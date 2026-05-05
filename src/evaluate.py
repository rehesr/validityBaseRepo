"""Compare model outputs against reference labels. Compute P/R/F1 per model."""

import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.client import call_model


TICKER_ALIASES = {
    "BF.B": "BF.A",
    "BRK.B": "BRK.A",
    "FOXA": "FOX",
    "GOOGL": "GOOG",
    "LGF.B": "LGF.A",
    "NWSA": "NWS",
    "RDSA": "RDS.A",
}


def canonicalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    normalized = normalized.replace("/", ".")
    if "-" in normalized and len(normalized.split("-", 1)[1]) == 1:
        normalized = normalized.replace("-", ".", 1)
    return TICKER_ALIASES.get(normalized, normalized)


def load_reference(ref_path: str | Path) -> dict[str, set[str]]:
    """Return {doc_id: set_of_tickers}."""
    with Path(ref_path).open(encoding="utf-8") as handle:
        ref = json.load(handle)
    return {
        doc["doc_id"]: {
            canonicalize_ticker(label["ticker"]) for label in doc["labels"] if label["label"] == 1
        }
        for doc in ref["documents"]
    }


def _parse_labels_content(content: str) -> tuple[list[dict[str, Any]], bool]:
    content = content.strip()
    fence_blocks = content.split("```")
    json_candidates = []
    for block in fence_blocks:
        candidate = block.strip()
        if not candidate:
            continue
        if "\n" in candidate:
            header, body = candidate.split("\n", 1)
            if header.strip().lower() in ("json", ""):
                json_candidates.append(body)
                continue
    if json_candidates:
        # Some models add commentary before or between fenced JSON blocks.
        # Prefer the last fenced JSON block instead of rejecting the whole response.
        content = json_candidates[-1]
    try:
        labels = json.loads(content)
    except json.JSONDecodeError:
        return [], True

    if isinstance(labels, dict):
        labels = labels.get("tickers")
    if not isinstance(labels, list):
        return [], True
    return labels, False


def _extract_tickers_and_names(
    labels: list[dict[str, Any]], threshold: float = 0.5
) -> tuple[set[str], dict[str, set[str]], bool]:
    tickers: set[str] = set()
    ticker_company_names: dict[str, set[str]] = defaultdict(set)

    for label in labels:
        if not isinstance(label, dict):
            return set(), {}, True

        ticker = label.get("ticker")
        confidence = label.get("confidence", 1.0)
        company_name = label.get("company_name")

        if not isinstance(ticker, str):
            return set(), {}, True
        if not isinstance(confidence, (int, float)):
            return set(), {}, True

        normalized_ticker = canonicalize_ticker(ticker)
        if not normalized_ticker:
            continue

        if confidence >= threshold:
            tickers.add(normalized_ticker)
            if isinstance(company_name, str) and company_name.strip():
                ticker_company_names[normalized_ticker].add(company_name.strip())

    return tickers, dict(ticker_company_names), False


def parse_model_tickers(result_path: str | Path, threshold: float = 0.5) -> tuple[set[str], bool]:
    """Extract ticker set from a model result file and flag parse failures."""
    tickers, _, parse_error = parse_model_entities(result_path, threshold)
    return tickers, parse_error


def parse_model_entities(
    result_path: str | Path, threshold: float = 0.5
) -> tuple[set[str], dict[str, set[str]], bool]:
    with Path(result_path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    content = data.get("content", "")
    if not content:
        return set(), {}, False

    labels, parse_error = _parse_labels_content(content)
    if parse_error:
        return set(), {}, True
    return _extract_tickers_and_names(labels, threshold)


def parse_result_entities(
    result: dict[str, Any], threshold: float = 0.5
) -> tuple[set[str], dict[str, set[str]], bool]:
    content = result.get("content", "")
    if not content:
        return set(), {}, False
    if not isinstance(content, str):
        return set(), {}, True

    labels, parse_error = _parse_labels_content(content)
    if parse_error:
        return set(), {}, True
    return _extract_tickers_and_names(labels, threshold)


def score(predicted: set[str], actual: set[str]) -> dict[str, float | int]:
    if not predicted and not actual:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(predicted & actual)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": len(predicted - actual),
        "fn": len(actual - predicted),
    }


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    total_tp = sum(int(row["tp"]) for row in rows)
    total_fp = sum(int(row["fp"]) for row in rows)
    total_fn = sum(int(row["fn"]) for row in rows)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def _ticker_vote_partition(
    rows: list[dict[str, Any]], ratio: float = 0.75
) -> dict[str, Any]:
    model_count = len(rows)
    min_votes = max(1, math.ceil(model_count * ratio))

    ticker_counts: dict[str, int] = defaultdict(int)
    ticker_files: dict[str, list[str]] = defaultdict(list)
    ticker_to_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for ticker in row["predicted"]:
            ticker_counts[ticker] += 1
            result_file = row.get("result_file")
            if isinstance(result_file, str):
                ticker_files[ticker].append(result_file)
            ticker_to_names[ticker].update(row.get("ticker_company_names", {}).get(ticker, set()))

    accepted = {ticker for ticker, count in ticker_counts.items() if count >= min_votes}
    review = {
        ticker for ticker, count in ticker_counts.items() if 2 <= count < min_votes
    }
    discarded = {ticker for ticker, count in ticker_counts.items() if count == 1}

    return {
        "model_count": model_count,
        "min_votes": min_votes,
        "ticker_counts": dict(ticker_counts),
        "ticker_files": {ticker: sorted(files) for ticker, files in ticker_files.items()},
        "ticker_company_names": {ticker: set(names) for ticker, names in ticker_to_names.items()},
        "accepted": accepted,
        "review": review,
        "discarded": discarded,
    }


def _set_consensus(rows: list[dict[str, Any]], ratio: float = 0.75) -> tuple[set[str] | None, int, int]:
    model_count = len(rows)
    min_votes = max(1, math.ceil(model_count * ratio))

    set_votes: dict[frozenset[str], int] = defaultdict(int)
    for row in rows:
        set_votes[frozenset(row["predicted"])] += 1

    if not set_votes:
        return set(), 0, min_votes

    winning_set, winning_votes = max(
        set_votes.items(),
        key=lambda item: (item[1], sorted(item[0])),
    )
    if winning_votes >= min_votes:
        return set(winning_set), winning_votes, min_votes
    return None, winning_votes, min_votes


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_comparison_summary(eval_dir: Path) -> dict[str, Any] | None:
    comparison_path = eval_dir / "all_docs_first_second_pass_comparison.csv"
    if not comparison_path.exists():
        return None

    with comparison_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    statuses = defaultdict(int)
    manual_docs: list[str] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for row in rows:
        status = row.get("second_pass_status", "")
        statuses[status] += 1
        if status == "manual":
            manual_docs.append(row["doc_id"])
        precision_values.append(float(row["precision_vs_first_pass"]))
        recall_values.append(float(row["recall_vs_first_pass"]))
        f1_values.append(float(row["f1_vs_first_pass"]))

    return {
        "rows": len(rows),
        "statuses": dict(statuses),
        "manual_docs": manual_docs,
        "avg_precision_vs_first_pass": sum(precision_values) / len(precision_values),
        "avg_recall_vs_first_pass": sum(recall_values) / len(recall_values),
        "avg_f1_vs_first_pass": sum(f1_values) / len(f1_values),
    }


def _read_second_pass_model_summary(eval_dir: Path) -> dict[str, dict[str, Any]] | None:
    scores_path = eval_dir / "first_pass_actual_second_pass_scores.csv"
    if not scores_path.exists():
        return None

    with scores_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = row.get("second_pass_model", "")
        if not model:
            continue
        by_model[model].append(row)

    if not by_model:
        return None

    summary: dict[str, dict[str, Any]] = {}
    for model, model_rows in sorted(by_model.items()):
        precision_values = [float(row["precision"]) for row in model_rows if row["precision"] != ""]
        recall_values = [float(row["recall"]) for row in model_rows if row["recall"] != ""]
        f1_values = [float(row["f1"]) for row in model_rows if row["f1"] != ""]
        summary[model] = {
            "rows": len(model_rows),
            "parse_errors": sum(1 for row in model_rows if row.get("parse_error") == "True"),
            "avg_precision": (sum(precision_values) / len(precision_values)) if precision_values else 0.0,
            "avg_recall": (sum(recall_values) / len(recall_values)) if recall_values else 0.0,
            "avg_f1": (sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        }
    return summary


def write_second_pass_csv_from_artifacts(exp_dir: Path) -> dict[str, int]:
    eval_dir = exp_dir / "eval"
    canonical_path = eval_dir / "canonical_reference.json"
    manual_path = eval_dir / "manual_label_queue.json"
    if not canonical_path.exists() or not manual_path.exists():
        raise ValueError(
            "Missing second-pass artifacts. Run --canonical-stage second-pass before second-pass-csv."
        )

    with canonical_path.open(encoding="utf-8") as handle:
        canonical_docs = json.load(handle).get("documents", [])
    with manual_path.open(encoding="utf-8") as handle:
        manual_docs = json.load(handle).get("documents", [])

    rows: list[dict[str, Any]] = []
    for doc in canonical_docs:
        labels = doc.get("labels", [])
        actual_tickers = sorted(
            [
                label.get("ticker")
                for label in labels
                if isinstance(label, dict) and isinstance(label.get("ticker"), str)
            ]
        )
        rows.append(
            {
                "doc_id": doc.get("doc_id"),
                "status": "canonical",
                "source": doc.get("source"),
                "actual": json.dumps(actual_tickers),
                "tickers": json.dumps(actual_tickers),
                "votes": doc.get("votes"),
                "min_votes": doc.get("min_votes"),
                "reason": "",
            }
        )

    for doc in manual_docs:
        rows.append(
            {
                "doc_id": doc.get("doc_id"),
                "status": "manual",
                "source": "",
                "actual": "[]",
                "tickers": "[]",
                "votes": doc.get("second_pass_votes", ""),
                "min_votes": doc.get("second_pass_min_votes", ""),
                "reason": doc.get("reason", ""),
            }
        )

    rows_sorted = sorted(rows, key=lambda row: (row.get("doc_id") or "", row.get("status") or ""))
    _write_csv(eval_dir / "second_pass_outcomes.csv", rows_sorted)
    return {"second_pass_csv_rows": len(rows_sorted)}


def _load_results_by_doc(exp_dir: Path) -> dict[str, list[dict[str, Any]]]:
    results_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result_file in sorted(glob.glob(str(exp_dir / "results" / "*.json"))):
        model_slug, doc_id = Path(result_file).stem.split("__", 1)
        predicted, ticker_company_names, parse_error = parse_model_entities(result_file)
        results_by_doc[doc_id].append(
            {
                "model": model_slug,
                "result_file": Path(result_file).name,
                "predicted": predicted,
                "ticker_company_names": ticker_company_names,
                "parse_error": parse_error,
            }
        )
    return results_by_doc


def write_all_docs_first_second_pass_comparison(exp_dir: Path, cfg: dict[str, Any]) -> dict[str, int]:
    eval_dir = exp_dir / "eval"
    first_pass_path = eval_dir / "canonical_first_pass.json"
    queue_path = eval_dir / "second_pass_queue.json"
    canonical_path = eval_dir / "canonical_reference.json"
    manual_path = eval_dir / "manual_label_queue.json"
    second_pass_results_dir = eval_dir / "second_pass_results"

    required_paths = [first_pass_path, queue_path, canonical_path, manual_path]
    if any(not path.exists() for path in required_paths):
        raise ValueError(
            "Missing canonical artifacts. Run --canonical-stage second-pass before writing comparison CSV."
        )

    with first_pass_path.open(encoding="utf-8") as handle:
        first_pass_docs = json.load(handle).get("documents", [])
    with queue_path.open(encoding="utf-8") as handle:
        queue_docs = json.load(handle).get("documents", [])
    with canonical_path.open(encoding="utf-8") as handle:
        canonical_docs = json.load(handle).get("documents", [])
    with manual_path.open(encoding="utf-8") as handle:
        manual_docs = json.load(handle).get("documents", [])

    canonical_by_doc = {doc["doc_id"]: doc for doc in canonical_docs}
    manual_by_doc = {doc["doc_id"]: doc for doc in manual_docs}
    queued_by_doc = {doc["doc_id"]: doc for doc in queue_docs}

    first_pass_canonical_by_doc = {doc["doc_id"]: doc for doc in first_pass_docs}
    rows: list[dict[str, Any]] = []

    all_doc_ids = sorted(set(first_pass_canonical_by_doc) | set(queued_by_doc))
    for doc_id in all_doc_ids:
        first_pass_doc = first_pass_canonical_by_doc.get(doc_id)
        queued_item = queued_by_doc.get(doc_id)
        second_rows: list[dict[str, Any]] = []

        if first_pass_doc is not None:
            first_pass_tickers = sorted(
                [
                    label.get("ticker")
                    for label in first_pass_doc.get("labels", [])
                    if isinstance(label, dict) and isinstance(label.get("ticker"), str)
                ]
            )
            first_pass_source = first_pass_doc.get("source", "first_pass_consensus")
            first_pass_votes = doc.get("votes", {}) if False else first_pass_doc.get("votes", {})
        else:
            first_pass_votes = queued_item.get("first_pass_votes", {})
            first_pass_tickers = sorted(first_pass_votes)
            first_pass_source = "first_pass_candidate_union"

        for model in cfg["models"]:
            model_slug = _model_slug(model["id"])
            result_path = second_pass_results_dir / f"{model_slug}__{doc_id}.json"
            if not result_path.exists():
                continue
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            predicted, ticker_company_names, parse_error = parse_result_entities(result)
            second_predictions_by_model[model_slug] = sorted(predicted)
            second_rows.append(
                {
                    "model": model_slug,
                    "predicted": predicted,
                    "ticker_company_names": ticker_company_names,
                    "parse_error": parse_error,
                }
            )

        second_partition = _ticker_vote_partition(second_rows, 0.75) if second_rows else {
            "ticker_counts": {},
            "accepted": set(),
            "review": set(),
            "min_votes": 0,
        }
        final_doc = canonical_by_doc.get(doc_id)
        manual_doc = manual_by_doc.get(doc_id)
        if first_pass_doc is not None:
            second_pass_status = "copied_from_first_pass"
            second_pass_source = "copied_from_first_pass"
            second_pass_tickers: Any = sorted(first_pass_tickers)
            second_pass_count = len(first_pass_tickers)
            second_pass_votes: Any = ""
            second_pass_min_votes: Any = ""
            metrics = score(set(first_pass_tickers), set(first_pass_tickers))
        elif manual_doc is not None:
            consensus_tickers = sorted(second_partition["accepted"])
            non_consensus_tickers = sorted(second_partition["review"])
            second_pass_status = "manual"
            second_pass_source = "manual_required"
            second_pass_tickers = {
                "consensus_tickers": consensus_tickers,
                "non_consensus_tickers": non_consensus_tickers,
                "vote_counts": second_partition["ticker_counts"],
            }
            second_pass_count = len(consensus_tickers)
            second_pass_votes = (
                max(second_partition["ticker_counts"].get(ticker, 0) for ticker in non_consensus_tickers)
                if non_consensus_tickers
                else ""
            )
            second_pass_min_votes = second_partition["min_votes"]
            metrics = score(set(consensus_tickers), set(first_pass_tickers))
        else:
            final_labels = sorted(
                [
                    label.get("ticker")
                    for label in final_doc.get("labels", [])
                    if isinstance(label, dict) and isinstance(label.get("ticker"), str)
                ]
            )
            second_pass_status = "canonical"
            second_pass_source = "second_pass_consensus_2_2"
            second_pass_tickers = final_labels
            second_pass_count = len(final_labels)
            second_pass_votes = (
                min(second_partition["ticker_counts"].get(ticker, second_partition["min_votes"]) for ticker in second_partition["accepted"])
                if second_partition["accepted"]
                else ""
            )
            second_pass_min_votes = second_partition["min_votes"]
            metrics = score(set(final_labels), set(first_pass_tickers))

        rows.append(
            {
                "doc_id": doc_id,
                "first_pass_source": first_pass_source,
                "first_pass_tickers": json.dumps(first_pass_tickers),
                "first_pass_count": len(first_pass_tickers),
                "second_pass_status": second_pass_status,
                "second_pass_source": second_pass_source,
                "second_pass_tickers": json.dumps(second_pass_tickers, ensure_ascii=False, sort_keys=True),
                "second_pass_count": second_pass_count,
                "second_pass_votes": second_pass_votes,
                "second_pass_min_votes": second_pass_min_votes,
                "precision_vs_first_pass": metrics["precision"],
                "recall_vs_first_pass": metrics["recall"],
                "f1_vs_first_pass": metrics["f1"],
                "tp_vs_first_pass": metrics["tp"],
                "fp_vs_first_pass": metrics["fp"],
                "fn_vs_first_pass": metrics["fn"],
            }
        )

    rows_sorted = sorted(rows, key=lambda row: row["doc_id"])
    _write_csv(eval_dir / "all_docs_first_second_pass_comparison.csv", rows_sorted)
    return {"all_docs_comparison_rows": len(rows_sorted)}


def write_all_docs_first_second_pass_comparison_from_first_pass_actual(
    exp_dir: Path,
    cfg: dict[str, Any],
    results_by_doc: dict[str, list[dict[str, Any]]],
    ratio: float = 0.75,
) -> dict[str, int]:
    eval_dir = exp_dir / "eval"
    second_pass_results_dir = eval_dir / "first_pass_actual_second_pass_results"
    if not second_pass_results_dir.exists():
        raise ValueError(
            "Missing second-pass artifacts. Run --canonical-stage second-pass or "
            "--canonical-stage first-pass-actual-second-pass before second-pass-csv."
        )

    rows: list[dict[str, Any]] = []
    for doc_id in sorted(results_by_doc):
        first_rows = results_by_doc[doc_id]
        first_partition = _ticker_vote_partition(first_rows, ratio)
        accepted_tickers = first_partition["accepted"]
        review_tickers = first_partition["review"]
        discarded_tickers = first_partition["discarded"]
        second_rows: list[dict[str, Any]] = []

        if review_tickers:
            for model in cfg["models"]:
                model_slug = _model_slug(model["id"])
                result_path = second_pass_results_dir / f"{model_slug}__{doc_id}.json"
                if not result_path.exists():
                    continue
                with result_path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
                predicted, ticker_company_names, parse_error = parse_result_entities(result)
                second_rows.append(
                    {
                        "model": model_slug,
                        "predicted": predicted,
                        "ticker_company_names": ticker_company_names,
                        "parse_error": parse_error,
                    }
                )

        second_partition = _ticker_vote_partition(second_rows, ratio) if second_rows else {
            "ticker_counts": {},
            "accepted": set(),
            "review": set(),
            "min_votes": 0,
        }
        combined_tickers = sorted(accepted_tickers | second_partition["accepted"])
        if not review_tickers:
            second_pass_status = "copied_from_first_pass"
            second_pass_source = "copied_from_first_pass"
            second_pass_tickers: Any = sorted(accepted_tickers)
            second_pass_count = len(accepted_tickers)
            second_pass_votes: Any = ""
            second_pass_min_votes: Any = ""
            metrics = score(set(accepted_tickers), set(accepted_tickers))
        elif second_partition["review"]:
            consensus_tickers = sorted(combined_tickers)
            non_consensus_tickers = sorted(second_partition["review"])
            second_pass_status = "manual"
            second_pass_source = "manual_required"
            second_pass_tickers = {
                "consensus_tickers": consensus_tickers,
                "non_consensus_tickers": non_consensus_tickers,
                "vote_counts": second_partition["ticker_counts"],
            }
            second_pass_count = len(consensus_tickers)
            second_pass_votes = (
                max(second_partition["ticker_counts"].get(ticker, 0) for ticker in non_consensus_tickers)
                if non_consensus_tickers
                else ""
            )
            second_pass_min_votes = second_partition["min_votes"]
            metrics = score(set(consensus_tickers), set(first_partition["ticker_counts"]))
        else:
            second_pass_status = "canonical"
            second_pass_source = "second_pass_consensus_2_2"
            second_pass_tickers = combined_tickers
            second_pass_count = len(combined_tickers)
            second_pass_votes = (
                min(second_partition["ticker_counts"].get(ticker, second_partition["min_votes"]) for ticker in second_partition["accepted"])
                if second_partition["accepted"]
                else ""
            )
            second_pass_min_votes = second_partition["min_votes"]
            metrics = score(set(combined_tickers), set(first_partition["ticker_counts"]))

        rows.append(
            {
                "doc_id": doc_id,
                "first_pass_source": "first_pass_candidate_union" if review_tickers else "first_pass_consensus",
                "first_pass_tickers": json.dumps(sorted(first_partition["ticker_counts"])),
                "first_pass_count": len(first_partition["ticker_counts"]),
                "second_pass_status": second_pass_status,
                "second_pass_source": second_pass_source,
                "second_pass_tickers": json.dumps(second_pass_tickers, ensure_ascii=False, sort_keys=True),
                "second_pass_count": second_pass_count,
                "second_pass_votes": second_pass_votes,
                "second_pass_min_votes": second_pass_min_votes,
                "precision_vs_first_pass": metrics["precision"],
                "recall_vs_first_pass": metrics["recall"],
                "f1_vs_first_pass": metrics["f1"],
                "tp_vs_first_pass": metrics["tp"],
                "fp_vs_first_pass": metrics["fp"],
                "fn_vs_first_pass": metrics["fn"],
            }
        )

    rows_sorted = sorted(rows, key=lambda row: row["doc_id"])
    _write_csv(eval_dir / "all_docs_first_second_pass_comparison.csv", rows_sorted)
    return {"all_docs_comparison_rows": len(rows_sorted)}


def _load_second_pass_prompt(prompt_path: str | Path | None) -> str:
    if prompt_path:
        path = Path(prompt_path).expanduser()
    else:
        path = Path(__file__).resolve().parents[1] / "prompts" / "canonical_subset_v1.txt"
    with path.open(encoding="utf-8") as handle:
        return handle.read()


def _build_candidate_pairs(
    rows: list[dict[str, Any]], allowed_tickers: set[str] | None = None
) -> list[dict[str, Any]]:
    ticker_to_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for ticker in row["predicted"]:
            if allowed_tickers is not None and ticker not in allowed_tickers:
                continue
            ticker_to_names[ticker].update(row["ticker_company_names"].get(ticker, set()))

    return [
        {
            "ticker": ticker,
            "company_names": sorted(ticker_to_names[ticker]),
        }
        for ticker in sorted(ticker_to_names)
    ]


def evaluate_second_pass_against_first_pass_actual(
    exp_dir: Path,
    cfg: dict[str, Any],
    results_by_doc: dict[str, list[dict[str, Any]]],
    second_pass_prompt_path: str | Path | None = None,
    ratio: float = 0.75,
) -> dict[str, int]:
    eval_dir = exp_dir / "eval"
    input_dir = Path(os.path.expanduser(cfg["input_dir"]))
    prompt_template = _load_second_pass_prompt(second_pass_prompt_path)

    models = cfg["models"]
    temperature = cfg.get("temperature", 0)
    max_tokens = cfg.get("max_tokens", 4096)

    results_dir = eval_dir / "first_pass_actual_second_pass_results"
    results_dir.mkdir(exist_ok=True)
    log_path = eval_dir / "first_pass_actual_second_pass_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    outcome_rows: list[dict[str, Any]] = []

    for doc_id in sorted(results_by_doc):
        first_rows = results_by_doc[doc_id]
        first_partition = _ticker_vote_partition(first_rows, ratio)
        review_set = first_partition["review"]
        if not review_set:
            continue
        actual_set = first_partition["accepted"]
        min_votes = first_partition["min_votes"]

        transcript_path = input_dir / f"{doc_id}.txt"
        if not transcript_path.exists():
            outcome_rows.append(
                {
                    "doc_id": doc_id,
                    "actual": json.dumps(sorted(actual_set)),
                    "actual_source": "first_pass_consensus",
                    "actual_min_votes": min_votes,
                    "first_pass_accepted": json.dumps(sorted(first_partition["accepted"])),
                    "first_pass_review": json.dumps(sorted(review_set)),
                    "second_pass_model": "",
                    "second_pass_predicted": "[]",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "parse_error": "",
                    "reason": "missing_transcript",
                }
            )
            continue

        transcript_text = transcript_path.read_text(encoding="utf-8")
        candidate_pairs_json = json.dumps(
            _build_candidate_pairs(first_rows, review_set), indent=2, ensure_ascii=False
        )

        for model in models:
            model_id = model["id"]
            prompt = (
                prompt_template.replace("{{transcript}}", transcript_text)
                .replace("{{candidate_pairs_json}}", candidate_pairs_json)
            )
            result = call_model(model_id, prompt, temperature=temperature, max_tokens=max_tokens)

            result_file = results_dir / f"{_model_slug(model_id)}__{doc_id}.json"
            with result_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            predicted, _, parse_error = parse_result_entities(result)
            error = result.get("error")
            metrics = (
                {"precision": "", "recall": "", "f1": "", "tp": "", "fp": "", "fn": ""}
                if error
                else score(predicted, actual_set)
            )
            outcome_rows.append(
                {
                    "doc_id": doc_id,
                    "actual": json.dumps(sorted(actual_set)),
                    "actual_source": "first_pass_consensus",
                    "actual_min_votes": min_votes,
                    "first_pass_accepted": json.dumps(sorted(first_partition["accepted"])),
                    "first_pass_review": json.dumps(sorted(review_set)),
                    "second_pass_model": _model_slug(model_id),
                    "second_pass_predicted": json.dumps(sorted(predicted)),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "parse_error": parse_error,
                    "second_pass_error": error or "",
                    "reason": "",
                }
            )

            _append_jsonl(
                log_path,
                {
                    "doc_id": doc_id,
                    "model": model_id,
                    "prompt_stage": "first_pass_actual_second_pass",
                    "prompt": prompt,
                    "response": result.get("response"),
                    "content": result.get("content"),
                    "latency_ms": result.get("latency_ms"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "finish_reason": result.get("finish_reason"),
                    "error": result.get("error"),
                },
            )

    _write_csv(eval_dir / "first_pass_actual_second_pass_scores.csv", outcome_rows)
    write_all_docs_first_second_pass_comparison_from_first_pass_actual(
        exp_dir=exp_dir,
        cfg=cfg,
        results_by_doc=results_by_doc,
        ratio=ratio,
    )
    return {
        "first_pass_actual_docs": len({row["doc_id"] for row in outcome_rows}),
        "second_pass_scored_rows": len(outcome_rows),
    }


def build_canonical_first_pass(
    exp_dir: Path,
    cfg: dict[str, Any],
    results_by_doc: dict[str, list[dict[str, Any]]],
    ratio: float = 0.75,
) -> dict[str, int]:
    eval_dir = exp_dir / "eval"
    input_dir = Path(os.path.expanduser(cfg["input_dir"]))

    canonical_docs: list[dict[str, Any]] = []
    second_pass_queue: list[dict[str, Any]] = []

    for doc_id in sorted(results_by_doc):
        first_rows = results_by_doc[doc_id]
        first_partition = _ticker_vote_partition(first_rows, ratio)
        accepted_tickers = first_partition["accepted"]
        review_tickers = first_partition["review"]
        if not review_tickers:
            canonical_docs.append(
                {
                    "doc_id": doc_id,
                    "labels": [{"ticker": ticker, "label": 1} for ticker in sorted(accepted_tickers)],
                    "source": "first_pass_ticker_votes",
                    "votes": first_partition["ticker_counts"],
                    "min_votes": first_partition["min_votes"],
                }
            )
            continue

        transcript_path = input_dir / f"{doc_id}.txt"
        second_pass_queue.append(
            {
                "doc_id": doc_id,
                "accepted_tickers": sorted(accepted_tickers),
                "review_tickers": sorted(review_tickers),
                "discarded_tickers": sorted(first_partition["discarded"]),
                "candidates": _build_candidate_pairs(first_rows, review_tickers),
                "first_pass_votes": first_partition["ticker_counts"],
                "first_pass_min_votes": first_partition["min_votes"],
                "transcript_path": str(transcript_path),
            }
        )

    _write_json(eval_dir / "canonical_first_pass.json", {"documents": canonical_docs})
    _write_json(eval_dir / "second_pass_queue.json", {"documents": second_pass_queue})

    return {
        "first_pass_canonical_docs": len(canonical_docs),
        "queued_for_second_pass_docs": len(second_pass_queue),
    }


def build_canonical_second_pass(
    exp_dir: Path,
    cfg: dict[str, Any],
    second_pass_prompt_path: str | Path | None = None,
    ratio: float = 0.75,
) -> dict[str, int]:
    eval_dir = exp_dir / "eval"
    queue_path = eval_dir / "second_pass_queue.json"
    first_pass_path = eval_dir / "canonical_first_pass.json"

    if not queue_path.exists() or not first_pass_path.exists():
        raise ValueError(
            "Missing first-pass artifacts. Run --canonical-stage first-pass before second-pass."
        )

    with queue_path.open(encoding="utf-8") as handle:
        queue_docs = json.load(handle).get("documents", [])
    with first_pass_path.open(encoding="utf-8") as handle:
        first_pass_docs = json.load(handle).get("documents", [])

    second_pass_results_dir = eval_dir / "second_pass_results"
    second_pass_results_dir.mkdir(exist_ok=True)

    second_pass_log = eval_dir / "second_pass_log.jsonl"
    if second_pass_log.exists():
        second_pass_log.unlink()

    second_pass_prompt = _load_second_pass_prompt(second_pass_prompt_path)

    models = cfg["models"]
    temperature = cfg.get("temperature", 0)
    max_tokens = cfg.get("max_tokens", 4096)

    second_pass_canonical: list[dict[str, Any]] = []
    manual_queue: list[dict[str, Any]] = []

    queue_docs_sorted = sorted(queue_docs, key=lambda doc: doc["doc_id"])
    total_docs = len(queue_docs_sorted)
    total_models = len(models)

    print(f"Second pass: {total_docs} docs queued; {total_models} models per doc.")

    for doc_index, item in enumerate(queue_docs_sorted, start=1):
        doc_id = item["doc_id"]
        transcript_path = Path(item["transcript_path"]).expanduser()
        candidates = item.get("candidates", [])
        print(f"[second-pass] doc {doc_index}/{total_docs}: {doc_id}")

        if not transcript_path.exists():
            manual_queue.append(
                {
                    "doc_id": doc_id,
                    "reason": "missing_transcript",
                    "transcript_path": str(transcript_path),
                    "accepted_tickers": item.get("accepted_tickers", []),
                    "review_tickers": item.get("review_tickers", []),
                    "candidates": candidates,
                    "first_pass_votes": item.get("first_pass_votes"),
                    "first_pass_min_votes": item.get("first_pass_min_votes"),
                }
            )
            continue

        transcript_text = transcript_path.read_text(encoding="utf-8")
        candidate_pairs_json = json.dumps(candidates, indent=2, ensure_ascii=False)

        second_rows: list[dict[str, Any]] = []
        for model_index, model in enumerate(models, start=1):
            model_id = model["id"]
            print(f"[second-pass]   model {model_index}/{total_models}: {model_id}")
            prompt = (
                second_pass_prompt.replace("{{transcript}}", transcript_text)
                .replace("{{candidate_pairs_json}}", candidate_pairs_json)
            )
            result = call_model(model_id, prompt, temperature=temperature, max_tokens=max_tokens)

            result_file = second_pass_results_dir / f"{_model_slug(model_id)}__{doc_id}.json"
            with result_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            predicted, ticker_company_names, parse_error = parse_result_entities(result)
            second_rows.append(
                {
                    "model": _model_slug(model_id),
                    "model_id": model_id,
                    "result_file": result_file.name,
                    "predicted": predicted,
                    "ticker_company_names": ticker_company_names,
                    "parse_error": parse_error,
                }
            )

            _append_jsonl(
                second_pass_log,
                {
                    "doc_id": doc_id,
                    "model": model_id,
                    "prompt_stage": "second_pass",
                    "prompt": prompt,
                    "response": result.get("response"),
                    "content": result.get("content"),
                    "latency_ms": result.get("latency_ms"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "finish_reason": result.get("finish_reason"),
                    "error": result.get("error"),
                },
            )

        second_partition = _ticker_vote_partition(second_rows, ratio)
        second_consensus_set = second_partition["accepted"]
        second_review_tickers = second_partition["review"]
        if not second_review_tickers:
            second_pass_canonical.append(
                {
                    "doc_id": doc_id,
                    "labels": [
                        {"ticker": ticker, "label": 1}
                        for ticker in sorted(set(item.get("accepted_tickers", [])) | second_consensus_set)
                    ],
                    "source": "first_pass_plus_second_pass_ticker_votes",
                    "votes": second_partition["ticker_counts"],
                    "min_votes": second_partition["min_votes"],
                    "candidates": candidates,
                }
            )
            continue

        manual_queue.append(
            {
                "doc_id": doc_id,
                "reason": "two_vote_tickers_remain_after_second_pass",
                "accepted_tickers": item.get("accepted_tickers", []),
                "review_tickers": item.get("review_tickers", []),
                "candidates": candidates,
                "first_pass_votes": item.get("first_pass_votes"),
                "first_pass_min_votes": item.get("first_pass_min_votes"),
                "second_pass_accepted_tickers": sorted(second_consensus_set),
                "second_pass_review_tickers": sorted(second_review_tickers),
                "second_pass_votes": second_partition["ticker_counts"],
                "second_pass_min_votes": second_partition["min_votes"],
                "second_pass_result_files": [row["result_file"] for row in second_rows],
            }
        )

    canonical_docs = sorted(first_pass_docs + second_pass_canonical, key=lambda doc: doc["doc_id"])
    _write_json(eval_dir / "canonical_reference.json", {"documents": canonical_docs})
    _write_json(eval_dir / "manual_label_queue.json", {"documents": manual_queue})
    write_second_pass_csv_from_artifacts(exp_dir)
    write_all_docs_first_second_pass_comparison(exp_dir, cfg)

    return {
        "first_pass_canonical_docs": len(first_pass_docs),
        "second_pass_canonical_docs": len(second_pass_canonical),
        "canonical_docs": len(canonical_docs),
        "manual_docs": len(manual_queue),
    }


def build_canonical_reference(
    exp_dir: Path,
    cfg: dict[str, Any],
    results_by_doc: dict[str, list[dict[str, Any]]],
    ratio: float = 0.75,
    second_pass_prompt_path: str | Path | None = None,
    canonical_mode: str = "full",
) -> dict[str, int]:
    if canonical_mode == "first-pass-actual-second-pass":
        stats = evaluate_second_pass_against_first_pass_actual(
            exp_dir=exp_dir,
            cfg=cfg,
            results_by_doc=results_by_doc,
            second_pass_prompt_path=second_pass_prompt_path,
            ratio=ratio,
        )
        stats["canonical_mode"] = canonical_mode
        return stats

    if canonical_mode == "first-pass":
        stats = build_canonical_first_pass(exp_dir, cfg, results_by_doc, ratio)
        stats["canonical_mode"] = canonical_mode
        return stats

    if canonical_mode == "second-pass":
        stats = build_canonical_second_pass(exp_dir, cfg, second_pass_prompt_path, ratio)
        stats["canonical_mode"] = canonical_mode
        return stats

    if canonical_mode == "second-pass-csv":
        try:
            stats = write_second_pass_csv_from_artifacts(exp_dir)
            stats.update(write_all_docs_first_second_pass_comparison(exp_dir, cfg))
        except ValueError:
            results_by_doc = _load_results_by_doc(exp_dir)
            stats = write_all_docs_first_second_pass_comparison_from_first_pass_actual(
                exp_dir, cfg, results_by_doc, ratio
            )
        stats["canonical_mode"] = canonical_mode
        return stats

    if canonical_mode == "full":
        first_stats = build_canonical_first_pass(exp_dir, cfg, results_by_doc, ratio)
        second_stats = build_canonical_second_pass(exp_dir, cfg, second_pass_prompt_path, ratio)
        return {
            **first_stats,
            **second_stats,
            "canonical_mode": canonical_mode,
        }

    raise ValueError(f"Unknown canonical_mode: {canonical_mode}")


def evaluate_experiment(
    exp_dir: str | Path,
    ref_path: str | Path,
    cfg: dict[str, Any] | None = None,
    build_canonical: bool = False,
    second_pass_prompt_path: str | Path | None = None,
    canonical_mode: str = "full",
) -> Path:
    exp_dir = Path(exp_dir)
    reference = load_reference(ref_path)

    results_by_model: dict[str, list[dict]] = defaultdict(list)
    results_by_doc: dict[str, list[dict]] = defaultdict(list)
    for result_file in sorted(glob.glob(str(exp_dir / "results" / "*.json"))):
        filename = Path(result_file).stem
        model_slug, doc_id = filename.split("__", 1)
        predicted, ticker_company_names, parse_error = parse_model_entities(result_file)
        actual = reference.get(doc_id, set())
        metrics = score(predicted, actual)
        metrics["model"] = model_slug
        metrics["doc_id"] = doc_id
        metrics["predicted"] = sorted(predicted)
        metrics["actual"] = sorted(actual)
        metrics["parse_error"] = parse_error
        results_by_model[model_slug].append(metrics)
        results_by_doc[doc_id].append(
            {
                "model": model_slug,
                "result_file": Path(result_file).name,
                "predicted": predicted,
                "ticker_company_names": ticker_company_names,
                "parse_error": parse_error,
            }
        )

    consensus_by_doc: dict[str, dict] = {}
    for doc_id, rows in results_by_doc.items():
        partition = _ticker_vote_partition(rows, 0.75)
        consensus_tickers = partition["accepted"]
        split_tie_tickers = partition["review"]
        union_tickers = consensus_tickers | split_tie_tickers
        disagreement_tickers = partition["discarded"]
        disagreement_files = {
            ticker: partition["ticker_files"].get(ticker, []) for ticker in sorted(disagreement_tickers)
        }
        disagreement_model_votes = {
            ticker: partition["ticker_counts"][ticker] for ticker in sorted(disagreement_tickers)
        }

        consensus_by_doc[doc_id] = {
            "models_for_doc": partition["model_count"],
            "consensus_min_votes": partition["min_votes"],
            "consensus_tickers": sorted(consensus_tickers),
            "disagreement_tickers": sorted(disagreement_tickers),
            "union_tickers": sorted(union_tickers),
            "ticker_votes": {
                ticker: partition["ticker_counts"][ticker] for ticker in sorted(partition["ticker_counts"])
            },
            "disagreement_files": disagreement_files,
            "disagreement_model_votes": disagreement_model_votes,
        }

    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    all_rows: list[dict] = []
    for rows in results_by_model.values():
        for row in rows:
            doc_consensus = consensus_by_doc.get(row["doc_id"], {})
            row["models_for_doc"] = doc_consensus.get("models_for_doc", 0)
            row["consensus_min_votes"] = doc_consensus.get("consensus_min_votes", 0)
            row["consensus_tickers"] = doc_consensus.get("consensus_tickers", [])
            row["disagreement_tickers"] = doc_consensus.get("disagreement_tickers", [])
            row["union_tickers"] = doc_consensus.get("union_tickers", [])
            row["ticker_votes"] = json.dumps(
                doc_consensus.get("ticker_votes", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            row["disagreement_files"] = json.dumps(
                doc_consensus.get("disagreement_files", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            row["disagreement_model_votes"] = json.dumps(
                doc_consensus.get("disagreement_model_votes", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            all_rows.append(row)

    if all_rows:
        with (eval_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    canonical_stats: dict[str, int] | None = None
    if build_canonical:
        if cfg is None:
            raise ValueError("cfg is required when build_canonical=True")
        canonical_stats = build_canonical_reference(
            exp_dir=exp_dir,
            cfg=cfg,
            results_by_doc=results_by_doc,
            ratio=0.75,
            second_pass_prompt_path=second_pass_prompt_path,
            canonical_mode=canonical_mode,
        )

    with (eval_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Evaluation Summary\n\n")
        if reference:
            handle.write("| Model | TP | FP | FN | Precision | Recall | F1 | Parse Errors | Docs |\n")
            handle.write("|-------|----|----|----|-----------|--------|----|--------------|------|\n")
            for model, rows in sorted(results_by_model.items()):
                totals = aggregate_scores(rows)
                parse_errors = sum(1 for row in rows if row["parse_error"])
                handle.write(
                    f"| {model} | {totals['tp']} | {totals['fp']} | {totals['fn']} | "
                    f"{totals['precision']:.3f} | {totals['recall']:.3f} | {totals['f1']:.3f} | "
                    f"{parse_errors} | {len(rows)} |\n"
                )
        else:
            handle.write(
                "Reference file is empty, so direct model-vs-reference metrics are omitted.\n"
            )

        if canonical_stats is not None:
            mode = canonical_stats["canonical_mode"]
            handle.write("\n## Canonical Labeling\n\n")
            handle.write(f"- Mode: `{mode}`\n")
            if mode == "first-pass":
                handle.write(
                    f"- First-pass canonical docs: {canonical_stats['first_pass_canonical_docs']}\n"
                )
                handle.write(
                    f"- Queued for second pass: {canonical_stats['queued_for_second_pass_docs']}\n"
                )
                handle.write("- First-pass output: `eval/canonical_first_pass.json`\n")
                handle.write("- Queue output: `eval/second_pass_queue.json`\n")
            elif mode in ("second-pass", "full"):
                handle.write(
                    f"- First-pass canonical docs: {canonical_stats['first_pass_canonical_docs']}\n"
                )
                handle.write(
                    f"- Second-pass canonical docs: {canonical_stats['second_pass_canonical_docs']}\n"
                )
                handle.write(f"- Total canonical docs: {canonical_stats['canonical_docs']}\n")
                handle.write(f"- Manual-review docs: {canonical_stats['manual_docs']}\n")
                handle.write("- Output: `eval/canonical_reference.json`\n")
                handle.write("- Manual queue: `eval/manual_label_queue.json`\n")
                handle.write("- CSV output: `eval/second_pass_outcomes.csv`\n")
            elif mode == "first-pass-actual-second-pass":
                handle.write(
                    f"- First-pass actual docs: {canonical_stats['first_pass_actual_docs']}\n"
                )
                handle.write(
                    f"- Second-pass scored rows: {canonical_stats['second_pass_scored_rows']}\n"
                )
                handle.write("- Output: `eval/first_pass_actual_second_pass_scores.csv`\n")
            else:
                if "second_pass_csv_rows" in canonical_stats:
                    handle.write(
                        f"- CSV rows written: {canonical_stats['second_pass_csv_rows']}\n"
                    )
                    handle.write("- CSV output: `eval/second_pass_outcomes.csv`\n")
                handle.write(
                    f"- Comparison rows written: {canonical_stats['all_docs_comparison_rows']}\n"
                )
                handle.write("- Comparison output: `eval/all_docs_first_second_pass_comparison.csv`\n")

        comparison_summary = _read_comparison_summary(eval_dir)
        if comparison_summary is not None:
            statuses = comparison_summary["statuses"]
            manual_docs = comparison_summary["manual_docs"]
            handle.write("\n## Final Results\n\n")
            handle.write(f"- Docs summarized: {comparison_summary['rows']}\n")
            handle.write(f"- Copied from first pass: {statuses.get('copied_from_first_pass', 0)}\n")
            handle.write(f"- Canonical after second pass: {statuses.get('canonical', 0)}\n")
            handle.write(f"- Manual inspection needed: {statuses.get('manual', 0)}\n")
            handle.write(
                f"- Avg precision vs first pass: {comparison_summary['avg_precision_vs_first_pass']:.3f}\n"
            )
            handle.write(
                f"- Avg recall vs first pass: {comparison_summary['avg_recall_vs_first_pass']:.3f}\n"
            )
            handle.write(
                f"- Avg F1 vs first pass: {comparison_summary['avg_f1_vs_first_pass']:.3f}\n"
            )
            if manual_docs:
                handle.write(
                    f"- Manual docs: {', '.join(sorted(manual_docs))}\n"
                )

        second_pass_model_summary = _read_second_pass_model_summary(eval_dir)
        if second_pass_model_summary is not None:
            handle.write("\n## Second-Pass Models\n\n")
            handle.write("| Model | Avg P | Avg R | Avg F1 | Parse Errors | Rows |\n")
            handle.write("|-------|-------|-------|--------|--------------|------|\n")
            for model, stats in second_pass_model_summary.items():
                handle.write(
                    f"| {model} | {stats['avg_precision']:.3f} | {stats['avg_recall']:.3f} | "
                    f"{stats['avg_f1']:.3f} | {stats['parse_errors']} | {stats['rows']} |\n"
                )

    print(f"Eval written to {eval_dir}")
    return eval_dir

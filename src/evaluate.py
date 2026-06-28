"""Compare model outputs against reference labels. Compute P/R/F1 per model."""

import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.client import call_model
from src.sweep import append_jsonl, load_reference_tickers, render_prompt, utc_now_iso


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
        if candidate[0] in "[{":
            json_candidates.append(candidate)
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
        decoder = json.JSONDecoder()
        labels = None
        for idx, char in enumerate(content):
            if char not in "[{":
                continue
            try:
                labels, _ = decoder.raw_decode(content[idx:])
                break
            except json.JSONDecodeError:
                continue
        if labels is None:
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


def parse_result_sentiments(result: dict[str, Any]) -> tuple[dict[str, dict[str, float]], bool]:
    content = result.get("content", "")
    if not content:
        return {}, False
    if not isinstance(content, str):
        return {}, True

    labels, parse_error = _parse_labels_content(content)
    if parse_error:
        return {}, True

    sentiments: dict[str, dict[str, float]] = {}
    for label in labels:
        if not isinstance(label, dict):
            return {}, True

        ticker = label.get("ticker")
        score_value = label.get("score")
        confidence_value = label.get("confidence")
        if (
            not isinstance(ticker, str)
            or not isinstance(score_value, (int, float))
            or not isinstance(confidence_value, (int, float))
        ):
            return {}, True

        normalized_ticker = canonicalize_ticker(ticker)
        if not normalized_ticker:
            continue
        sentiments[normalized_ticker] = {
            "score": float(score_value),
            "confidence": float(confidence_value),
        }

    return sentiments, False


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


def build_sentiment_consensus(
    exp_dir: str | Path,
    ref_path: str | Path,
    agree_max_abs_dev: float = 0.50,
    review_max_abs_dev: float = 0.50,
) -> Path:
    """Compute per-(document, ticker) sentiment consensus from model outputs."""
    exp_dir = Path(exp_dir).expanduser()
    reference = load_reference(ref_path)

    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    observations: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    parse_errors: list[dict[str, str]] = []
    models_by_doc: dict[str, set[str]] = defaultdict(set)

    for result_file in sorted(glob.glob(str(exp_dir / "results" / "*.json"))):
        filename = Path(result_file).stem
        model_slug, doc_id = filename.split("__", 1)
        models_by_doc[doc_id].add(model_slug)
        with Path(result_file).open(encoding="utf-8") as handle:
            result_data = json.load(handle)

        sentiments, parse_error = parse_result_sentiments(result_data)
        if parse_error:
            parse_errors.append(
                {
                    "doc_id": doc_id,
                    "model": model_slug,
                    "result_file": Path(result_file).name,
                }
            )
            continue

        for ticker, values in sentiments.items():
            observations[doc_id][ticker].append(
                {
                    "model": model_slug,
                    "score": values["score"],
                    "confidence": values["confidence"],
                }
            )

    rows: list[dict[str, Any]] = []
    consensus_docs: list[dict[str, Any]] = []
    review_docs: list[dict[str, Any]] = []

    all_doc_ids = sorted(set(reference) | set(observations))
    for doc_id in all_doc_ids:
        doc_tickers = sorted(reference.get(doc_id, set()) | set(observations.get(doc_id, {})))
        doc_consensus_labels: list[dict[str, Any]] = []
        doc_review_labels: list[dict[str, Any]] = []

        for ticker in doc_tickers:
            ticker_observations = observations.get(doc_id, {}).get(ticker, [])
            model_count = len(models_by_doc.get(doc_id, set()))
            scores = [item["score"] for item in ticker_observations]
            confidences = [item["confidence"] for item in ticker_observations]
            score_by_model = {item["model"]: item["score"] for item in ticker_observations}
            confidence_by_model = {item["model"]: item["confidence"] for item in ticker_observations}

            if scores:
                full_mean_score = sum(scores) / len(scores)
                outlier_idx = max(
                    range(len(scores)),
                    key=lambda idx: (abs(scores[idx] - full_mean_score), scores[idx]),
                )
                trimmed_observations = [
                    item for idx, item in enumerate(ticker_observations) if idx != outlier_idx
                ]
                trimmed_scores = [item["score"] for item in trimmed_observations]
                trimmed_confidences = [item["confidence"] for item in trimmed_observations]
                candidate_mean_score = (
                    sum(trimmed_scores) / len(trimmed_scores) if trimmed_scores else None
                )
                median_score = statistics.median(scores)
                mean_confidence = (
                    sum(trimmed_confidences) / len(trimmed_confidences)
                    if trimmed_confidences
                    else None
                )
                score_stddev = (
                    statistics.pstdev(trimmed_scores) if len(trimmed_scores) > 1 else 0.0
                )
                max_abs_dev = (
                    max(abs(score - candidate_mean_score) for score in trimmed_scores)
                    if candidate_mean_score is not None and trimmed_scores
                    else None
                )
                max_abs_diff = (
                    max(trimmed_scores) - min(trimmed_scores)
                    if trimmed_scores
                    else None
                )
                outlier = ticker_observations[outlier_idx]
                retained_score_by_model = {
                    item["model"]: item["score"] for item in trimmed_observations
                }
                if len(scores) < model_count or len(scores) < 4:
                    status = "review_insufficient_scores"
                elif max_abs_diff is not None and max_abs_diff <= agree_max_abs_dev:
                    status = "consensus"
                elif max_abs_diff is not None and max_abs_diff > review_max_abs_dev:
                    status = "review_disagreement"
                else:
                    status = "review_borderline"
                gold_score = candidate_mean_score if status == "consensus" else None
            else:
                full_mean_score = None
                candidate_mean_score = None
                gold_score = None
                median_score = None
                mean_confidence = None
                score_stddev = None
                max_abs_dev = None
                max_abs_diff = None
                outlier = None
                retained_score_by_model = {}
                trimmed_scores = []
                status = "review_missing_scores"

            row = {
                "doc_id": doc_id,
                "ticker": ticker,
                "status": status,
                "gold_score": round(gold_score, 4) if gold_score is not None else "",
                "candidate_mean_score": (
                    round(candidate_mean_score, 4) if candidate_mean_score is not None else ""
                ),
                "full_mean_score": round(full_mean_score, 4) if full_mean_score is not None else "",
                "median_score": round(median_score, 4) if median_score is not None else "",
                "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else "",
                "score_stddev": round(score_stddev, 4) if score_stddev is not None else "",
                "max_abs_dev": round(max_abs_dev, 4) if max_abs_dev is not None else "",
                "max_abs_diff": round(max_abs_diff, 4) if max_abs_diff is not None else "",
                "n_scores": len(scores),
                "n_scores_used": len(trimmed_scores),
                "n_models_for_doc": model_count,
                "removed_model": outlier["model"] if outlier else "",
                "removed_score": round(outlier["score"], 4) if outlier else "",
                "outlier_model": outlier["model"] if outlier else "",
                "outlier_score": round(outlier["score"], 4) if outlier else "",
                "raw_scores": json.dumps(score_by_model, ensure_ascii=False, sort_keys=True),
                "retained_scores": json.dumps(
                    retained_score_by_model,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "score_by_model": json.dumps(score_by_model, ensure_ascii=False, sort_keys=True),
                "confidence_by_model": json.dumps(
                    confidence_by_model,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            rows.append(row)

            label = {
                "ticker": ticker,
                "confidence": row["mean_confidence"],
                "score_stddev": row["score_stddev"],
                "max_abs_dev": row["max_abs_dev"],
                "max_abs_diff": row["max_abs_diff"],
                "n_scores": len(scores),
                "n_scores_used": row["n_scores_used"],
                "removed_model": row["removed_model"],
                "removed_score": row["removed_score"],
                "outlier_model": row["outlier_model"],
                "outlier_score": row["outlier_score"],
            }
            if status == "consensus":
                doc_consensus_labels.append({**label, "score": row["gold_score"]})
            else:
                doc_review_labels.append({**label, "status": status})

        if doc_consensus_labels:
            consensus_docs.append({"doc_id": doc_id, "labels": doc_consensus_labels})
        if doc_review_labels:
            review_docs.append({"doc_id": doc_id, "labels": doc_review_labels})

    _write_csv(eval_dir / "sentiment_consensus.csv", rows)
    _write_json(
        eval_dir / "sentiment_consensus.json",
        {
            "thresholds": {
                "agree_max_abs_dev": agree_max_abs_dev,
                "review_max_abs_dev": review_max_abs_dev,
            },
            "documents": consensus_docs,
        },
    )
    _write_json(
        eval_dir / "sentiment_review_queue.json",
        {
            "thresholds": {
                "agree_max_abs_dev": agree_max_abs_dev,
                "review_max_abs_dev": review_max_abs_dev,
            },
            "documents": review_docs,
            "parse_errors": parse_errors,
        },
    )

    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1

    with (eval_dir / "sentiment_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Sentiment Consensus Summary\n\n")
        handle.write("- Consensus score: mean after dropping the score farthest from the full mean\n")
        handle.write(f"- Agreement threshold: trimmed max abs difference <= {agree_max_abs_dev:.2f}\n")
        handle.write(f"- Human-review threshold: trimmed max abs difference > {review_max_abs_dev:.2f}\n")
        handle.write(f"- Document/ticker pairs: {len(rows)}\n")
        handle.write(f"- Consensus pairs: {status_counts.get('consensus', 0)}\n")
        handle.write(
            f"- Human-review pairs: {len(rows) - status_counts.get('consensus', 0)}\n"
        )
        handle.write(f"- Unique articles needing review: {len(review_docs)}\n")
        handle.write(f"- Parse-error result files: {len(parse_errors)}\n")
        handle.write("\nOutputs:\n")
        handle.write("- `eval/sentiment_consensus.csv`\n")
        handle.write("- `eval/sentiment_consensus.json`\n")
        handle.write("- `eval/sentiment_review_queue.json`\n")

    print(f"Sentiment consensus written to {eval_dir}")
    return eval_dir


def _resolve_sentiment_gold_path(gold_path: str | Path) -> Path:
    path = Path(gold_path).expanduser()
    if path.is_dir():
        path = path / "eval" / "sentiment_consensus.json"
    if not path.exists():
        raise FileNotFoundError(f"Sentiment gold file not found: {path}")
    return path


def _load_sentiment_gold_scores(gold_path: str | Path) -> dict[str, dict[str, float]]:
    """Return {doc_id: {ticker: gold_score}} from a consensus JSON or CSV."""
    path = _resolve_sentiment_gold_path(gold_path)
    gold_scores: dict[str, dict[str, float]] = defaultdict(dict)

    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                doc_id = row.get("doc_id")
                ticker = row.get("ticker")
                score_value = row.get("gold_score")
                if not doc_id or not ticker or score_value in (None, ""):
                    continue
                try:
                    score = float(score_value)
                except ValueError:
                    continue
                gold_scores[doc_id][canonicalize_ticker(ticker)] = score
        return gold_scores

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    for doc in payload.get("documents", []):
        doc_id = doc.get("doc_id")
        labels = doc.get("labels", [])
        if not isinstance(doc_id, str) or not isinstance(labels, list):
            continue
        for label in labels:
            if not isinstance(label, dict):
                continue
            ticker = label.get("ticker")
            score_value = label.get("score")
            if not isinstance(ticker, str) or not isinstance(score_value, (int, float)):
                continue
            gold_scores[doc_id][canonicalize_ticker(ticker)] = float(score_value)
    return gold_scores


def _sentiment_difference_histogram_rows(
    scored_rows: list[dict[str, Any]],
    bin_width: float = 0.10,
) -> list[dict[str, Any]]:
    """Build signed and absolute difference histogram rows by model and overall."""
    histogram_rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {"ALL": scored_rows}
    for row in scored_rows:
        groups.setdefault(row["model"], []).append(row)

    metric_specs = [
        ("difference", -2.0, 2.0),
        ("absolute_difference", 0.0, 2.0),
    ]
    for model, model_rows in groups.items():
        for metric, lower_bound, upper_bound in metric_specs:
            values = [float(row[metric]) for row in model_rows if row.get(metric) != ""]
            if not values:
                continue

            bin_counts: dict[float, int] = defaultdict(int)
            for value in values:
                clamped = min(max(value, lower_bound), upper_bound)
                if clamped == upper_bound:
                    bin_start = upper_bound - bin_width
                else:
                    bin_start = math.floor((clamped - lower_bound) / bin_width) * bin_width
                    bin_start += lower_bound
                bin_start = round(bin_start, 10)
                bin_counts[bin_start] += 1

            total = len(values)
            for bin_start in sorted(bin_counts):
                bin_end = round(bin_start + bin_width, 10)
                histogram_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "bin_start": round(bin_start, 4),
                        "bin_end": round(bin_end, 4),
                        "n": bin_counts[bin_start],
                        "share": round(bin_counts[bin_start] / total, 6),
                    }
                )
    return histogram_rows


def _sentiment_result_costs(exp_dir: Path) -> dict[str, dict[str, Any]]:
    """Return per-model request/token/cost totals from result payload usage."""
    costs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "result_files": 0,
            "total_cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    for result_file in sorted(glob.glob(str(exp_dir / "results" / "*.json"))):
        filename = Path(result_file).stem
        model_slug, _ = filename.split("__", 1)
        with Path(result_file).open(encoding="utf-8") as handle:
            result_data = json.load(handle)
        usage = (result_data.get("response") or {}).get("usage") or {}
        model_costs = costs[model_slug]
        model_costs["result_files"] += 1
        model_costs["total_cost"] += float(usage.get("cost") or 0.0)
        model_costs["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        model_costs["output_tokens"] += int(usage.get("completion_tokens") or 0)
        model_costs["total_tokens"] += int(usage.get("total_tokens") or 0)
    return costs


def _sentiment_cost_summary_rows(
    cost_by_model: dict[str, dict[str, Any]],
    scored_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a dedicated cost summary table by model and overall."""
    scored_pairs_by_model: dict[str, int] = defaultdict(int)
    for row in scored_rows:
        scored_pairs_by_model[row["model"]] += 1

    cost_rows: list[dict[str, Any]] = []
    for model, model_costs in sorted(cost_by_model.items()):
        total_cost = float(model_costs.get("total_cost") or 0.0)
        result_files = int(model_costs.get("result_files") or 0)
        scored_pairs = scored_pairs_by_model.get(model, 0)
        cost_rows.append(
            {
                "model": model,
                "total_cost": round(total_cost, 6),
                "result_files": result_files,
                "scored_pairs": scored_pairs,
                "input_tokens": int(model_costs.get("input_tokens") or 0),
                "output_tokens": int(model_costs.get("output_tokens") or 0),
                "total_tokens": int(model_costs.get("total_tokens") or 0),
                "cost_per_result_file": (
                    round(total_cost / result_files, 6) if result_files else ""
                ),
                "cost_per_scored_pair": (
                    round(total_cost / scored_pairs, 6) if scored_pairs else ""
                ),
            }
        )

    if cost_rows:
        totals = {
            "total_cost": sum(float(row["total_cost"]) for row in cost_rows),
            "result_files": sum(int(row["result_files"]) for row in cost_rows),
            "scored_pairs": sum(int(row["scored_pairs"]) for row in cost_rows),
            "input_tokens": sum(int(row["input_tokens"]) for row in cost_rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in cost_rows),
            "total_tokens": sum(int(row["total_tokens"]) for row in cost_rows),
        }
        cost_rows.insert(
            0,
            {
                "model": "ALL",
                "total_cost": round(totals["total_cost"], 6),
                "result_files": totals["result_files"],
                "scored_pairs": totals["scored_pairs"],
                "input_tokens": totals["input_tokens"],
                "output_tokens": totals["output_tokens"],
                "total_tokens": totals["total_tokens"],
                "cost_per_result_file": (
                    round(totals["total_cost"] / totals["result_files"], 6)
                    if totals["result_files"]
                    else ""
                ),
                "cost_per_scored_pair": (
                    round(totals["total_cost"] / totals["scored_pairs"], 6)
                    if totals["scored_pairs"]
                    else ""
                ),
            },
        )
    return cost_rows


def _write_sentiment_histogram_plots(
    eval_dir: Path,
    scored_rows: list[dict[str, Any]],
    bin_width: float = 0.10,
) -> list[Path]:
    """Write PNG histogram plots for signed and absolute gold-score differences."""
    if not scored_rows:
        return []

    mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict[str, Any]]] = {"ALL": scored_rows}
    for row in scored_rows:
        groups.setdefault(row["model"], []).append(row)

    plot_specs = [
        (
            "difference",
            "sentiment_difference_histogram.png",
            "Signed Difference vs Gold",
            "Model score - gold score",
            [round(-2.0 + idx * bin_width, 10) for idx in range(int(4.0 / bin_width) + 1)],
        ),
        (
            "absolute_difference",
            "sentiment_absolute_difference_histogram.png",
            "Absolute Difference vs Gold",
            "Absolute difference",
            [round(idx * bin_width, 10) for idx in range(int(2.0 / bin_width) + 1)],
        ),
    ]

    output_paths: list[Path] = []
    for metric, filename, title, xlabel, bins in plot_specs:
        model_names = sorted(groups)
        n_models = len(model_names)
        ncols = 2 if n_models > 1 else 1
        nrows = math.ceil(n_models / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.5 * ncols, 3.4 * nrows),
            squeeze=False,
        )
        fig.suptitle(title, fontsize=14)
        for ax, model in zip(axes.flat, model_names):
            values = [float(row[metric]) for row in groups[model] if row.get(metric) != ""]
            ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_title(f"{model} (n={len(values)})", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.flat[n_models:]:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        output_path = eval_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def build_sentiment_gold_comparison(
    exp_dir: str | Path,
    gold_path: str | Path,
) -> Path:
    """Write per-model sentiment score deltas against a gold consensus file."""
    exp_dir = Path(exp_dir).expanduser()
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    gold_scores = _load_sentiment_gold_scores(gold_path)
    rows: list[dict[str, Any]] = []

    for result_file in sorted(glob.glob(str(exp_dir / "results" / "*.json"))):
        filename = Path(result_file).stem
        model_slug, doc_id = filename.split("__", 1)
        doc_gold = gold_scores.get(doc_id, {})
        if not doc_gold:
            continue

        with Path(result_file).open(encoding="utf-8") as handle:
            result_data = json.load(handle)
        sentiments, parse_error = parse_result_sentiments(result_data)
        api_error = result_data.get("error")
        if api_error:
            result_status = "api_error"
        elif parse_error:
            result_status = "parse_error"
        else:
            result_status = None
        usage = (result_data.get("response") or {}).get("usage") or {}

        for ticker, gold_score in sorted(doc_gold.items()):
            model_values = sentiments.get(ticker)
            model_score = model_values["score"] if model_values else None
            diff = model_score - gold_score if model_score is not None else None
            status = result_status or ("scored" if model_values else "missing")
            rows.append(
                {
                    "ticker": ticker,
                    "article": doc_id,
                    "model": model_slug,
                    "sentiment_label": round(model_score, 4) if model_score is not None else "",
                    "gold_score": round(gold_score, 4),
                    "difference": round(diff, 4) if diff is not None else "",
                    "absolute_difference": round(abs(diff), 4) if diff is not None else "",
                    "confidence": (
                        round(model_values["confidence"], 4) if model_values is not None else ""
                    ),
                    "status": status,
                    "error": api_error or "",
                    "result_cost": round(float(usage.get("cost") or 0.0), 6),
                    "result_input_tokens": int(usage.get("prompt_tokens") or 0),
                    "result_output_tokens": int(usage.get("completion_tokens") or 0),
                    "result_total_tokens": int(usage.get("total_tokens") or 0),
                    "result_file": Path(result_file).name,
                }
            )

    output_path = eval_dir / "sentiment_gold_comparison.csv"
    _write_csv(output_path, rows)
    scored_rows = [row for row in rows if row["difference"] != ""]
    cost_by_model = _sentiment_result_costs(exp_dir)
    summary_groups = {"ALL": scored_rows}
    for row in scored_rows:
        summary_groups.setdefault(row["model"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for model, model_rows in summary_groups.items():
        differences = [float(row["difference"]) for row in model_rows]
        absolute_differences = [float(row["absolute_difference"]) for row in model_rows]
        sentiment_scores = [float(row["sentiment_label"]) for row in model_rows]
        gold_scores_for_rows = [float(row["gold_score"]) for row in model_rows]
        if not differences:
            continue
        summary_rows.append(
            {
                "model": model,
                "n": len(differences),
                "mean_difference": round(sum(differences) / len(differences), 4),
                "median_difference": round(statistics.median(differences), 4),
                "stddev_difference": (
                    round(statistics.pstdev(differences), 4) if len(differences) > 1 else 0.0
                ),
                "min_difference": round(min(differences), 4),
                "max_difference": round(max(differences), 4),
                "mean_absolute_difference": round(
                    sum(absolute_differences) / len(absolute_differences), 4
                ),
                "median_absolute_difference": round(
                    statistics.median(absolute_differences), 4
                ),
                "stddev_absolute_difference": (
                    round(statistics.pstdev(absolute_differences), 4)
                    if len(absolute_differences) > 1
                    else 0.0
                ),
                "stddev_sentiment_label": (
                    round(statistics.pstdev(sentiment_scores), 4)
                    if len(sentiment_scores) > 1
                    else 0.0
                ),
                "stddev_gold_score": (
                    round(statistics.pstdev(gold_scores_for_rows), 4)
                    if len(gold_scores_for_rows) > 1
                    else 0.0
                ),
                "min_absolute_difference": round(min(absolute_differences), 4),
                "max_absolute_difference": round(max(absolute_differences), 4),
                "total_cost": "",
                "result_files": "",
                "input_tokens": "",
                "output_tokens": "",
                "total_tokens": "",
                "cost_per_result_file": "",
                "cost_per_scored_pair": "",
            }
        )
        cost_rows = cost_by_model.values() if model == "ALL" else [cost_by_model.get(model, {})]
        total_cost = sum(float(item.get("total_cost") or 0.0) for item in cost_rows)
        result_files = sum(int(item.get("result_files") or 0) for item in cost_rows)
        input_tokens = sum(int(item.get("input_tokens") or 0) for item in cost_rows)
        output_tokens = sum(int(item.get("output_tokens") or 0) for item in cost_rows)
        total_tokens = sum(int(item.get("total_tokens") or 0) for item in cost_rows)
        summary_rows[-1].update(
            {
                "total_cost": round(total_cost, 6),
                "result_files": result_files,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_per_result_file": (
                    round(total_cost / result_files, 6) if result_files else ""
                ),
                "cost_per_scored_pair": (
                    round(total_cost / len(differences), 6) if differences else ""
                ),
            }
        )
    if summary_rows:
        _write_csv(eval_dir / "sentiment_gold_summary.csv", summary_rows)
        with output_path.open("a", newline="", encoding="utf-8") as handle:
            handle.write("\nSUMMARY\n")
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    cost_summary_rows = _sentiment_cost_summary_rows(cost_by_model, scored_rows)
    if cost_summary_rows:
        with output_path.open("a", newline="", encoding="utf-8") as handle:
            handle.write("\nCOST SUMMARY\n")
            writer = csv.DictWriter(handle, fieldnames=list(cost_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cost_summary_rows)
    histogram_rows = _sentiment_difference_histogram_rows(scored_rows)
    if histogram_rows:
        _write_csv(eval_dir / "sentiment_difference_histogram.csv", histogram_rows)
    _write_sentiment_histogram_plots(eval_dir, scored_rows)
    print(f"Sentiment gold comparison written to {output_path}")
    return output_path


def rerun_invalid_sentiment_results(
    exp_dir: str | Path,
    ref_path: str | Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Rerun sentiment result files with parse errors or missing reference tickers."""
    exp_dir = Path(exp_dir).expanduser()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY must be set before rerunning sentiment results.")

    template = (exp_dir / "prompt.txt").read_text(encoding="utf-8")
    reference_tickers = load_reference_tickers(ref_path)
    input_dir = Path(os.path.expanduser(cfg["input_dir"]))
    results_dir = exp_dir / "results"
    log_path = exp_dir / "log.jsonl"

    rerun_targets: list[dict[str, Any]] = []
    for model in cfg["models"]:
        model_slug = _model_slug(model["id"])
        for doc_id, expected_tickers in sorted(reference_tickers.items()):
            result_file = results_dir / f"{model_slug}__{doc_id}.json"
            if not result_file.exists():
                rerun_targets.append(
                    {
                        "model": model,
                        "doc_id": doc_id,
                        "reason": "missing_result_file",
                        "result_file": result_file,
                        "expected_tickers": expected_tickers,
                    }
                )
                continue

            with result_file.open(encoding="utf-8") as handle:
                result_data = json.load(handle)
            sentiments, parse_error = parse_result_sentiments(result_data)
            missing_tickers = sorted(set(expected_tickers) - set(sentiments))
            if parse_error or missing_tickers:
                rerun_targets.append(
                    {
                        "model": model,
                        "doc_id": doc_id,
                        "reason": "parse_error" if parse_error else "missing_tickers",
                        "missing_tickers": missing_tickers,
                        "result_file": result_file,
                        "expected_tickers": expected_tickers,
                    }
                )

    rerun_records: list[dict[str, Any]] = []
    for target in rerun_targets:
        model = target["model"]
        doc_id = target["doc_id"]
        article_path = input_dir / f"{doc_id}.txt"
        text = article_path.read_text(encoding="utf-8")
        prompt = render_prompt(template, text, target["expected_tickers"])

        print(f"Rerunning {model['name']} x {doc_id} ({target['reason']})...")
        result = call_model(
            model["id"],
            prompt,
            cfg.get("temperature", 0),
            cfg.get("max_tokens", 4096),
            provider=model.get("provider"),
        )
        rerun_sentiments, rerun_parse_error = parse_result_sentiments(result)
        still_missing_tickers = sorted(set(target["expected_tickers"]) - set(rerun_sentiments))
        replaced_result = (
            not result.get("error")
            and not rerun_parse_error
            and not still_missing_tickers
        )

        result_file = target["result_file"]
        if replaced_result:
            with result_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

        log_entry = {
            "timestamp": utc_now_iso(),
            "model": model["id"],
            "model_name": model["name"],
            "doc_id": doc_id,
            "rerun": True,
            "rerun_reason": target["reason"],
            "missing_tickers": target.get("missing_tickers", []),
            "replaced_result": replaced_result,
            "still_missing_tickers": still_missing_tickers,
            "request": {
                "prompt": prompt,
                "temperature": cfg.get("temperature", 0),
                "max_tokens": cfg.get("max_tokens", 4096),
            },
            "response": result.get("response"),
            "content": result.get("content"),
            "latency_ms": result.get("latency_ms"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "finish_reason": result.get("finish_reason"),
            "error": result.get("error"),
        }
        append_jsonl(log_path, log_entry)
        rerun_records.append(
            {
                "model": model["id"],
                "doc_id": doc_id,
                "reason": target["reason"],
                "missing_tickers": target.get("missing_tickers", []),
                "replaced_result": replaced_result,
                "still_missing_tickers": still_missing_tickers,
                "finish_reason": result.get("finish_reason"),
                "error": result.get("error"),
            }
        )

    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    _write_json(
        eval_dir / "sentiment_reruns.json",
        {
            "rerun_count": len(rerun_records),
            "reruns": rerun_records,
        },
    )
    return {
        "rerun_count": len(rerun_records),
        "reruns": rerun_records,
    }


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
        path = Path(__file__).resolve().parents[1] / "prompts" / "canonical_subset_v2.txt"
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
        with Path(result_file).open(encoding="utf-8") as handle:
            result_data = json.load(handle)
        predicted, ticker_company_names, parse_error = parse_result_entities(result_data)
        actual = reference.get(doc_id, set())
        metrics = score(predicted, actual)
        metrics["model"] = model_slug
        metrics["doc_id"] = doc_id
        metrics["predicted"] = sorted(predicted)
        metrics["actual"] = sorted(actual)
        metrics["parse_error"] = parse_error
        metrics["input_tokens"] = result_data.get("input_tokens") or 0
        metrics["output_tokens"] = result_data.get("output_tokens") or 0
        metrics["cost"] = (result_data.get("response") or {}).get("usage", {}).get("cost") or 0.0
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
            handle.write("| Model | TP | FP | FN | Precision | Recall | F1 | Parse Errors | Docs | Cost/Doc ($) | Cost/Ticker ($) |\n")
            handle.write("|-------|----|----|----|-----------|--------|----|--------------|------|--------------|----------------|\n")
            for model, rows in sorted(results_by_model.items()):
                totals = aggregate_scores(rows)
                parse_errors = sum(1 for row in rows if row["parse_error"])
                total_cost = sum(row.get("cost", 0.0) for row in rows)
                n_docs = len(rows)
                total_tickers = totals["tp"] + totals["fn"]
                cost_per_doc = f"${total_cost / n_docs:.4f}" if n_docs and total_cost else "—"
                cost_per_ticker = f"${total_cost / total_tickers:.4f}" if total_tickers and total_cost else "—"
                handle.write(
                    f"| {model} | {totals['tp']} | {totals['fp']} | {totals['fn']} | "
                    f"{totals['precision']:.3f} | {totals['recall']:.3f} | {totals['f1']:.3f} | "
                    f"{parse_errors} | {n_docs} | {cost_per_doc} | {cost_per_ticker} |\n"
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


    print(f"Eval written to {eval_dir}")
    return eval_dir

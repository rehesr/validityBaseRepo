"""Compare model outputs against reference labels. Compute P/R/F1 per model."""

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


def load_reference(ref_path: str | Path) -> dict[str, set[str]]:
    """Return {doc_id: set_of_tickers}."""
    with Path(ref_path).open(encoding="utf-8") as handle:
        ref = json.load(handle)
    return {
        doc["doc_id"]: {label["ticker"] for label in doc["labels"] if label["label"] == 1}
        for doc in ref["documents"]
    }


def parse_model_tickers(result_path: str | Path, threshold: float = 0.5) -> tuple[set[str], bool]:
    """Extract ticker set from a model result file and flag parse failures."""
    with Path(result_path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    content = data.get("content", "")
    if not content:
        return set(), False
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        labels = json.loads(content)
        if isinstance(labels, dict):
            labels = labels.get("tickers")
        if not isinstance(labels, list):
            return set(), True

        tickers: set[str] = set()
        for label in labels:
            if not isinstance(label, dict):
                return set(), True

            ticker = label.get("ticker")
            confidence = label.get("confidence", 1.0)
            if not isinstance(ticker, str):
                return set(), True
            if not isinstance(confidence, (int, float)):
                return set(), True
            if confidence >= threshold:
                tickers.add(ticker)

        return tickers, False
    except (json.JSONDecodeError, KeyError, TypeError):
        return set(), True


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


def evaluate_experiment(exp_dir: str | Path, ref_path: str | Path) -> Path:
    exp_dir = Path(exp_dir)
    reference = load_reference(ref_path)

    results_by_model: dict[str, list[dict]] = defaultdict(list)
    for result_file in sorted(glob.glob(str(exp_dir / "results" / "*.json"))):
        filename = Path(result_file).stem
        model_slug, doc_id = filename.split("__", 1)
        predicted, parse_error = parse_model_tickers(result_file)
        actual = reference.get(doc_id, set())
        metrics = score(predicted, actual)
        metrics["model"] = model_slug
        metrics["doc_id"] = doc_id
        metrics["predicted"] = sorted(predicted)
        metrics["actual"] = sorted(actual)
        metrics["parse_error"] = parse_error
        results_by_model[model_slug].append(metrics)

    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    all_rows = [row for rows in results_by_model.values() for row in rows]
    if all_rows:
        with (eval_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    with (eval_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Evaluation Summary\n\n")
        handle.write("| Model | Avg P | Avg R | Avg F1 | Parse Errors | Docs |\n")
        handle.write("|-------|-------|-------|--------|--------------|------|\n")
        for model, rows in sorted(results_by_model.items()):
            n_rows = len(rows)
            avg_p = sum(row["precision"] for row in rows) / n_rows
            avg_r = sum(row["recall"] for row in rows) / n_rows
            avg_f1 = sum(row["f1"] for row in rows) / n_rows
            parse_errors = sum(1 for row in rows if row["parse_error"])
            handle.write(
                f"| {model} | {avg_p:.3f} | {avg_r:.3f} | {avg_f1:.3f} | {parse_errors} | {n_rows} |\n"
            )

    print(f"Eval written to {eval_dir}")
    return eval_dir

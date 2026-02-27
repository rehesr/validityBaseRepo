import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_markdown_history(log_path: str = "outputs/inference_log.jsonl", out_path: str = "outputs/inference_history.md") -> None:
    records = load_jsonl(Path(log_path))
    lines: List[str] = ["# Inference History", ""]
    lines.append(f"Total records: {len(records)}")
    lines.append("")

    for idx, rec in enumerate(records, start=1):
        result = rec.get("result", {})
        holdings_count = len(result.get("holdings", [])) if isinstance(result, dict) else 0
        exclusions_count = len(result.get("exclusions", [])) if isinstance(result, dict) else 0
        lines.append(f"## Record {idx}")
        lines.append(f"- timestamp_utc: {rec.get('timestamp_utc', '')}")
        lines.append(f"- file: {rec.get('file', '')}")
        lines.append(f"- model: {rec.get('model', '')}")
        lines.append(f"- prompt_path: {rec.get('prompt_path', '')}")
        lines.append(f"- holdings_count: {holdings_count}")
        lines.append(f"- exclusions_count: {exclusions_count}")
        lines.append("- raw_result:")
        lines.append("```json")
        lines.append(json.dumps(result, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_markdown_history()

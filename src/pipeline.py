import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.label_schema import HoldingsExtraction
from src.logging_utils import append_jsonl, utc_timestamp
from src.models.openrouter_client import OpenRouterClient
from src.prompting import build_json_only_messages


LOG_PATH = "outputs/inference_log.jsonl"


def _extract_content(response_json: Dict[str, Any]) -> str:
    return response_json["choices"][0]["message"]["content"]


def run_one(
    file_path: str,
    model: str = "openai/gpt-4o-mini",
    prompt_path: str = "prompts/holdings_extractor.txt",
) -> HoldingsExtraction:
    load_dotenv()
    input_path = Path(file_path)
    text = input_path.read_text(encoding="utf-8")

    client = OpenRouterClient()
    messages = build_json_only_messages(text=text, file_name=input_path.name, prompt_path=prompt_path)
    raw_response = client.chat_completions(messages=messages, model=model)
    content = _extract_content(raw_response)

    parsed = HoldingsExtraction.model_validate(json.loads(content))

    append_jsonl(
        LOG_PATH,
        {
            "timestamp_utc": utc_timestamp(),
            "file": str(file_path),
            "model": model,
            "prompt_path": prompt_path,
            "result": parsed.model_dump(),
        },
    )
    return parsed


def run_from_folder(
    folder_path: str,
    model: str = "openai/gpt-4o-mini",
    prompt_path: str = "prompts/holdings_extractor.txt",
) -> List[Dict[str, Any]]:
    folder = Path(folder_path)
    results: List[Dict[str, Any]] = []

    for file_path in sorted(folder.glob("*.txt")):
        result = run_one(str(file_path), model=model, prompt_path=prompt_path)
        results.append({"file": str(file_path), "result": result.model_dump()})

    return results

from pathlib import Path
from typing import Dict, List


def build_json_only_messages(text: str, file_name: str, prompt_path: str = "prompts/holdings_extractor.txt") -> List[Dict[str, str]]:
    template = Path(prompt_path).read_text(encoding="utf-8")
    rendered_prompt = template.replace("{{PDF_NAME}}", file_name).replace("{{PDF_TEXT}}", text)
    system_prompt = (
        "You are an extraction assistant. Respond with JSON only. "
        "No markdown, no code fences, no extra text."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rendered_prompt},
    ]

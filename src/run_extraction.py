import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from pdf_utils import extract_pdf_text
from router_client import OpenRouterClient


SYSTEM_PROMPT = "You are a precise financial document extraction assistant. Return only valid JSON."


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_model(models_cfg: dict, alias: str) -> str:
    model = models_cfg.get("models", {}).get(alias)
    if not model:
        raise ValueError(f"Unknown model alias '{alias}'. Add it in config/models.yaml")
    return model


def build_user_prompt(template: str, pdf_name: str, pdf_text: str) -> str:
    return template.replace("{{PDF_NAME}}", pdf_name).replace("{{PDF_TEXT}}", pdf_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenRouter extraction over a PDF dataset")
    parser.add_argument("--run-config", required=True, help="Path to run config yaml")
    parser.add_argument("--models-config", default="config/models.yaml", help="Path to models yaml")
    args = parser.parse_args()

    run_cfg_path = Path(args.run_config)
    models_cfg_path = Path(args.models_config)

    run_cfg = load_yaml(run_cfg_path)
    models_cfg = load_yaml(models_cfg_path)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")

    base_url = models_cfg.get("base_url", "https://openrouter.ai/api/v1")
    model_alias = run_cfg["model_alias"]
    model = resolve_model(models_cfg, model_alias)

    prompt_path = Path(run_cfg["prompt_file"])
    prompt_template = load_text(prompt_path)

    dataset_glob = run_cfg["dataset_glob"]
    pdf_paths = sorted(Path(".").glob(dataset_glob))
    if not pdf_paths:
        raise FileNotFoundError(f"No files matched dataset_glob: {dataset_glob}")

    run_name = run_cfg.get("run_name", run_cfg_path.stem)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("outputs") / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenRouterClient(api_key=api_key, base_url=base_url)

    summary: list[dict] = []

    for pdf_path in pdf_paths:
        pdf_text = extract_pdf_text(pdf_path)
        user_prompt = build_user_prompt(prompt_template, pdf_path.name, pdf_text)

        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(run_cfg.get("temperature", 0.0)),
            max_tokens=int(run_cfg.get("max_tokens", 1200)),
            extra_headers={
                "HTTP-Referer": models_cfg.get("http_referer", "https://github.com"),
                "X-Title": models_cfg.get("x_title", "validityBaseRepo"),
            },
        )

        raw_file = out_dir / f"{pdf_path.stem}.raw.json"
        raw_file.write_text(json.dumps(response, indent=2), encoding="utf-8")

        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        parsed_file = out_dir / f"{pdf_path.stem}.parsed.json"
        parsed_file.write_text(content + "\n", encoding="utf-8")

        summary.append(
            {
                "pdf": str(pdf_path),
                "model": model,
                "raw_output": str(raw_file),
                "parsed_output": str(parsed_file),
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "processed": len(summary)}, indent=2))


if __name__ == "__main__":
    main()

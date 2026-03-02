"""Load config, render prompt for each article, call each model, save everything."""

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.client import call_model


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)
    models_file = config_path.parent / "models.yaml"
    with models_file.open(encoding="utf-8") as handle:
        all_models = yaml.safe_load(handle)
    sweep["models"] = all_models[sweep["model_group"]]
    return sweep


def load_prompt(template_path: str | Path) -> str:
    with Path(template_path).open(encoding="utf-8") as handle:
        return handle.read()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_experiment_dir(cfg: dict[str, Any]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(os.path.expanduser(cfg["output_base"])) / f"{cfg['name']}_{ts}"


def freeze_artifacts(exp_dir: Path, cfg: dict[str, Any], template: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with (exp_dir / "prompt.txt").open("w", encoding="utf-8") as handle:
        handle.write(template)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_sweep(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    template = load_prompt(cfg["prompt"])

    exp_dir = create_experiment_dir(cfg)
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    freeze_artifacts(exp_dir, cfg, template)

    input_dir = os.path.expanduser(cfg["input_dir"])
    articles = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    log_path = exp_dir / "log.jsonl"

    for model in cfg["models"]:
        for art_path in articles:
            doc_id = Path(art_path).stem
            with open(art_path, encoding="utf-8") as handle:
                text = handle.read()

            prompt = template.replace("{{transcript}}", text)
            print(f"{model['name']} x {doc_id}...")
            result = call_model(
                model["id"],
                prompt,
                cfg.get("temperature", 0),
                cfg.get("max_tokens", 4096),
            )

            result_file = results_dir / f"{model['id'].replace('/', '_')}__{doc_id}.json"
            with result_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            log_entry = {
                "timestamp": utc_now_iso(),
                "model": model["id"],
                "model_name": model["name"],
                "doc_id": doc_id,
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

    print(f"\nDone. Results in {exp_dir}")
    return exp_dir

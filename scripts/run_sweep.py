import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sweep import load_config, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model sweep experiment.")
    parser.add_argument("config", help="Path to sweep config YAML")
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated OpenRouter model IDs, overriding the config's model_group. "
            "Example: --models google/gemini-2.0-flash,anthropic/claude-3-5-haiku-20241022"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        cfg["models"] = [
            {"id": mid, "name": mid.split("/")[-1] if "/" in mid else mid}
            for mid in model_ids
        ]

    run_sweep(config_path, cfg=cfg)


if __name__ == "__main__":
    main()

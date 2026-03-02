from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_experiment


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_eval.py ~/data/tickerization/experiments/<folder>")
    exp_dir = Path(sys.argv[1]).expanduser()
    with (exp_dir / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    evaluate_experiment(exp_dir, Path(cfg["reference"]).expanduser())


if __name__ == "__main__":
    main()

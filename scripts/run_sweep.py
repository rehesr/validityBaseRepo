from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sweep import run_sweep


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_sweep.py configs/flagship_sweep.yaml")
    run_sweep(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

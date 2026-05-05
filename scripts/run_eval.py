import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Path to experiment folder")
    parser.add_argument(
        "--build-canonical",
        action="store_true",
        help="Alias for --canonical-stage full",
    )
    parser.add_argument(
        "--canonical-stage",
        choices=[
            "first-pass",
            "second-pass",
            "second-pass-csv",
            "full",
            "first-pass-actual-second-pass",
        ],
        default=None,
        help=(
            "Canonical mode: first-pass writes queue, second-pass consumes queue, "
            "second-pass-csv writes CSV from existing second-pass artifacts, "
            "first-pass-actual-second-pass fixes actual from first-pass consensus and "
            "scores second-pass outputs against it, full runs both canonical passes."
        ),
    )
    parser.add_argument(
        "--second-pass-prompt",
        default="prompts/canonical_subset_v1.txt",
        help="Prompt template path for second-stage canonical labeling",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).expanduser()
    with (exp_dir / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.build_canonical and args.canonical_stage:
        raise SystemExit("Use either --build-canonical or --canonical-stage, not both.")

    canonical_stage = args.canonical_stage
    if args.build_canonical:
        canonical_stage = "full"

    evaluate_experiment(
        exp_dir=exp_dir,
        ref_path=Path(cfg["reference"]).expanduser(),
        cfg=cfg,
        build_canonical=canonical_stage is not None,
        second_pass_prompt_path=args.second_pass_prompt,
        canonical_mode=canonical_stage or "full",
    )


if __name__ == "__main__":
    main()

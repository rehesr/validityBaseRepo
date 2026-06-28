import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (
    build_sentiment_gold_comparison,
    build_sentiment_consensus,
    evaluate_experiment,
    rerun_invalid_sentiment_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Path to experiment folder")
    parser.add_argument(
        "--reference",
        default=None,
        help="Path to reference JSON, overriding the reference field in config.yaml.",
    )
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
        default="prompts/canonical_subset_v2.txt",
        help="Prompt template path for second-stage canonical labeling",
    )
    parser.add_argument(
        "--sentiment-consensus",
        action="store_true",
        help="Build consensus sentiment labels from a sentiment sweep.",
    )
    parser.add_argument(
        "--sentiment-gold-comparison",
        action="store_true",
        help="Write per-model sentiment score deltas against a gold sentiment consensus.",
    )
    parser.add_argument(
        "--sentiment-gold",
        default=None,
        help=(
            "Path to a flagship sentiment experiment folder, "
            "eval/sentiment_consensus.json, or eval/sentiment_consensus.csv."
        ),
    )
    parser.add_argument(
        "--rerun-invalid-sentiments",
        action="store_true",
        help=(
            "Rerun sentiment model/document results with parse errors or missing "
            "reference ticker scores, then rebuild sentiment consensus."
        ),
    )
    parser.add_argument(
        "--sentiment-agree-max-abs-dev",
        type=float,
        default=0.50,
        help=(
            "Accept consensus when the max absolute difference among the "
            "three retained scores is at or below this."
        ),
    )
    parser.add_argument(
        "--sentiment-review-max-abs-dev",
        type=float,
        default=0.50,
        help=(
            "Flag human review when the max absolute difference among the "
            "three retained scores is above this."
        ),
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

    if args.reference:
        ref_path = Path(args.reference).expanduser()
    else:
        ref_path = Path(cfg["reference"]).expanduser()
        if not ref_path.is_absolute():
            ref_path = exp_dir / ref_path

    if args.sentiment_gold_comparison:
        if not args.sentiment_gold:
            raise SystemExit("--sentiment-gold-comparison requires --sentiment-gold.")
        build_sentiment_gold_comparison(
            exp_dir=exp_dir,
            gold_path=args.sentiment_gold,
        )
        return

    if args.sentiment_consensus or args.rerun_invalid_sentiments:
        if args.rerun_invalid_sentiments:
            rerun_invalid_sentiment_results(
                exp_dir=exp_dir,
                ref_path=ref_path,
                cfg=cfg,
            )
        build_sentiment_consensus(
            exp_dir=exp_dir,
            ref_path=ref_path,
            agree_max_abs_dev=args.sentiment_agree_max_abs_dev,
            review_max_abs_dev=args.sentiment_review_max_abs_dev,
        )
        return

    evaluate_experiment(
        exp_dir=exp_dir,
        ref_path=ref_path,
        cfg=cfg,
        build_canonical=canonical_stage is not None,
        second_pass_prompt_path=args.second_pass_prompt,
        canonical_mode=canonical_stage or "full",
    )


if __name__ == "__main__":
    main()

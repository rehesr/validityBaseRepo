# Tickerization Benchmark

This repo benchmarks LLM-based company-to-ticker extraction over transcript or article text files using OpenRouter.

The workflow is:

1. Put `.txt` inputs and a reference label file in an external data directory.
2. Run a sweep across a model group and prompt.
3. Save every model response in a timestamped experiment folder.
4. Evaluate model predictions against the reference labels.

## What This Repo Contains

- `configs/models.yaml`
  Named model groups for sweeps.
- `configs/flagship_sweep.yaml`
  A default high-end sweep configuration.
- `configs/light_sweep.yaml`
  A lighter and cheaper sweep configuration.
- `configs/ensemble_sweep.yaml`
  A Qwen ensemble sweep configuration.
- `configs/fast_sweep.yaml`
  A faster lower-cost sweep configuration.
- `configs/frontier_open_weights_sweep.yaml`
  A frontier open-weights sweep configuration.
- `prompts/tickerize_v1.txt`
  Original prompt template.
- `prompts/tickerize_v2.txt`
  Updated prompt template for explicit comparison-style ticker mentions.
- `src/client.py`
  Thin OpenRouter client using the OpenAI SDK.
- `src/sweep.py`
  Sweep runner: load config, render prompt, call models, save results.
- `src/evaluate.py`
  Evaluation logic: parse predictions, compare to reference labels, write reports.
- `scripts/run_sweep.py`
  CLI entrypoint for sweeps.
- `scripts/run_eval.py`
  CLI entrypoint for evaluation.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export your OpenRouter key in the shell before running sweeps:

```bash
export OPENROUTER_API_KEY='sk-or-...'
```

The code reads `OPENROUTER_API_KEY` from the shell environment. It does not auto-load `.env`.

## External Data Layout

Inputs and outputs live outside the repo.

Expected layout:

```text
~/data/tickerization/
├── inputs/
│   ├── ticker_articles/
│   │   ├── ticker_article_0001.txt
│   │   ├── ticker_article_0002.txt
│   │   └── ...
│   └── reference.json
└── experiments/
    └── flagship_YYYYMMDD_HHMMSS/
        ├── config.yaml
        ├── prompt.txt
        ├── log.jsonl
        ├── results/
        └── eval/
```

`reference.json` is expected to look like:

```json
{
  "documents": [
    {
      "doc_id": "ticker_article_0001",
      "labels": [
        { "ticker": "AMAT", "label": 1 },
        { "ticker": "LRCX", "label": 1 }
      ]
    }
  ]
}
```

## Running A Sweep

Run the flagship sweep:

```bash
python scripts/run_sweep.py configs/flagship_sweep.yaml
```

Run the lighter sweep:

```bash
python scripts/run_sweep.py configs/light_sweep.yaml
```

Run the ensemble sweep:

```bash
python scripts/run_sweep.py configs/ensemble_sweep.yaml
```

Run the faster sweep:

```bash
python scripts/run_sweep.py configs/fast_sweep.yaml
```

Run the frontier open-weights sweep:

```bash
python scripts/run_sweep.py configs/frontier_open_weights_sweep.yaml
```

Each run creates a new folder under `~/data/tickerization/experiments/`.

## Evaluating A Sweep

Run evaluation on a completed experiment folder:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339
```

Optional: build canonical labels with a second-pass union-candidate prompt for unresolved docs:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 --build-canonical
```

Canonical build behavior is automatic:

- First pass: accept tickers with `>= 3/4` votes, discard tickers with only `1/4` votes, and queue only `2/4` tickers for adjudication.
- Second pass: queued docs are re-prompted with `prompts/canonical_subset_v1.txt` using only the `2/4` candidate tickers.
- Remaining unresolved docs are written to the manual queue.

You can also run step 1 and step 2 separately:

```bash
# Step 1 only: build first-pass canonical labels and queue unresolved docs
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 \
  --canonical-stage first-pass

# Step 2 only: consume eval/second_pass_queue.json and finish canonical build
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 \
  --canonical-stage second-pass

# Step 2 CSV only: write CSV from existing second-pass artifacts
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 \
  --canonical-stage second-pass-csv
```

If you want the one-command scoring workflow instead, run:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 \
  --canonical-stage first-pass-actual-second-pass
```

That writes both:

- `eval/first_pass_actual_second_pass_scores.csv`
- `eval/all_docs_first_second_pass_comparison.csv`

This writes:

- `eval/scores.csv`
- `eval/summary.md`
- `eval/canonical_first_pass.json` (with `--canonical-stage first-pass` or `--build-canonical`)
- `eval/second_pass_queue.json` (with `--canonical-stage first-pass` or `--build-canonical`)
- `eval/canonical_reference.json` (with `--canonical-stage second-pass` or `--build-canonical`)
- `eval/manual_label_queue.json` (with `--canonical-stage second-pass` or `--build-canonical`)
- `eval/second_pass_outcomes.csv` (with `--canonical-stage second-pass`, `--canonical-stage second-pass-csv`, or `--build-canonical`)
- `eval/all_docs_first_second_pass_comparison.csv` (with `--canonical-stage second-pass`, `--canonical-stage second-pass-csv`, or `--build-canonical`)
- `eval/second_pass_results/` (with `--canonical-stage second-pass` or `--build-canonical`)

If the configured `reference.json` is empty, `eval/summary.md` will not report
model-vs-reference precision/recall/F1. Instead it will say:

```md
# Evaluation Summary

Reference file is empty, so direct model-vs-reference metrics are omitted.
```

If canonical labeling is run in the same evaluation command, the summary will
still include the `## Canonical Labeling` section below that notice.

## Output Format Notes

This repo currently supports two prediction output shapes during evaluation:

Raw list:

```json
[
  { "ticker": "AMAT", "confidence": 0.95 }
]
```

Object wrapper:

```json
{
  "tickers": [
    { "ticker": "AMAT", "confidence": 0.95 }
  ]
}
```

The evaluator handles both.

It also applies a small static ticker consolidation map for known duplicate listings
and share classes during parsing and scoring, for example `GOOGL -> GOOG`,
`BRK.B -> BRK.A`, `NWSA -> NWS`, and `RDSA -> RDS.A`.

## Inspecting Results

Open an experiment summary:

```bash
cat ~/data/tickerization/experiments/flagship_20260302_214339/eval/summary.md
```

Open per-document scores:

```bash
open ~/data/tickerization/experiments/flagship_20260302_214339/eval/scores.csv
```

Inspect raw model outputs:

```bash
ls ~/data/tickerization/experiments/flagship_20260302_214339/results
```

## Common Failure Modes

- `ModuleNotFoundError: No module named 'src'`
  Run the documented `python scripts/...` commands from the repo root.
- Empty or trivial predictions across all models
  Check the frozen `prompt.txt` in the experiment folder and confirm the transcript placeholder matched the sweep renderer.
- Evaluation crashes on model output shape
  The current evaluator supports both list outputs and `{ "tickers": [...] }` outputs.
- Bad metrics with apparently good predictions
  Check whether `reference.json` is empty, incomplete, or uses mismatched `doc_id` values.

## Prompt Notes

`prompts/tickerize_v2.txt` is the stronger default if your transcripts frequently contain comparison-style phrasing such as:

- `names like AMAT, LRCX, ASML, MU`
- `across names like WM, RSG, GFL, WCN`

It explicitly tells the model to treat those as valid positive mentions.

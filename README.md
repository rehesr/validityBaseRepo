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
│   ├── articles/
│   │   ├── article_1.txt
│   │   ├── article_2.txt
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
      "doc_id": "article_1",
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

Each run creates a new folder under `~/data/tickerization/experiments/`.

## Evaluating A Sweep

Run evaluation on a completed experiment folder:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339
```

This writes:

- `eval/scores.csv`
- `eval/summary.md`

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

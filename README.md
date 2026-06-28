# Tickerization Benchmark

This repo benchmarks LLM-based company-to-ticker extraction over transcript or article text files using OpenRouter. It is designed for running the same corpus through several models, saving the raw responses, and comparing those responses against a reference label file.

There are two related workflows in this repository. The first is the tickerization workflow: given a document, the model identifies which publicly traded companies are explicitly mentioned and returns their stock tickers. The second is the sentiment labeling workflow: given a document and a known list of reference tickers for that document, the model assigns a sentiment score to each ticker. Both workflows use the same sweep runner; the config and prompt determine which task is performed.

The workflow is:

1. Put `.txt` inputs and a reference label file in an external data directory.
2. Run a sweep across a model group and prompt.
3. Save every model response in a timestamped experiment folder.
4. Evaluate model predictions against the reference labels.

A sweep is intentionally append-only at the experiment level. Every run creates a new timestamped folder, freezes the resolved config and prompt into that folder, writes one result file per model/document pair, and appends request/response metadata to `log.jsonl`. This makes each experiment auditable after the fact.

## What This Repo Contains

The repository is organized around a small canonical runtime path. The `scripts/` files are the command-line entrypoints, while `src/` contains the implementation. Prompt wording lives in `prompts/`, and model/sweep choices live in `configs/`. There is also an optional Streamlit dashboard for interactive inspection and ad hoc runs, but the command-line scripts are the source of truth for benchmark sweeps.

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
- `configs/sentiment_*_sweep.yaml`
  Sweep configurations for generating sentiment scores for reference tickers.
- `prompts/tickerize_v1.txt`
  Original prompt template.
- `prompts/tickerize_v2.txt`
  Updated prompt template for explicit comparison-style ticker mentions.
- `prompts/tickerize_v0.txt`
  Legacy minimal ticker extraction prompt.
- `prompts/sentiment_labeling.txt`
  Sentiment scoring prompt used by `configs/sentiment_*_sweep.yaml`.
- `prompts/canonical_subset_v2.txt`
  Second-pass canonical labeling prompt for split ticker decisions.
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
- `dashboard.py`
  Optional Streamlit dashboard for running small interactive extractions and browsing experiment results.

## Setup

The main benchmark path is a small Python CLI. It does not require a database or a local service. You only need Python dependencies, text inputs, a reference JSON file, and an OpenRouter API key for any command that calls models.

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

The CLI code reads `OPENROUTER_API_KEY` from the shell environment. It does not auto-load `.env`. The optional Streamlit dashboard can load and save a repo-local `.env` file if `python-dotenv` is installed.

## External Data Layout

Inputs and outputs live outside the repo. This is deliberate: the repository should only contain code, prompts, and configs. Dataset files and experiment outputs can be large, private, or frequently changing, so they belong under `~/data/tickerization/` instead of being committed.

Expected layout:

```text
~/data/tickerization/
├── inputs/
│   ├── articles/
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

The `doc_id` should match the input filename without the `.txt` extension. For example, `~/data/tickerization/inputs/articles/ticker_article_0001.txt` should use `"doc_id": "ticker_article_0001"` in `reference.json`.

For tickerization evaluation, labels with `"label": 1` are treated as the positive tickers that should be found in the document. The sentiment workflow also uses those positive tickers as the ticker list to score for each document.

## Running A Sweep

A sweep loads one YAML config, resolves its model group from `configs/models.yaml`, reads every `.txt` file in the configured input directory, renders the configured prompt, and calls every configured model on every document. The output is a new experiment folder under `output_base`.

The standard tickerization configs use `prompts/tickerize_v2.txt`. Use these when you want models to extract tickers from raw documents and then compare those predicted tickers against `reference.json`.

Configs normally choose models through `model_group`, but the sweep loader also accepts an inline `models` list in a config file. The `--models` CLI override is a convenient way to create that inline model list at runtime.

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

Override the config's model group with explicit OpenRouter model IDs:

```bash
python scripts/run_sweep.py configs/light_sweep.yaml \
  --models google/gemini-2.0-flash,anthropic/claude-3-5-haiku-20241022
```

Use the model override for quick one-off tests without editing `configs/models.yaml`. The experiment still freezes the resolved model list into its `config.yaml`, so the run remains reproducible.

The sentiment configs use the same sweep runner, but they point at `prompts/sentiment_labeling.txt`. In this mode, the runner first looks up each document's positive tickers in `reference.json`, inserts them into the prompt, and asks the model to score sentiment for those tickers.

Generate sentiment labels for the tickers in `reference.json`:

```bash
python scripts/run_sweep.py configs/sentiment_flagship_sweep.yaml
```

Use `configs/sentiment_light_sweep.yaml`, `configs/sentiment_fast_sweep.yaml`,
or `configs/sentiment_frontier_sweep.yaml` for the other model groups. These
runs create folders such as `sentiment_flagship_YYYYMMDD_HHMMSS` or
`sentiment_fast_YYYYMMDD_HHMMSS`.

The sentiment sweep uses `prompts/sentiment_labeling.txt`. For each input
document, it fills `{{tickers}}` from that document's positive labels in
`reference.json`, then writes each model response under the experiment's
`results/` directory.

After a sentiment sweep finishes, compute per-document/ticker consensus:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/sentiment_flagship_YYYYMMDD_HHMMSS \
  --sentiment-consensus
```

The default consensus threshold accepts a retained three-score tuple when its
maximum pairwise score difference is `<= 0.50`. You can change the accept and
review thresholds with `--sentiment-agree-max-abs-dev` and
`--sentiment-review-max-abs-dev`.

If any model response was malformed, truncated, or omitted reference tickers,
rerun only those failed model/document results and rebuild consensus:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/sentiment_flagship_YYYYMMDD_HHMMSS \
  --rerun-invalid-sentiments
```

This rerun mode is the one evaluation path that can update files outside `eval/`: it replaces a result file only when the replacement parses cleanly and includes all expected tickers, and it appends a rerun record to the experiment's `log.jsonl`.

Consensus uses four model observations per document/ticker pair. It computes
the mean of all observations, removes the score farthest from that mean, then
uses the mean of the three remaining observations as the sentiment label only
when the tuple reaches consensus.
Mean confidence, trimmed score standard deviation, trimmed max absolute
deviation from the mean, and trimmed max absolute pairwise difference are
written as quality signals. Pairs with trimmed max absolute difference
`<= 0.50` are accepted into `eval/sentiment_consensus.json` with a gold score;
pairs with trimmed max absolute difference `> 0.50`, or pairs with missing scores, are flagged for human review without a generated score in
`eval/sentiment_review_queue.json`. The full pair-level table is written to
`eval/sentiment_consensus.csv`; it includes `gold_score`, `raw_scores`,
`retained_scores`, `removed_model`, and `removed_score` columns so the final
score and dropped observation are auditable. `eval/sentiment_summary.md` reports the unique article count left
for review. Rerun details, when requested, are written to
`eval/sentiment_reruns.json`.

To compare another sentiment sweep against gold scores produced by a previous
flagship consensus run:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/sentiment_fast_YYYYMMDD_HHMMSS \
  --sentiment-gold-comparison \
  --sentiment-gold ~/data/tickerization/experiments/sentiment_flagship_YYYYMMDD_HHMMSS
```

This writes research tables in the evaluated sweep folder:

- `eval/sentiment_gold_comparison.csv`: row-level ticker/article/model scores,
  flagship gold score, signed difference, absolute difference, result cost, and
  token counts. The CSV also appends `SUMMARY` and `COST SUMMARY` tables.
- `eval/sentiment_gold_summary.csv`: per-model and overall difference stats,
  raw-score standard deviations, token totals, total cost, cost per result
  file, and cost per scored pair.
- `eval/sentiment_difference_histogram.csv`: signed and absolute difference
  histograms by model and overall.
- `eval/sentiment_difference_histogram.png` and
  `eval/sentiment_absolute_difference_histogram.png`: presentation-ready
  histogram plots by model and overall.

`--sentiment-gold` can point to a flagship experiment folder,
`eval/sentiment_consensus.json`, or `eval/sentiment_consensus.csv`.

Each run creates a new folder under `~/data/tickerization/experiments/`.

## Evaluating A Sweep

Evaluation reads an existing experiment folder. It does not call models unless you ask for a canonical second pass, the first-pass-actual second-pass scoring mode, or a sentiment rerun. The normal evaluation path parses every file in `results/`, compares predicted tickers to the frozen reference path from `config.yaml`, and writes reports under `eval/`.

Use the basic evaluation command after a tickerization sweep. The most important outputs are `eval/scores.csv`, which contains per-document/per-model rows, and `eval/summary.md`, which contains aggregate precision, recall, F1, parse-error counts, and cost summaries when OpenRouter usage cost is present.

Run evaluation on a completed experiment folder:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339
```

Override the frozen `reference` path from `config.yaml` when needed:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 \
  --reference ~/data/tickerization/inputs/reference.json
```

Optional: build canonical labels with a second-pass union-candidate prompt for unresolved docs:

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260302_214339 --build-canonical
```

By default, second-pass canonical labeling uses
`prompts/canonical_subset_v2.txt`. To test another adjudication prompt, pass
`--second-pass-prompt path/to/prompt.txt`.

Canonical build behavior is automatic:

- First pass: accept tickers with `>= 3/4` votes, discard tickers with only `1/4` votes, and queue only `2/4` tickers for adjudication.
- Second pass: queued docs are re-prompted with `prompts/canonical_subset_v2.txt` using only the `2/4` candidate tickers.
- Remaining unresolved docs are written to the manual queue.

Canonical labeling is useful when you want to build a consensus-derived label file from the model outputs themselves. The first pass accepts strong agreement, rejects one-off model mentions, and sends only split decisions to a second adjudication prompt. This keeps the second pass focused on ambiguous ticker candidates rather than asking the model to re-extract everything from scratch.

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
- `eval/first_pass_actual_second_pass_results/`
- `eval/first_pass_actual_second_pass_log.jsonl`
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
- `eval/second_pass_log.jsonl` (with `--canonical-stage second-pass` or `--build-canonical`)

If the configured `reference.json` is empty, `eval/summary.md` will not report
model-vs-reference precision/recall/F1. Instead it will say:

```md
# Evaluation Summary

Reference file is empty, so direct model-vs-reference metrics are omitted.
```

If canonical labeling is run in the same evaluation command, the summary will
still include the `## Canonical Labeling` section below that notice.

## Output Format Notes

Model prompts have changed over time, and different models sometimes wrap JSON differently. The evaluator is intentionally tolerant of the two shapes below so older and newer experiment outputs can be scored by the same code.

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

The raw experiment folder is usually the best place to debug behavior. `prompt.txt` shows exactly what prompt template was used. `config.yaml` shows the resolved models and paths. `log.jsonl` contains one request/response record per API call. `results/` contains the standalone JSON payload for each model/document pair.

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

## Optional Dashboard

The repository includes `dashboard.py`, a Streamlit app for interactive tickerization work. It has three tabs:

- `Extract`: run ticker extraction or sentiment labeling for one pasted/uploaded article, one model group, or a small batch sweep from a local input directory or uploaded zip. Sentiment batch sweeps require a `reference.json` upload so the dashboard can fill each document's ticker list.
- `Sweep Results`: load experiment folders from disk or uploaded zips, run tickerization evaluation from an uploaded `reference.json`, build sentiment consensus for sentiment sweeps, and inspect per-model and per-document results.
- `Leaderboard`: compare evaluated experiments and summarize the best F1 scores by model.

Dashboard-created experiment folders include `config.yaml`, `prompt.txt`, `log.jsonl`, and `results/` so they can be inspected like script-created experiments. The dashboard is useful for exploration and review, but the reproducible benchmark path remains `scripts/run_sweep.py` followed by `scripts/run_eval.py`.

Sentiment gold-comparison research tables are still available through `scripts/run_eval.py`.

Run it with:

```bash
streamlit run dashboard.py
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

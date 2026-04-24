# AGENTS

## Purpose

This repo runs tickerization benchmark sweeps over transcript or article text files using OpenRouter models.

## Repo Contract

- Keep the codebase small and explicit.
- The canonical runtime path is:
  - `scripts/run_sweep.py`
  - `scripts/run_eval.py`
- Core logic belongs only in:
  - `src/client.py`
  - `src/sweep.py`
  - `src/evaluate.py`
- Prompt templates live in `prompts/`.
- Sweep and model definitions live in `configs/`.

## Data Contract

- Input text files live outside the repo under `~/data/tickerization/inputs/articles/`.
- Ground-truth labels live at `~/data/tickerization/inputs/reference.json`.
- Experiment outputs live under `~/data/tickerization/experiments/`.
- Do not commit dataset files or experiment outputs into this repository.

## Sweep Contract

- `configs/models.yaml` defines named model groups.
- `configs/*.yaml` sweep files choose the prompt, model group, input path, reference path, and output base.
- Each run must freeze the resolved config and prompt into the experiment folder.
- Each API call must append one record to `log.jsonl`.
- Each model-document pair must write one file under `results/`.

## Evaluation Contract

- Evaluation reads an existing experiment folder and the frozen `reference` path from `config.yaml`.
- Evaluation writes only to the experiment's `eval/` directory.

## Editing Rules

- Prefer minimal changes over abstraction.
- Do not reintroduce the old pipeline, experiment, or output-folder structures that were removed.
- Keep README instructions aligned with the actual CLI and file layout.

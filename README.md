# Tickerization Benchmark

Sweep a tickerization prompt across transcript or article text files and compare model results via OpenRouter.

## Repo Layout

- `prompts/tickerize_v1.txt`: prompt template with `{{transcript}}`
- `configs/models.yaml`: named model groups
- `configs/flagship_sweep.yaml`: sweep definition
- `src/client.py`: thin OpenRouter wrapper
- `src/sweep.py`: load config, call models, save outputs
- `src/evaluate.py`: score results against a reference file
- `scripts/run_sweep.py`: CLI entrypoint for sweeps
- `scripts/run_eval.py`: CLI entrypoint for evaluation

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

## Data Layout

```text
~/data/tickerization/
├── inputs/
│   ├── articles/
│   │   ├── art_001.txt
│   │   └── ...
│   └── reference.json
└── experiments/
    └── flagship_20260301_143000/
        ├── config.yaml
        ├── prompt.txt
        ├── log.jsonl
        ├── results/
        └── eval/
```

## Run

```bash
python scripts/run_sweep.py configs/flagship_sweep.yaml
```

This creates a timestamped experiment folder under `~/data/tickerization/experiments/`.

## Evaluate

```bash
python scripts/run_eval.py ~/data/tickerization/experiments/flagship_20260301_143000
```

This writes:

- `eval/scores.csv`
- `eval/summary.md`

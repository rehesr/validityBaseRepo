# OpenRouter Labeling Pipeline

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Run

```bash
python -c "from src.pipeline import run_from_folder; print(run_from_folder('data/raw'))"
```

Logs are written to `outputs/inference_log.jsonl`.

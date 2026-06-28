"""Streamlit dashboard for the tickerization benchmark."""

import ast
import io
import json
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import yaml
from openai import OpenAI

from src.client import call_model as _sweep_call_model
from src.evaluate import (
    _extract_tickers_and_names,
    _parse_labels_content,
    aggregate_scores,
    build_sentiment_consensus,
    canonicalize_ticker,
    evaluate_experiment,
    parse_result_sentiments,
)

REPO_ROOT = Path(__file__).parent
ENV_FILE = REPO_ROOT / ".env"

# Load saved API key from .env if present
try:
    from dotenv import load_dotenv, set_key as _dotenv_set_key
    load_dotenv(ENV_FILE)
    _dotenv_available = True
except ImportError:
    _dotenv_available = False


# ── helpers ───────────────────────────────────────────────────────────────────


def load_models() -> dict[str, list[dict]]:
    path = REPO_ROOT / "configs" / "models.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    path = REPO_ROOT / "prompts" / name
    with path.open(encoding="utf-8") as f:
        return f.read()


def reference_tickers_from_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    tickers_by_doc: dict[str, list[str]] = {}
    for doc in payload.get("documents", []):
        doc_id = doc.get("doc_id")
        labels = doc.get("labels", [])
        if not isinstance(doc_id, str) or not isinstance(labels, list):
            continue
        tickers = {
            canonicalize_ticker(label["ticker"])
            for label in labels
            if isinstance(label, dict)
            and label.get("label") == 1
            and isinstance(label.get("ticker"), str)
            and label["ticker"].strip()
        }
        tickers_by_doc[doc_id] = sorted(tickers)
    return tickers_by_doc


def resolve_config_path(exp_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else exp_dir / path


def list_experiments(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()],
        key=lambda p: p.name,
        reverse=True,
    )


def load_scores_csv(exp_dir: Path) -> pd.DataFrame | None:
    path = exp_dir / "eval" / "scores.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "parse_error" in df.columns:
        df["parse_error"] = df["parse_error"].map(
            {"True": True, "False": False, True: True, False: False}
        ).fillna(False)
    return df


def load_run_costs(exp_dir: Path) -> dict[str, float]:
    """Scan results/*.json and sum response.usage.cost per model slug."""
    costs: defaultdict[str, float] = defaultdict(float)
    results_dir = exp_dir / "results"
    if not results_dir.exists():
        return {}
    for result_file in results_dir.glob("*.json"):
        parts = result_file.stem.split("__", 1)
        if len(parts) != 2:
            continue
        model_slug = parts[0]
        try:
            with result_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        cost = (data.get("response") or {}).get("usage", {}).get("cost") or 0.0
        costs[model_slug] += cost
    return dict(costs)


def model_summary_df(
    df: pd.DataFrame,
    run_costs: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model"):
        agg = aggregate_scores(group.to_dict("records"))
        row: dict = {
            "model": model,
            "F1": agg["f1"],
            "Precision": agg["precision"],
            "Recall": agg["recall"],
            "TP": agg["tp"],
            "FP": agg["fp"],
            "FN": agg["fn"],
            "Docs": len(group),
            "Parse Errors": int(group["parse_error"].sum()),
        }
        if run_costs is not None and model in run_costs:
            total_cost = run_costs[model]
            total_tickers = agg["tp"] + agg["fn"]
            row["Cost/Doc ($)"] = round(total_cost / len(group), 4) if len(group) else None
            row["Cost/Ticker ($)"] = round(total_cost / total_tickers, 4) if total_tickers else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def call_model_direct(
    api_key: str,
    model_id: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.time() - t0) * 1000)
        usage = response.usage
        usage_dict = usage.model_dump(mode="json") if usage else {}
        return {
            "latency_ms": latency_ms,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "cost": usage_dict.get("cost"),
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "latency_ms": int((time.time() - t0) * 1000),
            "content": None,
            "finish_reason": None,
            "input_tokens": None,
            "output_tokens": None,
            "cost": None,
            "error": str(exc),
        }


def df_download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tickerization Dashboard", layout="wide")
st.title("Tickerization Dashboard")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "OpenRouter API Key",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
        help="Required for Extract and Run Sweep tabs.",
    )
    if st.button("Save API Key", key="save_api_key"):
        if _dotenv_available:
            ENV_FILE.touch(exist_ok=True)
            _dotenv_set_key(str(ENV_FILE), "OPENROUTER_API_KEY", api_key)
            os.environ["OPENROUTER_API_KEY"] = api_key
            st.success("Saved to .env")
        else:
            st.warning("Install python-dotenv to enable key persistence.")

    exp_base_str = st.text_input(
        "Experiments directory",
        value="",
        help="Base directory where sweep results are stored.",
    )
    exp_base = Path(exp_base_str).expanduser()

    st.divider()
    _zip_uploads = st.file_uploader(
        "Load experiments from zip",
        type=["zip"],
        accept_multiple_files=True,
        help="Upload experiment zips downloaded from a sweep to use in Sweep Results and Leaderboard.",
    )
    if "_loaded_exp_dirs" not in st.session_state:
        st.session_state["_loaded_exp_dirs"] = {}
    for _uz in (_zip_uploads or []):
        if _uz.name not in st.session_state["_loaded_exp_dirs"]:
            _tmp = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(io.BytesIO(_uz.read())) as _zf:
                _zf.extractall(_tmp)
            for _d in sorted(_tmp.iterdir()):
                if _d.is_dir() and (_d / "config.yaml").exists():
                    st.session_state["_loaded_exp_dirs"][_uz.name] = _d
                    break

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_extract, tab_results, tab_leaderboard = st.tabs(
    ["Extract", "Sweep Results", "Leaderboard"]
)

# ── Tab 1: Extract ────────────────────────────────────────────────────────────

with tab_extract:
    all_models = load_models()

    # ── Shared controls ───────────────────────────────────────────────────────
    sh_col1, sh_col2, sh_col3 = st.columns(3)
    with sh_col1:
        model_group = st.selectbox("Model group", list(all_models.keys()))
    model_options = all_models[model_group]
    with sh_col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, key="ex_temp")
    with sh_col3:
        max_tokens = st.number_input("Max tokens", 256, 12000, 4096, 256, key="ex_max_tokens")

    custom_model_ids = st.text_input(
        "Custom model IDs (overrides model group)",
        value="",
        placeholder="provider/model-name, provider/model-name",
        help="Comma-separated OpenRouter model IDs. When set, overrides the model group above.",
        key="ex_custom_models",
    )
    if custom_model_ids.strip():
        _custom_ids = [m.strip() for m in custom_model_ids.split(",") if m.strip()]
        model_options = [
            {"id": mid, "name": mid.split("/")[-1] if "/" in mid else mid}
            for mid in _custom_ids
        ]
        model_group = "custom"

    task = st.radio(
        "Task",
        ["Ticker extraction", "Sentiment labeling"],
        horizontal=True,
        key="extract_task",
    )
    is_sentiment_task = task == "Sentiment labeling"
    _default_prompt = load_prompt("sentiment_labeling.txt" if is_sentiment_task else "tickerize_v1.txt")
    prompt_help = (
        "`{{transcript}}` is replaced with article text; `{{tickers}}` is replaced with the ticker list"
        if is_sentiment_task
        else "`{{transcript}}` is replaced with the article text"
    )
    prompt_template = st.text_area(
        f"Prompt — edit freely; {prompt_help}",
        value=_default_prompt,
        height=300,
        key=f"prompt_template_{task}",
    )

    st.divider()

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["Single article", "Full sweep (batch)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if not api_key:
        st.caption("Enter your OpenRouter API key in the sidebar to run.")

    # ── Single article ────────────────────────────────────────────────────────
    if mode == "Single article":
        col_left, col_right = st.columns([1, 2])
        with col_left:
            sweep_mode = st.toggle("Sweep all models in group")
            if not sweep_mode:
                selected_model = st.selectbox(
                    "Model", model_options, format_func=lambda m: m["name"],
                )
            else:
                st.caption(f"Will run all {len(model_options)} models in **{model_group}** in parallel.")
            if is_sentiment_task:
                sentiment_tickers_text = st.text_input(
                    "Tickers to score",
                    value="",
                    placeholder="AAPL, MSFT, NVDA",
                )
                confidence_threshold = 0.0
            else:
                sentiment_tickers_text = ""
                confidence_threshold = st.slider(
                    "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
                )
        with col_right:
            uploaded = st.file_uploader("Upload article (.txt)", type=["txt"])
            pasted = st.text_area("Or paste article text here", height=220)

        article_text = ""
        if uploaded:
            article_text = uploaded.read().decode("utf-8")
        elif pasted.strip():
            article_text = pasted.strip()

        _doc_id = Path(uploaded.name).stem if uploaded else f"paste_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        sentiment_tickers = [
            ticker.strip().upper()
            for ticker in sentiment_tickers_text.split(",")
            if ticker.strip()
        ]
        can_run = bool(article_text) and bool(api_key) and (not is_sentiment_task or bool(sentiment_tickers))
        btn_label = f"Sweep {len(model_options)} models" if sweep_mode else ("Score Sentiment" if is_sentiment_task else "Extract")
        run_clicked = st.button(btn_label, disabled=not can_run, type="primary")
        if is_sentiment_task and article_text and api_key and not sentiment_tickers:
            st.caption("Enter at least one ticker to enable sentiment scoring.")

        if run_clicked and can_run:
            prompt = prompt_template.replace("{{transcript}}", article_text)
            if is_sentiment_task:
                prompt = prompt.replace("{{tickers}}", ", ".join(sentiment_tickers))

            if not sweep_mode:
                with st.spinner(f"Calling {selected_model['name']}..."):
                    result = call_model_direct(api_key, selected_model["id"], prompt, temperature, max_tokens)

                if result.get("error"):
                    st.error(f"API error: {result['error']}")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Latency", f"{result.get('latency_ms', 0):,} ms")
                    c2.metric("Input tokens", result.get("input_tokens") or "—")
                    c3.metric("Output tokens", result.get("output_tokens") or "—")

                    content = result.get("content", "") or ""
                    if is_sentiment_task:
                        sentiments, parse_error = parse_result_sentiments(result)
                        if parse_error:
                            st.warning("Response could not be parsed as sentiment JSON.")
                            st.code(content, language="json")
                        else:
                            sentiment_df = pd.DataFrame([
                                {
                                    "Ticker": ticker,
                                    "Score": values["score"],
                                    "Confidence": values["confidence"],
                                }
                                for ticker, values in sorted(sentiments.items())
                            ])
                            if sentiment_df.empty:
                                st.info("No sentiment scores returned.")
                            else:
                                st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
                                df_download_button(sentiment_df, "sentiment_scores.csv")
                    else:
                        labels, parse_error = _parse_labels_content(content)
                        if parse_error:
                            st.warning("Response could not be parsed as valid JSON.")
                            st.code(content, language="json")
                        else:
                            tickers, company_names, _ = _extract_tickers_and_names(
                                labels, threshold=confidence_threshold
                            )
                            if not tickers:
                                st.info("No tickers extracted above the confidence threshold.")
                            else:
                                conf_map: dict[str, Any] = {}
                                for label in labels:
                                    if isinstance(label, dict) and isinstance(label.get("ticker"), str):
                                        conf_map[canonicalize_ticker(label["ticker"])] = label.get("confidence", "—")
                                result_df = pd.DataFrame([
                                    {
                                        "Ticker": t,
                                        "Company Name": ", ".join(sorted(company_names.get(t, set()))) or "—",
                                        "Confidence": conf_map.get(t, "—"),
                                    }
                                    for t in sorted(tickers)
                                ])
                                st.markdown(f"**{len(result_df)} ticker(s) extracted**")
                                st.dataframe(result_df, use_container_width=True, hide_index=True)
                                df_download_button(result_df, "extracted_tickers.csv")

                    with st.expander("Raw model response"):
                        st.code(content, language="json")

                    st.session_state["_last_extract"] = {
                        "mode": "single",
                        "task": task,
                        "doc_id": _doc_id,
                        "prompt_template": prompt_template,
                        "rendered_prompt": prompt,
                        "model_options": [selected_model],
                        "results": {selected_model["name"]: result},
                        "model_group": model_group,
                        "temperature": temperature,
                        "max_tokens": int(max_tokens),
                    }

            else:
                with st.spinner(f"Calling {len(model_options)} models in parallel..."):
                    with ThreadPoolExecutor() as executor:
                        futures = {
                            executor.submit(call_model_direct, api_key, m["id"], prompt, temperature, max_tokens): m
                            for m in model_options
                        }
                        sweep_results: dict[str, dict] = {}
                        for future in as_completed(futures):
                            m = futures[future]
                            sweep_results[m["name"]] = future.result()

                st.markdown("### Model Metrics")
                metric_rows = []
                for m in model_options:
                    r = sweep_results.get(m["name"], {})
                    cost = r.get("cost")
                    metric_rows.append({
                        "Model": m["name"],
                        "Latency (ms)": r.get("latency_ms"),
                        "Input tokens": r.get("input_tokens"),
                        "Output tokens": r.get("output_tokens"),
                        "Cost ($)": round(cost, 5) if cost else None,
                        "Error": r.get("error") or "",
                    })
                metrics_df = pd.DataFrame(metric_rows)
                st.dataframe(
                    metrics_df.style.format({"Cost ($)": "${:.5f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )
                df_download_button(metrics_df, "sweep_metrics.csv")

                model_names = [m["name"] for m in model_options]
                if is_sentiment_task:
                    st.markdown("### Sentiment Comparison")
                    score_rows = []
                    for m in model_options:
                        r = sweep_results.get(m["name"], {})
                        sentiments, parse_error = parse_result_sentiments(r)
                        if parse_error:
                            score_rows.append({
                                "Model": m["name"],
                                "Ticker": "",
                                "Score": None,
                                "Confidence": None,
                                "Error": r.get("error") or "parse_error",
                            })
                            continue
                        for ticker, values in sorted(sentiments.items()):
                            score_rows.append({
                                "Model": m["name"],
                                "Ticker": ticker,
                                "Score": values["score"],
                                "Confidence": values["confidence"],
                                "Error": r.get("error") or "",
                            })
                    score_df = pd.DataFrame(score_rows)
                    st.dataframe(score_df, use_container_width=True, hide_index=True)
                    df_download_button(score_df, "sentiment_comparison.csv")
                else:
                    st.markdown("### Ticker Comparison")
                    parsed_sweep: dict[str, tuple] = {}
                    for m in model_options:
                        r = sweep_results.get(m["name"], {})
                        content = r.get("content", "") or ""
                        labels, parse_error = _parse_labels_content(content)
                        if not parse_error:
                            tickers, cnames, _ = _extract_tickers_and_names(labels, threshold=confidence_threshold)
                            cmap: dict[str, Any] = {}
                            for label in labels:
                                if isinstance(label, dict) and isinstance(label.get("ticker"), str):
                                    cmap[canonicalize_ticker(label["ticker"])] = label.get("confidence")
                            parsed_sweep[m["name"]] = (tickers, cnames, cmap)
                        else:
                            parsed_sweep[m["name"]] = (set(), {}, {})

                    all_sweep_tickers: set[str] = set()
                    all_sweep_names: dict[str, set] = {}
                    for tickers, cnames, _ in parsed_sweep.values():
                        all_sweep_tickers |= tickers
                        for t, n in cnames.items():
                            all_sweep_names.setdefault(t, set()).update(n)

                    if not all_sweep_tickers:
                        st.info("No tickers extracted by any model.")
                    else:
                        comp_rows = []
                        for t in sorted(all_sweep_tickers):
                            row: dict = {
                                "Ticker": t,
                                "Company Name": ", ".join(sorted(all_sweep_names.get(t, set()))) or "—",
                            }
                            votes = 0
                            for name in model_names:
                                tickers, _, cmap = parsed_sweep[name]
                                if t in tickers:
                                    row[name] = round(cmap.get(t) or 1.0, 2)
                                    votes += 1
                                else:
                                    row[name] = None
                            row["Votes"] = f"{votes}/{len(model_names)}"
                            comp_rows.append(row)
                        comp_df = pd.DataFrame(comp_rows)
                        st.dataframe(
                            comp_df.style.format({n: "{:.2f}" for n in model_names}, na_rep="—"),
                            use_container_width=True, hide_index=True,
                        )
                        df_download_button(comp_df, "ticker_comparison.csv")

                with st.expander("Raw responses"):
                    for m in model_options:
                        r = sweep_results.get(m["name"], {})
                        st.markdown(f"**{m['name']}**")
                        st.code(r.get("content") or r.get("error") or "", language="json")

                st.session_state["_last_extract"] = {
                    "mode": "sweep",
                    "task": task,
                    "doc_id": _doc_id,
                    "prompt_template": prompt_template,
                    "rendered_prompt": prompt,
                    "model_options": model_options,
                    "results": sweep_results,
                    "model_group": model_group,
                    "temperature": temperature,
                    "max_tokens": int(max_tokens),
                }

        if "_last_extract" in st.session_state:
            _ex = st.session_state["_last_extract"]
            st.divider()
            with st.expander("Save to experiments"):
                _save_name = st.text_input(
                    "Experiment name",
                    value=_ex["model_group"],
                    key="ex_save_name",
                )
                _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                _exp_folder_name = f"{_save_name}_{_ts}"

                _scol1, _scol2 = st.columns(2)

                # ── Save to disk ──────────────────────────────────────────────
                with _scol1:
                    st.caption("**Save locally**")
                    if not exp_base_str.strip():
                        st.warning("Set an experiments directory in the sidebar first.")
                    else:
                        st.caption(f"→ `{exp_base / _exp_folder_name}`")
                        if st.button("Save to disk", type="primary", key="ex_save_btn"):
                            try:
                                _new_exp = exp_base / _exp_folder_name
                                _res_dir = _new_exp / "results"
                                _res_dir.mkdir(parents=True, exist_ok=True)
                                for _m in _ex["model_options"]:
                                    _r = _ex["results"].get(_m["name"], {})
                                    _slug = _m["id"].replace("/", "_")
                                    _out = _res_dir / f"{_slug}__{_ex['doc_id']}.json"
                                    with _out.open("w", encoding="utf-8") as _f:
                                        json.dump(_r, _f, indent=2, ensure_ascii=False)
                                        _f.write("\n")
                                _cfg = {
                                    "name": _save_name,
                                    "model_group": _ex["model_group"],
                                    "task": "sentiment" if _ex.get("task") == "Sentiment labeling" else "tickerization",
                                    "prompt": "prompts/sentiment_labeling.txt" if _ex.get("task") == "Sentiment labeling" else "",
                                    "input_dir": "",
                                    "reference": "",
                                    "temperature": _ex["temperature"],
                                    "max_tokens": _ex["max_tokens"],
                                    "models": _ex["model_options"],
                                }
                                with (_new_exp / "config.yaml").open("w", encoding="utf-8") as _f:
                                    yaml.safe_dump(_cfg, _f, sort_keys=False)
                                with (_new_exp / "prompt.txt").open("w", encoding="utf-8") as _f:
                                    _f.write(_ex["prompt_template"])
                                with (_new_exp / "log.jsonl").open("w", encoding="utf-8") as _f:
                                    for _m in _ex["model_options"]:
                                        _r = _ex["results"].get(_m["name"], {})
                                        _f.write(json.dumps({
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "model": _m["id"],
                                            "model_name": _m["name"],
                                            "doc_id": _ex["doc_id"],
                                            "request": {
                                                "prompt": _ex["rendered_prompt"],
                                                "temperature": _ex["temperature"],
                                                "max_tokens": _ex["max_tokens"],
                                            },
                                            "content": _r.get("content"),
                                            "latency_ms": _r.get("latency_ms"),
                                            "input_tokens": _r.get("input_tokens"),
                                            "output_tokens": _r.get("output_tokens"),
                                            "finish_reason": _r.get("finish_reason"),
                                            "error": _r.get("error"),
                                        }, ensure_ascii=False) + "\n")
                                st.success(f"Saved to `{_new_exp}`")
                            except PermissionError:
                                st.warning(
                                    "Could not write to that path — this usually means the app "
                                    "is running on a remote server where your local filesystem "
                                    "isn't accessible. Use **Download as zip** on the right instead."
                                )
                            except Exception as _exc:
                                st.warning(
                                    f"Save failed: {_exc}  \n"
                                    "Try **Download as zip** on the right as an alternative."
                                )

                # ── Download as zip ───────────────────────────────────────────
                with _scol2:
                    st.caption("**Download as zip**")
                    st.caption("Unzip locally and load via Sweep Results.")
                    _zip_buf = io.BytesIO()
                    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                        for _m in _ex["model_options"]:
                            _r = _ex["results"].get(_m["name"], {})
                            _slug = _m["id"].replace("/", "_")
                            _zf.writestr(
                                f"{_exp_folder_name}/results/{_slug}__{_ex['doc_id']}.json",
                                json.dumps(_r, indent=2, ensure_ascii=False),
                            )
                        _cfg = {
                            "name": _save_name,
                            "model_group": _ex["model_group"],
                            "task": "sentiment" if _ex.get("task") == "Sentiment labeling" else "tickerization",
                            "prompt": "prompts/sentiment_labeling.txt" if _ex.get("task") == "Sentiment labeling" else "",
                            "input_dir": "",
                            "reference": "",
                            "temperature": _ex["temperature"],
                            "max_tokens": _ex["max_tokens"],
                            "models": _ex["model_options"],
                        }
                        _zf.writestr(
                            f"{_exp_folder_name}/config.yaml",
                            yaml.safe_dump(_cfg, sort_keys=False),
                        )
                        _zf.writestr(f"{_exp_folder_name}/prompt.txt", _ex["prompt_template"])
                        _log_lines = []
                        for _m in _ex["model_options"]:
                            _r = _ex["results"].get(_m["name"], {})
                            _log_lines.append(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "model": _m["id"],
                                "model_name": _m["name"],
                                "doc_id": _ex["doc_id"],
                                "request": {
                                    "prompt": _ex["rendered_prompt"],
                                    "temperature": _ex["temperature"],
                                    "max_tokens": _ex["max_tokens"],
                                },
                                "content": _r.get("content"),
                                "latency_ms": _r.get("latency_ms"),
                                "input_tokens": _r.get("input_tokens"),
                                "output_tokens": _r.get("output_tokens"),
                                "finish_reason": _r.get("finish_reason"),
                                "error": _r.get("error"),
                            }, ensure_ascii=False))
                        _zf.writestr(f"{_exp_folder_name}/log.jsonl", "\n".join(_log_lines) + "\n")
                    st.download_button(
                        "Download zip",
                        data=_zip_buf.getvalue(),
                        file_name=f"{_exp_folder_name}.zip",
                        mime="application/zip",
                        key="ex_download_btn",
                    )

    # ── Full sweep (batch) ────────────────────────────────────────────────────
    else:
        sw_col1, sw_col2 = st.columns(2)
        with sw_col1:
            sw_input_dir = st.text_input(
                "Input directory (articles)",
                value="",
                help="Local directory containing .txt article files.",
            )
            sw_zip_upload = st.file_uploader(
                "Or upload a zip of .txt articles",
                type=["zip"],
                help="Zip file containing .txt files (any folder depth).",
            )
        with sw_col2:
            sw_output_base = st.text_input(
                "Output directory",
                value=str(exp_base),
                help="Leave blank to download results as a zip instead.",
            )
        sw_exp_name = st.text_input(
            "Experiment name",
            value=f"sentiment_{model_group}" if is_sentiment_task else model_group,
            key="sw_exp_name",
            help="Prefix for the experiment folder (a timestamp is appended automatically).",
        )
        reference_payload: dict[str, Any] | None = None
        reference_json_text = ""
        reference_tickers_by_doc: dict[str, list[str]] = {}
        if is_sentiment_task:
            sw_reference_upload = st.file_uploader(
                "reference.json for sentiment tickers",
                type=["json"],
                key="sw_sentiment_reference",
                help="Positive labels in this file become the ticker list for each document.",
            )
            if sw_reference_upload:
                try:
                    reference_json_text = sw_reference_upload.read().decode("utf-8")
                    reference_payload = json.loads(reference_json_text)
                    reference_tickers_by_doc = reference_tickers_from_payload(reference_payload)
                    st.caption(f"Loaded ticker lists for {len(reference_tickers_by_doc)} document(s).")
                except Exception as exc:
                    st.warning(f"Could not parse reference.json: {exc}")

        # Resolve article list: directory → zip upload → nothing
        sw_input_path = Path(os.path.expanduser(sw_input_dir)) if sw_input_dir.strip() else None
        if sw_input_path and sw_input_path.exists():
            sw_articles: list[tuple[str, str]] = [
                (p.stem, p.read_text(encoding="utf-8"))
                for p in sorted(sw_input_path.glob("*.txt"))
            ]
            if not sw_articles:
                st.warning(
                    f"No .txt files found in `{sw_input_path}`. "
                    "Check the path or upload a zip of your articles above."
                )
        elif sw_input_path and not sw_input_path.exists():
            st.warning(
                f"Directory `{sw_input_path}` not found — it may be a local path not accessible "
                "from this server. Upload a zip of your articles above instead."
            )
            if sw_zip_upload:
                sw_articles = []
                with zipfile.ZipFile(io.BytesIO(sw_zip_upload.read())) as zf:
                    for name in sorted(zf.namelist()):
                        if name.endswith(".txt") and not Path(name).name.startswith("."):
                            sw_articles.append((Path(name).stem, zf.read(name).decode("utf-8")))
            else:
                sw_articles = []
        elif sw_zip_upload:
            sw_articles = []
            with zipfile.ZipFile(io.BytesIO(sw_zip_upload.read())) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(".txt") and not Path(name).name.startswith("."):
                        sw_articles.append((Path(name).stem, zf.read(name).decode("utf-8")))
        else:
            sw_articles = []

        missing_reference_docs = (
            [
                doc_id for doc_id, _ in sw_articles
                if doc_id not in reference_tickers_by_doc
            ]
            if is_sentiment_task and sw_articles
            else []
        )
        docs_without_tickers = (
            [
                doc_id for doc_id, _ in sw_articles
                if doc_id in reference_tickers_by_doc and not reference_tickers_by_doc[doc_id]
            ]
            if is_sentiment_task and sw_articles
            else []
        )
        st.caption(
            f"{len(sw_articles)} article(s) · {len(model_options)} model(s) · "
            f"{len(sw_articles) * len(model_options)} total API calls"
        )
        if missing_reference_docs:
            st.warning(
                "Reference is missing doc_id values for: "
                + ", ".join(missing_reference_docs[:10])
                + (" ..." if len(missing_reference_docs) > 10 else "")
            )
        if docs_without_tickers:
            st.warning(
                "Reference has no positive tickers for: "
                + ", ".join(docs_without_tickers[:10])
                + (" ..." if len(docs_without_tickers) > 10 else "")
            )

        can_sweep = (
            bool(api_key)
            and bool(sw_articles)
            and (
                not is_sentiment_task
                or (
                    bool(reference_tickers_by_doc)
                    and not missing_reference_docs
                    and not docs_without_tickers
                )
            )
        )
        sweep_clicked = st.button(
            "Run Sweep", disabled=not can_sweep, type="primary", key="run_sweep_btn"
        )
        if not sw_articles and api_key:
            st.caption("Add an input directory or upload a zip of .txt files to enable the sweep.")
        if is_sentiment_task and sw_articles and not reference_tickers_by_doc:
            st.caption("Upload `reference.json` to enable sentiment labeling.")

        if sweep_clicked and can_sweep:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            sw_exp_folder = f"{sw_exp_name}_{ts}"
            total_calls = len(model_options) * len(sw_articles)
            os.environ["OPENROUTER_API_KEY"] = api_key
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            done = 0
            failed = 0

            sw_results_mem: list[tuple[str, str, dict]] = []
            log_lines: list[str] = []

            for model in model_options:
                for doc_id, text in sw_articles:
                    prompt = prompt_template.replace("{{transcript}}", text)
                    if is_sentiment_task:
                        prompt = prompt.replace("{{tickers}}", ", ".join(reference_tickers_by_doc[doc_id]))
                    status_text.caption(f"[{done + 1}/{total_calls}] {model['name']} × {doc_id}")
                    result = _sweep_call_model(model["id"], prompt, temperature, int(max_tokens))
                    model_slug = model["id"].replace("/", "_")
                    sw_results_mem.append((model_slug, doc_id, result))
                    log_lines.append(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model["id"],
                        "model_name": model["name"],
                        "doc_id": doc_id,
                        "request": {
                            "prompt": prompt,
                            "temperature": temperature,
                            "max_tokens": int(max_tokens),
                        },
                        "latency_ms": result.get("latency_ms"),
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "finish_reason": result.get("finish_reason"),
                        "error": result.get("error"),
                    }, ensure_ascii=False))
                    if result.get("error"):
                        failed += 1
                    done += 1
                    progress_bar.progress(done / total_calls)

            status_text.empty()
            progress_bar.empty()
            msg = f"Sweep complete — {done} calls, {failed} error(s)"
            st.success(msg) if not failed else st.warning(msg)

            cfg_to_save = {
                "name": sw_exp_name,
                "model_group": model_group,
                "task": "sentiment" if is_sentiment_task else "tickerization",
                "prompt": "prompts/sentiment_labeling.txt" if is_sentiment_task else "",
                "input_dir": sw_input_dir,
                "output_base": sw_output_base,
                "reference": "",
                "temperature": temperature,
                "max_tokens": int(max_tokens),
                "models": model_options,
            }

            # ── Try saving to disk ────────────────────────────────────────────
            saved_to_disk = False
            if sw_output_base.strip():
                try:
                    new_exp_dir = Path(os.path.expanduser(sw_output_base)) / sw_exp_folder
                    results_dir_sw = new_exp_dir / "results"
                    results_dir_sw.mkdir(parents=True, exist_ok=True)
                    cfg_for_disk = dict(cfg_to_save)
                    if is_sentiment_task and reference_json_text:
                        ref_path = new_exp_dir / "reference.json"
                        ref_path.write_text(reference_json_text, encoding="utf-8")
                        cfg_for_disk["reference"] = str(ref_path)
                    for model_slug, doc_id, result in sw_results_mem:
                        result_file = results_dir_sw / f"{model_slug}__{doc_id}.json"
                        with result_file.open("w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                            f.write("\n")
                    with (new_exp_dir / "config.yaml").open("w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg_for_disk, f, sort_keys=False)
                    with (new_exp_dir / "prompt.txt").open("w", encoding="utf-8") as f:
                        f.write(prompt_template)
                    with (new_exp_dir / "log.jsonl").open("w", encoding="utf-8") as f:
                        f.write("\n".join(log_lines) + "\n")
                    st.success(f"Results saved to `{new_exp_dir}`.")
                    st.caption("Go to **Sweep Results** to run evaluation on this experiment.")
                    saved_to_disk = True
                except PermissionError:
                    st.warning(
                        "Could not write to that output path — the app is likely running on a "
                        "remote server. Download the results as a zip below."
                    )
                except Exception as exc:
                    st.warning(f"Save failed: {exc}  \nDownload the results as a zip below.")

            # ── Download as zip (fallback) ────────────────────────────────────
            if not saved_to_disk:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    cfg_for_zip = dict(cfg_to_save)
                    if is_sentiment_task and reference_json_text:
                        cfg_for_zip["reference"] = "reference.json"
                        zf.writestr(f"{sw_exp_folder}/reference.json", reference_json_text)
                    for model_slug, doc_id, result in sw_results_mem:
                        zf.writestr(
                            f"{sw_exp_folder}/results/{model_slug}__{doc_id}.json",
                            json.dumps(result, indent=2, ensure_ascii=False),
                        )
                    zf.writestr(
                        f"{sw_exp_folder}/config.yaml",
                        yaml.safe_dump(cfg_for_zip, sort_keys=False),
                    )
                    zf.writestr(
                        f"{sw_exp_folder}/prompt.txt",
                        prompt_template,
                    )
                    zf.writestr(
                        f"{sw_exp_folder}/log.jsonl",
                        "\n".join(log_lines) + "\n",
                    )
                st.download_button(
                    "Download results as zip",
                    data=zip_buf.getvalue(),
                    file_name=f"{sw_exp_folder}.zip",
                    mime="application/zip",
                )
                st.caption("Unzip locally and load the folder via Sweep Results.")

# ── Tab 2: Sweep Results ──────────────────────────────────────────────────────

with tab_results:
    st.subheader("Sweep experiment results")

    _disk_experiments = list_experiments(exp_base) if exp_base_str.strip() and exp_base.exists() else []
    if exp_base_str.strip() and not exp_base.exists():
        st.warning(
            f"Directory `{exp_base}` not found — it may be a local path not accessible from this server. "
            "Upload experiment zips in the sidebar instead."
        )
    _zip_experiments = list(st.session_state.get("_loaded_exp_dirs", {}).values())
    experiments = sorted(_disk_experiments + _zip_experiments, key=lambda p: p.name, reverse=True)
    if not experiments:
        st.info("No experiments found. Run a sweep or upload an experiment zip in the sidebar.")
    else:
        exp_names = [p.name for p in experiments]
        selected_exp_name = st.selectbox("Experiment", exp_names, key="results_exp_select")
        selected_exp = next(p for p in experiments if p.name == selected_exp_name)

        # ── Evaluate ──────────────────────────────────────────────────────────
        has_results = load_scores_csv(selected_exp) is not None
        exp_cfg: dict[str, Any] = {}
        config_path = selected_exp / "config.yaml"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                exp_cfg = yaml.safe_load(f) or {}
        is_sentiment_exp = (
            exp_cfg.get("task") == "sentiment"
            or "sentiment" in str(exp_cfg.get("name", "")).lower()
            or "sentiment_labeling" in str(exp_cfg.get("prompt", ""))
        )
        if not is_sentiment_exp:
            with st.expander("Run Evaluation", expanded=not has_results):
                st.caption("Upload your `reference.json` to score this experiment against ground-truth labels.")
                with st.expander("`reference.json` format"):
                    st.markdown(
                        "A JSON file with a `documents` array. Each entry maps a `doc_id` "
                        "(the article filename stem, without `.txt`) to its ground-truth tickers. "
                        "Set `\"label\": 1` for a true ticker and `\"label\": 0` to exclude one."
                    )
                    st.code(
                        '{\n'
                        '  "documents": [\n'
                        '    {\n'
                        '      "doc_id": "article_123",\n'
                        '      "labels": [\n'
                        '        {"ticker": "AAPL", "label": 1},\n'
                        '        {"ticker": "GOOG", "label": 1},\n'
                        '        {"ticker": "TSLA", "label": 0}\n'
                        '      ]\n'
                        '    },\n'
                        '    {\n'
                        '      "doc_id": "article_456",\n'
                        '      "labels": [\n'
                        '        {"ticker": "MSFT", "label": 1}\n'
                        '      ]\n'
                        '    }\n'
                        '  ]\n'
                        '}',
                        language="json",
                    )
                ref_file = st.file_uploader("reference.json", type=["json"], key="ref_upload")
                if ref_file:
                    if st.button("Run Evaluation", type="primary", key="run_eval_btn"):
                        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
                            tmp.write(ref_file.read())
                            tmp_path = tmp.name
                        try:
                            with st.spinner("Running evaluation..."):
                                evaluate_experiment(selected_exp, tmp_path)
                            st.success("Evaluation complete.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Evaluation failed: {exc}")
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)

        if is_sentiment_exp:
            consensus_exists = (selected_exp / "eval" / "sentiment_consensus.csv").exists()
            with st.expander("Build Sentiment Consensus", expanded=not consensus_exists):
                st.caption(
                    "Build per-document/ticker sentiment consensus from this sentiment sweep."
                )
                configured_ref = resolve_config_path(selected_exp, exp_cfg.get("reference"))
                ref_exists = configured_ref is not None and configured_ref.exists()
                if ref_exists:
                    st.caption(f"Using reference from config: `{configured_ref}`")
                sentiment_ref_upload = st.file_uploader(
                    "Optional reference.json override",
                    type=["json"],
                    key="sentiment_consensus_ref_upload",
                )
                agree_threshold = st.number_input(
                    "Consensus max pairwise difference",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.50,
                    step=0.05,
                    key="sentiment_agree_threshold",
                )
                if st.button(
                    "Build Sentiment Consensus",
                    type="primary",
                    key="build_sentiment_consensus_btn",
                    disabled=not (ref_exists or sentiment_ref_upload is not None),
                ):
                    tmp_path = None
                    try:
                        if sentiment_ref_upload is not None:
                            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
                                tmp.write(sentiment_ref_upload.read())
                                tmp_path = tmp.name
                            ref_path_for_consensus = Path(tmp_path)
                        elif configured_ref is not None:
                            ref_path_for_consensus = configured_ref
                        else:
                            raise ValueError("A reference.json file is required.")
                        with st.spinner("Building sentiment consensus..."):
                            build_sentiment_consensus(
                                exp_dir=selected_exp,
                                ref_path=ref_path_for_consensus,
                                agree_max_abs_dev=float(agree_threshold),
                                review_max_abs_dev=float(agree_threshold),
                            )
                        st.success("Sentiment consensus complete.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Sentiment consensus failed: {exc}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)

        # ── Results ───────────────────────────────────────────────────────────
        df = load_scores_csv(selected_exp)

        sentiment_consensus_path = selected_exp / "eval" / "sentiment_consensus.csv"
        if sentiment_consensus_path.exists():
            st.markdown("### Sentiment Consensus")
            sentiment_df = pd.read_csv(sentiment_consensus_path)
            st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
            df_download_button(sentiment_df, "sentiment_consensus.csv")
            sentiment_summary = selected_exp / "eval" / "sentiment_summary.md"
            if sentiment_summary.exists():
                with st.expander("Sentiment summary"):
                    st.markdown(sentiment_summary.read_text(encoding="utf-8"))

        if df is None:
            if is_sentiment_exp:
                if not sentiment_consensus_path.exists():
                    st.info("No sentiment consensus yet — build it above.")
            else:
                st.info("No results yet — upload a `reference.json` above to run evaluation.")
        else:
            n_models = df["model"].nunique()
            n_docs = df["doc_id"].nunique()
            st.caption(f"{len(df)} rows · {n_models} model(s) · {n_docs} doc(s)")

            st.markdown("### Model Summary")
            run_costs = load_run_costs(selected_exp)
            summary = model_summary_df(df, run_costs=run_costs)
            cost_cols = [c for c in ["Cost/Doc ($)", "Cost/Ticker ($)"] if c in summary.columns]
            fmt = {"F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"}
            fmt.update({c: "${:.4f}" for c in cost_cols})
            st.dataframe(summary.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)
            df_download_button(summary, "model_summary.csv")

            st.markdown("### F1 Score by Model")
            _chart_data = summary[["model", "F1", "Precision", "Recall"]].melt(
                id_vars="model", var_name="Metric", value_name="Score"
            )
            st.altair_chart(
                alt.Chart(_chart_data).mark_bar().encode(
                    x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                    color="Metric:N",
                    xOffset="Metric:N",
                ).properties(height=300),
                use_container_width=True,
            )

            st.markdown("### Per-Document Results")
            filter_model = st.selectbox(
                "Filter by model",
                ["All"] + sorted(df["model"].unique().tolist()),
                key="doc_model_filter",
            )
            display_df = df if filter_model == "All" else df[df["model"] == filter_model]
            show_cols = [c for c in
                ["doc_id", "model", "f1", "precision", "recall", "tp", "fp", "fn", "parse_error", "predicted", "actual"]
                if c in display_df.columns
            ]
            perdoc_df = display_df[show_cols].reset_index(drop=True)
            st.dataframe(perdoc_df, use_container_width=True, hide_index=True)
            df_download_button(perdoc_df, "per_doc_results.csv")

            # ── Doc drill-down ────────────────────────────────────────────────
            st.markdown("### Document Drill-Down")
            all_doc_ids = sorted(df["doc_id"].unique().tolist())
            selected_doc = st.selectbox("Select document", all_doc_ids, key="doc_drilldown")

            if selected_doc:
                doc_rows = df[df["doc_id"] == selected_doc]

                # Article text
                exp_config_path = selected_exp / "config.yaml"
                if exp_config_path.exists():
                    with exp_config_path.open(encoding="utf-8") as f:
                        exp_cfg = yaml.safe_load(f)
                    input_dir = Path(os.path.expanduser(exp_cfg.get("input_dir", "")))
                    article_path = input_dir / f"{selected_doc}.txt"
                    if article_path.exists():
                        with st.expander("Article text"):
                            st.text(article_path.read_text(encoding="utf-8"))
                    else:
                        st.caption(f"Article not found at `{article_path}`")

                # Reference tickers
                actual_tickers: list[str] = []
                if "actual" in doc_rows.columns:
                    raw_actual = doc_rows["actual"].dropna()
                    if not raw_actual.empty:
                        try:
                            actual_tickers = ast.literal_eval(str(raw_actual.iloc[0]))
                        except Exception:
                            actual_tickers = []
                if actual_tickers:
                    st.markdown(
                        f"**Reference tickers ({len(actual_tickers)}):** "
                        + "  ".join(f"`{t}`" for t in sorted(actual_tickers))
                    )

                # Per-model breakdown
                results_dir = selected_exp / "results"
                actual_set = set(actual_tickers)

                all_model_slugs = sorted(doc_rows["model"].unique().tolist())
                selected_model_slug = st.selectbox(
                    "Select model", all_model_slugs, key="drill_model_select"
                )

                model_row = doc_rows[doc_rows["model"] == selected_model_slug].iloc[0]
                is_parse_error = bool(model_row.get("parse_error", False))
                tp = int(model_row.get("tp", 0))
                fp = int(model_row.get("fp", 0))
                fn = int(model_row.get("fn", 0))
                f1_val = float(model_row.get("f1", 0))
                prec = float(model_row.get("precision", 0))
                rec = float(model_row.get("recall", 0))

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("F1", f"{f1_val:.3f}")
                m2.metric("Precision", f"{prec:.3f}")
                m3.metric("Recall", f"{rec:.3f}")
                m4.metric("TP", tp)
                m5.metric("FP", fp)
                m6.metric("FN", fn)

                if is_parse_error:
                    st.warning("⚠ This response failed to parse as valid JSON.")

                result_file = results_dir / f"{selected_model_slug}__{selected_doc}.json"
                if result_file.exists():
                    with result_file.open(encoding="utf-8") as f:
                        raw_result = json.load(f)
                    content = raw_result.get("content", "") or ""
                    if not is_parse_error:
                        try:
                            predicted = ast.literal_eval(str(model_row.get("predicted", "[]")))
                        except Exception:
                            predicted = []
                        predicted_set = set(predicted)
                        ticker_detail = []
                        for t in sorted(predicted_set):
                            ticker_detail.append({"Ticker": t, "Status": "✅ TP" if t in actual_set else "❌ FP"})
                        for t in sorted(actual_set - predicted_set):
                            ticker_detail.append({"Ticker": t, "Status": "⬜ FN"})
                        if ticker_detail:
                            st.dataframe(
                                pd.DataFrame(ticker_detail),
                                use_container_width=True, hide_index=True,
                            )
                    if st.toggle("Show raw response", key="raw_response_toggle"):
                        st.code(content, language="json")
                else:
                    st.caption("Result file not found.")

        # Expanders at bottom
        summary_md = selected_exp / "eval" / "summary.md"
        if summary_md.exists():
            with st.expander("Evaluation summary (summary.md)"):
                st.markdown(summary_md.read_text(encoding="utf-8"))

        config_path = selected_exp / "config.yaml"
        if config_path.exists():
            with st.expander("Sweep config"):
                st.code(config_path.read_text(encoding="utf-8"), language="yaml")


# ── Tab 3: Leaderboard ────────────────────────────────────────────────────────

with tab_leaderboard:
    st.subheader("Leaderboard — best F1 per model across selected experiments")

    _disk_experiments = list_experiments(exp_base) if exp_base_str.strip() and exp_base.exists() else []
    if exp_base_str.strip() and not exp_base.exists():
        st.warning(
            f"Directory `{exp_base}` not found — it may be a local path not accessible from this server. "
            "Upload experiment zips in the sidebar instead."
        )
    _zip_experiments = list(st.session_state.get("_loaded_exp_dirs", {}).values())
    experiments = sorted(_disk_experiments + _zip_experiments, key=lambda p: p.name, reverse=True)
    if not experiments:
        st.info("No experiments found. Run a sweep or upload an experiment zip in the sidebar.")
    else:
        _all_selected = all(
            st.session_state.get(f"lb_chk_{exp.name}", True) for exp in experiments
        )
        _transparent = (
            "background-color: transparent !important; border: 1px solid rgba(49,51,63,0.3);"
            " box-shadow: none; color: inherit;"
        ) if _all_selected else ""
        _hover = (
            ".st-key-lb_toggle_container button:hover"
            " { background-color: rgba(49,51,63,0.05) !important; }"
        ) if _all_selected else ""
        st.markdown(f"""<style>
.st-key-lb_toggle_container button {{
    font-size: 0.75rem; padding: 0.15rem 0.6rem; line-height: 1.3; {_transparent}
}}
{_hover}
</style>""", unsafe_allow_html=True)
        _lbc1, _lbc2 = st.columns([2, 14])
        with _lbc1:
            with st.container(key="lb_toggle_container"):
                if st.button(
                    "Unselect all" if _all_selected else "Select all",
                    key="lb_toggle_all",
                    type="secondary" if _all_selected else "primary",
                    use_container_width=True,
                ):
                    for exp in experiments:
                        st.session_state[f"lb_chk_{exp.name}"] = not _all_selected
        with st.expander(f"Select experiments ({len(experiments)} available)"):
            selected_experiments = []
            for exp in experiments:
                if st.checkbox(exp.name, value=True, key=f"lb_chk_{exp.name}"):
                    selected_experiments.append(exp)
        experiments = selected_experiments

    all_rows: list[dict] = []
    for exp_dir in experiments:
        df = load_scores_csv(exp_dir)
        if df is None:
            continue
        run_costs = load_run_costs(exp_dir)
        for model, group in df.groupby("model"):
            agg = aggregate_scores(group.to_dict("records"))
            total_tickers = agg["tp"] + agg["fn"]
            total_cost = run_costs.get(model, 0.0)
            all_rows.append({
                "Model": model,
                "Experiment": exp_dir.name,
                "F1": agg["f1"],
                "Precision": agg["precision"],
                "Recall": agg["recall"],
                "TP": agg["tp"],
                "FP": agg["fp"],
                "FN": agg["fn"],
                "Docs": len(group),
                "Parse Errors": int(group["parse_error"].sum()),
                "Cost/Doc ($)": round(total_cost / len(group), 4) if total_cost and len(group) else None,
                "Cost/Ticker ($)": round(total_cost / total_tickers, 4) if total_cost and total_tickers else None,
            })

    if not all_rows:
        st.info(f"No evaluated experiments found in `{exp_base}`.")
    else:
        full_df = pd.DataFrame(all_rows)

        best = (
            full_df.sort_values("F1", ascending=False)
            .drop_duplicates(subset=["Model"])
            .sort_values("F1", ascending=False)
            .reset_index(drop=True)
        )
        best.index += 1

        st.markdown("### Best F1 per Model")
        lb_fmt = {"F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"}
        lb_cost_cols = [c for c in ["Cost/Doc ($)", "Cost/Ticker ($)"] if c in best.columns]
        if lb_cost_cols:
            lb_fmt.update({c: "${:.4f}" for c in lb_cost_cols})
        st.dataframe(best.style.format(lb_fmt, na_rep="—"), use_container_width=True)
        df_download_button(best.reset_index(drop=True), "leaderboard.csv")

        st.markdown("### F1 · Precision · Recall (best run per model)")
        st.bar_chart(best.set_index("Model")[["F1", "Precision", "Recall"]])

        with st.expander("All experiment × model rows"):
            all_exp_df = full_df.sort_values(["Model", "F1"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(all_exp_df, use_container_width=True, hide_index=True)
            df_download_button(all_exp_df, "all_experiment_rows.csv")

"""Streamlit dashboard for the tickerization benchmark."""

import ast
import json
import os
import tempfile
import time
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
    canonicalize_ticker,
    evaluate_experiment,
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
        max_tokens = st.number_input("Max tokens", 256, 8192, 4096, 256, key="ex_max_tokens")

    _default_prompt = load_prompt("tickerize_v2.txt")
    prompt_template = st.text_area(
        "Prompt — edit freely; `{{transcript}}` is replaced with the article text",
        value=_default_prompt,
        height=300,
        key="prompt_template",
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

        can_run = bool(article_text) and bool(api_key)
        btn_label = f"Sweep {len(model_options)} models" if sweep_mode else "Extract"
        run_clicked = st.button(btn_label, disabled=not can_run, type="primary")

        if run_clicked and can_run:
            prompt = prompt_template.replace("{{transcript}}", article_text)

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
                        "doc_id": _doc_id,
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

                st.markdown("### Ticker Comparison")
                model_names = [m["name"] for m in model_options]
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
                    "doc_id": _doc_id,
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
                st.caption(f"Will save as **{_ex['doc_id']}** inside `{exp_base / (_save_name + '_<timestamp>')}`")
                if st.button("Save", type="primary", key="ex_save_btn"):
                    try:
                        _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                        _new_exp = exp_base / f"{_save_name}_{_ts}"
                        _res_dir = _new_exp / "results"
                        _res_dir.mkdir(parents=True, exist_ok=True)
                        for _m in _ex["model_options"]:
                            _r = _ex["results"].get(_m["name"], {})
                            _slug = _m["id"].replace("/", "_")
                            _out = _res_dir / f"{_slug}__{_ex['doc_id']}.json"
                            with _out.open("w", encoding="utf-8") as _f:
                                json.dump(_r, _f, indent=2, ensure_ascii=False)
                        _cfg = {
                            "name": _save_name,
                            "model_group": _ex["model_group"],
                            "input_dir": "",
                            "temperature": _ex["temperature"],
                            "max_tokens": _ex["max_tokens"],
                            "models": _ex["model_options"],
                        }
                        with (_new_exp / "config.yaml").open("w", encoding="utf-8") as _f:
                            yaml.safe_dump(_cfg, _f, sort_keys=False)
                        st.success(f"Saved to `{_new_exp}`")
                    except Exception as _exc:
                        st.error(f"Save failed: {_exc}")

    # ── Full sweep (batch) ────────────────────────────────────────────────────
    else:
        sw_col1, sw_col2 = st.columns(2)
        with sw_col1:
            sw_input_dir = st.text_input(
                "Input directory (articles)",
                value="",
                help="Directory containing .txt article files.",
            )
        with sw_col2:
            sw_output_base = st.text_input(
                "Output directory",
                value=str(exp_base),
                help="Base directory to save experiment results.",
            )
        sw_exp_name = st.text_input(
            "Experiment name",
            value=model_group,
            key="sw_exp_name",
            help="Prefix for the experiment folder (a timestamp is appended automatically).",
        )

        sw_input_path = Path(os.path.expanduser(sw_input_dir))
        sw_articles = sorted(sw_input_path.glob("*.txt")) if sw_input_path.exists() else []
        st.caption(
            f"{len(sw_articles)} article(s) found · {len(model_options)} model(s) · "
            f"{len(sw_articles) * len(model_options)} total API calls"
        )

        can_sweep = bool(api_key)
        sweep_clicked = st.button(
            "Run Sweep", disabled=not can_sweep, type="primary", key="run_sweep_btn"
        )

        if sweep_clicked and can_sweep:
            if not sw_articles:
                st.error(f"No .txt files found in `{sw_input_path}`.")
            else:
                sw_output_path = Path(os.path.expanduser(sw_output_base))
                total_calls = len(model_options) * len(sw_articles)

                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                new_exp_dir = sw_output_path / f"{sw_exp_name}_{ts}"
                results_dir_sw = new_exp_dir / "results"
                results_dir_sw.mkdir(parents=True, exist_ok=True)
                log_path = new_exp_dir / "log.jsonl"

                prompt_txt_path = str(new_exp_dir / "prompt.txt")
                cfg_to_save = {
                    "name": sw_exp_name,
                    "model_group": model_group,
                    "prompt": prompt_txt_path,
                    "input_dir": sw_input_dir,
                    "output_base": sw_output_base,
                    "temperature": temperature,
                    "max_tokens": int(max_tokens),
                    "models": model_options,
                }
                with (new_exp_dir / "config.yaml").open("w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg_to_save, f, sort_keys=False)
                with (new_exp_dir / "prompt.txt").open("w", encoding="utf-8") as f:
                    f.write(prompt_template)

                os.environ["OPENROUTER_API_KEY"] = api_key
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                done = 0
                failed = 0

                for model in model_options:
                    for art_path in sw_articles:
                        doc_id = art_path.stem
                        text = art_path.read_text(encoding="utf-8")
                        prompt = prompt_template.replace("{{transcript}}", text)

                        status_text.caption(f"[{done + 1}/{total_calls}] {model['name']} × {doc_id}")
                        result = _sweep_call_model(
                            model["id"], prompt, temperature, int(max_tokens)
                        )

                        result_file = results_dir_sw / f"{model['id'].replace('/', '_')}__{doc_id}.json"
                        with result_file.open("w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                            f.write("\n")

                        log_entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model["id"],
                            "model_name": model["name"],
                            "doc_id": doc_id,
                            "latency_ms": result.get("latency_ms"),
                            "input_tokens": result.get("input_tokens"),
                            "output_tokens": result.get("output_tokens"),
                            "finish_reason": result.get("finish_reason"),
                            "error": result.get("error"),
                        }
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                        if result.get("error"):
                            failed += 1
                        done += 1
                        progress_bar.progress(done / total_calls)

                status_text.empty()
                progress_bar.empty()
                msg = f"Sweep complete — {done} calls"
                if failed:
                    st.warning(f"{msg}, {failed} error(s). Results saved to `{new_exp_dir}`.")
                else:
                    st.success(f"{msg}, 0 errors. Results saved to `{new_exp_dir}`.")
                st.caption("Go to **Sweep Results** to run evaluation on this experiment.")

# ── Tab 2: Sweep Results ──────────────────────────────────────────────────────

with tab_results:
    st.subheader("Sweep experiment results")

    experiments = list_experiments(exp_base)
    if not experiments:
        st.info(f"No experiments found in `{exp_base}`. Run a sweep first.")
    else:
        exp_names = [p.name for p in experiments]
        selected_exp_name = st.selectbox("Experiment", exp_names, key="results_exp_select")
        selected_exp = exp_base / selected_exp_name

        # ── Evaluate ──────────────────────────────────────────────────────────
        has_results = load_scores_csv(selected_exp) is not None
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

        # ── Results ───────────────────────────────────────────────────────────
        df = load_scores_csv(selected_exp)

        if df is None:
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

    experiments = list_experiments(exp_base)
    if not experiments:
        st.info(f"No experiments found in `{exp_base}`.")
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

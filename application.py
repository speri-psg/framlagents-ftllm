"""
application.py — FRAML AI Assistant (Qwen2.5 via Ollama)

Run:
    python application.py        # http://127.0.0.1:5000

Override the LLM endpoint if needed:
    set OLLAMA_BASE_URL=http://localhost:11434/v1
    set OLLAMA_MODEL=qwen2.5:7b
"""

import json
import time
import sys
import os

# ── Ensure project root is on path so config + analytics imports work ─────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
import flask
import dash_bootstrap_components as dbc
from dash import Dash, callback, html, dcc, Input, Output, State, callback_context, ALL, no_update
from dash_chat import ChatComponent

from config import ALERTS_CSV, SS_CSV, OLLAMA_MODEL, OLLAMA_BASE_URL
from agents import OrchestratorAgent
import lambda_ss_performance

# ── Data loading ──────────────────────────────────────────────────────────────
_df_raw = pd.read_csv(ALERTS_CSV, sep="\t")
_df_raw = _df_raw.rename(columns={
    "AVG_TRXNS_WEEK":  "avg_num_trxns",
    "AVG_TRXN_AMT":    "avg_trxn_amt",
    "TRXN_AMT_MONTHLY":"trxn_amt_monthly",
    "FP":              "false_positives",
    "FN":              "false_negatives",
    "ALERT":           "alerts",
    "CUSTOMER_TYPE":   "customer_type",
})
_df_raw["alerts"]          = _df_raw["alerts"].map({"Yes": 1, "No": 0})
_df_raw["false_positives"] = _df_raw["false_positives"].map({"Yes": 1, "No": 0})
_df_raw["false_negatives"] = _df_raw["false_negatives"].map({"Yes": 1, "No": 0})
_df_raw["smart_segment_id"]= _df_raw["customer_type"].map({"BUSINESS": 0, "INDIVIDUAL": 1})
DF           = _df_raw
DF_BUSINESS  = DF[DF["smart_segment_id"] == 0]
DF_INDIVIDUAL= DF[DF["smart_segment_id"] == 1]

DF_SS = pd.read_csv(SS_CSV) if os.path.exists(SS_CSV) else None

_total      = len(DF)
_biz_count  = len(DF_BUSINESS)
_ind_count  = len(DF_INDIVIDUAL)
_alert_count= int(DF["alerts"].sum())
_fp_count   = int(DF["false_positives"].sum())
print(f"Alerts data: {_total:,} rows | Business={_biz_count:,} Individual={_ind_count:,}")
print(f"SS data: {'loaded (' + str(len(DF_SS)) + ' rows)' if DF_SS is not None else 'not found — run python ss_data_prep.py'}")

COL_MAP = {
    "AVG_TRXNS_WEEK":   "avg_num_trxns",
    "AVG_TRXN_AMT":     "avg_trxn_amt",
    "TRXN_AMT_MONTHLY": "trxn_amt_monthly",
}

orchestrator = OrchestratorAgent()

# ── Suggested prompts ─────────────────────────────────────────────────────────
SUGGESTED_PROMPTS = [
    "Show FP/FN threshold tuning for Business customers — weekly transaction count",
    "Show FP/FN threshold tuning for Individual customers — monthly transaction amount",
    "Cluster all customers into behavioral segments and show the treemap",
    "Cluster Business customers into 4 segments",
    "Show alerts and false positive distribution across segments",
    "What does AML policy say about structuring detection thresholds?",
]

# ── Welcome message ───────────────────────────────────────────────────────────
_WELCOME = (
    f"Hello! I am your **FRAML AI Assistant** powered by **{OLLAMA_MODEL}** running locally via Ollama.\n\n"
    "I can help you with:\n"
    "- **Threshold tuning** — analyze how FP/FN trade-offs shift as alert thresholds change\n"
    "- **Smart segmentation** — cluster customers into behavioral segments using K-Means\n"
    "- **AML policy Q&A** — answer compliance questions from the knowledge base\n\n"
    f"Dataset loaded: **{_total:,} accounts** ({_biz_count:,} Business / {_ind_count:,} Individual) "
    f"| **{_alert_count:,} alerts** | **{_fp_count:,} false positives**\n\n"
    "Click a suggested prompt on the left or type your question below."
)

INITIAL_MESSAGES = [{"role": "assistant", "content": _WELCOME}]


# ── Tool helpers ──────────────────────────────────────────────────────────────
def compute_threshold_stats(df_seg, threshold):
    """
    Compute threshold sweep and return:
      - a pre-written factual interpretation (Python-generated, always accurate)
      - the raw sweep table appended for reference

    The model is instructed to copy the PRE-COMPUTED ANALYSIS verbatim and only
    add one AML insight sentence — this eliminates hallucinated numbers.
    """
    t_min = df_seg[threshold].min()
    t_max = df_seg[threshold].max()
    step  = max(1, int((t_max - t_min) / 100))

    sweep = []
    t = t_min
    while t <= t_max + step:
        fp = df_seg[(df_seg[threshold] >= t) & (df_seg["false_positives"] == 1)].shape[0]
        fn = df_seg[(df_seg[threshold] <  t) & (df_seg["false_negatives"] == 1)].shape[0]
        sweep.append((t, fp, fn))
        t += step

    # ── Derive key facts ──────────────────────────────────────────────────────
    max_fp = sweep[0][1]
    max_fn = sweep[-1][2]

    fn_first_nonzero = next(((t, fn) for t, fp, fn in sweep if fn > 0), None)
    fp_first_zero    = next(((t, fp) for t, fp, fn in sweep if fp == 0), None)
    fn_zero_end      = fn_first_nonzero[0] - step if fn_first_nonzero else t_max

    crossover = min(sweep, key=lambda x: abs(x[1] - x[2]))

    optimal = [(t, fp, fn) for t, fp, fn in sweep
               if fp <= max_fp * 0.20 and fn <= max_fn * 0.20]

    # ── Build pre-written factual interpretation ──────────────────────────────
    lines = ["=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ==="]

    # FP behaviour
    lines.append(
        f"At the lowest threshold ({t_min}), there are {max_fp} false positives."
    )
    lines.append(
        f"False positives decrease as the threshold rises."
    )
    if fp_first_zero:
        lines.append(f"False positives reach zero at threshold {fp_first_zero[0]}.")
    else:
        lines.append(f"False positives do not reach zero within the threshold range ({t_min}–{t_max}).")

    # FN behaviour
    if fn_first_nonzero:
        lines.append(
            f"False negatives are zero for all thresholds from {t_min} up to and including {fn_zero_end}."
        )
        lines.append(
            f"False negatives first become non-zero at threshold {fn_first_nonzero[0]} (FN={fn_first_nonzero[1]})."
        )
        lines.append(
            f"False negatives increase as the threshold continues to rise, "
            f"reaching {max_fn} at the highest threshold ({t_max})."
        )
    else:
        lines.append(f"False negatives remain zero across the entire threshold range.")

    # Crossover
    lines.append(
        f"The crossover point — where false positives and false negatives are closest — "
        f"is at threshold {crossover[0]} (FP={crossover[1]}, FN={crossover[2]})."
    )

    # Optimal zone
    if optimal:
        lines.append(
            f"The optimal zone (both FP and FN below 20% of their respective maximums) "
            f"spans threshold {optimal[0][0]} to {optimal[-1][0]}."
        )
    else:
        lines.append(
            "No single threshold achieves both FP and FN below 20% of their maximums simultaneously."
        )

    lines.append("=== END PRE-COMPUTED ANALYSIS ===")
    lines.append("")
    lines.append(f"Raw sweep (threshold range {t_min}–{t_max}, step={step}):")
    lines += [f"  t={t}: FP={fp}, FN={fn}" for t, fp, fn in sweep]

    return "\n".join(lines)


def compute_segment_stats(df):
    lines = []
    for seg_id, name in [(0, "Business"), (1, "Individual")]:
        seg = df[df["smart_segment_id"] == seg_id]
        lines.append(
            f"{name}: total={len(seg):,}, alerts={int(seg['alerts'].sum())}, "
            f"FP={int(seg['false_positives'].sum())}, FN={int(seg['false_negatives'].sum())}"
        )
    return "\n".join(lines)


# ── Tool executor ─────────────────────────────────────────────────────────────
def tool_executor(tool_name, tool_input):
    """Execute a tool called by an agent. Returns (result_text, fig_or_None)."""
    global DF_SS

    if tool_name == "threshold_tuning":
        segment = tool_input.get("segment", "Business")
        col     = COL_MAP.get(tool_input.get("threshold_column", "AVG_TRXNS_WEEK"), "avg_num_trxns")
        df_seg  = DF_BUSINESS if segment == "Business" else DF_INDIVIDUAL
        stats   = compute_threshold_stats(df_seg, col)
        fig, _  = lambda_ss_performance.plot_thresholds_tuning(df_seg, col, 0.1, segment)
        return stats, fig

    elif tool_name == "segment_stats":
        return compute_segment_stats(DF), None

    elif tool_name == "alerts_distribution":
        fig = lambda_ss_performance.alerts_distribution(DF)
        return compute_segment_stats(DF), fig

    elif tool_name == "cluster_analysis":
        customer_type = tool_input.get("customer_type", "All")
        n_clusters    = tool_input.get("n_clusters", 4)
        df_src        = DF_SS if DF_SS is not None else DF
        scatter_fig, stats, df_clustered = lambda_ss_performance.perform_clustering(
            df_src, customer_type, n_clusters
        )
        ss_dims = {
            "BUSINESS":   ["ACCOUNT_TYPE", "ACCOUNT_AGE_CATEGORY"],
            "INDIVIDUAL": ["ACCOUNT_TYPE", "GENDER", "AGE_CATEGORY", "INCOME_BAND"],
        }
        treemap_fig = lambda_ss_performance.smartseg_tree_dynamic(
            df_clustered, customer_type, dims=ss_dims
        )
        return stats, (scatter_fig, treemap_fig)

    elif tool_name == "prepare_segmentation_data":
        import ss_data_prep
        try:
            df_out = ss_data_prep.prepare_data()
            DF_SS  = df_out
            summary = (
                f"Data prep complete — docs/ss_segmentation_data.csv\n"
                f"Rows: {len(df_out):,} | Columns: {len(df_out.columns)}\n"
                f"customer_type: {df_out['customer_type'].value_counts().to_dict()}"
            )
            return summary, None
        except Exception as e:
            return f"Data prep error: {e}", None

    elif tool_name == "ss_cluster_analysis":
        import ss_data_prep
        if DF_SS is None:
            print("ss_cluster_analysis: DF_SS not loaded — running prepare_data() first ...")
            DF_SS = ss_data_prep.prepare_data()
        customer_type = tool_input.get("customer_type", "All")
        n_clusters    = tool_input.get("n_clusters", 4)
        scatter_fig, stats, df_clustered = lambda_ss_performance.perform_clustering(
            DF_SS, customer_type, n_clusters
        )
        ss_dims = {
            "BUSINESS":   ["ACCOUNT_TYPE", "ACCOUNT_AGE_CATEGORY"],
            "INDIVIDUAL": ["ACCOUNT_TYPE", "GENDER", "AGE_CATEGORY", "INCOME_BAND"],
        }
        treemap_fig = lambda_ss_performance.smartseg_tree_dynamic(
            df_clustered, customer_type, dims=ss_dims
        )
        return stats, (scatter_fig, treemap_fig)

    return f"Unknown tool: {tool_name}", None


# ── Chart content builder ─────────────────────────────────────────────────────
def _chart_content(tool_name, tool_input, fig):
    """Convert a tool result figure into dash-chat content blocks."""
    blocks = []

    if tool_name == "threshold_tuning":
        seg = tool_input.get("segment", "")
        col = tool_input.get("threshold_column", "")
        figs   = [fig]
        labels = [f"Threshold Tuning — {seg} / {col}"]

    elif tool_name == "alerts_distribution":
        figs   = [fig]
        labels = ["Alerts & False Positives by Segment"]

    elif tool_name in ("cluster_analysis", "ss_cluster_analysis"):
        ct     = tool_input.get("customer_type", "All")
        prefix = "SS " if tool_name == "ss_cluster_analysis" else ""
        if isinstance(fig, tuple):
            figs   = list(fig)
            labels = [f"{prefix}Cluster Scatter — {ct}", f"{prefix}Smart Segment Treemap — {ct}"]
        else:
            figs   = [fig]
            labels = [f"{prefix}Cluster Analysis — {ct}"]

    else:
        figs   = [fig] if not isinstance(fig, tuple) else list(fig)
        labels = [tool_name] * len(figs)

    for label, f in zip(labels, figs):
        blocks.append({"type": "text", "text": f"**{label}**"})
        blocks.append({
            "type": "graph",
            "props": {
                "figure": f.to_dict(),
                "config": {"responsive": True},
                "style":  {"height": "460px"},
            },
        })
    return blocks


# ── Dash app ──────────────────────────────────────────────────────────────────
server = flask.Flask(__name__)
app = Dash(
    __name__,
    server=server,
    routes_pathname_prefix="/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="FRAML AI Assistant",
)

# ── Layout ────────────────────────────────────────────────────────────────────
_sidebar = dbc.Card([
    dbc.CardBody([
        html.H5("FRAML AI Assistant", className="fw-bold mb-1"),
        html.P("Powered by Qwen2.5 via Ollama", className="text-muted small mb-3"),

        html.Hr(className="my-2"),

        # Data summary badges
        html.Div([
            html.Span("Dataset", className="fw-semibold d-block mb-1 small"),
            dbc.Badge(f"{_total:,} accounts",    color="primary",  className="me-1 mb-1"),
            dbc.Badge(f"{_biz_count:,} business",  color="secondary",className="me-1 mb-1"),
            dbc.Badge(f"{_ind_count:,} individual", color="secondary",className="me-1 mb-1"),
            dbc.Badge(f"{_alert_count:,} alerts",   color="warning",  className="me-1 mb-1"),
            dbc.Badge(f"{_fp_count:,} FP",          color="danger",   className="me-1 mb-1"),
        ], className="mb-3"),

        html.Hr(className="my-2"),

        html.Span("Suggested Prompts", className="fw-semibold d-block mb-2 small"),
        html.Div([
            dbc.Button(
                p,
                id={"type": "prompt-btn", "index": i},
                color="outline-primary",
                size="sm",
                className="mb-2 text-start w-100",
                style={"whiteSpace": "normal", "height": "auto"},
                n_clicks=0,
            )
            for i, p in enumerate(SUGGESTED_PROMPTS)
        ]),

        html.Hr(className="my-2"),

        # Model info
        html.Div([
            html.Span("Model", className="fw-semibold d-block mb-1 small"),
            dbc.Badge(OLLAMA_MODEL, color="info", className="me-1"),
            dbc.Badge("Ollama", color="dark"),
        ]),
    ])
], className="h-100 overflow-auto", style={"fontSize": "0.85rem"})

_chat_panel = html.Div([
    ChatComponent(
        id="chat-component",
        messages=INITIAL_MESSAGES,
        class_name="FRAML AI",
    )
], style={
    "height": "calc(100vh - 80px)",
    "overflow": "auto",
    "overscrollBehaviorY": "contain",
})

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(
            html.H4("FRAML AI — Threshold Tuning & Smart Segmentation",
                    className="text-center my-3 fw-bold"),
        )
    ]),
    # Body
    dbc.Row([
        dbc.Col(_sidebar,    width=3, className="pe-2"),
        dbc.Col(_chat_panel, width=9, className="ps-2"),
    ], className="g-0"),

    # Store for pending prompt from sidebar buttons
    dcc.Store(id="pending-prompt", data=None),
], fluid=True, style={"height": "100vh", "overflow": "hidden"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("pending-prompt", "data"),
    Input({"type": "prompt-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def queue_prompt(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered = ctx.triggered[0]["prop_id"]
    try:
        idx = json.loads(triggered.split(".")[0])["index"]
    except Exception:
        return no_update
    # Attach timestamp so clicking the same prompt twice still fires
    return {"query": SUGGESTED_PROMPTS[idx], "ts": time.time()}


@callback(
    Output("chat-component", "messages"),
    Input("chat-component", "new_message"),
    Input("pending-prompt", "data"),
    State("chat-component", "messages"),
    prevent_initial_call=True,
)
def handle_chat(new_message, pending_prompt, messages):
    ctx     = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Determine query source
    if "pending-prompt" in trigger and pending_prompt:
        query    = pending_prompt["query"]
        user_msg = {"role": "user", "content": query}
    elif new_message:
        query    = new_message.get("content", "")
        user_msg = new_message
    else:
        return messages

    if not query.strip():
        return messages

    updated = messages + [user_msg]

    try:
        agent_text, chart_results = orchestrator.run(query, tool_executor)
    except Exception as e:
        import traceback
        traceback.print_exc()
        bot_response = {"role": "assistant", "content": f"Sorry, something went wrong: {e}"}
        return updated + [bot_response]

    if chart_results:
        content = [{"type": "text", "text": agent_text}] if agent_text else []
        for tool_name, tool_input, fig in chart_results:
            content.extend(_chart_content(tool_name, tool_input, fig))
        bot_response = {"role": "assistant", "content": content}
    else:
        bot_response = {"role": "assistant", "content": agent_text or "(No response)"}

    return updated + [bot_response]


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)

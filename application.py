"""
application.py — FRAML AI Assistant (Qwen2.5 via Ollama)

Run:
    python application.py        # http://127.0.0.1:5000

Override the LLM endpoint if needed:
    set OLLAMA_BASE_URL=http://localhost:11434/v1
    set OLLAMA_MODEL=qwen2.5:7b
"""

import json
import re
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

from config import ALERTS_CSV, SS_CSV, SAR_CSV, CLUSTER_LABELS_CSV, OLLAMA_MODEL, OLLAMA_BASE_URL
from agents import OrchestratorAgent
import lambda_ss_performance
import lambda_rule_analysis
import make_figures

MAX_SWEEP_ROWS = 10  # max sweep points passed to model (prevents token limit cutoff)

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

DF_SS  = pd.read_csv(SS_CSV) if os.path.exists(SS_CSV) else None
DF_SAR = pd.read_csv(SAR_CSV) if os.path.exists(SAR_CSV) else None
DF_SAR_BUSINESS    = DF_SAR[DF_SAR["smart_segment_id"] == 0] if DF_SAR is not None else None
DF_SAR_INDIVIDUAL  = DF_SAR[DF_SAR["smart_segment_id"] == 1] if DF_SAR is not None else None

_total      = len(DF)
_biz_count  = len(DF_BUSINESS)
_ind_count  = len(DF_INDIVIDUAL)
_alert_count= int(DF["alerts"].sum())
_fp_count   = int(DF["false_positives"].sum())
print(f"Alerts data: {_total:,} rows | Business={_biz_count:,} Individual={_ind_count:,}")
print(f"SS data: {'loaded (' + str(len(DF_SS)) + ' rows)' if DF_SS is not None else 'not found — run python ss_data_prep.py'}")
_sar_status = (f"loaded ({len(DF_SAR)} rows, {int(DF_SAR['is_sar'].sum())} SARs)" if DF_SAR is not None else "not found - run python simulate_sars.py")
print(f"SAR simulation: {_sar_status}")

DF_RULE_SWEEP = lambda_rule_analysis.load_rule_sweep_data()
_rule_status = (
    f"loaded ({len(DF_RULE_SWEEP)} rows, {DF_RULE_SWEEP['risk_factor'].nunique()} rules)"
    if DF_RULE_SWEEP is not None
    else "not found — run python prepare_rule_sweep_data.py"
)
print(f"Rule sweep data: {_rule_status}")

DF_CLUSTER_LABELS = None
if os.path.exists(CLUSTER_LABELS_CSV):
    DF_CLUSTER_LABELS = pd.read_csv(CLUSTER_LABELS_CSV)
    print(f"Cluster labels: loaded ({len(DF_CLUSTER_LABELS):,} customers)")
else:
    print("Cluster labels: not found — run python prepare_cluster_labels.py")

COL_MAP = {
    "AVG_TRXNS_WEEK":   "avg_num_trxns",
    "AVG_TRXN_AMT":     "avg_trxn_amt",
    "TRXN_AMT_MONTHLY": "trxn_amt_monthly",
}

# SAR simulation uses ss_segmentation_data column names (slightly different from main DF)
SAR_COL_MAP = {
    "AVG_TRXNS_WEEK":   "avg_num_trxns",
    "AVG_TRXN_AMT":     "avg_weekly_trxn_amt",
    "TRXN_AMT_MONTHLY": "trxn_amt_monthly",
}

orchestrator = OrchestratorAgent()

# ── Suggested prompts ─────────────────────────────────────────────────────────
SUGGESTED_PROMPTS = [
    # Threshold Tuning
    "Show FP/FN trade-off for Business customers by monthly transaction amount",
    "Run SAR backtest for Individual customers using average weekly transactions",
    "What is the crossover threshold for Business customers on average transaction amount?",
    # Smart Segmentation
    "Cluster Business customers by transaction behavior",
    "Show me the behavioral segments for Individual customers",
    "Which cluster of Business customers has the highest transaction volume?",
    # Rule-Level Sweep
    "Show SAR backtest for Activity Deviation ACH rule",
    "Run a 2D sweep for Elder Abuse varying floor amount and age threshold",
    "Show Elder Abuse SAR backtest for Cluster 4",
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
    import math as _math
    cur_val = df_seg[threshold].median()   # use median as operational center
    raw_step = cur_val / 4
    mag = 10 ** _math.floor(_math.log10(max(raw_step, 1)))
    n = raw_step / mag
    if n < 1.5:   nice = 1
    elif n < 3.5: nice = 2
    elif n < 7.5: nice = 5
    else:         nice = 10
    step  = int(nice * mag)
    t_min = max(step, int(cur_val - 4 * step))
    t_max = int(cur_val + 4 * step)

    sweep = []
    t = t_min
    while t <= t_max + step:
        t_r = round(t, 2)
        fp = df_seg[(df_seg[threshold] >= t_r) & (df_seg["false_positives"] == 1)].shape[0]
        fn = df_seg[(df_seg[threshold] <  t_r) & (df_seg["false_negatives"] == 1)].shape[0]
        sweep.append((t_r, fp, fn))
        t += step

    # ── Derive key facts ──────────────────────────────────────────────────────
    max_fp = sweep[0][1]
    t_last = sweep[-1][0]   # actual last threshold in sweep (may be t_max + step)
    max_fn = sweep[-1][2]

    fn_first_nonzero = next(((t, fn) for t, fp, fn in sweep if fn > 0), None)
    fp_first_zero    = next(((t, fp) for t, fp, fn in sweep if fp == 0), None)
    fn_zero_end      = max(t_min, round(fn_first_nonzero[0] - step, 2)) if fn_first_nonzero else t_max

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
        if fn_zero_end > t_min:
            lines.append(
                f"False negatives are zero for all thresholds from {t_min} up to and including {fn_zero_end}."
            )
        else:
            lines.append(
                f"False negatives are already non-zero at the lowest sweep threshold ({t_min}), "
                f"meaning some customers fall below the sweep floor."
            )
        lines.append(
            f"False negatives first become non-zero at threshold {fn_first_nonzero[0]} (FN={fn_first_nonzero[1]})."
        )
        lines.append(
            f"False negatives increase as the threshold continues to rise, "
            f"reaching {max_fn} at the highest threshold ({t_last})."
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
    lines.append("(Detailed sweep table shown in the chart below.)")

    return "\n".join(lines)


def compute_segment_stats(df):
    total_alerts = int(df["alerts"].sum())
    total_accounts = len(df)
    lines = ["=== PRE-COMPUTED SEGMENT STATS (copy verbatim, do not compute new numbers) ==="]
    for seg_id, name in [(0, "Business"), (1, "Individual")]:
        seg = df[df["smart_segment_id"] == seg_id]
        n         = len(seg)
        alerts    = int(seg["alerts"].sum())
        fp        = int(seg["false_positives"].sum())
        fn        = int(seg["false_negatives"].sum())
        fp_rate   = round(100 * fp / alerts, 1) if alerts > 0 else 0
        acct_pct  = round(100 * n / total_accounts, 1) if total_accounts > 0 else 0
        alert_pct = round(100 * alerts / total_alerts, 1) if total_alerts > 0 else 0
        lines.append(
            f"{name}: accounts={n:,} ({acct_pct}% of total), "
            f"alerts={alerts:,} ({alert_pct}% of all alerts), "
            f"FP={fp:,} (FP rate={fp_rate}% of alerts), FN={fn:,}"
        )
    lines.append("=== END PRE-COMPUTED SEGMENT STATS ===")
    return "\n".join(lines)


def compute_sar_backtest(df_sar_seg, sar_col, segment_name):
    """
    Sweep thresholds across SAR customers and report how many simulated SARs
    would be caught vs. missed at each threshold level.

    Returns PRE-COMPUTED text for the model to copy verbatim.
    """
    df = df_sar_seg.dropna(subset=[sar_col]).copy()
    total_sars = int(df["is_sar"].sum())
    total_alerted = len(df)

    if total_sars == 0:
        return "No simulated SARs found for this segment and threshold column."

    t_min = df[sar_col].min()
    t_max = df[sar_col].max()
    step  = max(1, int((t_max - t_min) / 100))

    sweep = []
    t = t_min
    while t <= t_max + step:
        caught = int(((df[sar_col] >= t) & (df["is_sar"] == 1)).sum())
        missed = total_sars - caught
        sweep.append((round(t, 2), caught, missed))
        t += step

    # ── Key statistics ────────────────────────────────────────────────────────
    # Threshold where SAR catch rate first drops below 90% / 80% / 50%
    def threshold_for_rate(target_rate):
        target = int(total_sars * target_rate)
        hit = next(((t, c, m) for t, c, m in sweep if c <= target), None)
        return hit

    t90 = threshold_for_rate(0.90)   # threshold where we drop below 90% catch
    t80 = threshold_for_rate(0.80)
    t50 = threshold_for_rate(0.50)

    # Threshold where SARs first start being missed
    first_miss = next(((t, c, m) for t, c, m in sweep if m > 0), None)

    lines = ["=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ==="]
    lines.append(f"Segment: {segment_name} | Column: {sar_col}")
    lines.append(f"Total simulated SARs: {total_sars} out of {total_alerted} alerted customers ({round(100*total_sars/total_alerted,1)}% SAR filing rate).")
    lines.append("")

    lines.append(f"At the lowest threshold ({sweep[0][0]}): {sweep[0][1]} SARs caught (100.0%), 0 missed.")
    if first_miss:
        lines.append(f"SARs first begin to be missed at threshold {first_miss[0]} ({first_miss[2]} missed).")

    if t90:
        prev = next((s for s in reversed(sweep) if s[0] < t90[0] and s[1] > int(total_sars*0.90)), None)
        t90_keep = prev[0] if prev else sweep[0][0]
        lines.append(f"To catch at least 90% of SARs, threshold must stay at or below {t90_keep} ({int(total_sars*0.90)+1} of {total_sars} caught).")
    if t80:
        prev = next((s for s in reversed(sweep) if s[0] < t80[0] and s[1] > int(total_sars*0.80)), None)
        t80_keep = prev[0] if prev else sweep[0][0]
        lines.append(f"To catch at least 80% of SARs, threshold must stay at or below {t80_keep} ({int(total_sars*0.80)+1} of {total_sars} caught).")
    if t50:
        prev = next((s for s in reversed(sweep) if s[0] < t50[0] and s[1] > int(total_sars*0.50)), None)
        t50_keep = prev[0] if prev else sweep[0][0]
        lines.append(f"To catch at least 50% of SARs, threshold must stay at or below {t50_keep} ({int(total_sars*0.50)+1} of {total_sars} caught).")

    lines.append(f"At the highest threshold ({sweep[-1][0]}): {sweep[-1][1]} SARs caught, {sweep[-1][2]} missed (100.0% missed).")
    lines.append("=== END PRE-COMPUTED SAR BACKTEST ===")
    lines.append("(Detailed sweep table shown in the chart below.)")

    return "\n".join(lines)


# ── Cluster filter helper ─────────────────────────────────────────────────────
def _filter_by_cluster(df_rule_sweep, cluster):
    """
    Return df_rule_sweep filtered to customers in a specific behavioral cluster.
    cluster: int (1-based) or None (no filtering).
    Requires DF_CLUSTER_LABELS to be loaded; silently skips filter if not available.
    """
    if cluster is None or DF_CLUSTER_LABELS is None:
        return df_rule_sweep
    cluster = int(cluster)
    ids = DF_CLUSTER_LABELS[DF_CLUSTER_LABELS["cluster"] == cluster]["customer_id"]
    filtered = df_rule_sweep[df_rule_sweep["customer_id"].isin(ids)]
    print(f"[cluster filter] cluster={cluster}: {len(filtered)} / {len(df_rule_sweep)} rows")
    return filtered


# ── Tool executor ─────────────────────────────────────────────────────────────
_cluster_cache = {}  # caches last clustering result for DISPLAY_CLUSTERS filtering
_current_query = ""  # set before each orchestrator.run() so tool_executor can read it

def tool_executor(tool_name, tool_input):
    """Execute a tool called by an agent. Returns (result_text, fig_or_None)."""
    global DF_SS, _cluster_cache

    if tool_name == "threshold_tuning":
        segment = tool_input.get("segment", "Business")
        raw_col = tool_input.get("threshold_column", "AVG_TRXNS_WEEK")
        col     = COL_MAP.get(raw_col)
        if col is None:
            return (f"'{raw_col}' is not a valid threshold_column. "
                    f"Valid options: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY."), None
        df_seg  = DF_BUSINESS if segment == "Business" else DF_INDIVIDUAL
        stats   = compute_threshold_stats(df_seg, col)
        line_fig, _ = lambda_ss_performance.plot_thresholds_tuning(df_seg, col, 0.1, segment)
        tbl_fig = make_figures.threshold_tuning_figure(df_seg, col, segment)
        return stats, (line_fig, tbl_fig)

    elif tool_name == "segment_stats":
        tbl_fig = make_figures.segment_stats_figure(DF)
        return compute_segment_stats(DF), tbl_fig

    elif tool_name == "sar_backtest":
        if DF_SAR is None:
            return "SAR simulation data not found. Run python simulate_sars.py first.", None
        segment    = tool_input.get("segment", "Business")
        raw_col    = tool_input.get("threshold_column", "TRXN_AMT_MONTHLY")
        col        = SAR_COL_MAP.get(raw_col)
        if col is None:
            return (f"'{raw_col}' is not a valid threshold_column. "
                    f"Valid options: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY."), None
        df_sar_seg = DF_SAR_BUSINESS if segment == "Business" else DF_SAR_INDIVIDUAL
        stats      = compute_sar_backtest(df_sar_seg, col, segment)
        tbl_fig    = make_figures.sar_backtest_figure(df_sar_seg, col, segment)
        return stats, tbl_fig

    elif tool_name == "rule_2d_sweep":
        if DF_RULE_SWEEP is None:
            return "Rule sweep data not found. Run python prepare_rule_sweep_data.py first.", None
        risk_factor = tool_input.get("risk_factor", "")
        param1      = tool_input.get("sweep_param_1") or None
        param2      = tool_input.get("sweep_param_2") or None
        cluster     = tool_input.get("cluster", None)
        df_sweep    = _filter_by_cluster(DF_RULE_SWEEP, cluster)
        text, grid  = lambda_rule_analysis.compute_rule_2d_sweep(df_sweep, risk_factor, param1, param2)
        heatmap     = make_figures.rule_2d_heatmap(grid) if grid else None
        return text, heatmap

    elif tool_name == "list_rules":
        if DF_RULE_SWEEP is None:
            return "Rule sweep data not found. Run python prepare_rule_sweep_data.py first.", None
        tbl_fig = make_figures.rule_list_figure(DF_RULE_SWEEP)
        return lambda_rule_analysis.list_rules_text(DF_RULE_SWEEP), tbl_fig

    elif tool_name == "rule_sar_backtest":
        if DF_RULE_SWEEP is None:
            return "Rule sweep data not found. Run python prepare_rule_sweep_data.py first.", None
        risk_factor = tool_input.get("risk_factor", tool_input.get("rule_code", ""))
        sweep_param = tool_input.get("sweep_param", None)
        cluster     = tool_input.get("cluster", None)
        df_sweep    = _filter_by_cluster(DF_RULE_SWEEP, cluster)
        stats   = lambda_rule_analysis.compute_rule_sar_sweep(df_sweep, risk_factor, sweep_param)
        tbl_fig = make_figures.rule_sweep_figure(df_sweep, risk_factor, sweep_param)
        return stats, tbl_fig

    elif tool_name == "alerts_distribution":
        bar_fig = lambda_ss_performance.alerts_distribution(DF)
        tbl_fig = make_figures.segment_stats_figure(DF)
        return compute_segment_stats(DF), (bar_fig, tbl_fig)

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
        customer_type   = tool_input.get("customer_type", "All")
        n_clusters      = tool_input.get("n_clusters", 4)
        filter_clusters = tool_input.get("filter_clusters", None)

        # GAP-1 patch disabled for V2 testing
        # _n_match = re.search(r'\b([2-8])\s*clusters?\b', _current_query, re.IGNORECASE)
        # if not _n_match:
        #     _n_match = re.search(r'\binto\s+([2-8])\b', _current_query, re.IGNORECASE)
        # if _n_match:
        #     n_clusters = int(_n_match.group(1))

        ss_dims = {
            "BUSINESS":   ["ACCOUNT_TYPE", "ACCOUNT_AGE_CATEGORY"],
            "INDIVIDUAL": ["ACCOUNT_TYPE", "GENDER", "AGE_CATEGORY", "INCOME_BAND"],
        }

        if filter_clusters and _cluster_cache:
            # Second call: use cached data, skip re-clustering
            df_clustered = _cluster_cache["df_clustered"]
            scatter_fig  = _cluster_cache["scatter_fig"]
            stats        = _cluster_cache["stats"]

            # Filter scatter: keep only traces for requested clusters
            import plotly.graph_objects as go
            keep_names = {f"Cluster {c}" for c in filter_clusters}
            filtered_scatter = go.Figure(
                data=[t for t in scatter_fig.data if t.name in keep_names],
                layout=scatter_fig.layout,
            )
            filtered_scatter.update_layout(title=filtered_scatter.layout.title.text +
                                           f" [filtered: clusters {filter_clusters}]")

            # Filter treemap: keep only rows for requested clusters (0-based internally)
            cluster_vals = sorted(df_clustered["cluster"].unique())
            keep_0based  = {cluster_vals[c - 1] for c in filter_clusters
                            if 1 <= c <= len(cluster_vals)}
            df_filtered  = df_clustered[df_clustered["cluster"].isin(keep_0based)].copy()
            treemap_fig  = lambda_ss_performance.smartseg_tree_dynamic(
                df_filtered, f"{customer_type} (clusters {filter_clusters})", dims=ss_dims
            )
            return f"Filtered to clusters {filter_clusters}.\n\n{stats}", (filtered_scatter, treemap_fig)

        else:
            # First call: run full clustering and cache results
            scatter_fig, stats, df_clustered = lambda_ss_performance.perform_clustering(
                DF_SS, customer_type, n_clusters
            )
            _cluster_cache.update({
                "df_clustered": df_clustered,
                "scatter_fig":  scatter_fig,
                "stats":        stats,
                "customer_type": customer_type,
            })
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
        figs   = list(fig) if isinstance(fig, tuple) else [fig]
        labels = [f"Threshold Tuning — {seg} / {col}", f"Threshold Table — {seg} / {col}"][:len(figs)]

    elif tool_name == "alerts_distribution":
        figs   = list(fig) if isinstance(fig, tuple) else [fig]
        labels = ["Alerts & False Positives by Segment", "Segment Statistics Table"][:len(figs)]

    elif tool_name == "segment_stats":
        figs   = [fig]
        labels = ["Segment Statistics"]

    elif tool_name == "sar_backtest":
        seg = tool_input.get("segment", "")
        col = tool_input.get("threshold_column", "")
        figs   = [fig]
        labels = [f"SAR Backtest — {seg} / {col}"]

    elif tool_name == "list_rules":
        figs   = [fig]
        labels = ["AML Rule Performance Overview"]

    elif tool_name == "rule_2d_sweep":
        from lambda_rule_analysis import RULE_CATALOGUE, _match_rule
        rf      = tool_input.get("risk_factor", "")
        p1      = tool_input.get("sweep_param_1") or None
        p2      = tool_input.get("sweep_param_2") or None
        cluster = tool_input.get("cluster", None)
        _, entry = _match_rule(rf)
        if entry and (p1 is None or p2 is None):
            d1, d2 = entry.get("default_2d", (None, None))
            if p1 is None: p1 = d1
            if p2 is None: p2 = d2
        cluster_tag = f" [Cluster {cluster}]" if cluster else ""
        figs   = [fig]
        labels = [f"2D Sweep — {rf} ({p1 or '?'} x {p2 or '?'}){cluster_tag}"]

    elif tool_name == "rule_sar_backtest":
        from lambda_rule_analysis import RULE_CATALOGUE, _match_rule
        rf      = tool_input.get("risk_factor", tool_input.get("rule_code", ""))
        sp      = tool_input.get("sweep_param") or None
        cluster = tool_input.get("cluster", None)
        _, entry = _match_rule(rf)
        if sp is None and entry:
            sp = entry["default_sweep"]
        cluster_tag = f" [Cluster {cluster}]" if cluster else ""
        figs   = [fig]
        labels = [f"Rule SAR Sweep — {rf} / {sp or 'default'}{cluster_tag}"]

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
        fig_h = f.layout.height if f.layout.height else 460
        blocks.append({"type": "text", "text": f"**{label}**"})
        blocks.append({
            "type": "graph",
            "props": {
                "figure": f.to_dict(),
                "config": {"responsive": True},
                "style":  {"height": f"{fig_h}px", "width": "100%"},
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
            *[html.Span("Threshold Tuning", className="text-muted d-block mb-1 small fst-italic")],
            *[html.Div([
                dbc.Button(
                    p,
                    id={"type": "prompt-btn", "index": i},
                    color="outline-primary",
                    size="sm",
                    className="text-start flex-grow-1",
                    style={"whiteSpace": "normal", "height": "auto"},
                    n_clicks=0,
                ),
                dcc.Clipboard(content=p, title="Copy", style={"cursor": "pointer", "paddingTop": "2px"}),
            ], className="d-flex align-items-start gap-1 mb-2") for i, p in enumerate(SUGGESTED_PROMPTS[:3])],
            *[html.Span("Smart Segmentation", className="text-muted d-block mb-1 mt-1 small fst-italic")],
            *[html.Div([
                dbc.Button(
                    p,
                    id={"type": "prompt-btn", "index": i + 3},
                    color="outline-success",
                    size="sm",
                    className="text-start flex-grow-1",
                    style={"whiteSpace": "normal", "height": "auto"},
                    n_clicks=0,
                ),
                dcc.Clipboard(content=p, title="Copy", style={"cursor": "pointer", "paddingTop": "2px"}),
            ], className="d-flex align-items-start gap-1 mb-2") for i, p in enumerate(SUGGESTED_PROMPTS[3:6])],
            *[html.Span("Rule-Level Sweep", className="text-muted d-block mb-1 mt-1 small fst-italic")],
            *[html.Div([
                dbc.Button(
                    p,
                    id={"type": "prompt-btn", "index": i + 6},
                    color="outline-warning",
                    size="sm",
                    className="text-start flex-grow-1",
                    style={"whiteSpace": "normal", "height": "auto"},
                    n_clicks=0,
                ),
                dcc.Clipboard(content=p, title="Copy", style={"cursor": "pointer", "paddingTop": "2px"}),
            ], className="d-flex align-items-start gap-1 mb-2") for i, p in enumerate(SUGGESTED_PROMPTS[6:])],
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
        messages=[],
        class_name="FRAML AI",
    )
], id="chat-scroll-container", style={
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
    dcc.Store(id="scroll-dummy"),
    # One-shot interval to inject welcome message after page load
    dcc.Interval(id="welcome-interval", interval=300, max_intervals=1),
], fluid=True, style={"height": "100vh", "overflow": "hidden"})


# ── Auto-scroll chat to bottom whenever messages update ───────────────────────
app.clientside_callback(
    """
    function(messages) {
        setTimeout(function() {
            var el = document.getElementById('chat-scroll-container');
            if (el) { el.scrollTop = el.scrollHeight; }
        }, 150);
        return null;
    }
    """,
    Output("scroll-dummy", "data"),
    Input("chat-component", "messages"),
    prevent_initial_call=True,
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("chat-component", "messages", allow_duplicate=True),
    Input("welcome-interval", "n_intervals"),
    prevent_initial_call=True,
)
def show_welcome(n):
    return INITIAL_MESSAGES


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
    elif new_message and new_message.get("role") == "user":
        query    = new_message.get("content", "")
        user_msg = new_message
    else:
        return messages

    if not query.strip():
        return messages

    updated = messages + [user_msg]

    try:
        global _current_query
        _current_query = query
        last_assistant = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant" and isinstance(m.get("content"), str)),
            ""
        )
        agent_text, chart_results = orchestrator.run(query, tool_executor, last_assistant)
    except Exception as e:
        import traceback
        traceback.print_exc()
        bot_response = {"role": "assistant", "content": f"Sorry, something went wrong: {e}"}
        return updated + [bot_response]

    # Parse DISPLAY_CLUSTERS directive and filter charts if present
    # Only honour the directive if the user actually asked to filter clusters
    _filter_keywords = re.search(
        r'\b(show only|only cluster|highest risk|lowest|top \d|filter)\b',
        _current_query, re.IGNORECASE
    )
    _dc_match = re.search(r'DISPLAY_CLUSTERS:\s*([\d,\s]+)', agent_text or "")
    if _dc_match and chart_results and _cluster_cache and _filter_keywords:
        filter_nums = [int(x.strip()) for x in _dc_match.group(1).split(',') if x.strip().isdigit()]
        if filter_nums:
            import plotly.graph_objects as _go  # noqa: already imported at top via px
            df_clustered  = _cluster_cache["df_clustered"]
            scatter_fig   = _cluster_cache["scatter_fig"]
            customer_type = _cluster_cache.get("customer_type", "All")
            ss_dims = {
                "BUSINESS":   ["ACCOUNT_TYPE", "ACCOUNT_AGE_CATEGORY"],
                "INDIVIDUAL": ["ACCOUNT_TYPE", "GENDER", "AGE_CATEGORY", "INCOME_BAND"],
            }
            # Filter scatter traces
            keep_names = {f"Cluster {c}" for c in filter_nums}
            filtered_scatter = _go.Figure(
                data=[t for t in scatter_fig.data if t.name in keep_names],
                layout=scatter_fig.layout,
            )
            # Filter treemap df to requested clusters (0-based internally)
            cluster_vals = sorted(df_clustered["cluster"].unique())
            keep_0based  = {cluster_vals[c - 1] for c in filter_nums if 1 <= c <= len(cluster_vals)}
            df_filtered  = df_clustered[df_clustered["cluster"].isin(keep_0based)].copy()
            treemap_fig  = lambda_ss_performance.smartseg_tree_dynamic(
                df_filtered, f"{customer_type} — clusters {filter_nums}", dims=ss_dims
            )
            chart_results = [("ss_cluster_analysis",
                              {"customer_type": customer_type, "filter_clusters": filter_nums},
                              (filtered_scatter, treemap_fig))]
    # Strip DISPLAY_CLUSTERS line and PRE-COMPUTED ANALYSIS markers from displayed text
    agent_text = re.sub(r'\s*DISPLAY_CLUSTERS:[\d,\s]*', '', agent_text or "").strip()
    agent_text = re.sub(r'===.*?PRE-COMPUTED ANALYSIS.*?===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===\s*END PRE-COMPUTED ANALYSIS\s*===\n?', '', agent_text).strip()
    agent_text = re.sub(r'PRE-COMPUTED ANALYSIS[:\s]*\n?', '', agent_text).strip()
    agent_text = re.sub(r'===.*?PRE-COMPUTED SEGMENT STATS.*?===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===\s*END PRE-COMPUTED SEGMENT STATS\s*===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===.*?PRE-COMPUTED SAR BACKTEST.*?===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===\s*END PRE-COMPUTED SAR BACKTEST\s*===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===.*?PRE-COMPUTED RULE SWEEP.*?===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===\s*END RULE SWEEP\s*===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===.*?PRE-COMPUTED RULE LIST.*?===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===\s*END RULE LIST\s*===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===.*?PRE-COMPUTED 2D SWEEP.*?===\n?', '', agent_text).strip()
    agent_text = re.sub(r'===\s*END 2D SWEEP\s*===\n?', '', agent_text).strip()

    if chart_results:
        content = [{"type": "text", "text": agent_text}] if agent_text else []
        for tool_name, tool_input, fig in chart_results:
            content.extend(_chart_content(tool_name, tool_input, fig))
        bot_response = {"role": "assistant", "content": content}
    else:
        bot_response = {"role": "assistant", "content": agent_text or "(No response)"}

    return updated + [bot_response]


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)

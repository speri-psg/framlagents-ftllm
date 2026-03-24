"""
generate_training_data.py

Generates FRAML tool-calling training traces for fine-tuning Qwen2.5-7B-Instruct.
Loads the real custs_accts_txns_alerts.csv to compute actual FP/FN statistics so
that every example contains real numbers, not synthetic placeholders.

Output: data/framl_train.jsonl  (OpenAI messages format, ready for Unsloth)

Usage:
    python generate_training_data.py
    python generate_training_data.py --data_path /path/to/custs_accts_txns_alerts.csv
    python generate_training_data.py --out data/my_train.jsonl
"""

import argparse
import json
import random
import uuid
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = str(_THIS_DIR.parent.parent / "framlagents" / "docs" / "custs_accts_txns_alerts.csv")
OUT_DIR = _THIS_DIR / "data"

SEED = 42

# ---------------------------------------------------------------------------
# System prompt (matches what the fine-tuned model will use at inference)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a FRAML (Fraud + AML) analytics AI assistant. "
    "You analyze false positive/false negative trade-offs in AML alert thresholds, "
    "perform customer behavioral segmentation, and interpret clustering results. "
    "Use the available tools to retrieve data, then provide clear, analytical insights. "
    "Be concise and reference specific numbers when interpreting results."
)

# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function-calling format
# (used as metadata in each training example so the model learns schema)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "threshold_tuning",
            "description": (
                "Analyze false positive / false negative trade-offs as a threshold column is swept "
                "for a given customer segment. FP decreases and FN increases as the threshold rises. "
                "Returns a sweep table of FP and FN counts at each threshold step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "enum": ["Business", "Individual"],
                        "description": "Customer segment to analyze.",
                    },
                    "threshold_column": {
                        "type": "string",
                        "enum": ["AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"],
                        "description": "Column to sweep as the alert threshold.",
                    },
                },
                "required": ["segment", "threshold_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_stats",
            "description": (
                "Return summary statistics (total accounts, alerts, false positives, false negatives) "
                "broken down by Business and Individual segments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_analysis",
            "description": (
                "Run K-Means clustering on customer/transaction features and return a PCA scatter plot "
                "and treemap showing cluster composition. Alert columns are excluded to avoid data leakage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "enum": ["Business", "Individual", "All"],
                        "description": "Filter to a specific customer type, or 'All' for both.",
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": (
                            "Number of K-Means clusters (2–8). "
                            "Use 0 to auto-select via the elbow method."
                        ),
                    },
                },
                "required": ["customer_type", "n_clusters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alerts_distribution",
            "description": (
                "Return a bar chart comparing total alerts and false positives "
                "across Business and Individual segments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ---------------------------------------------------------------------------
# Column name mappings
# ---------------------------------------------------------------------------
COL_MAP = {
    "AVG_TRXNS_WEEK": "avg_num_trxns",
    "AVG_TRXN_AMT": "avg_trxn_amt",
    "TRXN_AMT_MONTHLY": "trxn_amt_monthly",
}
COL_LABELS = {
    "AVG_TRXNS_WEEK": "average weekly transaction count",
    "AVG_TRXN_AMT": "average transaction amount",
    "TRXN_AMT_MONTHLY": "monthly transaction amount",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t")
    df = df.rename(columns={
        "AVG_TRXNS_WEEK": "avg_num_trxns",
        "AVG_TRXN_AMT": "avg_trxn_amt",
        "TRXN_AMT_MONTHLY": "trxn_amt_monthly",
        "FP": "false_positives",
        "FN": "false_negatives",
        "ALERT": "alerts",
        "CUSTOMER_TYPE": "customer_type",
    })
    df["alerts"] = df["alerts"].map({"Yes": 1, "No": 0})
    df["false_positives"] = df["false_positives"].map({"Yes": 1, "No": 0})
    df["false_negatives"] = df["false_negatives"].map({"Yes": 1, "No": 0})
    df["smart_segment_id"] = df["customer_type"].map({"BUSINESS": 0, "INDIVIDUAL": 1})
    return df


# ---------------------------------------------------------------------------
# Statistics computation helpers
# ---------------------------------------------------------------------------
def compute_threshold_sweep(df_seg: pd.DataFrame, col: str) -> dict:
    """
    Run the same sweep logic as application.py compute_threshold_stats().
    Returns a dict with the result_text (used as the tool result in training)
    plus key statistics for building the assistant's analytical response.
    """
    t_min = int(df_seg[col].min())
    t_max = int(df_seg[col].max())
    step = max(1, int((t_max - t_min) / 100))

    sweep = []
    t = t_min
    while t <= t_max + step:
        fp = int(df_seg[(df_seg[col] >= t) & (df_seg["false_positives"] == 1)].shape[0])
        fn = int(df_seg[(df_seg[col] < t) & (df_seg["false_negatives"] == 1)].shape[0])
        sweep.append({"t": t, "fp": fp, "fn": fn})
        t += step

    # Crossover: first point where FP drops below FN
    crossover = None
    for i in range(len(sweep) - 1):
        if sweep[i]["fp"] >= sweep[i]["fn"] and sweep[i + 1]["fp"] <= sweep[i + 1]["fn"]:
            crossover = sweep[i]
            break

    # Optimal: minimises FP + FN
    optimal = min(sweep, key=lambda x: x["fp"] + x["fn"])

    # Build result_text: first 10 rows + ellipsis + last 5 rows
    rows = [f"  threshold={r['t']}: FP={r['fp']}, FN={r['fn']}" for r in sweep]
    mid = "  ..." if len(rows) > 15 else ""
    result_text = (
        f"Threshold: {col}, range {t_min}-{t_max}\n"
        + "\n".join(rows[:10])
        + ("\n" + mid + "\n" + "\n".join(rows[-5:]) if mid else "")
    )

    return {
        "result_text": result_text,
        "sweep": sweep,
        "crossover": crossover,
        "t_min": t_min,
        "t_max": t_max,
        "step": step,
        "optimal": optimal,
        "fp_at_min": sweep[0]["fp"],
        "fn_at_max": sweep[-1]["fn"],
    }


def compute_segment_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for seg_id, name in [(0, "Business"), (1, "Individual")]:
        seg = df[df["smart_segment_id"] == seg_id]
        total = len(seg)
        alerts = int(seg["alerts"].sum())
        fp = int(seg["false_positives"].sum())
        fn = int(seg["false_negatives"].sum())
        stats[name] = {
            "total": total,
            "alerts": alerts,
            "fp": fp,
            "fn": fn,
            "fp_rate": round(fp / alerts * 100, 1) if alerts > 0 else 0.0,
        }
    result_text = "\n".join(
        f"{name}: total={s['total']}, alerts={s['alerts']}, FP={s['fp']}, FN={s['fn']}"
        for name, s in stats.items()
    )
    return {"stats": stats, "result_text": result_text}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cid() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def example(messages: list) -> dict:
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Example generators
# ---------------------------------------------------------------------------

def make_threshold_examples(df: pd.DataFrame) -> list:
    """24 examples — 6 segment×column combos × 4 query phrasings each."""

    query_variants = {
        ("Business", "AVG_TRXNS_WEEK"): [
            "Show me the FP/FN trade-off for business customers using weekly transaction count as the threshold",
            "How does adjusting the AVG_TRXNS_WEEK threshold affect false positives for business accounts?",
            "Tune the alert threshold for the business segment based on transaction frequency",
            "What's the optimal weekly transaction threshold for business AML alerts?",
        ],
        ("Business", "AVG_TRXN_AMT"): [
            "Show FP/FN analysis for business customers using average transaction amount threshold",
            "How does the avg transaction amount threshold impact false positives for business accounts?",
            "What's the best AVG_TRXN_AMT threshold to minimise unproductive alerts for business?",
            "Tune the average transaction amount threshold for the business segment",
        ],
        ("Business", "TRXN_AMT_MONTHLY"): [
            "Show me threshold tuning for the business segment using monthly transaction amount",
            "How does monthly transaction volume threshold affect FP/FN for business customers?",
            "What monthly amount threshold minimises wasted alert volume for business accounts?",
            "Tune the TRXN_AMT_MONTHLY threshold for business AML detection",
        ],
        ("Individual", "AVG_TRXNS_WEEK"): [
            "Show FP/FN trade-off for individual customers using weekly transaction count threshold",
            "How does the AVG_TRXNS_WEEK threshold affect retail customer alert quality?",
            "Tune the weekly transaction threshold for individual account holders",
            "What's the optimal transaction frequency threshold for individual AML alerts?",
        ],
        ("Individual", "AVG_TRXN_AMT"): [
            "Show threshold tuning for individual customers using average transaction amount",
            "How does avg transaction amount threshold impact false positives for retail accounts?",
            "What's the best average transaction amount threshold for individual customers?",
            "Tune AVG_TRXN_AMT for the individual segment to improve alert efficiency",
        ],
        ("Individual", "TRXN_AMT_MONTHLY"): [
            "Show FP/FN analysis for individual customers using monthly transaction amount threshold",
            "How does the monthly amount threshold affect alert quality for retail customers?",
            "Tune the TRXN_AMT_MONTHLY threshold for the individual segment",
            "What monthly volume threshold gives the best FP/FN balance for individual accounts?",
        ],
    }

    examples = []
    for (segment, col_key), queries in query_variants.items():
        df_col = COL_MAP[col_key]
        df_seg = df[df["smart_segment_id"] == (0 if segment == "Business" else 1)]
        s = compute_threshold_sweep(df_seg, df_col)
        col_label = COL_LABELS[col_key]

        crossover_text = (
            f"The FP and FN curves cross near threshold={s['crossover']['t']} "
            f"(FP={s['crossover']['fp']}, FN={s['crossover']['fn']})."
            if s["crossover"]
            else "The FP and FN curves do not clearly cross within this range — "
                 "consider whether the full customer population is being captured."
        )

        response = (
            f"Based on the threshold sweep for **{segment}** customers using "
            f"**{col_label}** (range {s['t_min']}–{s['t_max']}, step={s['step']}):\n\n"
            f"- **At minimum threshold ({s['t_min']}):** FP={s['fp_at_min']}, FN=0 — "
            f"maximum sensitivity, highest alert volume.\n"
            f"- **At maximum threshold ({s['t_max']}):** FP near 0, FN={s['fn_at_max']} — "
            f"almost no alerts fire, but true SAR-worthy activity is missed.\n"
            f"- **Crossover:** {crossover_text}\n"
            f"- **Optimal threshold:** {s['optimal']['t']} minimises total error "
            f"(FP={s['optimal']['fp']}, FN={s['optimal']['fn']}, "
            f"combined={s['optimal']['fp'] + s['optimal']['fn']}).\n\n"
            f"**Recommendation:** Set the {col_label} threshold near **{s['optimal']['t']}** "
            f"for {segment} accounts. This balances investigator workload against missed-SAR risk. "
            f"Validate against your team's daily alert capacity and document the rationale for "
            f"model risk management purposes before deployment."
        )

        for query in queries:
            call_id = cid()
            examples.append(example([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "threshold_tuning",
                            "arguments": json.dumps({
                                "segment": segment,
                                "threshold_column": col_key,
                            }),
                        },
                    }],
                },
                {"role": "tool", "content": s["result_text"], "tool_call_id": call_id},
                {"role": "assistant", "content": response},
            ]))

    return examples


def make_segment_stats_examples(seg_stats: dict) -> list:
    """6 examples — different phrasings of segment overview questions."""

    queries = [
        "Give me an overview of alert statistics across business and individual segments",
        "How many false positives do we have across both customer segments?",
        "What's the alert and FP breakdown for business vs individual customers?",
        "Show me summary alert stats for all segments",
        "How many accounts are generating alerts and how many are false positives by segment?",
        "Compare alert volumes and false positive counts across business and individual customer types",
    ]

    s = seg_stats["stats"]
    biz, ind = s["Business"], s["Individual"]
    biz_fp_pct = biz["fp_rate"]
    ind_fp_pct = ind["fp_rate"]
    total_alerts = biz["alerts"] + ind["alerts"]
    total_fp = biz["fp"] + ind["fp"]
    overall_fp_pct = round(total_fp / total_alerts * 100, 1) if total_alerts > 0 else 0.0

    response = (
        f"**Segment Alert Summary:**\n\n"
        f"| Segment | Accounts | Alerts | False Positives | False Negatives | FP Rate |\n"
        f"|---------|----------|--------|-----------------|-----------------|----------|\n"
        f"| Business | {biz['total']:,} | {biz['alerts']:,} | {biz['fp']:,} | {biz['fn']:,} | {biz_fp_pct}% |\n"
        f"| Individual | {ind['total']:,} | {ind['alerts']:,} | {ind['fp']:,} | {ind['fn']:,} | {ind_fp_pct}% |\n\n"
        f"**Key observations:**\n"
        f"- Total alerts: {total_alerts:,} — {total_fp:,} are false positives "
        f"({overall_fp_pct}% FP rate overall).\n"
        f"- Business FP rate ({biz_fp_pct}%) vs Individual ({ind_fp_pct}%) — "
        f"{'Business generates proportionally more FPs.' if biz_fp_pct > ind_fp_pct else 'Individual generates proportionally more FPs.'}\n"
        f"- Individual segment is {'larger' if ind['total'] > biz['total'] else 'smaller'} "
        f"({ind['total']:,} accounts vs {biz['total']:,}) and drives "
        f"{'more' if ind['alerts'] > biz['alerts'] else 'fewer'} absolute alerts.\n\n"
        f"Consider separate threshold tuning for each segment given the different FP rates."
    )

    examples = []
    for query in queries:
        call_id = cid()
        examples.append(example([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "segment_stats", "arguments": "{}"},
                }],
            },
            {"role": "tool", "content": seg_stats["result_text"], "tool_call_id": call_id},
            {"role": "assistant", "content": response},
        ]))

    return examples


def make_cluster_analysis_examples() -> list:
    """15 examples across 4 scenarios × varying query phrasings."""

    scenarios = [
        ("All", 4, [
            "Run a cluster analysis on all customers with 4 segments",
            "Segment all customers into 4 behavioral clusters",
            "Show me 4-cluster K-Means analysis across all customer types",
            "Cluster all accounts into 4 groups based on transaction behavior",
        ]),
        ("Business", 4, [
            "Run cluster analysis on business accounts with 4 clusters",
            "Segment business customers into 4 behavioral clusters",
            "Show me clustering for business accounts",
            "Identify 4 behavioral segments within the business customer population",
        ]),
        ("Individual", 4, [
            "Cluster individual customers by transaction behavior into 4 groups",
            "Run K-Means on retail accounts to find 4 behavioral segments",
            "Show me 4 clusters for individual account holders",
        ]),
        ("All", 0, [
            "Let the model choose the optimal number of clusters for all customers",
            "Auto-select the number of clusters using the elbow method",
            "Run clustering with automatic cluster count selection",
            "Use the elbow method to determine how many segments we should have",
        ]),
    ]

    examples = []
    for customer_type, n_clusters, queries in scenarios:
        n_desc = "auto-selected via elbow method" if n_clusters == 0 else str(n_clusters)
        tool_result = (
            f"Clustering complete. customer_type={customer_type}, "
            f"n_clusters={'auto' if n_clusters == 0 else n_clusters}. "
            f"Charts generated: PCA scatter plot, smart-segment treemap."
        )
        response = (
            f"K-Means clustering has been run on **{customer_type.lower()}** customers "
            f"with **{n_desc}** clusters. The PCA scatter and treemap are displayed above.\n\n"
            f"**Reading the results:**\n"
            f"- **Scatter plot:** Each point is a customer, coloured by cluster. "
            f"Tight, well-separated clusters indicate strong behavioural differentiation.\n"
            f"- **Treemap:** Drill-down from cluster → customer type → account type. "
            f"Tile size = number of accounts in that group.\n\n"
            f"**AML interpretation:** Look for clusters with high `avg_num_trxns` and low "
            f"`avg_trxn_amt` — a structuring signature. Clusters with very high "
            f"`trxn_amt_monthly` but few transactions may represent large-wire or trade-finance risk. "
            f"These high-risk clusters warrant tighter (lower) alert thresholds.\n\n"
            f"Use this segmentation to set **per-cluster thresholds** rather than a single "
            f"institution-wide threshold — run the `threshold_tuning` tool on each segment."
        )

        for query in queries:
            call_id = cid()
            examples.append(example([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "cluster_analysis",
                            "arguments": json.dumps({
                                "customer_type": customer_type,
                                "n_clusters": n_clusters,
                            }),
                        },
                    }],
                },
                {"role": "tool", "content": tool_result, "tool_call_id": call_id},
                {"role": "assistant", "content": response},
            ]))

    return examples


def make_alerts_distribution_examples(seg_stats: dict) -> list:
    """4 examples — different phrasings of alerts distribution queries."""

    queries = [
        "Show me how alerts are distributed across segments",
        "Compare alert volumes and false positives between business and individual customers",
        "What does the alerts distribution look like across customer segments?",
        "Show me a breakdown of alerts and FP counts by segment",
    ]

    s = seg_stats["stats"]
    biz, ind = s["Business"], s["Individual"]
    response = (
        f"The alerts distribution chart compares total alerts and false positives "
        f"for Business and Individual segments.\n\n"
        f"- **Business:** {biz['alerts']:,} alerts, {biz['fp']:,} false positives "
        f"({biz['fp_rate']}% FP rate)\n"
        f"- **Individual:** {ind['alerts']:,} alerts, {ind['fp']:,} false positives "
        f"({ind['fp_rate']}% FP rate)\n\n"
        f"Individual accounts generate {'more' if ind['alerts'] > biz['alerts'] else 'fewer'} "
        f"absolute alerts primarily due to their larger population size "
        f"({ind['total']:,} vs {biz['total']:,} accounts). "
        f"On a per-capita basis, "
        f"{'business accounts are more alert-prone.' if biz['alerts']/biz['total'] > ind['alerts']/ind['total'] else 'individual accounts are more alert-prone.'} "
        f"Consider per-capita rates and FP ratios before applying differential thresholds."
    )

    examples = []
    for query in queries:
        call_id = cid()
        examples.append(example([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "alerts_distribution", "arguments": "{}"},
                }],
            },
            {"role": "tool", "content": seg_stats["result_text"], "tool_call_id": call_id},
            {"role": "assistant", "content": response},
        ]))

    return examples


def make_multi_tool_examples(df: pd.DataFrame, seg_stats: dict) -> list:
    """
    3 examples where the agent calls segment_stats THEN threshold_tuning.
    Teaches the model to chain multiple tool calls when the query spans both.
    """
    df_biz = df[df["smart_segment_id"] == 0]
    s = compute_threshold_sweep(df_biz, "avg_num_trxns")
    seg = seg_stats["stats"]
    biz = seg["Business"]
    biz_fp_pct = biz["fp_rate"]

    queries = [
        "Give me a full overview: segment stats then tune the business threshold on weekly transactions",
        "First show alert stats across segments, then analyze FP/FN for business customers using AVG_TRXNS_WEEK",
        "I need both the segment summary and threshold tuning for business accounts using weekly transaction count",
    ]

    response = (
        f"**Segment Overview:**\n"
        f"Business has {biz['alerts']:,} alerts with a {biz_fp_pct}% FP rate "
        f"({biz['fp']:,} of {biz['alerts']:,} alerts are false positives). "
        f"Individual has {seg['Individual']['alerts']:,} alerts "
        f"({seg['Individual']['fp_rate']}% FP rate).\n\n"
        f"**Business Threshold Tuning — AVG_TRXNS_WEEK:**\n"
        f"Sweeping from {s['t_min']} to {s['t_max']} transactions/week (step={s['step']}):\n"
        f"- At minimum ({s['t_min']}): FP={s['fp_at_min']}, FN=0 — all activity flagged\n"
        f"- Optimal threshold: **{s['optimal']['t']}** "
        f"(FP={s['optimal']['fp']}, FN={s['optimal']['fn']}, "
        f"combined={s['optimal']['fp'] + s['optimal']['fn']})\n"
        f"- At maximum ({s['t_max']}): FP~0, FN={s['fn_at_max']} — most activity missed\n\n"
        f"**Recommendation:** Trial threshold={s['optimal']['t']} for business accounts "
        f"and monitor for 30 days before full deployment. "
        f"Document the FP/FN rationale for model risk management."
    )

    examples = []
    for query in queries:
        call1, call2 = cid(), cid()
        examples.append(example([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            # First tool call: segment_stats
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call1,
                    "type": "function",
                    "function": {"name": "segment_stats", "arguments": "{}"},
                }],
            },
            {"role": "tool", "content": seg_stats["result_text"], "tool_call_id": call1},
            # Second tool call: threshold_tuning
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call2,
                    "type": "function",
                    "function": {
                        "name": "threshold_tuning",
                        "arguments": json.dumps({
                            "segment": "Business",
                            "threshold_column": "AVG_TRXNS_WEEK",
                        }),
                    },
                }],
            },
            {"role": "tool", "content": s["result_text"], "tool_call_id": call2},
            {"role": "assistant", "content": response},
        ]))

    return examples


def make_policy_examples() -> list:
    """8 policy/compliance Q&A examples — no tool calls, pure reasoning."""

    pairs = [
        (
            "What is the FATF 40 Recommendations and how does it relate to our AML thresholds?",
            (
                "The FATF 40 Recommendations are the international standard for AML/CFT frameworks, "
                "adopted by 200+ jurisdictions. Key recommendations relevant to threshold tuning:\n\n"
                "- **R.10 (Customer Due Diligence):** Requires ongoing monitoring of transactions — "
                "your alert thresholds directly determine which transactions trigger CDD review.\n"
                "- **R.20 (Suspicious Transaction Reporting):** Obligates filing SARs for suspicious activity. "
                "A threshold set too high creates FNs — missed SARs — a direct regulatory violation.\n"
                "- **R.29 (Financial Intelligence Units):** FIUs expect SAR *quality* over quantity. "
                "Excessive FPs waste investigator capacity and dilute SAR value.\n\n"
                "**Implication:** Threshold optimisation must balance R.20 (catch true positives) "
                "against R.29 (avoid flooding FIUs with noise). The FP/FN sweep tool directly addresses this."
            ),
        ),
        (
            "What does BSA require for SAR filing thresholds?",
            (
                "Under the Bank Secrecy Act (BSA), financial institutions must file a SAR within 30 days "
                "of detecting suspicious activity. Key filing thresholds:\n\n"
                "- **$5,000+** — when the institution knows or suspects the customer is engaged in criminal activity.\n"
                "- **$25,000+** — when the institution suspects criminal activity but cannot identify a suspect.\n"
                "- **Structuring** — any deliberate breakup of transactions below $10,000 to avoid CTR "
                "reporting is a BSA violation regardless of amount.\n\n"
                "**For threshold tuning:** The BSA does not prescribe specific transaction-count or amount "
                "thresholds for *alert generation* — that is your institution's risk-based decision. However, "
                "your thresholds must be defensible to examiners. Document your FP/FN analysis and the "
                "reasoning behind each chosen threshold as part of your model risk management framework."
            ),
        ),
        (
            "What is structuring and how should our thresholds detect it?",
            (
                "Structuring (also called smurfing) is breaking large transactions into smaller amounts "
                "specifically to avoid the $10,000 CTR reporting threshold. It is a federal crime under "
                "31 U.S.C. § 5324.\n\n"
                "**Detection characteristics:**\n"
                "- Multiple transactions just below $10,000 within a short window\n"
                "- High transaction frequency with individually small amounts\n"
                "- Activity spread across multiple accounts or locations\n\n"
                "**Threshold implication:** `AVG_TRXNS_WEEK` is the most relevant column — structurers make "
                "many small transactions. Setting AVG_TRXNS_WEEK too high causes FNs (missed structuring); "
                "too low causes FPs from high-frequency legitimate customers (e.g., cash-intensive small businesses). "
                "Use the `threshold_tuning` tool with `threshold_column=AVG_TRXNS_WEEK` to find the balance."
            ),
        ),
        (
            "What is FinCEN's role and how does it relate to our alert thresholds?",
            (
                "FinCEN (Financial Crimes Enforcement Network) is the U.S. Treasury bureau administering the BSA. "
                "It collects SARs and CTRs and issues AML guidance. Key relevance to this system:\n\n"
                "- **SAR Repository:** FinCEN receives all SARs. Alert thresholds directly determine what "
                "suspicious activity gets reported. FNs = unreported activity; excess FPs = low-quality SAR filing.\n"
                "- **314(a) Requests:** FinCEN can require institutions to search records for specific subjects. "
                "The data includes a 314b column tracking voluntary information-sharing relationships.\n"
                "- **FinCEN Advisories:** Issue typology alerts (e.g., for specific laundering methods) "
                "that should inform threshold adjustments.\n\n"
                "FinCEN examination findings often cite inadequate transaction monitoring thresholds. "
                "Document your FP/FN analysis and threshold selection rationale for model risk management."
            ),
        ),
        (
            "What's the difference between false positives and false negatives in AML?",
            (
                "In AML alert systems:\n\n"
                "**False Positive (FP):** An alert fires for a customer who is *not* engaged in suspicious activity. "
                "The transaction looks suspicious by the threshold, but investigation reveals it's legitimate. "
                "High FPs waste investigator capacity, delay genuine SAR filing, and drive AML operational cost.\n\n"
                "**False Negative (FN):** A customer *is* engaged in suspicious activity, but no alert fires "
                "because the transaction falls below the threshold. This is the more serious risk — a SAR is "
                "never filed, leaving criminal activity undetected and creating regulatory exposure.\n\n"
                "**The trade-off:** Lowering the threshold catches more true positives (fewer FNs) but generates "
                "more FPs. Raising it reduces FP volume but increases missed-SAR risk. Use the `threshold_tuning` "
                "tool to sweep this curve and find the threshold that fits your institution's risk appetite "
                "and investigator capacity."
            ),
        ),
        (
            "What is layering in the context of money laundering?",
            (
                "Layering is the second stage of the classic three-stage money laundering model "
                "(placement → layering → integration). It involves moving illicit funds through a series "
                "of transactions designed to obscure the audit trail and distance money from its criminal origin.\n\n"
                "**Common layering techniques:**\n"
                "- Multiple wire transfers across jurisdictions\n"
                "- Conversion between currencies or asset classes\n"
                "- Use of shell companies or correspondent banks\n"
                "- Back-to-back loans\n\n"
                "**Detection implication:** Layering activity tends to produce high `trxn_amt_monthly` with "
                "moderate transaction counts. The `TRXN_AMT_MONTHLY` threshold is most relevant. "
                "Use `threshold_tuning` with `threshold_column=TRXN_AMT_MONTHLY` to tune detection for each segment."
            ),
        ),
        (
            "How should we document our threshold decisions for a regulatory exam?",
            (
                "Regulators (OCC, FDIC, FinCEN) expect AML model decisions to be documented under a "
                "Model Risk Management (MRM) framework. For threshold decisions:\n\n"
                "1. **Business justification:** Explain why the threshold was chosen — reference the FP/FN "
                "sweep and the crossover/optimal point.\n"
                "2. **Risk appetite alignment:** State the institution's tolerance for missed SARs (FN) vs "
                "investigator overload (FP).\n"
                "3. **Segmentation rationale:** Document why Business and Individual thresholds differ "
                "(different behavioural profiles, FP rates).\n"
                "4. **Validation evidence:** Show backtesting results — did the threshold catch known-bad "
                "actors in historical data?\n"
                "5. **Review cycle:** Thresholds should be reviewed at least annually or after significant "
                "portfolio changes.\n\n"
                "The FP/FN charts and sweep tables generated by this system can be attached directly "
                "to your MRM documentation."
            ),
        ),
        (
            "What is OFAC and how does it affect our customer screening?",
            (
                "OFAC (Office of Foreign Assets Control) administers U.S. economic and trade sanctions. "
                "Financial institutions must screen customers and transactions against OFAC's SDN "
                "(Specially Designated Nationals) list and other sanction lists.\n\n"
                "**Key requirements:**\n"
                "- Block or reject transactions involving sanctioned parties\n"
                "- Report blocked transactions to OFAC within 10 days\n"
                "- Annual OFAC compliance programme review\n\n"
                "**Relevance to this dataset:** The data includes an OFAC column indicating whether "
                "a customer has OFAC hits. This is a high-risk signal that should be incorporated as "
                "a clustering feature — OFAC-flagged customers may warrant a separate, tighter alert "
                "threshold regardless of their transaction profile. "
                "In the cluster analysis, check whether OFAC-flagged customers cluster together or "
                "are distributed across segments."
            ),
        ),
    ]

    return [
        example([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ])
        for q, a in pairs
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate FRAML fine-tuning data")
    parser.add_argument(
        "--data_path",
        default=DEFAULT_CSV,
        help="Path to custs_accts_txns_alerts.csv",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR / "framl_train.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_path} ...")
    df = load_data(args.data_path)
    biz_n = int((df["smart_segment_id"] == 0).sum())
    ind_n = int((df["smart_segment_id"] == 1).sum())
    print(f"  Loaded {len(df):,} rows — Business={biz_n:,}, Individual={ind_n:,}")

    seg_stats = compute_segment_stats(df)

    all_examples: list = []

    print("Generating threshold_tuning examples (24) ...")
    all_examples.extend(make_threshold_examples(df))

    print("Generating segment_stats examples (6) ...")
    all_examples.extend(make_segment_stats_examples(seg_stats))

    print("Generating cluster_analysis examples (15) ...")
    all_examples.extend(make_cluster_analysis_examples())

    print("Generating alerts_distribution examples (4) ...")
    all_examples.extend(make_alerts_distribution_examples(seg_stats))

    print("Generating multi-tool examples (3) ...")
    all_examples.extend(make_multi_tool_examples(df, seg_stats))

    print("Generating policy Q&A examples (8) ...")
    all_examples.extend(make_policy_examples())

    random.seed(SEED)
    random.shuffle(all_examples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(all_examples)} examples to {out_path}")

    # Breakdown
    tool_counts: dict = {}
    no_tool = 0
    for ex in all_examples:
        has_tool = False
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                has_tool = True
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    tool_counts[name] = tool_counts.get(name, 0) + 1
        if not has_tool:
            no_tool += 1

    print("\nTool call distribution:")
    for k, v in sorted(tool_counts.items()):
        print(f"  {k}: {v} calls")
    print(f"  no-tool (policy Q&A): {no_tool} examples")


if __name__ == "__main__":
    main()

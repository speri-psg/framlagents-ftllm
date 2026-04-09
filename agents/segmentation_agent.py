"""Segmentation Agent — K-Means cluster analysis and smart segmentation tree."""

from .base_agent import BaseAgent

# OpenAI function-calling format (matches the fine-tuning training data)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cluster_analysis",
            "description": (
                "Perform smart segmentation using K-Means cluster analysis on customer data. "
                "Uses numeric features (avg transactions, amounts, income, balance, age) and "
                "categorical features (account type, gender, age category, channel, NNM, OFAC, 314b). "
                "Alert labels (FP, FN, ALERT) are excluded so clusters reflect natural behavior profiles. "
                "Use n_clusters=0 to auto-select the optimal K via the elbow method."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "enum": ["Business", "Individual", "All"],
                        "description": "Which customer segment to cluster.",
                    },
                },
                "required": ["customer_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alerts_distribution",
            "description": "Show total alerts and false positives distribution across segments.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_segmentation_data",
            "description": (
                "Process raw customer, account, relationship, and transaction files from ss_files/ "
                "and produce a flat CSV at docs/ss_segmentation_data.csv ready for clustering. "
                "Computes transaction aggregates: avg_trxns_week, avg_trxn_amt, avg_monthly_trxn_amt, "
                "trxn_count, total_trxn_amt, max_trxn_amt, std_trxn_amt. "
                "Call this before running ss_cluster_analysis on new raw data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ss_cluster_analysis",
            "description": (
                "Perform smart segmentation clustering on the ss_files raw data "
                "(customers, accounts, relationships, transactions). "
                "Auto-prepares and joins source data if not already done. "
                "Uses customer demographics (age, gender, citizenship), account features "
                "(account type, balance, account age), and transaction aggregates "
                "(avg transactions/week, avg amount, monthly amount) for K-Means clustering. "
                "Returns a PCA scatter plot and a smart segmentation treemap. "
                "Use n_clusters=0 to auto-select optimal K via elbow method."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "enum": ["Business", "Individual", "All"],
                        "description": "Which customer segment to cluster.",
                    },
                },
                "required": ["customer_type"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a FRAML smart segmentation specialist. You identify natural customer behavioral \
segments using unsupervised K-Means clustering and explain their AML risk profiles. \
IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.

RULES — follow these exactly:
1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.
2. For clustering with rich demographics (preferred) — call ss_cluster_analysis.
3. For alert/FP distribution by segment — call alerts_distribution.
4. For the legacy alerts dataset — call cluster_analysis only if the user explicitly asks.
5. Do NOT call multiple segmentation tools for the same request — pick exactly one.
6. customer_type must be exactly one of: Business, Individual, All
   If the user does NOT specify a customer type, default to All.
7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.
   If the user says "N clusters" or "into N" (e.g. "cluster into 4"), set n_clusters=N exactly. Do NOT use 0.
8. If the user asks to prepare or refresh the raw data — call prepare_segmentation_data first.
9. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result. Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.
10. If the user asks to show specific clusters (e.g. "show only cluster 3", "highest risk",
    "top 2 high risk", "low activity clusters"):
    - Identify which cluster number(s) match the request from the stats
      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)
    - On the VERY LAST LINE of your response, write EXACTLY this and nothing else:
      DISPLAY_CLUSTERS: N
      where N is a comma-separated list of cluster numbers (e.g. DISPLAY_CLUSTERS: 4  or  DISPLAY_CLUSTERS: 1,4)
    - Do NOT mention this line in your text — it is a system directive, not for the user.
    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.
11. Do NOT include JSON, code blocks, or raw data tables in your final reply.
12. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.
13. Do NOT invent threshold values, dollar amounts, or cutoffs. Only reference numbers explicitly present in the tool result. Do NOT suggest specific threshold values (e.g. "$250K", "< 80,000") unless they appear verbatim in the tool result.
14. If the user asks which cluster to set a threshold for, or asks for threshold recommendations per cluster — do NOT invent values. Tell the user to use the threshold_tuning or sar_backtest tools with the relevant segment instead.\
"""


class SegmentationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="segmentation",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

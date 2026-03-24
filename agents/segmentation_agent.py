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
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters (2-8). Use 0 to auto-select. Default 4.",
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
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters (2-8). Use 0 to auto-select. Default 4.",
                    },
                },
                "required": ["customer_type"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a FRAML smart segmentation specialist. You identify natural customer behavioral \
segments using unsupervised K-Means clustering and explain their AML risk profiles.

RULES — follow these exactly:
1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.
2. For clustering with rich demographics (preferred) — call ss_cluster_analysis.
3. For alert/FP distribution by segment — call alerts_distribution.
4. For the legacy alerts dataset — call cluster_analysis only if the user explicitly asks.
5. Do NOT call multiple segmentation tools for the same request — pick exactly one.
6. customer_type must be exactly one of: Business, Individual, All
7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.
8. If the user asks to prepare or refresh the raw data — call prepare_segmentation_data first.
9. After receiving tool results, give a 3-5 sentence interpretation:
   - How many clusters were found and their behavioral profiles
   - Which cluster(s) are highest AML risk and why
   - How segment-specific thresholds could improve alert quality
10. Do NOT include JSON, code blocks, or raw data tables in your final reply.\
"""


class SegmentationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="segmentation",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

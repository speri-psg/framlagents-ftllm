"""
V44 training examples (2026-05-13).

Targets:

  AA_V44_1–5  Gap 1 reinforcement: 0-char response after list_rules.
              Fix: tool-call content is even more minimal; final response
              starts IMMEDIATELY with the PRE-COMPUTED block (no header
              preamble) so the model must generate content tokens before
              the analysis.  Two new datasets (F, G) prevent memorisation.

  AC_V44_1–6  Gap 3 reinforcement: "Age" / "oldest" / "youngest" phrasing
              triggered re-clustering in aria-v7 testing.  These examples
              use every synonym the user might reach for:
                - "highest age", "oldest customers", "youngest segment"
                - "average age by cluster", "rank by age"
                - "account age" vs bare "age"
              All 6 answered directly from [PREVIOUS CLUSTERING RESULT].

Combined: aria_train_combined_v43_full.jsonl (base 1031) + 11 new = 1042
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V43_BASE_PATH = DATA_DIR / "aria_train_combined_v43_full.jsonl"
V44_ONLY_PATH = DATA_DIR / "aria_train_v44.jsonl"
V44_FULL_PATH = DATA_DIR / "aria_train_combined_v44_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM           # noqa: E402
from write_v42 import (                                                # noqa: E402
    tc, prev_context, _CLUSTER_STATS,
)
from write_v43 import (                                                # noqa: E402
    _LIST_D, PC_LIST_D,
    _LIST_E, PC_LIST_E,
)

examples = []

# ═══════════════════════════════════════════════════════════════════════════
# Dataset F — Velocity Multiple wins precision at 30.0%
# ═══════════════════════════════════════════════════════════════════════════

_LIST_F = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=335, SAR=54, FP=281, precision=16.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=205, SAR=37, FP=168, precision=18.0%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=395, SAR=63, FP=332, precision=15.9%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=260, SAR=50, FP=210, precision=19.2%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=370, SAR=53, FP=317, precision=14.3%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=285, SAR=47, FP=238, precision=16.5%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=245, SAR=41, FP=204, precision=16.7%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=450, SAR=61, FP=389, precision=13.6%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=200, SAR=36, FP=164, precision=18.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=190, SAR=33, FP=157, precision=17.4%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=165, SAR=38, FP=127, precision=23.0%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=155, SAR=26, FP=129, precision=16.8%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=180, SAR=54, FP=126, precision=30.0%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=175, SAR=36, FP=139, precision=20.6%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=115, SAR=27, FP=88, precision=23.5%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=100, SAR=20, FP=80, precision=20.0%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_F = "Tool result for list_rules:\n" + _LIST_F

# ═══════════════════════════════════════════════════════════════════════════
# Dataset G — Round-trip wins precision at 33.3%
# ═══════════════════════════════════════════════════════════════════════════

_LIST_G = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=330, SAR=53, FP=277, precision=16.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=200, SAR=36, FP=164, precision=18.0%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=400, SAR=60, FP=340, precision=15.0%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=255, SAR=48, FP=207, precision=18.8%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=375, SAR=51, FP=324, precision=13.6%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=280, SAR=46, FP=234, precision=16.4%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=240, SAR=39, FP=201, precision=16.3%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=455, SAR=59, FP=396, precision=13.0%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=195, SAR=35, FP=160, precision=17.9%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=188, SAR=32, FP=156, precision=17.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=160, SAR=37, FP=123, precision=23.1%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=150, SAR=25, FP=125, precision=16.7%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=178, SAR=34, FP=144, precision=19.1%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=170, SAR=33, FP=137, precision=19.4%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=120, SAR=40, FP=80, precision=33.3%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=98, SAR=19, FP=79, precision=19.4%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_G = "Tool result for list_rules:\n" + _LIST_G

# ═══════════════════════════════════════════════════════════════════════════
# AA_V44_1–5  Gap 1 reinforcement
#
# Key change from V43:
#   1. tool-call content is even shorter — no detail about what we are
#      looking for; the model must not "pre-resolve" in its generation
#   2. Final response starts IMMEDIATELY with the verbatim PRE-COMPUTED
#      block so the first tokens the model generates are concrete data,
#      not an empty string
# ═══════════════════════════════════════════════════════════════════════════

# AA_V44_1: "highest precision" — Dataset F (Velocity Multiple wins, 30.0%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v44_1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v44_1", "content": PC_LIST_F},
    {"role": "assistant", "content": (
        _LIST_F + "\n\n"
        "**Top rules by precision:**\n"
        "1. **Velocity Multiple** — 30.0% (54 SAR, 126 FP)\n"
        "2. **Risky International Transfer** — 23.0% (38 SAR, 127 FP)\n"
        "3. **Round-trip** — 23.5% (27 SAR, 88 FP)\n\n"
        "Velocity Multiple leads with 30.0% precision — nearly double the system average."
    )},
]})

# AA_V44_2: "lowest false positive rate" — Dataset G (Round-trip wins, 33.3%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the lowest false positive rate?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v44_2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v44_2", "content": PC_LIST_G},
    {"role": "assistant", "content": (
        _LIST_G + "\n\n"
        "**Lowest false positive rate (highest precision):**\n"
        "1. **Round-trip** — 33.3% precision (40 SAR, 80 FP)\n"
        "2. **Risky International Transfer** — 23.1% (37 SAR, 123 FP)\n"
        "3. **Velocity Single** — 18.8% (48 SAR, 207 FP)\n\n"
        "Round-trip has the lowest false positive rate at 33.3% — "
        "1 in 3 alerts it generates is a confirmed SAR customer."
    )},
]})

# AA_V44_3: "top 3 by precision" — Dataset F (Velocity Multiple wins, 30.0%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me the top 3 rules by precision."},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v44_3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v44_3", "content": PC_LIST_F},
    {"role": "assistant", "content": (
        _LIST_F + "\n\n"
        "**Top 3 rules by precision:**\n"
        "1. **Velocity Multiple** — 30.0% (54 SAR, 126 FP)\n"
        "2. **Round-trip** — 23.5% (27 SAR, 88 FP)\n"
        "3. **Risky International Transfer** — 23.0% (38 SAR, 127 FP)"
    )},
]})

# AA_V44_4: "best hit rate" — Dataset G (Round-trip wins, 33.3%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the best hit rate for catching suspicious activity?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v44_4", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v44_4", "content": PC_LIST_G},
    {"role": "assistant", "content": (
        _LIST_G + "\n\n"
        "**Best hit rate (precision) for catching suspicious activity:**\n"
        "1. **Round-trip** — 33.3% (40 SAR, 80 FP) — highest precision in this dataset\n"
        "2. **Risky International Transfer** — 23.1% (37 SAR, 123 FP)\n"
        "3. **Velocity Single** — 18.8% (48 SAR, 207 FP)\n\n"
        "Round-trip's 33.3% precision means 1 in 3 alerts yields a confirmed SAR. "
        "CTR Client is the weakest at 13.0% (59 SAR, 396 FP)."
    )},
]})

# AA_V44_5: "most efficient rule" — Dataset E (Burst in Originator wins, 34.5%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule is the most efficient at identifying true SARs?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v44_5", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v44_5", "content": PC_LIST_E},
    {"role": "assistant", "content": (
        _LIST_E + "\n\n"
        "**Most efficient rule (highest precision):**\n"
        "1. **Burst in Originator Activity** — 34.5% (50 SAR, 95 FP)\n"
        "2. **Round-trip** — 24.3% (28 SAR, 87 FP)\n"
        "3. **Risky International Transfer** — 22.6% (35 SAR, 120 FP)\n\n"
        "Burst in Originator Activity is the most efficient at 34.5% — "
        "nearly 2.5× more precise than CTR Client (14.2%), the least efficient rule."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AC_V44_1–6  Gap 3 reinforcement — "Age" phrasing
#
# All six answered directly from [PREVIOUS CLUSTERING RESULT].
# No tool call. Field: "Account Age (years)" from _CLUSTER_STATS.
#
# Cluster 1: 4.1 yrs  |  Cluster 2: 7.3 yrs
# Cluster 3: 5.8 yrs  |  Cluster 4: 9.2 yrs
# ═══════════════════════════════════════════════════════════════════════════

# AC_V44_1: bare "age" — highest
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest age?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest average account age at **9.2 years**, "
        "followed by Cluster 2 (**7.3 years**), Cluster 3 (**5.8 years**), and Cluster 1 (**4.1 years**)."
    )},
]})

# AC_V44_2: "oldest customers"
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the oldest customers?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the oldest customers with an average account age "
        "of **9.2 years**, compared to Cluster 2 (**7.3 years**), Cluster 3 (**5.8 years**), "
        "and Cluster 1 (**4.1 years**)."
    )},
]})

# AC_V44_3: "youngest segment"
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich segment has the youngest account history?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the shortest account history at **4.1 years** — "
        "the youngest segment. Cluster 3 is next at **5.8 years**, then Cluster 2 at **7.3 years**, "
        "and Cluster 4 at **9.2 years**."
    )},
]})

# AC_V44_4: "rank clusters by account age"
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nCan you rank the clusters by account age?"},
    {"role": "assistant", "content": (
        "From the clustering result above, ranked by average account age (highest to lowest):\n\n"
        "1. **Cluster 4** — 9.2 years\n"
        "2. **Cluster 2** — 7.3 years\n"
        "3. **Cluster 3** — 5.8 years\n"
        "4. **Cluster 1** — 4.1 years"
    )},
]})

# AC_V44_5: "average account age for each cluster"
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhat is the average account age for each cluster?"},
    {"role": "assistant", "content": (
        "From the clustering result above, the average account age (years) for each cluster:\n\n"
        "- **Cluster 1** — 4.1 years (n=121)\n"
        "- **Cluster 2** — 7.3 years (n=98)\n"
        "- **Cluster 3** — 5.8 years (n=156)\n"
        "- **Cluster 4** — 9.2 years (n=110)"
    )},
]})

# AC_V44_6: "show me account age by segment" — minimal phrasing
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nShow me account age by segment."},
    {"role": "assistant", "content": (
        "From the clustering result above:\n\n"
        "| Cluster | Avg Account Age (years) | Size |\n"
        "|---------|------------------------|------|\n"
        "| 1       | 4.1                    | 121  |\n"
        "| 2       | 7.3                    | 98   |\n"
        "| 3       | 5.8                    | 156  |\n"
        "| 4       | 9.2                    | 110  |\n\n"
        "Cluster 4 has the most tenured customers; Cluster 1 the newest."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AC_V44_MT1–MT3  Multi-turn: 3-turn conversations with prior tool call
#
# Pattern that failed in aria-v7 testing:
#   Turn 1: ds_cluster_analysis called → result → model summarises clusters
#   Turn 2: follow-up Q answered from [PREVIOUS CLUSTERING RESULT]
#   Turn 3: "What about cluster X?" → model re-clustered ❌
#
# Fix: show the model that even when cluster_analysis already ran in this
# conversation, a follow-up about a specific cluster MUST be answered from
# the stats block — never by calling the tool again.
#
# _CLUSTER_STATS values (Business, 4 clusters):
#   C1 n=121: weekly=3.2, amt=$1,205, vol=$4,820,  inc=$72k,  bal=$18.5k, age=4.1yr
#   C2 n=98:  weekly=8.7, amt=$4,310, vol=$17,240, inc=$145k, bal=$62.3k, age=7.3yr
#   C3 n=156: weekly=14.2,amt=$2,914, vol=$41,380, inc=$98k,  bal=$31.1k, age=5.8yr
#   C4 n=110: weekly=22.6,amt=$8,750, vol=$197,500,inc=$312k, bal=$148k,  age=9.2yr
# ═══════════════════════════════════════════════════════════════════════════

_PC_CLUSTER_ANALYSIS = "Tool result for ds_cluster_analysis:\n" + _CLUSTER_STATS

_T1_CLUSTER_RESPONSE = (
    "Clustering complete for Business customers.\n\n"
    + _CLUSTER_STATS + "\n\n"
    "**Summary:** Cluster 4 (n=110) is the highest-activity, highest-value segment — "
    "22.6 weekly transactions and $197,500 monthly volume. "
    "Cluster 1 (n=121) is the lowest-activity segment with the newest accounts (4.1 yrs). "
    "Clusters 2 and 3 represent intermediate activity levels."
)

# MT1: T1 clusters → T2 highest monthly volume → T3 "What about cluster 4?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    # Turn 1
    {"role": "user", "content": "Can you cluster the Business customers by behavior?"},
    {"role": "assistant", "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v44_mt1", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v44_mt1", "content": _PC_CLUSTER_ANALYSIS},
    {"role": "assistant", "content": _T1_CLUSTER_RESPONSE},
    # Turn 2
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest monthly volume?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest monthly volume at **$197,500**, "
        "followed by Cluster 3 (**$41,380**), Cluster 2 (**$17,240**), and Cluster 1 (**$4,820**)."
    )},
    # Turn 3
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhat about cluster 4? Tell me more about it."},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 (n=110, 22.6% of customers) is the highest-activity segment:\n\n"
        "- **Avg Weekly Transactions:** 22.6 — 7× higher than Cluster 1\n"
        "- **Avg Txn Amount:** $8,750 — largest per-transaction size of any cluster\n"
        "- **Monthly Txn Volume:** $197,500 — by far the highest throughput\n"
        "- **Income:** $312,000 — wealthiest segment\n"
        "- **Current Balance:** $148,000\n"
        "- **Account Age:** 9.2 years — most established accounts\n\n"
        "Cluster 4 is the highest-risk segment for AML monitoring: high transaction velocity, "
        "large transaction sizes, and substantial monthly throughput make it the priority segment for threshold review."
    )},
]})

# MT2: T1 clusters → T2 youngest customers → T3 "What about cluster 2?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    # Turn 1
    {"role": "user", "content": "Run segmentation on Business customers."},
    {"role": "assistant", "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v44_mt2", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v44_mt2", "content": _PC_CLUSTER_ANALYSIS},
    {"role": "assistant", "content": _T1_CLUSTER_RESPONSE},
    # Turn 2
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the youngest customers?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the youngest customers with an average account age "
        "of **4.1 years**, followed by Cluster 3 (**5.8 years**), Cluster 2 (**7.3 years**), "
        "and Cluster 4 (**9.2 years**)."
    )},
    # Turn 3
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhat about cluster 2?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 2 (n=98) is a mid-tier segment:\n\n"
        "- **Avg Weekly Transactions:** 8.7\n"
        "- **Avg Txn Amount:** $4,310\n"
        "- **Monthly Txn Volume:** $17,240\n"
        "- **Income:** $145,000\n"
        "- **Current Balance:** $62,300\n"
        "- **Account Age:** 7.3 years\n\n"
        "Cluster 2 is more active and wealthier than Cluster 1, but considerably below "
        "Cluster 3 in transaction frequency and below Cluster 4 on every metric."
    )},
]})

# MT3: T1 clusters → T2 largest cluster → T3 "Tell me about cluster 1."
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    # Turn 1
    {"role": "user", "content": "Segment the Business customers for me."},
    {"role": "assistant", "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v44_mt3", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v44_mt3", "content": _PC_CLUSTER_ANALYSIS},
    {"role": "assistant", "content": _T1_CLUSTER_RESPONSE},
    # Turn 2
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster is the largest?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 3 is the largest segment with **156 customers**, "
        "followed by Cluster 1 (**121**), Cluster 4 (**110**), and Cluster 2 (**98**)."
    )},
    # Turn 3
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nTell me about cluster 1."},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 (n=121) is the lowest-activity segment:\n\n"
        "- **Avg Weekly Transactions:** 3.2 — the lowest of all clusters\n"
        "- **Avg Txn Amount:** $1,205\n"
        "- **Monthly Txn Volume:** $4,820\n"
        "- **Income:** $72,000 — lowest income segment\n"
        "- **Current Balance:** $18,500\n"
        "- **Account Age:** 4.1 years — newest accounts\n\n"
        "Cluster 1 represents low-activity, lower-income customers with newer account relationships. "
        "Lower transaction velocity makes it the lowest-priority segment for AML escalation."
    )},
]})

# ---------------------------------------------------------------------------
# Combine V43 base + V44 and write
# ---------------------------------------------------------------------------

def main():
    with open(V44_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V44] V44-only: {V44_ONLY_PATH.name} ({len(examples)} examples)")

    if V43_BASE_PATH.exists():
        v43_base = []
        with open(V43_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v43_base.append(json.loads(line))
        print(f"[V44] Loaded {len(v43_base)} base examples from {V43_BASE_PATH.name}")
        all_examples = v43_base + examples
        with open(V44_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V44] Combined: {V44_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V44] WARNING: V43 base not found at {V43_BASE_PATH}")


if __name__ == "__main__":
    main()

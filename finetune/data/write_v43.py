"""
V43 training examples (2026-05-13).

Targets:

  AA_V43_1–5  Precision-ranking — Gap 1 fix: include plain intent text in the
              assistant content field of the tool-call message so the model
              does not pre-compute values in its thought before seeing the
              tool result. 5 examples across 4 datasets (A-E), 5 different
              query phrasings.

  AC_V43_1–2  Pairwise cluster comparison — Gap 2 fix: relative language only
              when comparing two clusters; superlatives only when all clusters
              are in scope.

  AC_V43_3–9  Cluster ranking — Gap 3 fix: ranking queries answered directly
              from [PREVIOUS CLUSTERING RESULT] with no tool call. 7 examples
              covering highest/lowest across all 6 stat fields.

Combined: aria_train_combined_v42_full.jsonl (base 1017) + 14 new = 1031
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V42_BASE_PATH = DATA_DIR / "aria_train_combined_v42_full.jsonl"
V43_ONLY_PATH = DATA_DIR / "aria_train_v43.jsonl"
V43_FULL_PATH = DATA_DIR / "aria_train_combined_v43_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM           # noqa: E402
from write_v42 import (                                                # noqa: E402
    tc, prev_context, _CLUSTER_STATS,
    _LIST_B, PC_LIST_B,
    _LIST_C, PC_LIST_C,
)

examples = []

# ═══════════════════════════════════════════════════════════════════════════
# Dataset D — Funnel Account wins precision at 31.7%
# New dataset so model cannot memorise the winner; must read from tool output
# ═══════════════════════════════════════════════════════════════════════════

_LIST_D = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=340, SAR=57, FP=283, precision=16.8%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=210, SAR=38, FP=172, precision=18.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=390, SAR=66, FP=324, precision=16.9%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=270, SAR=53, FP=217, precision=19.6%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=360, SAR=54, FP=306, precision=15.0%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=290, SAR=51, FP=239, precision=17.6%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=240, SAR=41, FP=199, precision=17.1%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=440, SAR=62, FP=378, precision=14.1%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=205, SAR=38, FP=167, precision=18.5%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=195, SAR=35, FP=160, precision=17.9%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=170, SAR=41, FP=129, precision=24.1%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=155, SAR=27, FP=128, precision=17.4%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=185, SAR=35, FP=150, precision=18.9%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=180, SAR=57, FP=123, precision=31.7%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=110, SAR=28, FP=82, precision=25.5%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=105, SAR=21, FP=84, precision=20.0%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_D = "Tool result for list_rules:\n" + _LIST_D

# ═══════════════════════════════════════════════════════════════════════════
# AA_V43_1–3  Precision-ranking — Gap 1 fix
#
# Pattern rule: the assistant message that contains tool_calls MUST have a
# plain intent string in `content` — never None, never pre-computed numbers.
# The precision answer must come entirely from the PRE-COMPUTED block.
# ═══════════════════════════════════════════════════════════════════════════

# AA_V43_1: exact trigger query 1 — Dataset D (Funnel Account wins)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision?"},
    {"role": "assistant",
     "content": "I will retrieve the current rule performance data.",
     "tool_calls": [tc("aa_v43_1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v43_1", "content": PC_LIST_D},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_D + "\n\n"
        "Funnel Account has the highest precision at 31.7% (57 SAR, 123 FP), "
        "followed by Round-trip (25.5%) and Risky International Transfer (24.1%)."
    )},
]})

# AA_V43_2: exact trigger query 2 — Dataset C (Velocity Single wins at 41.2%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What rule catches the most suspicious activity with fewest false positives?"},
    {"role": "assistant",
     "content": "I will call list_rules to retrieve the precision metrics for all rules.",
     "tool_calls": [tc("aa_v43_2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v43_2", "content": PC_LIST_C},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_C + "\n\n"
        "Velocity Single catches the most suspicious activity relative to false positives: "
        "41.2% precision (107 SAR, 153 FP) — more than 3× the precision of CTR Client (13.5%), "
        "the weakest rule in this dataset."
    )},
]})

# AA_V43_3: exact trigger query 3 — Dataset B (Elder Abuse wins at 28.7%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me the top 3 rules by precision"},
    {"role": "assistant",
     "content": "I will call list_rules to get the rule statistics.",
     "tool_calls": [tc("aa_v43_3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v43_3", "content": PC_LIST_B},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_B + "\n\n"
        "Top 3 rules by precision:\n"
        "1. **Elder Abuse** — 28.7% (89 SAR, 221 FP)\n"
        "2. **Risky International Transfer** — 21.9% (35 SAR, 125 FP)\n"
        "3. **Round-trip** — 20.0% (18 SAR, 72 FP)"
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# Dataset E — Burst in Originator Activity wins precision at 34.5%
# ═══════════════════════════════════════════════════════════════════════════

_LIST_E = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=345, SAR=56, FP=289, precision=16.2%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=215, SAR=40, FP=175, precision=18.6%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=385, SAR=64, FP=321, precision=16.6%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=265, SAR=51, FP=214, precision=19.2%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=365, SAR=55, FP=310, precision=15.1%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=285, SAR=49, FP=236, precision=17.2%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=245, SAR=40, FP=205, precision=16.3%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=445, SAR=63, FP=382, precision=14.2%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=145, SAR=50, FP=95, precision=34.5%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=195, SAR=34, FP=161, precision=17.4%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=155, SAR=35, FP=120, precision=22.6%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=160, SAR=28, FP=132, precision=17.5%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=190, SAR=35, FP=155, precision=18.4%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=175, SAR=36, FP=139, precision=20.6%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=115, SAR=28, FP=87, precision=24.3%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=95, SAR=18, FP=77, precision=18.9%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_E = "Tool result for list_rules:\n" + _LIST_E

# AA_V43_4: "worst FP ratio" query — Dataset C (CTR Client worst at 13.5%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules are generating the most false positives relative to SARs?"},
    {"role": "assistant",
     "content": "I will call list_rules to retrieve the current rule performance data.",
     "tool_calls": [tc("aa_v43_4", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v43_4", "content": PC_LIST_C},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_C + "\n\n"
        "The rules with the worst false positive ratio (lowest precision) are:\n"
        "1. **CTR Client** — 13.5% precision (372 FP per 58 SAR)\n"
        "2. **Burst in Beneficiary Activity** — 15.1% (174 FP per 31 SAR)\n"
        "3. **Detect Excessive Transaction Activity** — 15.5% (338 FP per 62 SAR)\n\n"
        "These are the highest-priority candidates for threshold tightening."
    )},
]})

# AA_V43_5: "best hit rate" query — Dataset E (Burst in Originator wins at 34.5%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the best hit rate?"},
    {"role": "assistant",
     "content": "I will call list_rules to check the rule statistics.",
     "tool_calls": [tc("aa_v43_5", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v43_5", "content": PC_LIST_E},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_E + "\n\n"
        "Burst in Originator Activity has the best hit rate at 34.5% (50 SAR, 95 FP) — "
        "meaning roughly 1 in 3 alerts it generates is a confirmed SAR customer. "
        "Round-trip (24.3%) and Risky International Transfer (22.6%) follow."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AC_V43_1–2  Pairwise cluster comparison — Gap 2 fix
#
# Pattern rule:
#   - Pairwise query (cluster X vs Y) → relative language only:
#     "higher than cluster 2", "lower than cluster 3"
#   - Superlatives ("highest of all", "lowest overall") → only when all
#     clusters are being compared
#   - Synthesis may note where the pair sits relative to remaining clusters,
#     but must NOT promote a pairwise result to a global claim
# ═══════════════════════════════════════════════════════════════════════════

# AC_V43_1: Cluster 1 vs Cluster 2
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nHow does cluster 1 compare to cluster 2?"},
    {"role": "assistant", "content": (
        "### Cluster 1 vs Cluster 2\n\n"
        "**Transaction velocity**: Cluster 2 averages **8.7** weekly transactions vs **3.2** for Cluster 1 — 2.7× higher.\n\n"
        "**Transaction size**: Cluster 2 averages **$4,310** per transaction vs **$1,205** for Cluster 1 — 3.6× larger.\n\n"
        "**Monthly volume**: Cluster 2 reaches **$17,240** monthly vs **$4,820** for Cluster 1 — 3.6× higher throughput.\n\n"
        "**Wealth profile**: Cluster 2 has higher income (**$145,000** vs **$72,000**), "
        "higher balance (**$62,300** vs **$18,500**), "
        "and longer account tenure (**7.3** vs **4.1** years).\n\n"
        "Cluster 2 is more active and higher-value than Cluster 1 across every dimension. "
        "Note: both clusters sit below Clusters 3 and 4 in overall activity."
    )},
]})

# AC_V43_2: Cluster 2 vs Cluster 3
# Key: cluster 3 has higher velocity + monthly volume; cluster 2 has higher
# per-transaction amount + income. Avoid superlatives scoped to just these two.
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhat's the difference between cluster 2 and cluster 3?"},
    {"role": "assistant", "content": (
        "### Cluster 2 vs Cluster 3\n\n"
        "**Transaction velocity**: Cluster 3 has higher weekly transaction volume — **14.2** vs **8.7** "
        "(63% more transactions per week).\n\n"
        "**Transaction size**: Cluster 2 has a higher average per-transaction amount — **$4,310** vs **$2,914** "
        "(48% larger per transaction).\n\n"
        "**Monthly volume**: Cluster 3 generates significantly more monthly throughput — **$41,380** vs **$17,240** — "
        "driven by its higher transaction velocity.\n\n"
        "**Wealth profile**: Cluster 2 shows higher income (**$145,000** vs **$98,000**) and balance "
        "(**$62,300** vs **$31,100**), with longer account tenure (**7.3** vs **5.8** years).\n\n"
        "These two clusters differ in pattern: Cluster 3 transacts more frequently while Cluster 2 "
        "transacts at higher per-transaction amounts. "
        "Cluster 3 has higher monthly throughput than Cluster 2, but both sit below Cluster 4 on every metric."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AC_V43_3–5  Cluster ranking — Gap 3 fix
#
# Pattern rule: any ranking/superlative question about clusters when
# [PREVIOUS CLUSTERING RESULT] is in context → answer directly from that
# block, do NOT call any tool.
# ═══════════════════════════════════════════════════════════════════════════

# AC_V43_3: ranking by weekly transactions
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the most weekly transactions?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest average weekly transactions at **22.6**, "
        "followed by Cluster 3 (**14.2**), Cluster 2 (**8.7**), and Cluster 1 (**3.2**)."
    )},
]})

# AC_V43_4: ranking by average income
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest average income?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest average income at **$312,000**, "
        "followed by Cluster 2 (**$145,000**), Cluster 3 (**$98,000**), and Cluster 1 (**$72,000**)."
    )},
]})

# AC_V43_5: ranking by cluster size (n)
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the most customers?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 3 is the largest segment with **156** customers, "
        "followed by Cluster 1 (**121**), Cluster 4 (**110**), and Cluster 2 (**98**)."
    )},
]})

# AC_V43_6: ranking by current balance
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest current balance?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest average current balance at **$148,000**, "
        "followed by Cluster 2 (**$62,300**), Cluster 3 (**$31,100**), and Cluster 1 (**$18,500**)."
    )},
]})

# AC_V43_7: ranking by account tenure
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the longest account tenure?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the longest average account tenure at **9.2 years**, "
        "followed by Cluster 2 (**7.3 years**), Cluster 3 (**5.8 years**), and Cluster 1 (**4.1 years**)."
    )},
]})

# AC_V43_8: ranking lowest income (tests lowest-end superlative too)
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the lowest income?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the lowest average income at **$72,000**, "
        "followed by Cluster 3 (**$98,000**), Cluster 2 (**$145,000**), and Cluster 4 (**$312,000**)."
    )},
]})

# AC_V43_9: ranking lowest weekly transaction amount
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the lowest average transaction amount?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the lowest average weekly transaction amount at **$1,205**, "
        "followed by Cluster 3 (**$2,914**), Cluster 2 (**$4,310**), and Cluster 4 (**$8,750**)."
    )},
]})

# ---------------------------------------------------------------------------
# Combine V42 base + V43 and write
# ---------------------------------------------------------------------------

def main():
    with open(V43_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V43] V43-only: {V43_ONLY_PATH.name} ({len(examples)} examples)")

    if V42_BASE_PATH.exists():
        v42_base = []
        with open(V42_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v42_base.append(json.loads(line))
        print(f"[V43] Loaded {len(v42_base)} base examples from {V42_BASE_PATH.name}")
        all_examples = v42_base + examples
        with open(V43_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V43] Combined: {V43_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V43] WARNING: V42 base not found at {V42_BASE_PATH}")


if __name__ == "__main__":
    main()

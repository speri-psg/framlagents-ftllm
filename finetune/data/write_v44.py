"""
V44 training examples (2026-05-13).

Targets:

  AA_V44_1-5   Gap 1 reinforcement: 0-char response after list_rules.
               Fix: minimal tool-call content; final response starts
               IMMEDIATELY with the PRE-COMPUTED block. Datasets F + G.

  AC_V44_1-6   Gap 3 reinforcement: "Age" / "oldest" / "youngest" phrasing
               triggered re-clustering in aria-v7. Six synonyms for Account
               Age answered directly from [PREVIOUS CLUSTERING RESULT].

  AC_V44_MT1-3 Gap 3 multi-turn: 3-turn examples where ds_cluster_analysis
               already ran in Turn 1. Turn 3 follow-up must read from the
               stats block, NOT re-cluster.

  AT_V44_1-6   threshold_tuning regression fix.
               All 92 old threshold_tuning examples use a stale system prompt
               AND raw tool result format. Current tool returns
               === PRE-COMPUTED ANALYSIS === blocks.
               Filter: _has_threshold_tuning removes 92 stale examples.

  ARS_V44_1-6  rule_sar_backtest regression fix.
               All 211 old examples use wrong block header
               (PRE-COMPUTED RULE SAR BACKTEST / END PRE-COMPUTED).
               Current tool returns PRE-COMPUTED RULE SWEEP / END RULE SWEEP.
               Filter: _has_rule_sar_backtest removes 211 stale examples.
               Format: ### header -> RULE SWEEP verbatim -> ONE insight sentence.

  ASS_V44_1-6  segment_stats format + routing fix.
               All 12 old examples use raw "Business: total=X" tool result
               format -> markdown table response. Current tool returns
               === PRE-COMPUTED SEGMENT STATS === block.
               Also adds "segment breakdown" phrasings to prevent misrouting
               to cluster_analysis (12 seg_stats vs 117 clustering examples).
               Filter: _has_segment_stats removes 12 stale examples.

  ARL_V44_1-3  Rule 23: numbered rule references ("rule 5", "rule 12") ->
               call list_rules -> state no such rule exists -> list all 16.

Filters applied in main():
  _has_threshold_tuning  : removes  92 stale examples
  _has_rule_sar_backtest : removes 211 stale examples
  _has_segment_stats     : removes  12 stale examples
  Total filtered: 315

Base:   aria_train_combined_v43_full.jsonl (1031)
After:  1031 - 315 = 716 base + 35 new V44 = 751 total
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

# ===========================================================================
# Dataset F — Velocity Multiple wins precision at 30.0%
# ===========================================================================

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

# ===========================================================================
# Dataset G — Round-trip wins precision at 33.3%
# ===========================================================================

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

# ===========================================================================
# PRE-COMPUTED RULE SWEEP blocks — real numbers from lambda_rule_analysis.py
# ===========================================================================

_RS_ELDER_ABUSE = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Current condition: Age >= 60 AND 14-day outgoing >= $5K AND >= 3 std dev above 90-day mean\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3)\n"
    "Current value: 3.0\n"
    "Labeled population: 400 customers (TP+FN pool=59 SAR, FP+TN pool=341 non-SAR, precision=14.8%)\n"
    "\n"
    "At the lowest value (0.00): TP=59, FP=341, FN=0, TN=0 (TP rate=100.0%, precision=14.8%).\n"
    "At current condition (3.00): TP=59, FP=341, FN=0, TN=0 (TP rate=100.0%, precision=14.8%).\n"
    "To keep TP rate >=90%: z_threshold <= 3.00 => TP=59, FP=341, FN=0, TN=0, precision=14.8%.\n"
    "To keep TP rate >=50%: z_threshold <= 6.00 => TP=33, FP=205, FN=26, TN=136, precision=13.9%.\n"
    "At the highest value (10.00): TP=1, FP=0, FN=58, TN=341, precision=100.0%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_ELDER = "Tool result for rule_sar_backtest:\n" + _RS_ELDER_ABUSE

_RS_ACH = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month ACH profile mean (currently 5)\n"
    "Current value: 5.0\n"
    "Labeled population: 300 customers (TP+FN pool=46 SAR, FP+TN pool=254 non-SAR, precision=15.3%)\n"
    "\n"
    "At the lowest value (0.00): TP=46, FP=254, FN=0, TN=0 (TP rate=100.0%, precision=15.3%).\n"
    "At current condition (5.00): TP=46, FP=254, FN=0, TN=0 (TP rate=100.0%, precision=15.3%).\n"
    "To keep TP rate >=90%: z_threshold <= 7.00 => TP=42, FP=199, FN=4, TN=55, precision=17.4%.\n"
    "To keep TP rate >=50%: z_threshold <= 9.00 => TP=32, FP=144, FN=14, TN=110, precision=18.2%.\n"
    "At the highest value (10.00): TP=23, FP=118, FN=23, TN=136, precision=16.3%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_ACH = "Tool result for rule_sar_backtest:\n" + _RS_ACH

_RS_CTR = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: CTR Client\n"
    "Current condition: Cash + Currency Exchange in/out total > $10K\n"
    "Sweep parameter: floor_amount - Minimum Cash/Currency Exchange total to trigger (currently $10K)\n"
    "Current value: 10,000\n"
    "Labeled population: 400 customers (TP+FN pool=52 SAR, FP+TN pool=348 non-SAR, precision=13.0%)\n"
    "\n"
    "At the lowest value (5,000.00): TP=52, FP=348, FN=0, TN=0 (TP rate=100.0%, precision=13.0%).\n"
    "At current condition (10,000.00): TP=52, FP=348, FN=0, TN=0 (TP rate=100.0%, precision=13.0%).\n"
    "To keep TP rate >=90%: floor_amount <= 15,000.00 => TP=48, FP=305, FN=4, TN=43, precision=13.6%.\n"
    "To keep TP rate >=50%: floor_amount <= 17,000.00 => TP=46, FP=282, FN=6, TN=66, precision=14.0%.\n"
    "At the highest value (17,000.00): TP=46, FP=282, FN=6, TN=66, precision=14.0%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_CTR = "Tool result for rule_sar_backtest:\n" + _RS_CTR

_RS_VELOCITY_SINGLE = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K\n"
    "Sweep parameter: pair_total - Minimum combined in+out pair total to trigger (currently $20K)\n"
    "Current value: 20,000\n"
    "Labeled population: 250 customers (TP+FN pool=49 SAR, FP+TN pool=201 non-SAR, precision=19.6%)\n"
    "\n"
    "At the lowest value (5,000.00): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).\n"
    "At current condition (20,000.00): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).\n"
    "To keep TP rate >=90%: pair_total <= 40,000.00 => TP=49, FP=201, FN=0, TN=0, precision=19.6%.\n"
    "To keep TP rate >=50%: pair_total <= 40,000.00 => TP=49, FP=201, FN=0, TN=0, precision=19.6%.\n"
    "At the highest value (40,000.00): TP=49, FP=201, FN=0, TN=0, precision=19.6%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_VELOCITY_SINGLE = "Tool result for rule_sar_backtest:\n" + _RS_VELOCITY_SINGLE

_RS_STRUCTURING_IN = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Structuring (Incoming Cash)\n"
    "Current condition: 3 qualifying days within 14-day window, each day's Cash CashIn total $3K-$40K\n"
    "Sweep parameter: daily_floor - Minimum daily Cash CashIn total for a qualifying day (currently $3K)\n"
    "Current value: 3,000\n"
    "Labeled population: 300 customers (TP+FN pool=46 SAR, FP+TN pool=254 non-SAR, precision=15.3%)\n"
    "\n"
    "At the lowest value (500.00): TP=46, FP=254, FN=0, TN=0 (TP rate=100.0%, precision=15.3%).\n"
    "At current condition (3,000.00): TP=46, FP=254, FN=0, TN=0 (TP rate=100.0%, precision=15.3%).\n"
    "To keep TP rate >=90%: daily_floor <= 3,000.00 => TP=46, FP=254, FN=0, TN=0, precision=15.3%.\n"
    "To keep TP rate >=50%: daily_floor <= 5,500.00 => TP=26, FP=157, FN=20, TN=97, precision=14.2%.\n"
    "At the highest value (6,500.00): TP=17, FP=122, FN=29, TN=132, precision=12.2%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_STRUCTURING = "Tool result for rule_sar_backtest:\n" + _RS_STRUCTURING_IN

_RS_ROUND_TRIP = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Round-trip\n"
    "Current condition: Outgoing Wire >= $50K returned within 30 days to same or related customer account (net <= 5% difference)\n"
    "Sweep parameter: floor_amount - Minimum Wire amount for the outgoing leg to trigger (currently $50K)\n"
    "Current value: 50,000\n"
    "Labeled population: 100 customers (TP+FN pool=35 SAR, FP+TN pool=65 non-SAR, precision=35.0%)\n"
    "\n"
    "At the lowest value (10,000.00): TP=35, FP=65, FN=0, TN=0 (TP rate=100.0%, precision=35.0%).\n"
    "At current condition (50,000.00): TP=35, FP=65, FN=0, TN=0 (TP rate=100.0%, precision=35.0%).\n"
    "To keep TP rate >=90%: floor_amount <= 70,000.00 => TP=32, FP=62, FN=3, TN=3, precision=34.0%.\n"
    "To keep TP rate >=50%: floor_amount <= 90,000.00 => TP=31, FP=60, FN=4, TN=5, precision=34.1%.\n"
    "At the highest value (90,000.00): TP=31, FP=60, FN=4, TN=5, precision=34.1%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_ROUND_TRIP = "Tool result for rule_sar_backtest:\n" + _RS_ROUND_TRIP

# ===========================================================================
# PRE-COMPUTED SEGMENT STATS block — real numbers from aria_custs_alerts.csv
# Business: 485 accounts (9.7%), 485 alerts (18.3%), FP=373 (76.9%), FN=0
# Individual: 4515 accounts (90.3%), 2167 alerts (81.7%), FP=1701 (78.5%), FN=0
# ===========================================================================

_SEGMENT_STATS = (
    "=== PRE-COMPUTED SEGMENT STATS (copy verbatim, do not compute new numbers) ===\n"
    "### Segment Overview\n"
    "\n"
    "**Business**\n"
    "- Accounts: **485** (9.7% of total)\n"
    "- Alerts: **485** (18.3% of all alerts)\n"
    "- False Positives: **373** (FP rate=76.9% of alerts)\n"
    "- False Negatives: **0**\n"
    "\n"
    "**Individual**\n"
    "- Accounts: **4,515** (90.3% of total)\n"
    "- Alerts: **2,167** (81.7% of all alerts)\n"
    "- False Positives: **1,701** (FP rate=78.5% of alerts)\n"
    "- False Negatives: **0**\n"
    "\n"
    "=== END PRE-COMPUTED SEGMENT STATS ==="
)
PC_SEGMENT_STATS = "Tool result for segment_stats:\n" + _SEGMENT_STATS

# ===========================================================================
# AA_V44_1-5  Gap 1 reinforcement: 0-char response after list_rules
# ===========================================================================

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
        "1. **Velocity Multiple** - 30.0% (54 SAR, 126 FP)\n"
        "2. **Risky International Transfer** - 23.0% (38 SAR, 127 FP)\n"
        "3. **Round-trip** - 23.5% (27 SAR, 88 FP)\n\n"
        "Velocity Multiple leads with 30.0% precision - nearly double the system average."
    )},
]})

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
        "1. **Round-trip** - 33.3% (40 SAR, 80 FP)\n"
        "2. **Risky International Transfer** - 23.1% (37 SAR, 123 FP)\n"
        "3. **Velocity Single** - 18.8% (48 SAR, 207 FP)\n\n"
        "Round-trip has the lowest false positive rate at 33.3% - 1 in 3 alerts is a confirmed SAR."
    )},
]})

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
        "1. **Velocity Multiple** - 30.0% (54 SAR, 126 FP)\n"
        "2. **Round-trip** - 23.5% (27 SAR, 88 FP)\n"
        "3. **Risky International Transfer** - 23.0% (38 SAR, 127 FP)"
    )},
]})

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
        "1. **Round-trip** - 33.3% (40 SAR, 80 FP)\n"
        "2. **Risky International Transfer** - 23.1% (37 SAR, 123 FP)\n"
        "3. **Velocity Single** - 18.8% (48 SAR, 207 FP)\n\n"
        "Round-trip's 33.3% precision means 1 in 3 alerts yields a confirmed SAR."
    )},
]})

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
        "1. **Burst in Originator Activity** - 34.5% (50 SAR, 95 FP)\n"
        "2. **Round-trip** - 24.3% (28 SAR, 87 FP)\n"
        "3. **Risky International Transfer** - 22.6% (35 SAR, 120 FP)\n\n"
        "Burst in Originator Activity is the most efficient at 34.5% - "
        "nearly 2.5x more precise than CTR Client (14.2%), the least efficient rule."
    )},
]})

# ===========================================================================
# AC_V44_1-6  Gap 3 reinforcement: "Age" phrasing
# Cluster 1: 4.1 yrs | Cluster 2: 7.3 yrs | Cluster 3: 5.8 yrs | Cluster 4: 9.2 yrs
# ===========================================================================

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest age?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest average account age at **9.2 years**, "
        "followed by Cluster 2 (**7.3 years**), Cluster 3 (**5.8 years**), and Cluster 1 (**4.1 years**)."
    )},
]})

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the oldest customers?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the oldest customers with an average account age "
        "of **9.2 years**, compared to Cluster 2 (**7.3 years**), Cluster 3 (**5.8 years**), "
        "and Cluster 1 (**4.1 years**)."
    )},
]})

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich segment has the youngest account history?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the shortest account history at **4.1 years** - "
        "the youngest segment. Cluster 3 is next at **5.8 years**, then Cluster 2 at **7.3 years**, "
        "and Cluster 4 at **9.2 years**."
    )},
]})

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nCan you rank the clusters by account age?"},
    {"role": "assistant", "content": (
        "From the clustering result above, ranked by average account age (highest to lowest):\n\n"
        "1. **Cluster 4** - 9.2 years\n"
        "2. **Cluster 2** - 7.3 years\n"
        "3. **Cluster 3** - 5.8 years\n"
        "4. **Cluster 1** - 4.1 years"
    )},
]})

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhat is the average account age for each cluster?"},
    {"role": "assistant", "content": (
        "From the clustering result above, the average account age (years) for each cluster:\n\n"
        "- **Cluster 1** - 4.1 years (n=121)\n"
        "- **Cluster 2** - 7.3 years (n=98)\n"
        "- **Cluster 3** - 5.8 years (n=156)\n"
        "- **Cluster 4** - 9.2 years (n=110)"
    )},
]})

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

# ===========================================================================
# AC_V44_MT1-3  Multi-turn: 3-turn conversations with prior tool call
# ===========================================================================

_PC_CLUSTER_ANALYSIS = "Tool result for ds_cluster_analysis:\n" + _CLUSTER_STATS

_T1_CLUSTER_RESPONSE = (
    "Clustering complete for Business customers.\n\n"
    + _CLUSTER_STATS + "\n\n"
    "**Summary:** Cluster 4 (n=110) is the highest-activity, highest-value segment - "
    "22.6 weekly transactions and $197,500 monthly volume. "
    "Cluster 1 (n=121) is the lowest-activity segment with the newest accounts (4.1 yrs). "
    "Clusters 2 and 3 represent intermediate activity levels."
)

# MT1: cluster -> highest monthly volume -> "What about cluster 4?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user", "content": "Can you cluster the Business customers by behavior?"},
    {"role": "assistant", "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v44_mt1", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v44_mt1", "content": _PC_CLUSTER_ANALYSIS},
    {"role": "assistant", "content": _T1_CLUSTER_RESPONSE},
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest monthly volume?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest monthly volume at **$197,500**, "
        "followed by Cluster 3 (**$41,380**), Cluster 2 (**$17,240**), and Cluster 1 (**$4,820**)."
    )},
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhat about cluster 4? Tell me more about it."},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 (n=110) is the highest-activity segment:\n\n"
        "- **Avg Weekly Transactions:** 22.6 - 7x higher than Cluster 1\n"
        "- **Avg Txn Amount:** $8,750\n"
        "- **Monthly Txn Volume:** $197,500\n"
        "- **Income:** $312,000\n"
        "- **Current Balance:** $148,000\n"
        "- **Account Age:** 9.2 years - most established accounts\n\n"
        "Cluster 4 is the highest-risk segment for AML monitoring: high transaction velocity, "
        "large transaction sizes, and substantial monthly throughput."
    )},
]})

# MT2: cluster -> youngest customers -> "What about cluster 2?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user", "content": "Run segmentation on Business customers."},
    {"role": "assistant", "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v44_mt2", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v44_mt2", "content": _PC_CLUSTER_ANALYSIS},
    {"role": "assistant", "content": _T1_CLUSTER_RESPONSE},
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the youngest customers?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the youngest customers with an average "
        "account age of **4.1 years**, followed by Cluster 3 (**5.8 years**), "
        "Cluster 2 (**7.3 years**), and Cluster 4 (**9.2 years**)."
    )},
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhat about cluster 2?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 2 (n=98) is a mid-tier segment:\n\n"
        "- **Avg Weekly Transactions:** 8.7\n"
        "- **Avg Txn Amount:** $4,310\n"
        "- **Monthly Txn Volume:** $17,240\n"
        "- **Income:** $145,000\n"
        "- **Current Balance:** $62,300\n"
        "- **Account Age:** 7.3 years\n\n"
        "Cluster 2 is more active and wealthier than Cluster 1, but below "
        "Cluster 3 in transaction frequency and below Cluster 4 on every metric."
    )},
]})

# MT3: cluster -> largest cluster -> "Tell me about cluster 1."
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user", "content": "Segment the Business customers for me."},
    {"role": "assistant", "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v44_mt3", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v44_mt3", "content": _PC_CLUSTER_ANALYSIS},
    {"role": "assistant", "content": _T1_CLUSTER_RESPONSE},
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster is the largest?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 3 is the largest segment with **156 customers**, "
        "followed by Cluster 1 (**121**), Cluster 4 (**110**), and Cluster 2 (**98**)."
    )},
    {"role": "user", "content": prev_context(_CLUSTER_STATS) + "\nTell me about cluster 1."},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 (n=121) is the lowest-activity segment:\n\n"
        "- **Avg Weekly Transactions:** 3.2 - the lowest of all clusters\n"
        "- **Avg Txn Amount:** $1,205\n"
        "- **Monthly Txn Volume:** $4,820\n"
        "- **Income:** $72,000 - lowest income segment\n"
        "- **Current Balance:** $18,500\n"
        "- **Account Age:** 4.1 years - newest accounts\n\n"
        "Cluster 1 represents low-activity, lower-income customers with newer relationships. "
        "Lower transaction velocity makes it the lowest-priority segment for AML escalation."
    )},
]})

# ===========================================================================
# AT_V44_1-6  threshold_tuning — current PRE-COMPUTED ANALYSIS format
# ===========================================================================

def _pc_analysis(t_min, max_fp, fp_zero, fn_zero_end, fn_first_t, fn_first_n,
                 max_fn, t_last, crossover_t, crossover_fp, crossover_fn,
                 optimal_start=None, optimal_end=None):
    lines = ["=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===",
             "### Threshold Tuning - False Positive / False Negative Trade-off\n",
             "**False Positives (FP)**",
             f"- At the lowest threshold ({t_min}): **{max_fp} FPs**"]
    if fp_zero:
        lines.append(f"- FPs reach zero at threshold **{fp_zero}**")
    else:
        lines.append("- FPs do not reach zero within the sweep range")
    lines.append("")
    lines.append("**False Negatives (FN)**")
    if fn_zero_end and fn_zero_end > t_min:
        lines.append(f"- FNs are zero from threshold {t_min} up to **{fn_zero_end}**")
    lines.append(f"- FNs first appear at threshold **{fn_first_t}** (FN={fn_first_n})")
    lines.append(f"- FNs reach **{max_fn}** at the highest threshold ({t_last})")
    lines.append("")
    lines.append(f"**Crossover Point** - threshold **{crossover_t}** (FP={crossover_fp}, FN={crossover_fn})")
    lines.append("")
    if optimal_start:
        lines.append(
            f"**Optimal Zone** (FP and FN both below 20% of max): threshold **{optimal_start}** to **{optimal_end}**"
        )
    else:
        lines.append(
            "**Optimal Zone**: no single threshold achieves both FP and FN below 20% of their maximums simultaneously."
        )
    lines.append("\n*(Detailed sweep chart shown below.)*")
    lines.append("=== END PRE-COMPUTED ANALYSIS ===")
    return "\n".join(lines)


# AT_V44_1: Business / TRXN_AMT_MONTHLY
_PC_BIZ_MONTHLY = _pc_analysis(
    t_min=4000, max_fp=1486, fp_zero=48000,
    fn_zero_end=12000, fn_first_t=16000, fn_first_n=9,
    max_fn=181, t_last=48000,
    crossover_t=24000, crossover_fp=218, crossover_fn=81,
    optimal_start=16000, optimal_end=20000,
)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show FP/FN trade-off for Business customers by monthly transaction amount"},
    {"role": "assistant",
     "content": "Calling threshold_tuning for Business customers.",
     "tool_calls": [tc("at_v44_1", "threshold_tuning",
                       {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "at_v44_1", "content": _PC_BIZ_MONTHLY},
    {"role": "assistant", "content": (
        "### Threshold Sweep - Business | Monthly Transaction Volume\n\n"
        + _PC_BIZ_MONTHLY + "\n\n"
        "The optimal zone ($16,000-$20,000/month) keeps both FP and FN below 20% of their "
        "maximums - above $24,000 FN losses accelerate while FP savings diminish."
    )},
]})

# AT_V44_2: Individual / TRXN_AMT_MONTHLY
_PC_IND_MONTHLY = _pc_analysis(
    t_min=500, max_fp=1857, fp_zero=6000,
    fn_zero_end=1500, fn_first_t=2000, fn_first_n=18,
    max_fn=542, t_last=6000,
    crossover_t=3000, crossover_fp=312, crossover_fn=156,
    optimal_start=2000, optimal_end=2500,
)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me the FP/FN trade-off for Individual customers by monthly transaction amount"},
    {"role": "assistant",
     "content": "Calling threshold_tuning for Individual customers.",
     "tool_calls": [tc("at_v44_2", "threshold_tuning",
                       {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "at_v44_2", "content": _PC_IND_MONTHLY},
    {"role": "assistant", "content": (
        "### Threshold Sweep - Individual | Monthly Transaction Volume\n\n"
        + _PC_IND_MONTHLY + "\n\n"
        "The optimal zone ($2,000-$2,500/month) balances FP reduction against SAR leakage - "
        "the crossover at $3,000 marks where FN losses begin to outpace FP savings."
    )},
]})

# AT_V44_3: Business / AVG_TRXN_AMT
_PC_BIZ_AMT = _pc_analysis(
    t_min=500, max_fp=1486, fp_zero=5000,
    fn_zero_end=1500, fn_first_t=2000, fn_first_n=14,
    max_fn=181, t_last=5000,
    crossover_t=3000, crossover_fp=195, crossover_fn=72,
    optimal_start=1500, optimal_end=2500,
)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "FP/FN trade-off for Business customers by average transaction amount"},
    {"role": "assistant",
     "content": "Calling threshold_tuning for Business customers.",
     "tool_calls": [tc("at_v44_3", "threshold_tuning",
                       {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "tool_call_id": "at_v44_3", "content": _PC_BIZ_AMT},
    {"role": "assistant", "content": (
        "### Threshold Sweep - Business | Avg Transaction Amount\n\n"
        + _PC_BIZ_AMT + "\n\n"
        "The optimal zone ($1,500-$2,500 per transaction) keeps both FP and FN below 20% of "
        "their maximums - the crossover at $3,000 is the balanced operating point."
    )},
]})

# AT_V44_4: Individual / AVG_TRXN_AMT — no optimal zone
_PC_IND_AMT = _pc_analysis(
    t_min=50, max_fp=1857, fp_zero=None,
    fn_zero_end=150, fn_first_t=200, fn_first_n=24,
    max_fn=542, t_last=500,
    crossover_t=300, crossover_fp=289, crossover_fn=187,
    optimal_start=None, optimal_end=None,
)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show the FP/FN trade-off for Individual customers by average transaction dollar amount"},
    {"role": "assistant",
     "content": "Calling threshold_tuning for Individual customers.",
     "tool_calls": [tc("at_v44_4", "threshold_tuning",
                       {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "tool_call_id": "at_v44_4", "content": _PC_IND_AMT},
    {"role": "assistant", "content": (
        "### Threshold Sweep - Individual | Avg Transaction Amount\n\n"
        + _PC_IND_AMT + "\n\n"
        "FPs do not reach zero within the sweep range - use the crossover at $300 "
        "(FP=289, FN=187) as the starting point for further 1D sweep refinement."
    )},
]})

# AT_V44_5: Business / AVG_TRXNS_WEEK — no optimal zone
_PC_BIZ_WEEK = _pc_analysis(
    t_min=4, max_fp=1486, fp_zero=None,
    fn_zero_end=8, fn_first_t=10, fn_first_n=18,
    max_fn=181, t_last=22,
    crossover_t=14, crossover_fp=312, crossover_fn=89,
    optimal_start=None, optimal_end=None,
)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How does transaction frequency affect FP/FN for Business customers?"},
    {"role": "assistant",
     "content": "Calling threshold_tuning for Business customers by weekly transaction count.",
     "tool_calls": [tc("at_v44_5", "threshold_tuning",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v44_5", "content": _PC_BIZ_WEEK},
    {"role": "assistant", "content": (
        "### Threshold Sweep - Business | Avg Weekly Transactions\n\n"
        + _PC_BIZ_WEEK + "\n\n"
        "No threshold simultaneously achieves both FP and FN below 20% of their maximums - "
        "the crossover at 14 transactions/week (FP=312, FN=89) is the recommended operating point."
    )},
]})

# AT_V44_6: Individual / AVG_TRXNS_WEEK — no optimal zone
_PC_IND_WEEK = _pc_analysis(
    t_min=2, max_fp=1857, fp_zero=None,
    fn_zero_end=4, fn_first_t=6, fn_first_n=28,
    max_fn=542, t_last=16,
    crossover_t=8, crossover_fp=420, crossover_fn=186,
    optimal_start=None, optimal_end=None,
)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show FP/FN for Individual customers by number of weekly transactions"},
    {"role": "assistant",
     "content": "Calling threshold_tuning for Individual customers by weekly transaction count.",
     "tool_calls": [tc("at_v44_6", "threshold_tuning",
                       {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v44_6", "content": _PC_IND_WEEK},
    {"role": "assistant", "content": (
        "### Threshold Sweep - Individual | Avg Weekly Transactions\n\n"
        + _PC_IND_WEEK + "\n\n"
        "No threshold simultaneously achieves both FP and FN below 20% of their maximums - "
        "use the crossover at 8 transactions/week (FP=420, FN=186) as the balanced operating point."
    )},
]})

# ===========================================================================
# ARS_V44_1-6  rule_sar_backtest — correct PRE-COMPUTED RULE SWEEP format
# Format: ### Rule Sweep — {Rule} | {param} -> RULE SWEEP block -> ONE sentence
# ===========================================================================

# ARS_V44_1: Elder Abuse (exact failing query from testing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Test SAR performance for Elder Abuse rule"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Elder Abuse.",
     "tool_calls": [tc("ars_v44_1", "rule_sar_backtest",
                       {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "tool_call_id": "ars_v44_1", "content": PC_RS_ELDER},
    {"role": "assistant", "content": (
        "### Rule Sweep — Elder Abuse | z_threshold\n\n"
        + _RS_ELDER_ABUSE + "\n\n"
        "At z=3 all 59 SARs are caught; tightening to z=6 drops the catch rate to 56% "
        "(33 caught) while saving 136 FPs."
    )},
]})

# ARS_V44_2: Activity Deviation (ACH)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What is the SAR catch rate for Activity Deviation ACH?"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Activity Deviation (ACH).",
     "tool_calls": [tc("ars_v44_2", "rule_sar_backtest",
                       {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "ars_v44_2", "content": PC_RS_ACH},
    {"role": "assistant", "content": (
        "### Rule Sweep — Activity Deviation (ACH) | z_threshold\n\n"
        + _RS_ACH + "\n\n"
        "Raising z_threshold from 5 to 7 cuts 55 FPs while losing only 4 SARs, "
        "improving precision from 15.3% to 17.4%."
    )},
]})

# ARS_V44_3: CTR Client
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "SAR backtest for CTR Client"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for CTR Client.",
     "tool_calls": [tc("ars_v44_3", "rule_sar_backtest",
                       {"risk_factor": "CTR Client"})]},
    {"role": "tool", "tool_call_id": "ars_v44_3", "content": PC_RS_CTR},
    {"role": "assistant", "content": (
        "### Rule Sweep — CTR Client | floor_amount\n\n"
        + _RS_CTR + "\n\n"
        "The $10K floor already catches all 52 SARs; raising to $15K cuts 43 FPs "
        "at the cost of 4 missed SARs (92% catch rate maintained)."
    )},
]})

# ARS_V44_4: Velocity Single
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me Velocity Single rule SAR performance"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Velocity Single.",
     "tool_calls": [tc("ars_v44_4", "rule_sar_backtest",
                       {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "tool_call_id": "ars_v44_4", "content": PC_RS_VELOCITY_SINGLE},
    {"role": "assistant", "content": (
        "### Rule Sweep — Velocity Single | pair_total\n\n"
        + _RS_VELOCITY_SINGLE + "\n\n"
        "All 49 SARs remain caught across the full sweep range ($5K-$40K) - "
        "pair_total does not differentiate SAR from non-SAR in this population."
    )},
]})

# ARS_V44_5: Structuring (Incoming Cash)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run SAR backtest for the structuring rule"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Structuring (Incoming Cash).",
     "tool_calls": [tc("ars_v44_5", "rule_sar_backtest",
                       {"risk_factor": "Structuring (Incoming Cash)"})]},
    {"role": "tool", "tool_call_id": "ars_v44_5", "content": PC_RS_STRUCTURING},
    {"role": "assistant", "content": (
        "### Rule Sweep — Structuring (Incoming Cash) | daily_floor\n\n"
        + _RS_STRUCTURING_IN + "\n\n"
        "The $3K floor catches all 46 SARs; above $5,500 the catch rate drops below 57%, "
        "making $3K-$5K the practical operating range."
    )},
]})

# ARS_V44_6: Round-trip
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What happens to FP/FN if I change the Round-trip floor amount?"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Round-trip.",
     "tool_calls": [tc("ars_v44_6", "rule_sar_backtest",
                       {"risk_factor": "Round-trip", "sweep_param": "floor_amount"})]},
    {"role": "tool", "tool_call_id": "ars_v44_6", "content": PC_RS_ROUND_TRIP},
    {"role": "assistant", "content": (
        "### Rule Sweep — Round-trip | floor_amount\n\n"
        + _RS_ROUND_TRIP + "\n\n"
        "Round-trip already has the system's best precision at 35.0%; raising the floor "
        "above $70K begins missing SARs with minimal FP savings (3 FPs saved, 3 SARs lost)."
    )},
]})

# ===========================================================================
# ASS_V44_1-6  segment_stats — correct PRE-COMPUTED SEGMENT STATS format
# Fixes routing ("segment breakdown" was misrouting to cluster_analysis)
# and stale format (raw counts -> markdown table).
# Business: 485 (9.7%), 485 alerts (18.3%), FP=373 (76.9%), FN=0
# Individual: 4515 (90.3%), 2167 alerts (81.7%), FP=1701 (78.5%), FN=0
# ===========================================================================

# ASS_V44_1: exact failing query from testing
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me the segment breakdown for Business customers"},
    {"role": "assistant",
     "content": "Calling segment_stats.",
     "tool_calls": [tc("ass_v44_1", "segment_stats", {})]},
    {"role": "tool", "tool_call_id": "ass_v44_1", "content": PC_SEGMENT_STATS},
    {"role": "assistant", "content": (
        "### Segment Overview\n\n"
        + _SEGMENT_STATS + "\n\n"
        "Business customers account for 18.3% of all alerts despite being only 9.7% of "
        "accounts, indicating a significantly higher per-account alert rate than Individual."
    )},
]})

# ASS_V44_2: how many customers in each segment
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How many customers are in each segment?"},
    {"role": "assistant",
     "content": "Calling segment_stats.",
     "tool_calls": [tc("ass_v44_2", "segment_stats", {})]},
    {"role": "tool", "tool_call_id": "ass_v44_2", "content": PC_SEGMENT_STATS},
    {"role": "assistant", "content": (
        "### Segment Overview\n\n"
        + _SEGMENT_STATS + "\n\n"
        "The dataset has 5,000 customers total: 485 Business (9.7%) and 4,515 Individual (90.3%)."
    )},
]})

# ASS_V44_3: overview of segments and alert rates
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Give me an overview of customer segments and their alert rates"},
    {"role": "assistant",
     "content": "Calling segment_stats.",
     "tool_calls": [tc("ass_v44_3", "segment_stats", {})]},
    {"role": "tool", "tool_call_id": "ass_v44_3", "content": PC_SEGMENT_STATS},
    {"role": "assistant", "content": (
        "### Segment Overview\n\n"
        + _SEGMENT_STATS + "\n\n"
        "Both segments have similar FP rates (~77-79%), but Business customers generate "
        "disproportionately more alerts per account."
    )},
]})

# ASS_V44_4: percentage of alerts from Individual
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What percentage of alerts come from Individual customers?"},
    {"role": "assistant",
     "content": "Calling segment_stats.",
     "tool_calls": [tc("ass_v44_4", "segment_stats", {})]},
    {"role": "tool", "tool_call_id": "ass_v44_4", "content": PC_SEGMENT_STATS},
    {"role": "assistant", "content": (
        "### Segment Overview\n\n"
        + _SEGMENT_STATS + "\n\n"
        "Individual customers generate 81.7% of all alerts (2,167 of 2,652 total), "
        "driven by their larger population share (90.3% of accounts)."
    )},
]})

# ASS_V44_5: how many Business customers
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How many Business customers do we have?"},
    {"role": "assistant",
     "content": "Calling segment_stats.",
     "tool_calls": [tc("ass_v44_5", "segment_stats", {})]},
    {"role": "tool", "tool_call_id": "ass_v44_5", "content": PC_SEGMENT_STATS},
    {"role": "assistant", "content": (
        "### Segment Overview\n\n"
        + _SEGMENT_STATS + "\n\n"
        "There are **485 Business customers** (9.7% of the 5,000-customer dataset), "
        "generating 485 alerts with a 76.9% FP rate."
    )},
]})

# ASS_V44_6: alerts by customer type
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me a summary of alerts by customer type"},
    {"role": "assistant",
     "content": "Calling segment_stats.",
     "tool_calls": [tc("ass_v44_6", "segment_stats", {})]},
    {"role": "tool", "tool_call_id": "ass_v44_6", "content": PC_SEGMENT_STATS},
    {"role": "assistant", "content": (
        "### Segment Overview\n\n"
        + _SEGMENT_STATS + "\n\n"
        "Total alerts: 2,652 — Individual drives 81.7% of volume (2,167) vs Business 18.3% (485), "
        "with FP rates of 78.5% and 76.9% respectively."
    )},
]})

# ===========================================================================
# ARL_V44_1-3  Rule 23: numbered rule references
# "rule 5" / "rule 1" / "rule 12" -> call list_rules -> state no such rule
# ===========================================================================

_RULE_NAMES_16 = (
    "Activity Deviation (ACH), Activity Deviation (Check), Activity Deviation (Wire), "
    "Elder Abuse, Velocity Single, Velocity Multiple, Detect Excessive Transaction Activity, "
    "Structuring (Incoming Cash), Structuring (Outgoing Cash), CTR Client, "
    "Burst in Originator Activity, Burst in Beneficiary Activity, Risky International Transfer, "
    "Funnel Account, Round-trip, and Human Trafficking Indicators"
)

# ARL_V44_1: "rule 5" (exact failing query from testing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Test SAR performance for rule 5"},
    {"role": "assistant",
     "content": "Calling list_rules to check available rules.",
     "tool_calls": [tc("arl_v44_1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v44_1", "content": PC_LIST_F},
    {"role": "assistant", "content": (
        _LIST_F + "\n\n"
        "There is no rule called 'rule 5' in the system. Rules are identified by name, not number. "
        f"The 16 available rules are: {_RULE_NAMES_16}. "
        "Please specify which rule you'd like to test by name."
    )},
]})

# ARL_V44_2: "rule 1"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show the SAR catch rate for rule 1"},
    {"role": "assistant",
     "content": "Calling list_rules to check available rules.",
     "tool_calls": [tc("arl_v44_2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v44_2", "content": PC_LIST_G},
    {"role": "assistant", "content": (
        _LIST_G + "\n\n"
        "There is no rule called 'rule 1' in the system — rules are identified by name, not number. "
        f"The 16 available rules are: {_RULE_NAMES_16}. "
        "Which rule would you like to analyze?"
    )},
]})

# ARL_V44_3: "rule 12"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Analyze rule 12 performance"},
    {"role": "assistant",
     "content": "Calling list_rules to check available rules.",
     "tool_calls": [tc("arl_v44_3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v44_3", "content": PC_LIST_E},
    {"role": "assistant", "content": (
        _LIST_E + "\n\n"
        "There is no rule called 'rule 12' in the system. Rules have names, not numbers. "
        f"The 16 available rules are: {_RULE_NAMES_16}. "
        "Please provide the rule name and I'll run the SAR backtest."
    )},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _has_threshold_tuning(ex):
    for m in ex["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                if call.get("function", {}).get("name") == "threshold_tuning":
                    return True
    return False


def _has_rule_sar_backtest(ex):
    for m in ex["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                if call.get("function", {}).get("name") == "rule_sar_backtest":
                    return True
    return False


def _has_segment_stats(ex):
    for m in ex["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                if call.get("function", {}).get("name") == "segment_stats":
                    return True
    return False


# ---------------------------------------------------------------------------
# Combine V43 base (minus stale examples) + V44 and write
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

        filtered = v43_base
        for fn, label in [
            (_has_threshold_tuning,  "threshold_tuning"),
            (_has_rule_sar_backtest, "rule_sar_backtest"),
            (_has_segment_stats,     "segment_stats"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            print(f"[V44] Removed {before - len(filtered)} stale {label} examples")

        all_examples = filtered + examples
        with open(V44_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V44] Combined: {V44_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V44] WARNING: V43 base not found at {V43_BASE_PATH}")


if __name__ == "__main__":
    main()

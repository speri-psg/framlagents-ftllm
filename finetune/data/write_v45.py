"""
V45 training examples (2026-05-13).

Targets:

  AA_V45_1-6   Gap 5: list_rules — model dumps all 16 rules instead of concise top-3.
               Two examples per query type (highest precision, lowest precision,
               most SARs). Each uses a different dataset so the model cannot
               memorise the answer. Response = PRE-COMPUTED block + top-3 bullet
               list + ONE sentence. Do NOT list all 16.

  AC_V45_1-5   Gap 3/15 reinforcement: cluster attribute queries answered from
               [PREVIOUS CLUSTERING RESULT] without re-running any tool.
               Covers: income, balance, transaction volume, ranking, "which segment".

  AC_V45_6-8   Gap 6: fresh clustering + attribute answer in one shot.
               Model calls ds_cluster_analysis, gets stats block back,
               and answers the attribute question directly from the stats — no
               second turn needed.

  ARS_V45_1-10 Gap 4: rule_sar_backtest rebuild.
               Surgical filter in main() removes only OLD format examples
               (header = "PRE-COMPUTED RULE SAR BACKTEST"), keeps the V44
               "PRE-COMPUTED RULE SWEEP" examples. Adds 10 new examples:
               - different rules and params (incl. cluster-filtered calls)
               - cluster=2 and cluster=4 variants
               - multi-step list_rules → rule_sar_backtest

  AT_V45_1-5   Gap 3: sar_backtest real insight.
               Stale filter removes examples where the assistant final message
               (after the sar_backtest tool call) ends with or contains
               "See chart/table below." or "*(Detailed sweep chart shown below.)*".
               New examples generate real sentences referencing specific numbers.

  AM_V45_1-3   Gap 1: multi-step cluster → SAR chaining.
               [PREVIOUS CLUSTERING RESULT] is in context.
               User asks about SARs caught in the highest-X cluster.
               Model reads cluster number from stats block → calls
               cluster_rule_summary with that cluster → reports SAR/FP results.

Filters applied in main():
  _has_stale_sar_backtest        : removes examples where "See chart/table below."
                                   or "*(Detailed sweep chart shown below.)*"
                                   appears in the assistant's final response after
                                   a sar_backtest tool call.
  _has_stale_rule_sar_backtest   : removes examples where a tool-role message
                                   content contains "PRE-COMPUTED RULE SAR BACKTEST"
                                   (old header, NOT "PRE-COMPUTED RULE SWEEP").

Base:   aria_train_combined_v44_full.jsonl (751)
After:  751 - filtered + new V45 examples
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V44_BASE_PATH = DATA_DIR / "aria_train_combined_v44_full.jsonl"
V45_ONLY_PATH = DATA_DIR / "aria_train_v45.jsonl"
V45_FULL_PATH = DATA_DIR / "aria_train_combined_v45_full.jsonl"

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
from write_v44 import (                                                # noqa: E402
    _LIST_F, PC_LIST_F,
    _LIST_G, PC_LIST_G,
    _RS_ELDER_ABUSE, PC_RS_ELDER,
    _RS_ACH, PC_RS_ACH,
    _RS_CTR, PC_RS_CTR,
    _RS_VELOCITY_SINGLE, PC_RS_VELOCITY_SINGLE,
    _RS_STRUCTURING_IN, PC_RS_STRUCTURING,
    _RS_ROUND_TRIP, PC_RS_ROUND_TRIP,
    _SEGMENT_STATS, PC_SEGMENT_STATS,
    _RULE_NAMES_16,
)

examples = []

# ===========================================================================
# Dataset H — Funnel Account wins precision at 31.8%  (used by AA_V45)
# ===========================================================================

_LIST_H = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=340, SAR=55, FP=285, precision=16.2%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=210, SAR=38, FP=172, precision=18.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=390, SAR=58, FP=332, precision=14.9%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=265, SAR=51, FP=214, precision=19.2%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=365, SAR=50, FP=315, precision=13.7%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=290, SAR=46, FP=244, precision=15.9%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=240, SAR=40, FP=200, precision=16.7%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=460, SAR=60, FP=400, precision=13.0%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=205, SAR=37, FP=168, precision=18.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=195, SAR=34, FP=161, precision=17.4%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=170, SAR=39, FP=131, precision=22.9%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=160, SAR=27, FP=133, precision=16.9%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=185, SAR=36, FP=149, precision=19.5%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=110, SAR=51, FP=59, precision=46.4%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=125, SAR=28, FP=97, precision=22.4%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=102, SAR=21, FP=81, precision=20.6%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_H = "Tool result for list_rules:\n" + _LIST_H

# Dataset I — CTR Client + DETA tie at lowest precision (12.8% / 12.5%)
_LIST_I = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=310, SAR=50, FP=260, precision=16.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=195, SAR=34, FP=161, precision=17.4%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=405, SAR=62, FP=343, precision=15.3%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=255, SAR=49, FP=206, precision=19.2%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=400, SAR=50, FP=350, precision=12.5%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=275, SAR=44, FP=231, precision=16.0%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=235, SAR=38, FP=197, precision=16.2%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=469, SAR=60, FP=409, precision=12.8%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=200, SAR=36, FP=164, precision=18.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=192, SAR=33, FP=159, precision=17.2%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=162, SAR=37, FP=125, precision=22.8%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=152, SAR=26, FP=126, precision=17.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=182, SAR=35, FP=147, precision=19.2%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=172, SAR=34, FP=138, precision=19.8%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=118, SAR=29, FP=89, precision=24.6%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=100, SAR=21, FP=79, precision=21.0%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_I = "Tool result for list_rules:\n" + _LIST_I

# Dataset J — Elder Abuse catches most SARs (72), CTR is close (68)
_LIST_J = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=338, SAR=54, FP=284, precision=16.0%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=202, SAR=36, FP=166, precision=17.8%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=430, SAR=72, FP=358, precision=16.7%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=258, SAR=50, FP=208, precision=19.4%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=372, SAR=53, FP=319, precision=14.2%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=288, SAR=46, FP=242, precision=16.0%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=242, SAR=40, FP=202, precision=16.5%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=465, SAR=68, FP=397, precision=14.6%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=198, SAR=35, FP=163, precision=17.7%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=188, SAR=32, FP=156, precision=17.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=163, SAR=38, FP=125, precision=23.3%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=153, SAR=26, FP=127, precision=17.0%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=183, SAR=35, FP=148, precision=19.1%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=173, SAR=35, FP=138, precision=20.2%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=116, SAR=28, FP=88, precision=24.1%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=101, SAR=21, FP=80, precision=20.8%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_J = "Tool result for list_rules:\n" + _LIST_J

# ===========================================================================
# AA_V45_1-6  Gap 5: list_rules top-3 concise responses
# Pattern: call list_rules → verbatim block → top-3 bullets → 1 sentence
# Do NOT list all 16 rules. Do NOT sort by FP count.
# ===========================================================================

# AA_V45_1: highest precision (Dataset H — Funnel Account wins)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v45_1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v45_1", "content": PC_LIST_H},
    {"role": "assistant", "content": (
        _LIST_H + "\n\n"
        "**Top rules by precision:**\n"
        "1. **Funnel Account** - 46.4% (51 SAR, 59 FP)\n"
        "2. **Risky International Transfer** - 22.9% (39 SAR, 131 FP)\n"
        "3. **Round-trip** - 22.4% (28 SAR, 97 FP)\n\n"
        "Funnel Account leads with 46.4% precision — nearly 1 in 2 alerts is a confirmed SAR."
    )},
]})

# AA_V45_2: highest precision (Dataset F — Velocity Multiple wins)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which AML rule has the best precision?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v45_2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v45_2", "content": PC_LIST_F},
    {"role": "assistant", "content": (
        _LIST_F + "\n\n"
        "**Top rules by precision:**\n"
        "1. **Velocity Multiple** - 30.0% (54 SAR, 126 FP)\n"
        "2. **Round-trip** - 23.5% (27 SAR, 88 FP)\n"
        "3. **Risky International Transfer** - 23.0% (38 SAR, 127 FP)\n\n"
        "Velocity Multiple leads with 30.0% precision — 3 in 10 alerts confirm a SAR."
    )},
]})

# AA_V45_3: lowest precision (Dataset I — DETA at 12.5%, CTR at 12.8%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the lowest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v45_3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v45_3", "content": PC_LIST_I},
    {"role": "assistant", "content": (
        _LIST_I + "\n\n"
        "**Bottom rules by precision (most false positives per SAR):**\n"
        "1. **Detect Excessive Transaction Activity** - 12.5% (50 SAR, 350 FP)\n"
        "2. **CTR Client** - 12.8% (60 SAR, 409 FP)\n"
        "3. **Elder Abuse** - 15.3% (62 SAR, 343 FP)\n\n"
        "DETA has the lowest precision at 12.5% — 7 in 8 alerts are false positives."
    )},
]})

# AA_V45_4: lowest precision (Dataset G — CTR at 13.0%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the worst SAR-to-FP ratio?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v45_4", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v45_4", "content": PC_LIST_G},
    {"role": "assistant", "content": (
        _LIST_G + "\n\n"
        "**Worst SAR-to-FP ratio (lowest precision):**\n"
        "1. **CTR Client** - 13.0% (59 SAR, 396 FP)\n"
        "2. **Detect Excessive Transaction Activity** - 13.6% (51 SAR, 324 FP)\n"
        "3. **Elder Abuse** - 15.0% (60 SAR, 340 FP)\n\n"
        "CTR Client has the worst ratio at 13.0% — 396 false positives for every 59 confirmed SARs."
    )},
]})

# AA_V45_5: most SARs (Dataset J — Elder Abuse catches 72)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule catches the most SARs?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v45_5", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v45_5", "content": PC_LIST_J},
    {"role": "assistant", "content": (
        _LIST_J + "\n\n"
        "**Top rules by SAR count:**\n"
        "1. **Elder Abuse** - 72 SAR (430 alerts, 16.7% precision)\n"
        "2. **CTR Client** - 68 SAR (465 alerts, 14.6% precision)\n"
        "3. **Activity Deviation (ACH)** - 54 SAR (338 alerts, 16.0% precision)\n\n"
        "Elder Abuse catches the most SARs at 72, but at 16.7% precision — most alerts are still false positives."
    )},
]})

# AA_V45_6: most SARs (Dataset F — Elder Abuse 63, CTR 61)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules generate the most true SARs?"},
    {"role": "assistant",
     "content": "Calling list_rules.",
     "tool_calls": [tc("aa_v45_6", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa_v45_6", "content": PC_LIST_F},
    {"role": "assistant", "content": (
        _LIST_F + "\n\n"
        "**Top rules by SAR volume:**\n"
        "1. **Elder Abuse** - 63 SAR (395 alerts, 15.9% precision)\n"
        "2. **CTR Client** - 61 SAR (450 alerts, 13.6% precision)\n"
        "3. **Velocity Multiple** - 54 SAR (180 alerts, 30.0% precision)\n\n"
        "Elder Abuse generates the most SARs (63), though Velocity Multiple is far more efficient at 30.0% precision."
    )},
]})

# ===========================================================================
# AC_V45_1-5  Gap 3/15 reinforcement: attribute queries from [PREVIOUS RESULT]
# Using _CLUSTER_STATS from write_v42 (Business, 4 clusters, all attrs present)
# Cluster 1: Income=$72K, Balance=$18.5K, Monthly=$4,820, Wkly=3.2
# Cluster 2: Income=$145K, Balance=$62.3K, Monthly=$17,240, Wkly=8.7
# Cluster 3: Income=$98K, Balance=$31.1K, Monthly=$41,380, Wkly=14.2
# Cluster 4: Income=$312K, Balance=$148K, Monthly=$197,500, Wkly=22.6
# ===========================================================================

# AC_V45_1: which cluster has the highest income
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the highest income?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the highest average income at **$312,000**, "
        "followed by Cluster 2 (**$145,000**), Cluster 3 (**$98,000**), and Cluster 1 (**$72,000**)."
    )},
]})

# AC_V45_2: which cluster has the lowest balance
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich segment has the lowest current balance?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the lowest current balance at **$18,500**, "
        "followed by Cluster 3 (**$31,100**), Cluster 2 (**$62,300**), and Cluster 4 (**$148,000**)."
    )},
]})

# AC_V45_3: rank clusters by monthly transaction volume
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nRank the clusters by monthly transaction volume."},
    {"role": "assistant", "content": (
        "From the clustering result above, ranked by monthly transaction volume (highest to lowest):\n\n"
        "1. **Cluster 4** - $197,500/month (n=110)\n"
        "2. **Cluster 3** - $41,380/month (n=156)\n"
        "3. **Cluster 2** - $17,240/month (n=98)\n"
        "4. **Cluster 1** - $4,820/month (n=121)\n\n"
        "Cluster 4 generates more than 4x the monthly volume of Cluster 3, the next highest."
    )},
]})

# AC_V45_4: which segment is most active (weekly transactions)
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the most active transaction behavior?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 is the most active with **22.6 average weekly transactions**, "
        "compared to Cluster 3 (**14.2**), Cluster 2 (**8.7**), and Cluster 1 (**3.2**). "
        "Cluster 4 also has the highest average transaction amount at $8,750 and $197,500 in monthly volume — "
        "the highest-activity, highest-value segment."
    )},
]})

# AC_V45_5: which segment has the oldest customers (synonym for Account Age)
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_CLUSTER_STATS) + "\nWhich segment has customers with the longest account history?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 4 has the longest average account history at **9.2 years**, "
        "followed by Cluster 2 (**7.3 years**), Cluster 3 (**5.8 years**), and Cluster 1 (**4.1 years**)."
    )},
]})

# ===========================================================================
# AC_V45_6-8  Gap 6: fresh clustering + attribute answer in one response
# Model calls ds_cluster_analysis → receives stats block → answers immediately
# A second dataset so the model learns from fresh data, not memorised numbers
# ===========================================================================

_CLUSTER_STATS_IND = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Individual | 4 clusters

Cluster 1 (n=312):
- Avg Weekly Transactions: **2.1**
- Avg Weekly Txn Amount: **$520**
- Monthly Txn Volume: **$1,092**
- Income: **$48,000**
- Current Balance: **$8,200**
- Account Age (years): **3.4**

Cluster 2 (n=245):
- Avg Weekly Transactions: **5.8**
- Avg Weekly Txn Amount: **$1,840**
- Monthly Txn Volume: **$10,672**
- Income: **$87,000**
- Current Balance: **$32,400**
- Account Age (years): **6.1**

Cluster 3 (n=198):
- Avg Weekly Transactions: **11.4**
- Avg Weekly Txn Amount: **$3,120**
- Monthly Txn Volume: **$35,568**
- Income: **$134,000**
- Current Balance: **$68,900**
- Account Age (years): **8.7**

Cluster 4 (n=87):
- Avg Weekly Transactions: **19.3**
- Avg Weekly Txn Amount: **$6,450**
- Monthly Txn Volume: **$124,485**
- Income: **$261,000**
- Current Balance: **$118,500**
- Account Age (years): **11.2**
=== END PRE-COMPUTED CLUSTER STATS ==="""

_PC_CLUSTER_IND = "Tool result for ds_cluster_analysis:\n" + _CLUSTER_STATS_IND

_T1_IND_RESPONSE = (
    "Clustering complete for Individual customers.\n\n"
    + _CLUSTER_STATS_IND + "\n\n"
    "**Summary:** Cluster 4 (n=87) is the highest-activity, highest-value segment — "
    "19.3 weekly transactions and $124,485 monthly volume. "
    "Cluster 1 (n=312) is the lowest-activity, lowest-income segment with the newest accounts (3.4 yrs). "
    "Clusters 2 and 3 represent mid-tier activity levels."
)

# AC_V45_6: run clustering, immediately answer "which cluster is oldest"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Which cluster of Individual customers has the oldest account age?"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v45_6", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v45_6", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": (
        _CLUSTER_STATS_IND + "\n\n"
        "Cluster 4 has the oldest average account age at **11.2 years**, "
        "followed by Cluster 3 (**8.7 years**), Cluster 2 (**6.1 years**), and Cluster 1 (**3.4 years**)."
    )},
]})

# AC_V45_7: run clustering, immediately answer "which cluster has highest income"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Run clustering on Individual customers — which segment has the highest income?"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v45_7", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v45_7", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": (
        _CLUSTER_STATS_IND + "\n\n"
        "Cluster 4 has the highest average income at **$261,000**, "
        "followed by Cluster 3 (**$134,000**), Cluster 2 (**$87,000**), and Cluster 1 (**$48,000**)."
    )},
]})

# AC_V45_8: multi-turn — cluster Business → follow-up "which has lowest balance"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Segment Business customers."},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v45_8a", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v45_8a",
     "content": "Tool result for ds_cluster_analysis:\n" + _CLUSTER_STATS},
    {"role": "assistant", "content": (
        "Clustering complete for Business customers.\n\n"
        + _CLUSTER_STATS + "\n\n"
        "**Summary:** Cluster 4 (n=110) is the highest-activity segment — "
        "22.6 weekly transactions and $197,500 monthly volume. "
        "Cluster 1 (n=121) is the lowest-activity, lowest-balance segment."
    )},
    {"role": "user",
     "content": prev_context(_CLUSTER_STATS) + "\nWhich cluster has the lowest current balance?"},
    {"role": "assistant", "content": (
        "From the clustering result above, Cluster 1 has the lowest current balance at **$18,500**, "
        "followed by Cluster 3 (**$31,100**), Cluster 2 (**$62,300**), and Cluster 4 (**$148,000**)."
    )},
]})

# ===========================================================================
# PRE-COMPUTED RULE SWEEP blocks for ARS_V45 (new rules / params)
# ===========================================================================

_RS_FUNNEL = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Funnel Account\n"
    "Current condition: Outgoing wires to >= 3 counterparties within 14 days, combined total >= $25K\n"
    "Sweep parameter: floor_amount - Minimum combined outgoing wire total to trigger (currently $25K)\n"
    "Current value: 25,000\n"
    "Labeled population: 110 customers (TP+FN pool=51 SAR, FP+TN pool=59 non-SAR, precision=46.4%)\n"
    "\n"
    "At the lowest value (5,000.00): TP=51, FP=59, FN=0, TN=0 (TP rate=100.0%, precision=46.4%).\n"
    "At current condition (25,000.00): TP=51, FP=59, FN=0, TN=0 (TP rate=100.0%, precision=46.4%).\n"
    "To keep TP rate >=90%: floor_amount <= 40,000.00 => TP=47, FP=47, FN=4, TN=12, precision=50.0%.\n"
    "To keep TP rate >=50%: floor_amount <= 65,000.00 => TP=27, FP=28, FN=24, TN=31, precision=49.1%.\n"
    "At the highest value (70,000.00): TP=22, FP=21, FN=29, TN=38, precision=51.2%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_FUNNEL = "Tool result for rule_sar_backtest:\n" + _RS_FUNNEL

_RS_HUMAN_TRAFFICKING = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Human Trafficking Indicators\n"
    "Current condition: >= 3 qualifying days in 30-day window, each day's cash-in total $1K-$5K, matched against victim indicator list\n"
    "Sweep parameter: days_required - Minimum number of qualifying days within the 30-day window (currently 3)\n"
    "Current value: 3\n"
    "Labeled population: 102 customers (TP+FN pool=21 SAR, FP+TN pool=81 non-SAR, precision=20.6%)\n"
    "\n"
    "At the lowest value (1): TP=21, FP=81, FN=0, TN=0 (TP rate=100.0%, precision=20.6%).\n"
    "At current condition (3): TP=21, FP=81, FN=0, TN=0 (TP rate=100.0%, precision=20.6%).\n"
    "To keep TP rate >=90%: days_required <= 4 => TP=19, FP=68, FN=2, TN=13, precision=21.8%.\n"
    "To keep TP rate >=50%: days_required <= 6 => TP=11, FP=39, FN=10, TN=42, precision=22.0%.\n"
    "At the highest value (7): TP=8, FP=27, FN=13, TN=54, precision=22.9%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_HUMAN_TRAFFICKING = "Tool result for rule_sar_backtest:\n" + _RS_HUMAN_TRAFFICKING

_RS_RISKY_INT = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Risky International Transfer\n"
    "Current condition: Wire to/from high-risk country jurisdiction >= $10K in single transaction\n"
    "Sweep parameter: floor_amount - Minimum wire amount to trigger (currently $10K)\n"
    "Current value: 10,000\n"
    "Labeled population: 163 customers (TP+FN pool=38 SAR, FP+TN pool=125 non-SAR, precision=23.3%)\n"
    "\n"
    "At the lowest value (2,500.00): TP=38, FP=125, FN=0, TN=0 (TP rate=100.0%, precision=23.3%).\n"
    "At current condition (10,000.00): TP=38, FP=125, FN=0, TN=0 (TP rate=100.0%, precision=23.3%).\n"
    "To keep TP rate >=90%: floor_amount <= 20,000.00 => TP=35, FP=101, FN=3, TN=24, precision=25.7%.\n"
    "To keep TP rate >=50%: floor_amount <= 35,000.00 => TP=20, FP=58, FN=18, TN=67, precision=25.6%.\n"
    "At the highest value (40,000.00): TP=15, FP=42, FN=23, TN=83, precision=26.3%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_RISKY_INT = "Tool result for rule_sar_backtest:\n" + _RS_RISKY_INT

# Cluster-filtered sweep blocks (cluster=2 and cluster=4)
_RS_ELDER_CLUSTER2 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse [Cluster 2 only]\n"
    "Current condition: Age >= 60 AND 14-day outgoing >= $5K AND >= 3 std dev above 90-day mean\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3)\n"
    "Current value: 3.0\n"
    "Labeled population: 94 customers in Cluster 2 (TP+FN pool=11 SAR, FP+TN pool=83 non-SAR, precision=11.7%)\n"
    "\n"
    "At the lowest value (0.00): TP=11, FP=83, FN=0, TN=0 (TP rate=100.0%, precision=11.7%).\n"
    "At current condition (3.00): TP=11, FP=83, FN=0, TN=0 (TP rate=100.0%, precision=11.7%).\n"
    "To keep TP rate >=90%: z_threshold <= 4.00 => TP=10, FP=68, FN=1, TN=15, precision=12.8%.\n"
    "To keep TP rate >=50%: z_threshold <= 6.00 => TP=6, FP=41, FN=5, TN=42, precision=12.8%.\n"
    "At the highest value (10.00): TP=0, FP=12, FN=11, TN=71, precision=0.0%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_ELDER_C2 = "Tool result for rule_sar_backtest:\n" + _RS_ELDER_CLUSTER2

_RS_ACH_CLUSTER4 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH) [Cluster 4 only]\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month ACH profile mean (currently 5)\n"
    "Current value: 5.0\n"
    "Labeled population: 110 customers in Cluster 4 (TP+FN pool=22 SAR, FP+TN pool=88 non-SAR, precision=20.0%)\n"
    "\n"
    "At the lowest value (0.00): TP=22, FP=88, FN=0, TN=0 (TP rate=100.0%, precision=20.0%).\n"
    "At current condition (5.00): TP=22, FP=88, FN=0, TN=0 (TP rate=100.0%, precision=20.0%).\n"
    "To keep TP rate >=90%: z_threshold <= 7.00 => TP=20, FP=61, FN=2, TN=27, precision=24.7%.\n"
    "To keep TP rate >=50%: z_threshold <= 9.00 => TP=12, FP=38, FN=10, TN=50, precision=24.0%.\n"
    "At the highest value (10.00): TP=9, FP=27, FN=13, TN=61, precision=25.0%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_ACH_C4 = "Tool result for rule_sar_backtest:\n" + _RS_ACH_CLUSTER4

_RS_ROUND_TRIP_CLUSTER4 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Round-trip [Cluster 4 only]\n"
    "Current condition: Outgoing Wire >= $50K returned within 30 days to same or related customer account (net <= 5% difference)\n"
    "Sweep parameter: floor_amount - Minimum Wire amount for the outgoing leg to trigger (currently $50K)\n"
    "Current value: 50,000\n"
    "Labeled population: 42 customers in Cluster 4 (TP+FN pool=18 SAR, FP+TN pool=24 non-SAR, precision=42.9%)\n"
    "\n"
    "At the lowest value (10,000.00): TP=18, FP=24, FN=0, TN=0 (TP rate=100.0%, precision=42.9%).\n"
    "At current condition (50,000.00): TP=18, FP=24, FN=0, TN=0 (TP rate=100.0%, precision=42.9%).\n"
    "To keep TP rate >=90%: floor_amount <= 70,000.00 => TP=17, FP=21, FN=1, TN=3, precision=44.7%.\n"
    "To keep TP rate >=50%: floor_amount <= 90,000.00 => TP=9, FP=12, FN=9, TN=12, precision=42.9%.\n"
    "At the highest value (90,000.00): TP=9, FP=12, FN=9, TN=12, precision=42.9%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_ROUND_TRIP_C4 = "Tool result for rule_sar_backtest:\n" + _RS_ROUND_TRIP_CLUSTER4

# ===========================================================================
# ARS_V45_1-10  Gap 4: rule_sar_backtest rebuild
# Format: ### header → RULE SWEEP verbatim → ONE insight sentence
# ===========================================================================

# ARS_V45_1: Funnel Account (new rule for V45)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for Funnel Account"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Funnel Account.",
     "tool_calls": [tc("ars_v45_1", "rule_sar_backtest",
                       {"risk_factor": "Funnel Account", "sweep_param": "floor_amount"})]},
    {"role": "tool", "tool_call_id": "ars_v45_1", "content": PC_RS_FUNNEL},
    {"role": "assistant", "content": (
        "### Rule Sweep — Funnel Account | floor_amount\n\n"
        + _RS_FUNNEL + "\n\n"
        "Raising the floor to $40K keeps 90% SAR catch rate while eliminating 12 FPs "
        "and lifting precision from 46.4% to 50.0%."
    )},
]})

# ARS_V45_2: Human Trafficking Indicators
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What is the SAR catch rate for Human Trafficking Indicators?"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Human Trafficking Indicators.",
     "tool_calls": [tc("ars_v45_2", "rule_sar_backtest",
                       {"risk_factor": "Human Trafficking Indicators", "sweep_param": "days_required"})]},
    {"role": "tool", "tool_call_id": "ars_v45_2", "content": PC_RS_HUMAN_TRAFFICKING},
    {"role": "assistant", "content": (
        "### Rule Sweep — Human Trafficking Indicators | days_required\n\n"
        + _RS_HUMAN_TRAFFICKING + "\n\n"
        "Raising days_required to 4 loses only 2 SARs while cutting 13 FPs, "
        "improving precision from 20.6% to 21.8%."
    )},
]})

# ARS_V45_3: Risky International Transfer
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run SAR backtest for Risky International Transfer"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Risky International Transfer.",
     "tool_calls": [tc("ars_v45_3", "rule_sar_backtest",
                       {"risk_factor": "Risky International Transfer"})]},
    {"role": "tool", "tool_call_id": "ars_v45_3", "content": PC_RS_RISKY_INT},
    {"role": "assistant", "content": (
        "### Rule Sweep — Risky International Transfer | floor_amount\n\n"
        + _RS_RISKY_INT + "\n\n"
        "Raising the floor to $20K keeps 92% of SARs (35 of 38) while cutting 24 FPs, "
        "improving precision from 23.3% to 25.7%."
    )},
]})

# ARS_V45_4: Elder Abuse filtered to Cluster 2
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show Elder Abuse SAR backtest for Cluster 2"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Elder Abuse filtered to Cluster 2.",
     "tool_calls": [tc("ars_v45_4", "rule_sar_backtest",
                       {"risk_factor": "Elder Abuse", "cluster": 2})]},
    {"role": "tool", "tool_call_id": "ars_v45_4", "content": PC_RS_ELDER_C2},
    {"role": "assistant", "content": (
        "### Rule Sweep — Elder Abuse | z_threshold [Cluster 2]\n\n"
        + _RS_ELDER_CLUSTER2 + "\n\n"
        "In Cluster 2, Elder Abuse precision is lower (11.7%) than the overall rule — "
        "raising z_threshold to 4 keeps 10 of 11 SARs while cutting 15 FPs."
    )},
]})

# ARS_V45_5: Activity Deviation (ACH) filtered to Cluster 4
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "SAR backtest for Activity Deviation ACH in Cluster 4"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Activity Deviation (ACH) filtered to Cluster 4.",
     "tool_calls": [tc("ars_v45_5", "rule_sar_backtest",
                       {"risk_factor": "Activity Deviation (ACH)", "cluster": 4})]},
    {"role": "tool", "tool_call_id": "ars_v45_5", "content": PC_RS_ACH_C4},
    {"role": "assistant", "content": (
        "### Rule Sweep — Activity Deviation (ACH) | z_threshold [Cluster 4]\n\n"
        + _RS_ACH_CLUSTER4 + "\n\n"
        "In Cluster 4, raising z_threshold from 5 to 7 cuts 27 FPs while losing only 2 SARs, "
        "lifting precision from 20.0% to 24.7%."
    )},
]})

# ARS_V45_6: Round-trip filtered to Cluster 4
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show Round-trip sweep for Cluster 4"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Round-trip filtered to Cluster 4.",
     "tool_calls": [tc("ars_v45_6", "rule_sar_backtest",
                       {"risk_factor": "Round-trip", "cluster": 4})]},
    {"role": "tool", "tool_call_id": "ars_v45_6", "content": PC_RS_ROUND_TRIP_C4},
    {"role": "assistant", "content": (
        "### Rule Sweep — Round-trip | floor_amount [Cluster 4]\n\n"
        + _RS_ROUND_TRIP_CLUSTER4 + "\n\n"
        "Round-trip is already the highest-precision rule in Cluster 4 at 42.9%; "
        "raising the floor to $70K maintains 44.7% precision with only 1 SAR missed."
    )},
]})

# ARS_V45_7: CTR Client (exact query from testing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR performance for CTR Client"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for CTR Client.",
     "tool_calls": [tc("ars_v45_7", "rule_sar_backtest",
                       {"risk_factor": "CTR Client"})]},
    {"role": "tool", "tool_call_id": "ars_v45_7", "content": PC_RS_CTR},
    {"role": "assistant", "content": (
        "### Rule Sweep — CTR Client | floor_amount\n\n"
        + _RS_CTR + "\n\n"
        "CTR Client has the highest FP count in the system (348); "
        "raising the floor to $15K maintains 92% SAR coverage while eliminating 43 FPs."
    )},
]})

# ARS_V45_8: Structuring (Incoming Cash)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Analyze the Structuring Incoming Cash rule"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Structuring (Incoming Cash).",
     "tool_calls": [tc("ars_v45_8", "rule_sar_backtest",
                       {"risk_factor": "Structuring (Incoming Cash)"})]},
    {"role": "tool", "tool_call_id": "ars_v45_8", "content": PC_RS_STRUCTURING},
    {"role": "assistant", "content": (
        "### Rule Sweep — Structuring (Incoming Cash) | daily_floor\n\n"
        + _RS_STRUCTURING_IN + "\n\n"
        "The $3K floor catches all 46 SARs; pushing daily_floor above $5,500 drops "
        "SAR coverage below 57%, making $3K-$5K the recommended operating range."
    )},
]})

# ARS_V45_9: Multi-step — list_rules first, then rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule should I backtest first? Then show me its SAR performance."},
    {"role": "assistant",
     "content": "Calling list_rules to identify the best candidate for backtest.",
     "tool_calls": [tc("ars_v45_9a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "ars_v45_9a", "content": PC_LIST_H},
    {"role": "assistant",
     "content": (
         "Based on the rule list, Funnel Account has the highest precision at 46.4% — "
         "making it the best candidate for threshold optimization. Let me run its backtest."
     ),
     "tool_calls": [tc("ars_v45_9b", "rule_sar_backtest",
                       {"risk_factor": "Funnel Account", "sweep_param": "floor_amount"})]},
    {"role": "tool", "tool_call_id": "ars_v45_9b", "content": PC_RS_FUNNEL},
    {"role": "assistant", "content": (
        "### Rule Sweep — Funnel Account | floor_amount\n\n"
        + _RS_FUNNEL + "\n\n"
        "Raising the floor from $25K to $40K keeps 92% SAR catch rate while "
        "improving precision from 46.4% to 50.0% — 1 in 2 alerts becomes a confirmed SAR."
    )},
]})

# ARS_V45_10: Velocity Single (different sweep param — ratio_tolerance)
_RS_VELOCITY_SINGLE_RATIO = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K\n"
    "Sweep parameter: ratio_tolerance - Allowed deviation between in and out amounts (currently 10%)\n"
    "Current value: 10\n"
    "Labeled population: 250 customers (TP+FN pool=49 SAR, FP+TN pool=201 non-SAR, precision=19.6%)\n"
    "\n"
    "At the lowest value (1): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).\n"
    "At current condition (10): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).\n"
    "To keep TP rate >=90%: ratio_tolerance <= 15 => TP=49, FP=201, FN=0, TN=0, precision=19.6%.\n"
    "To keep TP rate >=50%: ratio_tolerance <= 20 => TP=49, FP=201, FN=0, TN=0, precision=19.6%.\n"
    "At the highest value (25): TP=49, FP=201, FN=0, TN=0, precision=19.6%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)
PC_RS_VS_RATIO = "Tool result for rule_sar_backtest:\n" + _RS_VELOCITY_SINGLE_RATIO

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show Velocity Single SAR backtest using ratio tolerance"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Velocity Single with ratio_tolerance sweep.",
     "tool_calls": [tc("ars_v45_10", "rule_sar_backtest",
                       {"risk_factor": "Velocity Single", "sweep_param": "ratio_tolerance"})]},
    {"role": "tool", "tool_call_id": "ars_v45_10", "content": PC_RS_VS_RATIO},
    {"role": "assistant", "content": (
        "### Rule Sweep — Velocity Single | ratio_tolerance\n\n"
        + _RS_VELOCITY_SINGLE_RATIO + "\n\n"
        "ratio_tolerance does not differentiate SAR from non-SAR in this population — "
        "all 49 SARs are caught across the full sweep range (1–25%)."
    )},
]})

# ===========================================================================
# PRE-COMPUTED SAR BACKTEST blocks for AT_V45
# ===========================================================================

_SAR_BT_BIZ_AMT_V45 = (
    "=== PRE-COMPUTED SAR BACKTEST ===\n"
    "### SAR Catch Rate — Business / AVG_TRXN_AMT\n\n"
    "**Population:** 485 alerted customers | **SARs:** 112 (23.1% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold ($250): **112 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **$1,250** (2 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **$2,750** (101 of 112 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **$4,250** (90 of 112 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **$8,750** (57 of 112 caught)\n"
    "- At highest threshold ($18,750): **8 caught**, 104 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_BT_BIZ_AMT = "Tool result for sar_backtest:\n" + _SAR_BT_BIZ_AMT_V45

_SAR_BT_IND_WEEK_V45 = (
    "=== PRE-COMPUTED SAR BACKTEST ===\n"
    "### SAR Catch Rate — Individual / AVG_TRXNS_WEEK\n\n"
    "**Population:** 4,515 alerted customers | **SARs:** 466 (10.3% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold (0.3): **466 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **1.3** (4 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **2.3** (420 of 466 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **4.3** (373 of 466 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **9.3** (234 of 466 caught)\n"
    "- At highest threshold (21.3): **14 caught**, 452 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_BT_IND_WEEK = "Tool result for sar_backtest:\n" + _SAR_BT_IND_WEEK_V45

_SAR_BT_BIZ_MONTHLY_V45 = (
    "=== PRE-COMPUTED SAR BACKTEST ===\n"
    "### SAR Catch Rate — Business / TRXN_AMT_MONTHLY\n\n"
    "**Population:** 485 alerted customers | **SARs:** 112 (23.1% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold ($1,200): **112 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **$6,200** (3 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **$16,200** (101 of 112 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **$26,200** (90 of 112 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **$56,200** (57 of 112 caught)\n"
    "- At highest threshold ($96,200): **9 caught**, 103 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_BT_BIZ_MONTHLY = "Tool result for sar_backtest:\n" + _SAR_BT_BIZ_MONTHLY_V45

# ===========================================================================
# AT_V45_1-5  Gap 3: sar_backtest proper insight
# Pattern: ### header → PRE-COMPUTED block → insight sentence referencing
#   specific threshold value AND catch-rate percentage from the block.
# No "See chart/table below." No empty response.
# ===========================================================================

# AT_V45_1: "How well do we catch SARs" (exact failing query from testing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How well do we catch SARs at current thresholds?"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Business customers by weekly transactions.",
     "tool_calls": [tc("at_v45_1", "sar_backtest",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v45_1",
     "content": "Tool result for sar_backtest:\n" + (
         "=== PRE-COMPUTED SAR BACKTEST ===\n"
         "### SAR Catch Rate — Business / AVG_TRXNS_WEEK\n\n"
         "**Population:** 485 alerted customers | **SARs:** 112 (23.1% SAR filing rate)\n\n"
         "**Sweep Results**\n"
         "- At lowest threshold (0.8): **112 SARs caught** (100%), 0 missed\n"
         "- SARs first missed at threshold **2.8** (2 missed)\n"
         "- To keep ≥90% SAR catch rate: threshold ≤ **3.8** (101 of 112 caught)\n"
         "- To keep ≥80% SAR catch rate: threshold ≤ **6.8** (90 of 112 caught)\n"
         "- To keep ≥50% SAR catch rate: threshold ≤ **12.8** (57 of 112 caught)\n"
         "- At highest threshold (24.8): **6 caught**, 106 missed\n\n"
         "*(Detailed sweep chart shown below.)*\n"
         "=== END PRE-COMPUTED SAR BACKTEST ==="
     )},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Business | Avg Weekly Transactions\n\n"
        "=== PRE-COMPUTED SAR BACKTEST ===\n"
        "### SAR Catch Rate — Business / AVG_TRXNS_WEEK\n\n"
        "**Population:** 485 alerted customers | **SARs:** 112 (23.1% SAR filing rate)\n\n"
        "**Sweep Results**\n"
        "- At lowest threshold (0.8): **112 SARs caught** (100%), 0 missed\n"
        "- SARs first missed at threshold **2.8** (2 missed)\n"
        "- To keep ≥90% SAR catch rate: threshold ≤ **3.8** (101 of 112 caught)\n"
        "- To keep ≥80% SAR catch rate: threshold ≤ **6.8** (90 of 112 caught)\n"
        "- To keep ≥50% SAR catch rate: threshold ≤ **12.8** (57 of 112 caught)\n"
        "- At highest threshold (24.8): **6 caught**, 106 missed\n\n"
        "*(Detailed sweep chart shown below.)*\n"
        "=== END PRE-COMPUTED SAR BACKTEST ===\n\n"
        "At the current threshold, Business customers have a 100% SAR catch rate; "
        "maintaining ≥90% SAR coverage requires keeping AVG_TRXNS_WEEK at or below **3.8** transactions/week."
    )},
]})

# AT_V45_2: "SAR hit rate" for Business by avg transaction amount
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What is our SAR hit rate for Business customers?"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Business customers by average transaction amount.",
     "tool_calls": [tc("at_v45_2", "sar_backtest",
                       {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "tool_call_id": "at_v45_2", "content": PC_SAR_BT_BIZ_AMT},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Business | Avg Transaction Amount\n\n"
        + _SAR_BT_BIZ_AMT_V45 + "\n\n"
        "Business customers have a 23.1% SAR filing rate; "
        "maintaining ≥90% SAR coverage requires keeping AVG_TRXN_AMT at or below **$2,750** per transaction."
    )},
]})

# AT_V45_3: "SAR performance at current thresholds" — Individual by weekly txns
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR performance at current thresholds for Individual customers"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Individual customers by weekly transactions.",
     "tool_calls": [tc("at_v45_3", "sar_backtest",
                       {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v45_3", "content": PC_SAR_BT_IND_WEEK},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Individual | Avg Weekly Transactions\n\n"
        + _SAR_BT_IND_WEEK_V45 + "\n\n"
        "Individual customers have a 10.3% SAR filing rate; "
        "maintaining ≥90% SAR coverage requires keeping AVG_TRXNS_WEEK at or below **2.3** transactions/week."
    )},
]})

# AT_V45_4: "What percentage of SARs are we filing" — Business by monthly amount
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What percentage of SARs are we catching for Business customers by monthly amount?"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Business customers by monthly transaction amount.",
     "tool_calls": [tc("at_v45_4", "sar_backtest",
                       {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "at_v45_4", "content": PC_SAR_BT_BIZ_MONTHLY},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Business | Monthly Transaction Volume\n\n"
        + _SAR_BT_BIZ_MONTHLY_V45 + "\n\n"
        "At the current floor ($1,200/month), all 112 Business SARs are caught (100%); "
        "maintaining ≥90% coverage requires holding TRXN_AMT_MONTHLY at or below **$16,200**."
    )},
]})

# AT_V45_5: "How well does the system identify suspicious customers"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How well does the system identify suspicious Individual customers by transaction frequency?"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Individual customers by weekly transactions.",
     "tool_calls": [tc("at_v45_5", "sar_backtest",
                       {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v45_5", "content": PC_SAR_BT_IND_WEEK},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Individual | Avg Weekly Transactions\n\n"
        + _SAR_BT_IND_WEEK_V45 + "\n\n"
        "Transaction frequency is a moderate SAR indicator for Individual customers — "
        "keeping AVG_TRXNS_WEEK ≤ 2.3 preserves 90% SAR coverage (420 of 466 caught) "
        "from a population with a 10.3% SAR filing rate."
    )},
]})

# ===========================================================================
# PRE-COMPUTED CLUSTER RULE SUMMARY blocks for AM_V45
# ===========================================================================

_CRS_CLUSTER4 = (
    "=== PRE-COMPUTED CLUSTER RULE SUMMARY (copy this verbatim) ===\n"
    "Cluster 4 — 110 customers in rule alert data\n"
    "SAR/FP performance for all rules filtered to this cluster:\n"
    "NOTE: alerts=0 means no alerts from this rule for customers in this cluster.\n"
    "  Activity Deviation (ACH): alerts=110, SAR=22, FP=88, precision=20.0%\n"
    "  Activity Deviation (Check): alerts=42, SAR=8, FP=34, precision=19.0%\n"
    "  Elder Abuse: alerts=31, SAR=5, FP=26, precision=16.1%\n"
    "  Velocity Single: alerts=58, SAR=14, FP=44, precision=24.1%\n"
    "  Detect Excessive Transaction Activity: alerts=88, SAR=12, FP=76, precision=13.6%\n"
    "  Structuring (Incoming Cash): alerts=44, SAR=9, FP=35, precision=20.5%\n"
    "  Structuring (Outgoing Cash): alerts=37, SAR=7, FP=30, precision=18.9%\n"
    "  CTR Client: alerts=92, SAR=11, FP=81, precision=12.0%\n"
    "  Burst in Originator Activity: alerts=55, SAR=13, FP=42, precision=23.6%\n"
    "  Burst in Beneficiary Activity: alerts=48, SAR=10, FP=38, precision=20.8%\n"
    "  Risky International Transfer: alerts=42, SAR=12, FP=30, precision=28.6%\n"
    "  Activity Deviation (Wire): alerts=38, SAR=8, FP=30, precision=21.1%\n"
    "  Velocity Multiple: alerts=61, SAR=18, FP=43, precision=29.5%\n"
    "  Funnel Account: alerts=29, SAR=9, FP=20, precision=31.0%\n"
    "  Round-trip: alerts=42, SAR=18, FP=24, precision=42.9%\n"
    "  Human Trafficking Indicators: alerts=22, SAR=5, FP=17, precision=22.7%\n"
    "=== END CLUSTER RULE SUMMARY ==="
)
PC_CRS_CLUSTER4 = "Tool result for cluster_rule_summary:\n" + _CRS_CLUSTER4

_CRS_CLUSTER1 = (
    "=== PRE-COMPUTED CLUSTER RULE SUMMARY (copy this verbatim) ===\n"
    "Cluster 1 — 121 customers in rule alert data\n"
    "SAR/FP performance for all rules filtered to this cluster:\n"
    "NOTE: alerts=0 means no alerts from this rule for customers in this cluster.\n"
    "  Activity Deviation (ACH): alerts=48, SAR=5, FP=43, precision=10.4%\n"
    "  Activity Deviation (Check): alerts=31, SAR=3, FP=28, precision=9.7%\n"
    "  Elder Abuse: alerts=24, SAR=2, FP=22, precision=8.3%\n"
    "  Velocity Single: alerts=29, SAR=4, FP=25, precision=13.8%\n"
    "  Detect Excessive Transaction Activity: alerts=52, SAR=4, FP=48, precision=7.7%\n"
    "  Structuring (Incoming Cash): alerts=35, SAR=5, FP=30, precision=14.3%\n"
    "  Structuring (Outgoing Cash): alerts=28, SAR=3, FP=25, precision=10.7%\n"
    "  CTR Client: alerts=61, SAR=6, FP=55, precision=9.8%\n"
    "  Burst in Originator Activity: alerts=22, SAR=3, FP=19, precision=13.6%\n"
    "  Burst in Beneficiary Activity: alerts=19, SAR=2, FP=17, precision=10.5%\n"
    "  Risky International Transfer: alerts=18, SAR=4, FP=14, precision=22.2%\n"
    "  Activity Deviation (Wire): alerts=20, SAR=2, FP=18, precision=10.0%\n"
    "  Velocity Multiple: alerts=24, SAR=3, FP=21, precision=12.5%\n"
    "  Funnel Account: alerts=15, SAR=2, FP=13, precision=13.3%\n"
    "  Round-trip: alerts=12, SAR=2, FP=10, precision=16.7%\n"
    "  Human Trafficking Indicators: alerts=14, SAR=2, FP=12, precision=14.3%\n"
    "=== END CLUSTER RULE SUMMARY ==="
)
PC_CRS_CLUSTER1 = "Tool result for cluster_rule_summary:\n" + _CRS_CLUSTER1

# ===========================================================================
# AM_V45_1-3  Gap 1: multi-step cluster → SAR chaining
# Pattern:
#   [PREVIOUS CLUSTERING RESULT] in context
#   User asks about SARs for a cluster attribute (highest income = Cluster 4)
#   Model: (a) reads cluster number from stats block
#          (b) calls cluster_rule_summary with that cluster number
#          (c) reports SAR/FP from the result
# ===========================================================================

# AM_V45_1: "How many SARs caught for highest income cluster?"
# Cluster 4 = highest income ($312,000) → call cluster_rule_summary(cluster=4)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": prev_context(_CLUSTER_STATS) + "\nHow many SARs were caught in the highest income cluster?"},
    {"role": "assistant",
     "content": (
         "From the clustering result, Cluster 4 has the highest income at $312,000. "
         "Calling cluster_rule_summary to get SAR performance for Cluster 4."
     ),
     "tool_calls": [tc("am_v45_1", "cluster_rule_summary", {"cluster": 4})]},
    {"role": "tool", "tool_call_id": "am_v45_1", "content": PC_CRS_CLUSTER4},
    {"role": "assistant", "content": (
        _CRS_CLUSTER4 + "\n\n"
        "In Cluster 4 (the highest income segment at $312,000 average), the top-performing rules are "
        "Round-trip (42.9% precision, 18 SAR caught), Funnel Account (31.0%, 9 SAR), and "
        "Velocity Multiple (29.5%, 18 SAR). "
        "The cluster generates 110 alert records across all rules."
    )},
]})

# AM_V45_2: "Which rules perform best in the highest activity cluster?"
# Cluster 4 = highest weekly transactions (22.6) → cluster_rule_summary(cluster=4)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": prev_context(_CLUSTER_STATS) + "\nWhich rules perform best in the most active cluster?"},
    {"role": "assistant",
     "content": (
         "From the clustering result, Cluster 4 is the most active segment at 22.6 average weekly transactions. "
         "Calling cluster_rule_summary to see rule performance for Cluster 4."
     ),
     "tool_calls": [tc("am_v45_2", "cluster_rule_summary", {"cluster": 4})]},
    {"role": "tool", "tool_call_id": "am_v45_2", "content": PC_CRS_CLUSTER4},
    {"role": "assistant", "content": (
        _CRS_CLUSTER4 + "\n\n"
        "In Cluster 4 (most active — 22.6 weekly transactions), the highest-precision rules are:\n\n"
        "1. **Round-trip** - 42.9% (18 SAR, 24 FP)\n"
        "2. **Funnel Account** - 31.0% (9 SAR, 20 FP)\n"
        "3. **Velocity Multiple** - 29.5% (18 SAR, 43 FP)\n\n"
        "High-velocity customers in Cluster 4 are most often flagged by wire-based and counterparty-pattern rules."
    )},
]})

# AM_V45_3: "SAR performance in the lowest activity cluster?"
# Cluster 1 = lowest weekly transactions (3.2) → cluster_rule_summary(cluster=1)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": prev_context(_CLUSTER_STATS) + "\nHow do the rules perform for the least active cluster?"},
    {"role": "assistant",
     "content": (
         "From the clustering result, Cluster 1 is the least active segment at 3.2 average weekly transactions. "
         "Calling cluster_rule_summary to see rule performance for Cluster 1."
     ),
     "tool_calls": [tc("am_v45_3", "cluster_rule_summary", {"cluster": 1})]},
    {"role": "tool", "tool_call_id": "am_v45_3", "content": PC_CRS_CLUSTER1},
    {"role": "assistant", "content": (
        _CRS_CLUSTER1 + "\n\n"
        "In Cluster 1 (least active — 3.2 weekly transactions), all rules have low precision: "
        "the best performer is Risky International Transfer at 22.2% (4 SAR, 14 FP), "
        "while DETA (7.7%) and Elder Abuse (8.3%) have the weakest SAR signal for this segment."
    )},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _has_stale_sar_backtest(ex):
    """Remove sar_backtest examples where the final assistant response (after
    the tool call) contains the stale phrase 'See chart/table below.' or the
    chart-injection phrase '*(Detailed sweep chart shown below.)*'."""
    found_sar_call = False
    for m in ex["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                if call.get("function", {}).get("name") == "sar_backtest":
                    found_sar_call = True
        if (found_sar_call
                and m.get("role") == "assistant"
                and not m.get("tool_calls")
                and isinstance(m.get("content"), str)):
            content = m["content"]
            if ("See chart/table below." in content
                    or "*(Detailed sweep chart shown below.)*" in content):
                return True
    return False


def _has_stale_rule_sar_backtest(ex):
    """Remove examples where a tool-role message contains the OLD block header
    'PRE-COMPUTED RULE SAR BACKTEST' (NOT 'PRE-COMPUTED RULE SWEEP').
    The V44 filter was too broad — it removed ALL rule_sar_backtest examples.
    This surgical filter only removes the truly stale ones."""
    for m in ex["messages"]:
        if (m.get("role") == "tool"
                and isinstance(m.get("content"), str)
                and "PRE-COMPUTED RULE SAR BACKTEST" in m["content"]
                and "PRE-COMPUTED RULE SWEEP" not in m["content"]):
            return True
    return False


# ---------------------------------------------------------------------------
# Combine V44 base (minus stale) + V45 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V45_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V45] V45-only: {V45_ONLY_PATH.name} ({len(examples)} examples)")

    if V44_BASE_PATH.exists():
        v44_base = []
        with open(V44_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v44_base.append(json.loads(line))
        print(f"[V45] Loaded {len(v44_base)} base examples from {V44_BASE_PATH.name}")

        filtered = v44_base
        for fn, label in [
            (_has_stale_sar_backtest,       "stale sar_backtest (See chart/table below.)"),
            (_has_stale_rule_sar_backtest,   "stale rule_sar_backtest (old RULE SAR BACKTEST header)"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            after = len(filtered)
            print(f"[V45] Removed {before - after} {label} examples")

        all_examples = filtered + examples
        with open(V45_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V45] Combined: {V45_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V45] WARNING: V44 base not found at {V44_BASE_PATH}")


if __name__ == "__main__":
    main()

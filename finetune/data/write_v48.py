"""
V48 training examples (2026-05-15).

Targets:

  AT_V48_1-4   Individual segment verbatim copy (Business works, Individual gives thin
               "Results shown below." response with no cluster breakdown text).
               Also covers "reduce FPs by treating clusters differently" phrasing that
               V47 routed to a hallucinated 'unsupervised' tool instead of
               cluster_threshold_analysis.

  ARL_V48_1-5  Short follow-up: "and the lowest?" / "how about the highest?" / "lowest?"
               after a precision exchange.  Model says "I need the data" instead of
               re-sorting the [PREVIOUS RULE LIST] already in context.
               Fix: 5 multi-turn examples where the final assistant turn re-sorts
               without calling list_rules again.

  AC_V48_1-2   Cluster short follow-up: "and the youngest?" / "and the fewest?" after a
               cluster exchange.  Same root cause as ARL — model loses context on bare
               follow-up words.  Two examples re-using prior ds_cluster_analysis result.

  ARL_V48_6-7  Gap 3: "what are all the AML rules?" after a precision or parameter
               exchange returns contaminated precision output instead of full list.
               Two examples anchoring the correct full-list response.

  ARL_V48_8-10 Parameter filter multi-turn: "days_required" follow-up gives definition
               instead of rule list; "what about X?" after a prior parameter query
               should filter from existing list_rules context.

  ARS_V48_1-2  sar_backtest (segment-level): model generates canned placeholder instead
               of copying the PRE-COMPUTED SAR BACKTEST block verbatim.
               Two examples with Business/AVG_TRXNS_WEEK and Individual/TRXN_AMT_MONTHLY.

  ARS_V48_3-4  rule_sar_backtest: same canned-phrase regression.
               Two fresh examples reusing existing RULE SWEEP datasets.

Filters applied in main():
  _has_sar_backtest_canned  : removes sar_backtest / rule_sar_backtest examples where
                              the final assistant response contains a canned placeholder
                              phrase without the PRE-COMPUTED block.
  _has_hardcoded_rule_count : removes examples where the assistant states "11" or "16"
                              as a hardcoded AML rule count without a list_rules call.

Base: aria_train_combined_v47_full.jsonl (549)
"""

import json, pathlib, re, sys

DATA_DIR      = pathlib.Path(__file__).parent
V47_FULL_PATH = DATA_DIR / "aria_train_combined_v47_full.jsonl"
V48_ONLY_PATH = DATA_DIR / "aria_train_v48.jsonl"
V48_FULL_PATH = DATA_DIR / "aria_train_combined_v48_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM           # noqa: E402
from write_v42 import tc, _CLUSTER_STATS                              # noqa: E402
from write_v44 import (                                                # noqa: E402
    _RS_ELDER_ABUSE, PC_RS_ELDER,
    _RS_ACH, PC_RS_ACH,
)
from write_v45 import _LIST_H, PC_LIST_H                              # noqa: E402
from write_v45 import _CLUSTER_STATS_IND, _PC_CLUSTER_IND, _T1_IND_RESPONSE  # noqa: E402
from write_v46 import _PC_CLUSTER_BIZ, _T1_BIZ_RESPONSE              # noqa: E402
from write_v47 import (                                                # noqa: E402
    _AT_C_BLOCK, _AT_C_RESPONSE, _PC_AT_C,
    _AT_A_BLOCK, _AT_A_RESPONSE, _PC_AT_A,
    _ARL_T1_HIGHEST,
)

examples = []

# ===========================================================================
# Shared constants
# ===========================================================================

_LIST_H_HEADER = "### AML Rule Performance Overview\n\n" + _LIST_H

# Bottom-3 precision from _LIST_H (CTR Client 13.0%, Detect Excessive 13.7%, Elder Abuse 14.9%)
_ARL_T1_LOWEST = (
    _LIST_H_HEADER + "\n\n"
    "The rules with the lowest precision are:\n"
    "- CTR Client (precision=13.0%)\n"
    "- Detect Excessive Transaction Activity (precision=13.7%)\n"
    "- Elder Abuse (precision=14.9%)"
)

# Full rule list response (for Gap 3 — "what are all the AML rules?" after context contamination)
_ARL_FULL_LIST = (
    "The system monitors the following AML rules: "
    "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, Velocity Single, "
    "Detect Excessive Transaction Activity, Structuring (Incoming Cash), "
    "Structuring (Outgoing Cash), CTR Client, Burst in Originator Activity, "
    "Burst in Beneficiary Activity, Risky International Transfer, Activity Deviation (Wire), "
    "Velocity Multiple, Funnel Account, Round-trip, Human Trafficking Indicators. "
    "See the chart for the full SAR/FP breakdown."
)

# ===========================================================================
# AT_V48_1-4  Individual verbatim copy + "reduce FPs" phrasing fix
# ===========================================================================

# AT_V48_1: "Run cluster-adaptive threshold analysis for Individual customers"
# Model must copy _AT_C block verbatim and add insight — NOT just say "Results shown below."
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run cluster-adaptive threshold analysis for Individual customers"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Individual customers.",
     "tool_calls": [tc("at_v48_1", "cluster_threshold_analysis",
                       {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v48_1", "content": _PC_AT_C},
    {"role": "assistant", "content": _AT_C_RESPONSE},
]})

# AT_V48_2: "Show per-cluster thresholds for Individual customers" — different phrasing
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show per-cluster adaptive thresholds for Individual customers"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Individual customers.",
     "tool_calls": [tc("at_v48_2", "cluster_threshold_analysis",
                       {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v48_2", "content": _PC_AT_C},
    {"role": "assistant", "content": _AT_C_RESPONSE},
]})

# ── Dataset AT-D: Individual / TRXN_AMT_MONTHLY ───────────────────────────────
_AT_D_CONTENT = """\
Segment: **Individual** | Column: TRXN_AMT_MONTHLY (Monthly Txn Volume) | Clusters: 4 | Target SAR catch: ≥90%

**Cluster 1 — High Volume** (31 customers, SAR pool: 9)
- Uniform $8,200: TP=9, FP=20, TP rate=100.0%, precision=31.0%
- Recommended $14,500: TP=9, FP=8, TP rate=100.0%, precision=52.9% (-12 FP, +0 SAR)

**Cluster 2 — Mid-High** (1,823 customers, SAR pool: 421)
- Uniform $8,200: TP=393, FP=1,237, TP rate=93.3%, precision=24.1%
- Recommended $8,600: TP=389, FP=1,201, TP rate=92.4%, precision=24.5% (-36 FP, -4 SAR)

**Cluster 3 — Mid-Low** (52 customers, SAR pool: 14)
- Uniform $8,200: TP=13, FP=35, TP rate=92.9%, precision=27.1%
- Recommended $7,800: TP=14, FP=37, TP rate=100.0%, precision=27.5% (+2 FP, +1 SAR)

**Cluster 4 — Low Volume** (11 customers, SAR pool: 3)
- Uniform $8,200: TP=3, FP=7, TP rate=100.0%, precision=30.0%
- Recommended $11,400: TP=3, FP=3, TP rate=100.0%, precision=50.0% (-4 FP, +0 SAR)

**ADAPTIVE SENSITIVITY SUMMARY**
- Uniform $8,200 applied to all Individual: TP=418, FP=1,299, TP rate=93.3%
- Cluster-adaptive thresholds: TP=415, FP=1,249, TP rate=92.6%
- Net change: **-50 FP**, -3 SARs (92.6% SAR retention)"""

_AT_D_BLOCK = (
    "=== PRE-COMPUTED CLUSTER THRESHOLD ANALYSIS (copy verbatim, do not alter numbers) ===\n"
    + _AT_D_CONTENT + "\n"
    "=== END CLUSTER THRESHOLD ANALYSIS ==="
)
_PC_AT_D = "Tool result for cluster_threshold_analysis:\n" + _AT_D_BLOCK

_AT_D_RESPONSE = (
    "### Cluster-Adaptive Threshold Analysis — Individual | Monthly Txn Volume\n\n"
    + _AT_D_BLOCK + "\n\n"
    "Cluster-adaptive thresholds cut 50 false positives while retaining 92.6% SAR coverage "
    "— Cluster 2 (Mid-High, 1,823 customers) drives the bulk of the FP reduction at -36."
)

# AT_V48_3: Individual / TRXN_AMT_MONTHLY
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show cluster thresholds for Individual customers using monthly transaction volume"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Individual / TRXN_AMT_MONTHLY.",
     "tool_calls": [tc("at_v48_3", "cluster_threshold_analysis",
                       {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "at_v48_3", "content": _PC_AT_D},
    {"role": "assistant", "content": _AT_D_RESPONSE},
]})

# AT_V48_4: "What strategies exist to reduce false positives using customer segmentation?"
# V47 routed this phrasing to a hallucinated "unsupervised" tool — fix by anchoring it to
# cluster_threshold_analysis with Business default.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": "What strategies exist to reduce false positives using customer segmentation?"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v48_4", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v48_4", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
]})

# ===========================================================================
# ARL_V48_1-5  Short follow-up: "and the lowest/highest?" — re-sort from prior data
#
# Critical: the final assistant turn has NO tool_calls.
# The model re-sorts from the PRE-COMPUTED RULE LIST already in the tool message
# from turn 1 without calling list_rules again.
#
# Precision order from _LIST_H:
#   DESC top 3: Funnel Account 46.4%, Risky International Transfer 22.9%, Round-trip 22.4%
#   ASC  bot 3: CTR Client 13.0%, Detect Excessive 13.7%, Elder Abuse 14.9%
# ===========================================================================

_ARL_BOT3 = (
    "The rules with the lowest precision are:\n"
    "- CTR Client (precision=13.0%)\n"
    "- Detect Excessive Transaction Activity (precision=13.7%)\n"
    "- Elder Abuse (precision=14.9%)"
)

_ARL_TOP3 = (
    "The rules with the highest precision are:\n"
    "- Funnel Account (precision=46.4%)\n"
    "- Risky International Transfer (precision=22.9%)\n"
    "- Round-trip (precision=22.4%)"
)

# ARL_V48_1: highest → "and the lowest?" (bare follow-up)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_1a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_1a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_HIGHEST},
    {"role": "user",   "content": "and the lowest?"},
    {"role": "assistant", "content": _ARL_BOT3},
]})

# ARL_V48_2: lowest → "and the highest?" (bare follow-up)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the lowest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_2a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_2a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_LOWEST},
    {"role": "user",   "content": "and the highest?"},
    {"role": "assistant", "content": _ARL_TOP3},
]})

# ARL_V48_3: top 3 → "how about the bottom 3?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me the top 3 rules by precision"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_3a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_3a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_HIGHEST},
    {"role": "user",   "content": "how about the bottom 3?"},
    {"role": "assistant", "content": _ARL_BOT3},
]})

# ARL_V48_4: highest → "lowest?" (single word)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What are the most precise AML rules?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_4a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_4a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_HIGHEST},
    {"role": "user",   "content": "lowest?"},
    {"role": "assistant", "content": _ARL_BOT3},
]})

# ARL_V48_5: lowest → "highest?" (single word)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "List the AML rules with the worst precision"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_5a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_5a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_LOWEST},
    {"role": "user",   "content": "highest?"},
    {"role": "assistant", "content": _ARL_TOP3},
]})

# ===========================================================================
# AC_V48_1-2  Cluster short follow-up — re-use prior ds_cluster_analysis result
#
# Business cluster data (_CLUSTER_STATS):
#   Cluster 1 (n=121): Account Age 4.1 yrs  ← youngest / fewest customers? no
#   Cluster 2 (n=98):  Account Age 7.3 yrs  ← fewest customers
#   Cluster 3 (n=156): Account Age 5.8 yrs  ← most customers
#   Cluster 4 (n=110): Account Age 9.2 yrs  ← oldest
# ===========================================================================

# AC_V48_1: clustering → "which cluster is oldest?" → "and the youngest?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Show me the behavioral segments for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v48_1a", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v48_1a", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
    {"role": "user",   "content": "which cluster is the oldest in account age?"},
    {"role": "assistant",
     "content": "Cluster 4 is the oldest with an average account age of **9.2 years**."},
    {"role": "user",   "content": "and the youngest?"},
    {"role": "assistant",
     "content": "Cluster 1 is the youngest with an average account age of **4.1 years**."},
]})

# AC_V48_2: clustering → "which cluster has the most customers?" → "and the fewest?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Cluster Business customers into behavioral groups"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v48_2a", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v48_2a", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
    {"role": "user",   "content": "which cluster has the most customers?"},
    {"role": "assistant",
     "content": "Cluster 3 has the most customers with **156 accounts**."},
    {"role": "user",   "content": "and the fewest?"},
    {"role": "assistant",
     "content": "Cluster 2 has the fewest customers with **98 accounts**."},
]})

# ===========================================================================
# ARL_V48_6-7  Gap 3: "what are all the AML rules?" after precision/parameter context
#
# After a precision or parameter exchange, the model contaminates the full-list
# response with precision context.  These examples anchor the correct clean response.
# ===========================================================================

# ARL_V48_6: after parameter filter → "what are all the AML rules available?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What rules have the z_threshold parameter?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_6a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_6a", "content": PC_LIST_H},
    {"role": "assistant",
     "content": (
         "The rules that have the z_threshold parameter are: "
         "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, "
         "Activity Deviation (Wire)."
     )},
    {"role": "user",   "content": "what are all the AML rules available?"},
    {"role": "assistant", "content": _ARL_FULL_LIST},
]})

# ARL_V48_7: after lowest precision exchange → "what are all the AML rules?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the lowest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v48_7a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_7a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_LOWEST},
    {"role": "user",   "content": "what are all the AML rules?"},
    {"role": "assistant", "content": _ARL_FULL_LIST},
]})

# ===========================================================================
# ARL_V48_8-10  Parameter filter — days_required / time_window + multi-turn
#
# Rule 29: call list_rules, then filter by sweep_params.  Never answer from memory.
# From _LIST_H sweep_params:
#   days_required : Structuring (Incoming Cash), Structuring (Outgoing Cash),
#                   Human Trafficking Indicators
#   time_window   : Detect Excessive Transaction Activity
#   ratio_tolerance: Velocity Single
# ===========================================================================

# ARL_V48_8: "What rules have the days_required parameter?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What rules have the days_required parameter?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep_params for all rules.",
     "tool_calls": [tc("arl_v48_8a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_8a", "content": PC_LIST_H},
    {"role": "assistant",
     "content": (
         "The rules that have the days_required parameter are: "
         "Structuring (Incoming Cash), Structuring (Outgoing Cash), "
         "and Human Trafficking Indicators."
     )},
]})

# ARL_V48_9: "Which rules use time_window?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules use the time_window parameter?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep_params for all rules.",
     "tool_calls": [tc("arl_v48_9a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_9a", "content": PC_LIST_H},
    {"role": "assistant",
     "content": (
         "The only rule that uses the time_window parameter is "
         "Detect Excessive Transaction Activity."
     )},
]})

# ARL_V48_10: multi-turn parameter filter — "z_threshold" → "what about days_required?"
# Model must NOT give a definition of days_required — it must filter from existing context.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What rules have the z_threshold parameter?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep_params for all rules.",
     "tool_calls": [tc("arl_v48_10a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v48_10a", "content": PC_LIST_H},
    {"role": "assistant",
     "content": (
         "The rules that have the z_threshold parameter are: "
         "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, "
         "Activity Deviation (Wire)."
     )},
    {"role": "user",   "content": "what about days_required?"},
    {"role": "assistant",
     "content": (
         "The rules that have the days_required parameter are: "
         "Structuring (Incoming Cash), Structuring (Outgoing Cash), "
         "and Human Trafficking Indicators."
     )},
]})

# ===========================================================================
# ARS_V48_1-2  sar_backtest — verbatim PRE-COMPUTED SAR BACKTEST block
#
# Model generates a canned placeholder instead of copying the block.
# After stripping, only header + insight are visible; chart carries the data.
# ===========================================================================

_SAR_BT_BIZ = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
    "### SAR Backtest — Business / avg_num_trxns\n\n"
    "**Population:** 485 alerted customers | **SARs:** 112 (23.1% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold (1.21): **112 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **4.21** (1 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **8.21** (101 of 112 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **13.21** (90 of 112 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **23.21** (57 of 112 caught)\n"
    "- At highest threshold (49.21): **6 caught**, 106 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
_PC_SAR_BT_BIZ = "Tool result for sar_backtest:\n" + _SAR_BT_BIZ

_SAR_BT_BIZ_RESPONSE = (
    "### SAR Catch Rate — Business | Avg Weekly Transactions\n\n"
    + _SAR_BT_BIZ + "\n\n"
    "Raising the threshold above 8.21 weekly transactions begins missing SARs — "
    "keeping it at or below 8.21 maintains ≥90% catch rate while reducing unnecessary investigations."
)

_SAR_BT_IND = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
    "### SAR Backtest — Individual / trxn_amt_monthly\n\n"
    "**Population:** 2,167 alerted customers | **SARs:** 466 (21.5% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold (500.0): **466 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **8,500.0** (1 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **18,500.0** (420 of 466 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **28,500.0** (373 of 466 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **48,500.0** (234 of 466 caught)\n"
    "- At highest threshold (98,500.0): **15 caught**, 451 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
_PC_SAR_BT_IND = "Tool result for sar_backtest:\n" + _SAR_BT_IND

_SAR_BT_IND_RESPONSE = (
    "### SAR Catch Rate — Individual | Monthly Transaction Volume\n\n"
    + _SAR_BT_IND + "\n\n"
    "Raising the threshold above $18,500 in monthly volume begins to meaningfully miss SARs — "
    "at $48,500 only half of the 466 simulated SARs are caught."
)

# ARS_V48_1: Business SAR backtest (AVG_TRXNS_WEEK default)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for Business customers"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Business customers.",
     "tool_calls": [tc("ars_v48_1", "sar_backtest",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "ars_v48_1", "content": _PC_SAR_BT_BIZ},
    {"role": "assistant", "content": _SAR_BT_BIZ_RESPONSE},
]})

# ARS_V48_2: Individual SAR backtest (TRXN_AMT_MONTHLY explicit)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What is the SAR catch rate for Individual customers by monthly volume?"},
    {"role": "assistant",
     "content": "Calling sar_backtest for Individual / TRXN_AMT_MONTHLY.",
     "tool_calls": [tc("ars_v48_2", "sar_backtest",
                       {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "ars_v48_2", "content": _PC_SAR_BT_IND},
    {"role": "assistant", "content": _SAR_BT_IND_RESPONSE},
]})

# ===========================================================================
# ARS_V48_3-4  rule_sar_backtest — verbatim PRE-COMPUTED RULE SWEEP block
# ===========================================================================

_RS_ELDER_RESPONSE = (
    "### SAR Backtest — Elder Abuse | z_threshold\n\n"
    + _RS_ELDER_ABUSE + "\n\n"
    "The Elder Abuse rule catches all 59 SARs at the current z_threshold of 3.0 — "
    "tightening to 6.0 still catches 33 SARs (55.9%) while cutting 136 false positives."
)

_RS_ACH_RESPONSE = (
    "### SAR Backtest — Activity Deviation (ACH) | z_threshold\n\n"
    + _RS_ACH + "\n\n"
    "Raising z_threshold to 7.0 cuts 55 false positives while missing only 4 SARs — "
    "a modest trade-off that keeps the TP rate at 91.3%."
)

# ARS_V48_3: Elder Abuse rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for the Elder Abuse rule"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Elder Abuse.",
     "tool_calls": [tc("ars_v48_3", "rule_sar_backtest",
                       {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "ars_v48_3", "content": PC_RS_ELDER},
    {"role": "assistant", "content": _RS_ELDER_RESPONSE},
]})

# ARS_V48_4: Activity Deviation ACH rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for Activity Deviation ACH"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Activity Deviation (ACH).",
     "tool_calls": [tc("ars_v48_4", "rule_sar_backtest",
                       {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "ars_v48_4", "content": PC_RS_ACH},
    {"role": "assistant", "content": _RS_ACH_RESPONSE},
]})


# ===========================================================================
# AC_V48_3-5  AC header format after non-OFAC/non-SAR prior turns
#
# AC_V47_1-3 already cover: OFAC prior, SAR policy prior, TP/FP def prior.
# The model still uses "Business Customer Behavioral Segments" header after
# threshold tuning, rule list, or general BSA/AML policy prior turns.
# Fix: 3 more examples, all using canonical "Clustering complete for X" header.
# ===========================================================================

_PRIOR_THRESHOLD = (
    "The FP/FN crossover for Business customers at monthly transaction volume is "
    "around $24,000 — FP=218, FN=81 at that point. "
    "The optimal zone ($16,000–$20,000/month) keeps both FP and FN below 20% of their "
    "maximums. Above $24,000 FN losses accelerate while FP savings diminish."
)

_PRIOR_RULE_LIST = (
    "The system monitors the following AML rules: "
    "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, Velocity Single, "
    "Detect Excessive Transaction Activity, Structuring (Incoming Cash), "
    "Structuring (Outgoing Cash), CTR Client, Burst in Originator Activity, "
    "Burst in Beneficiary Activity, Risky International Transfer, Activity Deviation (Wire), "
    "Velocity Multiple, Funnel Account, Round-trip, Human Trafficking Indicators. "
    "See the chart for the full SAR/FP breakdown."
)

_PRIOR_BSA = (
    "The Bank Secrecy Act (BSA) requires financial institutions to maintain an effective "
    "AML compliance program with four pillars: (1) internal policies and procedures, "
    "(2) a designated compliance officer, (3) ongoing employee training, and "
    "(4) independent testing. Currency Transaction Reports (CTRs) must be filed for "
    "transactions exceeding $10,000. BSA violations can result in civil money penalties "
    "and formal enforcement actions."
)

# AC_V48_3: threshold tuning prior → Business clustering
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "What is the FP/FN crossover for Business customers?"},
    {"role": "assistant", "content": _PRIOR_THRESHOLD},
    {"role": "user",      "content": "Now show me the behavioral segments for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v48_3", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v48_3", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
]})

# AC_V48_4: rule list prior → Individual clustering
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "List all the AML rules in the system"},
    {"role": "assistant", "content": _PRIOR_RULE_LIST},
    {"role": "user",      "content": "Cluster Individual customers into behavioral groups"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v48_4", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v48_4", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": _T1_IND_RESPONSE},
]})

# AC_V48_5: general BSA/AML policy prior → Business clustering
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "What are the BSA compliance requirements?"},
    {"role": "assistant", "content": _PRIOR_BSA},
    {"role": "user",      "content": "Run segmentation for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v48_5", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v48_5", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _has_sar_backtest_canned(ex):
    """Remove sar_backtest / rule_sar_backtest examples where the final assistant
    response contains a canned placeholder without copying the PRE-COMPUTED block.
    These teach the model to skip verbatim copying."""
    msgs = ex["messages"]
    has_sar_tool = any(
        call.get("function", {}).get("name", "") in ("sar_backtest", "rule_sar_backtest")
        for m in msgs if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    )
    if not has_sar_tool:
        return False
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            content = m.get("content") or ""
            canned = (
                "Detailed sweep table shown" in content
                or "Detailed sweep chart shown" in content
                or "See chart/table below" in content
                or "shown in the chart below" in content
            )
            if canned and "=== PRE-COMPUTED" not in content:
                return True
    return False


def _has_hardcoded_rule_count(ex):
    """Remove examples where the assistant states '11' or '16' as a hardcoded AML
    rule count without a list_rules tool call providing that count."""
    msgs = ex["messages"]
    has_list_rules = any(
        call.get("function", {}).get("name") == "list_rules"
        for m in msgs if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    )
    if has_list_rules:
        return False  # count derives from tool result — OK
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            content = m.get("content") or ""
            if re.search(
                r'\b(11|16)\s+(?:active\s+)?(?:AML\s+)?(?:monitoring\s+)?rules?\b',
                content, re.IGNORECASE
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Combine V47 full + V48 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V48_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V48] V48-only: {V48_ONLY_PATH.name} ({len(examples)} examples)")

    if V47_FULL_PATH.exists():
        v47_base = []
        with open(V47_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v47_base.append(json.loads(line))
        print(f"[V48] Loaded {len(v47_base)} base examples from {V47_FULL_PATH.name}")

        filtered = v47_base
        for fn, label in [
            (_has_sar_backtest_canned,
             "sar_backtest/rule_sar_backtest canned placeholder without PRE-COMPUTED block"),
            (_has_hardcoded_rule_count,
             "hardcoded '11' or '16' AML rule count without list_rules tool call"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            after = len(filtered)
            print(f"[V48] Removed {before - after} {label} examples")

        all_examples = filtered + examples
        with open(V48_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V48] Combined: {V48_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V48] WARNING: V47 base not found at {V47_FULL_PATH}")


if __name__ == "__main__":
    main()

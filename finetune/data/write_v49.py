"""
V49 training examples (2026-05-15).

Targets:

  ARL_V49_1-3  "as a sweep parameter" phrasing — "rules that have min_transactions
               as a sweep parameter" misroutes to FP ranking instead of calling
               list_rules and filtering by sweep_params.  Three examples covering
               min_transactions, min_counterparties, daily_floor with this phrasing.

  ARL_V49_4-6  Short follow-up "how about {param}?" — after a parameter-filter
               exchange, "how about min_counterparties?" / "how about daily_floor?"
               triggers OOS response instead of filtering from existing context.
               Three multi-turn examples: turn 2 answers from prior tool result
               WITHOUT calling list_rules again.

  ARL_V49_7-8  "give me a list of all the rules in the system" after a parameter-
               filter exchange → still returns parameter-filtered list (gap persists
               after ARL_V48_6-7 which only covered post-precision context).
               Two anchoring examples using the exact session phrasing.

  ARL_V49_9-10 "How many rules does the system monitor?" / "what is the total
               number of rules" → returns FP ranking instead of count.
               Three session failures; model must call list_rules and count 16.

  AC_V49_1     "cluster 1" (bare identifier) after a per-cluster follow-up dumps
               all 4 clusters instead of just cluster 1.  Single-cluster answer
               from prior context.

  ARS_V49_1-4  rule_sar_backtest canned phrase still firing for Elder Abuse and
               ACH after ARS_V48_3-4.  Root cause: filter was not catching the
               "see chart below for {rule}" variant (only "See chart/table below").
               Four new examples using CTR, Velocity Single, Round-trip, Structuring
               to broaden coverage.  Filter broadened to match the session variant.

  AT_V49_1-2   "How can we reduce false positives by treating different customer
               clusters differently?" → "Results shown below." (x3 in session).
               AT_V48_4 trained a different phrasing; this one still fails.
               Two examples with the exact failing phrasing → cluster_threshold_analysis.

  AT_V49_3     Multi-turn AT: after cluster_threshold_analysis block, "which cluster
               reduces the most false positive with tuning?" → "Results shown below."
               Expected: answer from PRE-COMPUTED block in context (Cluster 2, -23 FP).
               One multi-turn example; no tool call on the follow-up turn.

  ARL_V49_11-12 Two more rule-count phrasings from the session that still return
               FP ranking: "How many rules are there in the system" and
               "count the number of rules from list of AML rules".

Filters applied in main():
  _has_sar_backtest_canned  : broadened — now also catches "see chart below for"
                              and "backtest complete" variants.
  _has_hardcoded_rule_count : unchanged from V48.

Base: aria_train_combined_v48_full.jsonl (571)
"""

import json, pathlib, re, sys

DATA_DIR      = pathlib.Path(__file__).parent
V48_FULL_PATH = DATA_DIR / "aria_train_combined_v48_full.jsonl"
V49_ONLY_PATH = DATA_DIR / "aria_train_v49.jsonl"
V49_FULL_PATH = DATA_DIR / "aria_train_combined_v49_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM           # noqa: E402
from write_v42 import tc, _CLUSTER_STATS                              # noqa: E402
from write_v44 import (                                                # noqa: E402
    PC_RS_CTR, _RS_CTR,
    PC_RS_VELOCITY_SINGLE, _RS_VELOCITY_SINGLE,
    PC_RS_STRUCTURING, _RS_STRUCTURING_IN,
    PC_RS_ROUND_TRIP, _RS_ROUND_TRIP,
)
from write_v45 import PC_LIST_H                                       # noqa: E402
from write_v46 import _PC_CLUSTER_BIZ, _T1_BIZ_RESPONSE              # noqa: E402
from write_v47 import _AT_A_BLOCK, _AT_A_RESPONSE, _PC_AT_A          # noqa: E402
from write_v48 import _ARL_FULL_LIST                                  # noqa: E402

examples = []

# ===========================================================================
# Sweep-param lookup answers (derived from _LIST_H — do not alter numbers)
#
# min_transactions : Burst in Originator Activity, Burst in Beneficiary Activity
# min_counterparties: Velocity Multiple, Funnel Account
# daily_floor      : Structuring (Incoming Cash), Structuring (Outgoing Cash)
# ratio_tolerance  : Velocity Single
# return_window    : Round-trip
# ===========================================================================

_FILTER_MIN_TXN = (
    "The rules that have min_transactions as a sweep parameter are: "
    "Burst in Originator Activity and Burst in Beneficiary Activity."
)

_FILTER_MIN_CP = (
    "The rules that have min_counterparties as a sweep parameter are: "
    "Velocity Multiple and Funnel Account."
)

_FILTER_DAILY_FLOOR = (
    "The rules that have daily_floor as a sweep parameter are: "
    "Structuring (Incoming Cash) and Structuring (Outgoing Cash)."
)

_FILTER_RATIO_TOL = (
    "Velocity Single is the only rule that has ratio_tolerance as a sweep parameter."
)

_FILTER_RETURN_WIN = (
    "Round-trip is the only rule that has return_window as a sweep parameter."
)


# ===========================================================================
# ARL_V49_1-3  "as a sweep parameter" phrasing — call list_rules, filter
#
# All single-turn. Model must call list_rules then filter by sweep_params field.
# Must NOT route to FP/SAR ranking.
# ===========================================================================

# ARL_V49_1: "rules that have min_transactions as a sweep parameter"
# (exact session phrasing that misrouted to FP ranking)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "rules that have min_transactions as a sweep parameter"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_1a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_1a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_MIN_TXN},
]})

# ARL_V49_2: "which rules have min_counterparties as a sweep parameter"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "which rules have min_counterparties as a sweep parameter"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_2a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_2a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_MIN_CP},
]})

# ARL_V49_3: "what rules use daily_floor as a sweep parameter"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "what rules use daily_floor as a sweep parameter"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_3a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_3a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_DAILY_FLOOR},
]})


# ===========================================================================
# ARL_V49_4-6  "how about {param}?" short follow-up — filter from prior context
#
# Critical: the second assistant turn has NO tool_calls.
# The model MUST answer from the PRE-COMPUTED RULE LIST already in the tool
# message from turn 1 — it must NOT call list_rules again.
# ===========================================================================

# ARL_V49_4: min_transactions → "how about min_counterparties?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "which rules have min_transactions"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_4a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_4a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_MIN_TXN},
    {"role": "user",   "content": "how about min_counterparties"},
    {"role": "assistant", "content": _FILTER_MIN_CP},
]})

# ARL_V49_5: "min_transactions as a sweep parameter" → "how about daily_floor?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "rules with min_transactions as a sweep parameter"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_5a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_5a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_MIN_TXN},
    {"role": "user",   "content": "how about daily_floor?"},
    {"role": "assistant", "content": _FILTER_DAILY_FLOOR},
]})

# ARL_V49_6: ratio_tolerance → "how about return_window?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "which rules have ratio_tolerance"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_6a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_6a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_RATIO_TOL},
    {"role": "user",   "content": "how about return_window?"},
    {"role": "assistant", "content": _FILTER_RETURN_WIN},
]})


# ===========================================================================
# ARL_V49_7-8  "give me a list of all the rules" after parameter-filter exchange
#
# V48 ARL_V48_6-7 covered this after precision context only.
# Session showed the failure persists after parameter-filter context:
# "give me a list of all the rules in the system" → returns z_threshold-filtered list.
#
# Second turn: NO tool_calls. Full plain list from prior context, no filter applied.
# ===========================================================================

# ARL_V49_7: z_threshold filter → "give me a list of all the rules in the system"
# (exact session phrasing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "which rules have Z_threshold"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_7a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_7a", "content": PC_LIST_H},
    {"role": "assistant",
     "content": (
         "The rules that have the z_threshold parameter are: "
         "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, "
         "Activity Deviation (Wire)."
     )},
    {"role": "user",   "content": "give me a list of all the rules in the system"},
    {"role": "assistant", "content": _ARL_FULL_LIST},
]})

# ARL_V49_8: min_counterparties filter → "list all the AML rules"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "which rules have min_counterparties"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve sweep parameters for all rules.",
     "tool_calls": [tc("arl_v49_8a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_8a", "content": PC_LIST_H},
    {"role": "assistant", "content": _FILTER_MIN_CP},
    {"role": "user",   "content": "list all the AML rules"},
    {"role": "assistant", "content": _ARL_FULL_LIST},
]})


# ===========================================================================
# ARL_V49_9-10  "How many rules does the system monitor?" → FP ranking
#
# All three session attempts returned FP ranking (CTR Client 348 FP, ...).
# Model must call list_rules and count 16 rules — not FP-sort.
# ===========================================================================

_ARL_COUNT_16 = (
    "The system monitors **16** AML rules: "
    "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, Velocity Single, "
    "Detect Excessive Transaction Activity, Structuring (Incoming Cash), "
    "Structuring (Outgoing Cash), CTR Client, Burst in Originator Activity, "
    "Burst in Beneficiary Activity, Risky International Transfer, Activity Deviation (Wire), "
    "Velocity Multiple, Funnel Account, Round-trip, Human Trafficking Indicators. "
    "See the chart for the full SAR/FP breakdown."
)

# ARL_V49_9: "How many rules does the system monitor?" (exact session phrasing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How many rules does the system monitor?"},
    {"role": "assistant",
     "content": "Calling list_rules to count the active monitoring rules.",
     "tool_calls": [tc("arl_v49_9a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_9a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_COUNT_16},
]})

# ARL_V49_10: "what is the total number of rules in the system" (second session phrasing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "what is the total number of rules in the system"},
    {"role": "assistant",
     "content": "Calling list_rules to count the active monitoring rules.",
     "tool_calls": [tc("arl_v49_10a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_10a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_COUNT_16},
]})


# ===========================================================================
# AC_V49_1  "cluster 1" bare identifier → single cluster answer (not all clusters)
#
# Session: after "how about cluster 4" gives Cluster 4 details,
# then "cluster 1" (one word + number) dumps all 4 clusters.
# Fix: bare cluster name answered with just that cluster from prior context.
#
# Business cluster data from _CLUSTER_STATS (write_v42):
#   Cluster 1 (n=121): Avg Weekly Txn Amt $1,205, Monthly Vol $4,820, Age 4.1 yrs
#   Cluster 4 (n=110): Avg Weekly Txn Amt $8,750, Monthly Vol $197,500, Age 9.2 yrs
# ===========================================================================

examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Show me the behavioral segments for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v49_1a", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v49_1a", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
    {"role": "user",   "content": "how about cluster 4"},
    {"role": "assistant",
     "content": (
         "**Cluster 4** (n=110 customers)\n"
         "- Avg Weekly Transactions: **22.6**\n"
         "- Avg Weekly Txn Amount: **$8,750**\n"
         "- Monthly Txn Volume: **$197,500**\n"
         "- Current Balance: **$148,000**\n"
         "- Account Age: **9.2 years**\n\n"
         "Cluster 4 is the highest-volume, highest-value segment."
     )},
    {"role": "user",   "content": "cluster 1"},
    {"role": "assistant",
     "content": (
         "**Cluster 1** (n=121 customers)\n"
         "- Avg Weekly Transactions: **3.2**\n"
         "- Avg Weekly Txn Amount: **$1,205**\n"
         "- Monthly Txn Volume: **$4,820**\n"
         "- Current Balance: **$18,500**\n"
         "- Account Age: **4.1 years**\n\n"
         "Cluster 1 is the lowest-volume segment — smaller accounts with moderate tenure."
     )},
]})


# ===========================================================================
# ARS_V49_1-4  rule_sar_backtest — four more rules to broaden verbatim coverage
#
# ARS_V48_3-4 trained Elder Abuse and ACH but both still return canned phrase.
# Root cause: filter was not catching "see chart below for {rule}" / "backtest
# complete" variant — contaminating examples survive in V48 base.
# Filter below is broadened; four new rules added (CTR, Velocity Single,
# Round-trip, Structuring) to reinforce the verbatim-copy pattern.
# ===========================================================================

_RS_CTR_RESPONSE = (
    "### SAR Backtest — CTR Client | floor_amount\n\n"
    + _RS_CTR + "\n\n"
    "Raising the floor above $15,000 begins missing SARs — keeping it at or below "
    "$15,000 maintains ≥90% SAR catch rate while cutting 43 false positives."
)

_RS_VELOCITY_RESPONSE = (
    "### SAR Backtest — Velocity Single | pair_total\n\n"
    + _RS_VELOCITY_SINGLE + "\n\n"
    "Velocity Single catches all 49 SARs across the entire sweep range — "
    "raising pair_total to $40,000 eliminates no false positives, suggesting "
    "the parameter has little discriminatory power at current thresholds."
)

_RS_ROUND_TRIP_RESPONSE = (
    "### SAR Backtest — Round-trip | floor_amount\n\n"
    + _RS_ROUND_TRIP + "\n\n"
    "Round-trip has the highest baseline precision (35%) in the system — "
    "even at $90,000 it retains 31 of 35 SARs with only 5 false negatives."
)

_RS_STRUCTURING_RESPONSE = (
    "### SAR Backtest — Structuring (Incoming Cash) | daily_floor\n\n"
    + _RS_STRUCTURING_IN + "\n\n"
    "The current daily floor of $3,000 catches all 46 SARs — "
    "raising it above $5,500 begins missing more than half the SAR pool."
)

# ARS_V49_1: CTR Client
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for CTR Client"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for CTR Client.",
     "tool_calls": [tc("ars_v49_1", "rule_sar_backtest",
                       {"risk_factor": "CTR Client", "sweep_param": "floor_amount"})]},
    {"role": "tool", "tool_call_id": "ars_v49_1", "content": PC_RS_CTR},
    {"role": "assistant", "content": _RS_CTR_RESPONSE},
]})

# ARS_V49_2: Velocity Single
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for Velocity Single"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Velocity Single.",
     "tool_calls": [tc("ars_v49_2", "rule_sar_backtest",
                       {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "tool_call_id": "ars_v49_2", "content": PC_RS_VELOCITY_SINGLE},
    {"role": "assistant", "content": _RS_VELOCITY_RESPONSE},
]})

# ARS_V49_3: Round-trip
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for the Round-trip rule"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Round-trip.",
     "tool_calls": [tc("ars_v49_3", "rule_sar_backtest",
                       {"risk_factor": "Round-trip", "sweep_param": "floor_amount"})]},
    {"role": "tool", "tool_call_id": "ars_v49_3", "content": PC_RS_ROUND_TRIP},
    {"role": "assistant", "content": _RS_ROUND_TRIP_RESPONSE},
]})

# ARS_V49_4: Structuring Incoming Cash
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for Structuring Incoming Cash"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Structuring (Incoming Cash).",
     "tool_calls": [tc("ars_v49_4", "rule_sar_backtest",
                       {"risk_factor": "Structuring (Incoming Cash)",
                        "sweep_param": "daily_floor"})]},
    {"role": "tool", "tool_call_id": "ars_v49_4", "content": PC_RS_STRUCTURING},
    {"role": "assistant", "content": _RS_STRUCTURING_RESPONSE},
]})


# ===========================================================================
# AT_V49_1-2  "reduce FPs by treating clusters differently" → canned phrase
#
# AT_V48_4 trained "What strategies exist to reduce false positives using
# customer segmentation?" but the phrasing "by treating different customer
# clusters differently" still routes to "Results shown below." (×3 in session).
# Both examples call cluster_threshold_analysis for Business (default).
# ===========================================================================

# AT_V49_1: exact session phrasing (failed ×3)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": "How can we reduce false positives by treating different customer clusters differently?"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v49_1", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v49_1", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
]})

# AT_V49_2: variant phrasing with same intent
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": "What is the best way to reduce FPs using cluster-specific thresholds for Business?"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v49_2", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v49_2", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
]})


# ===========================================================================
# AT_V49_3  Multi-turn AT: "which cluster reduces the most false positives?"
#
# After a full cluster_threshold_analysis block, the follow-up question
# "which cluster reduces the most false positive with tuning?" returns
# "Results shown below." instead of answering from the PRE-COMPUTED block.
#
# From _AT_A_BLOCK (Business / AVG_TRXNS_WEEK):
#   Cluster 1: -6 FP, Cluster 2: -23 FP, Cluster 3: +2 FP, Cluster 4: -2 FP
#   → Cluster 2 (Mid-High) has the largest FP reduction at -23 FP.
#
# No tool call on the follow-up turn — answer from prior context.
# ===========================================================================

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run cluster-adaptive threshold analysis for Business customers"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v49_3a", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v49_3a", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
    {"role": "user",   "content": "which cluster reduces the most false positives with tuning?"},
    {"role": "assistant",
     "content": (
         "**Cluster 2 (Mid-High)** reduces the most false positives with adaptive tuning: "
         "**-23 FP** (from 612 down to 589) while retaining 93.9% SAR catch rate. "
         "Cluster 1 (High Volume) is the most efficient per customer at -6 FP across only 18 accounts."
     )},
]})


# ===========================================================================
# ARL_V49_11-12  Rule count — two more session phrasings still returning FP ranking
#
# ARL_V49_9-10 covered "How many rules does the system monitor?" and
# "what is the total number of rules in the system".
# These two phrasings from the same session still fail.
# ===========================================================================

# ARL_V49_11: "How many rules are there in the system" (exact session phrasing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How many rules are there in the system"},
    {"role": "assistant",
     "content": "Calling list_rules to count the active monitoring rules.",
     "tool_calls": [tc("arl_v49_11a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_11a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_COUNT_16},
]})

# ARL_V49_12: "count the number of rules from list of AML rules" (exact session phrasing)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "count the number of rules from list of AML rules"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve and count the active monitoring rules.",
     "tool_calls": [tc("arl_v49_12a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_12a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_COUNT_16},
]})


# ===========================================================================
# AC_V49_2  Second-hop cluster follow-up: "oldest" → "youngest"
#
# Test result: "which cluster is oldest" → Cluster 3, 6.7 yrs ✓
#              "and the youngest" → "I do not have the specific cluster age data" ✗
#
# AC_V48_1 trained this with "which cluster is the oldest in account age?" phrasing.
# The simpler "which cluster is oldest" → "and the youngest" still fails on turn 2.
# Three-message example so the model sees the exact failing phrasing.
#
# Business cluster data from _CLUSTER_STATS (write_v42):
#   Cluster 1 (n=121): Account Age 4.1 yrs  ← youngest
#   Cluster 2 (n=98):  Account Age 7.3 yrs
#   Cluster 3 (n=156): Account Age 5.8 yrs
#   Cluster 4 (n=110): Account Age 9.2 yrs  ← oldest
# ===========================================================================

examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Show me behavioral clusters for Business"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v49_2a", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v49_2a", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
    {"role": "user",   "content": "which cluster is oldest"},
    {"role": "assistant",
     "content": "Cluster 4 is the oldest with an average account age of **9.2 years**."},
    {"role": "user",   "content": "and the youngest"},
    {"role": "assistant",
     "content": "Cluster 1 is the youngest with an average account age of **4.1 years**."},
]})


# ===========================================================================
# ARL_V49_13-14  Precision highest → lowest: hallucinated numbers + invented rules
#
# Test result:
#   "Which rules have the highest precision" →
#      wrong numbers (Round-trip 35%, Risky Int'l 26.7%, Funnel 26%)
#   "and the lowest" →
#      invented rules ("High-Value Cash Deposit", "Rapid Account Activity",
#      "Multiple Small International Transfers") — none exist in the system
#
# Root cause: model reads precision values and rule names from parametric memory
# instead of from the PRE-COMPUTED RULE LIST in the tool result.
#
# Fix: two examples — one single-turn anchoring correct format (top-3 from
# _LIST_H), one multi-turn with "and the lowest" using exact failing phrasing.
# Critical: follow-up must use real rules only — CTR Client, Detect Excessive,
# Elder Abuse — NOT invented names.
# ===========================================================================

_ARL_TOP3_VERBATIM = (
    "The rules with the highest precision are:\n"
    "- **Funnel Account** (precision=46.4%)\n"
    "- **Risky International Transfer** (precision=22.9%)\n"
    "- **Round-trip** (precision=22.4%)\n\n"
    "See the chart for the full rule performance breakdown."
)

_ARL_BOT3_VERBATIM = (
    "The rules with the lowest precision are:\n"
    "- **CTR Client** (precision=13.0%)\n"
    "- **Detect Excessive Transaction Activity** (precision=13.7%)\n"
    "- **Elder Abuse** (precision=14.9%)\n\n"
    "These rules generate the highest ratio of false positives to SARs."
)

# ARL_V49_13: single-turn — anchor correct rule names and values for highest precision
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve precision for all rules.",
     "tool_calls": [tc("arl_v49_13a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_13a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_TOP3_VERBATIM},
]})

# ARL_V49_14: multi-turn — exact failing sequence; second turn must use real rules,
# real numbers, NO tool call
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve precision for all rules.",
     "tool_calls": [tc("arl_v49_14a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v49_14a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_TOP3_VERBATIM},
    {"role": "user",   "content": "and the lowest"},
    {"role": "assistant", "content": _ARL_BOT3_VERBATIM},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _has_sar_backtest_canned(ex):
    """Remove sar_backtest / rule_sar_backtest examples where the final assistant
    response contains a canned placeholder without copying the PRE-COMPUTED block.

    Broadened in V49 to catch the session variant:
      "SAR backtest complete — see chart below for {rule}"
    which the V48 filter missed (checked for "See chart/table below" with slash,
    not "see chart below for").
    """
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
                or "see chart below" in content.lower()       # NEW: "see chart below for X"
                or "backtest complete" in content.lower()     # NEW: "SAR backtest complete"
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
        return False
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
# Combine V48 full + V49 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V49_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V49] V49-only: {V49_ONLY_PATH.name} ({len(examples)} examples)")

    if V48_FULL_PATH.exists():
        v48_base = []
        with open(V48_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v48_base.append(json.loads(line))
        print(f"[V49] Loaded {len(v48_base)} base examples from {V48_FULL_PATH.name}")

        filtered = v48_base
        for fn, label in [
            (_has_sar_backtest_canned,
             "sar_backtest/rule_sar_backtest canned placeholder without PRE-COMPUTED block"),
            (_has_hardcoded_rule_count,
             "hardcoded '11' or '16' AML rule count without list_rules tool call"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            after = len(filtered)
            print(f"[V49] Removed {before - after} {label} examples")

        all_examples = filtered + examples
        with open(V49_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V49] Combined: {V49_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V49] WARNING: V48 base not found at {V48_FULL_PATH}")


if __name__ == "__main__":
    main()

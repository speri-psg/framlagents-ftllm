"""
V46 training examples (2026-05-14).

Targets:

  ARS_V46_1-4   Gap: multi-turn list_rules → rule_sar_backtest produces empty text.
                When list_rules runs in turn N and the user requests a specific rule
                backtest in turn N+1, the model calls the tool correctly but generates
                0 chars after the PRE-COMPUTED RULE SWEEP block arrives.
                Four variants covering different rules and query phrasings.

  AC_V46_1-6    Regression fix: fresh ds_cluster_analysis responses must always start
                with "Clustering complete for X customers." + full PRE-COMPUTED CLUSTER
                STATS block + summary. V45 AC_V45_6-7 dropped the header, teaching the
                model that it is optional. Filter removes those two examples; these
                replacements + three new "show me segments" examples re-anchor the format.

  ARS_V46_5-6   Generalization: rule_sar_backtest for fictitious rule names that do not
                exist in the catalogue. Forces the model to learn the general pattern
                "named rule → rule_sar_backtest" rather than matching on known names.
                Two examples with different fictitious rules and different phrasings.

Filters applied in main():
  _has_stale_11_rules:               removes pre-V40 examples with wrong rule count.
  _has_truncated_clustering_response: removes fresh clustering examples missing header.
  _has_dead_tool:                    removes examples using tool names that no longer
                                     exist (query_knowledge_base, search_policy_kb,
                                     ofac_screening).
  _has_wrong_tool_routing:           removes specific (query, tool) conflict pairs that
                                     contradict the correct routing established by the
                                     majority of examples.

Base: aria_train_combined_v45_full.jsonl
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V45_FULL_PATH = DATA_DIR / "aria_train_combined_v45_full.jsonl"
V46_ONLY_PATH = DATA_DIR / "aria_train_v46.jsonl"
V46_FULL_PATH = DATA_DIR / "aria_train_combined_v46_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM            # noqa: E402
from write_v42 import tc, _CLUSTER_STATS                               # noqa: E402
from write_v44 import (                                                 # noqa: E402
    _RS_ELDER_ABUSE, PC_RS_ELDER,
    _RS_ACH, PC_RS_ACH,
    _RS_CTR, PC_RS_CTR,
    _RS_ROUND_TRIP, PC_RS_ROUND_TRIP,
    _LIST_G, PC_LIST_G,
)
from write_v45 import (                                                 # noqa: E402
    _LIST_H, PC_LIST_H,
    _LIST_I, PC_LIST_I,
    _LIST_J, PC_LIST_J,
    _CLUSTER_STATS_IND, _PC_CLUSTER_IND, _T1_IND_RESPONSE,
)

# Business cluster tool result string (used in fresh-clustering examples)
_PC_CLUSTER_BIZ = "Tool result for ds_cluster_analysis:\n" + _CLUSTER_STATS

# Full "Clustering complete" response for Business customers (V44 format)
_T1_BIZ_RESPONSE = (
    "Clustering complete for Business customers.\n\n"
    + _CLUSTER_STATS + "\n\n"
    "**Summary:** Cluster 4 (n=110) is the highest-activity, highest-value segment — "
    "22.6 weekly transactions and $197,500 monthly volume. "
    "Cluster 1 (n=121) is the lowest-activity, lowest-balance segment with the newest accounts (4.1 yrs). "
    "Clusters 2 and 3 represent intermediate activity levels."
)

examples = []

# ===========================================================================
# AC_V46_1-6  Regression fix: full "Clustering complete" format for fresh clustering
#
# AC_V46_1-2  Corrected replacements for removed AC_V45_6-7 (had stats block but
#             no "Clustering complete" header — taught the model the header is optional).
# AC_V46_3-6  Simple "show me segments" queries — reinforce the canonical format:
#             "Clustering complete for X customers." + full stats block + summary.
# ===========================================================================

# AC_V46_1: corrected replacement for AC_V45_6 (oldest account age — Individual)
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Which cluster of Individual customers has the oldest account age?"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v46_1", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v46_1", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": (
        "Clustering complete for Individual customers.\n\n"
        + _CLUSTER_STATS_IND + "\n\n"
        "Cluster 4 has the oldest average account age at **11.2 years**, "
        "followed by Cluster 3 (**8.7 years**), Cluster 2 (**6.1 years**), and Cluster 1 (**3.4 years**)."
    )},
]})

# AC_V46_2: corrected replacement for AC_V45_7 (highest income — Individual)
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Run clustering on Individual customers — which segment has the highest income?"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v46_2", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v46_2", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": (
        "Clustering complete for Individual customers.\n\n"
        + _CLUSTER_STATS_IND + "\n\n"
        "Cluster 4 has the highest average income at **$261,000**, "
        "followed by Cluster 3 (**$134,000**), Cluster 2 (**$87,000**), and Cluster 1 (**$48,000**)."
    )},
]})

# AC_V46_3: simple "show me segments" — Individual
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Show me the behavioral segments for Individual customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v46_3", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v46_3", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": _T1_IND_RESPONSE},
]})

# AC_V46_4: simple "cluster customers" — Business
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Cluster Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v46_4", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v46_4", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
]})

# AC_V46_5: "run segmentation" — Business
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Run segmentation for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v46_5", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v46_5", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
]})

# AC_V46_6: "what are the customer segments" — Individual
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "What are the customer segments for Individual accounts?"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v46_6", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v46_6", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": _T1_IND_RESPONSE},
]})

# ===========================================================================
# ARS_V46_1-4  Gap: multi-turn list_rules → rule_sar_backtest empty response
#
# Pattern:
#   Turn 1 — user: simple "list rules" query
#             assistant: calls list_rules → receives PRE-COMPUTED RULE LIST
#                        → responds with verbatim block + brief summary sentence
#   Turn 2 — user: requests specific rule backtest (separate message)
#             assistant: calls rule_sar_backtest → receives PRE-COMPUTED RULE SWEEP
#                        → MUST generate full ### header + verbatim block + insight
#
# The gap is that the model generates 0 chars in Turn 2 after the tool result.
# ===========================================================================

# ARS_V46_1: list_rules → Elder Abuse SAR backtest
# Dataset H: Elder Abuse alerts=390, SAR=58, FP=332, precision=14.9%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What rules are available?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("ars_v46_1a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "ars_v46_1a", "content": PC_LIST_H},
    {"role": "assistant", "content": (
        _LIST_H + "\n\n"
        "The system contains 16 AML rules spanning volume, velocity, counterparty pattern, "
        "and behavioral deviation typologies."
    )},
    {"role": "user", "content": "Show me the SAR backtest for Elder Abuse"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Elder Abuse.",
     "tool_calls": [tc("ars_v46_1b", "rule_sar_backtest",
                       {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "tool_call_id": "ars_v46_1b", "content": PC_RS_ELDER},
    {"role": "assistant", "content": (
        "### Rule Sweep — Elder Abuse | z_threshold\n\n"
        + _RS_ELDER_ABUSE + "\n\n"
        "At z=3 all 59 SARs are caught; tightening to z=6 drops the catch rate to 56% "
        "(33 caught) while saving 136 FPs."
    )},
]})

# ARS_V46_2: list_rules → Activity Deviation (ACH) SAR backtest
# Dataset I: ACH alerts=310, SAR=50, FP=260, precision=16.1%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "List all AML monitoring rules"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("ars_v46_2a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "ars_v46_2a", "content": PC_LIST_I},
    {"role": "assistant", "content": (
        _LIST_I + "\n\n"
        "The system contains 16 AML rules. "
        "Detect Excessive Transaction Activity and CTR Client have the lowest precision "
        "(12.5% and 12.8%), while Round-trip leads at 24.6%."
    )},
    {"role": "user", "content": "Run the backtest for Activity Deviation ACH"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Activity Deviation (ACH).",
     "tool_calls": [tc("ars_v46_2b", "rule_sar_backtest",
                       {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "ars_v46_2b", "content": PC_RS_ACH},
    {"role": "assistant", "content": (
        "### Rule Sweep — Activity Deviation (ACH) | z_threshold\n\n"
        + _RS_ACH + "\n\n"
        "Raising z_threshold from 5 to 7 cuts 55 FPs while losing only 4 SARs, "
        "improving precision from 15.3% to 17.4%."
    )},
]})

# ARS_V46_3: list_rules → CTR Client SAR backtest
# Dataset G: CTR Client alerts=455, SAR=59, FP=396, precision=13.0%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules does the system have?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("ars_v46_3a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "ars_v46_3a", "content": PC_LIST_G},
    {"role": "assistant", "content": (
        _LIST_G + "\n\n"
        "The system contains 16 AML rules. "
        "Round-trip leads in precision at 33.3%, while CTR Client has the highest FP count at 396."
    )},
    {"role": "user", "content": "Show CTR Client backtest"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for CTR Client.",
     "tool_calls": [tc("ars_v46_3b", "rule_sar_backtest",
                       {"risk_factor": "CTR Client"})]},
    {"role": "tool", "tool_call_id": "ars_v46_3b", "content": PC_RS_CTR},
    {"role": "assistant", "content": (
        "### Rule Sweep — CTR Client | floor_amount\n\n"
        + _RS_CTR + "\n\n"
        "The $10K floor already catches all 52 SARs; raising to $15K cuts 43 FPs "
        "at the cost of 4 missed SARs (92% catch rate maintained)."
    )},
]})

# ARS_V46_4: list_rules → Round-trip SAR backtest
# Dataset J: Round-trip alerts=116, SAR=28, FP=88, precision=24.1%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Can you show me all AML rules?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("ars_v46_4a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "ars_v46_4a", "content": PC_LIST_J},
    {"role": "assistant", "content": (
        _LIST_J + "\n\n"
        "The system contains 16 AML rules. "
        "Elder Abuse catches the most SARs at 72, while Risky International Transfer "
        "leads precision at 23.3%."
    )},
    {"role": "user", "content": "Show me the Round-trip rule backtest"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Round-trip.",
     "tool_calls": [tc("ars_v46_4b", "rule_sar_backtest",
                       {"risk_factor": "Round-trip"})]},
    {"role": "tool", "tool_call_id": "ars_v46_4b", "content": PC_RS_ROUND_TRIP},
    {"role": "assistant", "content": (
        "### Rule Sweep — Round-trip | floor_amount\n\n"
        + _RS_ROUND_TRIP + "\n\n"
        "Round-trip has the highest precision in the system at 33.3%; "
        "raising the floor to $30K keeps 90% SAR coverage while cutting 20 FPs."
    )},
]})


# ===========================================================================
# ARS_V46_5-6  Generalization: rule_sar_backtest for fictitious rule names
#
# Both rules do not exist in the catalogue. The model must apply the general
# "named rule → rule_sar_backtest" pattern rather than matching on a name it
# has seen before. This teaches the pattern as a variable, not a fixed string.
# ===========================================================================

_RS_RAPID_FUND_CYCLING = """\
=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Rapid Fund Cycling
Current condition: 3-day incoming transfer total >= $25K with >= 4 distinct counterparties within 7 days
Sweep parameter: floor_amount - minimum 3-day transfer total to trigger the rule
Current value: 25,000
Labeled population: 180 customers (TP+FN pool=28 SAR, FP+TN pool=152 non-SAR, precision=15.6%)
At the lowest value (5,000.00): TP=28, FP=152, FN=0, TN=0 (TP rate=100.0%, precision=15.6%).
At current condition (25,000.00): TP=25, FP=120, FN=3, TN=32 (TP rate=89.3%, precision=17.2%).
To keep TP rate >=90%: floor_amount <= 22,000.00 => TP=26, FP=135, FN=2, TN=17, precision=16.1%.
To keep TP rate >=50%: floor_amount <= 48,000.00 => TP=15, FP=61, FN=13, TN=91, precision=19.7%.
At the highest value (100,000.00): TP=2, FP=8, FN=26, TN=144, precision=20.0%.
=== END RULE SWEEP ===
(Detailed sweep table shown in the chart below.)"""

_PC_RS_RAPID = "Tool result for rule_sar_backtest:\n" + _RS_RAPID_FUND_CYCLING

_RS_CROSS_BORDER = """\
=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Cross-Border Layering
Current condition: International wire >= $15K to >= 3 distinct high-risk jurisdictions within 30 days
Sweep parameter: floor_amount - minimum single international wire amount to trigger the rule
Current value: 15,000
Labeled population: 140 customers (TP+FN pool=32 SAR, FP+TN pool=108 non-SAR, precision=22.9%)
At the lowest value (5,000.00): TP=32, FP=108, FN=0, TN=0 (TP rate=100.0%, precision=22.9%).
At current condition (15,000.00): TP=30, FP=74, FN=2, TN=34 (TP rate=93.8%, precision=28.8%).
To keep TP rate >=90%: floor_amount <= 18,000.00 => TP=29, FP=62, FN=3, TN=46, precision=31.9%.
To keep TP rate >=50%: floor_amount <= 40,000.00 => TP=17, FP=29, FN=15, TN=79, precision=37.0%.
At the highest value (80,000.00): TP=3, FP=4, FN=29, TN=104, precision=42.9%.
=== END RULE SWEEP ===
(Detailed sweep table shown in the chart below.)"""

_PC_RS_CROSS_BORDER = "Tool result for rule_sar_backtest:\n" + _RS_CROSS_BORDER

# ARS_V46_5: fictitious rule "Rapid Fund Cycling"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What is the SAR catch rate for Rapid Fund Cycling?"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Rapid Fund Cycling.",
     "tool_calls": [tc("ars_v46_5", "rule_sar_backtest",
                       {"risk_factor": "Rapid Fund Cycling"})]},
    {"role": "tool", "tool_call_id": "ars_v46_5", "content": _PC_RS_RAPID},
    {"role": "assistant", "content": (
        "### Rule Sweep — Rapid Fund Cycling | floor_amount\n\n"
        + _RS_RAPID_FUND_CYCLING + "\n\n"
        "Lowering the floor to $22K recovers the 2 missed SARs while adding only 15 FPs "
        "relative to the current $25K condition."
    )},
]})

# ARS_V46_6: fictitious rule "Cross-Border Layering" — different phrasing
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me the backtest for Cross-Border Layering"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Cross-Border Layering.",
     "tool_calls": [tc("ars_v46_6", "rule_sar_backtest",
                       {"risk_factor": "Cross-Border Layering"})]},
    {"role": "tool", "tool_call_id": "ars_v46_6", "content": _PC_RS_CROSS_BORDER},
    {"role": "assistant", "content": (
        "### Rule Sweep — Cross-Border Layering | floor_amount\n\n"
        + _RS_CROSS_BORDER + "\n\n"
        "Cross-Border Layering has above-average precision at 28.8%; "
        "raising the floor to $18K improves precision to 31.9% while keeping 90% SAR coverage."
    )},
]})

# ===========================================================================
# R2D_V46_1-2  Generalization: rule_2d_sweep for fictitious rule names
#
# rule_2d_sweep has the same open-ended risk_factor problem as rule_sar_backtest.
# Two fictitious rules (not in the catalogue) with different param pairs force
# the model to generalise the pattern to any rule name it encounters.
# ===========================================================================

_2D_SMURFING = """\
=== PRE-COMPUTED 2D SWEEP RESULT (copy verbatim) ===
Rule: Smurfing Activity | Grid: 6x6=36 combinations
Current params: floor_amount=9500 min_transactions=3 => TP=41 FP=187 FN=0 TN=0
Best FP reduction cell: floor_amount=12000 min_transactions=4 => TP=36 FP=121 FN=5 TN=66 TP rate=87.8% precision=22.9%
Heatmap is displayed in the chart panel.
=== END PRE-COMPUTED ==="""

_PC_2D_SMURFING = "Tool result for rule_2d_sweep:\n" + _2D_SMURFING

_2D_GHOST_EMPLOYEE = """\
=== PRE-COMPUTED 2D SWEEP RESULT (copy verbatim) ===
Rule: Ghost Employee Diversion | Grid: 7x7=49 combinations
Current params: floor_amount=20000 z_threshold=2.5 => TP=29 FP=143 FN=0 TN=0
Best FP reduction cell: floor_amount=30000 z_threshold=3.5 => TP=24 FP=78 FN=5 TN=65 TP rate=82.8% precision=23.5%
Heatmap is displayed in the chart panel.
=== END PRE-COMPUTED ==="""

_PC_2D_GHOST_EMPLOYEE = "Tool result for rule_2d_sweep:\n" + _2D_GHOST_EMPLOYEE

# R2D_V46_1: fictitious "Smurfing Activity" — floor_amount x min_transactions
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me a 2D sweep for Smurfing Activity"},
    {"role": "assistant",
     "content": "Calling rule_2d_sweep for Smurfing Activity.",
     "tool_calls": [tc("r2d_v46_1", "rule_2d_sweep",
                       {"risk_factor": "Smurfing Activity",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "min_transactions"})]},
    {"role": "tool", "tool_call_id": "r2d_v46_1", "content": _PC_2D_SMURFING},
    {"role": "assistant", "content": (
        "### 2D Sweep — Smurfing Activity | Floor Amount × Min Transactions\n\n"
        + _2D_SMURFING + "\n\n"
        "Raising floor_amount to $12K and min_transactions to 4 cuts 66 FPs "
        "while keeping 88% SAR coverage."
    )},
]})

# R2D_V46_2: fictitious "Ghost Employee Diversion" — floor_amount x z_threshold
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run 2D analysis for Ghost Employee Diversion"},
    {"role": "assistant",
     "content": "Calling rule_2d_sweep for Ghost Employee Diversion.",
     "tool_calls": [tc("r2d_v46_2", "rule_2d_sweep",
                       {"risk_factor": "Ghost Employee Diversion",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "r2d_v46_2", "content": _PC_2D_GHOST_EMPLOYEE},
    {"role": "assistant", "content": (
        "### 2D Sweep — Ghost Employee Diversion | Floor Amount × Z Threshold\n\n"
        + _2D_GHOST_EMPLOYEE + "\n\n"
        "Tightening both parameters to $30K / z=3.5 recovers 65 FPs "
        "at a cost of 5 missed SARs, improving precision from 16.9% to 23.5%."
    )},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _has_stale_11_rules(ex):
    """Remove any example where a system prompt or assistant response mentions
    '11 active', 'exactly 11', or '11 rules' — stale rule count from pre-V40 data."""
    for m in ex["messages"]:
        content = m.get("content", "") or ""
        if not isinstance(content, str):
            continue
        if ("11 active" in content
                or "exactly 11" in content
                or "11 rules" in content
                or "11 active AML rules" in content):
            return True
    return False


_DEAD_TOOLS = {"query_knowledge_base", "search_policy_kb", "ofac_screening"}

_WRONG_TOOL_PAIRS = [
    ("which rules generate the most false positives", "cluster_rule_summary"),
    ("which rules generate the most false positives", "direct"),
    ("show elder abuse sar backtest for cluster 4",   "ds_cluster_analysis"),
    ("run a sar backtest for activity deviation ach in cluster 2", "ds_cluster_analysis"),
    ("run a 2d sweep for elder abuse for cluster 2",  "ds_cluster_analysis"),
    ("which cluster of business customers has the highest transaction volume", "cluster_analysis"),
]


def _first_tool(msgs):
    for m in msgs:
        if m.get("role") == "assistant":
            tc_ = m.get("tool_calls")
            if tc_:
                try: return tc_[0]["function"]["name"]
                except (KeyError, IndexError): pass
            return "direct"
    return "direct"


def _has_dead_tool(ex):
    """True if any assistant message calls a tool name that no longer exists."""
    for m in ex["messages"]:
        if m.get("role") == "assistant":
            for call in (m.get("tool_calls") or []):
                if call.get("function", {}).get("name", "") in _DEAD_TOOLS:
                    return True
    return False


def _has_wrong_tool_routing(ex):
    """True if this example matches a known (query_fragment, wrong_tool) conflict pair."""
    msgs = ex["messages"]
    first = _first_tool(msgs)
    for m in msgs:
        if m.get("role") == "user":
            q = (m.get("content") or "").lower()
            for frag, bad_tool in _WRONG_TOOL_PAIRS:
                if frag in q and first == bad_tool:
                    return True
            break
    return False


def _has_truncated_clustering_response(ex):
    """Remove fresh ds_cluster_analysis examples where the assistant response
    starts directly with the PRE-COMPUTED CLUSTER STATS block without the
    'Clustering complete' header. These are AC_V45_6 and AC_V45_7."""
    found_cluster_call = False
    for m in ex["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                if call.get("function", {}).get("name") == "ds_cluster_analysis":
                    found_cluster_call = True
        if (found_cluster_call
                and m.get("role") == "assistant"
                and not m.get("tool_calls")
                and isinstance(m.get("content"), str)):
            content = m["content"]
            if (content.startswith("=== PRE-COMPUTED CLUSTER STATS")
                    and "Clustering complete" not in content):
                return True
    return False


_RULE_2D_INVALID_PARAMS = {"height_threshold", "age_score"}
_RULE_2D_CAP = 6  # max rule_2d_sweep examples per rule name


def _make_rule_2d_filter():
    """
    Returns a stateful filter that:
      - removes examples with hallucinated rule_2d_sweep params
        (height_threshold, age_score — never valid params)
      - caps rule_2d_sweep examples per risk_factor at _RULE_2D_CAP
        (Elder Abuse 24x and Velocity Single 11x dominate; cap balances the set)
    Must be created fresh in main() so the counter resets each run.
    """
    from collections import Counter
    counts: Counter = Counter()

    def _filter(ex) -> bool:
        for m in ex.get("messages", []):
            if m.get("role") == "assistant":
                for call in (m.get("tool_calls") or []):
                    if call.get("function", {}).get("name") == "rule_2d_sweep":
                        try:
                            args = json.loads(call["function"]["arguments"])
                        except Exception:
                            args = {}
                        p1 = args.get("sweep_param_1", "")
                        p2 = args.get("sweep_param_2", "")
                        if p1 in _RULE_2D_INVALID_PARAMS or p2 in _RULE_2D_INVALID_PARAMS:
                            return True  # remove
                        rule = args.get("risk_factor", "")
                        counts[rule] += 1
                        if counts[rule] > _RULE_2D_CAP:
                            return True  # remove (over cap)
        return False

    return _filter


# ---------------------------------------------------------------------------
# Combine V45 full + V46 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V46_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V46] V46-only: {V46_ONLY_PATH.name} ({len(examples)} examples)")

    if V45_FULL_PATH.exists():
        v45_base = []
        with open(V45_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v45_base.append(json.loads(line))
        print(f"[V46] Loaded {len(v45_base)} base examples from {V45_FULL_PATH.name}")

        filtered = v45_base
        for fn, label in [
            (_has_stale_11_rules,
             "stale 11-rules count (pre-V40 examples)"),
            (_has_truncated_clustering_response,
             "truncated clustering response (missing 'Clustering complete' header)"),
            (_has_dead_tool,
             "dead tool name (query_knowledge_base / search_policy_kb / ofac_screening)"),
            (_has_wrong_tool_routing,
             "wrong-tool routing conflict"),
            (_make_rule_2d_filter(),
             f"rule_2d_sweep hallucinated params or over cap ({_RULE_2D_CAP}/rule)"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            after = len(filtered)
            print(f"[V46] Removed {before - after} {label} examples")

        all_examples = filtered + examples
        with open(V46_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V46] Combined: {V46_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V46] WARNING: V45 base not found at {V45_FULL_PATH}")


if __name__ == "__main__":
    main()

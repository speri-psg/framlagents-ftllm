"""
V47 training examples (2026-05-14).

Targets:

  AT_V47_1-5   Gap 11: cluster_threshold_analysis — zero training examples ever.
               Model asks for column clarification instead of defaulting to
               AVG_TRXNS_WEEK, generates wrong header ("SAR Catch Rate —"),
               hallucinates FP counts, omits insight sentence, and sometimes
               produces a second hallucinated text block.
               Five examples covering: two Business/AVG_TRXNS_WEEK phrasings,
               Business/AVG_TRXN_AMT, Individual/AVG_TRXNS_WEEK,
               and the conceptual "reduce FPs by treating clusters differently"
               phrasing that previously triggered a double-response.

  AC_V47_1-3   Gap 9: fresh clustering after a non-clustering policy exchange.
               Current AC examples all start from a neutral/empty history.
               When the prior turn contains OFAC, SAR policy, or TP/FP content,
               the model drops the "Clustering complete" header and produces only
               an insight. Three examples with different prior content types.

  ARL_V47_1-2  Gap 2: multi-turn list_rules precision re-sort without re-calling tool.
               After a highest-precision query uses list_rules, a follow-up
               "what about the lowest?" must re-sort the existing PRE-COMPUTED data
               rather than calling list_rules a second time or returning wrong data.

Filters applied in main():
  _has_arl_precision_trail   : removes list_rules examples whose final response
                               ends with a precision-sort sentence after the
                               END RULE LIST marker — these cause the model to
                               return a precision summary when the app strips the
                               PRE-COMPUTED block from the bubble.
  _has_segment_in_list_rules : removes list_rules examples where the assistant
                               response contains "Segment: " and "Total accounts:"
                               or "Active accounts:" — cluster context bleed.
  _has_plain_pre_computed_label : removes examples where the assistant response
                               emits "PRE-COMPUTED RULE LIST" or "PRE-COMPUTED
                               RULE SWEEP" as a plain-text line without === prefix.

Base: aria_train_combined_v46_full.jsonl (572)
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V46_FULL_PATH = DATA_DIR / "aria_train_combined_v46_full.jsonl"
V47_ONLY_PATH = DATA_DIR / "aria_train_v47.jsonl"
V47_FULL_PATH = DATA_DIR / "aria_train_combined_v47_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM           # noqa: E402
from write_v42 import tc, _CLUSTER_STATS                              # noqa: E402
from write_v45 import (                                                # noqa: E402
    _LIST_H, PC_LIST_H,
    _CLUSTER_STATS_IND, _PC_CLUSTER_IND, _T1_IND_RESPONSE,
)
from write_v46 import _T1_BIZ_RESPONSE, _PC_CLUSTER_BIZ               # noqa: E402

examples = []

# ===========================================================================
# AT_V47_1-5  Gap 11: cluster_threshold_analysis — first training examples ever
#
# Canonical header: "### Cluster-Adaptive Threshold Analysis — {segment} | {label}"
# Column labels (from lambda_cluster_threshold.py _COL_LABEL):
#   AVG_TRXNS_WEEK   → "Avg Weekly Transactions"
#   AVG_TRXN_AMT     → "Avg Weekly Txn Amount"
#   TRXN_AMT_MONTHLY → "Monthly Txn Volume"
#
# Format: full === PRE-COMPUTED CLUSTER THRESHOLD ANALYSIS block (new markdown
# format with **bold** cluster headers and bullet points) copied verbatim,
# followed by one insight sentence referencing net FP change and SAR retention.
# Default segment = Business, default column = AVG_TRXNS_WEEK (Rule 8 of SYSTEM_PROMPT).
# Model must NOT ask for clarification if column is unspecified.
# ===========================================================================

# ── Dataset AT-A: Business / AVG_TRXNS_WEEK, uniform threshold 2.30 ─────────
_AT_A_CONTENT = """\
Segment: **Business** | Column: AVG_TRXNS_WEEK (Avg Weekly Transactions) | Clusters: 4 | Target SAR catch: ≥90%

**Cluster 1 — High Volume** (18 customers, SAR pool: 6)
- Uniform 2.30: TP=5, FP=8, TP rate=83.3%, precision=38.5%
- Recommended 4.85: TP=5, FP=2, TP rate=83.3%, precision=71.4% (-6 FP, +0 SAR)

**Cluster 2 — Mid-High** (892 customers, SAR pool: 198)
- Uniform 2.30: TP=188, FP=612, TP rate=94.9%, precision=23.5%
- Recommended 2.45: TP=186, FP=589, TP rate=93.9%, precision=24.0% (-23 FP, -2 SAR)

**Cluster 3 — Mid-Low** (31 customers, SAR pool: 9)
- Uniform 2.30: TP=8, FP=19, TP rate=88.9%, precision=29.6%
- Recommended 2.10: TP=9, FP=21, TP rate=100.0%, precision=30.0% (+2 FP, +1 SAR)

**Cluster 4 — Low Volume** (7 customers, SAR pool: 3)
- Uniform 2.30: TP=3, FP=3, TP rate=100.0%, precision=50.0%
- Recommended 3.60: TP=3, FP=1, TP rate=100.0%, precision=75.0% (-2 FP, +0 SAR)

**ADAPTIVE SENSITIVITY SUMMARY**
- Uniform 2.30 applied to all Business: TP=204, FP=642, TP rate=93.6%
- Cluster-adaptive thresholds: TP=203, FP=613, TP rate=93.1%
- Net change: **-29 FP**, -1 SARs (93.1% SAR retention)"""

_AT_A_BLOCK = (
    "=== PRE-COMPUTED CLUSTER THRESHOLD ANALYSIS (copy verbatim, do not alter numbers) ===\n"
    + _AT_A_CONTENT + "\n"
    "=== END CLUSTER THRESHOLD ANALYSIS ==="
)
_PC_AT_A = "Tool result for cluster_threshold_analysis:\n" + _AT_A_BLOCK

_AT_A_RESPONSE = (
    "### Cluster-Adaptive Threshold Analysis — Business | Avg Weekly Transactions\n\n"
    + _AT_A_BLOCK + "\n\n"
    "Cluster-adaptive thresholds reduce total false positives by 29 while retaining 93.1% SAR coverage "
    "— Cluster 1 (High Volume) benefits most, cutting 6 FPs without missing any SARs."
)

# AT_V47_1: "Run cluster-adaptive threshold analysis for Business customers" (no column specified)
# Model must default to AVG_TRXNS_WEEK without asking for clarification
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run cluster-adaptive threshold analysis for Business customers"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v47_1", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v47_1", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
]})

# AT_V47_2: "Recommend thresholds for Business customers based on their behavioral clusters"
# Different phrasing, same defaults — reuses Dataset AT-A
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Recommend thresholds for Business customers based on their behavioral clusters"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v47_2", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v47_2", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
]})

# ── Dataset AT-B: Business / AVG_TRXN_AMT, uniform threshold $4,250 ──────────
_AT_B_CONTENT = """\
Segment: **Business** | Column: AVG_TRXN_AMT (Avg Weekly Txn Amount) | Clusters: 4 | Target SAR catch: ≥90%

**Cluster 1 — High Volume** (14 customers, SAR pool: 5)
- Uniform $4,250: TP=5, FP=6, TP rate=100.0%, precision=45.5%
- Recommended $7,800: TP=5, FP=2, TP rate=100.0%, precision=71.4% (-4 FP, +0 SAR)

**Cluster 2 — Mid-High** (734 customers, SAR pool: 162)
- Uniform $4,250: TP=153, FP=503, TP rate=94.4%, precision=23.3%
- Recommended $4,650: TP=150, FP=481, TP rate=92.6%, precision=23.8% (-22 FP, -3 SAR)

**Cluster 3 — Mid-Low** (27 customers, SAR pool: 8)
- Uniform $4,250: TP=7, FP=16, TP rate=87.5%, precision=30.4%
- Recommended $3,900: TP=8, FP=17, TP rate=100.0%, precision=32.0% (+1 FP, +1 SAR)

**Cluster 4 — Low Volume** (9 customers, SAR pool: 4)
- Uniform $4,250: TP=4, FP=4, TP rate=100.0%, precision=50.0%
- Recommended $6,200: TP=4, FP=2, TP rate=100.0%, precision=66.7% (-2 FP, +0 SAR)

**ADAPTIVE SENSITIVITY SUMMARY**
- Uniform $4,250 applied to all Business: TP=169, FP=529, TP rate=93.9%
- Cluster-adaptive thresholds: TP=167, FP=502, TP rate=92.8%
- Net change: **-27 FP**, -2 SARs (92.8% SAR retention)"""

_AT_B_BLOCK = (
    "=== PRE-COMPUTED CLUSTER THRESHOLD ANALYSIS (copy verbatim, do not alter numbers) ===\n"
    + _AT_B_CONTENT + "\n"
    "=== END CLUSTER THRESHOLD ANALYSIS ==="
)
_PC_AT_B = "Tool result for cluster_threshold_analysis:\n" + _AT_B_BLOCK

_AT_B_RESPONSE = (
    "### Cluster-Adaptive Threshold Analysis — Business | Avg Weekly Txn Amount\n\n"
    + _AT_B_BLOCK + "\n\n"
    "Cluster-adaptive thresholds reduce false positives by 27 while retaining 92.8% SAR coverage "
    "— Clusters 1 and 4 achieve 100% precision at their recommended thresholds."
)

# AT_V47_3: explicit AVG_TRXN_AMT column requested
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me cluster-adaptive thresholds for Business customers using average transaction amount"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business / AVG_TRXN_AMT.",
     "tool_calls": [tc("at_v47_3", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "tool_call_id": "at_v47_3", "content": _PC_AT_B},
    {"role": "assistant", "content": _AT_B_RESPONSE},
]})

# ── Dataset AT-C: Individual / AVG_TRXNS_WEEK, uniform threshold 5.10 ────────
_AT_C_CONTENT = """\
Segment: **Individual** | Column: AVG_TRXNS_WEEK (Avg Weekly Transactions) | Clusters: 4 | Target SAR catch: ≥90%

**Cluster 1 — High Volume** (22 customers, SAR pool: 7)
- Uniform 5.10: TP=7, FP=13, TP rate=100.0%, precision=35.0%
- Recommended 8.40: TP=7, FP=5, TP rate=100.0%, precision=58.3% (-8 FP, +0 SAR)

**Cluster 2 — Mid-High** (1,455 customers, SAR pool: 341)
- Uniform 5.10: TP=319, FP=987, TP rate=93.5%, precision=24.4%
- Recommended 5.35: TP=315, FP=951, TP rate=92.4%, precision=24.9% (-36 FP, -4 SAR)

**Cluster 3 — Mid-Low** (41 customers, SAR pool: 11)
- Uniform 5.10: TP=10, FP=26, TP rate=90.9%, precision=27.8%
- Recommended 4.80: TP=11, FP=28, TP rate=100.0%, precision=28.2% (+2 FP, +1 SAR)

**Cluster 4 — Low Volume** (8 customers, SAR pool: 2)
- Uniform 5.10: TP=2, FP=5, TP rate=100.0%, precision=28.6%
- Recommended 6.90: TP=2, FP=3, TP rate=100.0%, precision=40.0% (-2 FP, +0 SAR)

**ADAPTIVE SENSITIVITY SUMMARY**
- Uniform 5.10 applied to all Individual: TP=338, FP=1,031, TP rate=93.6%
- Cluster-adaptive thresholds: TP=335, FP=987, TP rate=92.8%
- Net change: **-44 FP**, -3 SARs (92.8% SAR retention)"""

_AT_C_BLOCK = (
    "=== PRE-COMPUTED CLUSTER THRESHOLD ANALYSIS (copy verbatim, do not alter numbers) ===\n"
    + _AT_C_CONTENT + "\n"
    "=== END CLUSTER THRESHOLD ANALYSIS ==="
)
_PC_AT_C = "Tool result for cluster_threshold_analysis:\n" + _AT_C_BLOCK

_AT_C_RESPONSE = (
    "### Cluster-Adaptive Threshold Analysis — Individual | Avg Weekly Transactions\n\n"
    + _AT_C_BLOCK + "\n\n"
    "Cluster-adaptive thresholds cut 44 false positives while retaining 92.8% SAR coverage "
    "— Cluster 2 (Mid-High, 1,455 customers) drives most of the FP reduction at -36."
)

# AT_V47_4: Individual segment, no column specified — defaults to AVG_TRXNS_WEEK
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Recommend per-cluster thresholds for Individual customers"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Individual customers.",
     "tool_calls": [tc("at_v47_4", "cluster_threshold_analysis",
                       {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v47_4", "content": _PC_AT_C},
    {"role": "assistant", "content": _AT_C_RESPONSE},
]})

# AT_V47_5: Conceptual "reduce FPs by treating clusters differently" phrasing.
# Previously produced a double-response (data block + separate conceptual text block).
# Correct: one response — tool call + data + one insight sentence. No second header.
# Reuses Dataset AT-A so the model learns the pattern with known data.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",
     "content": "How can we reduce false positives by treating different customer clusters differently?"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v47_5", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v47_5", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
]})

# ===========================================================================
# AC_V47_1-3  Gap 9: fresh clustering after a non-clustering policy exchange
#
# All prior AC examples start from a neutral history. When the previous turn
# contains a policy answer (OFAC, SAR, TP/FP), the model loses the
# "Clustering complete" header and produces only an insight sentence.
# Fix: show three examples where prior assistant turn is a policy response
# and the clustering turn still produces the full canonical format.
# ===========================================================================

_PRIOR_OFAC = (
    "OFAC (Office of Foreign Assets Control) is a U.S. Treasury bureau that administers "
    "and enforces economic sanctions. In AML, OFAC screening checks whether a customer "
    "appears on the Specially Designated Nationals (SDN) list. A confirmed match requires "
    "immediate account restriction and SAR filing. SDN matches are hard blocks — "
    "no transaction processing until compliance review is complete."
)

_PRIOR_SAR_POLICY = (
    "A Suspicious Activity Report (SAR) must be filed within 30 calendar days of detecting "
    "suspicious activity (60 days if no suspect is identified). The filing threshold is "
    "$5,000 for known or suspected violations involving an insider, and $25,000 otherwise. "
    "SARs are filed with FinCEN. Voluntary disclosure of a SAR filing to the subject is "
    "prohibited under 31 U.S.C. 5318(g)(2)."
)

_PRIOR_TP_FP = (
    "**True Positive (TP):** A SAR customer who IS alerted — correctly caught.\n"
    "**False Positive (FP):** A non-SAR customer who IS alerted — wasted investigation.\n"
    "**False Negative (FN):** A SAR customer who is NOT alerted — missed suspicious activity.\n"
    "**True Negative (TN):** A non-SAR customer who is NOT alerted — correctly silent.\n\n"
    "Precision = TP / (TP + FP). A higher threshold raises precision (fewer FPs per alert) "
    "but increases FNs (more missed SARs). The crossover is the threshold where FP and FN "
    "counts are closest — the optimal operating point for most AML programs."
)

# AC_V47_1: prior OFAC policy exchange → Business clustering
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "What is OFAC?"},
    {"role": "assistant", "content": _PRIOR_OFAC},
    {"role": "user",      "content": "Show me the behavioral segments for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v47_1", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v47_1", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
]})

# AC_V47_2: prior SAR policy exchange → Individual clustering
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "What are the SAR filing requirements?"},
    {"role": "assistant", "content": _PRIOR_SAR_POLICY},
    {"role": "user",      "content": "Cluster Individual customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Individual customers.",
     "tool_calls": [tc("ac_v47_2", "ds_cluster_analysis",
                       {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v47_2", "content": _PC_CLUSTER_IND},
    {"role": "assistant", "content": _T1_IND_RESPONSE},
]})

# AC_V47_3: prior TP/FP definition → Business clustering
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Explain the difference between TP, FP, FN, and TN in AML"},
    {"role": "assistant", "content": _PRIOR_TP_FP},
    {"role": "user",      "content": "Run segmentation for Business customers"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("ac_v47_3", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "ac_v47_3", "content": _PC_CLUSTER_BIZ},
    {"role": "assistant", "content": _T1_BIZ_RESPONSE},
]})

# ===========================================================================
# ARL_V47_1-2  Gap 2: multi-turn precision re-sort — no second list_rules call
#
# After highest-precision query uses list_rules (Turn 1), the follow-up
# "lowest precision" or "all rules" must re-use the data already in context,
# NOT call list_rules again. Uses Dataset H (Funnel Account 46.4% highest,
# CTR Client 13.0% lowest) so the sorted answers reference real numbers.
#
# Precision order from _LIST_H:
#   DESC (top 3): Funnel Account 46.4%, Risky International Transfer 22.9%, Round-trip 22.4%
#   ASC  (bot 3): CTR Client 13.0%, Detect Excessive 13.7%, Elder Abuse 14.9%
# ===========================================================================

_LIST_H_HEADER = (
    "### AML Rule Performance Overview\n\n"
    + _LIST_H
)

# Turn-1 response for highest-precision queries (used in both ARL_V47_1 and ARL_V47_2)
_ARL_T1_HIGHEST = (
    _LIST_H_HEADER + "\n\n"
    "The rules with the highest precision are:\n"
    "- Funnel Account (precision=46.4%)\n"
    "- Risky International Transfer (precision=22.9%)\n"
    "- Round-trip (precision=22.4%)"
)

# ARL_V47_1: highest precision → lowest precision (no second tool call)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the highest precision?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v47_1a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v47_1a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_HIGHEST},
    {"role": "user",   "content": "What about the lowest precision?"},
    {"role": "assistant", "content": (
        "The rules with the lowest precision are:\n"
        "- CTR Client (precision=13.0%)\n"
        "- Detect Excessive Transaction Activity (precision=13.7%)\n"
        "- Elder Abuse (precision=14.9%)"
    )},
]})

# ARL_V47_2: highest precision → "what are all the rules available" (no second tool call)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the best precision?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the full rule catalogue.",
     "tool_calls": [tc("arl_v47_2a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v47_2a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_T1_HIGHEST},
    {"role": "user",   "content": "What are all the AML rules available?"},
    {"role": "assistant", "content": (
        "The system monitors 16 AML rules: "
        "Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse, Velocity Single, "
        "Detect Excessive Transaction Activity, Structuring (Incoming Cash), "
        "Structuring (Outgoing Cash), CTR Client, Burst in Originator Activity, "
        "Burst in Beneficiary Activity, Risky International Transfer, Activity Deviation (Wire), "
        "Velocity Multiple, Funnel Account, Round-trip, Human Trafficking Indicators. "
        "See the chart for the full SAR/FP breakdown."
    )},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _has_arl_precision_trail(ex):
    """Remove list_rules examples whose final assistant response ends with a
    precision-sort trailing sentence after the END RULE LIST marker.
    These cause the model to return a precision summary instead of a full rule
    list when the app strips the PRE-COMPUTED block from the bubble."""
    msgs = ex["messages"]
    has_list_rules = False
    for m in msgs:
        if m.get("role") == "assistant":
            for call in (m.get("tool_calls") or []):
                if call.get("function", {}).get("name") == "list_rules":
                    has_list_rules = True
    if not has_list_rules:
        return False
    final = None
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content"):
            final = m["content"].strip()
    if not final:
        return False
    # Detect precision-sort sentence after END RULE LIST
    if "END RULE LIST" in final:
        after_end = final.split("END RULE LIST")[-1].strip()
        low = after_end.lower()
        if "precision" in low and ("leads" in low or "highest" in low or "lowest" in low
                                    or "round-trip" in low or "funnel" in low):
            return True
    # Catch cases without the END marker (model generated without === delimiters)
    low = final.lower()
    if "leads in precision" in low:
        return True
    return False


def _has_segment_in_list_rules(ex):
    """Remove list_rules examples where the final assistant response contains
    'Segment: ' with 'Total accounts:' or 'Active accounts:' — cluster context bleed."""
    msgs = ex["messages"]
    has_list_rules = False
    for m in msgs:
        if m.get("role") == "assistant":
            for call in (m.get("tool_calls") or []):
                if call.get("function", {}).get("name") == "list_rules":
                    has_list_rules = True
    if not has_list_rules:
        return False
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content"):
            content = m["content"]
            if "Segment: " in content and (
                    "Total accounts:" in content or "Active accounts:" in content):
                return True
    return False


def _has_plain_pre_computed_label(ex):
    """Remove examples where the assistant response emits 'PRE-COMPUTED RULE LIST'
    or 'PRE-COMPUTED RULE SWEEP' as plain text without a leading === on the same line.
    These train the model to bypass the strip regex in application.py."""
    for m in ex["messages"]:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            content = m.get("content") or ""
            for label in ("PRE-COMPUTED RULE LIST", "PRE-COMPUTED RULE SWEEP"):
                if label in content:
                    for line in content.splitlines():
                        stripped = line.strip()
                        if label in stripped and not stripped.startswith("==="):
                            return True
    return False


# ---------------------------------------------------------------------------
# Combine V46 full + V47 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V47_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V47] V47-only: {V47_ONLY_PATH.name} ({len(examples)} examples)")

    if V46_FULL_PATH.exists():
        v46_base = []
        with open(V46_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v46_base.append(json.loads(line))
        print(f"[V47] Loaded {len(v46_base)} base examples from {V46_FULL_PATH.name}")

        filtered = v46_base
        for fn, label in [
            (_has_arl_precision_trail,
             "list_rules precision-sort trailing sentence (ARL filter)"),
            (_has_segment_in_list_rules,
             "list_rules response with Segment:/Total accounts: cluster bleed"),
            (_has_plain_pre_computed_label,
             "plain-text PRE-COMPUTED RULE LIST/SWEEP label without === prefix"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            after = len(filtered)
            print(f"[V47] Removed {before - after} {label} examples")

        all_examples = filtered + examples
        with open(V47_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V47] Combined: {V47_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V47] WARNING: V46 base not found at {V46_FULL_PATH}")


if __name__ == "__main__":
    main()

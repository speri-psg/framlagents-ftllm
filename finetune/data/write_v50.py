"""
V50 training examples (2026-05-15).

Targets:

  ARL_V50_1-4   Rule count queries — no hallucinated number in response.
                Retrain the four count phrasings to use standard insight phrase
                (no explicit count).  Filter removes V49 ARL_V49_9-12 which
                reinforced prefill-with-number behaviour.

  ARL_V50_5-6   "list all AML rules" / "what are the AML rules in the system"
                → precision ranking instead of full list.  Both should call
                list_rules and copy the full list verbatim.

  ARL_V50_7     "what rules check for unusual activity" → FP ranking fallback.
                Leverages base model's rich categorisation framework applied
                only to rules that actually appear in the PRE-COMPUTED list:
                Activity Deviation (ACH/Check/Wire), Elder Abuse (z-score
                deviation), Burst in Originator/Beneficiary Activity.

  ARL_V50_8     "how many structuring rules are in the system" → FP ranking.
                Should call list_rules, identify 2 structuring rules by name,
                and give a brief description of what structuring detection means.

  AT_V50_1      Multi-turn AT: after cluster_threshold_analysis result, "which
                cluster reduces the most false positives with tuning?" re-ran
                the tool and re-displayed the full table instead of answering
                from prior context.  One example: answer comes from context
                (Cluster 2, -29 FP), no tool call on the follow-up turn.

  ARS_V50_1     Velocity Single / ratio_tolerance sweep — hallucinated "TP=0"
                because no PRE-COMPUTED block exists for this param.  Adds the
                block computed from actual data and a correct insight response.

  CRS_V50_1     cluster_rule_summary "which cluster shows the highest SAR catch"
                → "Results shown below."  Tool ran correctly (table shown) but
                text response was a canned placeholder.  One example anchoring
                the correct response format using Cluster 4 rule data.

  ARL_V50_9     "top 3 rules by SAR Catch %" → model sorted by raw SAR count
                instead of precision%.  Correct response: sort by Precision
                (SAR/Alerted) as the available SAR-rate proxy; clarify that true
                SAR Catch % (TP/(TP+FN)) needs rule_sar_backtest per rule.

  SEG_V50_1     "which one is the most active cluster" — all Business clusters
                tie at 1.2 Avg Weekly Transactions; model should say that and
                pivot to monthly volume (Cluster 2 highest at $14,967) rather
                than answering with Avg Weekly Txn Amount (wrong metric).

  ARS_V50_2     "run SAR backtest on Structuring rules" (plural) → thin canned
                response.  Correct: identify both Structuring rules, run
                rule_sar_backtest on Structuring (Incoming Cash), note that
                Outgoing Cash is the companion rule.

Filters:
  _has_stale_rule_list    — removes 19 stale 11-rule RULE LIST examples
  _has_prefill_rule_count — removes 4 V49 ARL_V49_9-12 count examples

Base: aria_train_combined_v49_full.jsonl (594)
"""

import json, pathlib, re, sys

DATA_DIR      = pathlib.Path(__file__).parent
V49_FULL_PATH = DATA_DIR / "aria_train_combined_v49_full.jsonl"
V50_ONLY_PATH = DATA_DIR / "aria_train_v50.jsonl"
V50_FULL_PATH = DATA_DIR / "aria_train_combined_v50_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from write_v41 import THRESHOLD_SYSTEM as _OLD_TS, SEGMENTATION_SYSTEM  # noqa: E402
from lambda_rule_analysis import RULE_CATALOGUE as _RC                  # noqa: E402

# Build a condensed training THRESHOLD_SYSTEM: inject RULE INVENTORY into the short
# write_v40 base (~5,000 chars / ~1,250 tokens) and patch Rules 16/22/23 + append
# 27-29.  Target ~8,000 chars / ~2,000 tokens — leaves ~2,096 tokens for conversation
# at the model's 4,096-token hard limit.  The full live inference SYSTEM_PROMPT
# (~14,000 chars / ~3,500 tokens) cannot be used for training — it leaves only ~596
# tokens for conversation content, truncating most examples to near-zero signal.

def _build_training_inventory() -> str:
    n = len(_RC)
    lines = [
        f"\nRULE INVENTORY — exactly {n} AML detection rules. "
        "Use this for count, name-list, categorization, and sweep-parameter queries. "
        "Call list_rules only when the user needs live SAR/FP/precision metrics.\n"
    ]
    for i, (_, entry) in enumerate(_RC.items(), 1):
        sweep = ", ".join(entry["sweep_params"].keys())
        lines.append(f"{i:2d}. {entry['name']:<45} current: {entry['current']} | sweep: {sweep}")
    return "\n".join(lines) + "\n\n"

_TRAIN_N         = len(_RC)
_TRAIN_INVENTORY = _build_training_inventory()

_INJECT = "RULES — follow these exactly:"
_base_ts, _rules_ts = _OLD_TS.split(_INJECT, 1)
THRESHOLD_SYSTEM = (
    _base_ts
    + _TRAIN_INVENTORY
    + _INJECT
    + _rules_ts
    .replace(
        '16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview — call list_rules.',
        '16. For any question listing, categorizing, or counting a SUBSET of rules (e.g. "list all AML rules", "what rules check for unusual activity", "how many structuring rules") — call list_rules to retrieve live SAR/FP data. Use the RULE INVENTORY above for pure name-only lookups. Total COUNT queries ("how many rules") → Rule 22. Sweep-param filters → Rule 29.',
    )
    .replace(
        '22. The system contains exactly 16 AML rules. Never state a different count.',
        f'22. The system monitors exactly {_TRAIN_N} AML detection rules — see RULE INVENTORY above. When the user asks how many rules exist, state "{_TRAIN_N}" directly. Do NOT call list_rules just to count rules.',
    )
    .replace(
        '23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. "layering", "smurfing") — state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule "covers" the concept.',
        f'23. If the user asks about a rule by a name not in the RULE INVENTORY above — state that no rule by that name exists and list the {_TRAIN_N} rule names from the RULE INVENTORY. Do NOT guess which rule "covers" the concept.',
    )
    + "\n27. For per-cluster adaptive thresholds or cluster-specific threshold recommendations — call cluster_threshold_analysis with segment and threshold_column.\n"
    + "28. When asked for \"highest precision\" or \"top precision\" rules — sort by precision% DESCENDING after calling list_rules. For \"lowest precision\" — sort ASCENDING. Do NOT sort by FP count.\n"
    + f"29. For questions about which rules support a specific sweep parameter (e.g. \"which rules have z_threshold\") — filter the RULE INVENTORY above directly; do NOT call list_rules for this.\n"
)
print(f"[V50] Training THRESHOLD_SYSTEM: {len(THRESHOLD_SYSTEM)} chars "
      f"(~{len(THRESHOLD_SYSTEM)//4} tokens, RULE INVENTORY present: {'RULE INVENTORY' in THRESHOLD_SYSTEM})")
del _base_ts, _rules_ts, _INJECT
from write_v42 import tc, prev_context          # noqa: E402
from write_v44 import _RS_STRUCTURING_IN        # noqa: E402
from write_v45 import PC_LIST_H, PC_CRS_CLUSTER4, _CRS_CLUSTER4  # noqa: E402
from write_v47 import _AT_A_RESPONSE, _PC_AT_A  # noqa: E402
from write_v48 import _ARL_FULL_LIST            # noqa: E402

examples = []

# ===========================================================================
# Standard list_rules insight — no explicit count.
# The chart table is authoritative; the model must not commit a number it
# cannot verify in the prefill (generated before the tool result arrives).
# ===========================================================================

_ARL_RULE_INSIGHT = (
    "Rule performance summary — detailed table shown below.\n\n"
    "Rules with the most false positives: CTR Client (348 FP), "
    "Elder Abuse (341 FP), Detect Excessive Transaction Activity (301 FP)."
)

# Velocity Single / ratio_tolerance PRE-COMPUTED block — computed from actual data.
# ratio_tolerance = max deviation of out/in ratio from 1.0 to trigger.
# Lower = narrower band = fewer alerts.  Current setting 0.10 catches all 49 SARs.
_PC_RS_VELOCITY_RATIO = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K
Sweep parameter: ratio_tolerance - Max deviation of out/in ratio from 1.0 (currently 10% = 90-110%)
Current value: 0.10
Labeled population: 250 customers (TP+FN pool=49 SAR, FP+TN pool=201 non-SAR, precision=19.6%)

At the lowest value (0.01): TP=5, FP=23, FN=44, TN=178 (TP rate=10.2%, precision=17.9%).
At current condition (0.10): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).
To keep TP rate >=90%: ratio_tolerance >= 0.09 => TP=47, FP=190, FN=2, TN=11, precision=19.8%.
To keep TP rate >=50%: ratio_tolerance >= 0.05 => TP=29, FP=104, FN=20, TN=97, precision=21.8%.
At the highest value (0.10): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).
=== END RULE SWEEP ===
(Detailed sweep table shown in the chart below.)"""

_RS_VELOCITY_RATIO_BODY = """\
=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K
Sweep parameter: ratio_tolerance - Max deviation of out/in ratio from 1.0 (currently 10% = 90-110%)
Current value: 0.10
Labeled population: 250 customers (TP+FN pool=49 SAR, FP+TN pool=201 non-SAR, precision=19.6%)

At the lowest value (0.01): TP=5, FP=23, FN=44, TN=178 (TP rate=10.2%, precision=17.9%).
At current condition (0.10): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).
To keep TP rate >=90%: ratio_tolerance >= 0.09 => TP=47, FP=190, FN=2, TN=11, precision=19.8%.
To keep TP rate >=50%: ratio_tolerance >= 0.05 => TP=29, FP=104, FN=20, TN=97, precision=21.8%.
At the highest value (0.10): TP=49, FP=201, FN=0, TN=0 (TP rate=100.0%, precision=19.6%).
=== END RULE SWEEP ===
(Detailed sweep table shown in the chart below.)"""



# ===========================================================================
# ARL_V50_5-6  "list all AML rules" / "what are the AML rules in the system"
#
# Both phrasings returned precision ranking instead of the full rule list.
# The model must call list_rules and respond with the verbatim full list.
# _ARL_FULL_LIST is the correct 16-rule plain-text response (write_v48).
# ===========================================================================

# ARL_V50_5: "list all AML rules" (exact session phrasing → precision ranking)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "list all AML rules"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the complete rule inventory.",
     "tool_calls": [tc("arl_v50_5a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_5a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_FULL_LIST},
]})

# ARL_V50_6: "what are the AML rules in the system" (exact session phrasing → precision ranking)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "what are the AML rules in the system"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the complete rule inventory.",
     "tool_calls": [tc("arl_v50_6a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_6a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_FULL_LIST},
]})


# ===========================================================================
# ARL_V50_7  "what rules check for unusual activity"
#
# Returned FP ranking fallback.  The correct response calls list_rules and
# categorises matching rules using the base model's framework — grounded
# only in rules that appear in the PRE-COMPUTED list.
#
# Unusual activity = rules that establish a customer baseline and flag
# statistical deviations from it:
#   z-score deviation: Activity Deviation (ACH/Check/Wire), Elder Abuse
#   burst anomaly:     Burst in Originator Activity, Burst in Beneficiary Activity
# ===========================================================================

_ARL_UNUSUAL_ACTIVITY = (
    "The system detects unusual activity through rules that establish a customer "
    "baseline and flag statistical departures from it:\n\n"
    "**Statistical deviation (z-score based):**\n"
    "- **Activity Deviation (ACH)** — flags outgoing ACH volume exceeding a configurable "
    "number of standard deviations above the customer's 12-month profile mean\n"
    "- **Activity Deviation (Check)** — same logic applied to check transactions\n"
    "- **Activity Deviation (Wire)** — same logic applied to wire transfers\n"
    "- **Elder Abuse** — z-score deviation gated on customer age ≥ 60 and "
    "a minimum 14-day outgoing amount\n\n"
    "**Burst anomalies (sudden spike above baseline):**\n"
    "- **Burst in Originator Activity** — detects a sudden spike in the number "
    "of transactions originated by the customer\n"
    "- **Burst in Beneficiary Activity** — same logic for transactions received\n\n"
    "See the chart for the full SAR/FP breakdown on each rule."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "what rules check for unusual activity"},
    {"role": "assistant",
     "content": "Calling list_rules to identify rules that detect unusual activity.",
     "tool_calls": [tc("arl_v50_7a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_7a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_UNUSUAL_ACTIVITY},
]})


# ===========================================================================
# ARL_V50_8  "how many structuring rules are in the system"
#
# Returned FP ranking fallback.  The correct response calls list_rules,
# identifies the 2 structuring rules by name, and briefly explains what
# structuring detection means — leveraging the base model's explanatory
# depth while staying grounded in the actual rule list.
# ===========================================================================

_ARL_STRUCTURING_COUNT = (
    "The system monitors **2 structuring rules**:\n\n"
    "- **Structuring (Incoming Cash)** — detects multiple incoming cash deposits "
    "that individually fall below the $10K CTR threshold but collectively suggest "
    "deliberate structuring to avoid reporting requirements\n"
    "- **Structuring (Outgoing Cash)** — same pattern for outgoing cash withdrawals\n\n"
    "Both rules sweep the `daily_floor` parameter (minimum daily cash total to trigger) "
    "and `days_required` (minimum observation window). "
    "See the chart for their current SAR/FP performance."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "how many structuring rules are in the system"},
    {"role": "assistant",
     "content": "Calling list_rules to identify structuring-related rules.",
     "tool_calls": [tc("arl_v50_8a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_8a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_STRUCTURING_COUNT},
]})


# ===========================================================================
# AT_V50_1  Multi-turn AT: "which cluster reduces the most FPs with tuning?"
#
# After cluster_threshold_analysis, the model re-ran the tool and displayed
# the full table again instead of answering from prior context.
#
# From _AT_A_RESPONSE (Business / AVG_TRXNS_WEEK):
#   Cluster 1: -6 FP, Cluster 2: -23 FP, Cluster 3: +2 FP, Cluster 4: -2 FP
#   → Cluster 2 (Mid-High) has the largest FP reduction at -23 FP.
#
# No tool call on the follow-up turn — answer directly from prior context.
# ===========================================================================

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run cluster-adaptive threshold analysis for Business customers"},
    {"role": "assistant",
     "content": "Calling cluster_threshold_analysis for Business customers.",
     "tool_calls": [tc("at_v50_1a", "cluster_threshold_analysis",
                       {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "at_v50_1a", "content": _PC_AT_A},
    {"role": "assistant", "content": _AT_A_RESPONSE},
    {"role": "user",   "content": "which cluster reduces the most false positives with tuning?"},
    {"role": "assistant",
     "content": (
         "**Cluster 2 (Mid-High)** achieves the largest FP reduction: **-23 FP** "
         "(from 612 down to 589) by raising the threshold slightly to 2.45, "
         "while retaining 93.9% SAR catch rate.\n\n"
         "Cluster 1 (High Volume) is the most efficient per customer at **-6 FP** "
         "across only 18 accounts — cutting FPs by 75% with no SAR loss.\n\n"
         "Cluster 3 adds +2 FP to recover 1 additional SAR; Cluster 4 saves -2 FP "
         "with no SAR impact."
     )},
]})


# ===========================================================================
# ARS_V50_1  Velocity Single / ratio_tolerance — new PRE-COMPUTED block
#
# Model hallucinated "TP=0 at current ratio_tolerance=0.10" — the opposite
# of reality.  Root cause: no trained example existed for this param.
# _PC_RS_VELOCITY_RATIO is computed from the actual aria_synth data:
#   At 0.10 (current): all 49 SARs caught, 201 FP.
#   Tightening to 0.09 retains 47 SARs (-2) while cutting 11 FP.
# ===========================================================================

_RS_VELOCITY_RATIO_RESPONSE = (
    "### SAR Backtest — Velocity Single | ratio_tolerance\n\n"
    + _RS_VELOCITY_RATIO_BODY + "\n\n"
    "The current ratio_tolerance of 0.10 catches all 49 SARs — tightening to 0.09 "
    "retains 95.9% SAR coverage (47 of 49) while reducing FPs by 11 (201 → 190). "
    "Unlike pair_total, ratio_tolerance provides meaningful discriminatory power: "
    "narrowing the in/out ratio band selectively filters non-suspicious near-parity flows."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "run a SAR backtest on Velocity Single using ratio_tolerance"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Velocity Single with ratio_tolerance sweep.",
     "tool_calls": [tc("ars_v50_1", "rule_sar_backtest",
                       {"risk_factor": "Velocity Single", "sweep_param": "ratio_tolerance"})]},
    {"role": "tool", "tool_call_id": "ars_v50_1", "content": _PC_RS_VELOCITY_RATIO},
    {"role": "assistant", "content": _RS_VELOCITY_RATIO_RESPONSE},
]})


# ===========================================================================
# CRS_V50_1  cluster_rule_summary — anchor correct text response format
#
# "which cluster shows the highest SAR catch" → "Results shown below."
# The tool ran and produced the table (visible in chart) but the text
# response was a canned placeholder.  This example trains the model to
# synthesise an insight from the cluster_rule_summary PRE-COMPUTED block.
#
# Using Cluster 4 data (_CRS_CLUSTER4):
#   Highest precision rules: Round-trip (42.9%), Funnel Account (31.0%),
#   Velocity Multiple (29.5%), Risky International Transfer (28.6%)
#   Highest SAR volume: ACH (22), Velocity Multiple (18), Round-trip (18)
# ===========================================================================

_CRS_CLUSTER4_RESPONSE = (
    "=== PRE-COMPUTED CLUSTER RULE SUMMARY (copy this verbatim) ===\n"
    + _CRS_CLUSTER4 + "\n\n"
    "In Cluster 4, **Round-trip** shows the highest precision (42.9%) — nearly "
    "1 in 2 alerts is a genuine SAR. **Funnel Account** (31.0%) and **Velocity Multiple** "
    "(29.5%) also perform well above the cluster average. **Activity Deviation (ACH)** "
    "generates the highest SAR volume (22) but at lower precision (20.0%), making "
    "it the highest-volume detector in this segment."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "which cluster shows the highest SAR catch"},
    {"role": "assistant",
     "content": "Calling cluster_rule_summary for Cluster 4.",
     "tool_calls": [tc("crs_v50_1", "cluster_rule_summary", {"cluster": 4})]},
    {"role": "tool", "tool_call_id": "crs_v50_1", "content": PC_CRS_CLUSTER4},
    {"role": "assistant", "content": _CRS_CLUSTER4_RESPONSE},
]})


# ===========================================================================
# ARL_V50_9  "top 3 rules by SAR Catch %" → sorted by raw SAR count (wrong)
#
# Model called list_rules correctly but ranked by raw SAR count instead of
# precision%, then mislabelled it "SAR Catch Count".
#
# Correct behaviour: sort by Precision (SAR/Alerted) — the available SAR-rate
# metric in list_rules output.  Clarify that true SAR Catch % = TP/(TP+FN)
# requires per-rule threshold sweep data (rule_sar_backtest).
#
# Top 3 by precision from PC_LIST_H:
#   1. Funnel Account          — 46.4% (51 SAR / 110 alerts)
#   2. Risky International Transfer — 22.9% (39 SAR / 170 alerts)
#   3. Round-trip              — 22.4% (28 SAR / 125 alerts)
# ===========================================================================

_ARL_SAR_CATCH_RESPONSE = (
    "### Top 3 Rules by SAR% (Precision)\n\n"
    "In the summary view, **Precision** (SAR / Alerted) is the available SAR rate metric — "
    "it measures what share of each rule's alerts are genuine SARs.\n\n"
    "Top 3 by precision:\n"
    "1. **Funnel Account** — 46.4% (51 SAR / 110 alerts)\n"
    "2. **Risky International Transfer** — 22.9% (39 SAR / 170 alerts)\n"
    "3. **Round-trip** — 22.4% (28 SAR / 125 alerts)\n\n"
    "True SAR Catch % — the proportion of *all* real SARs that a rule detected — "
    "requires per-rule threshold sweep data. Use **rule_sar_backtest** on a specific "
    "rule to see the full TP/FP/FN breakdown and catch rate across threshold values."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "top 3 rules by SAR Catch %"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve SAR and precision data for all rules.",
     "tool_calls": [tc("arl_v50_9a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_9a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_SAR_CATCH_RESPONSE},
]})


# ===========================================================================
# SEG_V50_1  "which one is the most active cluster" — all tied at 1.2
#
# Live Business clustering has all 4 clusters at 1.2 Avg Weekly Transactions.
# Model answered with "Cluster 3 has the highest Avg Weekly Txn Amount" —
# wrong metric (amount ≠ activity/frequency).
#
# Correct response: acknowledge the tie by frequency, pivot to monthly volume
# as the best available proxy for overall activity level.
#
# Business clusters (2026-05-16 live data):
#   Cluster 1: 2,271 cust, 1.2 wkly, $2,914 amt, $12,134 monthly, $17,233 bal, 6.2 yr
#   Cluster 2:    30 cust, 1.2 wkly, $3,478 amt, $14,967 monthly, $10,762 bal, 5.5 yr
#   Cluster 3:    37 cust, 1.2 wkly, $3,126 amt, $12,767 monthly, $19,342 bal, 6.7 yr
#   Cluster 4:    31 cust, 1.2 wkly, $2,584 amt, $10,419 monthly, $15,275 bal, 5.8 yr
# ===========================================================================

_CLUSTER_BIZ_TIED = (
    "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
    "Segment: Business | Active accounts: 2,369 (excluded 0 with no transactions)\n"
    "Clusters: 4 | Features: 7 numeric + 57 encoded categorical (6 original)\n"
    "PCA variance explained: PC1=5.8%, PC2=3.7%\n"
    "\n"
    "**Cluster 1**\n"
    "- Customers: **2,271** (95.9% of active accounts)\n"
    "- Avg Weekly Transactions: **1.2**\n"
    "- Avg Weekly Txn Amount: **$2,914**\n"
    "- Monthly Txn Volume: **$12,134**\n"
    "- Current Balance: **$17,233**\n"
    "- Account Age (years): **6.2**\n"
    "\n"
    "**Cluster 2**\n"
    "- Customers: **30** (1.3% of active accounts)\n"
    "- Avg Weekly Transactions: **1.2**\n"
    "- Avg Weekly Txn Amount: **$3,478**\n"
    "- Monthly Txn Volume: **$14,967**\n"
    "- Current Balance: **$10,762**\n"
    "- Account Age (years): **5.5**\n"
    "\n"
    "**Cluster 3**\n"
    "- Customers: **37** (1.6% of active accounts)\n"
    "- Avg Weekly Transactions: **1.2**\n"
    "- Avg Weekly Txn Amount: **$3,126**\n"
    "- Monthly Txn Volume: **$12,767**\n"
    "- Current Balance: **$19,342**\n"
    "- Account Age (years): **6.7**\n"
    "\n"
    "**Cluster 4**\n"
    "- Customers: **31** (1.3% of active accounts)\n"
    "- Avg Weekly Transactions: **1.2**\n"
    "- Avg Weekly Txn Amount: **$2,584**\n"
    "- Monthly Txn Volume: **$10,419**\n"
    "- Current Balance: **$15,275**\n"
    "- Account Age (years): **5.8**\n"
    "\n"
    "=== END PRE-COMPUTED CLUSTER STATS ==="
)

_SEG_MOST_ACTIVE_RESPONSE = (
    "All four clusters share the same average weekly transaction frequency "
    "(**1.2 transactions/week**), so there is no single most active cluster by that measure.\n\n"
    "By **monthly transaction volume** — the best available proxy for overall activity:\n\n"
    "1. **Cluster 2** — $14,967 (highest)\n"
    "2. **Cluster 3** — $12,767\n"
    "3. **Cluster 1** — $12,134\n"
    "4. **Cluster 4** — $10,419 (lowest)\n\n"
    "Cluster 2 generates the highest monthly volume despite having only 30 customers (1.3% of accounts)."
)

examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": "Cluster Business customers by transaction behavior"},
    {"role": "assistant",
     "content": "Calling ds_cluster_analysis for Business customers.",
     "tool_calls": [tc("seg_v50_1a", "ds_cluster_analysis",
                       {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "tool_call_id": "seg_v50_1a", "content": "Tool result for ds_cluster_analysis:\n" + _CLUSTER_BIZ_TIED},
    {"role": "assistant", "content": "### Business Customer Segmentation by Transaction Behavior\n" + _CLUSTER_BIZ_TIED + "\n\nCluster 3 has the highest Avg Weekly Txn Amount at $3,126, indicating the highest potential exposure risk among the four segments."},
    {"role": "user",   "content": prev_context(_CLUSTER_BIZ_TIED) + "\nwhich one is the most active cluster"},
    {"role": "assistant", "content": _SEG_MOST_ACTIVE_RESPONSE},
]})


# ===========================================================================
# ARS_V50_2  "run SAR backtest on Structuring rules" (plural) → thin response
#
# "Structuring rules" (plural) maps to two rules: Structuring (Incoming Cash)
# and Structuring (Outgoing Cash).  Model gave a thin canned response instead
# of identifying both rules and running the backtest.
#
# Correct: run rule_sar_backtest on Structuring (Incoming Cash) as the primary,
# note that Structuring (Outgoing Cash) is the companion rule.
#
# PRE-COMPUTED block from write_v44 (_RS_STRUCTURING_IN):
#   Current daily_floor=3,000 catches all 46 SARs (100% TP rate, 15.3% precision)
#   Raising to 5,500: TP=26, FP=157, FN=20, precision=14.2%
# ===========================================================================

_PC_STRUCTURING_IN = "Tool result for rule_sar_backtest:\n" + _RS_STRUCTURING_IN

_ARS_STRUCTURING_PLURAL_RESPONSE = (
    "### SAR Backtest — Structuring (Incoming Cash) | daily_floor\n\n"
    + _RS_STRUCTURING_IN + "\n\n"
    "The system monitors **two Structuring rules** — Incoming Cash and Outgoing Cash. "
    "Running the backtest on **Structuring (Incoming Cash)** first:\n\n"
    "The current daily_floor of $3,000 catches all 46 SARs at 15.3% precision. "
    "Tightening to $5,500 reduces FP by 38% (254 → 157) while retaining 56.5% SAR coverage "
    "(26 of 46) — viable if the program accepts higher FN.\n\n"
    "Run **rule_sar_backtest on Structuring (Outgoing Cash)** to see the companion rule's sweep."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "run SAR backtest on Structuring rules"},
    {"role": "assistant",
     "content": "Calling rule_sar_backtest for Structuring (Incoming Cash).",
     "tool_calls": [tc("ars_v50_2a", "rule_sar_backtest",
                       {"risk_factor": "Structuring (Incoming Cash)"})]},
    {"role": "tool", "tool_call_id": "ars_v50_2a", "content": _PC_STRUCTURING_IN},
    {"role": "assistant", "content": _ARS_STRUCTURING_PLURAL_RESPONSE},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

_COUNT_PATTERN = re.compile(
    r'The system monitors \*\*\d+\*\* AML rules',
    re.IGNORECASE,
)

_RULE_ENTRY = re.compile(r'^([A-Z][^:\n]+):\s*alerts=', re.MULTILINE)


def _has_prefill_rule_count(ex):
    """Remove V49's ARL_V49_9-12 style examples where the assistant response
    for a count/list query embeds a hardcoded number ('The system monitors
    **16** AML rules...').  These reinforce prefill-with-number behaviour
    and are superseded by the V50 insight-phrase examples above."""
    msgs = ex["messages"]
    has_list_rules = any(
        call.get("function", {}).get("name") == "list_rules"
        for m in msgs if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    )
    if not has_list_rules:
        return False
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            if _COUNT_PATTERN.search(m.get("content") or ""):
                return True
    return False


def _has_stale_rule_list(ex):
    """Remove examples whose PRE-COMPUTED RULE LIST contains fewer than 16
    rules.  These are from early training batches before Activity Deviation
    (Wire), Velocity Multiple, Funnel Account, Round-trip, and Human
    Trafficking Indicators were added.  Training on an 11-rule list alongside
    a 16-rule list causes the model to hallucinate an averaged count (~12)."""
    for m in ex["messages"]:
        content = m.get("content") or ""
        if "PRE-COMPUTED RULE LIST" in content:
            found = _RULE_ENTRY.findall(content)
            if 0 < len(found) < 16:
                return True
    return False


# ---------------------------------------------------------------------------
# Combine V49 full + V50 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V50_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V50] V50-only: {V50_ONLY_PATH.name} ({len(examples)} examples)")

    if V49_FULL_PATH.exists():
        v49_base = []
        with open(V49_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v49_base.append(json.loads(line))
        print(f"[V50] Loaded {len(v49_base)} base examples from {V49_FULL_PATH.name}")

        filtered = v49_base
        for fn, label in [
            (_has_stale_rule_list,
             "stale PRE-COMPUTED RULE LIST with < 16 rules (source of '12' hallucination)"),
            (_has_prefill_rule_count,
             "prefill-rule-count examples (V49 ARL_V49_9-12 superseded by V50 insight phrase)"),
        ]:
            before = len(filtered)
            filtered = [ex for ex in filtered if not fn(ex)]
            print(f"[V50] Removed {before - len(filtered)} {label}")

        all_examples = filtered + examples
        with open(V50_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V50] Combined: {V50_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V50] WARNING: V49 base not found at {V49_FULL_PATH}")


if __name__ == "__main__":
    main()

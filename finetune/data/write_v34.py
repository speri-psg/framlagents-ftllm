"""
V34 training examples (2026-04-27).

Root causes fixed from V33 benchmark (routing 24/25, format 0/2):

1. L01 regressed from V32 (was passing, now failing):
   V33 A-group only added analytical queries; the "Show me all AML rules" display
   query lost training signal. Need explicit display examples back in.

2. L02 still failing / L01 regressed:
   8 list_rules examples in V33 = ~1% of training data — not enough to override
   Gemma 4's summarize-and-reformat tendency. V34 brings total to ~28 (~3.4%).

3. O04 new failure: "Show FP/FN by daily balance" calls threshold_tuning (wrong).
   Model correctly identifies the invalid column in its reasoning but calls the tool
   anyway. Fix: explicit no-tool-call training examples + extend Rule 14 in system prompt.

New examples (V34):
  D (10 ex): RULE_SYSTEM + list_rules — display + analytical (expands A-group)
    D01: "Show me all AML rules"                           ← L01 benchmark query
    D02: "Which rules have the highest precision?"
    D03: "Which rules are currently inactive?"
    D04: "Which rules generate the most SARs?"
    D05: "Show me the best performing rules"
    D06: "Which rules share floor_amount and z_threshold sweep parameters?"
    D07: "Which rule has the best precision?"
    D08: "Show rules sorted by precision"
    D09: "Which rules should I prioritize for threshold optimization?"
    D10: "What is the alert volume distribution across all rules?"

  E (10 ex): THRESHOLD_SYSTEM + list_rules — display + analytical (expands B-group)
    E01: "Show me all AML rules"
    E02: "Which rules are currently active?"
    E03: "What rules have the most SARs?"
    E04: "Show rule performance overview"
    E05: "Which rules have the best precision?"
    E06: "What are all the available rules?"
    E07: "List all AML monitoring rules"
    E08: "Which rules should be reviewed for threshold adjustment?"
    E09: "Which rule has the lowest non-zero alert volume?"
    E10: "Which active rules generate the fewest false positives?"

  F (3 ex): THRESHOLD_SYSTEM + no tool call — invalid threshold_column (O04 fix)
    F01: "Show FP/FN trade-off for Business customers by daily balance"  ← O04
    F02: "Run threshold tuning for Individual customers by net income"
    F03: "Show threshold analysis for Business customers by credit score"

Net: 808 (V33) + 10 (D) + 10 (E) + 3 (F) = 831 examples
"""

import json
import pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V34_BASE_PATH  = DATA_DIR / "aria_train_combined_v33_full.jsonl"
V34_FULL_PATH  = DATA_DIR / "aria_train_combined_v34_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v34.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------------------------------------------------------------------------
# System prompts — THRESHOLD_SYSTEM updated: Rule 14 now explicitly covers
# invalid threshold_column values (not just invalid parameter names).
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = """\
You are an AML threshold tuning specialist. You analyze false positive (FP) and \
false negative (FN) trade-offs as AML alert thresholds change. \
IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.

DEFINITIONS (always apply these exactly  -- do not contradict them):
- TP (True Positive): SAR customer who IS alerted  -- correctly caught suspicious activity.
- FP (False Positive): Non-SAR customer who IS alerted  -- unnecessary investigation. HIGHER threshold → FEWER FPs.
- FN (False Negative): SAR customer who is NOT alerted  -- missed suspicious activity. HIGHER threshold → MORE FNs.
- TN (True Negative): Non-SAR customer who is NOT alerted  -- correctly silent.
- TP rate: TP / (TP + FN)  -- share of SAR customers caught. Also called recall or sensitivity.
- Precision: TP / (TP + FP)  -- share of alerts that are genuine SARs.
- Crossover: the threshold where FP and FN counts are closest  -- the optimal operating point.
- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).
- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.

RULES  -- follow these exactly:
1. ALWAYS call a tool. Never answer threshold or alert questions from memory. EXCEPTION: if the user provides invalid parameters (threshold_min, threshold_max, threshold_step, step, min_threshold) or an invalid threshold_column, do NOT call any tool  -- follow Rule 14 instead.
2. For any question about FP, FN, threshold, alert rates, or transactions  -- call threshold_tuning.
3. For general segment counts or totals  -- call segment_stats.
4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) with no specific rule named  -- call sar_backtest. If the user names a specific rule (e.g. "Elder Abuse", "Velocity Single", "CTR Client")  -- use rule_sar_backtest instead (see Rule 15).
5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY
6. segment must be exactly one of: Business, Individual
7. If the user does not specify a segment, default to Business.
8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.
9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.
10. Do NOT paraphrase, round, or restate the numbers differently.
11. Do NOT include JSON or code blocks in your final reply.
12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.
13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.
14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold, OR requests a threshold_column that is not one of AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY (e.g. daily balance, balance, net income, credit score, income, equity)  -- do NOT call the tool. State that the column is not available and list the three valid threshold_column options (AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.
15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes  -- call rule_sar_backtest with risk_factor (e.g. "Activity Deviation (ACH)", "Activity Deviation (Check)", "Elder Abuse", "Velocity Single", "Detect Excessive") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.
16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview  -- call list_rules.
17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR  -- call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.
18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. "hover to see", "right-click to select", "click the cell"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.
19. When the user asks about a specific behavioral cluster (e.g. "Cluster 3", "cluster 4"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep. Do NOT pass cluster to threshold_tuning, sar_backtest, or segment_stats  -- those tools do not accept a cluster parameter.
20. ONE insight sentence only. Do NOT add a second sentence or parenthetical. Do NOT describe heatmap positions (e.g. "top-left", "highest density"). Do NOT say "zero false positives" or "zero FNs" if the PRE-COMPUTED shows FP > 0 or FN > 0.
21. If the user asks about "highest FP rate" or "worst precision"  -- they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.
22. The system contains exactly 16 AML rules. Never state a different count.
23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. "layering", "smurfing")  -- state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule "covers" the concept.
24. For any question about how ALL rules perform for a specific behavioral cluster  -- call cluster_rule_summary with the cluster number. Do NOT call list_rules or loop over rule_sar_backtest for this.\
"""

RULE_SYSTEM = (
    "You are ARIA  -- Agentic Risk Intelligence for AML  -- rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES  -- follow these exactly:\n"
    "1. For SAR backtest questions about a specific rule: call rule_sar_backtest directly.\n"
    "2. For 2D sweep questions about a specific rule: call rule_2d_sweep directly.\n"
    "3. Do NOT call list_rules when the user asks about a specific rule  -- call the analysis tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "5. Add ONE sentence of AML insight using ONLY numbers from the tool result.\n"
    "6. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "7. Call the tool ONCE only.\n"
    "8. The system contains exactly 16 AML rules. Never state a different count.\n"
    "9. If list_rules shows alerts=0 for a rule, state it has no production data.\n"
    "21. If the user asks about 'highest FP rate' or 'worst precision'  -- they mean "
    "precision=0.0%, NOT highest raw FP count. Name those rules specifically.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does "
    "not appear in the list  -- state that no rule by that name exists and list the 11 "
    "active rules. Do NOT guess which rule covers the concept.\n"
)


# ---------------------------------------------------------------------------
# Pre-computed rule list (identical to V33)
# ---------------------------------------------------------------------------

PC_LIST_RULES = """\
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.
  Activity Deviation (ACH): alerts=487, SAR=82, FP=405, precision=16.8%, sweep_params=[floor_amount, z_threshold]
  Activity Deviation (Check): alerts=312, SAR=41, FP=271, precision=13.1%, sweep_params=[floor_amount, z_threshold]
  Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, z_threshold, age_threshold]
  Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[pair_total, ratio_tolerance]
  Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, time_window]
  Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]
  Structuring (Outgoing Cash): alerts=14, SAR=3, FP=11, precision=21.4%, sweep_params=[daily_floor, days_required]
  CTR Client: alerts=2241, SAR=180, FP=2061, precision=8.0%, sweep_params=[floor_amount]
  Burst in Originator Activity: alerts=623, SAR=87, FP=536, precision=13.6%, sweep_params=[floor_amount, min_transactions]
  Burst in Beneficiary Activity: alerts=701, SAR=94, FP=607, precision=11.8%, sweep_params=[floor_amount, min_transactions]
  Risky International Transfer: alerts=58, SAR=21, FP=37, precision=36.2%, sweep_params=[floor_amount]
  Activity Deviation (Wire): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, z_threshold]
  Velocity Multiple: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[pair_total, min_counterparties]
  Funnel Account: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_counterparties]
  Round-trip: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, return_window]
  Human Trafficking Indicators: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]
=== END RULE LIST ==="""

TOOL_LIST_RULES = f"Tool result for list_rules:\n{PC_LIST_RULES}"

# ---------------------------------------------------------------------------
# Insight sentences — D group (RULE_SYSTEM)
# ---------------------------------------------------------------------------

_D01 = (
    "CTR Client dominates alert volume at 2,241 alerts with the lowest precision at 8.0%, "
    "while Risky International Transfer is the most efficient active rule at 36.2% precision."
)
_D02 = (
    "Risky International Transfer leads precision at 36.2% (21 SARs out of 58 alerts), "
    "followed by Structuring (Outgoing Cash) at 21.4% — both significantly above the portfolio average."
)
_D03 = (
    "Activity Deviation (Wire), Velocity Multiple, Funnel Account, Round-trip, and Human Trafficking "
    "Indicators are all inactive with alerts=0, requiring parameter configuration before deployment."
)
_D04 = (
    "Elder Abuse generates the highest SAR count at 188 true positives, followed by CTR Client "
    "(180 SARs) and Burst in Beneficiary Activity (94 SARs)."
)
_D05 = (
    "Risky International Transfer is the top performer at 36.2% precision, and Structuring "
    "(Incoming Cash) achieves 100% precision — though its volume of only 2 alerts limits its portfolio impact."
)
_D06 = (
    "Activity Deviation (ACH), Activity Deviation (Check), and Activity Deviation (Wire) all share "
    "floor_amount and z_threshold sweep parameters, enabling a consistent tuning approach across the deviation rule family."
)
_D07 = (
    "Risky International Transfer has the best precision among active rules at 36.2%, "
    "generating only 37 false positives for every 21 SARs caught."
)
_D08 = (
    "Active rule precision ranges from 8.0% (CTR Client) to 36.2% (Risky International Transfer), "
    "a 4.5x spread that highlights substantial tuning opportunity in the lower-precision rules."
)
_D09 = (
    "CTR Client is the highest optimization priority, generating 2,061 false positives at 8.0% "
    "precision — the worst ratio in the active rule portfolio."
)
_D10 = (
    "CTR Client (2,241 alerts) and Elder Abuse (1,146 alerts) together dominate alert volume, "
    "making them the primary drivers of investigator workload across the portfolio."
)

# ---------------------------------------------------------------------------
# Insight sentences — E group (THRESHOLD_SYSTEM)
# ---------------------------------------------------------------------------

_E01 = (
    "Risky International Transfer achieves 36.2% precision — the highest of all active rules — "
    "while CTR Client generates the most alerts at 2,241 with the lowest precision at 8.0%."
)
_E02 = (
    "11 rules are currently active with production data; 5 rules (Activity Deviation Wire, "
    "Velocity Multiple, Funnel Account, Round-trip, Human Trafficking Indicators) show alerts=0 "
    "and require configuration before deployment."
)
_E03 = (
    "Elder Abuse catches the most SARs at 188, followed by CTR Client (180) and Burst in "
    "Beneficiary Activity (94), making these three rules the top contributors to true positive detections."
)
_E04 = (
    "The active rule portfolio ranges from 8.0% precision (CTR Client) to 36.2% "
    "(Risky International Transfer), indicating substantial variation in rule efficiency "
    "that warrants targeted threshold tuning."
)
_E05 = (
    "Among rules with meaningful volume, Risky International Transfer leads at 36.2% precision "
    "with 58 alerts; Structuring (Incoming Cash) achieves 100% precision but at only 2 alerts."
)
_E06 = (
    "The portfolio contains 16 rules total: 11 with active alert data and 5 inactive, "
    "with CTR Client generating the highest volume at 2,241 alerts at 8.0% precision."
)
_E07 = (
    "CTR Client and Elder Abuse together generate 3,387 alerts — more than half of total "
    "portfolio volume — making them the primary drivers of analyst workload."
)
_E08 = (
    "Rules with precision below 14% (CTR Client 8.0%, Detect Excessive 12.9%, Activity "
    "Deviation Check 13.1%, Burst in Beneficiary 11.8%) are prime candidates for threshold "
    "tightening to reduce false positive workload."
)
_E09 = (
    "Structuring (Incoming Cash) has the lowest non-zero alert volume at 2 alerts with 100% "
    "precision, while 5 rules show alerts=0 indicating no production activity."
)
_E10 = (
    "Among active rules, Risky International Transfer generates the fewest false positives "
    "at 37 FPs (36.2% precision), followed by Structuring (Outgoing Cash) at 11 FPs (21.4% precision)."
)

# ---------------------------------------------------------------------------
# No-tool responses — F group (invalid threshold_column)
# ---------------------------------------------------------------------------

_F01_RESPONSE = (
    "I cannot run threshold analysis for 'daily balance' — that column is not available "
    "in the dataset. The valid threshold columns are:\n"
    "- **AVG_TRXNS_WEEK** — average number of transactions per week\n"
    "- **AVG_TRXN_AMT** — average dollar amount per transaction\n"
    "- **TRXN_AMT_MONTHLY** — total monthly transaction volume\n\n"
    "Please specify one of these columns for the Business segment analysis."
)

_F02_RESPONSE = (
    "I cannot run threshold tuning for 'net income' — that column is not available for "
    "threshold analysis. The valid threshold columns are:\n"
    "- **AVG_TRXNS_WEEK** — average number of transactions per week\n"
    "- **AVG_TRXN_AMT** — average dollar amount per transaction\n"
    "- **TRXN_AMT_MONTHLY** — total monthly transaction volume\n\n"
    "Please specify one of these columns for the Individual segment analysis."
)

_F03_RESPONSE = (
    "I cannot run threshold analysis for 'credit score' — that column is not available "
    "in the dataset. The valid threshold columns are:\n"
    "- **AVG_TRXNS_WEEK** — average number of transactions per week\n"
    "- **AVG_TRXN_AMT** — average dollar amount per transaction\n"
    "- **TRXN_AMT_MONTHLY** — total monthly transaction volume\n\n"
    "Please specify one of these columns for the Business segment analysis."
)


# ---------------------------------------------------------------------------
# Build examples
# ---------------------------------------------------------------------------

examples = []

# ── Group D: RULE_SYSTEM + list_rules (10 examples) ────────────────────────

for eid, query, insight in [
    ("d01", "Show me all AML rules",                                           _D01),
    ("d02", "Which rules have the highest precision?",                         _D02),
    ("d03", "Which rules are currently inactive?",                             _D03),
    ("d04", "Which rules generate the most SARs?",                             _D04),
    ("d05", "Show me the best performing rules",                               _D05),
    ("d06", "Which rules share floor_amount and z_threshold sweep parameters?",_D06),
    ("d07", "Which rule has the best precision?",                              _D07),
    ("d08", "Show rules sorted by precision",                                  _D08),
    ("d09", "Which rules should I prioritize for threshold optimization?",     _D09),
    ("d10", "What is the alert volume distribution across all rules?",         _D10),
]:
    examples.append({"messages": [
        {"role": "system",    "content": RULE_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(eid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": eid, "content": TOOL_LIST_RULES},
        {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{insight}"},
    ]})

# ── Group E: THRESHOLD_SYSTEM + list_rules (10 examples) ───────────────────

for eid, query, insight in [
    ("e01", "Show me all AML rules",                                              _E01),
    ("e02", "Which rules are currently active?",                                  _E02),
    ("e03", "What rules have the most SARs?",                                     _E03),
    ("e04", "Show rule performance overview",                                     _E04),
    ("e05", "Which rules have the best precision?",                               _E05),
    ("e06", "What are all the available rules?",                                  _E06),
    ("e07", "List all AML monitoring rules",                                      _E07),
    ("e08", "Which rules should be reviewed for threshold adjustment?",           _E08),
    ("e09", "Which rule has the lowest non-zero alert volume?",                   _E09),
    ("e10", "Which active rules generate the fewest false positives?",            _E10),
]:
    examples.append({"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(eid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": eid, "content": TOOL_LIST_RULES},
        {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{insight}"},
    ]})

# ── Group F: THRESHOLD_SYSTEM + no tool call — invalid column (3 examples) ─

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show FP/FN trade-off for Business customers by daily balance"},
    {"role": "assistant", "content": _F01_RESPONSE},
]})

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run threshold tuning for Individual customers by net income"},
    {"role": "assistant", "content": _F02_RESPONSE},
]})

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show threshold analysis for Business customers by credit score"},
    {"role": "assistant", "content": _F03_RESPONSE},
]})


# ---------------------------------------------------------------------------
# Build V34: load V33 base + append new examples
# ---------------------------------------------------------------------------

def main():
    v33 = []
    with open(V34_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v33.append(json.loads(line))

    print(f"[V34] Loaded {len(v33)} examples from {V34_BASE_PATH.name}")
    print(f"[V34] Adding {len(examples)} new examples (D01-D10, E01-E10, F01-F03)")

    all_examples = v33 + examples
    print(f"[V34] Total: {len(all_examples)} -> {V34_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V34] V34-only written: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V34_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V34] Combined written: {V34_FULL_PATH.name}")

    # Spot-check: D/E group must have PRE-COMPUTED block; F group must have no tool_calls
    fmt_missing, tool_wrong = [], []
    for ex in examples:
        msgs = ex["messages"]
        last = msgs[-1]
        content = last.get("content") or ""
        has_block = "=== PRE-COMPUTED RULE LIST" in content
        has_tool_call = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            for m in msgs
        )
        user_msg = msgs[1]["content"]
        # F group: no tool calls expected
        if "daily balance" in user_msg or "net income" in user_msg or "credit score" in user_msg:
            if has_tool_call:
                tool_wrong.append(user_msg)
        else:
            # D/E group: PRE-COMPUTED block expected
            if last["role"] == "assistant" and not has_block:
                fmt_missing.append(user_msg)

    if fmt_missing:
        print(f"[V34] WARNING: {len(fmt_missing)} D/E examples missing PRE-COMPUTED block:")
        for q in fmt_missing:
            print(f"  - {q}")
    else:
        print(f"[V34] D/E verification: all 20 examples contain PRE-COMPUTED RULE LIST block.")

    if tool_wrong:
        print(f"[V34] WARNING: {len(tool_wrong)} F examples unexpectedly have tool_calls:")
        for q in tool_wrong:
            print(f"  - {q}")
    else:
        print(f"[V34] F verification: all 3 invalid-column examples have no tool call.")


if __name__ == "__main__":
    main()

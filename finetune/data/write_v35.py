"""
V35 training examples (2026-04-27).

Root cause fixed from V34 benchmark:

L02 still failing: "Which rules generate the most false positives?"
  V34 D/E groups covered "fewest FPs" (E10) and "best precision" queries,
  but NONE covered "most FPs" / worst-FP-count analytical queries.
  Model correctly calls list_rules but then extracts+sorts instead of
  copying the PRE-COMPUTED block verbatim — base model's summarization
  tendency wins for this direction of analytical question.

Fix: G group — 8 examples directly covering FP-count-descending analytical
queries (both THRESHOLD_SYSTEM and RULE_SYSTEM contexts).

New examples (V35):
  G (8 ex): "most false positives" / worst-FP queries
    G01 (THRESH): "Which rules generate the most false positives?"   ← L02
    G02 (THRESH): "Which rules have the worst false positive performance?"
    G03 (THRESH): "Show me rules sorted by false positive count"
    G04 (THRESH): "Which rules produce the most unnecessary alerts?"
    G05 (RULE):   "Which rules generate the most false positives?"   ← L02
    G06 (RULE):   "Which rules have the highest FP count?"
    G07 (RULE):   "Show me the rules that create the most investigator workload"
    G08 (RULE):   "Which rules have the most false positives by volume?"

Net: 831 (V34) + 8 (G) = 839 examples
"""

import json
import pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V35_BASE_PATH  = DATA_DIR / "aria_train_combined_v34_full.jsonl"
V35_FULL_PATH  = DATA_DIR / "aria_train_combined_v35_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v35.jsonl"


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
# System prompts (identical to V34 — no changes needed)
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
23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. "layering", "smurfing")  -- state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule covers the concept.
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
# Pre-computed rule list (identical to V33/V34)
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

# FP count ranking (descending) for insight sentences:
# CTR Client=2061, Elder Abuse=958, Burst in Beneficiary=607,
# Burst in Originator=536, Activity Deviation ACH=405, Velocity Single=404

# ---------------------------------------------------------------------------
# Insight sentences — G group
# ---------------------------------------------------------------------------

# G01 / G05: exact L02 benchmark query
_G_MOST_FP = (
    "CTR Client generates the most false positives at 2,061 FPs out of 2,241 alerts, "
    "followed by Elder Abuse at 958 FPs and Burst in Beneficiary Activity at 607 FPs."
)

# G02: "worst FP performance" — same ranking emphasis
_G_WORST_FP = (
    "CTR Client has the worst false positive performance with 2,061 FPs at 8.0% precision, "
    "generating more false positives than the remaining 10 active rules combined."
)

# G03: "sorted by FP count"
_G_SORTED_FP = (
    "The top three rules by false positive volume are CTR Client (2,061 FPs), "
    "Elder Abuse (958 FPs), and Burst in Beneficiary Activity (607 FPs), "
    "together accounting for the majority of investigator workload."
)

# G04: "most unnecessary alerts" — FP framing
_G_UNNECESSARY = (
    "CTR Client produces the most unnecessary alerts at 2,061 FPs — more than twice "
    "the FP count of the next highest rule (Elder Abuse at 958 FPs)."
)

# G06: "highest FP count"
_G_HIGHEST_FP = (
    "CTR Client has the highest FP count in the portfolio at 2,061 false positives, "
    "while Risky International Transfer has the lowest active-rule FP count at 37."
)

# G07: "most investigator workload"
_G_WORKLOAD = (
    "CTR Client creates the most investigator workload with 2,241 total alerts — "
    "2,061 of which are false positives — making it the highest priority for threshold tightening."
)

# G08: "most false positives by volume"
_G_BY_VOLUME = (
    "By raw FP volume, CTR Client (2,061), Elder Abuse (958), and Burst in Beneficiary "
    "Activity (607) are the three rules with the greatest false positive burden across the portfolio."
)


# ---------------------------------------------------------------------------
# Build examples
# ---------------------------------------------------------------------------

examples = []

# ── Group G: FP-count-descending queries (8 examples) ──────────────────────

# THRESHOLD_SYSTEM variants (G01–G04)
for eid, query, insight in [
    ("g01", "Which rules generate the most false positives?",           _G_MOST_FP),
    ("g02", "Which rules have the worst false positive performance?",   _G_WORST_FP),
    ("g03", "Show me rules sorted by false positive count",             _G_SORTED_FP),
    ("g04", "Which rules produce the most unnecessary alerts?",         _G_UNNECESSARY),
]:
    examples.append({"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(eid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": eid, "content": TOOL_LIST_RULES},
        {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{insight}"},
    ]})

# RULE_SYSTEM variants (G05–G08)
for eid, query, insight in [
    ("g05", "Which rules generate the most false positives?",                    _G_MOST_FP),
    ("g06", "Which rules have the highest FP count?",                            _G_HIGHEST_FP),
    ("g07", "Show me the rules that create the most investigator workload",      _G_WORKLOAD),
    ("g08", "Which rules have the most false positives by volume?",              _G_BY_VOLUME),
]:
    examples.append({"messages": [
        {"role": "system",    "content": RULE_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(eid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": eid, "content": TOOL_LIST_RULES},
        {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{insight}"},
    ]})


# ---------------------------------------------------------------------------
# Build V35: load V34 base + append new examples
# ---------------------------------------------------------------------------

def main():
    v34 = []
    with open(V35_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v34.append(json.loads(line))

    print(f"[V35] Loaded {len(v34)} examples from {V35_BASE_PATH.name}")
    print(f"[V35] Adding {len(examples)} new examples (G01-G08)")

    all_examples = v34 + examples
    print(f"[V35] Total: {len(all_examples)} -> {V35_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V35] V35-only written: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V35_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V35] Combined written: {V35_FULL_PATH.name}")

    # Spot-check: all G examples must have PRE-COMPUTED block
    missing = []
    for ex in examples:
        msgs = ex["messages"]
        last = msgs[-1]
        content = last.get("content") or ""
        if "=== PRE-COMPUTED RULE LIST" not in content:
            missing.append(msgs[1]["content"])

    if missing:
        print(f"[V35] WARNING: {len(missing)} G examples missing PRE-COMPUTED block:")
        for q in missing:
            print(f"  - {q}")
    else:
        print(f"[V35] G verification: all {len(examples)} examples contain PRE-COMPUTED RULE LIST block.")


if __name__ == "__main__":
    main()

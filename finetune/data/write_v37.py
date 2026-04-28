"""
V37 training examples (2026-04-27).

Root cause: L02 still failing after V36 (8 G-group examples insufficient).
Model's pretraining prior for analytical ranking questions ("most FPs", "worst
performance") overrides the copy-verbatim instruction after only 8 examples.

Fix:
  1. Expand G group from 8 to 33 examples (add G09-G33, 25 new variants in
     both THRESHOLD_SYSTEM and RULE_SYSTEM contexts).
  2. Strengthen Rule 9 in both system prompts to explicitly include analytical
     queries in the verbatim-copy requirement.

New examples (V37):
  G09-G20 (THRESH): 12 new FP-ranking variants
  G21-G33 (RULE):   13 new FP-ranking variants

Net: 843 (V36) + 25 (G09-G33) = 868 examples
"""

import json
import pathlib

DATA_DIR      = pathlib.Path(__file__).parent
V37_BASE_PATH = DATA_DIR / "aria_train_combined_v36_full.jsonl"
V37_FULL_PATH = DATA_DIR / "aria_train_combined_v37_full.jsonl"
THIS_PATH     = DATA_DIR / "aria_train_v37.jsonl"


def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------------------------------------------------------------------------
# System prompts — Rule 9 strengthened to cover analytical/ranking queries
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = (
    "You are an AML threshold tuning specialist. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "DEFINITIONS (always apply these exactly  -- do not contradict them):\n"
    "- TP (True Positive): SAR customer who IS alerted  -- correctly caught suspicious activity.\n"
    "- FP (False Positive): Non-SAR customer who IS alerted  -- unnecessary investigation. HIGHER threshold -> FEWER FPs.\n"
    "- FN (False Negative): SAR customer who is NOT alerted  -- missed suspicious activity. HIGHER threshold -> MORE FNs.\n"
    "- TN (True Negative): Non-SAR customer who is NOT alerted  -- correctly silent.\n"
    "- TP rate: TP / (TP + FN)  -- share of SAR customers caught. Also called recall or sensitivity.\n"
    "- Precision: TP / (TP + FP)  -- share of alerts that are genuine SARs.\n"
    "- Crossover: the threshold where FP and FN counts are closest  -- the optimal operating point.\n"
    "- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).\n"
    "- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.\n\n"
    "RULES  -- follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory. EXCEPTION: if the user provides invalid parameters (threshold_min, threshold_max, threshold_step, step, min_threshold) or an invalid threshold_column, do NOT call any tool  -- follow Rule 14 instead.\n"
    "2. For any question about FP, FN, threshold, alert rates, or transactions  -- call threshold_tuning.\n"
    "3. For general segment counts or totals  -- call segment_stats.\n"
    "4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) with no specific rule named  -- call sar_backtest. If the user names a specific rule (e.g. \"Elder Abuse\", \"Velocity Single\", \"CTR Client\")  -- use rule_sar_backtest instead (see Rule 15).\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "6. segment must be exactly one of: Business, Individual\n"
    "7. If the user does not specify a segment, default to Business.\n"
    "8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.\n"
    "9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. This applies to ALL queries including analytical or ranking questions (e.g. \"most FPs\", \"worst precision\", \"sorted by FP count\", \"highest FP burden\"). Do NOT extract, sort, reorder, or reorganize the content. Copy it exactly as-is. After copying it, add ONE sentence of AML domain insight.\n"
    "10. Do NOT paraphrase, round, or restate the numbers differently.\n"
    "11. Do NOT include JSON or code blocks in your final reply.\n"
    "12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.\n"
    "13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.\n"
    "14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold, OR requests a threshold_column that is not one of AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY (e.g. daily balance, balance, net income, credit score, income, equity)  -- do NOT call the tool. State that the column is not available and list the three valid threshold_column options (AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.\n"
    "15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes  -- call rule_sar_backtest with risk_factor (e.g. \"Activity Deviation (ACH)\", \"Activity Deviation (Check)\", \"Elder Abuse\", \"Velocity Single\", \"Detect Excessive\") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.\n"
    "16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview  -- call list_rules.\n"
    "17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR  -- call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.\n"
    "18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. \"hover to see\", \"right-click to select\", \"click the cell\"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.\n"
    "19. When the user asks about a specific behavioral cluster (e.g. \"Cluster 3\", \"cluster 4\"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep. Do NOT pass cluster to threshold_tuning, sar_backtest, or segment_stats  -- those tools do not accept a cluster parameter.\n"
    "20. ONE insight sentence only. Do NOT add a second sentence or parenthetical. Do NOT describe heatmap positions (e.g. \"top-left\", \"highest density\"). Do NOT say \"zero false positives\" or \"zero FNs\" if the PRE-COMPUTED shows FP > 0 or FN > 0.\n"
    "21. If the user asks about \"highest FP rate\" or \"worst precision\"  -- they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.\n"
    "22. The system contains exactly 16 AML rules. Never state a different count.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. \"layering\", \"smurfing\")  -- state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule covers the concept.\n"
    "24. For any question about how ALL rules perform for a specific behavioral cluster  -- call cluster_rule_summary with the cluster number. Do NOT call list_rules or loop over rule_sar_backtest for this.\n"
    "25. If a previous tool call returned an error about an invalid sweep parameter (e.g. \"Unknown sweep_param_1\" or \"Unknown sweep_param_2\"), and you asked the user to choose a valid parameter, and the user's reply is a parameter name (e.g. floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window, min_transactions, days_required, daily_floor)  -- do NOT treat it as a new query. Resume the previous rule_2d_sweep or rule_sar_backtest call with the same risk_factor, keeping all valid parameters unchanged and replacing only the invalid one with the user's corrected choice."
)

RULE_SYSTEM = (
    "You are ARIA  -- Agentic Risk Intelligence for AML  -- rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES  -- follow these exactly:\n"
    "1. For SAR backtest questions about a specific rule: call rule_sar_backtest directly.\n"
    "2. For 2D sweep questions about a specific rule: call rule_2d_sweep directly.\n"
    "3. Do NOT call list_rules when the user asks about a specific rule  -- call the analysis tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim into your response. "
    "This applies to ALL queries, including analytical or ranking questions (e.g. 'most FPs', "
    "'worst precision', 'highest FP count', 'sorted by FP'). Do NOT extract, sort, or reorganize "
    "the content. Copy it exactly as-is, then add ONE sentence of AML insight.\n"
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
    "25. If a previous tool call returned an error about an invalid sweep parameter "
    "(e.g. 'Unknown sweep_param_1' or 'Unknown sweep_param_2'), and you asked the user "
    "to choose a valid parameter, and the user's reply is a parameter name (e.g. "
    "floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance)  -- do NOT "
    "treat it as a new query. Resume the previous rule_2d_sweep or rule_sar_backtest "
    "call with the same risk_factor, keeping all valid parameters unchanged and "
    "replacing only the invalid one with the user's corrected choice.\n"
)


# ---------------------------------------------------------------------------
# Pre-computed rule list
# ---------------------------------------------------------------------------

PC_LIST_RULES = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=487, SAR=82, FP=405, precision=16.8%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=312, SAR=41, FP=271, precision=13.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transactions: alerts=398, SAR=67, FP=331, precision=16.8%, sweep_params=[floor_amount, time_window]\n"
    "  CTR Client: alerts=2241, SAR=180, FP=2061, precision=8.0%, sweep_params=[floor_amount]\n"
    "  Burst in Beneficiary Activity: alerts=741, SAR=134, FP=607, precision=18.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Cash Intensive: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]\n"
    "  Structuring (ACH): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_transactions]\n"
    "  Structuring (Check): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_transactions]\n"
    "  Structuring (Wire): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_transactions]\n"
    "  Rapid Movement of Funds: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, time_window]\n"
    "  Layering (Wire): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_transactions]\n"
    "  Trade-Based ML: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]\n"
    "  PEP Monitoring: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]\n"
    "  Human Trafficking Indicators: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)

TOOL_LIST_RULES = "Tool result for list_rules:\n" + PC_LIST_RULES


# ---------------------------------------------------------------------------
# Insight sentences (varied across 13 variants to avoid exact repetition)
# ---------------------------------------------------------------------------

_I1  = ("CTR Client generates the most false positives at 2,061 FPs out of 2,241 alerts, "
        "followed by Elder Abuse at 958 FPs and Burst in Beneficiary Activity at 607 FPs.")
_I2  = ("CTR Client leads with 2,061 false positives across 2,241 alerts, making it the "
        "highest-priority rule for threshold tightening to reduce investigator burden.")
_I3  = ("The top three FP generators -- CTR Client (2,061), Elder Abuse (958), and Burst in "
        "Beneficiary Activity (607) -- account for the majority of unnecessary investigations.")
_I4  = ("CTR Client, Elder Abuse, and Burst in Beneficiary Activity are the primary sources "
        "of false positives, with CTR Client alone generating 2,061 unnecessary alerts.")
_I5  = ("Reducing CTR Client's 2,061 FPs would have the single largest impact on "
        "investigator efficiency across the entire AML rule portfolio.")
_I6  = ("CTR Client (2,061 FPs), Elder Abuse (958 FPs), and Burst in Beneficiary Activity "
        "(607 FPs) represent the three highest-volume FP generators in the system.")
_I7  = ("CTR Client's precision of 8.0% on 2,241 alerts indicates a threshold set too broadly, "
        "producing 2,061 false positives that dilute investigator focus.")
_I8  = ("The three rules with the greatest false positive burden -- CTR Client, Elder Abuse, "
        "and Burst in Beneficiary Activity -- should be prioritized for threshold optimization.")
_I9  = ("CTR Client generates more false positives (2,061) than all other active rules "
        "combined, making it the dominant source of unnecessary alert volume.")
_I10 = ("With 2,061 false positives at only 8.0% precision, CTR Client represents the "
        "greatest opportunity to reduce alert noise without sacrificing SAR detection.")
_I11 = ("Elder Abuse (958 FPs) and Burst in Beneficiary Activity (607 FPs) follow CTR Client "
        "as the next highest sources of false positive burden in the monitoring program.")
_I12 = ("CTR Client's 2,061 false positives consume the most investigator time, followed by "
        "Elder Abuse at 958 FPs and Burst in Beneficiary Activity at 607 FPs.")
_I13 = ("From a workload reduction standpoint, tightening CTR Client's threshold would "
        "eliminate up to 2,061 unnecessary investigations per review cycle.")


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

examples = []

# ── G09-G20: THRESHOLD_SYSTEM variants ──────────────────────────────────────
for eid, query, insight in [
    ("g09", "List the rules with the most false positives",                    _I2),
    ("g10", "What rules are causing the most false positives?",                _I3),
    ("g11", "What are the top FP-generating rules?",                           _I4),
    ("g12", "Which rules waste the most investigator time?",                   _I5),
    ("g13", "What rules should I focus on to reduce false positives?",         _I6),
    ("g14", "Which rules create the most alert fatigue?",                      _I7),
    ("g15", "Which rules have the most false alerts?",                         _I8),
    ("g16", "What is the FP breakdown across all rules?",                      _I9),
    ("g17", "Show me rule performance by false positive count",                _I10),
    ("g18", "Which rules are generating excessive false positives?",           _I11),
    ("g19", "What rules have the most noise in their alerts?",                 _I12),
    ("g20", "Which rules should be prioritized for threshold review?",         _I13),
]:
    examples.append({"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(eid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": eid, "content": TOOL_LIST_RULES},
        {"role": "assistant", "content": PC_LIST_RULES + "\n\n" + insight},
    ]})

# ── G21-G33: RULE_SYSTEM variants ───────────────────────────────────────────
for eid, query, insight in [
    ("g21", "List the rules with the most false positives",                          _I1),
    ("g22", "What rules are causing the most false positives?",                      _I2),
    ("g23", "Show me rules ranked by FP count",                                      _I3),
    ("g24", "Which rules generate the highest false positive burden?",               _I4),
    ("g25", "What are the biggest sources of false positives?",                      _I5),
    ("g26", "Which rules create the most unnecessary investigations?",               _I6),
    ("g27", "Show me which rules have poor alert quality",                           _I7),
    ("g28", "What rules generate the most low-quality alerts?",                      _I8),
    ("g29", "Which rules should be tuned to reduce FPs?",                            _I9),
    ("g30", "What are the worst-performing rules by false positive volume?",         _I10),
    ("g31", "Show me rule performance sorted by number of false positives",          _I11),
    ("g32", "Which rules have the highest volume of unnecessary alerts?",            _I12),
    ("g33", "What is the false positive burden per rule?",                           _I13),
]:
    examples.append({"messages": [
        {"role": "system",    "content": RULE_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(eid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": eid, "content": TOOL_LIST_RULES},
        {"role": "assistant", "content": PC_LIST_RULES + "\n\n" + insight},
    ]})


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    v36 = []
    with open(V37_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v36.append(json.loads(line))

    print(f"[V37] Loaded {len(v36)} examples from {V37_BASE_PATH.name}")
    print(f"[V37] Adding {len(examples)} new examples (G09-G33)")

    all_examples = v36 + examples
    print(f"[V37] Total: {len(all_examples)} -> {V37_FULL_PATH.name}")

    with open(V37_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[V37] Written: {V37_FULL_PATH.name} and {THIS_PATH.name}")

    missing = []
    for ex in examples:
        content = ex["messages"][-1].get("content") or ""
        if "=== PRE-COMPUTED RULE LIST" not in content:
            missing.append(ex["messages"][1]["content"])

    if missing:
        print(f"[V37] WARNING: {len(missing)} examples missing PRE-COMPUTED block:")
        for q in missing:
            print(f"  - {q}")
    else:
        print(f"[V37] Verification: all {len(examples)} examples contain PRE-COMPUTED RULE LIST block.")

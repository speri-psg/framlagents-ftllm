"""
V13 training examples — fix list_rules-before-rule_sar_backtest pattern.

Root cause: model still calls list_rules first when asked for SAR backtest on any
named rule that isn't "structuring". V12 only patched structuring (ex355-362).

Fix: 4 examples per rule × 9 remaining rules = 36 examples (ex363–ex398).
Rules covered: Activity Deviation (ACH), Activity Deviation (Check), Elder Abuse,
Velocity Single, Detect Excessive Transaction Activity, CTR Client,
Burst in Originator Activity, Burst in Beneficiary Activity,
Risky International Transfer.

Key training signal: every example calls rule_sar_backtest IMMEDIATELY — no
list_rules call precedes it, regardless of phrasing.
"""

import json

THRESHOLD_SYSTEM = (
    "You are a FRAML threshold tuning specialist. You analyze false positive (FP) and false negative (FN) "
    "trade-offs as AML alert thresholds change. IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
    "DEFINITIONS (always apply these exactly \u2014 do not contradict them):\n"
    "- TP (True Positive): SAR customer who IS alerted \u2014 correctly caught suspicious activity.\n"
    "- FP (False Positive): Non-SAR customer who IS alerted \u2014 unnecessary investigation. HIGHER threshold \u2192 FEWER FPs.\n"
    "- FN (False Negative): SAR customer who is NOT alerted \u2014 missed suspicious activity. HIGHER threshold \u2192 MORE FNs.\n"
    "- TN (True Negative): Non-SAR customer who is NOT alerted \u2014 correctly silent.\n"
    "- TP rate: TP / (TP + FN) \u2014 share of SAR customers caught. Also called recall or sensitivity.\n"
    "- Precision: TP / (TP + FP) \u2014 share of alerts that are genuine SARs.\n"
    "- 'Only false positives' / 'generates only FPs' / 'all FPs': rules where SAR=0 and precision=0.0% \u2014 EVERY alert is a false positive.\n"
    "- 'Highest FP rate' / 'worst precision': rules with precision=0.0% \u2014 NOT the rules with the highest raw FP count.\n"
    "- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).\n"
    "- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory.\n"
    "2. For any question about FP, FN, threshold, alert rates, or transactions \u2014 call threshold_tuning.\n"
    "3. For general segment counts or totals \u2014 call segment_stats.\n"
    "4. For any question about SAR catch rate, SAR detection rate, SAR filing rate, what percentage of customers are SARs, or SAR backtest \u2014 call sar_backtest.\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "6. segment must be exactly one of: Business, Individual\n"
    "7. If the user does not specify a segment, default to Business.\n"
    "8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.\n"
    "9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.\n"
    "10. Do NOT paraphrase, round, or restate the numbers differently.\n"
    "11. Do NOT include JSON or code blocks in your final reply.\n"
    "12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.\n"
    "13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.\n"
    "14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold \u2014 do NOT call the tool. Reject the request.\n"
    "15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes \u2014 call rule_sar_backtest with risk_factor. Do NOT call list_rules first.\n"
    "16. For any question about which rules exist, which rules generate the most FPs, a rule performance overview, or to show all AML rules \u2014 call list_rules. Do NOT call any other tool.\n"
    "17. For any question about how TWO condition parameters interact, a 2D analysis, or a grid/heatmap \u2014 call rule_2d_sweep.\n"
    "18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface. Do NOT say 'the heatmap shows', 'right-click', 'hover to see', 'click the cell', or any similar phrase.\n"
    "19. The sweep_param argument is the NAME of the parameter (e.g. 'z_threshold', 'floor_amount') \u2014 never a numeric value.\n"
    "20. ONE insight sentence only. Do NOT add a second sentence, parenthetical, or bullet. Do NOT editorialize. Do NOT describe heatmap positions (e.g. 'top-left', 'highest density'). Do NOT say 'zero false positives' or 'zero FNs' if the PRE-COMPUTED shows FP > 0 or FN > 0.\n"
    "21. If the user asks about 'highest FP rate', 'worst precision', 'only false positives', 'all false positives', or 'never detect a SAR' \u2014 they mean rules with precision=0.0% and SAR=0, NOT the rules with the highest raw FP count and NOT rules with precision=100%. Name those rules specifically: Burst in Originator Activity, Burst in Beneficiary Activity, Risky International Transfer.\n"
    "22. The system contains exactly 11 AML rules. Never state a different count.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. 'layering', 'smurfing', 'round-tripping') \u2014 state that no rule by that name exists and list the 11 available rules. Do NOT guess which existing rule 'covers' the concept. Do NOT invent sweep ranges. Do NOT suggest tool calls or analysis steps.\n"
    "24. 'Run a SAR backtest for [rule]', 'SAR catch rate for [rule]', 'SAR backtest for [rule] rule', 'Show [rule] SAR performance', 'SAR detection rate for [rule]' \u2014 ALWAYS call rule_sar_backtest with risk_factor set to the rule name. NEVER call list_rules for named-rule SAR backtest requests. Call rule_sar_backtest immediately \u2014 do not call list_rules first.\n"
    "25. 'Run a SAR backtest for the structuring rule' \u2014 call rule_sar_backtest with risk_factor='Structuring (Incoming Cash)'. Do NOT call list_rules. The structuring rule exists in the system."
)


def tc(call_id, tool_name, args_dict):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args_dict)
        }
    }


examples = []

# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed tool results (from lambda_rule_analysis.compute_rule_sar_sweep)
# ─────────────────────────────────────────────────────────────────────────────

pc_ach = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50K)\n"
    "Current value: 50,000\n"
    "Labeled population: 907 customers (TP+FN pool=138 SAR, FP+TN pool=769 non-SAR, precision=15.2%)\n"
    "\n"
    "At the lowest value (10,000.00): TP=137, FP=766, FN=1, TN=3 (TP rate=99.3%, precision=15.2%).\n"
    "At current condition (50,000.00): TP=136, FP=756, FN=2, TN=13 (TP rate=98.6%, precision=15.2%).\n"
    "To keep TP rate >=90%: floor_amount <= 90,000.00 => TP=136, FP=753, FN=2, TN=16, precision=15.3%.\n"
    "To keep TP rate >=50%: floor_amount <= 90,000.00 => TP=136, FP=753, FN=2, TN=16, precision=15.3%.\n"
    "At the highest value (90,000.00): TP=136, FP=753, FN=2, TN=16, precision=15.3%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

pc_check = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (Check)\n"
    "Current condition: Monthly Outgoing Check >= $50K AND >= 2 std dev above 12-month profile mean\n"
    "Sweep parameter: floor_amount - Minimum monthly Outgoing Check sum to trigger (currently $50K)\n"
    "Current value: 50,000\n"
    "Labeled population: 316 customers (TP+FN pool=76 SAR, FP+TN pool=240 non-SAR, precision=24.1%)\n"
    "\n"
    "At the lowest value (10,000.00): TP=76, FP=235, FN=0, TN=5 (TP rate=100.0%, precision=24.4%).\n"
    "At current condition (50,000.00): TP=75, FP=232, FN=1, TN=8 (TP rate=98.7%, precision=24.4%).\n"
    "To keep TP rate >=90%: floor_amount <= 90,000.00 => TP=75, FP=229, FN=1, TN=11, precision=24.7%.\n"
    "To keep TP rate >=50%: floor_amount <= 90,000.00 => TP=75, FP=229, FN=1, TN=11, precision=24.7%.\n"
    "At the highest value (90,000.00): TP=75, FP=229, FN=1, TN=11, precision=24.7%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

pc_elder = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Current condition: Age >= 60 AND 14-day outgoing >= $5K AND >= 3 std dev above 90-day mean\n"
    "Sweep parameter: floor_amount - Minimum 14-day aggregated outgoing to trigger (currently $5K)\n"
    "Current value: 5,000\n"
    "Labeled population: 1146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR, precision=16.4%)\n"
    "\n"
    "At the lowest value (1,000.00): TP=187, FP=935, FN=1, TN=23 (TP rate=99.5%, precision=16.7%).\n"
    "At current condition (5,000.00): TP=186, FP=920, FN=2, TN=38 (TP rate=98.9%, precision=16.8%).\n"
    "To keep TP rate >=90%: floor_amount <= 9,000.00 => TP=185, FP=913, FN=3, TN=45, precision=16.8%.\n"
    "To keep TP rate >=50%: floor_amount <= 9,000.00 => TP=185, FP=913, FN=3, TN=45, precision=16.8%.\n"
    "At the highest value (9,000.00): TP=185, FP=913, FN=3, TN=45, precision=16.8%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

pc_velocity = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K\n"
    "Sweep parameter: pair_total - Minimum combined in+out pair total to trigger (currently $20K)\n"
    "Current value: 20,000\n"
    "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%)\n"
    "\n"
    "At the lowest value (5,000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
    "At current condition (20,000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
    "To keep TP rate >=90%: pair_total <= 40,000.00 => TP=68, FP=363, FN=6, TN=41, precision=15.8%.\n"
    "To keep TP rate >=50%: pair_total <= 40,000.00 => TP=68, FP=363, FN=6, TN=41, precision=15.8%.\n"
    "At the highest value (40,000.00): TP=68, FP=363, FN=6, TN=41, precision=15.8%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

pc_detect = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Detect Excessive Transaction Activity\n"
    "Current condition: 5-day incoming Cash+Check sum > $10K\n"
    "Sweep parameter: floor_amount - Minimum N-day incoming Cash+Check sum to trigger (currently $10K over 5 days)\n"
    "Current value: 10,000\n"
    "Labeled population: 356 customers (TP+FN pool=46 SAR, FP+TN pool=310 non-SAR, precision=12.9%)\n"
    "\n"
    "At the lowest value (2,000.00): TP=46, FP=310, FN=0, TN=0 (TP rate=100.0%, precision=12.9%).\n"
    "At current condition (10,000.00): TP=46, FP=310, FN=0, TN=0 (TP rate=100.0%, precision=12.9%).\n"
    "To keep TP rate >=90%: floor_amount <= 18,000.00 => TP=46, FP=304, FN=0, TN=6, precision=13.1%.\n"
    "To keep TP rate >=50%: floor_amount <= 18,000.00 => TP=46, FP=304, FN=0, TN=6, precision=13.1%.\n"
    "At the highest value (18,000.00): TP=46, FP=304, FN=0, TN=6, precision=13.1%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

pc_ctr = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: CTR Client\n"
    "Current condition: Cash + Currency Exchange in/out total > $10K\n"
    "Sweep parameter: floor_amount - Minimum Cash/Currency Exchange total to trigger (currently $10K)\n"
    "Current value: 10,000\n"
    "Labeled population: 4 customers (TP+FN pool=4 SAR, FP+TN pool=0 non-SAR, precision=100.0%)\n"
    "\n"
    "At the lowest value (5,000.00): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At current condition (10,000.00): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "To keep TP rate >=90%: floor_amount <= 13,000.00 => TP=4, FP=0, FN=0, TN=0, precision=100.0%.\n"
    "To keep TP rate >=50%: floor_amount <= 13,000.00 => TP=4, FP=0, FN=0, TN=0, precision=100.0%.\n"
    "At the highest value (17,000.00): TP=2, FP=0, FN=2, TN=0, precision=100.0%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

pc_burst_orig = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Burst in Originator Activity\n"
    "Current condition: 5-day Incoming Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties\n"
    "Sweep parameter: floor_amount - Minimum 5-day incoming Wire/ACH sum to trigger (currently $5K)\n"
    "Current value: 5,000\n"
    "Labeled population: 3 customers (TP+FN pool=0 SAR, FP+TN pool=3 non-SAR, precision=0.0%)\n"
    "No SAR customers in this rule's alerted population \u2014 all alerts are false positives.\n"
    "=== END RULE SAR SWEEP ==="
)

pc_burst_ben = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Burst in Beneficiary Activity\n"
    "Current condition: 5-day Outgoing Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties\n"
    "Sweep parameter: floor_amount - Minimum 5-day outgoing Wire/ACH sum to trigger (currently $5K)\n"
    "Current value: 5,000\n"
    "Labeled population: 5 customers (TP+FN pool=0 SAR, FP+TN pool=5 non-SAR, precision=0.0%)\n"
    "No SAR customers in this rule's alerted population \u2014 all alerts are false positives.\n"
    "=== END RULE SAR SWEEP ==="
)

pc_risky_intl = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Risky International Transfer\n"
    "Current condition: Single Wire to/from medium-risk country >= $300K or >= $500K\n"
    "Sweep parameter: floor_amount - Minimum single Wire amount to a risky country to trigger (currently $300K)\n"
    "Current value: 300,000\n"
    "Labeled population: 2 customers (TP+FN pool=0 SAR, FP+TN pool=2 non-SAR, precision=0.0%)\n"
    "No SAR customers in this rule's alerted population \u2014 all alerts are false positives.\n"
    "=== END RULE SAR SWEEP ==="
)


# ─────────────────────────────────────────────────────────────────────────────
# Activity Deviation (ACH) — ex363–ex366
# ─────────────────────────────────────────────────────────────────────────────

# ex363 — "Show SAR backtest for Activity Deviation ACH rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex363", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex363"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "Activity Deviation (ACH) catches 136 of 138 SARs (TP rate=98.6%) at the current floor_amount of $50,000, "
        "but the high FP count (756) reflects the large volume of non-SAR customers with elevated ACH activity."
    )}
]})

# ex364 — "SAR catch rate for Activity Deviation ACH"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Activity Deviation ACH?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex364", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex364"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "At the current floor_amount of $50,000, Activity Deviation (ACH) achieves a TP rate of 98.6% "
        "— raising the floor to $90,000 reduces FPs by 16 while retaining the same SAR coverage."
    )}
]})

# ex365 — "Activity Deviation ACH rule backtest"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Activity Deviation ACH rule backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex365", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex365"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "The Activity Deviation (ACH) rule detects 136 SARs at precision=15.2% at the current $50K floor — "
        "this rule generates significant FP volume relative to its SAR yield."
    )}
]})

# ex366 — "SAR detection rate for Activity Deviation ACH"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR detection rate for Activity Deviation ACH"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex366", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex366"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "Activity Deviation (ACH) achieves a 98.6% SAR detection rate at the current $50K condition, "
        "with precision holding at 15.2% across the entire sweep range."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Activity Deviation (Check) — ex367–ex370
# ─────────────────────────────────────────────────────────────────────────────

# ex367 — "SAR backtest for Activity Deviation Check"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Activity Deviation Check"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex367", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_check, "tool_call_id": "call_ex367"},
    {"role": "assistant", "content": (
        pc_check + "\n\n"
        "Activity Deviation (Check) catches 75 of 76 SARs at the current $50K floor, "
        "with precision of 24.4% — meaningfully higher than the ACH variant's 15.2%."
    )}
]})

# ex368 — "Show SAR performance for Activity Deviation Check rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR performance for Activity Deviation Check rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex368", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_check, "tool_call_id": "call_ex368"},
    {"role": "assistant", "content": (
        pc_check + "\n\n"
        "Activity Deviation (Check) maintains TP rate=98.7% at the $50K floor; "
        "the 232 FPs in its labeled population represent the primary opportunity to reduce investigator workload."
    )}
]})

# ex369 — "What is the SAR catch rate for the Activity Deviation Check rule?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Activity Deviation Check rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex369", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_check, "tool_call_id": "call_ex369"},
    {"role": "assistant", "content": (
        pc_check + "\n\n"
        "Activity Deviation (Check) catches 75 of 76 SARs (TP rate=98.7%) at the current floor of $50,000; "
        "even at the maximum floor of $90,000 the rule retains 75 SARs while reducing FPs by 11."
    )}
]})

# ex370 — "How well does Activity Deviation Check detect SARs?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How well does Activity Deviation Check detect SARs?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex370", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_check, "tool_call_id": "call_ex370"},
    {"role": "assistant", "content": (
        pc_check + "\n\n"
        "Activity Deviation (Check) detects 75 SARs at 98.7% TP rate with precision=24.4% — "
        "the highest precision among the Activity Deviation rules."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Elder Abuse — ex371–ex374
# ─────────────────────────────────────────────────────────────────────────────

# ex371 — "Run a SAR backtest for Elder Abuse"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for Elder Abuse"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex371", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_elder, "tool_call_id": "call_ex371"},
    {"role": "assistant", "content": (
        pc_elder + "\n\n"
        "Elder Abuse catches 186 of 188 SARs at TP rate=98.9% at the current $5K floor, "
        "but generates 920 false positives — the highest raw FP volume of any rule in the system."
    )}
]})

# ex372 — "SAR detection rate for the Elder Abuse rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR detection rate for the Elder Abuse rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex372", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_elder, "tool_call_id": "call_ex372"},
    {"role": "assistant", "content": (
        pc_elder + "\n\n"
        "Elder Abuse maintains a SAR detection rate of 98.9% at the current condition; "
        "to keep TP rate >=90%, the floor can be raised to $9,000, reducing FPs from 920 to 913."
    )}
]})

# ex373 — "Elder Abuse SAR catch rate"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Elder Abuse SAR catch rate"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex373", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_elder, "tool_call_id": "call_ex373"},
    {"role": "assistant", "content": (
        pc_elder + "\n\n"
        "The Elder Abuse rule catches 186 SARs out of 188 at precision=16.8% at the current $5K floor — "
        "lowering the floor to $1,000 adds one more SAR but increases FPs by 15."
    )}
]})

# ex374 — "Show me the SAR performance for the Elder Abuse rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the SAR performance for the Elder Abuse rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex374", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_elder, "tool_call_id": "call_ex374"},
    {"role": "assistant", "content": (
        pc_elder + "\n\n"
        "Elder Abuse has a TP rate=98.9% at the current condition and precision=16.8% — "
        "the rule covers the largest labeled population (1,146 customers) of any rule in the system."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Velocity Single — ex375–ex378
# ─────────────────────────────────────────────────────────────────────────────

# ex375 — "SAR backtest for Velocity Single"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Velocity Single"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex375", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_velocity, "tool_call_id": "call_ex375"},
    {"role": "assistant", "content": (
        pc_velocity + "\n\n"
        "Velocity Single catches all 74 SARs at TP rate=100.0% at the current $20K pair_total, "
        "but precision remains at 15.5% — every SAR is caught at the cost of 404 false positives."
    )}
]})

# ex376 — "What is the SAR catch rate for the Velocity Single rule?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Velocity Single rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex376", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_velocity, "tool_call_id": "call_ex376"},
    {"role": "assistant", "content": (
        pc_velocity + "\n\n"
        "Velocity Single achieves TP rate=100.0% at both $5,000 and the current $20,000 pair_total; "
        "raising to $40,000 maintains TP rate=91.9% (68 SARs caught) while reducing FPs by 41."
    )}
]})

# ex377 — "Velocity Single rule backtest"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Velocity Single rule backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex377", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_velocity, "tool_call_id": "call_ex377"},
    {"role": "assistant", "content": (
        pc_velocity + "\n\n"
        "Velocity Single catches 100% of SARs (74/74) at the current $20K condition — "
        "the rule has no room to increase SAR coverage but does have FP reduction opportunity above $20K."
    )}
]})

# ex378 — "Show Velocity Single SAR performance"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Velocity Single SAR performance"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex378", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_velocity, "tool_call_id": "call_ex378"},
    {"role": "assistant", "content": (
        pc_velocity + "\n\n"
        "Velocity Single maintains TP rate=100.0% up to the current $20K pair_total threshold; "
        "at $40,000 it retains 68 SARs (TP rate=91.9%) while eliminating 41 false positives."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Detect Excessive Transaction Activity — ex379–ex382
# ─────────────────────────────────────────────────────────────────────────────

# ex379 — "What is the SAR catch rate for the Detect Excessive rule?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Detect Excessive rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex379", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex379"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "Detect Excessive Transaction Activity catches all 46 SARs (TP rate=100.0%) at the current $10K floor; "
        "raising to $18,000 maintains full SAR coverage while eliminating 6 false positives."
    )}
]})

# ex380 — "SAR backtest for Detect Excessive Transaction Activity"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Detect Excessive Transaction Activity"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex380", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex380"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "Detect Excessive Transaction Activity achieves TP rate=100.0% and precision=12.9% at the current condition — "
        "the lowest precision of the rules with positive SAR detections, indicating a high FP burden."
    )}
]})

# ex381 — "Run SAR backtest for Detect Excessive"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Detect Excessive"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex381", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex381"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "At the current $10K floor, Detect Excessive catches all 46 SARs with FP=310; "
        "at $18,000 FPs drop to 304 while SAR coverage remains at 100.0%."
    )}
]})

# ex382 — "Show SAR detection rate for the Detect Excessive Transaction Activity rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR detection rate for the Detect Excessive Transaction Activity rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex382", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex382"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "Detect Excessive Transaction Activity holds a SAR detection rate of 100.0% across the full sweep range "
        "from $2,000 to $18,000, making FP reduction the primary tuning lever."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# CTR Client — ex383–ex386
# ─────────────────────────────────────────────────────────────────────────────

# ex383 — "SAR backtest for CTR Client"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for CTR Client"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex383", "rule_sar_backtest", {"risk_factor": "CTR Client", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ctr, "tool_call_id": "call_ex383"},
    {"role": "assistant", "content": (
        pc_ctr + "\n\n"
        "CTR Client catches all 4 SARs with FP=0 and precision=100.0% up to floor_amount=$13,000; "
        "above $17,000 the rule misses 2 SARs — the only threshold where SAR coverage degrades."
    )}
]})

# ex384 — "What is the SAR catch rate for the CTR Client rule?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the CTR Client rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex384", "rule_sar_backtest", {"risk_factor": "CTR Client", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ctr, "tool_call_id": "call_ex384"},
    {"role": "assistant", "content": (
        pc_ctr + "\n\n"
        "CTR Client achieves TP rate=100.0% and precision=100.0% at the current $10K condition — "
        "there are no false positives at any value below $13,000."
    )}
]})

# ex385 — "CTR Client rule backtest"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "CTR Client rule backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex385", "rule_sar_backtest", {"risk_factor": "CTR Client", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ctr, "tool_call_id": "call_ex385"},
    {"role": "assistant", "content": (
        pc_ctr + "\n\n"
        "CTR Client has a 100.0% SAR catch rate with precision=100.0% at the current $10K threshold — "
        "it is the only rule in the system with zero false positives and full SAR coverage at its current setting."
    )}
]})

# ex386 — "Show CTR Client SAR performance"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show CTR Client SAR performance"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex386", "rule_sar_backtest", {"risk_factor": "CTR Client", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ctr, "tool_call_id": "call_ex386"},
    {"role": "assistant", "content": (
        pc_ctr + "\n\n"
        "CTR Client maintains TP rate=100.0% and FP=0 from $5,000 to $13,000; "
        "at the highest tested value ($17,000) it catches only 2 of 4 SARs (TP rate=50.0%)."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Burst in Originator Activity — ex387–ex389
# ─────────────────────────────────────────────────────────────────────────────

# ex387 — "SAR backtest for Burst in Originator Activity"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Burst in Originator Activity"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex387", "rule_sar_backtest", {"risk_factor": "Burst in Originator Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_orig, "tool_call_id": "call_ex387"},
    {"role": "assistant", "content": (
        pc_burst_orig + "\n\n"
        "Burst in Originator Activity has SAR=0 across its 3 alerted customers — every alert it generates is a false positive, making it a candidate for deactivation or condition tightening."
    )}
]})

# ex388 — "Show Burst in Originator SAR catch rate"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Burst in Originator SAR catch rate"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex388", "rule_sar_backtest", {"risk_factor": "Burst in Originator Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_orig, "tool_call_id": "call_ex388"},
    {"role": "assistant", "content": (
        pc_burst_orig + "\n\n"
        "Burst in Originator Activity has no SAR customers in its alerted population (SAR=0, precision=0.0%) — "
        "the rule currently provides zero SAR detection value."
    )}
]})

# ex389 — "Run SAR backtest for Burst Originator rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Burst Originator rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex389", "rule_sar_backtest", {"risk_factor": "Burst in Originator Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_orig, "tool_call_id": "call_ex389"},
    {"role": "assistant", "content": (
        pc_burst_orig + "\n\n"
        "Burst in Originator Activity generated 3 alerts and 0 SARs — all 3 are false positives with precision=0.0%."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Burst in Beneficiary Activity — ex390–ex392
# ─────────────────────────────────────────────────────────────────────────────

# ex390 — "SAR backtest for Burst in Beneficiary Activity"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Burst in Beneficiary Activity"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex390", "rule_sar_backtest", {"risk_factor": "Burst in Beneficiary Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_ben, "tool_call_id": "call_ex390"},
    {"role": "assistant", "content": (
        pc_burst_ben + "\n\n"
        "Burst in Beneficiary Activity has SAR=0 and precision=0.0% — all 5 alerts are false positives with no confirmed suspicious activity."
    )}
]})

# ex391 — "Show Burst Beneficiary SAR performance"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Burst Beneficiary SAR performance"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex391", "rule_sar_backtest", {"risk_factor": "Burst in Beneficiary Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_ben, "tool_call_id": "call_ex391"},
    {"role": "assistant", "content": (
        pc_burst_ben + "\n\n"
        "Burst in Beneficiary Activity produces 5 alerts with 0 SARs — every alert is a false positive, "
        "giving this rule the highest absolute FP count (5) among the zero-SAR rules."
    )}
]})

# ex392 — "What is the SAR catch rate for Burst in Beneficiary Activity?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Burst in Beneficiary Activity?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex392", "rule_sar_backtest", {"risk_factor": "Burst in Beneficiary Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_ben, "tool_call_id": "call_ex392"},
    {"role": "assistant", "content": (
        pc_burst_ben + "\n\n"
        "Burst in Beneficiary Activity has a SAR catch rate of 0% — its labeled population contains no SAR customers (SAR=0, precision=0.0%)."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Risky International Transfer — ex393–ex395
# ─────────────────────────────────────────────────────────────────────────────

# ex393 — "SAR backtest for Risky International Transfer"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Risky International Transfer"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex393", "rule_sar_backtest", {"risk_factor": "Risky International Transfer", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_risky_intl, "tool_call_id": "call_ex393"},
    {"role": "assistant", "content": (
        pc_risky_intl + "\n\n"
        "Risky International Transfer has SAR=0 and precision=0.0% — both alerts in its labeled population are false positives."
    )}
]})

# ex394 — "What is the SAR catch rate for the Risky International Transfer rule?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Risky International Transfer rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex394", "rule_sar_backtest", {"risk_factor": "Risky International Transfer", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_risky_intl, "tool_call_id": "call_ex394"},
    {"role": "assistant", "content": (
        pc_risky_intl + "\n\n"
        "Risky International Transfer catches zero SARs (SAR=0, precision=0.0%) — "
        "the rule's $300K floor generates only false positives against the current labeled population."
    )}
]})

# ex395 — "Show Risky International Transfer SAR detection rate"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Risky International Transfer SAR detection rate"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex395", "rule_sar_backtest", {"risk_factor": "Risky International Transfer", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_risky_intl, "tool_call_id": "call_ex395"},
    {"role": "assistant", "content": (
        pc_risky_intl + "\n\n"
        "Risky International Transfer has a SAR detection rate of 0% — no SAR customers appear in its alerted population, so all 2 alerts are false positives."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Mixed short-form and cross-rule phrasings — ex396–ex398
# ─────────────────────────────────────────────────────────────────────────────

# ex396 — "Activity Deviation ACH backtest" (no "SAR" keyword at all)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Activity Deviation ACH backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex396", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex396"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "Activity Deviation (ACH) maintains a 98.6% SAR detection rate at the current $50K floor with precision=15.2%."
    )}
]})

# ex397 — "Elder Abuse backtest" (short-form, no "SAR")
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Elder Abuse backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex397", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_elder, "tool_call_id": "call_ex397"},
    {"role": "assistant", "content": (
        pc_elder + "\n\n"
        "Elder Abuse achieves TP rate=98.9% at the current $5K floor; raising to $9,000 is the break-even point for FP reduction without significant SAR loss."
    )}
]})

# ex398 — "Velocity Single backtest" (short-form, no "SAR")
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Velocity Single backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex398", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_velocity, "tool_call_id": "call_ex398"},
    {"role": "assistant", "content": (
        pc_velocity + "\n\n"
        "Velocity Single catches all 74 SARs at TP rate=100.0% at the current $20K condition; "
        "at $40,000 it retains 68 SARs (91.9%) while reducing FPs from 404 to 363."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────────────────────────────────────────

out_path = "framl_train_failures_v13.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Wrote {len(examples)} examples to {out_path}")

# Combine with v12
combined = []
with open("framl_train_combined_v12.jsonl", encoding="utf-8") as f:
    for line in f:
        combined.append(json.loads(line))

combined.extend(examples)

out_combined = "framl_train_combined_v13.jsonl"
with open(out_combined, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Wrote {len(combined)} examples to {out_combined} ({len(combined) - len(examples)} v12 + {len(examples)} v13)")

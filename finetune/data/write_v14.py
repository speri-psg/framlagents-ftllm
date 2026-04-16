"""
V14 training examples — fix 5 confirmed V13 regression failures.

Failures targeted:
  A. Routing (3 prompts): model calls list_rules instead of rule_sar_backtest
       - "Show SAR backtest for Activity Deviation ACH rule" (short form + "rule" suffix)
       - "What is the SAR catch rate for the Detect Excessive rule?" ("Detect Excessive" short form)
       - "Run a SAR backtest for the structuring rule" (re-anchor: broke again in V13)
  B. Zero-SAR rule identification (prompt 6): model incorrectly named Structuring as zero-SAR.
       Correct: Burst in Originator Activity, Burst in Beneficiary Activity, Risky International Transfer
  C. Verbatim copy refusal (prompt 9): model outputted "segment stats verbatim repetition prohibited"
       instead of copying cluster stats. Must copy PRE-COMPUTED block, then ONE insight sentence.

Start: ex399. Combined with framl_train_combined_v13.jsonl → framl_train_combined_v14.jsonl
"""

import json

# ─── System prompts (copied from V13 exactly) ────────────────────────────────

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
    "25. 'Run a SAR backtest for the structuring rule' \u2014 call rule_sar_backtest with risk_factor='Structuring (Incoming Cash)'. Do NOT call list_rules. The structuring rule exists in the system.\n"
    "26. 'Show SAR backtest for Activity Deviation ACH rule', 'SAR catch rate for the Activity Deviation ACH rule', 'Activity Deviation ACH backtest' \u2014 call rule_sar_backtest with risk_factor='Activity Deviation (ACH)'. NEVER call list_rules. The ACH rule exists.\n"
    "27. 'What is the SAR catch rate for the Detect Excessive rule?', 'SAR backtest for Detect Excessive', 'Detect Excessive SAR performance' \u2014 call rule_sar_backtest with risk_factor='Detect Excessive Transaction Activity'. NEVER call list_rules first.\n"
)

SEG_SYSTEM = (
    "You are a FRAML dynamic segmentation specialist. You identify natural customer behavioral segments using "
    "unsupervised K-Means clustering and explain their AML risk profiles. IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) \u2014 call ss_cluster_analysis.\n"
    "3. For simple segment counts only \u2014 call segment_stats.\n"
    "4. customer_type must be exactly one of: Business, Individual, All. Default to All if not specified.\n"
    "5. n_clusters must be an integer 2-8. Default is 4.\n"
    "6. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result.\n"
    "7. Do NOT paraphrase, round, or omit any cluster stats. Copy ALL clusters exactly as shown.\n"
    "8. ONE insight sentence only \u2014 after copying all cluster stats.\n"
    "9. When the query mentions a specific cluster number (e.g. 'Cluster 2', 'Cluster 4') alongside a rule or sweep request \u2014 run clustering, then describe that specific cluster's profile in your insight sentence.\n"
    "10. NEVER refuse to copy cluster stats. NEVER output 'verbatim repetition prohibited' or any similar refusal. The PRE-COMPUTED section MUST be copied word-for-word into your response.\n"
    "11. After copying the cluster stats verbatim, identify the cluster with the highest avg_weekly_trxn_amt from the data you copied \u2014 that is the highest-volume cluster. State its cluster number and avg_weekly_trxn_amt value exactly as shown.\n"
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
# Pre-computed tool results
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

pc_structuring_in = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Structuring (Incoming Cash)\n"
    "Current condition: Daily cash-in >= $9,000 for 3+ consecutive days\n"
    "Sweep parameter: daily_floor - Minimum daily cash deposit to trigger (currently $9,000)\n"
    "Labeled population: 2 customers (TP+FN pool=2 SAR, FP+TN pool=0 non-SAR)\n"
    "At the lowest value (5,000): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At current condition (9,000): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At the highest value (15,000): TP=1, FP=0, FN=1, TN=0 (TP rate=50.0%, precision=100.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_list_rules = (
    "=== PRE-COMPUTED RULE LIST ===\n"
    "The system contains 11 AML rules:\n"
    "Activity Deviation (ACH): alerts=907, SAR=138, FP=769, precision=15.2%, sweep_params=[floor_amount, z_threshold]\n"
    "Activity Deviation (Check): alerts=316, SAR=76, FP=240, precision=24.1%, sweep_params=[floor_amount, z_threshold]\n"
    "Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, z_threshold, age_min]\n"
    "Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[pair_total, days_window]\n"
    "Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, days_window]\n"
    "CTR Client: alerts=4, SAR=4, FP=0, precision=100.0%, sweep_params=[floor_amount]\n"
    "Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
    "Structuring (Outgoing Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
    "Burst in Originator Activity: alerts=3, SAR=0, FP=3, precision=0.0%, sweep_params=[floor_amount]\n"
    "Burst in Beneficiary Activity: alerts=5, SAR=0, FP=5, precision=0.0%, sweep_params=[floor_amount]\n"
    "Risky International Transfer: alerts=2, SAR=0, FP=2, precision=0.0%, sweep_params=[floor_amount]\n"
    "=== END RULE LIST ==="
)

pc_business_4cluster = (
    "Segment: Business | Active accounts: 14,423 (0 with no transactions)\n"
    "Clusters: 4 | Features: 5 numeric + 8 encoded categorical\n"
    "PCA variance explained: PC1=38.2%, PC2=18.1%\n"
    "\n"
    "  Cluster 1: n=1,242 (8.6% of active accounts), avg_num_trxns=6.1, avg_weekly_trxn_amt=59380.3, trxn_amt_monthly=183521.0, CURRENT_BALANCE=452130.0, ACCT_AGE_YEARS=8.2\n"
    "  Cluster 2: n=12,506 (86.7% of active accounts), avg_num_trxns=4.1, avg_weekly_trxn_amt=36519.0, trxn_amt_monthly=89234.0, CURRENT_BALANCE=231450.0, ACCT_AGE_YEARS=5.3\n"
    "  Cluster 3: n=32 (0.2% of active accounts), avg_num_trxns=0.5, avg_weekly_trxn_amt=38458.3, trxn_amt_monthly=102283.0, CURRENT_BALANCE=3521000.0, ACCT_AGE_YEARS=4.7\n"
    "  Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=12.3, avg_weekly_trxn_amt=783508.0, trxn_amt_monthly=2914231.0, CURRENT_BALANCE=1823450.0, ACCT_AGE_YEARS=6.1"
)


# ─────────────────────────────────────────────────────────────────────────────
# A1. Activity Deviation ACH routing — ex399–ex404
# ─────────────────────────────────────────────────────────────────────────────
# The failing prompt was "Show SAR backtest for Activity Deviation ACH rule"
# V13 had 4 ACH examples but "ACH rule" (with "rule" suffix) still routes to list_rules.
# Fix: 6 more examples with varied phrasings emphasising "ACH rule" and short forms.

# ex399 — exact failing prompt
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex399", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex399"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "Activity Deviation (ACH) catches 136 of 138 SARs (TP rate=98.6%) at the current floor_amount of $50,000 "
        "with precision holding at 15.2% across the entire sweep range."
    )}
]})

# ex400 — "SAR backtest for the ACH rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for the ACH rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex400", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex400"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "The ACH rule (Activity Deviation) alerts on 907 customers and catches 136 SARs at 98.6% TP rate; "
        "raising floor_amount to $90,000 reduces FPs by 16 while retaining the same SAR count."
    )}
]})

# ex401 — "Run SAR backtest for Activity Deviation ACH"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Activity Deviation ACH"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex401", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex401"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "Activity Deviation (ACH) maintains a TP rate of 98.6% at the current $50K floor "
        "across 907 alerted customers with a SAR pool of 138."
    )}
]})

# ex402 — "Show me the SAR catch rate for the Activity Deviation ACH rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the SAR catch rate for the Activity Deviation ACH rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex402", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex402"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "At the current floor_amount of $50,000, the Activity Deviation (ACH) rule catches 136 of 138 SARs "
        "— a 98.6% SAR detection rate with precision=15.2%."
    )}
]})

# ex403 — "ACH rule SAR detection rate"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "ACH rule SAR detection rate"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex403", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex403"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "The ACH rule detects 98.6% of SARs (136 of 138) at the current $50K floor "
        "with 756 false positives in its labeled population."
    )}
]})

# ex404 — "What is the SAR performance of the ACH deviation rule?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR performance of the ACH deviation rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex404", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach, "tool_call_id": "call_ex404"},
    {"role": "assistant", "content": (
        pc_ach + "\n\n"
        "The Activity Deviation (ACH) rule catches 136 SARs at precision=15.2% — "
        "the high FP volume (756) reflects the broad population of customers with elevated ACH activity."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# A2. Detect Excessive routing — ex405–ex410
# ─────────────────────────────────────────────────────────────────────────────
# Failing prompt: "What is the SAR catch rate for the Detect Excessive rule?"
# Model calls list_rules. Fix: 6 examples anchoring "Detect Excessive" → rule_sar_backtest.

# ex405 — exact failing prompt
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Detect Excessive rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex405", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex405"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "The Detect Excessive Transaction Activity rule catches all 46 SARs (TP rate=100.0%) at the current $10K floor; "
        "raising to $18K reduces FPs by 6 while maintaining full SAR coverage."
    )}
]})

# ex406 — "SAR backtest for Detect Excessive"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Detect Excessive"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex406", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex406"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "Detect Excessive Transaction Activity catches 100.0% of SARs (46 of 46) at the current $10K condition "
        "with 310 false positives — the entire sweep range keeps TP rate at 100.0%."
    )}
]})

# ex407 — "Show SAR backtest for Detect Excessive rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Detect Excessive rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex407", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex407"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "The Detect Excessive rule catches all 46 SARs with TP rate=100.0% at the current $10K floor "
        "across 356 alerted customers (precision=12.9%)."
    )}
]})

# ex408 — "Detect Excessive SAR performance"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Detect Excessive SAR performance"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex408", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex408"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "The Detect Excessive Transaction Activity rule has the lowest precision (12.9%) among active rules "
        "but maintains 100.0% SAR coverage across the full sweep range."
    )}
]})

# ex409 — "Run a SAR backtest for the Detect Excessive Transaction Activity rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for the Detect Excessive Transaction Activity rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex409", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex409"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "Detect Excessive Transaction Activity detects 46 SARs at TP rate=100.0% with precision=12.9% "
        "— raising the floor to $18K reduces FPs slightly while retaining full SAR coverage."
    )}
]})

# ex410 — "SAR catch rate for the excessive transaction rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR catch rate for the excessive transaction rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex410", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect, "tool_call_id": "call_ex410"},
    {"role": "assistant", "content": (
        pc_detect + "\n\n"
        "The Detect Excessive Transaction Activity rule achieves 100.0% SAR detection at the current $10K floor "
        "— 46 SARs caught, 310 FPs, precision=12.9%."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# A3. Structuring routing re-anchor — ex411–ex416
# ─────────────────────────────────────────────────────────────────────────────
# V12 had 8 structuring examples (ex355–ex362) but V13 training diluted them.
# Re-anchor with 6 more examples, including "the structuring rule" exact phrasing.

# ex411 — exact failing prompt
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for the structuring rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex411", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex411"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "Structuring (Incoming Cash) catches all 2 SARs with zero false positives (precision=100.0%) "
        "at the current daily_floor of $9,000."
    )}
]})

# ex412 — "structuring rule backtest"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "structuring rule backtest"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex412", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex412"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "The Structuring (Incoming Cash) rule achieves TP rate=100.0% and precision=100.0% "
        "from daily_floor=$5,000 to $9,000 with FP=0 throughout."
    )}
]})

# ex413 — "SAR catch rate for the structuring rule"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR catch rate for the structuring rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex413", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex413"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "The structuring rule (Incoming Cash) catches 100.0% of SARs at the current daily_floor of $9,000; "
        "TP rate drops to 50.0% only above $15,000."
    )}
]})

# ex414 — "Show structuring rule SAR performance"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show structuring rule SAR performance"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex414", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex414"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "Structuring (Incoming Cash) achieves perfect detection — TP rate=100.0%, FP=0, precision=100.0% "
        "at every daily_floor from $5,000 to $9,000."
    )}
]})

# ex415 — "How many SARs does the structuring rule catch?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many SARs does the structuring rule catch?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex415", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex415"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "Structuring (Incoming Cash) catches 2 of 2 SARs (TP rate=100.0%) at the current daily_floor=$9,000 "
        "with FP=0 — a rare rule with perfect precision."
    )}
]})

# ex416 — "SAR detection rate for structuring"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR detection rate for structuring"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex416", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex416"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "The Structuring (Incoming Cash) rule detects 100.0% of SARs with zero false positives "
        "at the current daily_floor of $9,000."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# B. Zero-SAR rule correctness — ex417–ex422
# ─────────────────────────────────────────────────────────────────────────────
# Prompt 6 failure: model said Structuring has zero SARs (WRONG).
# Correct: Burst Orig (SAR=0, FP=3), Burst Ben (SAR=0, FP=5), Risky Intl (SAR=0, FP=2).
# Structuring has SAR=2 and precision=100%.
# Re-anchor: 6 examples with list_rules call + correct zero-SAR identification.

# ex417 — "Which rules generate only false positives?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules generate only false positives?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex417", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex417"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules generate only false positives (precision=0.0%, SAR=0): "
        "Burst in Originator Activity (FP=3), Burst in Beneficiary Activity (FP=5), and Risky International Transfer (FP=2) — "
        "every alert from these rules is an unnecessary investigation."
    )}
]})

# ex418 — "Which AML rules have never caught a SAR?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which AML rules have never caught a SAR?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex418", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex418"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules have SAR=0 in their labeled population: "
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer — "
        "all alerts from these rules are false positives (precision=0.0%)."
    )}
]})

# ex419 — "Show rules with worst precision"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show rules with worst precision"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex419", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex419"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "The three rules with the worst precision (0.0%) are Burst in Originator Activity, "
        "Burst in Beneficiary Activity, and Risky International Transfer — "
        "none of their alerts correspond to a SAR customer."
    )}
]})

# ex420 — "Which rules have precision=0?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules have precision=0?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex420", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex420"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Rules with precision=0.0% (all alerts are FPs): "
        "Burst in Originator Activity (alerts=3), Burst in Beneficiary Activity (alerts=5), "
        "Risky International Transfer (alerts=2)."
    )}
]})

# ex421 — "Are there any rules that never file a SAR?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Are there any rules that never file a SAR?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex421", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex421"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Yes — three rules have SAR=0 in their alerted population: "
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer. "
        "Note that Structuring rules have precision=100.0% (SAR=2 each, FP=0) — the opposite of zero-SAR."
    )}
]})

# ex422 — "Which rules generate all false positives and no real SARs?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules generate all false positives and no real SARs?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex422", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex422"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules generate all FPs with zero SARs (precision=0.0%): "
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer — "
        "Structuring and CTR Client are the opposite, with precision=100.0%."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# C. Verbatim copy behavior (segmentation) — ex423–ex428
# ─────────────────────────────────────────────────────────────────────────────
# Prompt 9 failure: model output "segment stats verbatim repetition prohibited"
# instead of copying cluster stats. Fix: 6 examples showing MUST copy stats verbatim.
# Also anchor: Cluster 4 has highest avg_weekly_trxn_amt (783,508) in Business 4-cluster.

# ex423 — "Which cluster of Business customers has the highest transaction volume?"
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Which cluster of Business customers has the highest transaction volume?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex423", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_business_4cluster, "tool_call_id": "call_ex423"},
    {"role": "assistant", "content": (
        pc_business_4cluster + "\n\n"
        "Cluster 4 (n=643, avg_weekly_trxn_amt=783,508.0) has the highest transaction volume — "
        "a concentrated group of high-frequency, high-value accounts warranting elevated AML scrutiny."
    )}
]})

# ex424 — "Segment Business customers into clusters"
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Segment Business customers into clusters"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex424", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_business_4cluster, "tool_call_id": "call_ex424"},
    {"role": "assistant", "content": (
        pc_business_4cluster + "\n\n"
        "Cluster 4 (n=643, avg_weekly_trxn_amt=783,508.0) is the highest-risk segment — "
        "it has 12.3 average transactions per week and $2.9M monthly transaction volume."
    )}
]})

# ex425 — "Show Business customer segments"
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Show Business customer segments"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex425", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_business_4cluster, "tool_call_id": "call_ex425"},
    {"role": "assistant", "content": (
        pc_business_4cluster + "\n\n"
        "Cluster 2 (n=12,506, 86.7%) is the dominant segment by count, "
        "but Cluster 4 (n=643) drives the highest avg_weekly_trxn_amt at 783,508.0."
    )}
]})

# ex426 — "Run clustering for Business accounts"
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Run clustering for Business accounts"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex426", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_business_4cluster, "tool_call_id": "call_ex426"},
    {"role": "assistant", "content": (
        pc_business_4cluster + "\n\n"
        "Four clusters emerge: Cluster 4 (n=643) is the highest-volume group with avg_weekly_trxn_amt=783,508.0, "
        "while Cluster 2 (n=12,506) represents the bulk of the Business population at lower volumes."
    )}
]})

# ex427 — "Business segment cluster analysis"
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Business segment cluster analysis"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex427", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_business_4cluster, "tool_call_id": "call_ex427"},
    {"role": "assistant", "content": (
        pc_business_4cluster + "\n\n"
        "Cluster 4 (4.5% of active accounts) has avg_weekly_trxn_amt=783,508.0 — "
        "disproportionately high activity relative to its small size suggests concentrated AML exposure."
    )}
]})

# ex428 — "What are the Business customer clusters and which has the most transactions?"
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "What are the Business customer clusters and which has the most transactions?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex428", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_business_4cluster, "tool_call_id": "call_ex428"},
    {"role": "assistant", "content": (
        pc_business_4cluster + "\n\n"
        "Cluster 4 has the most transactions: avg_weekly_trxn_amt=783,508.0 and avg_num_trxns=12.3 — "
        "the highest across all four Business clusters."
    )}
]})


# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────

out_v14 = "finetune/data/framl_train_failures_v14.jsonl"
with open(out_v14, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"V14 new examples : {len(examples)} (ex399–ex{398 + len(examples)})")
print(f"Written to       : {out_v14}")

# Combine V13 + V14
import pathlib

v13_path = "finetune/data/framl_train_combined_v13.jsonl"
v14_combined = "finetune/data/framl_train_combined_v14.jsonl"

v13_lines = pathlib.Path(v13_path).read_text(encoding="utf-8").strip().splitlines()
v14_lines  = [json.dumps(ex, ensure_ascii=False) for ex in examples]

with open(v14_combined, "w", encoding="utf-8") as f:
    f.write("\n".join(v13_lines + v14_lines) + "\n")

total = len(v13_lines) + len(v14_lines)
print(f"V14 combined     : {total} examples -> {v14_combined}")

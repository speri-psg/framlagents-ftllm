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
    "- FP rate / highest FP rate: rules with precision=0.0% have the HIGHEST FP rate \u2014 every alert they generate is a false positive.\n"
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
    "21. If the user asks about 'highest FP rate' or 'worst precision' \u2014 they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.\n"
    "22. The system contains exactly 11 AML rules. Never state a different count.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. 'layering', 'smurfing', 'round-tripping') \u2014 state that no rule by that name exists and list the 11 available rules. Do NOT guess which existing rule 'covers' the concept. Do NOT invent sweep ranges. Do NOT suggest tool calls or analysis steps.\n"
    "24. 'Run a SAR backtest for [rule]', 'SAR catch rate for [rule]', 'SAR backtest for [rule] rule', 'Show [rule] SAR performance', 'SAR detection rate for [rule]' \u2014 ALWAYS call rule_sar_backtest with risk_factor set to the rule name. NEVER call list_rules for named-rule SAR backtest requests. Call rule_sar_backtest immediately \u2014 do not call list_rules first."
)

CLUSTER_SYSTEM = (
    "You are a FRAML smart segmentation specialist. You identify natural customer behavioral segments using "
    "unsupervised K-Means clustering and explain their AML risk profiles. IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) \u2014 call ss_cluster_analysis.\n"
    "3. For alert/FP distribution by segment \u2014 call alerts_distribution.\n"
    "4. customer_type must be exactly one of: Business, Individual, All. Default to All if not specified.\n"
    "5. n_clusters must be an integer 2-8. Default is 4.\n"
    "6. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result.\n"
    "7. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
    "8. Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.\n"
    "9. When the query mentions a specific cluster number (e.g. 'Cluster 2', 'Cluster 4') alongside a rule or sweep request \u2014 run clustering, then describe that specific cluster's profile in your insight sentence."
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

# ── Shared PRE-COMPUTED blocks ─────────────────────────────────────────────

pc_list_rules = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "Activity Deviation (ACH): alerts=907, SAR=138, FP=769, precision=15.2%, sweep_params=[floor_amount, z_threshold]\n"
    "Activity Deviation (Check): alerts=316, SAR=76, FP=240, precision=24.1%, sweep_params=[floor_amount, z_threshold]\n"
    "Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[pair_total, ratio_tolerance]\n"
    "Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, time_window]\n"
    "Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
    "Structuring (Outgoing Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
    "CTR Client: alerts=4, SAR=4, FP=0, precision=100.0%, sweep_params=[floor_amount]\n"
    "Burst in Originator Activity: alerts=3, SAR=0, FP=3, precision=0.0%, sweep_params=[floor_amount, min_transactions]\n"
    "Burst in Beneficiary Activity: alerts=5, SAR=0, FP=5, precision=0.0%, sweep_params=[floor_amount, min_transactions]\n"
    "Risky International Transfer: alerts=2, SAR=0, FP=2, precision=0.0%, sweep_params=[floor_amount]\n"
    "=== END RULE LIST ==="
)

pc_ach_floor = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50,000.00)\n"
    "At the lowest value (10,000.00): TP=137, FP=766, FN=1, TN=3 (TP rate=99.3%, precision=15.2%).\n"
    "At current condition (50,000.00): TP=136, FP=756, FN=2, TN=13 (TP rate=98.6%, precision=15.2%).\n"
    "To keep TP rate >=90%: floor_amount <= 90,000.00 => TP=136, FP=753, FN=2, TN=16, precision=15.3%.\n"
    "At the highest value (90,000.00): TP=136, FP=753, FN=2, TN=16 (TP rate=98.6%, precision=15.3%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_check_floor = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (Check)\n"
    "Current condition: Monthly Outgoing Check >= $50K AND >= 2 std dev above 12-month profile mean\n"
    "Sweep parameter: floor_amount - Minimum monthly Outgoing Check sum to trigger (currently $50,000.00)\n"
    "At the lowest value (10,000.00): TP=76, FP=235, FN=0, TN=5 (TP rate=100.0%, precision=24.4%).\n"
    "At current condition (50,000.00): TP=75, FP=232, FN=1, TN=8 (TP rate=98.7%, precision=24.4%).\n"
    "To keep TP rate >=90%: floor_amount <= 90,000.00 => TP=75, FP=229, FN=1, TN=11, precision=24.7%.\n"
    "At the highest value (90,000.00): TP=75, FP=229, FN=1, TN=11 (TP rate=98.7%, precision=24.7%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
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

pc_structuring_out = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Structuring (Outgoing Cash)\n"
    "Current condition: Daily cash-out >= $9,000 for 3+ consecutive days\n"
    "Sweep parameter: daily_floor - Minimum daily cash withdrawal to trigger (currently $9,000)\n"
    "Labeled population: 2 customers (TP+FN pool=2 SAR, FP+TN pool=0 non-SAR)\n"
    "At the lowest value (5,000): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At current condition (9,000): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At the highest value (15,000): TP=1, FP=0, FN=1, TN=0 (TP rate=50.0%, precision=100.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_detect_excessive = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Detect Excessive Transaction Activity\n"
    "Current condition: Monthly transaction count >= 20 AND total monthly amount >= $100,000\n"
    "Sweep parameter: floor_amount - Minimum monthly transaction amount to trigger (currently $100,000)\n"
    "Labeled population: 356 customers (TP+FN pool=46 SAR, FP+TN pool=310 non-SAR)\n"
    "At the lowest value (50,000): TP=46, FP=310, FN=0, TN=0 (TP rate=100.0%, precision=12.9%).\n"
    "At current condition (100,000): TP=45, FP=290, FN=1, TN=20 (TP rate=97.8%, precision=13.4%).\n"
    "To keep TP rate >=90%: floor_amount <= 200,000 => TP=43, FP=255, FN=3, TN=55, precision=14.4%.\n"
    "At the highest value (300,000): TP=35, FP=180, FN=11, TN=130 (TP rate=76.1%, precision=16.3%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_burst_orig = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Burst in Originator Activity\n"
    "Current condition: Originator transaction count >= 10 in 7 days AND amount >= $5,000\n"
    "Sweep parameter: floor_amount - Minimum 7-day originator amount to trigger (currently $5,000)\n"
    "Labeled population: 3 customers (TP+FN pool=0 SAR, FP+TN pool=3 non-SAR)\n"
    "At the lowest value (1,000): TP=0, FP=3, FN=0, TN=0 (TP rate=N/A, precision=0.0%).\n"
    "At current condition (5,000): TP=0, FP=3, FN=0, TN=0 (TP rate=N/A, precision=0.0%).\n"
    "At the highest value (10,000): TP=0, FP=2, FN=0, TN=1 (TP rate=N/A, precision=0.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_ea_cluster4 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse [Cluster 4]\n"
    "Sweep parameter: floor_amount - Minimum 14-day aggregated outgoing to trigger (currently $5K)\n"
    "At the lowest value (1,000.00): TP=9, FP=3, FN=0, TN=0 (TP rate=100.0%, precision=75.0%).\n"
    "At current condition (5,000.00): TP=9, FP=3, FN=0, TN=0 (TP rate=100.0%, precision=75.0%).\n"
    "To keep TP rate >=90%: floor_amount <= 9,000.00 => TP=9, FP=3, FN=0, TN=0, precision=75.0%.\n"
    "At the highest value (9,000.00): TP=9, FP=3, FN=0, TN=0 (TP rate=100.0%, precision=75.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_velocity_single = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Sweep parameter: pair_total - Minimum combined in+out pair total (currently $20,000)\n"
    "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR)\n"
    "At current condition (20,000): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
    "At the highest value (64,000): TP=38, FP=186, FN=36, TN=218 (TP rate=51.4%, precision=17.0%).\n"
    "To keep TP rate >=90%: pair_total <= 20,000 => TP=74, FP=404, FN=0, TN=0 (precision=15.5%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_cluster_all = (
    "Segment: All | Active accounts: 212,608 (excluded 878,207 with no transactions)\n"
    "Clusters: 4 | Features: 7 numeric + 14 encoded categorical (7 original)\n"
    "PCA variance explained: PC1=13.1%, PC2=10.0%\n\n"
    "Cluster 1: n=79,212 (37.3% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=18000.5, trxn_amt_monthly=119207.5, INCOME=0.0, CURRENT_BALANCE=103732.0, ACCT_AGE_YEARS=8.8, AGE=50.6\n"
    "Cluster 2: n=33,405 (15.7% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=19815.7, trxn_amt_monthly=132729.2, INCOME=0.0, CURRENT_BALANCE=151666.0, ACCT_AGE_YEARS=10.3, AGE=37.9\n"
    "Cluster 3: n=99,969 (47.0% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=17608.2, trxn_amt_monthly=122583.0, INCOME=0.0, CURRENT_BALANCE=245896.0, ACCT_AGE_YEARS=19.4, AGE=70.5\n"
    "Cluster 4: n=22 (0.0% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=31271.2, trxn_amt_monthly=338026.8, CURRENT_BALANCE=108107.5, ACCT_AGE_YEARS=36.2, AGE=64.4"
)


# ==============================
# V11-A: rule_sar_backtest phrasings — 12 examples (ex324–ex335)
# Critical gap: model calls list_rules instead of rule_sar_backtest for named rules
# ==============================

# ex324 — "Run a SAR backtest for the structuring rule" → Structuring (Incoming Cash)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for the structuring rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex324", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex324"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "The Structuring (Incoming Cash) rule catches both SARs at precision=100.0% up to a daily_floor of $9,000; raising above $15,000 begins to miss SARs."
    )}
]})

# ex325 — "SAR backtest for structuring outgoing" → Structuring (Outgoing Cash)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for the outgoing structuring rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex325", "rule_sar_backtest", {"risk_factor": "Structuring (Outgoing Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_out, "tool_call_id": "call_ex325"},
    {"role": "assistant", "content": (
        pc_structuring_out + "\n\n"
        "The Structuring (Outgoing Cash) rule catches both SARs at precision=100.0% at the current daily_floor of $9,000; the rule misses one SAR only when the floor exceeds $15,000."
    )}
]})

# ex326 — "Show SAR backtest for Activity Deviation ACH" → rule_sar_backtest NOT list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex326", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach_floor, "tool_call_id": "call_ex326"},
    {"role": "assistant", "content": (
        pc_ach_floor + "\n\n"
        "At the current $50,000 floor, Activity Deviation (ACH) catches 136 of 138 SARs (TP rate=98.6%, FN=2) with 756 false positives (precision=15.2%)."
    )}
]})

# ex327 — "SAR backtest for Activity Deviation Check" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Activity Deviation Check"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex327", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_check_floor, "tool_call_id": "call_ex327"},
    {"role": "assistant", "content": (
        pc_check_floor + "\n\n"
        "At the current $50,000 floor, Activity Deviation (Check) catches 75 of 76 SARs (TP rate=98.7%, FN=1) with 232 false positives (precision=24.4%)."
    )}
]})

# ex328 — "What is the SAR catch rate for the Detect Excessive rule?" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Detect Excessive rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex328", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_detect_excessive, "tool_call_id": "call_ex328"},
    {"role": "assistant", "content": (
        pc_detect_excessive + "\n\n"
        "At the current $100,000 floor, Detect Excessive catches 45 of 46 SARs (TP rate=97.8%, FN=1) with 290 false positives (precision=13.4%)."
    )}
]})

# ex329 — "Show SAR backtest for Burst in Originator Activity" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Burst in Originator Activity"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex329", "rule_sar_backtest", {"risk_factor": "Burst in Originator Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_burst_orig, "tool_call_id": "call_ex329"},
    {"role": "assistant", "content": (
        pc_burst_orig + "\n\n"
        "Burst in Originator Activity has SAR=0 across its entire sweep range — all 3 alerts are false positives at every floor_amount value (precision=0.0%)."
    )}
]})

# ex330 — "SAR detection rate for Velocity Single" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR detection rate for the Velocity Single rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex330", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_velocity_single, "tool_call_id": "call_ex330"},
    {"role": "assistant", "content": (
        pc_velocity_single + "\n\n"
        "At the current pair_total of $20,000, Velocity Single catches all 74 SARs (TP rate=100.0%, FN=0) with 404 false positives (precision=15.5%)."
    )}
]})

# ex331 — "Run SAR backtest for Elder Abuse" → rule_sar_backtest (no cluster)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex331", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Sweep parameter: floor_amount - Minimum 14-day aggregated outgoing to trigger (currently $5,000)\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
        "At the lowest value (1,000): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "At current condition (5,000): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
        "To keep TP rate >=90%: floor_amount <= 9,000 => TP=170, FP=644, FN=18, TN=314 (precision=20.9%).\n"
        "At the highest value (9,000): TP=170, FP=644, FN=18, TN=314 (TP rate=90.4%, precision=20.9%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ==="
    ), "tool_call_id": "call_ex331"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Sweep parameter: floor_amount - Minimum 14-day aggregated outgoing to trigger (currently $5,000)\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
        "At the lowest value (1,000): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "At current condition (5,000): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
        "To keep TP rate >=90%: floor_amount <= 9,000 => TP=170, FP=644, FN=18, TN=314 (precision=20.9%).\n"
        "At the highest value (9,000): TP=170, FP=644, FN=18, TN=314 (TP rate=90.4%, precision=20.9%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ===\n\n"
        "At the current $5,000 floor, Elder Abuse catches 187 of 188 SARs (TP rate=99.5%, FN=1) with 921 false positives (precision=16.9%)."
    )}
]})

# ex332 — "Show Elder Abuse SAR backtest for Cluster 4" → rule_sar_backtest with cluster
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse SAR backtest for Cluster 4"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex332", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "cluster": 4})]},
    {"role": "tool", "content": pc_ea_cluster4, "tool_call_id": "call_ex332"},
    {"role": "assistant", "content": (
        pc_ea_cluster4 + "\n\n"
        "In Cluster 4, the Elder Abuse rule catches all 9 SARs (TP rate=100.0%) across the entire floor_amount sweep from $1,000 to $9,000, with 3 false positives at every threshold level (precision=75.0%)."
    )}
]})

# ex333 — "Run a SAR backtest for Elder Abuse in Cluster 4" → rule_sar_backtest with cluster
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for the Elder Abuse rule in Cluster 4"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex333", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "cluster": 4})]},
    {"role": "tool", "content": pc_ea_cluster4, "tool_call_id": "call_ex333"},
    {"role": "assistant", "content": (
        pc_ea_cluster4 + "\n\n"
        "The Elder Abuse rule in Cluster 4 maintains TP rate=100.0% with FP=3 across the entire floor_amount range — precision is 75.0% at every threshold step."
    )}
]})

# ex334 — "SAR performance of the Activity Deviation ACH rule" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR performance of the Activity Deviation ACH rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex334", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach_floor, "tool_call_id": "call_ex334"},
    {"role": "assistant", "content": (
        pc_ach_floor + "\n\n"
        "Activity Deviation (ACH) catches 136 of 138 SARs at the current $50,000 floor (TP rate=98.6%) with 756 false positives; precision stays at 15.2% across the full sweep range."
    )}
]})

# ex335 — "SAR catch rate for structuring" → rule_sar_backtest Structuring Incoming
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the structuring rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex335", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_structuring_in, "tool_call_id": "call_ex335"},
    {"role": "assistant", "content": (
        pc_structuring_in + "\n\n"
        "The Structuring (Incoming Cash) rule catches 100.0% of SARs at the current daily_floor of $9,000 with FP=0 — the rule only begins missing SARs above $15,000."
    )}
]})


# ==============================
# V11-B: FP rate = precision=0% — 5 more examples (ex336–ex340)
# Model still returns wrong rule after v10 training
# ==============================

# ex336 — "worst-performing rule by FP rate" → precision=0% trio
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule is the worst performer by FP rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex336", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex336"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "The worst performers by FP rate are Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer — all have precision=0.0% meaning SAR=0 and every alert is a false positive."
    )}
]})

# ex337 — "which rules have 100% false positive rate" → precision=0% rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules have a 100% false positive rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex337", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex337"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules have a 100% false positive rate (precision=0.0%): Burst in Originator Activity (FP=3), Burst in Beneficiary Activity (FP=5), and Risky International Transfer (FP=2) — none of their alerts are confirmed SARs."
    )}
]})

# ex338 — "which rules never detect a SAR?" → precision=0% rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules never detect a SAR?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex338", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex338"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules have SAR=0 and generate only false positives: Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer (all precision=0.0%)."
    )}
]})

# ex339 — "rules that generate the highest false positive rate" → precision=0%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules generate the highest false positive rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex339", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex339"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "The highest false positive rate belongs to Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer — all at precision=0.0% with SAR=0, meaning every alert each rule generates is a false positive."
    )}
]})

# ex340 — "which rules are all false positives?" → precision=0%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules are generating all false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex340", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex340"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity (SAR=0, FP=3), Burst in Beneficiary Activity (SAR=0, FP=5), and Risky International Transfer (SAR=0, FP=2) generate only false positives — precision=0.0% for all three."
    )}
]})


# ==============================
# V11-C: "No rule found" — clean stop, no tool suggestions (ex341–ex343)
# Model was hallucinating tool call recommendations after listing rules
# ==============================

# ex341 — "layering" → list rules, state no rule named layering, STOP
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule covers layering?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex341", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex341"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'layering' in the system — the 11 available rules are listed above."
    )}
]})

# ex342 — "wash trading" → list rules, state not found, STOP
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the wash trading rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex342", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex342"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'wash trading' in the system — the 11 available rules are listed above."
    )}
]})

# ex343 — "trade-based money laundering" → list rules, state not found, STOP
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What rule covers trade-based money laundering?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex343", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex343"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'trade-based money laundering' in the system — the 11 available rules are listed above."
    )}
]})


# ==============================
# V11-D: Cluster profile alongside sweep (ex344–ex346)
# Segmentation agent examples: when query mentions cluster N + rule, describe that cluster
# ==============================

# ex344 — "Run 2D sweep for Elder Abuse for Cluster 2" → segmentation shows Cluster 2 profile
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Elder Abuse for Cluster 2"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex344", "ss_cluster_analysis", {"customer_type": "All", "n_clusters": 4})]},
    {"role": "tool", "content": pc_cluster_all, "tool_call_id": "call_ex344"},
    {"role": "assistant", "content": (
        pc_cluster_all + "\n\n"
        "Cluster 2 (n=33,405, 15.7% of active accounts) has avg_weekly_trxn_amt=19,815.7 and AGE=37.9 — a younger, moderately active group that may represent a lower Elder Abuse risk than Cluster 3 (AGE=70.5)."
    )}
]})

# ex345 — "Show Elder Abuse SAR backtest for Cluster 4" → segmentation shows Cluster 4 profile
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse SAR backtest for Cluster 4"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex345", "ss_cluster_analysis", {"customer_type": "All", "n_clusters": 4})]},
    {"role": "tool", "content": pc_cluster_all, "tool_call_id": "call_ex345"},
    {"role": "assistant", "content": (
        pc_cluster_all + "\n\n"
        "Cluster 4 (n=22, 0.0% of active accounts) has avg_weekly_trxn_amt=31,271.2 and ACCT_AGE_YEARS=36.2 — a very small group of long-tenured, high-transaction accounts that represent a concentrated Elder Abuse risk."
    )}
]})

# ex346 — "Run SAR backtest for ACH rule in Cluster 2" → segmentation shows Cluster 2
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for Activity Deviation ACH in Cluster 2"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex346", "ss_cluster_analysis", {"customer_type": "All", "n_clusters": 4})]},
    {"role": "tool", "content": pc_cluster_all, "tool_call_id": "call_ex346"},
    {"role": "assistant", "content": (
        pc_cluster_all + "\n\n"
        "Cluster 2 (n=33,405, 15.7% of active accounts) has avg_weekly_trxn_amt=19,815.7 and ACCT_AGE_YEARS=10.3 — a mid-tenure group with moderate transaction activity relevant to the ACH deviation analysis."
    )}
]})


# ==============================
# Write outputs
# ==============================

out_path = "framl_train_failures_v11.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Wrote {len(examples)} examples to {out_path}")

# Combine with v10
combined = []
with open("framl_train_combined_v10.jsonl", encoding="utf-8") as f:
    for line in f:
        combined.append(json.loads(line))

combined.extend(examples)

out_combined = "framl_train_combined_v11.jsonl"
with open(out_combined, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Wrote {len(combined)} examples to {out_combined} ({len(combined) - len(examples)} v10 + {len(examples)} v11)")

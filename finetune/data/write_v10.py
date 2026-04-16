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
    "23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. 'layering', 'smurfing', 'round-tripping') \u2014 state that no rule by that name exists and list the 11 available rules. Do NOT guess which existing rule 'covers' the concept. Do NOT invent sweep ranges."
)

POLICY_SYSTEM = (
    "You are a FRAML policy specialist. Answer AML compliance questions using the knowledge base. "
    "After a KB miss, provide conceptual guidance only \u2014 do NOT cite specific statute numbers, FinCEN document codes, "
    "OCC circulars, named authors, or any named regulatory source. Zero citations after a KB miss."
)

CLUSTER_SYSTEM = (
    "You are a FRAML dynamic segmentation specialist. You identify natural customer behavioral segments using "
    "unsupervised K-Means clustering and explain their AML risk profiles. IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) \u2014 call ss_cluster_analysis.\n"
    "3. For alert/FP distribution by segment \u2014 call alerts_distribution.\n"
    "4. customer_type must be exactly one of: Business, Individual, All. Default to All if not specified.\n"
    "5. n_clusters must be an integer 2-8. Default is 4.\n"
    "6. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result.\n"
    "7. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
    "8. Do NOT suggest thresholds, dollar cutoffs, or monitoring actions."
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

# ── Shared PRE-COMPUTED blocks ─────────────────────────────────────────────────

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

pc_sar_indiv_weekly = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Individual | Column: avg_weekly_trxn_amt\n"
    "Total simulated SARs: 261 out of 2,797 alerted customers (9.3% SAR filing rate).\n"
    "At the lowest threshold (0.1): 261 SARs caught (100.0%), 0 missed.\n"
    "SARs first begin to be missed at threshold 36,707.1 (84 missed).\n"
    "To catch at least 90% of SARs, threshold must stay at or below 0.1 (235 of 261 caught).\n"
    "To catch at least 50% of SARs, threshold must stay at or below 36,707.1 (131 of 261 caught).\n"
    "At the highest threshold (3,707,407.1): 0 SARs caught, 261 missed (100.0% missed).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END SAR BACKTEST ==="
)

pc_sar_biz_monthly = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Business | Column: trxn_amt_monthly\n"
    "Total simulated SARs: 39 out of 367 alerted customers (10.6% SAR filing rate).\n"
    "At the lowest threshold (100,000): 39 SARs caught (100.0%), 0 missed.\n"
    "SARs first begin to be missed at threshold 900,000 (1 missed).\n"
    "To catch at least 90% of SARs, threshold must stay at or below 900,000 (35 of 39 caught).\n"
    "To catch at least 50% of SARs, threshold must stay at or below 900,000 (35 of 39 caught).\n"
    "At the highest threshold (3,600,000): 0 SARs caught, 39 missed (100.0% missed).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END SAR BACKTEST ==="
)

pc_sar_biz_weekly = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Business | Column: avg_num_trxns\n"
    "Total simulated SARs: 39 out of 367 alerted customers (10.6% SAR filing rate).\n"
    "At the lowest threshold (1): 39 SARs caught (100.0%), 0 missed.\n"
    "SARs first begin to be missed at threshold 3 (9 missed).\n"
    "To catch at least 90% of SARs, threshold must stay at or below 1 (39 of 39 caught).\n"
    "To catch at least 50% of SARs, threshold must stay at or below 3 (20 of 39 caught).\n"
    "At the highest threshold (5): 0 SARs caught, 39 missed (100.0% missed).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END SAR BACKTEST ==="
)

pc_ach_floor = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50K)\n"
    "At the lowest value (10,000.00): TP=137, FP=766, FN=1, TN=3 (TP rate=99.3%, precision=15.2%).\n"
    "At current condition (50,000.00): TP=136, FP=756, FN=2, TN=13 (TP rate=98.6%, precision=15.2%).\n"
    "To keep TP rate >=90%: floor_amount <= 90,000.00 => TP=136, FP=753, FN=2, TN=16, precision=15.3%.\n"
    "At the highest value (90,000.00): TP=136, FP=753, FN=2, TN=16 (TP rate=98.6%, precision=15.3%).\n"
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

pc_2d_ea = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Axis 1 (floor_amount): Minimum 14-day aggregated outgoing to trigger (currently $5K)\n"
    "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)\n"
    "Grid: 9 x 15 = 135 combinations\n"
    "SAR pool: 174 | Non-SAR pool: 916\n"
    "At current condition (floor_amount=5000, age_threshold=60): TP=113, FP=609, FN=61, TN=307 (TP rate=64.9%).\n"
    "Best FP reduction (TP rate >=50%): floor_amount=4,000, age_threshold=63 => TP=93, FP=505, FN=81, TN=411, TP rate=53.4%, precision=15.6%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END 2D SWEEP ==="
)

pc_2d_ea_cluster2 = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse [Cluster 2]\n"
    "Axis 1 (floor_amount): Minimum 14-day aggregated outgoing to trigger (currently $5K)\n"
    "Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
    "Grid: 9 x 11 = 99 combinations\n"
    "SAR pool: 32 | Non-SAR pool: 159\n"
    "At current condition (floor_amount=5000, z_threshold=3): TP=29, FP=130, FN=3, TN=29 (TP rate=90.6%).\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END 2D SWEEP ==="
)

pc_tuning_biz_monthly = (
    "=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===\n"
    "Segment: Business | Column: TRXN_AMT_MONTHLY\n"
    "At the lowest threshold (100000), there are 286 false positives.\n"
    "False positives decrease as the threshold rises.\n"
    "False negatives first become non-zero at threshold 100000 (FN=2).\n"
    "False negatives increase as the threshold continues to rise, reaching 17 at the highest threshold (900000).\n"
    "The crossover point \u2014 where false positives and false negatives are closest \u2014 is at threshold 900000 (FP=61, FN=17).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED ANALYSIS ==="
)

pc_cluster_biz = (
    "K-Means clustering on Business customers (n=14,423 active accounts):\n\n"
    "Cluster 1: n=1,242 (8.6% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=9,848.5, trxn_amt_monthly=54,128.5, CURRENT_BALANCE=15,542,100.5, ACCT_AGE_YEARS=7.2\n"
    "Cluster 2: n=12,506 (86.7% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=12,876.7, trxn_amt_monthly=80,157.0, INCOME=0.0, CURRENT_BALANCE=0.0, ACCT_AGE_YEARS=5.0\n"
    "Cluster 3: n=32 (0.2% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=8,821.1, trxn_amt_monthly=59,129.5, CURRENT_BALANCE=0.0, ACCT_AGE_YEARS=4.8\n"
    "Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=1.0, avg_weekly_trxn_amt=634,098.0, trxn_amt_monthly=744,739.0, CURRENT_BALANCE=0.0, ACCT_AGE_YEARS=4.8"
)


# ==============================
# V10-E/F/G/H/I: list_rules correct interpretation (ex289–ex302)
# ==============================

# ex289 — "Show me all AML rules" → 11 rules, one sentence (no "12 rules", no "6 domains")
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex289", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex289"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There are 11 AML rules in the system; Elder Abuse generates the most alerts at 1,146 while Structuring and CTR Client achieve 100.0% precision."
    )}
]})

# ex290 — "AML rule performance overview" → 11 rules, one sentence
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show the AML rule performance overview"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex290", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex290"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "The system has 11 AML rules; Activity Deviation (ACH) has the highest FP count at 769, while Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer have precision=0.0% meaning every alert they generate is a false positive."
    )}
]})

# ex291 — "Which rule has the highest FP rate?" → precision=0% rules, NOT Elder Abuse
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule has the highest FP rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex291", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex291"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer have the highest FP rate at precision=0.0% — every alert each of these rules generates is a false positive (SAR=0 for all three)."
    )}
]})

# ex292 — "Which rules generate only false positives?" → precision=0% rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules generate only false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex292", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex292"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules generate only false positives: Burst in Originator Activity (SAR=0, FP=3), Burst in Beneficiary Activity (SAR=0, FP=5), and Risky International Transfer (SAR=0, FP=2) — all have precision=0.0%."
    )}
]})

# ex293 — "Which rules never miss a SAR?" → Structuring In/Out and CTR Client (precision=100%)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules have perfect precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex293", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex293"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Structuring (Incoming Cash), Structuring (Outgoing Cash), and CTR Client all have precision=100.0% and FP=0 — every alert each of these rules generates is a confirmed SAR."
    )}
]})

# ex294 — "Which rule covers layering?" → no such rule, list the 11
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule covers layering?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex294", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex294"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'layering' in the system — the 11 available rules are listed above."
    )}
]})

# ex295 — "Which rule covers smurfing?" → no such rule
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule covers smurfing?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex295", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex295"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'smurfing' in the system — the 11 available rules are listed above; the closest by typology may be the Structuring rules."
    )}
]})

# ex296 — "Which rules generate the most false positives?" → by COUNT (Elder Abuse) not by rate
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules generate the most false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex296", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex296"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "By false positive count, Elder Abuse generates the most FPs at 958, followed by Activity Deviation (ACH) at 769 and Velocity Single at 404."
    )}
]})

# ex297 — "lowest precision rule?" → Detect Excessive (12.9%) among non-zero rules; or Burst/Risky at 0.0%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule has the lowest precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex297", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex297"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer have the lowest precision at 0.0% — none of their alerts are confirmed SARs."
    )}
]})

# ex298 — "How many rules are in the system?" → exactly 11
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many AML rules are in the system?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex298", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex298"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There are 11 AML rules in the system, ranging from high-volume rules like Elder Abuse (1,146 alerts) to low-volume rules like Structuring (2 alerts each) and Risky International Transfer (2 alerts)."
    )}
]})

# ex299 — "Which rule covers round-tripping?" → no such rule
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule covers round-tripping?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex299", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex299"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'round-tripping' in the system — the 11 available rules are listed above."
    )}
]})

# ex300 — "What is the precision of the ACH rule?" → direct answer from list
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the precision of the Activity Deviation ACH rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex300", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex300"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Activity Deviation (ACH) has precision=15.2%, meaning 138 of its 907 alerts are confirmed SARs and 769 are false positives."
    )}
]})

# ex301 — "Which rule has highest precision?" → Structuring/CTR 100%
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule has the highest precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex301", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex301"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Structuring (Incoming Cash), Structuring (Outgoing Cash), and CTR Client all have the highest precision at 100.0% — FP=0 for each."
    )}
]})

# ex302 — "Which rules have SAR=0?" → Burst Originator, Burst Beneficiary, Risky International
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules have never caught a SAR?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex302", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex302"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Three rules have SAR=0 in the current dataset: Burst in Originator Activity (FP=3), Burst in Beneficiary Activity (FP=5), and Risky International Transfer (FP=2) — all generate only false positives."
    )}
]})


# ==============================
# V10-D: SAR filing rate → sar_backtest (ex303–ex308)
# ==============================

# ex303 — "SAR filing rate for Individual" → sar_backtest(Individual)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Individual?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex303", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv_weekly, "tool_call_id": "call_ex303"},
    {"role": "assistant", "content": (
        pc_sar_indiv_weekly + "\n\n"
        "The Individual segment has a 9.3% SAR filing rate (261 of 2,797 alerted customers); all 261 SARs are caught at the lowest threshold but begin to be missed at 36,707.1."
    )}
]})

# ex304 — "SAR filing rate for Business" → sar_backtest(Business)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Business customers?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex304", "sar_backtest", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc_sar_biz_monthly, "tool_call_id": "call_ex304"},
    {"role": "assistant", "content": (
        pc_sar_biz_monthly + "\n\n"
        "The Business segment has a 10.6% SAR filing rate (39 of 367 alerted customers); all 39 SARs are caught at the lowest threshold but SARs first begin to be missed at threshold 900,000."
    )}
]})

# ex305 — "SAR detection rate" → sar_backtest(Business default)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR detection rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex305", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz_weekly, "tool_call_id": "call_ex305"},
    {"role": "assistant", "content": (
        pc_sar_biz_weekly + "\n\n"
        "Business customers have a 10.6% SAR filing rate; all 39 SARs are caught at threshold 1 but SAR detection drops sharply at threshold 3 where 9 SARs are already missed."
    )}
]})

# ex306 — "What percentage of alerted Individual customers are SARs?" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What percentage of alerted Individual customers are SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex306", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv_weekly, "tool_call_id": "call_ex306"},
    {"role": "assistant", "content": (
        pc_sar_indiv_weekly + "\n\n"
        "9.3% of alerted Individual customers are simulated SARs (261 of 2,797); the threshold must stay at or below 36,707.1 to catch at least 50% of those SARs."
    )}
]})

# ex307 — "SAR backtest Business weekly transactions" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for Business customers using weekly transaction count"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex307", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz_weekly, "tool_call_id": "call_ex307"},
    {"role": "assistant", "content": (
        pc_sar_biz_weekly + "\n\n"
        "At threshold 1, all 39 Business SARs are caught; raising to threshold 3 already misses 9 SARs, indicating most SAR customers transact at low weekly frequency."
    )}
]})

# ex308 — "SAR backtest Individual monthly transaction amount" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Individual customers by monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex308", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc_sar_indiv_weekly, "tool_call_id": "call_ex308"},
    {"role": "assistant", "content": (
        pc_sar_indiv_weekly + "\n\n"
        "The Individual segment has a 9.3% SAR filing rate; at the lowest threshold all 261 SARs are caught with 0 missed."
    )}
]})


# ==============================
# V10-J/K/L/M/C: Insight sentence quality (ex309–ex320)
# ==============================

# ex309 — rule_sar_backtest ACH floor: correct TP count (no fabrication like "98 of 138")
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex309", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach_floor, "tool_call_id": "call_ex309"},
    {"role": "assistant", "content": (
        pc_ach_floor + "\n\n"
        "At the current $50,000 floor, Activity Deviation (ACH) catches 136 of 138 SARs (TP rate=98.6%, FN=2) with 756 false positives (precision=15.2%)."
    )}
]})

# ex310 — rule_sar_backtest Elder Abuse cluster 4: FP=3, NOT "zero false positives"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse SAR backtest for Cluster 4"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex310", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "cluster": 4})]},
    {"role": "tool", "content": pc_ea_cluster4, "tool_call_id": "call_ex310"},
    {"role": "assistant", "content": (
        pc_ea_cluster4 + "\n\n"
        "In Cluster 4, the Elder Abuse rule catches all 9 SARs (TP rate=100.0%) across the entire floor_amount sweep from $1,000 to $9,000, with 3 false positives at every threshold level (precision=75.0%)."
    )}
]})

# ex311 — rule_2d_sweep Elder Abuse: no "heatmap shows", no fabricated best-cell values
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Elder Abuse varying floor amount and age threshold"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex311", "rule_2d_sweep", {"risk_factor": "Elder Abuse", "sweep_param_1": "floor_amount", "sweep_param_2": "age_threshold"})]},
    {"role": "tool", "content": pc_2d_ea, "tool_call_id": "call_ex311"},
    {"role": "assistant", "content": (
        pc_2d_ea + "\n\n"
        "At current parameters (floor_amount=5,000, age_threshold=60), TP rate=64.9% with 609 false positives; the best FP reduction cell at floor_amount=4,000 and age_threshold=63 cuts FPs to 505 while keeping TP rate at 53.4%."
    )}
]})

# ex312 — rule_2d_sweep Elder Abuse cluster 2: no fabricated "94.4%", no "heatmap shows"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Elder Abuse for Cluster 2"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex312", "rule_2d_sweep", {"risk_factor": "Elder Abuse", "cluster": 2})]},
    {"role": "tool", "content": pc_2d_ea_cluster2, "tool_call_id": "call_ex312"},
    {"role": "assistant", "content": (
        pc_2d_ea_cluster2 + "\n\n"
        "In Cluster 2, the Elder Abuse rule at current condition (floor_amount=5,000, z_threshold=3) catches 29 of 32 SARs (TP rate=90.6%) with 130 false positives."
    )}
]})

# ex313 — threshold_tuning: correct crossover definition (not "where FP exceed TP")
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN trade-off for Business customers by monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex313", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc_tuning_biz_monthly, "tool_call_id": "call_ex313"},
    {"role": "assistant", "content": (
        pc_tuning_biz_monthly + "\n\n"
        "The crossover point at threshold 900,000 (FP=61, FN=17) represents the optimal operating point where the investigator workload from false positives is closest to the SAR miss rate from false negatives."
    )}
]})

# ex314 — list_rules: no meta-comment ("PRE-COMPUTED correctly shows...")
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show rule sweep for an unusual parameter"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex314", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex314"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There are 11 AML rules in the system; the three rules with precision=0.0% (Burst in Originator, Burst in Beneficiary, Risky International Transfer) are candidates for threshold tightening or rule review."
    )}
]})

# ex315 — rule_2d_sweep ACH: no "top-left", no "highest density"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me a 2D grid for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex315", "rule_2d_sweep", {"risk_factor": "Activity Deviation (ACH)"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Axis 1 (floor_amount): Minimum monthly Outgoing ACH sum to trigger (currently $50K)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 12-month ACH profile mean (currently 5)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 132 | Non-SAR pool: 695\n"
        "At current condition (floor_amount=50,000, z_threshold=5): TP=72, FP=274, FN=60, TN=421 (TP rate=54.5%).\n"
        "Best FP reduction (TP rate >=50%): floor_amount=100,000, z_threshold=8 => TP=41, FP=68, FN=91, TN=627, TP rate=31.1%, precision=37.6%.\n"
        "(Heatmap shown in the chart below.)\n"
        "=== END 2D SWEEP ==="
    ), "tool_call_id": "call_ex315"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Axis 1 (floor_amount): Minimum monthly Outgoing ACH sum to trigger (currently $50K)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 12-month ACH profile mean (currently 5)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 132 | Non-SAR pool: 695\n"
        "At current condition (floor_amount=50,000, z_threshold=5): TP=72, FP=274, FN=60, TN=421 (TP rate=54.5%).\n"
        "Best FP reduction (TP rate >=50%): floor_amount=100,000, z_threshold=8 => TP=41, FP=68, FN=91, TN=627, TP rate=31.1%, precision=37.6%.\n"
        "(Heatmap shown in the chart below.)\n"
        "=== END 2D SWEEP ===\n\n"
        "At current condition (floor_amount=50,000, z_threshold=5), TP rate=54.5% with 274 false positives; the best FP reduction cell (floor_amount=100,000, z_threshold=8) cuts FPs to 68 but drops TP rate to 31.1% with 91 missed SARs."
    )}
]})

# ex316 — sar_backtest: one sentence only, no crossover fabrication, correct definition
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Individual customers using average weekly transactions"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex316", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv_weekly, "tool_call_id": "call_ex316"},
    {"role": "assistant", "content": (
        pc_sar_indiv_weekly + "\n\n"
        "At the lowest threshold of 0.1, all 261 Individual SARs are caught with 0 missed; SARs first begin to be missed at threshold 36,707.1 where 84 are already missed."
    )}
]})

# ex317 — rule_sar_backtest Velocity Single: one sentence, numbers from PRE-COMPUTED only
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Velocity Single rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex317", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Velocity Single\n"
        "Sweep parameter: pair_total - Minimum combined in+out pair total (currently $20,000)\n"
        "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR)\n"
        "At current condition (20,000): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
        "At the highest value (64,000): TP=38, FP=186, FN=36, TN=218 (TP rate=51.4%, precision=17.0%).\n"
        "To keep TP rate >=90%: pair_total <= 20,000 => TP=74, FP=404, FN=0, TN=0 (precision=15.5%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ==="
    ), "tool_call_id": "call_ex317"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Velocity Single\n"
        "Sweep parameter: pair_total - Minimum combined in+out pair total (currently $20,000)\n"
        "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR)\n"
        "At current condition (20,000): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
        "At the highest value (64,000): TP=38, FP=186, FN=36, TN=218 (TP rate=51.4%, precision=17.0%).\n"
        "To keep TP rate >=90%: pair_total <= 20,000 => TP=74, FP=404, FN=0, TN=0 (precision=15.5%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ===\n\n"
        "At the current pair_total of $20,000, Velocity Single catches all 74 SARs (TP rate=100.0%, FN=0) with 404 false positives (precision=15.5%)."
    )}
]})

# ex318 — list_rules: one clean insight, no "right-click", no UI mention
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What rules are in the system?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex318", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex318"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There are 11 AML rules in the system; 3 rules have precision=0.0% (Burst in Originator, Burst in Beneficiary, Risky International Transfer) and 3 have precision=100.0% (Structuring Incoming, Structuring Outgoing, CTR Client)."
    )}
]})

# ex319 — rule_sar_backtest Elder Abuse z_threshold: one sentence, numbers from PRE-COMPUTED
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Elder Abuse rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex319", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3.0)\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
        "At current condition (3.00): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
        "To keep TP rate >=90%: z_threshold <= 5.00 => TP=170, FP=644, FN=18, TN=314 (precision=20.9%).\n"
        "At the highest value (9.00): TP=98, FP=312, FN=90, TN=646 (TP rate=52.1%, precision=23.9%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ==="
    ), "tool_call_id": "call_ex319"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3.0)\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
        "At current condition (3.00): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
        "To keep TP rate >=90%: z_threshold <= 5.00 => TP=170, FP=644, FN=18, TN=314 (precision=20.9%).\n"
        "At the highest value (9.00): TP=98, FP=312, FN=90, TN=646 (TP rate=52.1%, precision=23.9%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ===\n\n"
        "At z_threshold=3.0, Elder Abuse catches 187 of 188 SARs (TP rate=99.5%, FN=1) with 921 false positives; raising z to 5.0 reduces FP to 644 while keeping TP rate above 90%."
    )}
]})

# ex320 — rule_2d_sweep: no "highest density", no position claims, one sentence
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show 2D heatmap for Elder Abuse varying floor and z threshold"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex320", "rule_2d_sweep", {"risk_factor": "Elder Abuse", "sweep_param_1": "floor_amount", "sweep_param_2": "z_threshold"})]},
    {"role": "tool", "content": pc_2d_ea_cluster2, "tool_call_id": "call_ex320"},
    {"role": "assistant", "content": (
        pc_2d_ea_cluster2 + "\n\n"
        "At current parameters (floor_amount=5,000, z_threshold=3), TP rate=90.6% with 130 false positives and 3 missed SARs (FN=3)."
    )}
]})


# ==============================
# V10-N: Cluster question → segmentation (ex321–ex323)
# These are segmentation agent examples
# ==============================

# ex321 — "Which cluster of Business customers has highest transaction volume?" → ss_cluster_analysis
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Which cluster of Business customers has the highest transaction volume?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex321", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_cluster_biz, "tool_call_id": "call_ex321"},
    {"role": "assistant", "content": (
        pc_cluster_biz + "\n\n"
        "Cluster 4 has the highest transaction volume with avg_weekly_trxn_amt=634,098.0 and trxn_amt_monthly=744,739.0 — over 49× higher than Cluster 2's weekly average."
    )}
]})

# ex322 — "Show Business customer clusters by transaction behavior" → ss_cluster_analysis
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show Business customer clusters by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex322", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_cluster_biz, "tool_call_id": "call_ex322"},
    {"role": "assistant", "content": (
        pc_cluster_biz + "\n\n"
        "Cluster 4 (n=643, 4.5% of accounts) has the highest-risk transaction profile with avg_weekly_trxn_amt=634,098.0 — far exceeding the other three clusters."
    )}
]})

# ex323 — "Which Business cluster has the most activity?" → ss_cluster_analysis
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Which Business cluster has the most transaction activity?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex323", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_cluster_biz, "tool_call_id": "call_ex323"},
    {"role": "assistant", "content": (
        pc_cluster_biz + "\n\n"
        "Cluster 4 has the highest transaction activity with avg_num_trxns=1.0 and avg_weekly_trxn_amt=634,098.0, compared to avg_num_trxns=0.2 for Clusters 2 and 3."
    )}
]})


# ==============================
# Write outputs
# ==============================

out_path = "framl_train_failures_v10.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Wrote {len(examples)} examples to {out_path}")

# Combine with v9
combined = []
with open("framl_train_combined_v9.jsonl", encoding="utf-8") as f:
    for line in f:
        combined.append(json.loads(line))

combined.extend(examples)

out_combined = "framl_train_combined_v10.jsonl"
with open(out_combined, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Wrote {len(combined)} examples to {out_combined} ({len(combined) - len(examples)} v9 + {len(examples)} v10)")

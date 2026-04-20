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
    "- FP rate: FP / (FP + TN) OR FP / total alerts \u2014 share of alerts that are false positives. High FP rate = low precision.\n"
    "- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).\n"
    "- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory.\n"
    "2. For any question about FP, FN, threshold, alert rates, or transactions \u2014 call threshold_tuning.\n"
    "3. For general segment counts or totals \u2014 call segment_stats.\n"
    "4. For any question about SAR catch rate, SAR detection, how many SARs a threshold catches, or SAR backtest \u2014 call sar_backtest.\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "6. segment must be exactly one of: Business, Individual\n"
    "7. If the user does not specify a segment, default to Business.\n"
    "8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.\n"
    "9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.\n"
    "10. Do NOT paraphrase, round, or restate the numbers differently.\n"
    "11. Do NOT include JSON or code blocks in your final reply.\n"
    "12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.\n"
    "13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.\n"
    "14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold \u2014 do NOT call the tool. Reject the request and state that the only valid parameters are segment (Business or Individual) and threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.\n"
    "15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes \u2014 call rule_sar_backtest with risk_factor and optionally sweep_param. Do NOT call list_rules first.\n"
    "16. For any question about which rules exist, which rules generate the most FPs, a rule performance overview, or to show all AML rules \u2014 call list_rules. Do NOT call any other tool.\n"
    "17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR \u2014 call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.\n"
    "18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. 'hover to see', 'right-click to select', 'click the cell'). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.\n"
    "19. The sweep_param argument is the NAME of the parameter (e.g. 'z_threshold', 'daily_floor', 'pair_total') \u2014 never a numeric value. If the user asks 'what happens at value X', call the sweep tool with the correct sweep_param name, then answer from the returned sweep data.\n"
    "20. ONE insight sentence only. Do NOT add a second sentence, parenthetical, or bullet after the insight. Do NOT characterize results as 'strong', 'weak', 'excellent', or any editorial adjective not in the PRE-COMPUTED data. Do NOT describe cross-cell heatmap trends, heatmap patterns, or movement across axes \u2014 report only the current cell and the best-FP-reduction cell as stated in the PRE-COMPUTED output.\n"
    "21. If the user asks about 'highest FP rate' or 'worst precision', they mean the ratio FP/(FP+TP), NOT the raw FP count. Rules with zero SARs (precision=0.0%) have the highest FP rate."
)

POLICY_SYSTEM = (
    "You are a FRAML policy specialist. Answer AML compliance questions using the knowledge base. "
    "After a KB miss, provide conceptual guidance only \u2014 do NOT cite specific statute numbers, FinCEN document codes, "
    "OCC circulars, named authors, or any named regulatory source. Zero citations after a KB miss."
)

CLUSTER_SYSTEM = (
    "You are a FRAML dynamic segmentation specialist. You identify natural customer behavioral segments using "
    "unsupervised K-Means clustering and explain their AML risk profiles. IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) \u2014 call ds_cluster_analysis.\n"
    "3. For alert/FP distribution by segment \u2014 call alerts_distribution.\n"
    "4. For the legacy alerts dataset \u2014 call cluster_analysis only if the user explicitly asks.\n"
    "5. Do NOT call multiple segmentation tools for the same request \u2014 pick exactly one.\n"
    "6. customer_type must be exactly one of: Business, Individual, All\n"
    "   If the user does NOT specify a customer type, default to All.\n"
    "7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.\n"
    "8. If the user asks to prepare or refresh the raw data \u2014 call prepare_segmentation_data first.\n"
    "9. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result. Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.\n"
    "10. If the user asks to show specific clusters (e.g. 'show only cluster 3', 'highest risk', 'top 2 high risk', 'low activity clusters'):\n"
    "    - Identify which cluster number(s) match the request from the stats\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY this and nothing else:\n"
    "      DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers\n"
    "    - Do NOT mention this line in your text \u2014 it is a system directive, not for the user.\n"
    "    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.\n"
    "11. Do NOT include JSON, code blocks, or raw data tables in your final reply.\n"
    "12. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
    "13. Do NOT invent threshold values, dollar amounts, or cutoffs.\n"
    "14. If the user asks which cluster to set a threshold for, or asks for threshold recommendations per cluster \u2014 do NOT invent values. Tell the user to use the threshold_tuning tool with the relevant segment and threshold_column parameters instead."
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

pc_struct_in = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Structuring (Incoming Cash)\n"
    "Sweep parameter: daily_floor - Minimum daily Cash CashIn total for a qualifying day (currently $3K)\n"
    "Current value: 3,000\n"
    "Labeled population: 2 customers (TP+FN pool=2 SAR, FP+TN pool=0 non-SAR, precision=100.0%)\n"
    "At the lowest value (500.00): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At current condition (3,000.00): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "To keep TP rate >=90%: daily_floor <= 6,500.00 => TP=2, FP=0, FN=0, TN=0, precision=100.0%.\n"
    "At the highest value (6,500.00): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_struct_out = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Structuring (Outgoing Cash)\n"
    "Sweep parameter: daily_floor - Minimum daily Cash CashOut total for a qualifying day (currently $7K)\n"
    "Current value: 7,000\n"
    "Labeled population: 2 customers (TP+FN pool=2 SAR, FP+TN pool=0 non-SAR, precision=100.0%)\n"
    "At the lowest value (1,000.00): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "At current condition (7,000.00): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "To keep TP rate >=90%: daily_floor <= 11,000.00 => TP=2, FP=0, FN=0, TN=0, precision=100.0%.\n"
    "At the highest value (11,000.00): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_ach_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month ACH profile mean (currently 5.0)\n"
    "Labeled population: 827 customers (TP+FN pool=132 SAR, FP+TN pool=695 non-SAR)\n"
    "At the lowest value (0.00): TP=126, FP=672, FN=6, TN=23 (TP rate=95.5%, precision=15.8%).\n"
    "At current condition (5.00): TP=72, FP=274, FN=60, TN=421 (TP rate=54.5%, precision=20.8%).\n"
    "To keep TP rate >=90%: z_threshold <= 1.00 => TP=120, FP=637, FN=12, TN=58 (precision=15.9%).\n"
    "At the highest value (10.00): TP=30, FP=75, FN=102, TN=620 (TP rate=22.7%, precision=28.6%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_ea_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3.0)\n"
    "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
    "At current condition (3.00): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
    "To keep TP rate >=90%: z_threshold <= 5.00 => TP=170, FP=644, FN=18, TN=314 (precision=20.9%).\n"
    "At the highest value (9.00): TP=98, FP=312, FN=90, TN=646 (TP rate=52.1%, precision=23.9%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_vs_pt = (
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

pc_det_exc = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Detect Excessive Transaction Activity\n"
    "Sweep parameter: floor_amount - Minimum total outgoing to trigger (currently $10,000)\n"
    "Labeled population: 356 customers (TP+FN pool=46 SAR, FP+TN pool=310 non-SAR)\n"
    "At the lowest value (1,000): TP=44, FP=302, FN=2, TN=8 (TP rate=95.7%, precision=12.7%).\n"
    "At current condition (10,000): TP=38, FP=248, FN=8, TN=62 (TP rate=82.6%, precision=13.3%).\n"
    "To keep TP rate >=90%: floor_amount <= 4,000 => TP=42, FP=291, FN=4, TN=19 (precision=12.6%).\n"
    "At the highest value (40,000): TP=18, FP=94, FN=28, TN=216 (TP rate=39.1%, precision=16.1%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_check_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (Check)\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month Check profile mean (currently 2.0)\n"
    "Labeled population: 316 customers (TP+FN pool=76 SAR, FP+TN pool=240 non-SAR)\n"
    "At current condition (2.00): TP=36, FP=73, FN=40, TN=167 (TP rate=47.4%, precision=33.0%).\n"
    "To keep TP rate >=90%: z_threshold <= 0.00 => TP=69, FP=195, FN=7, TN=45 (precision=26.1%).\n"
    "At the highest value (8.00): TP=12, FP=18, FN=64, TN=222 (TP rate=15.8%, precision=40.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

pc_2d_vs = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Axis 1 (pair_total): Minimum combined in+out pair total to trigger (currently $20K)\n"
    "Axis 2 (ratio_tolerance): Max deviation of out/in ratio from 1.0 to trigger (currently 10% = 90-110%)\n"
    "Grid: 8 x 9 = 72 combinations\n"
    "SAR pool: 74 | Non-SAR pool: 404\n"
    "At current condition (pair_total=20,000, ratio_tolerance=10%): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=50%): pair_total=40,000, ratio_tolerance=7.5% => TP=52, FP=273, FN=22, TN=131, TP rate=70.3%, precision=16.0%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END 2D SWEEP ==="
)

pc_2d_ach = (
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
)

pc_sar_biz = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Business | threshold_column: AVG_TRXNS_WEEK\n"
    "Total customers: 367 | SAR customers: 36 | Non-SAR: 331\n"
    "At current threshold: TP=32, FP=218, FN=4, TN=113 (TP rate=88.9%, precision=12.8%).\n"
    "=== END SAR BACKTEST ==="
)

pc_sar_indiv = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Individual | threshold_column: AVG_TRXN_AMT\n"
    "Total customers: 4,668 | SAR customers: 684 | Non-SAR: 3,984\n"
    "At current threshold: TP=612, FP=1,847, FN=72, TN=2,137 (TP rate=89.5%, precision=24.9%).\n"
    "=== END SAR BACKTEST ==="
)


# ==============================
# V9-C: SAR backtest for named rule → rule_sar_backtest directly (NOT list_rules) (ex243–ex250)
# ==============================

# ex243
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for the structuring rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex243", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_struct_in, "tool_call_id": "call_ex243"},
    {"role": "assistant", "content": (
        pc_struct_in + "\n\n"
        "The Structuring (Incoming Cash) rule catches both SAR customers at every daily_floor from $500 to $6,500 with zero false positives — "
        "the current $3,000 floor already achieves 100% detection and 100% precision."
    )}
]})

# ex244
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me a SAR backtest for the Elder Abuse rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex244", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ea_z, "tool_call_id": "call_ex244"},
    {"role": "assistant", "content": (
        pc_ea_z + "\n\n"
        "The current z_threshold=3.0 catches 99.5% of Elder Abuse SARs (FN=1) but generates 921 false positives (precision=16.9%); "
        "raising z to 5.0 reduces FP to 644 while keeping TP rate above 90%."
    )}
]})

# ex245
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for the Detect Excessive rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex245", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_det_exc, "tool_call_id": "call_ex245"},
    {"role": "assistant", "content": (
        pc_det_exc + "\n\n"
        "At the current $10,000 floor_amount, Detect Excessive catches 82.6% of SARs with 248 false positives (precision=13.3%); "
        "lowering to $4,000 raises TP rate to 90% at the cost of 43 additional false positives."
    )}
]})

# ex246 — "SAR backtest for Activity Deviation ACH" → rule_sar_backtest, NOT list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex246", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_ex246"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "At z_threshold=5.0, Activity Deviation (ACH) catches 54.5% of SARs with 274 false positives (precision=20.8%); "
        "to reach 90% TP rate, z must be lowered to 1.0 which raises FP to 637."
    )}
]})

# ex247 — "SAR performance for Check rule" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR performance for the Activity Deviation Check rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex247", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_check_z, "tool_call_id": "call_ex247"},
    {"role": "assistant", "content": (
        pc_check_z + "\n\n"
        "Activity Deviation (Check) at z=2.0 catches 47.4% of Check SARs with 73 false positives; "
        "to maintain a 90% TP rate, z must be lowered to 0.0 which raises FP to 195."
    )}
]})

# ex248 — "Run SAR backtest for structuring outgoing" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for the structuring outgoing cash rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex248", "rule_sar_backtest", {"risk_factor": "Structuring (Outgoing Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_struct_out, "tool_call_id": "call_ex248"},
    {"role": "assistant", "content": (
        pc_struct_out + "\n\n"
        "Structuring (Outgoing Cash) detects both SAR customers at all daily_floor values from $1,000 to $11,000 with zero false positives — "
        "precision remains 100.0% across the entire sweep range."
    )}
]})

# ex249 — "SAR catch rate for Velocity Single" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR catch rate for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex249", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_vs_pt, "tool_call_id": "call_ex249"},
    {"role": "assistant", "content": (
        pc_vs_pt + "\n\n"
        "At the current $20,000 pair_total, Velocity Single catches all 74 SARs (TP rate=100.0%, FN=0) with 404 false positives (precision=15.5%)."
    )}
]})

# ex250 — "show me ACH SAR analysis" → rule_sar_backtest with floor_amount
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the ACH rule SAR analysis"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex250", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50,000)\n"
        "Labeled population: 827 customers (TP+FN pool=132 SAR, FP+TN pool=695 non-SAR)\n"
        "At the lowest value (10,000): TP=131, FP=694, FN=1, TN=1 (TP rate=99.2%, precision=15.9%).\n"
        "At current condition (50,000): TP=129, FP=673, FN=3, TN=22 (TP rate=97.7%, precision=16.1%).\n"
        "To keep TP rate >=90%: floor_amount <= 200,000 => TP=119, FP=584, FN=13, TN=111 (precision=16.9%).\n"
        "At the highest value (500,000): TP=68, FP=214, FN=64, TN=481 (TP rate=51.5%, precision=24.1%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ==="
    ), "tool_call_id": "call_ex250"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50,000)\n"
        "Labeled population: 827 customers (TP+FN pool=132 SAR, FP+TN pool=695 non-SAR)\n"
        "At the lowest value (10,000): TP=131, FP=694, FN=1, TN=1 (TP rate=99.2%, precision=15.9%).\n"
        "At current condition (50,000): TP=129, FP=673, FN=3, TN=22 (TP rate=97.7%, precision=16.1%).\n"
        "To keep TP rate >=90%: floor_amount <= 200,000 => TP=119, FP=584, FN=13, TN=111 (precision=16.9%).\n"
        "At the highest value (500,000): TP=68, FP=214, FN=64, TN=481 (TP rate=51.5%, precision=24.1%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ===\n\n"
        "The ACH floor_amount sweep shows TP rate remains above 97% up to $50,000 (FN=3 at current); "
        "raising to $500,000 cuts FP from 673 to 214 but drops TP rate to 51.5% with 64 missed SARs."
    )}
]})


# ==============================
# V9-D: Specific metric question → direct answer (ex251–ex256)
# ==============================

# ex251 — "precision of Velocity Single" → list_rules → answer DIRECTLY with Velocity Single number
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the precision of the Velocity Single rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex251", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex251"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Velocity Single has precision=15.5%, meaning 15.5% of its 478 alerts are genuine SARs (74 SAR, 404 FP)."
    )}
]})

# ex252 — "how many SARs does Elder Abuse generate?" → list_rules → direct answer for Elder Abuse
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many SARs does the Elder Abuse rule generate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex252", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex252"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Elder Abuse generates 188 SAR alerts out of 1,146 total alerts (FP=958, precision=16.4%)."
    )}
]})

# ex253 — "how many alerts does Detect Excessive generate?" → list_rules → direct answer
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many alerts does the Detect Excessive rule generate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex253", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex253"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Detect Excessive Transaction Activity generates 356 alerts (46 SAR, 310 FP, precision=12.9%)."
    )}
]})

# ex254 — "SAR count for ACH" → list_rules → direct answer for ACH
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR count for the Activity Deviation ACH rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex254", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex254"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Activity Deviation (ACH) has 138 confirmed SARs out of 907 total alerts (FP=769, precision=15.2%)."
    )}
]})

# ex255 — "show all AML rules" → list_rules (not policy)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex255", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex255"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There are 11 AML rules in the system; Elder Abuse generates the most alerts (1,146) while Structuring, CTR Client, "
        "and the Burst rules each generate 5 or fewer."
    )}
]})

# ex256 — "list rules" → list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "List all the AML rules in the system"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex256", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex256"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "The system monitors 11 AML rules across structuring, velocity, deviation, and other typologies; "
        "the three highest-alert rules are Elder Abuse (1,146), Activity Deviation ACH (907), and Velocity Single (478)."
    )}
]})


# ==============================
# V9-F: Threshold value as question → correct sweep_param name (ex257–ex262)
# ==============================

# ex257 — "what does structuring flag at threshold 8000?" → sweep_param="daily_floor", note 8000 is above sweep max
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What does the structuring rule flag at threshold 8000?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex257", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)", "sweep_param": "daily_floor"})]},
    {"role": "tool", "content": pc_struct_in, "tool_call_id": "call_ex257"},
    {"role": "assistant", "content": (
        pc_struct_in + "\n\n"
        "A daily_floor of $8,000 is above the sweep maximum of $6,500 — the sweep data does not extend to that value, "
        "but at the highest tested value ($6,500), both SAR customers are still caught (TP=2, FP=0)."
    )}
]})

# ex258 — "ACH rule at z=8" → sweep_param="z_threshold" (name, not value)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What does the ACH rule flag at z_threshold 8?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex258", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_ex258"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "At z_threshold=8.0 (between the current 5.0 and the sweep max of 10.0), the sweep table shows TP rate continuing to decline below 54.5%; "
        "at z=10.0, TP=30 (22.7%) with FP=75, so z=8 falls in between those values."
    )}
]})

# ex259 — "Elder Abuse at z=5" → sweep_param="z_threshold"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the Elder Abuse performance at z_threshold 5?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex259", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ea_z, "tool_call_id": "call_ex259"},
    {"role": "assistant", "content": (
        pc_ea_z + "\n\n"
        "At z_threshold=5.0 (the 90% TP rate boundary per the PRE-COMPUTED results), Elder Abuse catches 170 of 188 SARs (TP rate=90.4%) "
        "with FP=644 and precision=20.9%."
    )}
]})

# ex260 — "Velocity Single at pair_total 40000" → sweep_param="pair_total"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What does Velocity Single flag at pair_total 40000?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex260", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_vs_pt, "tool_call_id": "call_ex260"},
    {"role": "assistant", "content": (
        pc_vs_pt + "\n\n"
        "The PRE-COMPUTED sweep does not list pair_total=40,000 explicitly, but the highest value shown ($64,000) gives TP=38 (51.4%); "
        "at $40,000, results fall between the current $20,000 (TP=74, FP=404) and $64,000 (TP=38, FP=186)."
    )}
]})

# ex261 — "Detect Excessive at floor 25000" → sweep_param="floor_amount"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What does the Detect Excessive rule catch at a floor of 25000?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex261", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_det_exc, "tool_call_id": "call_ex261"},
    {"role": "assistant", "content": (
        pc_det_exc + "\n\n"
        "floor_amount=25,000 falls between the current $10,000 (TP=38, FP=248) and the highest value $40,000 (TP=18, FP=94); "
        "the PRE-COMPUTED sweep table (shown in chart) contains the exact row for $25,000."
    )}
]})

# ex262 — "Check rule at z 4" → sweep_param="z_threshold"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What happens to the Check rule at z_threshold 4?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex262", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_check_z, "tool_call_id": "call_ex262"},
    {"role": "assistant", "content": (
        pc_check_z + "\n\n"
        "z_threshold=4.0 falls between the current 2.0 (TP=36, FP=73) and the highest tested 8.0 (TP=12, FP=18); "
        "the detailed sweep table in the chart contains the exact TP/FP values at z=4.0."
    )}
]})


# ==============================
# V9-G: Segment-level SAR filing rate → sar_backtest or segment_stats, NOT list_rules (ex263–ex268)
# ==============================

# ex263 — "SAR filing rate for Individual" → sar_backtest (Individual), NOT list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Individual customers?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex263", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv, "tool_call_id": "call_ex263"},
    {"role": "assistant", "content": (
        pc_sar_indiv + "\n\n"
        "Among 4,668 Individual customers, 684 are confirmed SARs; the current threshold catches 612 of them "
        "(TP rate=89.5%) with 1,847 false positives (precision=24.9%)."
    )}
]})

# ex264 — "SAR rate for Business" → sar_backtest (Business)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR rate for Business customers?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex264", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz, "tool_call_id": "call_ex264"},
    {"role": "assistant", "content": (
        pc_sar_biz + "\n\n"
        "Among 367 Business customers, 36 are confirmed SARs; the current threshold catches 32 of them "
        "(TP rate=88.9%) with 218 false positives (precision=12.8%)."
    )}
]})

# ex265 — "how many Individual customers are SARs" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many Individual customers are confirmed SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex265", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv, "tool_call_id": "call_ex265"},
    {"role": "assistant", "content": (
        pc_sar_indiv + "\n\n"
        "684 of 4,668 Individual customers are confirmed SARs; the current threshold alerts 612 of them (TP rate=89.5%, FN=72)."
    )}
]})

# ex266 — "what proportion of Business are SARs" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What proportion of Business customers file SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex266", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz, "tool_call_id": "call_ex266"},
    {"role": "assistant", "content": (
        pc_sar_biz + "\n\n"
        "36 of 367 Business customers are confirmed SARs; the current threshold catches 32 of them (TP rate=88.9%, precision=12.8%)."
    )}
]})

# ex267 — "SAR detection rate for Individual" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show the SAR detection rate for Individual customers"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex267", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv, "tool_call_id": "call_ex267"},
    {"role": "assistant", "content": (
        pc_sar_indiv + "\n\n"
        "The Individual SAR detection rate is 89.5% — 612 of 684 confirmed Individual SARs are caught at the current threshold, "
        "with 72 missed (FN=72)."
    )}
]})

# ex268 — "SAR filing rate for Business" → sar_backtest (not list_rules)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the SAR filing rate for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex268", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz, "tool_call_id": "call_ex268"},
    {"role": "assistant", "content": (
        pc_sar_biz + "\n\n"
        "Business SAR detection rate is 88.9% — 32 of 36 confirmed Business SARs are alerted at the current threshold, "
        "with 218 false positives (precision=12.8%)."
    )}
]})


# ==============================
# V9-H: "Highest FP rate" = precision=0%, NOT highest FP count (ex269–ex273)
# ==============================

# ex269 — "Which rule has highest FP rate?" → rules with precision=0%, not highest FP count
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule has the highest FP rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex269", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex269"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer each have a 100% FP rate "
        "(precision=0.0%) — every alert generated by these three rules is a false positive with zero SARs detected."
    )}
]})

# ex270 — "which rules generate only false positives?" → precision=0% rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules generate only false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex270", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex270"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity (3 FP), Burst in Beneficiary Activity (5 FP), and Risky International Transfer (2 FP) "
        "have precision=0.0% — no SARs have been detected by any of these three rules."
    )}
]})

# ex271 — "which rule has worst precision?" → precision=0% rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule has the worst precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex271", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex271"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer have the worst precision "
        "at 0.0% — zero SARs detected across all their combined 10 alerts."
    )}
]})

# ex272 — "which rule flags the most non-SARs proportionally?" → 100% FP rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule flags the most non-SARs proportionally?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex272", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex272"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer each flag non-SARs 100% of the time "
        "(precision=0.0%) — all their alerts are false positives."
    )}
]})

# ex273 — "rank rules by FP rate" → correct ordering from highest to lowest FP rate
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Rank the AML rules from highest to lowest FP rate"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex273", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex273"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "By FP rate (highest first): Burst Originator, Burst Beneficiary, and Risky International Transfer (all 100% FP), "
        "then Detect Excessive (87.1%), Activity Deviation ACH (84.8%), Velocity Single (84.5%), Elder Abuse (83.6%), "
        "Activity Deviation Check (75.9%) — Structuring and CTR Client have 0% FP rate."
    )}
]})


# ==============================
# V7-I: No fabricated numbers in sweep insight (ex274–ex281)
# ==============================

# ex274 — ACH z_threshold sweep: insight must not invent floor amounts or fractions
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show z_threshold sweep for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex274", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_ex274"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "Raising z_threshold from 5.0 to 10.0 reduces FP from 274 to 75 but drops TP rate from 54.5% to 22.7%, "
        "missing 102 SARs instead of the current 60."
    )}
]})

# ex275 — 2D Velocity Single: no "exactly halving" or invented percentage reductions
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a 2D threshold sweep for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex275", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc_2d_vs, "tool_call_id": "call_ex275"},
    {"role": "assistant", "content": (
        pc_2d_vs + "\n\n"
        "The best FP reduction cell (pair_total=40,000, ratio_tolerance=7.5%) reduces FP from 404 to 273 "
        "and lowers TP rate from 100.0% to 70.3% with 22 missed SARs."
    )}
]})

# ex276 — 2D ACH: no invented cross-cell trends or heatmap pattern descriptions
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex276", "rule_2d_sweep", {"risk_factor": "Activity Deviation (ACH)"})]},
    {"role": "tool", "content": pc_2d_ach, "tool_call_id": "call_ex276"},
    {"role": "assistant", "content": (
        pc_2d_ach + "\n\n"
        "At the current condition (floor_amount=50,000, z_threshold=5), TP rate=54.5% with FP=274; "
        "the best FP reduction cell (floor_amount=100,000, z_threshold=8) cuts FP to 68 but lowers TP rate to 31.1%."
    )}
]})

# ex277 — Elder Abuse floor sweep: no invented z values or fractions
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show floor_amount sweep for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex277", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Sweep parameter: floor_amount - Minimum 14-day outgoing to trigger (currently $5,000)\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
        "At the lowest value (500): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "At current condition (5,000): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
        "To keep TP rate >=90%: floor_amount <= 9,000 => TP=170, FP=796, FN=18, TN=162 (precision=17.6%).\n"
        "At the highest value (25,000): TP=120, FP=482, FN=68, TN=476 (TP rate=63.8%, precision=19.9%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ==="
    ), "tool_call_id": "call_ex277"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Sweep parameter: floor_amount - Minimum 14-day outgoing to trigger (currently $5,000)\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
        "At the lowest value (500): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "At current condition (5,000): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
        "To keep TP rate >=90%: floor_amount <= 9,000 => TP=170, FP=796, FN=18, TN=162 (precision=17.6%).\n"
        "At the highest value (25,000): TP=120, FP=482, FN=68, TN=476 (TP rate=63.8%, precision=19.9%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ===\n\n"
        "Raising floor_amount from $5,000 to $9,000 keeps TP rate above 90% while reducing FP from 921 to 796 — "
        "the current $5,000 setting catches all but 1 SAR (TP rate=99.5%)."
    )}
]})

# ex278 — Detect Excessive: no invented parameters or derived numbers
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show floor_amount sweep for Detect Excessive"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex278", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_det_exc, "tool_call_id": "call_ex278"},
    {"role": "assistant", "content": (
        pc_det_exc + "\n\n"
        "At the current $10,000 floor, Detect Excessive catches 82.6% of SARs with 248 false positives (precision=13.3%); "
        "raising to $40,000 reduces FP by 154 but drops TP rate to 39.1% with 28 missed SARs."
    )}
]})

# ex279 — Check z sweep: no invented ratios or floor values
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show z_threshold sweep for Activity Deviation Check"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex279", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_check_z, "tool_call_id": "call_ex279"},
    {"role": "assistant", "content": (
        pc_check_z + "\n\n"
        "At the current z=2.0, Activity Deviation (Check) catches 47.4% of SARs with 73 false positives; "
        "raising to z=8.0 further reduces FP to 18 but at the cost of catching only 15.8% of SARs (FN=64)."
    )}
]})

# ex280 — Velocity Single pair_total: no invented % reductions
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show pair_total sweep for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex280", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_vs_pt, "tool_call_id": "call_ex280"},
    {"role": "assistant", "content": (
        pc_vs_pt + "\n\n"
        "At the current $20,000 pair_total, Velocity Single catches all 74 SARs (TP rate=100.0%, FN=0); "
        "raising to $64,000 reduces FP from 404 to 186 but misses 36 SARs (TP rate=51.4%)."
    )}
]})

# ex281 — 2D Elder Abuse: no invented heatmap trends
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex281", "rule_2d_sweep", {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Axis 1 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
        "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)\n"
        "Grid: 9 x 7 = 63 combinations\n"
        "SAR pool: 188 | Non-SAR pool: 958\n"
        "At current condition (z_threshold=3, age_threshold=60): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%).\n"
        "Best FP reduction (TP rate >=50%): z_threshold=7, age_threshold=65 => TP=102, FP=298, FN=86, TN=660, TP rate=54.3%, precision=25.5%.\n"
        "(Heatmap shown in the chart below.)\n"
        "=== END 2D SWEEP ==="
    ), "tool_call_id": "call_ex281"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Axis 1 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
        "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)\n"
        "Grid: 9 x 7 = 63 combinations\n"
        "SAR pool: 188 | Non-SAR pool: 958\n"
        "At current condition (z_threshold=3, age_threshold=60): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%).\n"
        "Best FP reduction (TP rate >=50%): z_threshold=7, age_threshold=65 => TP=102, FP=298, FN=86, TN=660, TP rate=54.3%, precision=25.5%.\n"
        "(Heatmap shown in the chart below.)\n"
        "=== END 2D SWEEP ===\n\n"
        "The best FP reduction cell (z_threshold=7, age_threshold=65) cuts FP from 921 to 298 but drops TP rate from 99.5% to 54.3%, "
        "missing 86 Elder Abuse SARs."
    )}
]})


# ==============================
# Gap 29: "High-risk segment" → redirect to threshold with rule or segment clarification (ex282–ex284)
# ==============================

# ex282 — "SAR backtest for the high-risk segment" → ask for segment or rule clarification
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me a SAR backtest for the high-risk segment"},
    {"role": "assistant", "content": (
        "There is no 'high-risk segment' as a parameter for SAR backtest. "
        "Please specify either:\n"
        "- A customer segment: Business or Individual (for segment-level SAR backtest)\n"
        "- A rule name such as 'Activity Deviation (ACH)' or 'Elder Abuse' (for rule-level SAR backtest)"
    )}
]})

# ex283 — "SAR backtest for VIP customers" → redirect
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for VIP customers"},
    {"role": "assistant", "content": (
        "VIP customers is not a recognized segment. "
        "The valid segments are Business and Individual. "
        "Please specify segment=Business or segment=Individual, or name a specific AML rule for a rule-level SAR backtest."
    )}
]})

# ex284 — "backtest for suspicious customers" → redirect
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR backtest for suspicious customers"},
    {"role": "assistant", "content": (
        "'Suspicious customers' is not a segment parameter. "
        "For a segment-level SAR backtest, specify segment=Business or segment=Individual. "
        "For a rule-level SAR backtest, specify a rule name such as 'Velocity Single' or 'Elder Abuse'."
    )}
]})


# ==============================
# V9-Orch-C: "layering rule" / "show all rules" → list_rules tool (ex285–ex288)
# ==============================

# ex285 — "What transactions are flagged by the layering rule?" → list_rules (not policy)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What transactions are flagged by the layering rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex285", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex285"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule named 'layering' — the closest rules covering layering typologies are "
        "Velocity Single (rapid in/out pairs), Activity Deviation (ACH and Check), and Structuring (Incoming/Outgoing Cash), "
        "all shown in the table above."
    )}
]})

# ex286 — "Which rule covers layering?" → list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule covers layering?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex286", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex286"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There is no rule explicitly named 'layering' — Velocity Single and Activity Deviation (ACH/Check) "
        "are the rules most associated with layering typologies based on their sweep parameters."
    )}
]})

# ex287 — "Show me all AML rules" → list_rules (not policy)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex287", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex287"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "The system has 11 AML rules; Structuring, CTR Client, Burst, and Risky International Transfer each generate "
        "5 or fewer alerts, while Elder Abuse and Activity Deviation ACH are the highest-volume rules."
    )}
]})

# ex288 — "What rules are in the system?" → list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What AML rules are in the system?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex288", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex288"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "There are 11 AML rules covering structuring, velocity deviation, elder abuse, excessive activity, "
        "CTR reporting, burst patterns, and risky international transfers."
    )}
]})


# ==============================
# Write output
# ==============================
out_path = r"C:\Users\Aaditya\PycharmProjects\framlagents-ftllm\finetune\data\framl_train_failures_v9.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} examples to {out_path}")

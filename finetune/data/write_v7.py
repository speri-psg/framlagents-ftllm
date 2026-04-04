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
    "- Crossover: the threshold where FP and FN counts are closest \u2014 the optimal operating point.\n"
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
    "15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes \u2014 call rule_sar_backtest with risk_factor and optionally sweep_param.\n"
    "16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview \u2014 call list_rules.\n"
    "17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR \u2014 call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.\n"
    "18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. \"hover to see\", \"right-click to select\", \"click the cell\"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart."
)

POLICY_SYSTEM = (
    "You are a FRAML policy specialist. Answer AML compliance questions using the knowledge base. "
    "After a KB miss, provide conceptual guidance only \u2014 do NOT cite specific statute numbers, FinCEN document codes, "
    "OCC circulars, named authors, or any named regulatory source. Zero citations after a KB miss."
)

CLUSTER_SYSTEM = (
    "You are a FRAML smart segmentation specialist. You identify natural customer behavioral segments using "
    "unsupervised K-Means clustering and explain their AML risk profiles. IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES \u2014 follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) \u2014 call ss_cluster_analysis.\n"
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

# ==============================
# GAP V7-I: 2D sweep (ex131-ex135)
# ==============================

pc131 = (
    "=== PRE-COMPUTED 2D SWEEP RESULT (copy verbatim) ===\n"
    "Rule: Velocity Single | Grid: 8x9=72 combinations\n"
    "Current params: pair_total=20000 ratio_tolerance=7.5% => TP=74 FP=404 FN=0 TN=235\n"
    "Best FP reduction cell: pair_total=40000 ratio_tolerance=7.5% => TP=52 FP=273 FN=22 TN=157 TP rate=70.3% precision=16.0%\n"
    "Heatmap is displayed in the chart panel.\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me a 2D grid for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex131", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc131, "tool_call_id": "call_ex131"},
    {"role": "assistant", "content": (
        pc131 + "\n\n"
        "Raising pair_total to $40,000 while holding ratio_tolerance at 7.5% cuts FP from 404 to 273 and improves precision to 16.0%, "
        "at the cost of introducing 22 missed SARs (FN=22)."
    )}
]})

pc132 = (
    "=== PRE-COMPUTED 2D SWEEP RESULT (copy verbatim) ===\n"
    "Rule: Elder Abuse | Grid: 9x11=99 combinations\n"
    "Current params: floor_amount=5000 z_threshold=3 => TP=138 FP=624 FN=46 TN=481\n"
    "Best FP reduction cell: floor_amount=8000 z_threshold=7 => TP=92 FP=308 FN=92 TN=557 TP rate=50.0% precision=23.0%\n"
    "Heatmap is displayed in the chart panel.\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me a 2D sweep for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex132", "rule_2d_sweep", {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "content": pc132, "tool_call_id": "call_ex132"},
    {"role": "assistant", "content": (
        pc132 + "\n\n"
        "The best FP reduction cell at floor_amount=8000 and z_threshold=7 cuts FP from 624 to 308 and improves precision to 23.0%, "
        "but raises FN from 46 to 92, meaning half of Elder Abuse SARs would be missed at that parameter combination."
    )}
]})

pc133 = (
    "=== PRE-COMPUTED 2D SWEEP RESULT (copy verbatim) ===\n"
    "Rule: Activity Deviation (ACH) | Grid: 9x11=99 combinations\n"
    "Current params: floor_amount=50000 z_threshold=5 => TP=72 FP=274 FN=60 TN=533\n"
    "Best FP reduction cell: floor_amount=90000 z_threshold=8 => TP=42 FP=98 FN=90 TN=597 TP rate=31.8% precision=30.0%\n"
    "Heatmap is displayed in the chart panel.\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show 2D grid for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex133", "rule_2d_sweep", {"risk_factor": "Activity Deviation (ACH)"})]},
    {"role": "tool", "content": pc133, "tool_call_id": "call_ex133"},
    {"role": "assistant", "content": (
        pc133 + "\n\n"
        "The best FP reduction cell at floor_amount=90000 and z_threshold=8 improves precision to 30.0% but TP rate falls to 31.8%, "
        "meaning the rule would miss the majority of ACH-related SARs at that combination."
    )}
]})

pc134 = (
    "=== PRE-COMPUTED 2D SWEEP RESULT (copy verbatim) ===\n"
    "Rule: Detect Excessive Transaction Activity | Grid: 9x5=45 combinations\n"
    "Current params: floor_amount=10000 time_window=5 => TP=46 FP=310 FN=0 TN=176\n"
    "Best FP reduction cell: floor_amount=18000 time_window=5 => TP=24 FP=156 FN=22 TN=154 TP rate=52.2% precision=13.3%\n"
    "Heatmap is displayed in the chart panel.\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "2D sweep for Detect Excessive Transaction Activity"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex134", "rule_2d_sweep", {"risk_factor": "Detect Excessive Transaction Activity"})]},
    {"role": "tool", "content": pc134, "tool_call_id": "call_ex134"},
    {"role": "assistant", "content": (
        pc134 + "\n\n"
        "Raising the floor_amount to $18,000 cuts FP by more than half (310 to 156) while keeping time_window at 5, "
        "but reduces the TP rate to 52.2% \u2014 meaning nearly half of excessive transaction SARs would be missed."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Give me the 2D heatmap for Velocity Single with pair_total and ratio_tolerance"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex135", "rule_2d_sweep", {"risk_factor": "Velocity Single", "sweep_param_1": "pair_total", "sweep_param_2": "ratio_tolerance"})]},
    {"role": "tool", "content": pc131, "tool_call_id": "call_ex135"},
    {"role": "assistant", "content": (
        pc131 + "\n\n"
        "The pair_total=40000 and ratio_tolerance=7.5% combination achieves the best FP reduction in the 8x9 grid, "
        "improving precision to 16.0% while a TP rate of 70.3% reflects 22 additional missed SARs versus the current configuration."
    )}
]})

# ==============================
# GAP V7-M: FN direction inverted (ex136-ex139)
# ==============================

pc136 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (ACH) | sweep_param: z_threshold\n"
    "Current: z_threshold=5 => TP=72 FP=274 FN=60 precision=20.8%\n"
    "At highest value z_threshold=10: TP=30 FP=75 FN=102 precision=28.6%\n"
    "To keep TP rate>=90%: z_threshold<=1 => TP=120 FP=637 FN=12\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show z_threshold sweep for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex136", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc136, "tool_call_id": "call_ex136"},
    {"role": "assistant", "content": (
        pc136 + "\n\n"
        "Raising z_threshold reduces FP but INCREASES FN \u2014 at z=10, FN=102 vs FN=60 at current z=5, "
        "meaning 42 additional SARs would be missed as the threshold becomes more restrictive."
    )}
]})

pc137 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Velocity Single | sweep_param: pair_total\n"
    "Current: pair_total=20000 => TP=74 FP=404 FN=0 precision=15.5%\n"
    "At highest value pair_total=40000: TP=52 FP=273 FN=22 precision=16.0%\n"
    "To keep TP rate>=90%: pair_total<=20000 => TP=74 FP=404 FN=0\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show floor_amount sweep for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex137", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc137, "tool_call_id": "call_ex137"},
    {"role": "assistant", "content": (
        pc137 + "\n\n"
        "Raising pair_total increases FN from 0 to 22 at pair_total=40000 \u2014 the rule currently catches all SARs at the current threshold, "
        "and any increase introduces missed SARs as the minimum qualifying transaction amount becomes more restrictive."
    )}
]})

pc138 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Elder Abuse | sweep_param: floor_amount\n"
    "Current: floor_amount=5000 => TP=187 FP=921 FN=1 precision=16.9%\n"
    "At highest value floor_amount=10000: TP=140 FP=700 FN=48 precision=16.7%\n"
    "To keep TP rate>=90%: floor_amount<=6000 => TP=170 FP=834 FN=18\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the floor_amount sweep for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex138", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc138, "tool_call_id": "call_ex138"},
    {"role": "assistant", "content": (
        pc138 + "\n\n"
        "Raising the floor to $10,000 cuts FP from 921 to 700 but INCREASES FN from 1 to 48 \u2014 "
        "the higher floor excludes lower-dollar elder abuse transactions that are genuine SARs, "
        "a pattern consistent with financial exploitation targeting smaller, more frequent withdrawals."
    )}
]})

pc139 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (Check) | sweep_param: z_threshold\n"
    "Current: z_threshold=2 => TP=36 FP=73 FN=29\n"
    "At z_threshold=5: TP=20 FP=40 FN=45\n"
    "At z_threshold=10: TP=8 FP=4 FN=57\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What happens to missed SARs if I raise the z-threshold for Activity Deviation Check?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex139", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc139, "tool_call_id": "call_ex139"},
    {"role": "assistant", "content": (
        pc139 + "\n\n"
        "Raising z_threshold increases FN (missed SARs) \u2014 at z=10, 57 SARs are missed vs 29 at current z=2, "
        "while FP drops from 73 to 4, demonstrating the classic sensitivity-specificity trade-off for deviation-based rules."
    )}
]})

# ==============================
# GAP V7-E: Wrong direction for floor/z (ex140-ex142)
# ==============================

pc140 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Elder Abuse | sweep_param: floor_amount\n"
    "Current: floor_amount=5000 => TP=187 FP=921 FN=1 precision=16.9%\n"
    "To keep TP rate>=90%: floor_amount<=7000 => TP=170 FP=796 FN=18 precision=17.6%\n"
    "At highest value floor_amount=10000: TP=140 FP=700 FN=48 precision=16.7%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse floor_amount sweep"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex140", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc140, "tool_call_id": "call_ex140"},
    {"role": "assistant", "content": (
        pc140 + "\n\n"
        "Raising the floor_amount from $5K reduces false positives \u2014 to maintain a 90% catch rate the floor can be raised to $7,000, "
        "cutting FP from 921 to 796 (precision 17.6%)."
    )}
]})

pc141 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (ACH) | sweep_param: z_threshold\n"
    "Current: z_threshold=5 => TP=72 FP=274 FN=60 precision=20.8%\n"
    "To keep TP rate>=90%: z_threshold<=1 => TP=120 FP=637 FN=12\n"
    "To keep TP rate>=50%: z_threshold<=5 => TP=72 FP=274 FN=60\n"
    "At z_threshold=10: TP=30 FP=75 FN=102 precision=28.6%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How should I tune the z_threshold for Activity Deviation ACH to reduce investigator workload?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex141", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc141, "tool_call_id": "call_ex141"},
    {"role": "assistant", "content": (
        pc141 + "\n\n"
        "Raising z_threshold above 5 reduces FP below 274 but increases missed SARs \u2014 "
        "to keep TP rate above 50% the threshold must stay at or below z=5."
    )}
]})

pc142 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Velocity Single | sweep_param: pair_total\n"
    "Current: pair_total=20000 => TP=74 FP=404 FN=0 precision=15.5%\n"
    "At pair_total=40000: TP=52 FP=273 FN=22 precision=16.0%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the optimal floor_amount for Velocity Single to improve precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex142", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc142, "tool_call_id": "call_ex142"},
    {"role": "assistant", "content": (
        pc142 + "\n\n"
        "Raising pair_total from $20K to $40K reduces FP from 404 to 273 (precision 16.0%) while catching 52 of 74 SARs (FN increases from 0 to 22)."
    )}
]})

# ==============================
# GAP V7-A: Dollar sign on z-score (ex143-ex145)
# ==============================

pc143 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (Check) | sweep_param: z_threshold\n"
    "Current: z_threshold=2 => TP=36 FP=73 FN=29 precision=33.0%\n"
    "To keep TP rate>=90%: z_threshold<=0 => TP=52 FP=111 FN=13\n"
    "To keep TP rate>=50%: z_threshold<=2 => TP=36 FP=73 FN=29\n"
    "At z_threshold=10: TP=8 FP=4 FN=57 precision=66.7%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show z_threshold sweep for Activity Deviation Check"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex143", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc143, "tool_call_id": "call_ex143"},
    {"role": "assistant", "content": (
        pc143 + "\n\n"
        "To maintain a 90% catch rate on Activity Deviation Check, z_threshold must stay at or below 0, "
        "catching 52 of 65 SARs with 111 false positives \u2014 z_threshold is a dimensionless standard deviation multiplier, not a dollar value."
    )}
]})

pc144 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Elder Abuse | sweep_param: z_threshold\n"
    "To keep TP rate>=80%: z_threshold<=4 => TP=155 FP=580 FN=33 precision=21.1%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What z threshold keeps 80% SAR catch for Elder Abuse?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex144", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc144, "tool_call_id": "call_ex144"},
    {"role": "assistant", "content": (
        pc144 + "\n\n"
        "To keep an 80% SAR catch rate on Elder Abuse, z_threshold must stay at or below 4, "
        "catching 155 of 188 SARs with 580 false positives (precision 21.1%)."
    )}
]})

pc145 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (ACH) | sweep_param: z_threshold\n"
    "Current: z_threshold=5 => TP=72 FP=274 FN=60 precision=20.8%\n"
    "To keep TP rate>=50%: z_threshold<=5 => TP=72 FP=274 FN=60\n"
    "At z_threshold=10: TP=30 FP=75 FN=102 precision=28.6%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Activity Deviation ACH z sweep"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex145", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc145, "tool_call_id": "call_ex145"},
    {"role": "assistant", "content": (
        pc145 + "\n\n"
        "At z_threshold=5 the rule catches 72 SARs with 274 false positives (precision 20.8%); "
        "raising to z=10 reduces FP to 75 but increases FN to 102 \u2014 "
        "z_threshold is a standard deviation multiplier with no dollar sign."
    )}
]})

# ==============================
# GAP V7-B: FN/FP confusion (ex146-ex147)
# ==============================

pc146 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (Check) | sweep_param: z_threshold\n"
    "To keep TP rate>=90%: z_threshold<=0 => TP=52 FP=111 FN=13 precision=31.9%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Activity Deviation Check 1D sweep with z_threshold"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex146", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc146, "tool_call_id": "call_ex146"},
    {"role": "assistant", "content": (
        pc146 + "\n\n"
        "To maintain a 90% catch rate on Activity Deviation Check, the threshold must stay at or below z=0, "
        "catching 52 of 65 SARs with 111 false positives and 13 missed SARs (precision 31.9%)."
    )}
]})

pc147 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (Check) | sweep_param: z_threshold\n"
    "At z_threshold=2: TP=36 FP=73 FN=29 precision=33.0%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the FP count at z=2 for Check?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex147", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc147, "tool_call_id": "call_ex147"},
    {"role": "assistant", "content": (
        pc147 + "\n\n"
        "At z_threshold=2, Activity Deviation Check generates 73 false positives (investigated but not SAR) "
        "while missing 29 SARs (FN=29), with 36 correctly caught (precision 33.0%)."
    )}
]})

# ==============================
# GAP V7-C: Precision/TP rate confusion (ex148-ex150)
# ==============================

pc148 = (
    "=== PRE-COMPUTED LIST_RULES RESULT (copy verbatim) ===\n"
    "Elder Abuse: total_alerts=1146 SAR=188 FP=958 precision=16.4% TP_rate=100.0%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How good is Elder Abuse at catching SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex148", "list_rules", {})]},
    {"role": "tool", "content": pc148, "tool_call_id": "call_ex148"},
    {"role": "assistant", "content": (
        pc148 + "\n\n"
        "Elder Abuse catches 188 SARs with 958 false positives (precision 16.4%) \u2014 meaning only 1 in 6 alerts is a genuine SAR, "
        "though the rule currently alerts all 188 identified SARs in the alerted population."
    )}
]})

pc149 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Elder Abuse | sweep_param: floor_amount\n"
    "At floor_amount=5000 (current): TP=187 FP=921 FN=1 TP rate=99.5% precision=16.9%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show floor_amount sweep for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex149", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc149, "tool_call_id": "call_ex149"},
    {"role": "assistant", "content": (
        pc149 + "\n\n"
        "The current $5K floor catches 99.5% of Elder Abuse SARs (TP rate) but only 16.9% of alerts are genuine SARs (precision) \u2014 "
        "the rule generates many false positives at this threshold."
    )}
]})

pc150 = (
    "=== PRE-COMPUTED LIST_RULES RESULT (copy verbatim) ===\n"
    "Activity Deviation (ACH): total_alerts=907 SAR=138 FP=769 precision=15.2%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Is the precision good for Activity Deviation ACH?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex150", "list_rules", {})]},
    {"role": "tool", "content": pc150, "tool_call_id": "call_ex150"},
    {"role": "assistant", "content": (
        pc150 + "\n\n"
        "Activity Deviation ACH has a precision of 15.2% \u2014 meaning 769 of 907 alerts are false positives; "
        "raising the z_threshold or floor_amount can improve precision at the cost of catching fewer SARs."
    )}
]})

# ==============================
# GAP V7-D: list_rules wrong counts and UI hallucination (ex151-ex153)
# ==============================

pc_list_rules = (
    "=== PRE-COMPUTED LIST_RULES RESULT (copy verbatim) ===\n"
    "1. Activity Deviation (ACH): total_alerts=907 SAR=138 FP=769 precision=15.2%\n"
    "2. Activity Deviation (Check): total_alerts=229 SAR=65 FP=164 precision=28.4%\n"
    "3. Elder Abuse: total_alerts=1146 SAR=188 FP=958 precision=16.4%\n"
    "4. Velocity Single: total_alerts=478 SAR=74 FP=404 precision=15.5%\n"
    "5. Detect Excessive Transaction Activity: total_alerts=356 SAR=46 FP=310 precision=12.9%\n"
    "6. Structuring Incoming Cash: total_alerts=12 SAR=12 FP=0 precision=100.0%\n"
    "7. Structuring Outgoing Cash: total_alerts=8 SAR=8 FP=0 precision=100.0%\n"
    "8. CTR Client: total_alerts=5 SAR=5 FP=0 precision=100.0%\n"
    "9. Burst in Originator Activity: total_alerts=3 SAR=0 FP=3 precision=0.0%\n"
    "10. Burst in Beneficiary Activity: total_alerts=5 SAR=0 FP=5 precision=0.0%\n"
    "11. Risky International Transfer: total_alerts=2 SAR=0 FP=2 precision=0.0%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me all AML rules and their SAR performance"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex151", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex151"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "11 AML rules are active; Structuring (Incoming/Outgoing Cash) and CTR Client have 100% precision with small populations, "
        "while Elder Abuse (16.4%), Activity Deviation ACH (15.2%), and Burst rules (0.0%) show the widest FP ranges."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What rules have the highest FP rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex152", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex152"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Burst in Originator Activity (precision=0.0%, 3 alerts), Burst in Beneficiary Activity (0.0%, 5 alerts), "
        "and Risky International Transfer (0.0%, 2 alerts) have the lowest precision \u2014 though all have fewer than 10 alerts; "
        "among high-volume rules, Detect Excessive Transaction Activity (12.9%, 356 alerts) has the lowest precision."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Give me a summary of all rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex153", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex153"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Across all 11 active rules, Structuring and CTR rules achieve 100% precision on small populations, "
        "high-volume behavioral rules (Elder Abuse, ACH, Velocity Single) operate in the 15\u201328% precision range, "
        "and the three Burst/Risky International rules currently produce no confirmed SARs."
    )}
]})

# ==============================
# GAP V7-G/H: Fabricated segment qualifier and editorial claims (ex154-ex156)
# ==============================

pc154 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (Check) | sweep_param: z_threshold\n"
    "Population: 229 customers\n"
    "Current: z_threshold=2 => TP=36 FP=73 FN=29 precision=33.0%\n"
    "To keep TP rate>=90%: z_threshold<=0 => TP=52 FP=111 FN=13\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show z_threshold sweep for Activity Deviation Check"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex154", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc154, "tool_call_id": "call_ex154"},
    {"role": "assistant", "content": (
        pc154 + "\n\n"
        "Across 229 customers, z_threshold=2 currently catches 36 SARs with 73 false positives; "
        "lowering to z=0 maintains a 90% catch rate while raising FP to 111."
    )}
]})

pc155 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Elder Abuse | sweep_param: floor_amount\n"
    "Current: floor_amount=5000 => TP=187 FP=921 FN=1 precision=16.9%\n"
    "To keep TP rate>=90%: floor_amount<=7000 => TP=170 FP=796 FN=18 precision=17.6%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse floor_amount sweep"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex155", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc155, "tool_call_id": "call_ex155"},
    {"role": "assistant", "content": (
        pc155 + "\n\n"
        "At the current $5K floor the rule generates 921 false positives (precision 16.9%); "
        "raising the floor to $7,000 cuts FP to 796 while maintaining a 90% SAR catch rate (precision 17.6%)."
    )}
]})

pc156 = (
    "=== PRE-COMPUTED RULE SAR BACKTEST (copy verbatim) ===\n"
    "Rule: Activity Deviation (ACH) | sweep_param: z_threshold\n"
    "Population: 827 customers\n"
    "Current: z_threshold=5 => TP=72 FP=274 FN=60 precision=20.8%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show ACH 1D sweep"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex156", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc156, "tool_call_id": "call_ex156"},
    {"role": "assistant", "content": (
        pc156 + "\n\n"
        "Across 827 customers, Activity Deviation ACH at z_threshold=5 generates 72 true positives and 274 false positives (precision 20.8%), "
        "with 60 missed SARs at the current setting."
    )}
]})

# ==============================
# GAP 24: Invalid parameters rejected (ex157-ex159)
# ==============================

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN tuning for Business with threshold_min=100 threshold_step=10"},
    {"role": "assistant", "content": (
        "threshold_min and threshold_step are not valid parameters for this tool. "
        "The only valid parameters are segment (Business or Individual) and threshold_column "
        "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
        "Please specify which threshold column you would like to analyze."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run FP analysis with threshold_max=50000 and step=5000"},
    {"role": "assistant", "content": (
        "threshold_max and step are not valid parameters. "
        "Please specify segment (Business or Individual) and threshold_column "
        "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY)."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Threshold tuning for Individual, min_threshold=500"},
    {"role": "assistant", "content": (
        "min_threshold is not a valid parameter. "
        "The only valid parameters are segment and threshold_column. "
        "To analyze Individual customers, please specify which column: AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY."
    )}
]})

# ==============================
# GAP 26/27: No citations after KB miss (ex160-ex162)
# ==============================

examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What does AML policy say about structuring detection thresholds?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex160", "query_knowledge_base", {"query": "structuring detection thresholds AML policy"})]},
    {"role": "tool", "content": "KB result: No relevant chunks found.", "tool_call_id": "call_ex160"},
    {"role": "assistant", "content": (
        "Structuring detection thresholds are typically calibrated around the $10,000 CTR reporting baseline, "
        "with institutions setting daily cash floor amounts (e.g. $3K\u2013$7K) and requiring multiple qualifying days "
        "within a rolling window to balance sensitivity against false alert volume."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the regulatory basis for SAR filing?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex161", "query_knowledge_base", {"query": "regulatory basis SAR filing requirements"})]},
    {"role": "tool", "content": (
        "KB result: BSA requires financial institutions to file SARs for transactions involving potential money laundering "
        "or other suspicious activity above applicable thresholds. Institutions must use a risk-based approach."
    ), "tool_call_id": "call_ex161"},
    {"role": "assistant", "content": (
        "The Bank Secrecy Act requires financial institutions to file a SAR when they detect transactions that cannot be "
        "explained as lawful and involve amounts above applicable thresholds, using a risk-based approach calibrated "
        "to their customer profile."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What guidance exists on elder financial exploitation monitoring?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex162", "query_knowledge_base", {"query": "elder financial exploitation monitoring guidance"})]},
    {"role": "tool", "content": "KB result: No relevant chunks found.", "tool_call_id": "call_ex162"},
    {"role": "assistant", "content": (
        "Elder financial exploitation monitoring typically focuses on detecting unusual third-party transaction patterns, "
        "sudden changes in beneficiary designations, large cash withdrawals inconsistent with prior behavior, "
        "and deviations from established spending baselines for senior customers \u2014 "
        "institutions generally apply behavioral deviation rules with lower floor amounts to capture the smaller, "
        "more frequent transactions common in this typology."
    )}
]})

# ==============================
# GAP 29: Cluster routing (ex163-ex164)
# ==============================

pc163 = (
    "=== PRE-COMPUTED CLUSTER ANALYSIS (copy verbatim) ===\n"
    "Customer type: Business | n_clusters=4\n"
    "Cluster 0: n=112 avg_trxns_week=3.2 avg_trxn_amt=8400 sar_rate=0.04 fp_count=87\n"
    "Cluster 1: n=89 avg_trxns_week=7.1 avg_trxn_amt=22000 sar_rate=0.18 fp_count=143\n"
    "Cluster 2: n=64 avg_trxns_week=12.4 avg_trxn_amt=51000 sar_rate=0.31 fp_count=198\n"
    "Cluster 3: n=102 avg_trxns_week=1.8 avg_trxn_amt=3200 sar_rate=0.01 fp_count=19\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Which Business cluster has the most false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex163", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc163, "tool_call_id": "call_ex163"},
    {"role": "assistant", "content": (
        pc163 + "\n\n"
        "Cluster 2 has the most false positives (198) among Business clusters, "
        "with the highest avg_trxns_week (12.4) and avg_trxn_amt ($51,000) and a SAR rate of 0.31."
    )}
]})

pc164 = (
    "=== PRE-COMPUTED CLUSTER ANALYSIS (copy verbatim) ===\n"
    "Customer type: Business | n_clusters=4\n"
    "Cluster 0: n=112 avg_trxns_week=3.2 avg_trxn_amt=8400 sar_rate=0.04 fp_count=87\n"
    "Cluster 1: n=89 avg_trxns_week=7.1 avg_trxn_amt=22000 sar_rate=0.18 fp_count=143\n"
    "Cluster 2: n=64 avg_trxns_week=12.4 avg_trxn_amt=51000 sar_rate=0.31 fp_count=198\n"
    "Cluster 3: n=102 avg_trxns_week=1.8 avg_trxn_amt=3200 sar_rate=0.01 fp_count=19\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show me the high-risk Business cluster"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex164", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc164, "tool_call_id": "call_ex164"},
    {"role": "assistant", "content": (
        pc164 + "\n\n"
        "Cluster 2 is the highest-risk Business cluster with a SAR rate of 0.31, avg_trxns_week=12.4, and avg_trxn_amt=$51,000 \u2014 "
        "it also generates the most false positives (198) among the four clusters.\n"
        "DISPLAY_CLUSTERS: 2"
    )}
]})

# ==============================
# GAP V7-N: SAR filing rate routing (ex165-ex166)
# ==============================

pc165 = (
    "=== PRE-COMPUTED SAR BACKTEST (copy verbatim) ===\n"
    "Segment: Individual | threshold_column: AVG_TRXN_AMT\n"
    "Total customers: 4668 | SAR customers: 684 | Non-SAR: 3984\n"
    "At current threshold: TP=612 FP=1847 FN=72 TN=2137 TP rate=89.5% precision=24.9%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Individual customers?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex165", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc165, "tool_call_id": "call_ex165"},
    {"role": "assistant", "content": (
        pc165 + "\n\n"
        "Among Individual customers, the current threshold catches 89.5% of SARs (TP rate=89.5%) with a precision of 24.9%, "
        "meaning roughly 1 in 4 alerted Individual customers is a genuine SAR."
    )}
]})

pc166 = (
    "=== PRE-COMPUTED SAR BACKTEST (copy verbatim) ===\n"
    "Segment: Business | threshold_column: TRXN_AMT_MONTHLY\n"
    "Total customers: 367 | SAR customers: 52 | Non-SAR: 315\n"
    "At current threshold: TP=48 FP=127 FN=4 TN=188 TP rate=92.3% precision=27.4%\n"
    "=== END PRE-COMPUTED ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me SAR detection rate for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex166", "sar_backtest", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc166, "tool_call_id": "call_ex166"},
    {"role": "assistant", "content": (
        pc166 + "\n\n"
        "Business customers show a TP rate of 92.3% at the current monthly amount threshold, "
        "catching 48 of 52 SARs with 127 false positives (precision 27.4%)."
    )}
]})

# ==============================
# GAP V7-J: "FN=0 everywhere" — must read FN from best cell (ex167-ex170)
# ==============================

pc_2d_vs = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Axis 1 (pair_total): Minimum combined in+out pair total to trigger (currently $20K)\n"
    "Axis 2 (ratio_tolerance): Max deviation of out/in ratio from 1.0 to trigger (currently 10% = 90-110%)\n"
    "Grid: 8 x 9 = 72 combinations\n"
    "SAR pool: 74\n"
    "Non-SAR pool: 404\n"
    "\n"
    "At current condition (pair_total=20,000, ratio_tolerance=10%): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=50%): pair_total=35,000, ratio_tolerance=10% => TP=42, FP=178, FN=32, TN=226, TP rate=56.8%, precision=19.1%.\n"
    "=== END 2D SWEEP ===\n"
    "(Heatmap shown in the chart below.)"
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Velocity Single and find the best FP reduction"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7j1", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc_2d_vs, "tool_call_id": "call_v7j1"},
    {"role": "assistant", "content": (
        pc_2d_vs + "\n\n"
        "The best FP reduction cell (pair_total=35,000, ratio_tolerance=10%) cuts FP from 404 to 178 and improves precision to 19.1%, "
        "but FN rises to 32 — 32 SARs that are caught at current settings would be missed."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many SARs are missed at the best FP reduction point for Velocity Single?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7j2", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc_2d_vs, "tool_call_id": "call_v7j2"},
    {"role": "assistant", "content": (
        pc_2d_vs + "\n\n"
        "At the best FP reduction cell (pair_total=35,000, ratio_tolerance=10%), FN=32 — meaning 32 SARs are missed "
        "compared to FN=0 at the current operational setting."
    )}
]})

pc_2d_ea = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Axis 1 (floor_amount): Minimum 14-day aggregated outgoing to trigger (currently $5K)\n"
    "Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
    "Grid: 9 x 11 = 99 combinations\n"
    "SAR pool: 188\n"
    "Non-SAR pool: 958\n"
    "\n"
    "At current condition (floor_amount=5,000, z_threshold=3): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%).\n"
    "Best FP reduction (TP rate >=50%): floor_amount=9,000, z_threshold=7 => TP=97, FP=345, FN=91, TN=613, TP rate=51.6%, precision=21.9%.\n"
    "=== END 2D SWEEP ===\n"
    "(Heatmap shown in the chart below.)"
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Elder Abuse 2D sweep — where is the best FP reduction?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7j3", "rule_2d_sweep", {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "content": pc_2d_ea, "tool_call_id": "call_v7j3"},
    {"role": "assistant", "content": (
        pc_2d_ea + "\n\n"
        "The best FP reduction cell (floor_amount=9,000, z_threshold=7) cuts FP from 921 to 345 and improves precision to 21.9%, "
        "but FN rises to 91 — nearly half of the 188 SARs in the population would be missed at that combination."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Optimize Elder Abuse 2D to minimize false positives"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7j4", "rule_2d_sweep", {"risk_factor": "Elder Abuse", "sweep_param_1": "floor_amount", "sweep_param_2": "z_threshold"})]},
    {"role": "tool", "content": pc_2d_ea, "tool_call_id": "call_v7j4"},
    {"role": "assistant", "content": (
        pc_2d_ea + "\n\n"
        "The best FP reduction point at floor_amount=9,000 and z_threshold=7 reduces FP to 345, "
        "but FN=91 means 91 of 188 Elder Abuse SARs would not be alerted at that combination."
    )}
]})

# ==============================
# GAP V7-L + V7-K: No fabricated crossover, no invented constraints (ex171-ex173)
# ==============================

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the crossover point for Velocity Single?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7l1", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc_2d_vs, "tool_call_id": "call_v7l1"},
    {"role": "assistant", "content": (
        pc_2d_vs + "\n\n"
        "The PRE-COMPUTED section shows two key reference points: the current setting (TP=74, FP=404, FN=0) "
        "and the best FP reduction cell (TP=42, FP=178, FN=32, precision=19.1%) — "
        "no crossover point where FP equals FN is identified in the data shown."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Can I only change one parameter at a time for Velocity Single?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7k1", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc_2d_vs, "tool_call_id": "call_v7k1"},
    {"role": "assistant", "content": (
        pc_2d_vs + "\n\n"
        "The 2D sweep shows 72 combinations across 8 pair_total values and 9 ratio_tolerance values — "
        "both parameters can be varied independently; the best FP reduction cell uses pair_total=35,000 and ratio_tolerance=10%."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse 2D sweep — what constraints apply to the parameters?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7k2", "rule_2d_sweep", {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "content": pc_2d_ea, "tool_call_id": "call_v7k2"},
    {"role": "assistant", "content": (
        pc_2d_ea + "\n\n"
        "The PRE-COMPUTED section covers a 9×11 grid of floor_amount and z_threshold combinations; "
        "no additional constraints beyond the data shown are identified."
    )}
]})

# ==============================
# GAP V7-O: ACH rule SAR analysis — call rule_sar_backtest, not sar_backtest (ex174-ex177)
# ==============================

pc_ach_floor = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: floor_amount - Minimum monthly ACH floor ($) (currently $50K)\n"
    "Current value: 50,000\n"
    "Labeled population: 142 customers (TP+FN pool=43 SAR, FP+TN pool=99 non-SAR, precision=30.3%)\n"
    "\n"
    "At the lowest value (10,000): TP=43, FP=99, FN=0, TN=0 (TP rate=100.0%, precision=30.3%).\n"
    "At current condition (50,000): TP=40, FP=63, FN=3, TN=36 (TP rate=93.0%, precision=38.8%).\n"
    "To keep TP rate >=90%: floor_amount <= 50,000 => TP=40, FP=63, FN=3, TN=36, precision=38.8%.\n"
    "To keep TP rate >=50%: floor_amount <= 90,000 => TP=22, FP=18, FN=21, TN=81, precision=55.0%.\n"
    "At the highest value (130,000): TP=5, FP=3, FN=38, TN=96, precision=62.5%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show the SAR backtest for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7o1", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)"})]},
    {"role": "tool", "content": pc_ach_floor, "tool_call_id": "call_v7o1"},
    {"role": "assistant", "content": (
        pc_ach_floor + "\n\n"
        "Activity Deviation ACH currently catches 40 of 43 SARs (TP rate=93.0%) at the $50K floor with 63 false positives (precision=38.8%); "
        "raising the floor to $90K maintains a 50% catch rate while cutting FP to 18."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How does the ACH rule perform at different floor amounts?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7o2", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach_floor, "tool_call_id": "call_v7o2"},
    {"role": "assistant", "content": (
        pc_ach_floor + "\n\n"
        "Raising the ACH floor_amount reduces FP but increases FN — at $130K the rule has only 3 false positives but misses 38 of 43 SARs (precision=62.5%)."
    )}
]})

pc_ach_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month ACH profile mean (currently 5)\n"
    "Current value: 5\n"
    "Labeled population: 142 customers (TP+FN pool=43 SAR, FP+TN pool=99 non-SAR, precision=30.3%)\n"
    "\n"
    "At the lowest value (0): TP=43, FP=99, FN=0, TN=0 (TP rate=100.0%, precision=30.3%).\n"
    "At current condition (5): TP=40, FP=63, FN=3, TN=36 (TP rate=93.0%, precision=38.8%).\n"
    "To keep TP rate >=90%: z_threshold <= 5 => TP=40, FP=63, FN=3, TN=36, precision=38.8%.\n"
    "To keep TP rate >=50%: z_threshold <= 8 => TP=22, FP=14, FN=21, TN=85, precision=61.1%.\n"
    "At the highest value (10): TP=11, FP=4, FN=32, TN=95, precision=73.3%.\n"
    "=== END RULE SWEEP ===\n"
    "(Detailed sweep table shown in the chart below.)"
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run a SAR sweep on Activity Deviation ACH using z_threshold"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7o3", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_v7o3"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "Raising z_threshold above 5 for Activity Deviation ACH reduces FP but increases FN — "
        "at z=10 the rule catches only 11 of 43 SARs with 4 false positives (FN=32, precision=73.3%)."
    )}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What threshold for ACH minimizes false positives while keeping SAR detection above 90%?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_v7o4", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc_ach_floor, "tool_call_id": "call_v7o4"},
    {"role": "assistant", "content": (
        pc_ach_floor + "\n\n"
        "To keep TP rate above 90% for Activity Deviation ACH, the floor_amount must stay at or below $50,000 (current setting), "
        "catching 40 of 43 SARs with 63 false positives (precision=38.8%)."
    )}
]})

# ==============================
# GAP 24: Invalid params — additional reinforcement (ex178-ex182)
# Still failing after V6 training — 5 more varied examples
# ==============================

REJECT_V7 = (
    "threshold_min, threshold_max, threshold_step, step, and min_threshold are not valid parameters. "
    "The only accepted parameters are:\n"
    "- segment: Business or Individual\n"
    "- threshold_column: AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY\n\n"
    "Please specify which segment and threshold column you would like to analyze."
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run tuning for Business from threshold_min=1000 to threshold_max=50000 with step=500"},
    {"role": "assistant", "content": REJECT_V7}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "threshold_min=100, threshold_step=10 for Individual"},
    {"role": "assistant", "content": REJECT_V7}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Set threshold range from 500 to 100000 for Business weekly transactions"},
    {"role": "assistant", "content": REJECT_V7}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run FP analysis for Business with min_threshold=2000 step=200"},
    {"role": "assistant", "content": REJECT_V7}
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Sweep threshold_min=0 threshold_max=10000 for Individual customers"},
    {"role": "assistant", "content": REJECT_V7}
]})

# Write the file
out_path = r"C:\Users\Aaditya\PycharmProjects\framlagents-ftllm\finetune\data\framl_train_failures_v7.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} examples to {out_path}")

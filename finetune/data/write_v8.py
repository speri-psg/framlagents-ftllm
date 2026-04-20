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

# ==============================
# GAP 24: Invalid params — 15 more examples (ex183–ex197)
# ==============================

REJECT = (
    "threshold_min, threshold_max, threshold_step, step, and min_threshold are NOT valid parameters. "
    "The only valid parameters are segment (Business or Individual) and threshold_column "
    "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
    "Please specify one of these instead."
)

REJECT_MIN = (
    "threshold_min is not a valid parameter. "
    "The only valid parameters are segment (Business or Individual) and threshold_column "
    "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY)."
)

REJECT_STEP = (
    "step is not a valid parameter. "
    "The only valid parameters are segment (Business or Individual) and threshold_column "
    "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY)."
)

REJECT_RANGE = (
    "threshold_min and threshold_max are not valid parameters. "
    "The only valid parameters are segment (Business or Individual) and threshold_column "
    "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
    "Please specify which column you want to analyze instead."
)

# ex183
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run FP/FN tuning with threshold_min=1000 and step=500"},
    {"role": "assistant", "content": REJECT}
]})

# ex184
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Tune Business thresholds from threshold_min=5000 to threshold_max=50000"},
    {"role": "assistant", "content": REJECT_RANGE}
]})

# ex185
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP analysis for Individual with step=200"},
    {"role": "assistant", "content": REJECT_STEP}
]})

# ex186
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "threshold tuning for Business min_threshold=2500 threshold_step=250"},
    {"role": "assistant", "content": REJECT}
]})

# ex187
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "FP/FN sweep Business threshold_min=0 threshold_max=100000 threshold_step=10000"},
    {"role": "assistant", "content": REJECT}
]})

# ex188
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show threshold tuning with min_threshold=500 for Individual AVG_TRXN_AMT"},
    {"role": "assistant", "content": (
        "min_threshold is not a valid parameter. "
        "To analyze Individual customers by AVG_TRXN_AMT, please use: segment=Individual, threshold_column=AVG_TRXN_AMT."
    )}
]})

# ex189
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Sweep from 1000 to 20000 step 1000 for Business weekly transactions"},
    {"role": "assistant", "content": (
        "step, threshold_min, and threshold_max are not valid parameters. "
        "To analyze Business weekly transactions, use segment=Business and threshold_column=AVG_TRXNS_WEEK."
    )}
]})

# ex190
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run threshold sweep threshold_min=100 threshold_step=10 for Business"},
    {"role": "assistant", "content": REJECT}
]})

# ex191
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "FP/FN analysis Business threshold_max=80000"},
    {"role": "assistant", "content": (
        "threshold_max is not a valid parameter. "
        "The only valid parameters are segment (Business or Individual) and threshold_column "
        "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
        "To analyze Business customers, please specify which threshold column."
    )}
]})

# ex192
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Tune Individual TRXN_AMT_MONTHLY with threshold_min=10000 step=5000"},
    {"role": "assistant", "content": (
        "threshold_min and step are not valid parameters. "
        "To analyze Individual customers by TRXN_AMT_MONTHLY, use segment=Individual and threshold_column=TRXN_AMT_MONTHLY "
        "with no additional range parameters."
    )}
]})

# ex193
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me threshold sweep from 500 to 5000 for Business"},
    {"role": "assistant", "content": (
        "threshold_min and threshold_max are not valid parameters. "
        "The only valid parameters are segment and threshold_column. "
        "Please specify which column to analyze: AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY."
    )}
]})

# ex194
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Can you sweep threshold_min=200 threshold_max=2000 threshold_step=200 for Individual?"},
    {"role": "assistant", "content": REJECT}
]})

# ex195
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run FP analysis with min=1000 max=10000 step=500 for Business"},
    {"role": "assistant", "content": (
        "min, max, and step are not valid parameters. "
        "The only valid parameters are segment (Business or Individual) and threshold_column "
        "(AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY)."
    )}
]})

# ex196
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "threshold_step=1000 Business AVG_TRXN_AMT"},
    {"role": "assistant", "content": (
        "threshold_step is not a valid parameter. "
        "To analyze Business customers by AVG_TRXN_AMT, use segment=Business and threshold_column=AVG_TRXN_AMT "
        "with no additional parameters."
    )}
]})

# ex197
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Sweep Business AVG_TRXNS_WEEK from min_threshold=1 to max 20 step 2"},
    {"role": "assistant", "content": (
        "min_threshold, max, and step are not valid parameters. "
        "To analyze Business customers by AVG_TRXNS_WEEK, use segment=Business and threshold_column=AVG_TRXNS_WEEK."
    )}
]})


# ==============================
# MULTI-SENTENCE: One insight sentence only (ex198–ex212)
# ==============================

pc198 = (
    "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim) ===\n"
    "Segment: Business | threshold_column: AVG_TRXN_AMT\n"
    "At the lowest threshold ($8,421): TP=34, FP=286, FN=2, TN=0 (TP rate=94.4%, precision=10.6%).\n"
    "At current condition ($45,000): TP=29, FP=198, FN=7, TN=88 (TP rate=80.6%, precision=12.8%).\n"
    "At the highest threshold ($120,000): TP=18, FP=95, FN=18, TN=191 (TP rate=50.0%, precision=15.9%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED ==="
)

# ex198
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show threshold tuning for Business customers using AVG_TRXN_AMT"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex198", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc198, "tool_call_id": "call_ex198"},
    {"role": "assistant", "content": (
        pc198 + "\n\n"
        "At the current $45,000 threshold, Business customers show TP rate=80.6% and precision=12.8%; "
        "raising to $120,000 improves precision to 15.9% but reduces TP rate to 50.0% with 18 missed SARs."
    )}
]})

pc199 = (
    "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim) ===\n"
    "Segment: Individual | threshold_column: AVG_TRXNS_WEEK\n"
    "At the lowest threshold (0.1): TP=612, FP=1847, FN=72, TN=0 (TP rate=89.5%, precision=24.9%).\n"
    "At current condition (2.0): TP=580, FP=1423, FN=104, TN=424 (TP rate=84.8%, precision=28.9%).\n"
    "At the highest threshold (10.0): TP=310, FP=412, FN=374, TN=1435 (TP rate=45.3%, precision=42.9%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED ==="
)

# ex199
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "FP/FN analysis for Individual customers by weekly transaction count"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex199", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc199, "tool_call_id": "call_ex199"},
    {"role": "assistant", "content": (
        pc199 + "\n\n"
        "The current threshold of 2.0 weekly transactions catches 84.8% of Individual SARs with 1,423 false positives (precision=28.9%)."
    )}
]})

pc200 = (
    "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim) ===\n"
    "Segment: Business | threshold_column: TRXN_AMT_MONTHLY\n"
    "At the lowest threshold ($19,723): TP=34, FP=254, FN=2, TN=3 (TP rate=94.4%, precision=11.8%).\n"
    "At current condition ($60,000): TP=28, FP=172, FN=8, TN=85 (TP rate=77.8%, precision=14.0%).\n"
    "At the highest threshold ($109,723): TP=18, FP=102, FN=18, TN=155 (TP rate=50.0%, precision=15.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED ==="
)

# ex200
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Business threshold tuning by monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex200", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc200, "tool_call_id": "call_ex200"},
    {"role": "assistant", "content": (
        pc200 + "\n\n"
        "Raising the Business monthly amount threshold from $60,000 to $109,723 reduces FP from 172 to 102 "
        "but drops TP rate from 77.8% to 50.0%, missing 10 additional SARs."
    )}
]})

# ex201
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Tune Business monthly amount thresholds"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex201", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc200, "tool_call_id": "call_ex201"},
    {"role": "assistant", "content": (
        pc200 + "\n\n"
        "The Business TRXN_AMT_MONTHLY sweep shows TP rate declining from 94.4% to 50.0% as the threshold rises "
        "from $19,723 to $109,723, with FP falling from 254 to 102."
    )}
]})

pc202 = (
    "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim) ===\n"
    "Segment: Individual | threshold_column: AVG_TRXN_AMT\n"
    "At the lowest threshold ($500): TP=520, FP=2104, FN=48, TN=0 (TP rate=91.5%, precision=19.8%).\n"
    "At current condition ($3,000): TP=488, FP=1621, FN=80, TN=483 (TP rate=85.9%, precision=23.1%).\n"
    "At the highest threshold ($15,000): TP=312, FP=704, FN=256, TN=1400 (TP rate=54.9%, precision=30.7%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED ==="
)

# ex202
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Individual FP/FN tuning by AVG_TRXN_AMT"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex202", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc202, "tool_call_id": "call_ex202"},
    {"role": "assistant", "content": (
        pc202 + "\n\n"
        "Individual customers at the current $3,000 threshold show TP rate=85.9% with 1,621 false positives; "
        "raising to $15,000 improves precision to 30.7% but TP rate falls to 54.9%."
    )}
]})

pc203 = (
    "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim) ===\n"
    "Segment: Business | threshold_column: AVG_TRXNS_WEEK\n"
    "At the lowest threshold (0.2): TP=36, FP=312, FN=0, TN=0 (TP rate=100.0%, precision=10.3%).\n"
    "At current condition (1.0): TP=32, FP=218, FN=4, TN=94 (TP rate=88.9%, precision=12.8%).\n"
    "At the highest threshold (5.0): TP=14, FP=68, FN=22, TN=244 (TP rate=38.9%, precision=17.1%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED ==="
)

# ex203
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Business weekly transactions threshold tuning"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex203", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc203, "tool_call_id": "call_ex203"},
    {"role": "assistant", "content": (
        pc203 + "\n\n"
        "The current threshold of 1.0 weekly transactions catches 88.9% of Business SARs with 218 false positives (precision=12.8%)."
    )}
]})

# Rule sweep one-sentence examples
pc204 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month ACH profile mean (currently 5.0)\n"
    "Labeled population: 769 customers (TP+FN pool=132 SAR, FP+TN pool=637 non-SAR)\n"
    "At current condition (5.00): TP=72, FP=274, FN=60, TN=363 (TP rate=54.5%, precision=20.8%).\n"
    "At the highest value (10.00): TP=30, FP=75, FN=102, TN=562 (TP rate=22.7%, precision=28.6%).\n"
    "To keep TP rate >=90%: z_threshold <= 1.00 => TP=120, FP=637, FN=12, TN=0 (precision=15.9%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

# ex204
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show z_threshold sweep for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex204", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc204, "tool_call_id": "call_ex204"},
    {"role": "assistant", "content": (
        pc204 + "\n\n"
        "At z_threshold=5.0, the rule catches 54.5% of ACH SARs with 274 false positives (precision=20.8%); "
        "raising to z=10 reduces FP to 75 but drops TP rate to 22.7% with 102 missed SARs."
    )}
]})

pc205 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Sweep parameter: floor_amount - Minimum 14-day outgoing to trigger (currently $5,000)\n"
    "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
    "At current condition (5,000): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
    "At the highest value (15,000): TP=142, FP=601, FN=46, TN=357 (TP rate=75.5%, precision=19.1%).\n"
    "To keep TP rate >=90%: floor_amount <= 7,000 => TP=170, FP=796, FN=18, TN=162 (precision=17.6%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

# ex205
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show floor_amount sweep for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex205", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": pc205, "tool_call_id": "call_ex205"},
    {"role": "assistant", "content": (
        pc205 + "\n\n"
        "At the current $5,000 floor the Elder Abuse rule catches 99.5% of SARs (FN=1); "
        "raising to $7,000 maintains a 90% TP rate while reducing FP from 921 to 796."
    )}
]})

pc206 = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (Check)\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month Check profile mean (currently 2.0)\n"
    "Labeled population: 316 customers (TP+FN pool=76 SAR, FP+TN pool=240 non-SAR)\n"
    "At current condition (2.00): TP=36, FP=73, FN=40, TN=167 (TP rate=47.4%, precision=33.0%).\n"
    "At the highest value (8.00): TP=12, FP=18, FN=64, TN=222 (TP rate=15.8%, precision=40.0%).\n"
    "To keep TP rate >=90%: z_threshold <= 0.00 => TP=69, FP=195, FN=7, TN=45 (precision=26.1%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

# ex206
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Activity Deviation Check z_threshold sweep"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex206", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc206, "tool_call_id": "call_ex206"},
    {"role": "assistant", "content": (
        pc206 + "\n\n"
        "The current z=2.0 condition catches 47.4% of Check deviation SARs with 73 false positives (precision=33.0%); "
        "to reach 90% TP rate, z must be lowered to 0.0 which increases FP to 195."
    )}
]})

# 2D sweep one-sentence examples
pc207 = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Grid: 8 x 9 = 72 combinations\n"
    "SAR pool: 74 | Non-SAR pool: 404\n"
    "At current condition (pair_total=20,000, ratio_tolerance=10%): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=50%): pair_total=40,000, ratio_tolerance=7.5% => TP=52, FP=273, FN=22, TN=131, TP rate=70.3%, precision=16.0%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END 2D SWEEP ==="
)

# ex207
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex207", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "content": pc207, "tool_call_id": "call_ex207"},
    {"role": "assistant", "content": (
        pc207 + "\n\n"
        "The best FP reduction cell (pair_total=40,000, ratio_tolerance=7.5%) cuts FP from 404 to 273 and improves precision to 16.0%, "
        "at the cost of 22 missed SARs (FN=22) versus FN=0 at the current setting."
    )}
]})

pc208 = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Grid: 9 x 11 = 99 combinations\n"
    "SAR pool: 188 | Non-SAR pool: 958\n"
    "At current condition (floor_amount=5,000, z_threshold=3): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%).\n"
    "Best FP reduction (TP rate >=50%): floor_amount=9,000, z_threshold=7 => TP=97, FP=345, FN=91, TN=613, TP rate=51.6%, precision=21.9%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END 2D SWEEP ==="
)

# ex208
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "2D heatmap for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex208", "rule_2d_sweep", {"risk_factor": "Elder Abuse"})]},
    {"role": "tool", "content": pc208, "tool_call_id": "call_ex208"},
    {"role": "assistant", "content": (
        pc208 + "\n\n"
        "The best FP reduction cell at floor_amount=9,000 and z_threshold=7 reduces FP from 921 to 345, "
        "but FN rises to 91 — nearly half of Elder Abuse SARs would be missed at that combination."
    )}
]})

# ex209 — multi-sentence trap: no crossover narrative
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Where do FP and FN cross for Business monthly amount?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex209", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc200, "tool_call_id": "call_ex209"},
    {"role": "assistant", "content": (
        pc200 + "\n\n"
        "The PRE-COMPUTED sweep shows FP declining from 254 to 102 and FN rising from 2 to 18 as the threshold "
        "increases from $19,723 to $109,723 — no exact crossover value is given in the sweep results."
    )}
]})

# ex210 — no crossover narrative for rule sweep
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show ACH z_threshold sweep and find the crossover"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex210", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc204, "tool_call_id": "call_ex210"},
    {"role": "assistant", "content": (
        pc204 + "\n\n"
        "The PRE-COMPUTED sweep shows FP decreasing and FN increasing as z_threshold rises from 1.0 to 10.0 — "
        "the exact crossover point is not listed in the sweep results."
    )}
]})

# ex211 — no fabricated ratios
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What share of Individual alerts are genuine SARs at current thresholds?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex211", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc202, "tool_call_id": "call_ex211"},
    {"role": "assistant", "content": (
        pc202 + "\n\n"
        "At the current $3,000 threshold, precision=23.1% — meaning 23.1% of alerted Individual customers are genuine SARs."
    )}
]})

# ex212 — precision is not TP rate
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How often do Business alerts turn out to be genuine SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex212", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc203, "tool_call_id": "call_ex212"},
    {"role": "assistant", "content": (
        pc203 + "\n\n"
        "At the current threshold of 1.0 weekly transactions, precision=12.8% — meaning 12.8% of Business alerts are genuine SARs, "
        "not to be confused with the TP rate of 88.9% which measures how many SARs the threshold catches."
    )}
]})


# ==============================
# V7-C: Precision vs TP rate distinction (ex213–ex217)
# ==============================

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

# ex213 — ACH precision is 15.2% not "half"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rule has the highest false positive count?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex213", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex213"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Activity Deviation (ACH) generates the most false positives (769) across all rules, "
        "with precision=15.2% — meaning 15.2% of ACH alerts are genuine SARs, or about 1 in 7."
    )}
]})

# ex214 — precision != TP rate in rule sweep
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show ACH floor_amount sweep — what fraction of alerts are real SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex214", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})]},
    {"role": "tool", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Sweep parameter: floor_amount\n"
        "At current condition (50,000): TP=136, FP=756, FN=2, TN=13 (TP rate=98.6%, precision=15.2%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ==="
    ), "tool_call_id": "call_ex214"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Sweep parameter: floor_amount\n"
        "At current condition (50,000): TP=136, FP=756, FN=2, TN=13 (TP rate=98.6%, precision=15.2%).\n"
        "(Detailed sweep table shown in the chart below.)\n"
        "=== END RULE SWEEP ===\n\n"
        "At the current $50,000 floor, precision=15.2% means 15.2% of ACH alerts are genuine SARs — "
        "distinct from the TP rate of 98.6% which measures how many actual SARs the rule catches."
    )}
]})

# ex215 — Velocity Single precision is 15.5% not "most" or "half"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the precision for Velocity Single?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex215", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex215"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Velocity Single has precision=15.5%, meaning 15.5% of its 478 alerts are genuine SARs (74 SAR, 404 FP)."
    )}
]})

# ex216 — Structuring has 100% precision because FP=0, not because TP rate is high
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Which rules have perfect precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex216", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex216"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Structuring (Incoming Cash), Structuring (Outgoing Cash), and CTR Client all show precision=100.0% "
        "because FP=0 for each — every alert generated by these rules corresponds to a confirmed SAR."
    )}
]})

# ex217 — Elder Abuse: precision 16.4% is not "most" alerts being SARs
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many Elder Abuse alerts are genuine SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex217", "list_rules", {})]},
    {"role": "tool", "content": pc_list_rules, "tool_call_id": "call_ex217"},
    {"role": "assistant", "content": (
        pc_list_rules + "\n\n"
        "Elder Abuse generates 1,146 alerts of which 188 are genuine SARs and 958 are false positives — "
        "precision=16.4%, meaning about 1 in 6 Elder Abuse alerts is a confirmed SAR."
    )}
]})


# ==============================
# V7-N + V8-A: SAR routing (ex218–ex232)
# ==============================

pc_sar_indiv = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Individual | threshold_column: AVG_TRXN_AMT\n"
    "Total customers: 4,668 | SAR customers: 684 | Non-SAR: 3,984\n"
    "At current threshold: TP=612, FP=1,847, FN=72, TN=2,137 (TP rate=89.5%, precision=24.9%).\n"
    "=== END SAR BACKTEST ==="
)

# ex218 — "SAR filing rate" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Individual customers?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex218", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv, "tool_call_id": "call_ex218"},
    {"role": "assistant", "content": (
        pc_sar_indiv + "\n\n"
        "Among Individual customers, 684 of 4,668 are confirmed SARs; the current threshold catches 612 of them "
        "(TP rate=89.5%) with precision=24.9%."
    )}
]})

pc_sar_biz = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim) ===\n"
    "Segment: Business | threshold_column: AVG_TRXNS_WEEK\n"
    "Total customers: 14,423 | SAR customers: 36 | Non-SAR: 14,387\n"
    "At current threshold: TP=32, FP=218, FN=4, TN=14,169 (TP rate=88.9%, precision=12.8%).\n"
    "=== END SAR BACKTEST ==="
)

# ex219 — "SAR detection rate" → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR detection rate for Business customers?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex219", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz, "tool_call_id": "call_ex219"},
    {"role": "assistant", "content": (
        pc_sar_biz + "\n\n"
        "Business customers show a SAR detection rate of 88.9% — 32 of 36 confirmed SARs are caught at the current threshold, "
        "with 218 false positive alerts (precision=12.8%)."
    )}
]})

# ex220 — "how many SARs" for segment → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many Individual customers are flagged as SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex220", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv, "tool_call_id": "call_ex220"},
    {"role": "assistant", "content": (
        pc_sar_indiv + "\n\n"
        "684 of 4,668 Individual customers are confirmed SARs; the current threshold alerts 612 of them (TP rate=89.5%)."
    )}
]})

# ex221 — "SAR rate" for Business → sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the SAR rate for Business"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex221", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz, "tool_call_id": "call_ex221"},
    {"role": "assistant", "content": (
        pc_sar_biz + "\n\n"
        "Business customers have 36 confirmed SARs out of 14,423 total; the current threshold catches 32 (TP rate=88.9%, precision=12.8%)."
    )}
]})

# ex222 — "SAR backtest for ACH rule" → rule_sar_backtest, NOT list_rules, NOT floor_amount sweep
pc_ach_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month ACH profile mean (currently 5.0)\n"
    "Labeled population: 769 customers (TP+FN pool=132 SAR, FP+TN pool=637 non-SAR)\n"
    "At current condition (5.00): TP=72, FP=274, FN=60, TN=363 (TP rate=54.5%, precision=20.8%).\n"
    "At the highest value (10.00): TP=30, FP=75, FN=102, TN=562 (TP rate=22.7%, precision=28.6%).\n"
    "To keep TP rate >=90%: z_threshold <= 1.00 => TP=120, FP=637, FN=12, TN=0 (precision=15.9%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex222", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_ex222"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "The ACH rule at z_threshold=5.0 catches 54.5% of SARs with 274 false positives (precision=20.8%); "
        "to maintain 90% SAR catch rate, z must be lowered to 1.0 which increases FP to 637."
    )}
]})

# ex223 — "SAR backtest for Elder Abuse" → rule_sar_backtest
pc_ea_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3.0)\n"
    "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR)\n"
    "At current condition (3.00): TP=187, FP=921, FN=1, TN=37 (TP rate=99.5%, precision=16.9%).\n"
    "At the highest value (9.00): TP=98, FP=312, FN=90, TN=646 (TP rate=52.1%, precision=23.9%).\n"
    "To keep TP rate >=90%: z_threshold <= 5.00 => TP=170, FP=644, FN=18, TN=314 (precision=20.9%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex223", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ea_z, "tool_call_id": "call_ex223"},
    {"role": "assistant", "content": (
        pc_ea_z + "\n\n"
        "At z_threshold=3.0, the Elder Abuse rule catches 99.5% of SARs (FN=1) with 921 false positives (precision=16.9%)."
    )}
]})

# ex224 — "SAR performance for Velocity Single" → rule_sar_backtest
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

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR performance for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex224", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_vs_pt, "tool_call_id": "call_ex224"},
    {"role": "assistant", "content": (
        pc_vs_pt + "\n\n"
        "At the current $20,000 pair_total threshold, Velocity Single catches 100% of SARs (FN=0) with 404 false positives (precision=15.5%)."
    )}
]})

# ex225 — "How many SARs does ACH catch?" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "How many SARs does the Activity Deviation ACH rule catch?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex225", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_ex225"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "At the current z_threshold=5.0, Activity Deviation (ACH) catches 72 of 132 SARs in the labeled population (TP rate=54.5%, FN=60)."
    )}
]})

# ex226 — "SAR catch rate for Elder Abuse" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Elder Abuse rule?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex226", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ea_z, "tool_call_id": "call_ex226"},
    {"role": "assistant", "content": (
        pc_ea_z + "\n\n"
        "At z_threshold=3.0, the Elder Abuse rule catches 187 of 188 SARs in the labeled population (TP rate=99.5%, FN=1)."
    )}
]})

# ex227 — "rule-level SAR analysis" → rule_sar_backtest, not list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Analyze SAR outcomes for the Velocity Single rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex227", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_vs_pt, "tool_call_id": "call_ex227"},
    {"role": "assistant", "content": (
        pc_vs_pt + "\n\n"
        "Velocity Single at pair_total=$20,000 catches all 74 SARs in the labeled population (TP rate=100.0%) "
        "but generates 404 false positives — raising pair_total to $64,000 reduces FP to 186 at the cost of 36 missed SARs."
    )}
]})

# ex228 — "What does ACH rule look like for SARs?" → rule_sar_backtest
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What does the Activity Deviation ACH rule look like for SAR coverage?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex228", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_ach_z, "tool_call_id": "call_ex228"},
    {"role": "assistant", "content": (
        pc_ach_z + "\n\n"
        "Activity Deviation (ACH) covers a labeled population of 769 customers; at z_threshold=5.0, 72 of 132 SARs are caught (TP rate=54.5%, precision=20.8%)."
    )}
]})

# ex229 — "SAR backtest for Check rule" → rule_sar_backtest
pc_check_z = (
    "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (Check)\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 12-month Check profile mean (currently 2.0)\n"
    "Labeled population: 316 customers (TP+FN pool=76 SAR, FP+TN pool=240 non-SAR)\n"
    "At current condition (2.00): TP=36, FP=73, FN=40, TN=167 (TP rate=47.4%, precision=33.0%).\n"
    "To keep TP rate >=90%: z_threshold <= 0.00 => TP=69, FP=195, FN=7, TN=45 (precision=26.1%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END RULE SWEEP ==="
)

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "SAR backtest for Activity Deviation Check"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex229", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)", "sweep_param": "z_threshold"})]},
    {"role": "tool", "content": pc_check_z, "tool_call_id": "call_ex229"},
    {"role": "assistant", "content": (
        pc_check_z + "\n\n"
        "The Activity Deviation (Check) rule at z=2.0 catches 47.4% of Check SARs (TP=36, FN=40); "
        "lowering z to 0.0 raises the TP rate to 90%+ but increases FP from 73 to 195."
    )}
]})

# ex230 — "SAR filing rate for Business" → sar_backtest (segment level, not rule level)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Business customers overall?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex230", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc_sar_biz, "tool_call_id": "call_ex230"},
    {"role": "assistant", "content": (
        pc_sar_biz + "\n\n"
        "Across 14,423 Business customers, 36 are confirmed SARs (0.25%); the current threshold catches 32 of them (TP rate=88.9%)."
    )}
]})

# ex231 — "SAR filing for Individual" → sar_backtest (segment level, not policy KB)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What proportion of Individual customers file SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex231", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc_sar_indiv, "tool_call_id": "call_ex231"},
    {"role": "assistant", "content": (
        pc_sar_indiv + "\n\n"
        "684 of 4,668 Individual customers are confirmed SARs; the current threshold generates alerts for 612 of them "
        "plus 1,847 false positives (precision=24.9%)."
    )}
]})

# ex232 — "SAR backtest for Velocity Single" → rule_sar_backtest (not sar_backtest)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me the SAR backtest for the Velocity Single rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex232", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})]},
    {"role": "tool", "content": pc_vs_pt, "tool_call_id": "call_ex232"},
    {"role": "assistant", "content": (
        pc_vs_pt + "\n\n"
        "Velocity Single covers 478 customers in the labeled population; at the current $20,000 pair_total, "
        "all 74 SARs are caught (TP rate=100.0%) with 404 false positives (precision=15.5%)."
    )}
]})


# ==============================
# No fabricated crossover in threshold tuning (ex233–ex237)
# ==============================

# ex233 — no crossover narrative, no invented intersection
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Where do false positives and false negatives cross for Business monthly amount?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex233", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "content": pc200, "tool_call_id": "call_ex233"},
    {"role": "assistant", "content": (
        pc200 + "\n\n"
        "The PRE-COMPUTED results show FP declining from 254 to 102 and FN rising from 2 to 18 across the sweep range; "
        "the exact threshold where FP equals FN is not stated in the results."
    )}
]})

# ex234 — no crossover for Individual AVG_TRXNS_WEEK
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Find the optimal threshold where FP equals FN for Individual weekly transactions"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex234", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc199, "tool_call_id": "call_ex234"},
    {"role": "assistant", "content": (
        pc199 + "\n\n"
        "The sweep shows FP declining and FN increasing as the threshold rises from 0.1 to 10.0; "
        "a specific FP=FN crossover threshold is not listed in the PRE-COMPUTED results."
    )}
]})

# ex235 — numbers only from PRE-COMPUTED, no derived stats
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Tune Business AVG_TRXN_AMT — I want to minimize total errors"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex235", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc198, "tool_call_id": "call_ex235"},
    {"role": "assistant", "content": (
        pc198 + "\n\n"
        "The PRE-COMPUTED sweep shows TP rate declining from 94.4% to 50.0% as the threshold rises from $8,421 to $120,000 — "
        "to choose the optimal point, compare the FP and FN counts for each threshold in the sweep table."
    )}
]})

# ex236 — no "investigator workload" or editorial
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Business weekly transaction threshold tuning"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex236", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "content": pc203, "tool_call_id": "call_ex236"},
    {"role": "assistant", "content": (
        pc203 + "\n\n"
        "The current threshold of 1.0 weekly transactions yields TP rate=88.9% with 218 false positives (precision=12.8%); "
        "the sweep ranges from 0.2 (TP rate=100.0%, FP=312) to 5.0 (TP rate=38.9%, FP=68)."
    )}
]})

# ex237 — no fabricated "balance point" claim
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show Individual AVG_TRXN_AMT tuning and find the balance point"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex237", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "content": pc202, "tool_call_id": "call_ex237"},
    {"role": "assistant", "content": (
        pc202 + "\n\n"
        "Across the Individual AVG_TRXN_AMT sweep, FP falls from 2,104 to 704 and FN rises from 48 to 256 "
        "as the threshold increases from $500 to $15,000 — the exact balance point is not specified in the PRE-COMPUTED results."
    )}
]})


# ==============================
# DISPLAY_CLUSTERS with insight sentence (ex238–ex242)
# ==============================

pc_biz_clusters = (
    "Segment: Business | Active accounts: 14,423 (excluded 67,203 with no transactions)\n"
    "Clusters: 4 | Features: 7 numeric + 8 encoded categorical (7 original)\n"
    "PCA variance explained: PC1=18.9%, PC2=16.6%\n\n"
    "Cluster 1: n=1,242 (8.6% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=9,848.5, "
    "trxn_amt_monthly=54,128.5, CURRENT_BALANCE=15,542,100.5, ACCT_AGE_YEARS=7.1\n"
    "Cluster 2: n=12,506 (86.7% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=12,876.7, "
    "trxn_amt_monthly=80,157.0, INCOME=0.0, CURRENT_BALANCE=0.0, ACCT_AGE_YEARS=5.0\n"
    "Cluster 3: n=32 (0.2% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=8,821.1, "
    "trxn_amt_monthly=59,129.5, CURRENT_BALANCE=0.0, ACCT_AGE_YEARS=4.8\n"
    "Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=1.0, avg_weekly_trxn_amt=634,098.0, "
    "trxn_amt_monthly=744,739.0, CURRENT_BALANCE=0.0, ACCT_AGE_YEARS=4.8\n\n"
    "HIGHEST-RISK CLUSTER: Cluster 4 (n=643, avg_num_trxns=1.0)."
)

# ex238 — "show only highest-risk cluster" → ds_cluster_analysis + DISPLAY_CLUSTERS
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show only the highest-risk Business cluster"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex238", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_biz_clusters, "tool_call_id": "call_ex238"},
    {"role": "assistant", "content": (
        pc_biz_clusters + "\n\n"
        "Cluster 4 is the highest-risk Business cluster with avg_num_trxns=1.0 and avg_weekly_trxn_amt=634,098.0 — "
        "significantly higher transaction activity than the other three clusters.\n"
        "DISPLAY_CLUSTERS: 4"
    )}
]})

# ex239 — "show only high risk" → DISPLAY_CLUSTERS
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show me only the high-risk Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex239", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_biz_clusters, "tool_call_id": "call_ex239"},
    {"role": "assistant", "content": (
        pc_biz_clusters + "\n\n"
        "Cluster 4 represents the highest-risk Business segment (n=643) with avg_weekly_trxn_amt=634,098.0 and avg_num_trxns=1.0, "
        "far exceeding the other clusters.\n"
        "DISPLAY_CLUSTERS: 4"
    )}
]})

pc_indiv_clusters = (
    "Segment: Individual | Active accounts: 212,608 (excluded 878,207 with no transactions)\n"
    "Clusters: 4\n"
    "PCA variance explained: PC1=13.1%, PC2=10.0%\n\n"
    "Cluster 1: n=5,197 (2.4% of active accounts), avg_num_trxns=1.0, avg_weekly_trxn_amt=754,931.0, "
    "trxn_amt_monthly=814,450.0, INCOME=0.0, CURRENT_BALANCE=204,624.0, ACCT_AGE_YEARS=11.8, AGE=61.2\n"
    "Cluster 2: n=77,106 (36.3% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=16,999.8, "
    "trxn_amt_monthly=113,280.4, INCOME=0.0, CURRENT_BALANCE=103,813.0, ACCT_AGE_YEARS=8.8, AGE=50.6\n"
    "Cluster 3: n=32,826 (15.4% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=19,159.6, "
    "trxn_amt_monthly=128,233.9, INCOME=0.0, CURRENT_BALANCE=151,676.0, ACCT_AGE_YEARS=10.3, AGE=37.9\n"
    "Cluster 4: n=97,479 (45.8% of active accounts), avg_num_trxns=0.2, avg_weekly_trxn_amt=16,791.4, "
    "trxn_amt_monthly=117,399.8, INCOME=0.0, CURRENT_BALANCE=244,183.0, ACCT_AGE_YEARS=19.4, AGE=70.5\n\n"
    "HIGHEST-RISK CLUSTER: Cluster 1 (n=5,197, avg_num_trxns=1.0)."
)

# ex240 — Individual highest-risk cluster → DISPLAY_CLUSTERS: 1
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show only the highest-risk Individual cluster"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex240", "ds_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "content": pc_indiv_clusters, "tool_call_id": "call_ex240"},
    {"role": "assistant", "content": (
        pc_indiv_clusters + "\n\n"
        "Cluster 1 is the highest-risk Individual cluster (n=5,197) with avg_num_trxns=1.0 and avg_weekly_trxn_amt=754,931.0 — "
        "the only cluster with transaction activity well above the population baseline.\n"
        "DISPLAY_CLUSTERS: 1"
    )}
]})

# ex241 — "show top risk cluster" → DISPLAY_CLUSTERS
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Show the top risk cluster for Business"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex241", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_biz_clusters, "tool_call_id": "call_ex241"},
    {"role": "assistant", "content": (
        pc_biz_clusters + "\n\n"
        "Cluster 4 is the top-risk Business cluster with the highest avg_num_trxns (1.0) and avg_weekly_trxn_amt (634,098.0) "
        "among all four Business clusters (n=643, 4.5% of active accounts).\n"
        "DISPLAY_CLUSTERS: 4"
    )}
]})

# ex242 — "which cluster should I focus on?" → DISPLAY_CLUSTERS with insight
examples.append({"messages": [
    {"role": "system", "content": CLUSTER_SYSTEM},
    {"role": "user", "content": "Which Business cluster should I focus on for AML risk?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("call_ex242", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": pc_biz_clusters, "tool_call_id": "call_ex242"},
    {"role": "assistant", "content": (
        pc_biz_clusters + "\n\n"
        "Cluster 4 has the highest AML risk indicators — avg_weekly_trxn_amt=634,098.0 and avg_num_trxns=1.0 — "
        "compared to the other three clusters which show avg_weekly_trxn_amt below $20,000.\n"
        "DISPLAY_CLUSTERS: 4"
    )}
]})


# ==============================
# Write output
# ==============================
out_path = r"C:\Users\Aaditya\PycharmProjects\framlagents-ftllm\finetune\data\framl_train_failures_v8.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} examples to {out_path}")

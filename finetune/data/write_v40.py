"""
V40 training examples (2026-05-06).

Targets — gaps discovered during V39 app testing:

  X1–X4   Gap 1: Lowest FP direction — model returned highest FP instead of lowest.
           Correct: call list_rules, pick MINIMUM FP count / MAXIMUM precision.

  X5–X8   Gap 2: Precision synthesis — model gave no answer after calling list_rules.
           Correct: rank rules by precision column (already in tool result), name winner.

  X9–X12  Gap 3: Multi-turn data isolation — model copied prior tool result (Elder Abuse)
           when user asked about a different rule in the next turn.
           Correct: ignore prior tool data in history, call the correct fresh tool.

  X13–X16 Gap 4: Cluster stats hallucination — model invented cluster stats instead of
           reading from [PREVIOUS CLUSTERING RESULT] context block.
           Correct: answer from context block, NO tool call.

  X17–X20 Gap 5: Non-AML OFAC/sanctions context → out_of_scope.
           Correct: classifier returns out_of_scope when OFAC/sanctions appear as names
           in a clearly non-AML sentence.

Conventions (match write_v38.py / write_v39.py):
  - System prompts copied verbatim from agents/*.py
  - Tool content: "Tool result for <tool>:\n{pc_body}"
  - Assistant content after tool call: pc_body + "\n\n" + ONE insight sentence
  - Cluster follow-up context: prev_context(stats_plain_text) injected in user message
  - stats blocks use bold markdown (no === markers, no "Tool result:" prefix)

Combined: aria_train_combined_v39_full.jsonl (base) + 20 V40 = aria_train_combined_v40_full.jsonl
"""

import json, pathlib

DATA_DIR      = pathlib.Path(__file__).parent
V39_BASE_PATH = DATA_DIR / "aria_train_combined_v39_full.jsonl"
V40_ONLY_PATH = DATA_DIR / "aria_train_v40.jsonl"
V40_FULL_PATH = DATA_DIR / "aria_train_combined_v40_full.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

def prev_context(stats):
    return f"[PREVIOUS CLUSTERING RESULT]\n{stats}\n[END PREVIOUS RESULT]"

# ---------------------------------------------------------------------------
# System prompts — copied verbatim from agents/*.py (do not abbreviate)
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change, and run SAR backtests and "
    "2D sweeps for AML rule performance. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "DEFINITIONS (always apply these exactly — do not contradict them):\n"
    "- TP (True Positive): SAR customer who IS alerted — correctly caught suspicious activity.\n"
    "- FP (False Positive): Non-SAR customer who IS alerted — unnecessary investigation. HIGHER threshold → FEWER FPs.\n"
    "- FN (False Negative): SAR customer who is NOT alerted — missed suspicious activity. HIGHER threshold → MORE FNs.\n"
    "- TN (True Negative): Non-SAR customer who is NOT alerted — correctly silent.\n"
    "- TP rate: TP / (TP + FN) — share of SAR customers caught. Also called recall or sensitivity.\n"
    "- Precision: TP / (TP + FP) — share of alerts that are genuine SARs.\n"
    "- Crossover: the threshold where FP and FN counts are closest — the optimal operating point.\n"
    "- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).\n"
    "- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory. EXCEPTION: if the user provides invalid parameters (threshold_min, threshold_max, threshold_step, step, min_threshold) or an invalid threshold_column, do NOT call any tool — follow Rule 14 instead.\n"
    "2. For any question about FP, FN, threshold, alert rates, or transactions — call threshold_tuning.\n"
    "3. For general segment counts, totals, or dataset summaries (\"how many customers\", \"how many alerts\", \"summary of the data\", \"total accounts\", \"customers and alerts in the dataset\") — call segment_stats. NEVER answer count questions from memory.\n"
    "4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) with no specific rule named — call sar_backtest. If the user names a specific rule (e.g. \"Elder Abuse\", \"Velocity Single\", \"CTR Client\") — use rule_sar_backtest instead (see Rule 15).\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "6. segment must be exactly one of: Business, Individual\n"
    "7. If the user does not specify a segment, default to Business.\n"
    "8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.\n"
    "9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.\n"
    "10. Do NOT paraphrase, round, or restate the numbers differently.\n"
    "11. Do NOT include JSON or code blocks in your final reply.\n"
    "12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.\n"
    "13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.\n"
    "14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold, OR requests a threshold_column that is not one of AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY (e.g. daily balance, balance, net income, credit score, income, equity) — do NOT call the tool. State that the column is not available and list the three valid threshold_column options (AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.\n"
    "15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes — call rule_sar_backtest with risk_factor (e.g. \"Activity Deviation (ACH)\", \"Activity Deviation (Check)\", \"Elder Abuse\", \"Velocity Single\", \"Detect Excessive\") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.\n"
    "16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview — call list_rules.\n"
    "17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR — call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.\n"
    "18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. \"hover to see\", \"right-click to select\", \"click the cell\"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.\n"
    "19. When the user asks about a specific behavioral cluster (e.g. \"Cluster 3\", \"cluster 4\"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep. Do NOT pass cluster to threshold_tuning, sar_backtest, or segment_stats — those tools do not accept a cluster parameter.\n"
    "20. ONE insight sentence only. Do NOT add a second sentence or parenthetical. Do NOT describe heatmap positions (e.g. \"top-left\", \"highest density\"). Do NOT say \"zero false positives\" or \"zero FNs\" if the PRE-COMPUTED shows FP > 0 or FN > 0.\n"
    "21. If the user asks about \"highest FP rate\" or \"worst precision\" — they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.\n"
    "22. The system contains exactly 16 AML rules. Never state a different count.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. \"layering\", \"smurfing\") — state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule \"covers\" the concept.\n"
    "24. For any question about how ALL rules perform for a specific behavioral cluster — call cluster_rule_summary with the cluster number. Do NOT call list_rules or loop over rule_sar_backtest for this.\n"
    "25. If a previous tool call returned an error about an invalid sweep parameter (e.g. \"Unknown sweep_param_1\" or \"Unknown sweep_param_2\"), and you asked the user to choose a valid parameter, and the user's reply is a parameter name (e.g. floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window, min_transactions, days_required, daily_floor) — do NOT treat it as a new query. Resume the previous rule_2d_sweep or rule_sar_backtest call with the same risk_factor, keeping all valid parameters unchanged and replacing only the invalid one with the user's corrected choice.\n"
    "26. For pure definitional questions about TP, FP, FN, TN, precision, recall, crossover, the effect of raising or lowering thresholds on FP/FN counts, or what a 2D grid/sweep shows — answer DIRECTLY from the DEFINITIONS section above. Do NOT call any tool. Answer in 2–3 sentences using only the definitions listed above."
)

SEGMENTATION_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML. You identify natural customer behavioral "
    "segments using unsupervised K-Means clustering and explain their AML risk profiles. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) — call ds_cluster_analysis.\n"
    "3. For alert/FP distribution by segment — call alerts_distribution.\n"
    "4. For the legacy alerts dataset — call cluster_analysis only if the user explicitly asks.\n"
    "5. Do NOT call multiple segmentation tools for the same request — pick exactly one.\n"
    "6. customer_type must be exactly one of: Business, Individual, All\n"
    "   If the user does NOT specify a customer type, default to All.\n"
    "7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.\n"
    "   If the user says \"N clusters\", \"into N\", \"only N\", or \"I want N\" (e.g. \"cluster into 3\", \"I only want 2 clusters\"),\n"
    "   set n_clusters=N exactly in the tool call. Do NOT ignore the user's requested count and do NOT default to 4.\n"
    "8. If the user asks to prepare or refresh the raw data — call prepare_segmentation_data first.\n"
    "9. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result. Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.\n"
    "10. If the user asks to show specific clusters (e.g. \"show only cluster 3\", \"highest risk\",\n"
    "    \"top 2 high risk\", \"low activity clusters\"):\n"
    "    - Identify which cluster number(s) match the request from the stats\n"
    "      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY this and nothing else:\n"
    "      DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers (e.g. DISPLAY_CLUSTERS: 4  or  DISPLAY_CLUSTERS: 1,4)\n"
    "    - Do NOT mention this line in your text — it is a system directive, not for the user.\n"
    "    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.\n"
    "11. Do NOT include JSON, code blocks, or raw data tables in your final reply.\n"
    "12. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
    "13. Do NOT invent threshold values, dollar amounts, or cutoffs. Only reference numbers explicitly present in the tool result. Do NOT suggest specific threshold values (e.g. \"$250K\", \"< 80,000\") unless they appear verbatim in the tool result.\n"
    "14. If the user asks which cluster to set a threshold for, or asks for threshold recommendations per cluster — do NOT invent values. Tell the user to use the threshold_tuning or sar_backtest tools with the relevant segment instead.\n"
    "15. If a [PREVIOUS CLUSTERING RESULT] block is provided in the context AND the user is asking to characterize, describe, compare, or explain a specific cluster — answer from that data WITHOUT calling any tool. Compare the named cluster's stats (avg_trxn_amt, monthly volume, balance, account age) against the other clusters to identify what makes it distinctive. Name the risk profile in one sentence. Do NOT re-run clustering."
)

CLASSIFY_SYSTEM = (
    "You are a routing classifier for ARIA. Given a user query, respond with "
    "one or more of these labels (comma-separated, no other text):\n\n"
    "  threshold    — user wants to RUN threshold tuning analysis on OUR LOCAL DATA (FP/FN trade-off charts, sweep analysis)\n"
    "  segmentation — user wants to RUN clustering/segmentation on OUR LOCAL DATA (K-Means, treemap, behavioral groups)\n"
    "  ofac         — user wants to RUN OFAC sanctions screening on OUR LOCAL CUSTOMER DATA (SDN list hits, sanctioned country exposure)\n"
    "  policy       — user is asking a GENERAL KNOWLEDGE question about AML, compliance, regulations, industry practices, or best practices — does NOT require running local data analysis\n"
    "  greeting     — query is a greeting or social pleasantry (hello, hi, how are you, etc.)\n"
    "  out_of_scope — query is not related to any of the above AML topics\n\n"
    "Key distinction:\n"
    "- \"Show FP/FN tuning for Business customers\" → threshold  (run local analysis)\n"
    "- \"Show FP/FN threshold tuning for Individual customers\" → threshold\n"
    "- \"Run SAR backtest for Individual customers\" → threshold  (SAR backtest is a threshold tool)\n"
    "- \"What threshold catches 90% of SARs?\" → threshold\n"
    "- \"SAR catch rate for Business monthly transaction amount\" → threshold\n"
    "- \"Run SAR backtest\" → threshold\n"
    "- \"Show me a 2D grid for Activity Deviation ACH\" → threshold  (2D sweep is a threshold tool)\n"
    "- \"How do floor amount and sigma interact for Activity Deviation?\" → threshold\n"
    "- \"Show me the ACH deviation rule performance\" → threshold\n"
    "- \"What is the SAR catch rate for Activity Deviation Check?\" → threshold\n"
    "- \"Show the heatmap for Elder Abuse\" → threshold\n"
    "- \"2D analysis for Velocity Single\" → threshold\n"
    "- \"How does time window interact with floor amount for Detect Excessive?\" → threshold\n"
    "- \"Show me the AML rule performance overview\" → threshold  (list_rules is a threshold tool)\n"
    "- \"Which rules generate the most false positives?\" → threshold\n"
    "- \"What is the SAR catch rate for the Activity Deviation rule?\" → threshold\n"
    "- \"Show rule-level FP analysis\" → threshold\n"
    "- \"What happens to FP if I raise the age threshold for Elder Abuse?\" → threshold\n"
    "- \"How do banks manage alert volumes?\" → policy  (general knowledge question)\n"
    "- \"What is AML?\" → policy  (general knowledge question)\n"
    "- \"Cluster all customers\" → segmentation  (run local analysis)\n"
    "- \"What does AML policy say about structuring?\" → policy  (general knowledge + knowledge base)\n"
    "- \"Show alerts and false positive distribution across segments\" → segmentation  (distribution chart, NOT threshold tuning)\n"
    "- \"Show alert distribution\" → segmentation\n"
    "- \"How are alerts spread across segments?\" → segmentation\n"
    "- \"Which segment has the most alerts?\" → segmentation\n"
    "- \"What is the average transaction amount for Business customers?\" → threshold  (segment_stats tool)\n"
    "- \"How many alerts does the Individual segment have?\" → threshold  (segment_stats tool)\n"
    "- \"What are the transaction stats for Business customers?\" → threshold\n"
    "- \"Show me Business customer stats\" → threshold\n"
    "- \"Show me all AML rules\" → threshold  (list_rules is a threshold tool — NOT policy)\n"
    "- \"What rules are in the system?\" → threshold\n"
    "- \"List all the AML rules\" → threshold\n"
    "- \"What transactions are flagged by the layering rule?\" → threshold  (list_rules — 'layering' is not a KB topic)\n"
    "- \"Which rule covers layering?\" → threshold  (list_rules)\n"
    "- \"Show rule sweep for xyz_column\" → threshold  (rule sweep request, even with unknown param — NOT out_of_scope)\n"
    "- \"Show rule sweep for an invalid parameter\" → threshold\n"
    "- \"What is the SAR filing rate for Individual?\" → threshold  (sar_backtest is a threshold tool)\n"
    "- \"SAR filing rate for Business\" → threshold\n"
    "- \"Which rule has the highest FP rate?\" → threshold  (list_rules)\n"
    "- \"Which rules generate only false positives?\" → threshold\n"
    "- \"Run a SAR backtest for the structuring rule\" → threshold  (rule_sar_backtest — NOT out_of_scope)\n"
    "- \"SAR backtest for Elder Abuse\" → threshold\n"
    "- \"Show Elder Abuse sweep for Cluster 4\" → threshold  (cluster-filtered rule sweep)\n"
    "- \"Run SAR backtest for Activity Deviation ACH in Cluster 2\" → threshold\n"
    "- \"Show 2D heatmap for Elder Abuse for Cluster 3\" → threshold\n"
    "- \"Which cluster has the most false positives for Velocity Single?\" → threshold\n"
    "- \"Which cluster of Business customers has the highest transaction volume?\" → segmentation\n"
    "- \"Which Business cluster has the most activity?\" → segmentation\n"
    "- \"Which cluster has the most transaction activity?\" → segmentation\n"
    "- \"Show Business customer clusters by transaction behavior\" → segmentation\n"
    "- \"Run OFAC screening\" → ofac\n"
    "- \"Show OFAC sanctions exposure\" → ofac\n"
    "- \"Which customers are on the sanctions list?\" → ofac\n"
    "- \"How many customers are from sanctioned countries?\" → ofac\n"
    "- \"Show me OFAC hits\" → ofac\n"
    "- \"Screen customers against SDN list\" → ofac\n"
    "- \"What is our Iran/North Korea customer exposure?\" → ofac\n"
    "- \"Show comprehensive sanctions hits\" → ofac\n"
    "- \"Show me a 2D grid for Elder Abuse\" → threshold  (2D grid = 2D sweep, same tool)\n"
    "- \"Show 2D analysis for Detect Excessive Transaction Activity\" → threshold  (2D analysis = 2D sweep)\n"
    "- \"Run a 2D grid analysis for Velocity Single\" → threshold\n"
    "- \"Show grid analysis for Activity Deviation ACH\" → threshold\n"
    "- \"What are Canada's suspicious transaction reporting requirements?\" → policy\n"
    "- \"What are Canada's AML rules?\" → policy\n"
    "- \"What does FINTRAC require?\" → policy\n"
    "- \"What is AML structuring?\" → policy  (prefix 'AML' does not change the topic — still a policy question)\n"
    "- \"What is tructuring?\" → policy  (typo for 'structuring' — still an AML definition question)\n"
    "- \"What is smurfing?\" → policy  (synonym for structuring — AML definition question)\n"
    "- \"What is AML layering?\" → policy\n"
    "- \"What is AML typology?\" → policy\n"
    "- \"cluster into 3 groups\" → segmentation  (user specifying cluster count is still a segmentation request)\n"
    "- \"I only want 2 business clusters\" → segmentation\n"
    "- \"show me 4 clusters for Individual customers\" → segmentation\n"
    "- \"I want k-means with 3 clusters\" → segmentation\n"
    "- \"What are the EU requirements for beneficial ownership registers?\" → policy  (EU regulatory question)\n"
    "- \"What does the 4th AMLD require for customer due diligence?\" → policy\n"
    "- \"What does the 5th AMLD say about virtual assets?\" → policy\n"
    "- \"What are FATF recommendations for banks?\" → policy\n"
    "- \"What does UN Security Council Resolution 1373 require of banks?\" → policy\n"
    "- \"What are EBA guidelines on ML/TF risk factors?\" → policy\n"
    "- \"What are the beneficial ownership disclosure requirements?\" → policy\n"
    "- \"What does the EU AML Regulation require?\" → policy\n"
    "- \"What is the AMLA?\" → policy\n"
    "- \"Does UNODC have guidance on AML?\" → policy\n"
    "- \"What are PEP requirements under AML regulations?\" → policy\n"
    "- \"Thanks, that was helpful!\" → greeting\n"
    "- \"Thanks, that's great\" → greeting\n"
    "- \"Got it, thanks\" → greeting\n"
    "- \"Thank you\" → greeting\n"
    "- \"That was useful, thanks\" → greeting\n"
    "- \"Can you send this to my compliance team?\" → out_of_scope  (action request, not an AML analysis task)\n"
    "- \"Can you email this to someone?\" → out_of_scope\n"
    "- \"Can you export this as a PDF?\" → out_of_scope\n"
    "- \"What is a false positive?\" → threshold  (FP/FN definition is a threshold concept, not policy)\n"
    "- \"What is a false negative?\" → threshold\n"
    "- \"What is the difference between FP and FN?\" → threshold\n"
    "- \"Explain false positives in AML monitoring\" → threshold\n"
    "- \"What does FP mean?\" → threshold\n"
    "- \"Can you explain false positives and false negatives?\" → threshold\n"
    "- \"What is a 2D grid?\" → threshold  (2D grid = rule_2d_sweep — a threshold tool concept)\n"
    "- \"What is a 2D sweep?\" → threshold\n"
    "- \"How does a 2D grid work?\" → threshold\n"
    "- \"Are you ARIA?\" → greeting  (identity question — not an AML topic)\n"
    "- \"What is your name?\" → greeting\n"
    "- \"Who are you?\" → greeting\n"
    "- \"Ahoy!\" → greeting\n"
    "- \"Ahoy matey!\" → greeting\n"
    "- \"What are true positives in AML monitoring?\" → threshold  (TP/TN definitions are threshold/confusion-matrix concepts)\n"
    "- \"What are true negatives?\" → threshold\n"
    "- \"What is the difference between TP and TN?\" → threshold\n"
    "- \"What is OFAC?\" → policy  (definition question — NOT a screening request)\n"
    "- \"What does OFAC stand for?\" → policy\n"
    "- \"My dog OFAC met a cat the other day\" → out_of_scope  (OFAC here is a name, not AML topic)\n"
    "- \"OFAC said hello\" → out_of_scope  (not an AML query)\n"
    "- \"Is OFAC the same as sanctions screening?\" → policy  (terminology question — NOT a screening request)\n"
    "- \"What does OFAC stand for?\" → policy  (terminology question)\n"
    "- \"What is OFAC?\" → policy\n"
    "- \"What are the rules that have z_threshold as a parameter?\" → threshold  (list_rules — filter by parameter)\n"
    "- \"Which rule shows the highest SAR count?\" → threshold  (list_rules tool)\n\n"
    "Rules:\n"
    "- Output ONLY the label(s), comma-separated. No explanation, no punctuation other than commas.\n"
    "- A query can map to multiple labels (e.g. threshold,segmentation).\n"
    "- When in doubt between out_of_scope and a AML label, prefer the AML label."
)

# ---------------------------------------------------------------------------
# Pre-computed tool results
# Body constants (=== ... === block only) — used in assistant responses.
# Full constants (with "Tool result for ..." prefix) — used as tool content.
# ---------------------------------------------------------------------------

_RULE_LIST_BODY = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=487, SAR=82, FP=405, precision=16.8%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=312, SAR=41, FP=271, precision=13.1%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=14, SAR=3, FP=11, precision=21.4%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=2241, SAR=180, FP=2061, precision=8.0%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=623, SAR=87, FP=536, precision=13.6%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=701, SAR=94, FP=607, precision=11.8%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=58, SAR=21, FP=37, precision=36.2%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)

PC_LIST_RULES = "Tool result for list_rules:\n" + _RULE_LIST_BODY

_SAR_ACH_BODY = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Activity Deviation (ACH)\n"
    "Current condition: ACH transactions >= floor_amount AND >= z_threshold std dev above 90-day mean\n"
    "Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3.0)\n"
    "Current value: 3.0\n"
    "Labeled population: 487 customers (TP+FN pool=82 SAR, FP+TN pool=405 non-SAR, precision=16.8%)\n\n"
    "At the lowest value (0.00): TP=82, FP=405, FN=0, TN=0 (TP rate=100.0%, precision=16.8%).\n"
    "At current condition (3.00): TP=82, FP=405, FN=0, TN=0 (TP rate=100.0%, precision=16.8%).\n"
    "To keep TP rate >=90%: z_threshold <= 5.00 => TP=76, FP=298, FN=6, TN=107, precision=20.3%.\n"
    "To keep TP rate >=50%: z_threshold <= 9.00 => TP=44, FP=128, FN=38, TN=277, precision=25.6%.\n"
    "At the highest value (12.00): TP=28, FP=74, FN=54, TN=331 (TP rate=34.1%, precision=27.5%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)

PC_RULE_SAR_ACH = "Tool result for rule_sar_backtest:\n" + _SAR_ACH_BODY

_SAR_VS_BODY = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Current condition: Combined in+out pair total >= $20K AND ratio deviation <= 10%\n"
    "Sweep parameter: pair_total - Minimum combined in+out pair total (currently $20K)\n"
    "Current value: 20000\n"
    "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%)\n\n"
    "At the lowest value (5000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
    "At current condition (20000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
    "To keep TP rate >=90%: pair_total <= 35000.00 => TP=68, FP=286, FN=6, TN=118, precision=19.2%.\n"
    "To keep TP rate >=50%: pair_total <= 65000.00 => TP=40, FP=104, FN=34, TN=300, precision=27.8%.\n"
    "At the highest value (80000.00): TP=28, FP=62, FN=46, TN=342 (TP rate=37.8%, precision=31.1%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)

PC_RULE_SAR_VS = "Tool result for rule_sar_backtest:\n" + _SAR_VS_BODY

_2D_BIO_BODY = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Burst in Originator Activity\n"
    "Axis 1 (floor_amount): Minimum 30-day incoming transaction amount to trigger (currently $25K)\n"
    "Axis 2 (min_transactions): Minimum number of incoming transactions in window (currently 5)\n"
    "Grid: 9 x 5 = 45 combinations\n"
    "SAR pool: 87  Non-SAR pool: 536\n\n"
    "At current condition (floor_amount=25000, min_transactions=5): TP=87, FP=536, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=90%): floor_amount=35000, min_transactions=5 => TP=80, FP=412, FN=7, TN=124, TP rate=92.0%, precision=16.3%.\n"
    "Best FP reduction (TP rate >=50%): floor_amount=75000, min_transactions=7 => TP=46, FP=184, FN=41, TN=352, TP rate=52.9%, precision=20.0%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END PRE-COMPUTED 2D SWEEP ==="
)

PC_RULE_2D_BIO = "Tool result for rule_2d_sweep:\n" + _2D_BIO_BODY

_2D_ELDER_BODY = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Axis 1 (floor_amount): Minimum 14-day outgoing floor to trigger (currently $5K)\n"
    "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60 years)\n"
    "Grid: 9 x 15 = 135 combinations\n"
    "SAR pool: 59  Non-SAR pool: 341\n\n"
    "At current condition (floor_amount=5000, age_threshold=60): TP=59, FP=341, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=90%): floor_amount=6000, age_threshold=62 => TP=54, FP=261, FN=5, TN=80, TP rate=91.5%, precision=17.1%.\n"
    "Best FP reduction (TP rate >=50%): floor_amount=8000, age_threshold=66 => TP=30, FP=163, FN=29, TN=178, TP rate=50.8%, precision=15.5%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END PRE-COMPUTED 2D SWEEP ==="
)

PC_RULE_2D_ELDER = "Tool result for rule_2d_sweep:\n" + _2D_ELDER_BODY

# ---------------------------------------------------------------------------
# Cluster stats blocks (write_v38 convention: bold markdown, no === markers)
# These simulate last_assistant text that the orchestrator injects as context.
# ---------------------------------------------------------------------------

STATS_BIZ = """\
Segment: Business | Active accounts: 3,551 (excluded 0 with no transactions)
Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=48.3%, PC2=19.1%

**Cluster 1**
- Customers: **1,114** (31.4% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **30,487.4**
- Monthly Txn Volume ($): **300,181.8**
- Current Balance ($): **42,310.5**
- Account Age (years): **6.2**

**Cluster 2**
- Customers: **1,088** (30.6% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **27,690.0**
- Monthly Txn Volume ($): **246,301.4**
- Current Balance ($): **38,204.1**
- Account Age (years): **5.8**

**Cluster 3**
- Customers: **837** (23.6% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **24,505.9**
- Monthly Txn Volume ($): **189,046.9**
- Current Balance ($): **29,118.3**
- Account Age (years): **5.1**

**Cluster 4**
- Customers: **512** (14.4% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **17,441.0**
- Monthly Txn Volume ($): **121,217.7**
- Current Balance ($): **17,842.6**
- Account Age (years): **4.3**"""

STATS_IND = """\
Segment: Individual | Active accounts: 2,184 (excluded 0 with no transactions)
Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=44.1%, PC2=21.3%

**Cluster 1**
- Customers: **712** (32.6% of active accounts)
- Avg Weekly Transactions: **8.2**
- Avg Weekly Txn Amount ($): **12,340.5**
- Monthly Txn Volume ($): **98,724.0**
- Current Balance ($): **18,420.3**
- Account Age (years): **4.7**

**Cluster 2**
- Customers: **638** (29.2% of active accounts)
- Avg Weekly Transactions: **7.9**
- Avg Weekly Txn Amount ($): **9,875.2**
- Monthly Txn Volume ($): **78,001.6**
- Current Balance ($): **14,230.8**
- Account Age (years): **4.1**

**Cluster 3**
- Customers: **521** (23.9% of active accounts)
- Avg Weekly Transactions: **7.4**
- Avg Weekly Txn Amount ($): **7,210.0**
- Monthly Txn Volume ($): **57,680.0**
- Current Balance ($): **9,840.2**
- Account Age (years): **3.5**

**Cluster 4**
- Customers: **313** (14.3% of active accounts)
- Avg Weekly Transactions: **6.1**
- Avg Weekly Txn Amount ($): **4,320.7**
- Monthly Txn Volume ($): **34,565.6**
- Current Balance ($): **5,210.4**
- Account Age (years): **2.8**"""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

examples = []

# ═══════════════════════════════════════════════════════════════════════════
# GAP 1 — Lowest FP direction (X1–X4)
# Pattern: tool call → copy _RULE_LIST_BODY verbatim → ONE insight sentence (lowest-FP focus)
# ═══════════════════════════════════════════════════════════════════════════

# X1: "Which rule has the lowest FP rate?" — must pick MINIMUM, not maximum
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the lowest FP rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x1", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Among rules with meaningful alert volumes, Risky International Transfer has the lowest "
        "false positive count at 37 (precision=36.2%) — its 36% precision is 4.5× higher than CTR Client."
    )},
]})

# X2: "Which rule generates the fewest false positives?" — synonym for lowest FP
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule generates the fewest false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x2", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Excluding rules with no production data (alerts=0), Risky International Transfer generates "
        "the fewest false positives at 37 out of 58 alerts (precision=36.2%)."
    )},
]})

# X3: "Which rule has the best precision?" — precision = TP/(TP+FP), already in tool result
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the best precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x3", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Risky International Transfer has the highest precision at 36.2% among rules with "
        "meaningful alert volumes — 21 of its 58 alerts are confirmed SAR customers."
    )},
]})

# X4: "Which AML rule is the most efficient?"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which AML rule is the most efficient — least wasted investigator effort?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x4", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x4", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Risky International Transfer is the most efficient rule at 36.2% precision — "
        "investigators find a confirmed SAR in more than 1 in 3 alerts, versus 8.0% for CTR Client."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 2 — Precision synthesis: after list_rules, name the winner (X5–X8)
# Pattern: same as Gap 1 — copy _RULE_LIST_BODY verbatim → ONE insight sentence (highest-precision focus)
# ═══════════════════════════════════════════════════════════════════════════

# X5: "Which rule has the highest precision rate?" — must produce a non-empty answer after list_rules
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the highest precision rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x5", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x5", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Risky International Transfer has the highest precision at 36.2% among rules with active alerts — "
        "21 SAR customers out of 58 alerts, compared to the system low of 8.0% for CTR Client."
    )},
]})

# X6: "Rank all AML rules by precision from highest to lowest"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Rank all AML rules by precision from highest to lowest"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x6", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x6", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Ranked by precision descending: Risky International Transfer (36.2%) > "
        "Structuring Outgoing (21.4%) > AD ACH (16.8%) > Elder Abuse (16.4%) > "
        "Velocity Single (15.5%) > BIO (13.6%) > AD Check (13.1%) > DETA (12.9%) > "
        "BBA (11.8%) > CTR Client (8.0%), with five rules at n/a (no production data)."
    )},
]})

# X7: "Which rule has the best true positive ratio?" — precision synonym
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the best true positive ratio?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x7", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x7", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "True positive ratio equals precision; Risky International Transfer leads at 36.2% "
        "(21 SAR, 37 FP, 58 total alerts) — nearly 4.5× higher than CTR Client (8.0%)."
    )},
]})

# X8: Confirms lowest FP rate = highest precision, then names the winner
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule shows the lowest false positive rate? That’s the same as highest precision right?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("x8", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "x8", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BODY + "\n\n"
        "Correct — lowest FP rate equals highest precision; "
        "Risky International Transfer has both the fewest false positives (37) and the highest precision (36.2%) "
        "among rules with meaningful production data."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 3 — Multi-turn data isolation: don’t copy prior tool results (X9–X12)
# Pattern: prior turn in history → new query → call CORRECT fresh tool → copy new pc_body verbatim
# ═══════════════════════════════════════════════════════════════════════════

# X9: Prior = Elder Abuse 2D sweep; Current = SAR backtest for AD ACH
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run a 2D sweep for Elder Abuse varying floor amount and age threshold"},
    {"role": "assistant", "content": "The best trade-off in the Elder Abuse 2D sweep is floor_amount=8,000"},
    {"role": "user",      "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x9", "rule_sar_backtest", {"risk_factor": "Activity Deviation ACH"})
    ]},
    {"role": "tool", "tool_call_id": "x9", "content": PC_RULE_SAR_ACH},
    {"role": "assistant", "content": (
        _SAR_ACH_BODY + "\n\n"
        "Raising z_threshold to 5.00 cuts false positives by 26% (405 to 298) while retaining "
        "92.7% of the 82 SAR customers."
    )},
]})

# X10: Prior = Elder Abuse sweep; Current = SAR backtest for Velocity Single
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run a 2D sweep for Elder Abuse varying floor amount and age threshold"},
    {"role": "assistant", "content": "The best trade-off: floor_amount=8,000, age_threshold=66, TP=30, FP=163"},
    {"role": "user",      "content": "Now show me the SAR backtest for Velocity Single rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x10", "rule_sar_backtest", {"risk_factor": "Velocity Single"})
    ]},
    {"role": "tool", "tool_call_id": "x10", "content": PC_RULE_SAR_VS},
    {"role": "assistant", "content": (
        _SAR_VS_BODY + "\n\n"
        "Raising the pair_total to $35K retains 91.9% of the 74 SAR customers while cutting "
        "false positives by 29% (404 to 286)."
    )},
]})

# X11: Prior = CTR Client SAR backtest; Current = 2D sweep for Burst in Originator
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run SAR backtest for CTR Client rule"},
    {"role": "assistant", "content": "CTR Client: TP=180, FP=2,061, precision=8.0% at current $10K floor."},
    {"role": "user",      "content": "Show 2D sweep for Burst in Originator Activity"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x11", "rule_2d_sweep", {
            "risk_factor": "Burst in Originator Activity",
            "sweep_param_1": "floor_amount",
            "sweep_param_2": "min_transactions",
        })
    ]},
    {"role": "tool", "tool_call_id": "x11", "content": PC_RULE_2D_BIO},
    {"role": "assistant", "content": (
        _2D_BIO_BODY + "\n\n"
        "Raising the 30-day floor to $35K retains 92% of the 87 SAR customers while cutting "
        "false positives by 23% (536 to 412)."
    )},
]})

# X12: Prior = list_rules summary; Current = 2D sweep for Elder Abuse
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rules generate the most false positives?"},
    {"role": "assistant", "content": "CTR Client leads with FP=2,061, followed by Elder Abuse FP=958."},
    {"role": "user",      "content": "Run a 2D sweep for Elder Abuse varying floor amount and age threshold"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x12", "rule_2d_sweep", {
            "risk_factor": "Elder Abuse",
            "sweep_param_1": "floor_amount",
            "sweep_param_2": "age_threshold",
        })
    ]},
    {"role": "tool", "tool_call_id": "x12", "content": PC_RULE_2D_ELDER},
    {"role": "assistant", "content": (
        _2D_ELDER_BODY + "\n\n"
        "Raising the floor to $6K and the age threshold to 62 retains 91.5% of the 59 SAR "
        "customers while cutting false positives by 23% (341 to 261)."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 4 — Cluster stats hallucination: read from context, no tool call (X13–X16)
# Pattern: prev_context(STATS_*) injected in user message → answer directly, NO tool call
# (Rule 15 of SEGMENTATION_SYSTEM: answer from [PREVIOUS CLUSTERING RESULT] WITHOUT calling any tool)
# ═══════════════════════════════════════════════════════════════════════════

# X13: Business clusters in context; "how about cluster 3?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\nhow about cluster 3?"},
    {"role": "assistant", "content": (
        "Cluster 3 contains **837 Business customers** (23.6% of active accounts).\n\n"
        "- Avg Weekly Txn Amount: $24,505.9\n"
        "- Monthly Txn Volume: $189,046.9\n"
        "- Current Balance: $29,118.3\n"
        "- Account Age: 5.1 years\n\n"
        "Cluster 3 sits in the middle tier — its $189K monthly volume is below Clusters 1 and 2 "
        "but above Cluster 4, representing established mid-tier Business accounts with moderate AML exposure."
    )},
]})

# X14: Individual clusters in context; "tell me about cluster 4"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_IND) + "\n\ntell me about cluster 4"},
    {"role": "assistant", "content": (
        "Cluster 4 contains **313 Individual customers** (14.3% of active accounts).\n\n"
        "- Avg Weekly Transactions: 6.1\n"
        "- Avg Weekly Txn Amount: $4,320.7\n"
        "- Monthly Txn Volume: $34,565.6\n"
        "- Current Balance: $5,210.4\n"
        "- Account Age: 2.8 years\n\n"
        "Cluster 4 is the lowest-activity segment — fewest transactions, lowest balances, "
        "and shortest account tenure; these 313 newer low-volume accounts present the lowest "
        "transaction-based AML risk of the four Individual clusters."
    )},
]})

# X15: Business clusters in context; "what about cluster 2?"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\nwhat about cluster 2?"},
    {"role": "assistant", "content": (
        "Cluster 2 contains **1,088 Business customers** (30.6% of active accounts).\n\n"
        "- Avg Weekly Txn Amount: $27,690.0\n"
        "- Monthly Txn Volume: $246,301.4\n"
        "- Current Balance: $38,204.1\n"
        "- Account Age: 5.8 years\n\n"
        "Cluster 2 closely mirrors Cluster 1 in size and behaviour — similar age (5.8 vs 6.2 years) "
        "and balance ($38K vs $42K) — together Clusters 1 and 2 represent 62% of active Business accounts "
        "and the highest aggregate transaction activity."
    )},
]})

# X16: Business clusters in context; "cluster 1 characteristics please"
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\ncluster 1 characteristics please"},
    {"role": "assistant", "content": (
        "Cluster 1 is the largest group with **1,114 Business customers** (31.4% of active accounts).\n\n"
        "- Avg Weekly Txn Amount: $30,487.4\n"
        "- Monthly Txn Volume: $300,181.8\n"
        "- Current Balance: $42,310.5\n"
        "- Account Age: 6.2 years\n\n"
        "Cluster 1 has the highest transaction volumes and balances of all four groups — "
        "these are the most established, highest-activity Business accounts and the primary "
        "candidates for velocity, structuring, and elder abuse rule monitoring."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 5 — Non-AML OFAC/sanctions context → out_of_scope (classifier, X17–X20)
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "My dog OFAC met a cat called sanctions"},
    {"role": "assistant", "content": "out_of_scope"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "ARIA sanctions my morning coffee ritual"},
    {"role": "assistant", "content": "out_of_scope"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "My friend BSA went to the bank yesterday to deposit money"},
    {"role": "assistant", "content": "out_of_scope"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "The AML club at my school meets every Thursday for practice"},
    {"role": "assistant", "content": "out_of_scope"},
]})

# ---------------------------------------------------------------------------
# Combine V39 base + V40 new examples and write
# ---------------------------------------------------------------------------

def main():
    # Write V40-only file
    with open(V40_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V40] V40-only: {V40_ONLY_PATH.name} ({len(examples)} examples)")

    # Attempt to combine with V39 base
    if V39_BASE_PATH.exists():
        v39_base = []
        with open(V39_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v39_base.append(json.loads(line))
        print(f"[V40] Loaded {len(v39_base)} base examples from {V39_BASE_PATH.name}")
        all_examples = v39_base + examples
        with open(V40_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V40] Combined: {V40_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V40] WARNING: V39 base not found at {V39_BASE_PATH}")
        print(f"[V40] V40-only file written — combine manually if needed.")


if __name__ == "__main__":
    main()

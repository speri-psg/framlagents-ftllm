"""
Generate V6 training examples — fix remaining V4 gaps identified in production testing.

Gaps covered:
  17/22  — Fabricates FP/FN for sweep points not in computed range
  24     — Accepts invalid threshold_tuning parameters (threshold_min, threshold_step)
  26/27  — CFR/statute citations after KB-miss disclaimer
  28     — Wrong inline number in cluster summary sentence (10x scale error)
  29     — Wrong threshold_tuning params in cluster-to-threshold redirect
  31     — Doesn't call alerts_distribution; gives text instead of tool call
  New-A  — Echoes DISPLAY_CLUSTERS rule text verbatim in response
  New-B  — Correct SEG_STATS format (no trailing instruction text)
"""

import json, pathlib

OUT = pathlib.Path(__file__).parent / "data" / "framl_train_failures_v5.jsonl"

# ── System prompts (current production versions) ──────────────────────────────

SYS_THRESHOLD = (
    "You are a FRAML threshold tuning specialist. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change. "
    "IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
    "DEFINITIONS (always apply these exactly — do not contradict them):\n"
    "- False Positive (FP): an alert that fires on a non-suspicious transaction. HIGHER threshold → FEWER FPs.\n"
    "- False Negative (FN): a suspicious transaction that did NOT trigger an alert. HIGHER threshold → MORE FNs.\n"
    "- Crossover: the threshold where FP and FN counts are closest — the optimal operating point.\n"
    "- Raising the threshold reduces investigator workload (fewer alerts) but risks missing SAR-worthy activity.\n"
    "- Lowering the threshold catches more suspicious activity but overwhelms investigators with false alarms.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory.\n"
    "2. For any question about FP, FN, threshold, alert rates, or transactions — call threshold_tuning.\n"
    "3. For general segment counts or totals — call segment_stats.\n"
    "4. For any question about SAR catch rate, SAR detection, how many SARs a threshold catches, "
    "or SAR backtest — call sar_backtest.\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "6. segment must be exactly one of: Business, Individual\n"
    "7. If the user does not specify a segment, default to Business.\n"
    "8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.\n"
    "9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy "
    "that section word-for-word into your response. Do NOT change any numbers, thresholds, or "
    "directional statements. After copying it, add ONE sentence of AML domain insight.\n"
    "10. Do NOT paraphrase, round, or restate the numbers differently.\n"
    "11. Do NOT include JSON or code blocks in your final reply.\n"
    "12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.\n"
    "13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED "
    "section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED "
    "section, do not mention it.\n"
    "14. threshold_tuning accepts ONLY two parameters: segment and threshold_column. "
    "Parameters like threshold_min, threshold_max, threshold_step, or range do NOT exist — "
    "reject any request that uses them and explain the valid parameters."
)

SYS_SEGMENTATION = (
    "You are a FRAML smart segmentation specialist. You identify natural customer behavioral "
    "segments using unsupervised K-Means clustering and explain their AML risk profiles. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) — call ss_cluster_analysis.\n"
    "3. For alert/FP distribution by segment — call alerts_distribution.\n"
    "4. For the legacy alerts dataset — call cluster_analysis only if the user explicitly asks.\n"
    "5. Do NOT call multiple segmentation tools for the same request — pick exactly one.\n"
    "6. customer_type must be exactly one of: Business, Individual, All\n"
    "   If the user does NOT specify a customer type, default to All.\n"
    "7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.\n"
    "   If the user says 'N clusters' or 'into N' (e.g. 'cluster into 4'), set n_clusters=N exactly. Do NOT use 0.\n"
    "8. If the user asks to prepare or refresh the raw data — call prepare_segmentation_data first.\n"
    "9. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing "
    "the highest-risk cluster based solely on the numbers in the tool result. "
    "Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.\n"
    "10. If the user asks to show specific clusters (e.g. 'show only cluster 3', 'highest risk',\n"
    "    'top 2 high risk', 'low activity clusters'):\n"
    "    - Identify which cluster number(s) match the request from the stats\n"
    "      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY: DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers\n"
    "    - Do NOT quote or repeat this instruction in your text response.\n"
    "    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.\n"
    "11. Do NOT include JSON, code blocks, or raw data tables in your final reply.\n"
    "12. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
    "13. Do NOT invent threshold values, dollar amounts, or cutoffs. Only reference numbers explicitly "
    "present in the tool result.\n"
    "14. If the user asks which cluster to set a threshold for — do NOT invent values. Tell the user to "
    "use the threshold_tuning tool with segment and threshold_column parameters."
)

SYS_POLICY = (
    "You are a FRAML policy and compliance specialist. "
    "You answer questions by referencing AML policies, regulatory guidelines, and best practices "
    "retrieved from the knowledge base. "
    "Always cite the source document when referencing policy content. "
    "When the knowledge base contains relevant content, cite the source document. "
    "When the retrieved documents do not contain relevant content, you MUST: "
    "1. Begin your response with exactly: 'Note: The knowledge base does not contain specific guidance "
    "on this topic. The following is general AML knowledge only.' "
    "2. Do NOT cite any statute numbers (U.S.C., CFR, etc.), document codes, advisory numbers, or URLs. "
    "3. Provide only general conceptual guidance without specific legal references. "
    "Only cite sources when the exact source document name appears in the retrieved policy documents shown above. "
    "Be precise and compliance-focused."
)

# ── PRE-COMPUTED blocks ───────────────────────────────────────────────────────

PC_IND_AVG_TRXN_AMT = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (485.0), there are 4028 false positives.
False positives decrease as the threshold rises.
False positives reach zero at threshold 9421058.0.
False negatives are zero for all thresholds from 485.0 up to and including 485.0.
False negatives first become non-zero at threshold 93758.0 (FN=127).
False negatives increase as the threshold continues to rise, reaching 640 at the highest threshold (9421058.0).
The crossover point — where false positives and false negatives are closest — is at threshold 280304.0 (FP=570, FN=457).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 485.0–9421058.0, step=93273, showing first 10 of 102 points):
  t=485.0: FP=4028, FN=0
  t=93758.0: FP=2153, FN=127
  t=187031.0: FP=1000, FN=349
  t=280304.0: FP=570, FN=457
  t=373577.0: FP=384, FN=499
  t=466850.0: FP=283, FN=538
  t=560123.0: FP=219, FN=555
  t=653396.0: FP=166, FN=568
  t=746669.0: FP=128, FN=584
  t=839942.0: FP=105, FN=595
  ... (92 additional points not shown)"""

PC_IND_TRXN_AMT_MONTHLY = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (485.0), there are 4028 false positives.
False positives decrease as the threshold rises.
False positives reach zero at threshold 22378954.0.
False negatives are zero for all thresholds from 485.0 up to and including 485.0.
False negatives first become non-zero at threshold 222054.0 (FN=14).
False negatives increase as the threshold continues to rise, reaching 640 at the highest threshold (22378954.0).
The crossover point — where false positives and false negatives are closest — is at threshold 1329899.0 (FP=542, FN=511).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 485.0–22378954.0, step=221569, showing first 10 of 102 points):
  t=485.0: FP=4028, FN=0
  t=222054.0: FP=3497, FN=14
  t=443623.0: FP=2473, FN=91
  t=665192.0: FP=1639, FN=197
  t=886761.0: FP=1145, FN=296
  t=1108330.0: FP=790, FN=441
  t=1329899.0: FP=542, FN=511
  t=1551468.0: FP=367, FN=537
  t=1773037.0: FP=254, FN=561
  t=1994606.0: FP=186, FN=588
  ... (92 additional points not shown)"""

PC_BIZ_AVG_TRXN_AMT = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (266.38), there are 323 false positives.
False positives decrease as the threshold rises.
False positives do not reach zero within the threshold range (266.38–45725188.38).
False negatives are zero for all thresholds from 266.38 up to and including 266.38.
False negatives first become non-zero at threshold 452988.38 (FN=23).
False negatives increase as the threshold continues to rise, reaching 44 at the highest threshold (45725188.38).
The crossover point — where false positives and false negatives are closest — is at threshold 905710.38 (FP=33, FN=30).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 266.38–45725188.38, step=452722, showing first 10 of 102 points):
  t=266.38: FP=323, FN=0
  t=452988.38: FP=45, FN=23
  t=905710.38: FP=33, FN=30
  t=1358432.38: FP=25, FN=32
  t=1811154.38: FP=21, FN=32
  t=2263876.38: FP=15, FN=32
  t=2716598.38: FP=12, FN=33
  t=3169320.38: FP=9, FN=33
  t=3622042.38: FP=8, FN=33
  t=4074764.38: FP=8, FN=34
  ... (92 additional points not shown)"""

SEG_STATS = """\
=== PRE-COMPUTED SEGMENT STATS (copy verbatim, do not compute new numbers) ===
Business: accounts=367 (7.3% of total), alerts=355 (7.3% of all alerts), FP=323 (FP rate=91.0% of alerts), FN=44
Individual: accounts=4,668 (92.7% of total), alerts=4,480 (92.7% of all alerts), FP=4,028 (FP rate=89.9% of alerts), FN=640
=== END PRE-COMPUTED SEGMENT STATS ==="""

# Real cluster stats from live tool output (corrected from 10x error in earlier training data)
BIZ_4CLUSTER = """\
Segment: Business | Active accounts: 14,423 (878,207 excluded with no transactions)
Clusters: 4 | Features: 7 numeric + 8 encoded categorical
PCA variance explained: PC1=18.9%, PC2=16.6%

  Cluster 1: n=1,242 (8.6% of active accounts), avg_num_trxns=0.6, avg_weekly_trxn_amt=59380.3, trxn_amt_monthly=137374.5, INCOME=nan, CURRENT_BALANCE=77116508.8, ACCT_AGE_YEARS=10.8, AGE=nan
  Cluster 2: n=12,506 (86.7% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=36519.0, trxn_amt_monthly=140904.0, INCOME=0.0, CURRENT_BALANCE=1540453.6, ACCT_AGE_YEARS=5.8, AGE=nan
  Cluster 3: n=32 (0.2% of active accounts), avg_num_trxns=0.5, avg_weekly_trxn_amt=38458.3, trxn_amt_monthly=102282.7, INCOME=nan, CURRENT_BALANCE=3524868.0, ACCT_AGE_YEARS=4.7, AGE=nan
  Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=1.0, avg_weekly_trxn_amt=783507.6, trxn_amt_monthly=915021.7, INCOME=nan, CURRENT_BALANCE=2961738.9, ACCT_AGE_YEARS=5.9, AGE=nan
HIGHEST-RISK CLUSTER: Cluster 4 (n=643, avg_num_trxns=1.0) is the highest-risk segment by transaction frequency and volume."""

IND_4CLUSTER = """\
Segment: Individual | Active accounts: 198,185 (680,022 excluded with no transactions)
Clusters: 4 | Features: 7 numeric + 12 encoded categorical
PCA variance explained: PC1=22.1%, PC2=13.4%

  Cluster 1: n=8,432 (4.3% of active accounts), avg_num_trxns=3.2, avg_weekly_trxn_amt=1250.0, trxn_amt_monthly=4820.0, INCOME=52000.0, AGE=34.2, CURRENT_BALANCE=8420.0, ACCT_AGE_YEARS=3.1
  Cluster 2: n=18,923 (9.6% of active accounts), avg_num_trxns=1.8, avg_weekly_trxn_amt=620.0, trxn_amt_monthly=2180.0, INCOME=38000.0, AGE=45.1, CURRENT_BALANCE=4100.0, ACCT_AGE_YEARS=6.2
  Cluster 3: n=7,241 (3.7% of active accounts), avg_num_trxns=6.4, avg_weekly_trxn_amt=4830.0, trxn_amt_monthly=18920.0, INCOME=74000.0, AGE=38.7, CURRENT_BALANCE=31200.0, ACCT_AGE_YEARS=5.8
  Cluster 4: n=163,589 (82.5% of active accounts), avg_num_trxns=0.8, avg_weekly_trxn_amt=410.0, trxn_amt_monthly=1640.0, INCOME=31000.0, AGE=41.8, CURRENT_BALANCE=2100.0, ACCT_AGE_YEARS=4.9
HIGHEST-RISK CLUSTER: Cluster 3 (n=7,241, avg_num_trxns=6.4) is the highest-risk segment by transaction frequency and weekly volume."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def ex(system, messages):
    return {"messages": [{"role": "system", "content": system}] + messages}

def tc(call_id, name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}

def tr(call_id, content):
    return {"role": "tool", "content": content, "tool_call_id": call_id}

def user(text):
    return {"role": "user", "content": text}

def asst(text):
    return {"role": "assistant", "content": text}

# ── Examples ──────────────────────────────────────────────────────────────────

examples = [

    # ── Gap 17/22: Refuse to fabricate FP/FN for points not in sweep ──────────

    # ex84
    ex(SYS_THRESHOLD, [
        user("Run a threshold sweep for Individual AVG_TRXN_AMT."),
        tc("t84", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t84", PC_IND_AVG_TRXN_AMT),
        asst(PC_IND_AVG_TRXN_AMT + "\n\nThe crossover at 280,304 (FP=570, FN=457) is the optimal operating point for Individual average transaction amount thresholds."),
    ]),

    # ex85 — follow-up asking for a point not in the sweep
    ex(SYS_THRESHOLD, [
        user("Run a threshold sweep for Individual AVG_TRXN_AMT."),
        tc("t85a", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t85a", PC_IND_AVG_TRXN_AMT),
        asst(PC_IND_AVG_TRXN_AMT + "\n\nThe crossover at 280,304 (FP=570, FN=457) is the optimal operating point for Individual average transaction amount thresholds."),
        user("What would FP be at threshold 175000?"),
        asst("Threshold 175,000 is not a computed sweep point. The nearest points in the sweep are t=93,758 (FP=2,153, FN=127) and t=187,031 (FP=1,000, FN=349). To get the exact figure at 175,000, re-run the sweep with a tighter range centered on that value."),
    ]),

    # ex86 — asking for a point not in sweep (monthly)
    ex(SYS_THRESHOLD, [
        user("Show threshold tuning for Individual monthly transaction amount."),
        tc("t86a", "threshold_tuning", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t86a", PC_IND_TRXN_AMT_MONTHLY),
        asst(PC_IND_TRXN_AMT_MONTHLY + "\n\nThe crossover at 1,329,899 (FP=542, FN=511) is the recommended starting point for Individual monthly transaction amount threshold calibration."),
        user("What is FP at 500000?"),
        asst("Threshold 500,000 is not a computed sweep point. The nearest points in the sweep are t=443,623 (FP=2,473, FN=91) and t=665,192 (FP=1,639, FN=197). To get the exact figure at 500,000, re-run the sweep with a tighter range around that value."),
    ]),

    # ex87 — asking for a point not in sweep (Business)
    ex(SYS_THRESHOLD, [
        user("Run threshold sweep for Business AVG_TRXN_AMT."),
        tc("t87a", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t87a", PC_BIZ_AVG_TRXN_AMT),
        asst(PC_BIZ_AVG_TRXN_AMT + "\n\nThe crossover at 905,710 (FP=33, FN=30) is the optimal operating point for Business average transaction amount thresholds."),
        user("What would FP be at 200000?"),
        asst("Threshold 200,000 is not a computed sweep point. The nearest points in the sweep are t=266.38 (FP=323, FN=0) and t=452,988 (FP=45, FN=23). To get the exact figure at 200,000, re-run the sweep with a tighter range centered on that value."),
    ]),

    # ex88 — direct question about a threshold not yet swept
    ex(SYS_THRESHOLD, [
        user("What FP would I get at threshold 1000000 for Individual monthly transaction amount?"),
        tc("t88", "threshold_tuning", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t88", PC_IND_TRXN_AMT_MONTHLY),
        asst(PC_IND_TRXN_AMT_MONTHLY + "\n\nThreshold 1,000,000 is not a computed sweep point; the nearest points are t=886,761 (FP=1,145, FN=296) and t=1,108,330 (FP=790, FN=441) — re-run with a tighter range to get the exact figure."),
    ]),

    # ── Gap 24: Reject invalid threshold_tuning parameters ────────────────────

    # ex89
    ex(SYS_THRESHOLD, [
        user("Run threshold sweep with threshold_min=5000 and threshold_step=1000."),
        asst("threshold_min and threshold_step are not valid parameters for threshold_tuning. "
             "The tool accepts exactly two parameters: segment (Business or Individual) and "
             "threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
             "Please restate your request with those parameters."),
    ]),

    # ex90
    ex(SYS_THRESHOLD, [
        user("Can you sweep thresholds from 10000 to 500000 with a step of 5000?"),
        asst("threshold_tuning does not accept a range or step size. "
             "The tool automatically determines the sweep range from your data. "
             "Valid parameters are segment (Business or Individual) and "
             "threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
             "Please specify those and the tool will run the sweep."),
    ]),

    # ex91
    ex(SYS_THRESHOLD, [
        user("Run a sweep for Business with threshold_max=100000."),
        asst("threshold_max is not a valid parameter for threshold_tuning. "
             "The valid parameters are segment (Business or Individual) and "
             "threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
             "The sweep range is computed automatically from your data."),
    ]),

    # ── Gap 26/27: No CFR citations after KB-miss disclaimer ──────────────────

    # ex92
    ex(SYS_POLICY, [
        user("What does AML policy say about structuring detection thresholds?"),
        asst("Note: The knowledge base does not contain specific guidance on this topic. "
             "The following is general AML knowledge only.\n\n"
             "Structuring detection thresholds are typically set just below the $10,000 Currency "
             "Transaction Report trigger to identify customers who may be breaking up transactions "
             "to avoid mandatory reporting. Risk-based programs calibrate these thresholds to the "
             "institution's customer mix and transaction profile."),
    ]),

    # ex93
    ex(SYS_POLICY, [
        user("What is the CTR reporting threshold?"),
        asst("Note: The knowledge base does not contain specific guidance on this topic. "
             "The following is general AML knowledge only.\n\n"
             "The Currency Transaction Report threshold is $10,000. Financial institutions are "
             "required to file a CTR for cash transactions exceeding that amount in a single business day."),
    ]),

    # ex94
    ex(SYS_POLICY, [
        user("What are the rules around SAR filing obligations?"),
        asst("Note: The knowledge base does not contain specific guidance on this topic. "
             "The following is general AML knowledge only.\n\n"
             "Suspicious Activity Reports must be filed when a transaction involves $5,000 or more "
             "and the institution knows, suspects, or has reason to suspect the funds involve illegal "
             "activity, structuring, or have no lawful purpose. The decision is based on the "
             "institution's risk-based assessment of the activity."),
    ]),

    # ── Gap 28: Correct inline numbers in cluster summary ────────────────────

    # ex95 — Business 4-cluster: correct highest-risk identification
    ex(SYS_SEGMENTATION, [
        user("Cluster Business customers into 4 segments."),
        tc("t95", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t95", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 (n=643, avg_num_trxns=1.0, avg_weekly_trxn_amt=783,507.6) is the highest-risk segment by both transaction frequency and weekly volume across all four clusters."),
    ]),

    # ex96 — Business 4-cluster: different user phrasing
    ex(SYS_SEGMENTATION, [
        user("Show me 4 behavioral clusters for Business customers."),
        tc("t96", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t96", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 has the highest avg_num_trxns (1.0) and avg_weekly_trxn_amt (783,507.6) of all four clusters, making it the primary focus for Business AML review."),
    ]),

    # ex97 — Individual 4-cluster: correct highest-risk
    ex(SYS_SEGMENTATION, [
        user("Cluster Individual customers into 4 groups."),
        tc("t97", "ss_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4}),
        tr("t97", IND_4CLUSTER),
        asst(IND_4CLUSTER + "\n\nCluster 3 (n=7,241, avg_num_trxns=6.4, avg_weekly_trxn_amt=4,830.0) is the highest-risk segment by transaction frequency and weekly volume across all four Individual clusters."),
    ]),

    # ex98 — Business 4-cluster: user asks which cluster is riskiest
    ex(SYS_SEGMENTATION, [
        user("Which Business cluster is the highest risk?"),
        tc("t98", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t98", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 is the highest-risk segment with avg_num_trxns=1.0 and avg_weekly_trxn_amt=783,507.6 — the largest values in both metrics across all four clusters."),
    ]),

    # ex99 — Business 4-cluster: show highest risk cluster only
    ex(SYS_SEGMENTATION, [
        user("Show me only the highest risk Business cluster."),
        tc("t99", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t99", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 (avg_num_trxns=1.0, avg_weekly_trxn_amt=783,507.6) is the highest-risk segment and the primary focus for Business threshold review.\nDISPLAY_CLUSTERS: 4"),
    ]),

    # ── Gap 29: Correct threshold redirect with right params ──────────────────

    # ex100
    ex(SYS_SEGMENTATION, [
        user("Cluster Business customers into 4 segments."),
        tc("t100a", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t100a", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 (n=643, avg_num_trxns=1.0, avg_weekly_trxn_amt=783,507.6) is the highest-risk segment by both transaction frequency and weekly volume across all four clusters."),
        user("Which cluster should I set the threshold for?"),
        asst("I cannot recommend a specific threshold per cluster. Use the threshold_tuning tool "
             "with segment=Business and a threshold_column (AVG_TRXN_AMT, TRXN_AMT_MONTHLY, or "
             "AVG_NUM_TRXNS) to find the optimal FP/FN operating point for the Business segment."),
    ]),

    # ex101
    ex(SYS_SEGMENTATION, [
        user("Cluster Business customers into 4 segments."),
        tc("t101a", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t101a", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 (n=643, avg_num_trxns=1.0, avg_weekly_trxn_amt=783,507.6) is the highest-risk segment by both transaction frequency and weekly volume across all four clusters."),
        user("What threshold should I apply to Cluster 4?"),
        asst("I cannot recommend a threshold for a specific cluster. To find the optimal threshold "
             "for Business customers, use threshold_tuning with segment=Business and "
             "threshold_column set to AVG_TRXN_AMT or TRXN_AMT_MONTHLY."),
    ]),

    # ex102
    ex(SYS_SEGMENTATION, [
        user("What threshold is best for the high-risk Business cluster?"),
        asst("I cannot recommend a threshold per cluster. Use the threshold_tuning tool "
             "with segment=Business and threshold_column (AVG_TRXN_AMT, TRXN_AMT_MONTHLY, or "
             "AVG_NUM_TRXNS) to derive the optimal FP/FN operating point for Business customers."),
    ]),

    # ── Gap 31: Call alerts_distribution instead of describing it ─────────────

    # ex103
    ex(SYS_SEGMENTATION, [
        user("Which segment has the most alerts?"),
        tc("t103", "alerts_distribution", {}),
        tr("t103", SEG_STATS),
        asst(SEG_STATS + "\n\nIndividual accounts generate 92.7% of all alerts and 92.7% of false positives, far exceeding the Business segment in absolute volume."),
    ]),

    # ex104
    ex(SYS_SEGMENTATION, [
        user("Show me the alerts distribution across Business and Individual."),
        tc("t104", "alerts_distribution", {}),
        tr("t104", SEG_STATS),
        asst(SEG_STATS + "\n\nIndividual accounts dominate in both total alerts (4,480 vs 355) and false positives (4,028 vs 323), reflecting their much larger share of the customer population."),
    ]),

    # ex105
    ex(SYS_SEGMENTATION, [
        user("How are false positives distributed between segments?"),
        tc("t105", "alerts_distribution", {}),
        tr("t105", SEG_STATS),
        asst(SEG_STATS + "\n\nBoth segments show similar FP rates (91.0% for Business, 89.9% for Individual), but Individual generates 4,028 FPs versus 323 for Business due to its far larger account base."),
    ]),

    # ex106
    ex(SYS_SEGMENTATION, [
        user("Which segment generates more false positives?"),
        tc("t106", "alerts_distribution", {}),
        tr("t106", SEG_STATS),
        asst(SEG_STATS + "\n\nIndividual generates 4,028 false positives versus 323 for Business — 12.5 times more in absolute terms — proportional to its 92.7% share of total accounts."),
    ]),

    # ── New-A: DISPLAY_CLUSTERS — silent, no rule echo ─────────────────────────

    # ex107 — user asks for all clusters (no filter) → no DISPLAY_CLUSTERS line
    ex(SYS_SEGMENTATION, [
        user("Cluster all customers and show the treemap."),
        tc("t107", "ss_cluster_analysis", {"customer_type": "All", "n_clusters": 4}),
        tr("t107", """\
Segment: All | Active accounts: 212,608 (878,207 excluded with no transactions)
Clusters: 4 | Features: 6 numeric + 10 encoded categorical
PCA variance explained: PC1=21.3%, PC2=14.8%

  Cluster 1: n=5,197 (2.4% of active accounts), avg_num_trxns=1.1, avg_weekly_trxn_amt=905773.0, trxn_amt_monthly=978336.5, CURRENT_BALANCE=2841200.0, ACCT_AGE_YEARS=6.8
  Cluster 2: n=84,312 (39.7% of active accounts), avg_num_trxns=0.3, avg_weekly_trxn_amt=28400.0, trxn_amt_monthly=98200.0, CURRENT_BALANCE=1240000.0, ACCT_AGE_YEARS=4.9
  Cluster 3: n=91,843 (43.2% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=31200.0, trxn_amt_monthly=112400.0, CURRENT_BALANCE=980000.0, ACCT_AGE_YEARS=5.2
  Cluster 4: n=31,256 (14.7% of active accounts), avg_num_trxns=0.6, avg_weekly_trxn_amt=52100.0, trxn_amt_monthly=184300.0, CURRENT_BALANCE=3120000.0, ACCT_AGE_YEARS=7.1
HIGHEST-RISK CLUSTER: Cluster 1 (n=5,197, avg_num_trxns=1.1) is the highest-risk segment by transaction frequency and weekly volume."""),
        asst("""\
Segment: All | Active accounts: 212,608 (878,207 excluded with no transactions)
Clusters: 4 | Features: 6 numeric + 10 encoded categorical
PCA variance explained: PC1=21.3%, PC2=14.8%

  Cluster 1: n=5,197 (2.4% of active accounts), avg_num_trxns=1.1, avg_weekly_trxn_amt=905773.0, trxn_amt_monthly=978336.5, CURRENT_BALANCE=2,841,200.0, ACCT_AGE_YEARS=6.8
  Cluster 2: n=84,312 (39.7% of active accounts), avg_num_trxns=0.3, avg_weekly_trxn_amt=28400.0, trxn_amt_monthly=98200.0, CURRENT_BALANCE=1,240,000.0, ACCT_AGE_YEARS=4.9
  Cluster 3: n=91,843 (43.2% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=31200.0, trxn_amt_monthly=112400.0, CURRENT_BALANCE=980,000.0, ACCT_AGE_YEARS=5.2
  Cluster 4: n=31,256 (14.7% of active accounts), avg_num_trxns=0.6, avg_weekly_trxn_amt=52100.0, trxn_amt_monthly=184300.0, CURRENT_BALANCE=3,120,000.0, ACCT_AGE_YEARS=7.1
HIGHEST-RISK CLUSTER: Cluster 1 (n=5,197, avg_num_trxns=1.1) is the highest-risk segment by transaction frequency and weekly volume.

Cluster 1 has the highest avg_weekly_trxn_amt (905,773.0) and avg_num_trxns (1.1) of all four clusters."""),
    ]),

    # ex108 — user asks for highest risk cluster → DISPLAY_CLUSTERS silently
    ex(SYS_SEGMENTATION, [
        user("Show me only the highest risk Business cluster."),
        tc("t108", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t108", BIZ_4CLUSTER),
        asst(BIZ_4CLUSTER + "\n\nCluster 4 (avg_num_trxns=1.0, avg_weekly_trxn_amt=783,507.6) has the highest transaction activity of all four Business clusters.\nDISPLAY_CLUSTERS: 4"),
    ]),

    # ex109 — user asks for top 2 high risk → DISPLAY_CLUSTERS silently
    ex(SYS_SEGMENTATION, [
        user("Show the top 2 highest-risk Individual clusters."),
        tc("t109", "ss_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4}),
        tr("t109", IND_4CLUSTER),
        asst(IND_4CLUSTER + "\n\nClusters 3 and 1 have the highest transaction frequency and weekly volume among all four Individual clusters.\nDISPLAY_CLUSTERS: 3,1"),
    ]),

]

# ── Write output ──────────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e) + "\n")

print(f"Wrote {len(examples)} examples to {OUT}")

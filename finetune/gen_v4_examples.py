"""
Generate V3 gap training examples (ex37–ex50).

Gaps addressed:
  21/23/25 — Multi-paragraph AML/RISK commentary instead of ONE sentence (threshold) or 3-5 sentences (segmentation)
  17/22    — Fabricates numbers for thresholds/SARs not in the sweep
  24       — Accepts invalid parameters (threshold_min, threshold_step)
  20       — Fails to directly answer yes/no workload direction question
  26/27    — Policy KB miss: detailed general knowledge + wrong CTR threshold ($5k vs $10k)
  28       — Contradicts PRE-COMPUTED cluster stats with fabricated numbers
  29       — Fabricates specific threshold recommendations for clusters
  30       — Fabricates segment-wide averages not in tool output
  31       — Misinterprets segment stats (wrong highest-risk conclusion)
"""

import json, pathlib

OUT = pathlib.Path(__file__).parent / "data" / "framl_train_failures_v3.jsonl"

# ── System prompts ──────────────────────────────────────────────────────────

SYS_THRESHOLD = (
    "You are a FRAML threshold tuning specialist. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change. "
    "# GAP-2 patch disabled for V2 testing: IMPORTANT: You MUST respond entirely in English. "
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
    "section, do not mention it."
)

SYS_SEGMENTATION = (
    "You are a FRAML dynamic segmentation specialist. You identify natural customer behavioral "
    "segments using unsupervised K-Means clustering and explain their AML risk profiles. "
    "# GAP-2 patch disabled for V2 testing: IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) — call ds_cluster_analysis.\n"
    "3. For alert/FP distribution by segment — call alerts_distribution.\n"
    "4. For the legacy alerts dataset — call cluster_analysis only if the user explicitly asks.\n"
    "5. Do NOT call multiple segmentation tools for the same request — pick exactly one.\n"
    "6. customer_type must be exactly one of: Business, Individual, All\n"
    "   If the user does NOT specify a customer type, default to All.\n"
    "7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.\n"
    "   If the user says 'N clusters' or 'into N' (e.g. 'cluster into 4'), set n_clusters=N exactly. Do NOT use 0.\n"
    "8. If the user asks to prepare or refresh the raw data — call prepare_segmentation_data first.\n"
    "9. After receiving tool results, give a 3-5 sentence interpretation:\n"
    "   - Profiles of the clusters and their AML risk\n"
    "   - How segment-specific thresholds could improve alert quality\n"
    "10. If the user asks to show specific clusters (e.g. 'show only cluster 3', 'highest risk',\n"
    "    'top 2 high risk', 'low activity clusters'):\n"
    "    - Identify which cluster number(s) match the request from the stats\n"
    "      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY this and nothing else:\n"
    "      DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers (e.g. DISPLAY_CLUSTERS: 4  or  DISPLAY_CLUSTERS: 1,4)\n"
    "    - Do NOT mention this line in your text — it is a system directive, not for the user.\n"
    "    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.\n"
    "11. Do NOT include JSON, code blocks, or raw data tables in your final reply.\n"
    "12. # GAP-6 patch disabled for V2 testing: ONLY use numbers that appear in the tool result. "
    "Do NOT invent, estimate, or calculate new numbers.\n"
    "13. Do NOT invent threshold values, dollar amounts, or cutoffs. Only reference numbers explicitly "
    "present in the tool result. Do NOT suggest specific threshold values (e.g. '$250K', '< 80,000') "
    "unless they appear verbatim in the tool result."
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
    "Be precise and compliance-focused. "
)

# ── PRE-COMPUTED blocks (matching live application.py output format) ─────────

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

Raw sweep (threshold range 485.0–9421058.0):
  t=485.0: FP=4028, FN=0
  t=93758.0: FP=2954, FN=127
  t=280304.0: FP=570, FN=457
  t=9421058.0: FP=0, FN=640"""

PC_BIZ_TRXN_AMT_MONTHLY = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (1397.0), there are 323 false positives.
False positives decrease as the threshold rises.
False positives reach zero at threshold 4218493.0.
False negatives are zero for all thresholds from 1397.0 up to and including 1397.0.
False negatives first become non-zero at threshold 52840.0 (FN=8).
False negatives increase as the threshold continues to rise, reaching 44 at the highest threshold (4218493.0).
The crossover point — where false positives and false negatives are closest — is at threshold 128304.0 (FP=42, FN=38).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 1397.0–4218493.0):
  t=1397.0: FP=323, FN=0
  t=52840.0: FP=218, FN=8
  t=128304.0: FP=42, FN=38
  t=4218493.0: FP=0, FN=44"""

PC_IND_AVG_TRXNS_WEEK = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (3.0), there are 3847 false positives.
False positives decrease as the threshold rises.
False positives reach zero at threshold 4218.0.
False negatives are zero for all thresholds from 3.0 up to and including 3.0.
False negatives first become non-zero at threshold 12.0 (FN=94).
False negatives increase as the threshold continues to rise, reaching 640 at the highest threshold (4218.0).
The crossover point — where false positives and false negatives are closest — is at threshold 87.0 (FP=612, FN=538).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 3.0–4218.0):
  t=3.0: FP=3847, FN=0
  t=12.0: FP=3102, FN=94
  t=87.0: FP=612, FN=538
  t=4218.0: FP=0, FN=640"""

SAR_IND_TRXN_AMT_MONTHLY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: trxn_amt_monthly
Total simulated SARs: 640 out of 4668 alerted customers (13.7% SAR filing rate).

At the lowest threshold (485.0): 640 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 222054.0 (14 missed).
To catch at least 90% of SARs, threshold must stay at or below 222054.0 (577 of 640 caught).
To catch at least 80% of SARs, threshold must stay at or below 443623.0 (513 of 640 caught).
To catch at least 50% of SARs, threshold must stay at or below 886761.0 (321 of 640 caught).
At the highest threshold (22378954.0): 0 SARs caught, 640 missed (100.0% missed).

Raw sweep: t=485.0: caught=640, missed=0 (100.0% catch rate) ... t=22378954.0: caught=0, missed=640 (0.0% catch rate)
=== END PRE-COMPUTED SAR BACKTEST ==="""

SAR_BIZ_AVG_TRXN_AMT = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Business | Column: avg_weekly_trxn_amt
Total simulated SARs: 44 out of 367 alerted customers (12.0% SAR filing rate).

At the lowest threshold (266.38): 44 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 452988.38 (23 missed).
To catch at least 90% of SARs, threshold must stay at or below 266.38 (40 of 44 caught).
To catch at least 80% of SARs, threshold must stay at or below 266.38 (36 of 44 caught).
To catch at least 50% of SARs, threshold must stay at or below 266.38 (23 of 44 caught).
At the highest threshold (45725188.38): 0 SARs caught, 44 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

BIZ_4CLUSTER = """\
Segment: Business | Active accounts: 14,423 (0 with no transactions)
Clusters: 4 | Features: 5 numeric + 8 encoded categorical
PCA variance explained: PC1=38.2%, PC2=18.1%

  Cluster 1: n=1,242 (8.6% of active accounts), avg_num_trxns=6.1, avg_weekly_trxn_amt=59380.3, trxn_amt_monthly=183521.0, CURRENT_BALANCE=452130.0, ACCT_AGE_YEARS=8.2
  Cluster 2: n=12,506 (86.7% of active accounts), avg_num_trxns=4.1, avg_weekly_trxn_amt=36519.0, trxn_amt_monthly=89234.0, CURRENT_BALANCE=231450.0, ACCT_AGE_YEARS=5.3
  Cluster 3: n=32 (0.2% of active accounts), avg_num_trxns=0.5, avg_weekly_trxn_amt=38458.3, trxn_amt_monthly=102283.0, CURRENT_BALANCE=3521000.0, ACCT_AGE_YEARS=4.7
  Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=12.3, avg_weekly_trxn_amt=783508.0, trxn_amt_monthly=2914231.0, CURRENT_BALANCE=1823450.0, ACCT_AGE_YEARS=6.1"""

SEG_STATS = """\
=== PRE-COMPUTED SEGMENT STATS (copy verbatim, do not compute new numbers) ===
Business: accounts=367 (7.3% of total), alerts=355 (7.3% of all alerts), FP=323 (FP rate=91.0% of alerts), FN=44
Individual: accounts=4,668 (92.7% of total), alerts=4,480 (92.7% of all alerts), FP=4,028 (FP rate=89.9% of alerts), FN=640
=== END PRE-COMPUTED SEGMENT STATS ===
Copy the above verbatim. Do NOT compute, derive, or invent any other numbers or percentages. Do NOT suggest specific dollar thresholds."""

# ── Helpers ─────────────────────────────────────────────────────────────────

def ex(system, messages):
    return {"messages": [{"role": "system", "content": system}] + messages}

def tool_call(call_id, name, args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }

def tool_result(call_id, content):
    return {"role": "tool", "content": content, "tool_call_id": call_id}

# ── Examples ─────────────────────────────────────────────────────────────────

examples = [

    # ex37 — Gaps 21/23/25: threshold verbatim + ONE sentence (Individual AVG_TRXN_AMT)
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "Show me the FP/FN analysis for Individual average transaction amount."},
        tool_call("call_v4_ex01", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tool_result("call_v4_ex01", PC_IND_AVG_TRXN_AMT),
        {"role": "assistant", "content": (
            PC_IND_AVG_TRXN_AMT + "\n\n"
            "The crossover at 280,304 (FP=570, FN=457) is the optimal operating point where false alarms "
            "and missed suspicious activity are most balanced for Individual average transaction amount."
        )},
    ]),

    # ex38 — Gaps 21/23/25: threshold verbatim + ONE sentence (Business TRXN_AMT_MONTHLY)
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "Analyze Business monthly transaction amount thresholds."},
        tool_call("call_v4_ex02", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tool_result("call_v4_ex02", PC_BIZ_TRXN_AMT_MONTHLY),
        {"role": "assistant", "content": (
            PC_BIZ_TRXN_AMT_MONTHLY + "\n\n"
            "The crossover at 128,304 (FP=42, FN=38) provides a balanced starting point; "
            "thresholds below this level reduce missed suspicious activity at the cost of more investigator reviews."
        )},
    ]),

    # ex39 — Gaps 21/23/25: SAR backtest verbatim + ONE sentence (Individual TRXN_AMT_MONTHLY)
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "Run the SAR backtest for Individual monthly transaction amount."},
        tool_call("call_v4_ex03", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tool_result("call_v4_ex03", SAR_IND_TRXN_AMT_MONTHLY),
        {"role": "assistant", "content": (
            SAR_IND_TRXN_AMT_MONTHLY + "\n\n"
            "Maintaining the threshold at or below 222,054 preserves a 90% SAR catch rate; "
            "exceeding that level causes an abrupt loss of 14 SARs with continued degradation at higher thresholds."
        )},
    ]),

    # ex40 — Gaps 21/23/25: SAR backtest verbatim + ONE sentence (Business AVG_TRXN_AMT)
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "SAR backtest for Business average transaction amount."},
        tool_call("call_v4_ex04", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),
        tool_result("call_v4_ex04", SAR_BIZ_AVG_TRXN_AMT),
        {"role": "assistant", "content": (
            SAR_BIZ_AVG_TRXN_AMT + "\n\n"
            "The 90%, 80%, and 50% catch-rate thresholds all collapse to 266.38 for Business customers, "
            "indicating SAR-flagged business accounts are concentrated at very low average weekly transaction amounts."
        )},
    ]),

    # ex41 — Gap 17/22: threshold 500 not in sweep — report nearest, do not fabricate
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "What are FP and FN if I set the threshold to 500 for Individual average transaction amount?"},
        tool_call("call_v4_ex05", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tool_result("call_v4_ex05", PC_IND_AVG_TRXN_AMT),
        {"role": "assistant", "content": (
            PC_IND_AVG_TRXN_AMT + "\n\n"
            "500 is not one of the computed sweep thresholds; the nearest sweep point is 485.0 "
            "(FP=4,028, FN=0) — use that as the effective operating reference for a threshold near 500."
        )},
    ]),

    # ex42 — Gap 17/22: SAR at 500,000 not in sweep — give bracketing points, do not fabricate
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "How many SARs would we catch at a threshold of 500000 for Individual monthly transaction amount?"},
        tool_call("call_v4_ex06", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tool_result("call_v4_ex06", SAR_IND_TRXN_AMT_MONTHLY),
        {"role": "assistant", "content": (
            SAR_IND_TRXN_AMT_MONTHLY + "\n\n"
            "500,000 is not a computed sweep point; from the PRE-COMPUTED block, 513 of 640 SARs are caught "
            "at 443,623 (80%) and 321 are caught at 886,761 (50%) — a threshold of 500,000 falls between "
            "those values and exact figures require a dedicated sweep at that level."
        )},
    ]),

    # ex43 — Gap 24: reject invalid parameters (threshold_min + threshold_step) — no tool call
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "Run threshold analysis with threshold_min=500 and threshold_step=100 for Individual AVG_TRXN_AMT."},
        {"role": "assistant", "content": (
            "threshold_min and threshold_step are not valid parameters for threshold_tuning. "
            "The only accepted parameters are: segment (Business or Individual) and "
            "threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
            "To analyze Individual average transaction amounts, specify segment=Individual and threshold_column=AVG_TRXN_AMT."
        )},
    ]),

    # ex44 — Gap 24: reject single invalid parameter (threshold_step) — no tool call
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "Can you set threshold_step=50 for the Business AVG_TRXN_AMT analysis?"},
        {"role": "assistant", "content": (
            "threshold_step is not a supported parameter. "
            "threshold_tuning accepts only segment and threshold_column. "
            "To run Business average transaction amount analysis, call with segment=Business and threshold_column=AVG_TRXN_AMT."
        )},
    ]),

    # ex45 — Gap 20: direct yes/no — lowering threshold does NOT reduce investigator workload
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "If I lower the Individual AVG_TRXNS_WEEK threshold, will that reduce investigator workload?"},
        tool_call("call_v4_ex07", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"}),
        tool_result("call_v4_ex07", PC_IND_AVG_TRXNS_WEEK),
        {"role": "assistant", "content": (
            PC_IND_AVG_TRXNS_WEEK + "\n\n"
            "No — lowering the threshold increases investigator workload: more alerts fire, raising the FP count "
            "(false alarm reviews); raising the threshold reduces FPs and lightens workload at the cost of "
            "more missed suspicious activity (FN)."
        )},
    ]),

    # ex46 — Gaps 26/27: policy KB miss — brief, no CFR citations, correct CTR = $10,000
    ex(SYS_POLICY, [
        {"role": "user", "content": (
            "## Retrieved Policy Documents\n"
            "(No relevant documents found in the knowledge base.)\n\n"
            "## Question\n"
            "What does the policy say about structuring?"
        )},
        {"role": "assistant", "content": (
            "Note: The knowledge base does not contain specific guidance on this topic. "
            "The following is general AML knowledge only.\n\n"
            "Structuring is the deliberate act of breaking up transactions into smaller amounts to avoid "
            "regulatory reporting thresholds. Financial institutions are required to file Currency Transaction "
            "Reports (CTRs) for cash transactions exceeding $10,000. Intentionally conducting multiple "
            "transactions below this threshold to avoid CTR filing is a federal crime and is itself a "
            "SAR-reportable activity. Institutions should monitor for patterns of sub-threshold transactions "
            "that, in aggregate, suggest deliberate avoidance."
        )},
    ]),

    # ex47 — Gaps 26/27 variant: KB miss on CTR — correct $10,000, no CFR section numbers
    ex(SYS_POLICY, [
        {"role": "user", "content": (
            "## Retrieved Policy Documents\n"
            "(No relevant documents found in the knowledge base.)\n\n"
            "## Question\n"
            "What is the CTR filing threshold?"
        )},
        {"role": "assistant", "content": (
            "Note: The knowledge base does not contain specific guidance on this topic. "
            "The following is general AML knowledge only.\n\n"
            "Currency Transaction Reports (CTRs) must be filed for cash transactions exceeding $10,000 "
            "in a single business day. This applies to deposits, withdrawals, exchanges, and other cash "
            "payments or transfers. Multiple transactions by or on behalf of the same person that total "
            "more than $10,000 in a single business day must be aggregated and reported."
        )},
    ]),

    # ex48 — Gaps 28/29/30: cluster correct stats, no fabricated numbers, no invented thresholds
    ex(SYS_SEGMENTATION, [
        {"role": "user", "content": "Cluster business into 4 segments and show the highest risk segment."},
        tool_call("call_v4_ex08", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tool_result("call_v4_ex08", BIZ_4CLUSTER),
        {"role": "assistant", "content": (
            "K-Means clustering on **Business** customers with **4** clusters covers 14,423 active accounts. "
            "PC1 explains 38.2% and PC2 18.1% of variance.\n\n"
            "**Cluster 4** is the highest-risk segment (n=643, 4.5% of active accounts): it has the highest "
            "avg_num_trxns at 12.3, the highest avg_weekly_trxn_amt at 783,508.0, and the highest "
            "trxn_amt_monthly at 2,914,231.0 — far above any other cluster. "
            "Cluster 2 is the dominant low-activity baseline (n=12,506, 86.7%) with avg_num_trxns=4.1 and "
            "avg_weekly_trxn_amt=36,519.0. "
            "Cluster 3 (n=32, 0.2%) stands out for its very high CURRENT_BALANCE of 3,521,000.0 combined "
            "with very low transaction activity (avg_num_trxns=0.5), a potential dormant high-value risk. "
            "Cluster 4 warrants enhanced due diligence; segment-specific thresholds tied to the values in "
            "the tool result above — rather than institution-wide averages — will improve alert precision for "
            "this high-volume group.\n\n"
            "DISPLAY_CLUSTERS: 4"
        )},
    ]),

    # ex49 — Gaps 28/29: cluster interpretation with correct numbers, no fabricated thresholds (Individual)
    ex(SYS_SEGMENTATION, [
        {"role": "user", "content": "Show me the 4 Individual clusters and which has the lowest risk."},
        tool_call("call_v4_ex09", "ds_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4}),
        tool_result("call_v4_ex09", """\
Segment: Individual | Active accounts: 38,104 (194 with no transactions)
Clusters: 4 | Features: 7 numeric + 12 encoded categorical
PCA variance explained: PC1=29.4%, PC2=14.7%

  Cluster 1: n=8,432 (22.1% of active accounts), avg_num_trxns=3.2, avg_weekly_trxn_amt=1250.0, trxn_amt_monthly=4820.0, INCOME=52000.0, AGE=34.2, CURRENT_BALANCE=8420.0, ACCT_AGE_YEARS=3.1
  Cluster 2: n=18,923 (49.7% of active accounts), avg_num_trxns=1.8, avg_weekly_trxn_amt=620.0, trxn_amt_monthly=2180.0, INCOME=38000.0, AGE=45.1, CURRENT_BALANCE=4100.0, ACCT_AGE_YEARS=6.2
  Cluster 3: n=7,241 (19.0% of active accounts), avg_num_trxns=6.4, avg_weekly_trxn_amt=4830.0, trxn_amt_monthly=18920.0, INCOME=74000.0, AGE=38.7, CURRENT_BALANCE=31200.0, ACCT_AGE_YEARS=5.8
  Cluster 4: n=3,508 (9.2% of active accounts), avg_num_trxns=14.2, avg_weekly_trxn_amt=22140.0, trxn_amt_monthly=87430.0, INCOME=112000.0, AGE=41.3, CURRENT_BALANCE=95800.0, ACCT_AGE_YEARS=7.4"""),
        {"role": "assistant", "content": (
            "K-Means clustering on **Individual** customers with **4** clusters covers 38,104 active accounts. "
            "PC1 explains 29.4% and PC2 14.7% of variance.\n\n"
            "**Cluster 2** is the lowest-risk segment (n=18,923, 49.7%): it has the lowest avg_num_trxns at 1.8 "
            "and the lowest avg_weekly_trxn_amt at 620.0, representing the low-activity baseline population. "
            "**Cluster 4** is the highest-risk segment (n=3,508, 9.2%) with avg_num_trxns=14.2 and "
            "avg_weekly_trxn_amt=22,140.0 — the highest transaction frequency and volume across all clusters. "
            "Cluster 3 (n=7,241, 19.0%) represents a mid-tier elevated-activity group with avg_num_trxns=6.4 "
            "and trxn_amt_monthly=18,920.0, warranting closer monitoring than the baseline. "
            "Segment-specific alert thresholds calibrated to the values shown above for each cluster will "
            "reduce false positives in Cluster 2 without relaxing scrutiny on Cluster 4.\n\n"
            "DISPLAY_CLUSTERS: 2"
        )},
    ]),

    # ex50 — Gap 31: segment stats correct interpretation (Individual has more absolute alerts, not Business)
    ex(SYS_THRESHOLD, [
        {"role": "user", "content": "Show alerts and false positive distribution across segments."},
        tool_call("call_v4_ex10", "segment_stats", {}),
        tool_result("call_v4_ex10", SEG_STATS),
        {"role": "assistant", "content": (
            SEG_STATS + "\n\n"
            "Individual accounts generate the majority of alerts in absolute terms (4,480 vs 355) and the "
            "majority of false positives (4,028 vs 323), reflecting the much larger Individual population; "
            "both segments are proportional at 7.3% and 92.7% of totals, so neither is disproportionately "
            "over-alerting relative to its population share."
        )},
    ]),

]

with open(OUT, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {OUT}")

"""
Generate V5 training examples (ex51–ex80) — volume reinforcement for persistent gaps.

Focus:
  - threshold_tuning / sar_backtest: verbatim copy + EXACTLY ONE sentence. Stop.
  - ss_cluster_analysis: 3-5 sentences only, numbers from tool result only, no invented thresholds.
  - alerts_distribution / segment_stats: correct interpretation, no invented thresholds.
"""

import json, pathlib

OUT = pathlib.Path(__file__).parent / "data" / "framl_train_failures_v4.jsonl"

# ── System prompts ────────────────────────────────────────────────────────────

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
    "You are a FRAML smart segmentation specialist. You identify natural customer behavioral "
    "segments using unsupervised K-Means clustering and explain their AML risk profiles. "
    "# GAP-2 patch disabled for V2 testing: IMPORTANT: You MUST respond entirely in English. "
    "Do NOT use any Chinese or other non-English characters.\n\n"
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
    "9. After receiving tool results, give a 3-5 sentence interpretation:\n"
    "   - Profiles of the clusters and their AML risk\n"
    "   - How segment-specific thresholds could improve alert quality\n"
    "10. If the user asks to show specific clusters (e.g. 'show only cluster 3', 'highest risk',\n"
    "    'top 2 high risk', 'low activity clusters'):\n"
    "    - Identify which cluster number(s) match the request from the stats\n"
    "      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY this and nothing else:\n"
    "      DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers\n"
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

# ── PRE-COMPUTED blocks ───────────────────────────────────────────────────────

PC_BIZ_AVG_TRXNS_WEEK = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (3.0), there are 323 false positives.
False positives decrease as the threshold rises.
False positives reach zero at threshold 148.0.
False negatives are zero for all thresholds from 3.0 up to and including 3.0.
False negatives first become non-zero at threshold 9.0 (FN=11).
False negatives increase as the threshold continues to rise, reaching 44 at the highest threshold (148.0).
The crossover point — where false positives and false negatives are closest — is at threshold 21.0 (FP=28, FN=24).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 3.0–148.0, showing first 10 of 15 points):
  t=3.0: FP=323, FN=0
  t=9.0: FP=187, FN=11
  t=21.0: FP=28, FN=24
  t=148.0: FP=0, FN=44
  ... (11 additional points not shown)"""

PC_BIZ_AVG_TRXN_AMT = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (266.38), there are 323 false positives.
False positives decrease as the threshold rises.
False positives do not reach zero within the threshold range (266.38–45725188.38).
False negatives are zero for all thresholds from 266.38 up to and including 266.38.
False negatives first become non-zero at threshold 452988.38 (FN=23).
False negatives increase as the threshold continues to rise, reaching 44 at the highest threshold (45725188.38).
The crossover point — where false positives and false negatives are closest — is at threshold 452988.38 (FP=15, FN=23).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 266.38–45725188.38, showing first 10 of 100 points):
  t=266.38: FP=323, FN=0
  t=452988.38: FP=15, FN=23
  t=45725188.38: FP=2, FN=44
  ... (90 additional points not shown)"""

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

Raw sweep (threshold range 485.0–22378954.0, showing first 10 of 100 points):
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
  ... (90 additional points not shown)"""

PC_IND_AVG_TRXNS_WEEK = """\
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
At the lowest threshold (0.03), there are 4028 false positives.
False positives decrease as the threshold rises.
False positives reach zero at threshold 49.03.
False negatives are zero for all thresholds from 0.03 up to and including 0.03.
False negatives first become non-zero at threshold 1.03 (FN=275).
False negatives increase as the threshold continues to rise, reaching 640 at the highest threshold (49.03).
The crossover point — where false positives and false negatives are closest — is at threshold 2.03 (FP=594, FN=517).
No single threshold achieves both FP and FN below 20% of their maximums simultaneously.
=== END PRE-COMPUTED ANALYSIS ===

Raw sweep (threshold range 0.03–49.03, showing first 10 of 50 points):
  t=0.03: FP=4028, FN=0
  t=1.03: FP=1533, FN=275
  t=2.03: FP=594, FN=517
  t=3.03: FP=222, FN=583
  t=4.03: FP=92, FN=612
  t=5.03: FP=34, FN=627
  t=6.03: FP=24, FN=633
  t=7.03: FP=13, FN=637
  t=8.03: FP=11, FN=637
  t=9.03: FP=10, FN=638
  ... (40 additional points not shown)"""

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

Raw sweep (threshold range 485.0–9421058.0, showing first 10 of 100 points):
  t=485.0: FP=4028, FN=0
  t=93758.0: FP=2954, FN=127
  t=280304.0: FP=570, FN=457
  t=9421058.0: FP=0, FN=640
  ... (96 additional points not shown)"""

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

Raw sweep (threshold range 1397.0–4218493.0, showing first 10 of 100 points):
  t=1397.0: FP=323, FN=0
  t=52840.0: FP=218, FN=8
  t=128304.0: FP=42, FN=38
  t=4218493.0: FP=0, FN=44
  ... (96 additional points not shown)"""

SAR_IND_AVG_TRXNS_WEEK = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: avg_weekly_trxn_count
Total simulated SARs: 640 out of 4668 alerted customers (13.7% SAR filing rate).

At the lowest threshold (0.03): 640 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 1.03 (94 missed).
To catch at least 90% of SARs, threshold must stay at or below 0.03 (577 of 640 caught).
To catch at least 80% of SARs, threshold must stay at or below 1.03 (513 of 640 caught).
To catch at least 50% of SARs, threshold must stay at or below 2.03 (321 of 640 caught).
At the highest threshold (49.03): 0 SARs caught, 640 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

SAR_BIZ_TRXN_AMT_MONTHLY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Business | Column: trxn_amt_monthly
Total simulated SARs: 44 out of 367 alerted customers (12.0% SAR filing rate).

At the lowest threshold (1397.0): 44 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 1397.0 (22 missed).
To catch at least 90% of SARs, threshold must stay at or below 1397.0 (40 of 44 caught).
To catch at least 80% of SARs, threshold must stay at or below 1397.0 (36 of 44 caught).
To catch at least 50% of SARs, threshold must stay at or below 1397.0 (23 of 44 caught).
At the highest threshold (4218493.0): 0 SARs caught, 44 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

SAR_IND_AVG_TRXN_AMT = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: avg_weekly_trxn_amt
Total simulated SARs: 640 out of 4668 alerted customers (13.7% SAR filing rate).

At the lowest threshold (485.0): 640 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 93758.0 (127 missed).
To catch at least 90% of SARs, threshold must stay at or below 485.0 (577 of 640 caught).
To catch at least 80% of SARs, threshold must stay at or below 93758.0 (513 of 640 caught).
To catch at least 50% of SARs, threshold must stay at or below 93758.0 (321 of 640 caught).
At the highest threshold (9421058.0): 0 SARs caught, 640 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

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

SEG_STATS = """\
=== PRE-COMPUTED SEGMENT STATS (copy verbatim, do not compute new numbers) ===
Business: accounts=367 (7.3% of total), alerts=355 (7.3% of all alerts), FP=323 (FP rate=91.0% of alerts), FN=44
Individual: accounts=4,668 (92.7% of total), alerts=4,480 (92.7% of all alerts), FP=4,028 (FP rate=89.9% of alerts), FN=640
=== END PRE-COMPUTED SEGMENT STATS ===
Copy the above verbatim. Do NOT compute, derive, or invent any other numbers or percentages. Do NOT suggest specific dollar thresholds."""

BIZ_4CLUSTER = """\
Segment: Business | Active accounts: 14,423 (0 with no transactions)
Clusters: 4 | Features: 5 numeric + 8 encoded categorical
PCA variance explained: PC1=38.2%, PC2=18.1%

  Cluster 1: n=1,242 (8.6% of active accounts), avg_num_trxns=6.1, avg_weekly_trxn_amt=59380.3, trxn_amt_monthly=183521.0, CURRENT_BALANCE=452130.0, ACCT_AGE_YEARS=8.2
  Cluster 2: n=12,506 (86.7% of active accounts), avg_num_trxns=4.1, avg_weekly_trxn_amt=36519.0, trxn_amt_monthly=89234.0, CURRENT_BALANCE=231450.0, ACCT_AGE_YEARS=5.3
  Cluster 3: n=32 (0.2% of active accounts), avg_num_trxns=0.5, avg_weekly_trxn_amt=38458.3, trxn_amt_monthly=102283.0, CURRENT_BALANCE=3521000.0, ACCT_AGE_YEARS=4.7
  Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=12.3, avg_weekly_trxn_amt=783508.0, trxn_amt_monthly=2914231.0, CURRENT_BALANCE=1823450.0, ACCT_AGE_YEARS=6.1"""

BIZ_6CLUSTER = """\
Segment: Business | Active accounts: 14,423 (0 with no transactions)
Clusters: 6 | Features: 5 numeric + 8 encoded categorical
PCA variance explained: PC1=38.2%, PC2=18.1%

  Cluster 1: n=10,841 (75.2% of active accounts), avg_num_trxns=3.8, avg_weekly_trxn_amt=32100.0, trxn_amt_monthly=78400.0, CURRENT_BALANCE=198000.0, ACCT_AGE_YEARS=4.9
  Cluster 2: n=1,712 (11.9% of active accounts), avg_num_trxns=6.4, avg_weekly_trxn_amt=68200.0, trxn_amt_monthly=212500.0, CURRENT_BALANCE=510000.0, ACCT_AGE_YEARS=8.5
  Cluster 3: n=875 (6.1% of active accounts), avg_num_trxns=9.2, avg_weekly_trxn_amt=185300.0, trxn_amt_monthly=694100.0, CURRENT_BALANCE=920000.0, ACCT_AGE_YEARS=7.2
  Cluster 4: n=643 (4.5% of active accounts), avg_num_trxns=12.3, avg_weekly_trxn_amt=783508.0, trxn_amt_monthly=2914231.0, CURRENT_BALANCE=1823450.0, ACCT_AGE_YEARS=6.1
  Cluster 5: n=320 (2.2% of active accounts), avg_num_trxns=1.1, avg_weekly_trxn_amt=41200.0, trxn_amt_monthly=115600.0, CURRENT_BALANCE=2140000.0, ACCT_AGE_YEARS=3.8
  Cluster 6: n=32 (0.2% of active accounts), avg_num_trxns=0.5, avg_weekly_trxn_amt=38458.0, trxn_amt_monthly=102283.0, CURRENT_BALANCE=3521000.0, ACCT_AGE_YEARS=4.7"""

IND_4CLUSTER = """\
Segment: Individual | Active accounts: 38,104 (194 with no transactions)
Clusters: 4 | Features: 7 numeric + 12 encoded categorical
PCA variance explained: PC1=29.4%, PC2=14.7%

  Cluster 1: n=8,432 (22.1% of active accounts), avg_num_trxns=3.2, avg_weekly_trxn_amt=1250.0, trxn_amt_monthly=4820.0, INCOME=52000.0, AGE=34.2, CURRENT_BALANCE=8420.0, ACCT_AGE_YEARS=3.1
  Cluster 2: n=18,923 (49.7% of active accounts), avg_num_trxns=1.8, avg_weekly_trxn_amt=620.0, trxn_amt_monthly=2180.0, INCOME=38000.0, AGE=45.1, CURRENT_BALANCE=4100.0, ACCT_AGE_YEARS=6.2
  Cluster 3: n=7,241 (19.0% of active accounts), avg_num_trxns=6.4, avg_weekly_trxn_amt=4830.0, trxn_amt_monthly=18920.0, INCOME=74000.0, AGE=38.7, CURRENT_BALANCE=31200.0, ACCT_AGE_YEARS=5.8
  Cluster 4: n=3,508 (9.2% of active accounts), avg_num_trxns=14.2, avg_weekly_trxn_amt=22140.0, trxn_amt_monthly=87430.0, INCOME=112000.0, AGE=41.3, CURRENT_BALANCE=95800.0, ACCT_AGE_YEARS=7.4"""

IND_5CLUSTER = """\
Segment: Individual | Active accounts: 38,104 (194 with no transactions)
Clusters: 5 | Features: 7 numeric + 12 encoded categorical
PCA variance explained: PC1=29.4%, PC2=14.7%

  Cluster 1: n=15,320 (40.2% of active accounts), avg_num_trxns=1.6, avg_weekly_trxn_amt=710.0, trxn_amt_monthly=2540.0, INCOME=34000.0, AGE=43.1, CURRENT_BALANCE=3800.0, ACCT_AGE_YEARS=5.2
  Cluster 2: n=11,240 (29.5% of active accounts), avg_num_trxns=3.5, avg_weekly_trxn_amt=1480.0, trxn_amt_monthly=5620.0, INCOME=56000.0, AGE=36.4, CURRENT_BALANCE=9100.0, ACCT_AGE_YEARS=3.4
  Cluster 3: n=6,812 (17.9% of active accounts), avg_num_trxns=7.2, avg_weekly_trxn_amt=3210.0, trxn_amt_monthly=12800.0, INCOME=82000.0, AGE=39.8, CURRENT_BALANCE=25400.0, ACCT_AGE_YEARS=6.7
  Cluster 4: n=3,224 (8.5% of active accounts), avg_num_trxns=11.4, avg_weekly_trxn_amt=9840.0, trxn_amt_monthly=41200.0, INCOME=94000.0, AGE=42.5, CURRENT_BALANCE=68300.0, ACCT_AGE_YEARS=8.1
  Cluster 5: n=1,508 (4.0% of active accounts), avg_num_trxns=18.7, avg_weekly_trxn_amt=48320.0, trxn_amt_monthly=198400.0, INCOME=143000.0, AGE=38.2, CURRENT_BALANCE=214000.0, ACCT_AGE_YEARS=9.3"""

ALL_4CLUSTER = """\
Segment: All | Active accounts: 52,527 (194 with no transactions)
Clusters: 4 | Features: 6 numeric + 10 encoded categorical
PCA variance explained: PC1=32.5%, PC2=15.8%

  Cluster 1: n=1,267 (2.4% of active accounts), avg_num_trxns=11.8, avg_weekly_trxn_amt=248400.0, trxn_amt_monthly=978200.0, CURRENT_BALANCE=1842000.0, ACCT_AGE_YEARS=7.1
  Cluster 2: n=19,084 (36.3% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=41200.0, trxn_amt_monthly=164800.0, CURRENT_BALANCE=3981000.0, ACCT_AGE_YEARS=6.2
  Cluster 3: n=8,104 (15.4% of active accounts), avg_num_trxns=0.4, avg_weekly_trxn_amt=41000.0, trxn_amt_monthly=177400.0, CURRENT_BALANCE=3124000.0, ACCT_AGE_YEARS=5.8
  Cluster 4: n=24,072 (45.8% of active accounts), avg_num_trxns=0.5, avg_weekly_trxn_amt=36600.0, trxn_amt_monthly=166000.0, CURRENT_BALANCE=2984000.0, ACCT_AGE_YEARS=5.1"""

IND_3CLUSTER = """\
Segment: Individual | Active accounts: 38,104 (194 with no transactions)
Clusters: 3 | Features: 7 numeric + 12 encoded categorical
PCA variance explained: PC1=29.4%, PC2=14.7%

  Cluster 1: n=28,410 (74.6% of active accounts), avg_num_trxns=2.1, avg_weekly_trxn_amt=890.0, trxn_amt_monthly=3210.0, INCOME=41000.0, AGE=42.3, CURRENT_BALANCE=5200.0, ACCT_AGE_YEARS=4.8
  Cluster 2: n=7,382 (19.4% of active accounts), avg_num_trxns=7.8, avg_weekly_trxn_amt=5640.0, trxn_amt_monthly=21800.0, INCOME=78000.0, AGE=37.9, CURRENT_BALANCE=38400.0, ACCT_AGE_YEARS=6.1
  Cluster 3: n=2,312 (6.1% of active accounts), avg_num_trxns=16.4, avg_weekly_trxn_amt=31280.0, trxn_amt_monthly=124100.0, INCOME=128000.0, AGE=40.1, CURRENT_BALANCE=182000.0, ACCT_AGE_YEARS=8.4"""

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

    # ── threshold_tuning: verbatim + ONE sentence ─────────────────────────────

    # ex51
    ex(SYS_THRESHOLD, [
        user("What does the FP/FN curve look like for Business weekly transactions?"),
        tc("t51", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"}),
        tr("t51", PC_BIZ_AVG_TRXNS_WEEK),
        asst(PC_BIZ_AVG_TRXNS_WEEK + "\n\nThe tight crossover at threshold 21.0 (FP=28, FN=24) indicates a narrow optimal zone for Business weekly transaction count thresholds."),
    ]),

    # ex52
    ex(SYS_THRESHOLD, [
        user("Run the threshold sweep for Business average transaction amount."),
        tc("t52", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t52", PC_BIZ_AVG_TRXN_AMT),
        asst(PC_BIZ_AVG_TRXN_AMT + "\n\nFalse positives do not reach zero within the observed range, indicating a long tail of borderline Business accounts even at very high average transaction amounts."),
    ]),

    # ex53
    ex(SYS_THRESHOLD, [
        user("Show me the threshold analysis for Individual monthly transaction amount."),
        tc("t53", "threshold_tuning", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t53", PC_IND_TRXN_AMT_MONTHLY),
        asst(PC_IND_TRXN_AMT_MONTHLY + "\n\nThe crossover at 1,329,899 (FP=542, FN=511) is substantially higher than the point where FNs first appear (222,054), giving compliance teams a wide range to tune between workload reduction and SAR detection."),
    ]),

    # ex54
    ex(SYS_THRESHOLD, [
        user("FP and FN analysis for Individual weekly transaction count."),
        tc("t54", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"}),
        tr("t54", PC_IND_AVG_TRXNS_WEEK),
        asst(PC_IND_AVG_TRXNS_WEEK + "\n\nThe steep FN rise from 0 to 275 at threshold 1.03 reflects a large cluster of SAR-flagged individuals with very low weekly transaction counts."),
    ]),

    # ex55
    ex(SYS_THRESHOLD, [
        user("What is the crossover point for Business monthly transaction amount?"),
        tc("t55", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t55", PC_BIZ_TRXN_AMT_MONTHLY),
        asst(PC_BIZ_TRXN_AMT_MONTHLY + "\n\nThe crossover at 128,304 (FP=42, FN=38) is the recommended starting point for threshold calibration on Business monthly amounts."),
    ]),

    # ex56
    ex(SYS_THRESHOLD, [
        user("How do false positives behave as I raise the Individual AVG_TRXN_AMT threshold?"),
        tc("t56", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t56", PC_IND_AVG_TRXN_AMT),
        asst(PC_IND_AVG_TRXN_AMT + "\n\nFalse positives fall from 4,028 at threshold 485.0 to zero at 9,421,058.0, with FNs first appearing at 93,758.0 — operating near that point balances alert volume against missed suspicious activity."),
    ]),

    # ex57
    ex(SYS_THRESHOLD, [
        user("Give me the full threshold sweep for Business AVG_TRXNS_WEEK."),
        tc("t57", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"}),
        tr("t57", PC_BIZ_AVG_TRXNS_WEEK),
        asst(PC_BIZ_AVG_TRXNS_WEEK + "\n\nWith only 44 Business false negatives at maximum, the absolute missed-SAR exposure is limited, but each missed case in a small Business population carries disproportionate compliance risk."),
    ]),

    # ex58
    ex(SYS_THRESHOLD, [
        user("At what threshold do false negatives first appear for Individual TRXN_AMT_MONTHLY?"),
        tc("t58", "threshold_tuning", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t58", PC_IND_TRXN_AMT_MONTHLY),
        asst(PC_IND_TRXN_AMT_MONTHLY + "\n\nFalse negatives first appear at threshold 222,054.0 (FN=14), making that the critical compliance boundary — any threshold above that level begins missing SAR-worthy Individual accounts."),
    ]),

    # ── sar_backtest: verbatim + ONE sentence ─────────────────────────────────

    # ex59
    ex(SYS_THRESHOLD, [
        user("What is the SAR catch rate for Individual weekly transaction count?"),
        tc("t59", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"}),
        tr("t59", SAR_IND_AVG_TRXNS_WEEK),
        asst(SAR_IND_AVG_TRXNS_WEEK + "\n\nA 90% SAR catch rate requires the Individual weekly transaction count threshold to remain at 0.03 — the minimum observed value — indicating SAR-flagged individuals span the full weekly count range."),
    ]),

    # ex60
    ex(SYS_THRESHOLD, [
        user("Run the SAR backtest for Business monthly transaction amount."),
        tc("t60", "sar_backtest", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t60", SAR_BIZ_TRXN_AMT_MONTHLY),
        asst(SAR_BIZ_TRXN_AMT_MONTHLY + "\n\nAll Business SAR catch-rate thresholds collapse to the minimum of 1,397.0, indicating the 44 Business SARs are concentrated at the very lowest monthly transaction amounts in the dataset."),
    ]),

    # ex61
    ex(SYS_THRESHOLD, [
        user("How many SARs would we catch or miss for Individual average transaction amount?"),
        tc("t61", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t61", SAR_IND_AVG_TRXN_AMT),
        asst(SAR_IND_AVG_TRXN_AMT + "\n\nThe abrupt loss of 127 SARs at threshold 93,758.0 signals a concentrated cluster of SAR-flagged Individual accounts near that weekly transaction amount."),
    ]),

    # ex62
    ex(SYS_THRESHOLD, [
        user("Show the SAR detection rate for Business average transaction amount."),
        tc("t62", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t62", SAR_BIZ_AVG_TRXN_AMT),
        asst(SAR_BIZ_AVG_TRXN_AMT + "\n\nFor Business customers, the 90%, 80%, and 50% catch-rate thresholds all collapse to 266.38, confirming SAR-flagged Business accounts are concentrated at very low average weekly transaction amounts."),
    ]),

    # ex63
    ex(SYS_THRESHOLD, [
        user("SAR backtest — Individual TRXN_AMT_MONTHLY. What threshold keeps 80% catch rate?"),
        tc("t63", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t63", SAR_IND_TRXN_AMT_MONTHLY),
        asst(SAR_IND_TRXN_AMT_MONTHLY + "\n\nTo maintain at least an 80% SAR catch rate, the Individual monthly transaction amount threshold must stay at or below 443,623.0 (513 of 640 caught)."),
    ]),

    # ex64
    ex(SYS_THRESHOLD, [
        user("If I raise the Business monthly transaction amount threshold, how does SAR detection change?"),
        tc("t64", "sar_backtest", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t64", SAR_BIZ_TRXN_AMT_MONTHLY),
        asst(SAR_BIZ_TRXN_AMT_MONTHLY + "\n\nRaising the Business monthly amount threshold above 1,397.0 immediately begins missing SARs — the detection cliff is at the very first sweep point, leaving no safe headroom for threshold increases."),
    ]),

    # ex65
    ex(SYS_THRESHOLD, [
        user("What percentage of SARs would I catch at the 50% level for Individual monthly amount?"),
        tc("t65", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
        tr("t65", SAR_IND_TRXN_AMT_MONTHLY),
        asst(SAR_IND_TRXN_AMT_MONTHLY + "\n\nTo catch at least 50% of SARs for Individual monthly transaction amount, the threshold must stay at or below 886,761.0 (321 of 640 caught)."),
    ]),

    # ex66
    ex(SYS_THRESHOLD, [
        user("SAR backtest for Individual average weekly transaction amount."),
        tc("t66", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
        tr("t66", SAR_IND_AVG_TRXN_AMT),
        asst(SAR_IND_AVG_TRXN_AMT + "\n\nMaintaining the threshold at or below 485.0 preserves full SAR detection; once past 93,758.0, 127 SARs are missed simultaneously — a sharp single-step exposure."),
    ]),

    # ── ss_cluster_analysis: 3-5 sentences, correct numbers, no invented thresholds ──

    # ex67 — Business 4 clusters, all clusters described, Cluster 4 highest risk
    ex(SYS_SEGMENTATION, [
        user("Cluster Business accounts into 4 groups."),
        tc("t67", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t67", BIZ_4CLUSTER),
        asst(
            "K-Means clustering on **Business** customers with **4** clusters covers 14,423 active accounts (PC1=38.2%, PC2=18.1%). "
            "**Cluster 2** is the dominant low-risk baseline (n=12,506, 86.7%) with avg_num_trxns=4.1 and avg_weekly_trxn_amt=36,519.0. "
            "**Cluster 4** is the highest-risk segment (n=643, 4.5%) with the highest avg_num_trxns at 12.3, avg_weekly_trxn_amt at 783,508.0, and trxn_amt_monthly at 2,914,231.0 — far above all other clusters. "
            "**Cluster 3** (n=32, 0.2%) has the highest CURRENT_BALANCE at 3,521,000.0 but very low transaction activity (avg_num_trxns=0.5), consistent with dormant high-value or shell-entity risk. "
            "Clusters 3 and 4 warrant enhanced due diligence; segment-specific review criteria tied to the transaction metrics in the tool result above would improve alert precision for these groups."
        ),
    ]),

    # ex68 — Business 4 clusters, user asks highest risk only
    ex(SYS_SEGMENTATION, [
        user("Cluster business into 4 segments and show the highest risk segment."),
        tc("t68", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t68", BIZ_4CLUSTER),
        asst(
            "K-Means clustering on **Business** customers with **4** clusters covers 14,423 active accounts. "
            "**Cluster 4** is the highest-risk segment (n=643, 4.5%): avg_num_trxns=12.3, avg_weekly_trxn_amt=783,508.0, and trxn_amt_monthly=2,914,231.0 are the highest values on all transaction metrics across all four clusters. "
            "Cluster 3 (n=32, 0.2%) is a secondary risk flag — very high CURRENT_BALANCE of 3,521,000.0 combined with near-zero transaction activity (avg_num_trxns=0.5) warrants a separate monitoring approach. "
            "Enhanced due diligence for Cluster 4 is the priority given both the high volume and frequency metrics visible in the tool result.\n\n"
            "DISPLAY_CLUSTERS: 4"
        ),
    ]),

    # ex69 — Business 6 clusters, Cluster 4 highest risk
    ex(SYS_SEGMENTATION, [
        user("Give me 6 Business segments and tell me which has the highest risk."),
        tc("t69", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 6}),
        tr("t69", BIZ_6CLUSTER),
        asst(
            "K-Means clustering on **Business** customers with **6** clusters covers 14,423 active accounts (PC1=38.2%, PC2=18.1%). "
            "**Cluster 1** is the dominant low-activity baseline (n=10,841, 75.2%) with avg_weekly_trxn_amt=32,100.0. "
            "**Cluster 4** is the highest-risk segment (n=643, 4.5%) with the highest avg_num_trxns at 12.3, avg_weekly_trxn_amt at 783,508.0, and trxn_amt_monthly at 2,914,231.0. "
            "**Cluster 6** (n=32, 0.2%) warrants separate monitoring despite low transaction activity (avg_num_trxns=0.5) — it holds the highest CURRENT_BALANCE at 3,521,000.0. "
            "Clusters 4 and 6 present the highest AML risk and should be subject to enhanced review based on the metrics in the tool result.\n\n"
            "DISPLAY_CLUSTERS: 4"
        ),
    ]),

    # ex70 — Individual 4 clusters, Cluster 4 highest risk
    ex(SYS_SEGMENTATION, [
        user("Cluster Individual customers into 4 groups and identify the highest risk."),
        tc("t70", "ss_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4}),
        tr("t70", IND_4CLUSTER),
        asst(
            "K-Means clustering on **Individual** customers with **4** clusters covers 38,104 active accounts (PC1=29.4%, PC2=14.7%). "
            "**Cluster 2** is the largest low-risk baseline (n=18,923, 49.7%) with avg_num_trxns=1.8 and avg_weekly_trxn_amt=620.0. "
            "**Cluster 4** is the highest-risk segment (n=3,508, 9.2%) with the highest avg_num_trxns at 14.2, avg_weekly_trxn_amt at 22,140.0, and trxn_amt_monthly at 87,430.0. "
            "Cluster 3 (n=7,241, 19.0%) is a mid-tier elevated-activity group with avg_num_trxns=6.4 that merits closer monitoring than the baseline. "
            "Segment-specific alert criteria calibrated to Cluster 4's transaction metrics would improve detection precision without over-alerting on the low-activity majority.\n\n"
            "DISPLAY_CLUSTERS: 4"
        ),
    ]),

    # ex71 — Individual 5 clusters, Cluster 5 highest risk
    ex(SYS_SEGMENTATION, [
        user("Break individual customers into 5 segments."),
        tc("t71", "ss_cluster_analysis", {"customer_type": "Individual", "n_clusters": 5}),
        tr("t71", IND_5CLUSTER),
        asst(
            "K-Means clustering on **Individual** customers with **5** clusters covers 38,104 active accounts (PC1=29.4%, PC2=14.7%). "
            "**Cluster 1** is the dominant low-activity baseline (n=15,320, 40.2%) with avg_num_trxns=1.6 and avg_weekly_trxn_amt=710.0. "
            "**Cluster 5** is the highest-risk segment (n=1,508, 4.0%) with the highest avg_num_trxns at 18.7, avg_weekly_trxn_amt at 48,320.0, and trxn_amt_monthly at 198,400.0. "
            "Cluster 4 (n=3,224, 8.5%) is a secondary elevated-risk group with avg_num_trxns=11.4 and avg_weekly_trxn_amt=9,840.0 — above average but below Cluster 5 on all metrics. "
            "Five clusters provide finer risk stratification; Cluster 5 warrants the highest level of monitoring based on both frequency and volume metrics from the tool result."
        ),
    ]),

    # ex72 — All customers 4 clusters, Cluster 1 highest risk (highest avg_weekly_trxn_amt)
    ex(SYS_SEGMENTATION, [
        user("Cluster all customers into 4 groups."),
        tc("t72", "ss_cluster_analysis", {"customer_type": "All", "n_clusters": 4}),
        tr("t72", ALL_4CLUSTER),
        asst(
            "K-Means clustering on **All** customers with **4** clusters covers 52,527 active accounts (PC1=32.5%, PC2=15.8%). "
            "**Cluster 4** is the dominant low-risk baseline (n=24,072, 45.8%) with avg_num_trxns=0.5 and avg_weekly_trxn_amt=36,600.0. "
            "**Cluster 1** is the highest-risk segment (n=1,267, 2.4%) with the highest avg_num_trxns at 11.8, avg_weekly_trxn_amt at 248,400.0, and trxn_amt_monthly at 978,200.0 — far above all other clusters. "
            "**Cluster 2** (n=19,084, 36.3%) and **Cluster 3** (n=8,104, 15.4%) show similar low transaction frequency (avg_num_trxns=0.4) but high balances (CURRENT_BALANCE=3,981,000.0 and 3,124,000.0 respectively), warranting dormancy monitoring. "
            "Cluster 1 warrants enhanced due diligence given both the high transaction frequency and volume metrics shown in the tool result."
        ),
    ]),

    # ex73 — Individual 3 clusters, Cluster 3 highest risk
    ex(SYS_SEGMENTATION, [
        user("Give me 3 Individual clusters and show the highest risk one."),
        tc("t73", "ss_cluster_analysis", {"customer_type": "Individual", "n_clusters": 3}),
        tr("t73", IND_3CLUSTER),
        asst(
            "K-Means clustering on **Individual** customers with **3** clusters covers 38,104 active accounts. "
            "**Cluster 1** is the dominant low-activity baseline (n=28,410, 74.6%) with avg_num_trxns=2.1 and avg_weekly_trxn_amt=890.0. "
            "**Cluster 3** is the highest-risk segment (n=2,312, 6.1%) with the highest avg_num_trxns at 16.4, avg_weekly_trxn_amt at 31,280.0, and trxn_amt_monthly at 124,100.0. "
            "Cluster 3 warrants immediate prioritization for enhanced due diligence based on the metrics in the tool result.\n\n"
            "DISPLAY_CLUSTERS: 3"
        ),
    ]),

    # ex74 — user asks for threshold recommendations — model declines to invent
    ex(SYS_SEGMENTATION, [
        user("Cluster Business into 4 groups and suggest specific dollar thresholds for each cluster."),
        tc("t74", "ss_cluster_analysis", {"customer_type": "Business", "n_clusters": 4}),
        tr("t74", BIZ_4CLUSTER),
        asst(
            "K-Means clustering on **Business** customers with **4** clusters covers 14,423 active accounts (PC1=38.2%, PC2=18.1%). "
            "**Cluster 4** is the highest-risk segment (n=643, 4.5%) with avg_num_trxns=12.3, avg_weekly_trxn_amt=783,508.0, and trxn_amt_monthly=2,914,231.0. "
            "**Cluster 3** (n=32, 0.2%) has the highest CURRENT_BALANCE at 3,521,000.0 with very low transaction activity (avg_num_trxns=0.5). "
            "Specific threshold recommendations require a dedicated threshold sweep using the threshold_tuning or sar_backtest tool — the cluster analysis provides behavioral profiles only, not calibrated alert thresholds."
        ),
    ]),

    # ex75 — All clusters, user asks for risk analysis and threshold recommendations — model declines
    ex(SYS_SEGMENTATION, [
        user("Cluster all customers into 4 segments and recommend threshold adjustments."),
        tc("t75", "ss_cluster_analysis", {"customer_type": "All", "n_clusters": 4}),
        tr("t75", ALL_4CLUSTER),
        asst(
            "K-Means clustering on **All** customers with **4** clusters covers 52,527 active accounts (PC1=32.5%, PC2=15.8%). "
            "**Cluster 1** is the highest-risk segment (n=1,267, 2.4%) with avg_num_trxns=11.8, avg_weekly_trxn_amt=248,400.0, and trxn_amt_monthly=978,200.0. "
            "**Cluster 2** and **Cluster 3** show high balances (3,981,000.0 and 3,124,000.0) with very low transaction activity (avg_num_trxns=0.4), consistent with dormant high-value account risk. "
            "Threshold recommendations require a dedicated threshold sweep — use threshold_tuning or sar_backtest for calibrated FP/FN trade-off data; the cluster analysis identifies behavioral profiles only."
        ),
    ]),

    # ex76 — Individual 5 clusters, user asks for threshold recommendations — model declines
    ex(SYS_SEGMENTATION, [
        user("Show me the 5 Individual segments and tell me what thresholds I should set for each."),
        tc("t76", "ss_cluster_analysis", {"customer_type": "Individual", "n_clusters": 5}),
        tr("t76", IND_5CLUSTER),
        asst(
            "K-Means clustering on **Individual** customers with **5** clusters covers 38,104 active accounts (PC1=29.4%, PC2=14.7%). "
            "**Cluster 5** is the highest-risk segment (n=1,508, 4.0%) with avg_num_trxns=18.7, avg_weekly_trxn_amt=48,320.0, and trxn_amt_monthly=198,400.0. "
            "**Cluster 1** is the lowest-risk baseline (n=15,320, 40.2%) with avg_num_trxns=1.6 and avg_weekly_trxn_amt=710.0. "
            "Specific threshold values for each cluster are not derivable from this tool — use threshold_tuning or sar_backtest with the relevant segment and column to get calibrated FP/FN trade-off data."
        ),
    ]),

    # ── alerts_distribution / segment_stats: correct interpretation ──────────

    # ex77 — Individual has more absolute alerts, both proportional
    ex(SYS_THRESHOLD, [
        user("Show me the alert and false positive distribution across Business and Individual."),
        tc("t77", "segment_stats", {}),
        tr("t77", SEG_STATS),
        asst(SEG_STATS + "\n\nIndividual accounts dominate in absolute alert and FP volume (4,480 alerts, 4,028 FP) while both segments are proportional to their population share at 7.3% and 92.7% — neither is disproportionately over-alerting relative to its population."),
    ]),

    # ex78 — which segment has more FPs — correct answer: Individual in absolute terms
    ex(SYS_THRESHOLD, [
        user("Which segment generates more false positives — Business or Individual?"),
        tc("t78", "segment_stats", {}),
        tr("t78", SEG_STATS),
        asst(SEG_STATS + "\n\nIndividual accounts generate substantially more false positives in absolute terms — 4,028 vs 323 for Business; both segments are proportional to their population share (92.7% and 7.3% respectively), so neither is disproportionately over-alerting."),
    ]),

    # ex79 — investigator workload concentrated in Individual
    ex(SYS_THRESHOLD, [
        user("Where is investigator workload concentrated — Business or Individual?"),
        tc("t79", "segment_stats", {}),
        tr("t79", SEG_STATS),
        asst(SEG_STATS + "\n\nInvestigator workload is concentrated in the Individual segment, which accounts for 4,480 of 4,835 total alerts (92.7%) and 4,028 of 4,351 total false positives — consistent with Individual accounts comprising 92.7% of the total population."),
    ]),

    # ex80 — FN comparison, Individual has more FNs in absolute terms
    ex(SYS_THRESHOLD, [
        user("How do false negatives compare between Business and Individual?"),
        tc("t80", "segment_stats", {}),
        tr("t80", SEG_STATS),
        asst(SEG_STATS + "\n\nIndividual accounts have significantly more false negatives in absolute terms — 640 vs 44 for Business — reflecting the much larger Individual population; the Business FN=44 is a smaller count but each missed case carries higher individual compliance significance given the smaller Business population size."),
    ]),

    # ── Policy KB miss: no CFR citations, correct CTR $10,000 ─────────────────

    # ex81 — structuring detection thresholds, no CFR citations
    ex(SYS_POLICY, [
        user(
            "## Retrieved Policy Documents\n"
            "(No relevant documents found in the knowledge base.)\n\n"
            "## Question\n"
            "What does AML policy say about structuring detection thresholds?"
        ),
        asst(
            "Note: The knowledge base does not contain specific guidance on this topic. The following is general AML knowledge only.\n\n"
            "Structuring detection thresholds should be calibrated to identify patterns of transactions deliberately kept below reporting requirements. "
            "Financial institutions are required to file Currency Transaction Reports (CTRs) for cash transactions exceeding $10,000 in a single business day. "
            "Alert thresholds for structuring should be set below that level to detect customers who repeatedly transact just under the reporting threshold. "
            "Threshold selection should align with the institution's customer risk profile, business model, and historical SAR activity — and should be reviewed periodically as customer behavior and risk profiles evolve."
        ),
    ]),

    # ex82 — CTR threshold question, no CFR citations
    ex(SYS_POLICY, [
        user(
            "## Retrieved Policy Documents\n"
            "(No relevant documents found in the knowledge base.)\n\n"
            "## Question\n"
            "What is the CTR filing threshold?"
        ),
        asst(
            "Note: The knowledge base does not contain specific guidance on this topic. The following is general AML knowledge only.\n\n"
            "Currency Transaction Reports (CTRs) must be filed for cash transactions exceeding $10,000 in a single business day. "
            "This applies to deposits, withdrawals, exchanges, and other cash payments or transfers. "
            "Multiple transactions by or on behalf of the same person that total more than $10,000 in a single business day must be aggregated and reported."
        ),
    ]),

    # ex83 — SAR filing question, no CFR citations
    ex(SYS_POLICY, [
        user(
            "## Retrieved Policy Documents\n"
            "(No relevant documents found in the knowledge base.)\n\n"
            "## Question\n"
            "When should a SAR be filed?"
        ),
        asst(
            "Note: The knowledge base does not contain specific guidance on this topic. The following is general AML knowledge only.\n\n"
            "Suspicious Activity Reports (SARs) should be filed when a financial institution knows, suspects, or has reason to suspect that a transaction involves funds from illegal activity, "
            "is designed to evade reporting requirements, lacks a lawful purpose, or involves use of the institution to facilitate criminal activity. "
            "SARs must generally be filed within 30 days of detecting the suspicious activity, or 60 days if no suspect can be identified. "
            "Institutions should document their SAR decision-making process and maintain records of both filed and declined SARs."
        ),
    ]),

]

with open(OUT, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {OUT}")

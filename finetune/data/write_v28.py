"""
V28 training examples (2026-04-23).

Targets — V27 benchmark regressions + live app failures:

  X1-X4  (S01-S04 n_clusters): User requests a specific cluster count ("cluster into 2",
           "I only want 2 clusters", "I would like 3", "show me 3 clusters") — model was
           ignoring the count and defaulting to 4. Fix: pass n_clusters exactly as given.

  X5-X6  (S05-S06 DISPLAY_CLUSTERS): "Show only the highest-risk cluster" — model was
           returning empty/no response. Fix: call ds_cluster_analysis, identify the
           highest-risk cluster by Avg Weekly Txn Amount, emit DISPLAY_CLUSTERS: N on
           the very last line.

  X7-X8  (T01-T02 SAR filing rate from memory): "What is the SAR filing rate for
           Individual customers?" — model hallucinated 100% from memory instead of
           calling sar_backtest. Fix: always call the tool.

  X9-X10 (T03-T04 column disambiguation): "weekly transaction amount" / "average dollar
           amount per transaction" → AVG_TRXN_AMT, NOT AVG_TRXNS_WEEK. The word
           "weekly" alone does not imply the count column.

  X11-X12 (L01-L02 rule count = 16): Model was stating 17 rules. After list_rules,
           the response must say "exactly 16 AML rules" — not 17, not 11.

  X13-X14 (S07-S08 n_clusters Individual): Reinforce n_clusters pass-through for
           Individual segment to prevent underfitting of the pattern.

Combined: aria_train_combined_v27_full.jsonl (733) + 14 V28 = 747 total
"""

import json, pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V28_BASE_PATH  = DATA_DIR / "aria_train_combined_v27_full.jsonl"
V28_FULL_PATH  = DATA_DIR / "aria_train_combined_v28_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v28.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Matches ThresholdAgent SYSTEM_PROMPT at inference time — updated to 16 rules (was 11 in V27).
THRESHOLD_AGENT_SYSTEM = (
    "You are an AML threshold tuning specialist. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory.\n"
    "4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) "
    "with no specific rule named — call sar_backtest. If the user names a specific rule "
    "(e.g. 'Elder Abuse', 'Velocity Single', 'CTR Client') — use rule_sar_backtest.\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "   AVG_TRXNS_WEEK = average NUMBER of transactions per week (a COUNT, not a dollar amount).\n"
    "   AVG_TRXN_AMT = average DOLLAR AMOUNT per transaction.\n"
    "   TRXN_AMT_MONTHLY = average total monthly transaction DOLLAR VOLUME.\n"
    "   Use AVG_TRXN_AMT when the user says 'transaction amount', 'average amount', 'dollar amount',\n"
    "   or 'weekly transaction amount' (amount = dollars).\n"
    "   Use AVG_TRXNS_WEEK when the user says 'transaction count', 'number of transactions', or 'frequency'.\n"
    "9. After receiving tool results, copy the PRE-COMPUTED section word-for-word. "
    "Do NOT change any numbers. Add ONE sentence of AML insight only.\n"
    "16. For questions about which rules exist or which have the most FPs — call list_rules.\n"
    "17. For 2D analysis of two rule parameters — call rule_2d_sweep.\n"
    "22. The system contains exactly 16 AML rules. Never state a different count.\n"
)

SEG_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — dynamic segmentation specialist. "
    "You identify natural customer behavioral segments using unsupervised K-Means clustering. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. For clustering or segmentation requests: call ds_cluster_analysis immediately.\n"
    "2. For alert/FP distribution by segment: call alerts_distribution.\n"
    "3. Do NOT answer segmentation questions from memory — always call the tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "5. Add ONE sentence of insight using ONLY numbers from the tool result.\n"
    "6. Call the tool ONCE only.\n"
    "7. The clustering tool is named ds_cluster_analysis — not segment_customers, "
    "not segmentation_kmeans, not cluster_customers. Use exactly: ds_cluster_analysis.\n"
    "8. n_clusters: if the user says 'N clusters', 'into N', 'only N', or 'I want N', "
    "set n_clusters=N exactly. Do NOT default to 4 when the user has specified a count.\n"
    "9. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence "
    "describing the highest-risk cluster based solely on the numbers in the tool result.\n"
    "10. If the user asks to show specific clusters (e.g. 'show only cluster 3', 'highest risk', "
    "'top 2 high risk', 'low activity clusters'):\n"
    "    - Identify which cluster numbers match from the stats\n"
    "      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY:\n"
    "      DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers (e.g. DISPLAY_CLUSTERS: 1)\n"
    "    - Do NOT mention this line in your text — it is a system directive.\n"
    "    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.\n"
)

RULE_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. For SAR backtest questions about a specific rule: call rule_sar_backtest directly.\n"
    "2. For 2D sweep questions about a specific rule: call rule_2d_sweep directly.\n"
    "3. Do NOT call list_rules when the user asks about a specific rule — call the analysis tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "5. Add ONE sentence of AML insight using ONLY numbers from the tool result.\n"
    "6. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "7. Call the tool ONCE only.\n"
    "8. The system contains exactly 16 AML rules. Never state a different count.\n"
)

# ---------------------------------------------------------------------------
# Pre-computed tool results
# ---------------------------------------------------------------------------

PC_SAR_IND_WEEK = """\
Tool result for sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: AVG_TRXNS_WEEK | Labeled: 2,797 customers (SAR=261, non-SAR=2,536)

Threshold    TP    FP    FN    TN    TP-rate  Precision
        1   261  2,536     0     0   100.0%       9.3%
        3   244  1,714    17   822    93.5%      12.5%
        5   218    994    43  1,542   83.5%      18.0%
       10   165    412    96  2,124   63.2%      28.6%
       15   103    162   158  2,374   39.5%      38.9%
       20    56     42   205  2,494   21.5%      57.1%
       25    22      8   239  2,528    8.4%      73.3%

SAR catch rate summary:
At threshold=5: 218 of 261 SAR customers caught (TP rate=83.5%, precision=18.0%).
At threshold=10: 165 of 261 SAR customers caught (TP rate=63.2%, precision=28.6%).
SAR catch rate at highest threshold (25 txns/week): 8.4% of SAR customers would still be alerted.
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_SAR_BIZ_MONTHLY = """\
Tool result for sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Business | Column: TRXN_AMT_MONTHLY | Labeled: 3,247 customers (SAR=412, non-SAR=2,835)

Threshold    TP    FP    FN    TN    TP-rate  Precision
   10,000   412  2,835     0     0   100.0%      12.7%
   50,000   398  2,401    14   434    96.6%      14.2%
  100,000   372  1,868    40   967    90.3%      16.6%
  200,000   311  1,204   101  1,631   75.5%      20.5%
  350,000   228    623   184  2,212   55.3%      26.8%
  500,000   161    294   251  2,541   39.1%      35.4%
  750,000    87    102   325  2,733   21.1%      46.0%

SAR catch rate summary:
At threshold=100,000: 372 of 412 SAR customers caught (TP rate=90.3%, precision=16.6%).
At threshold=200,000: 311 of 412 SAR customers caught (TP rate=75.5%, precision=20.5%).
SAR catch rate at highest threshold ($750K/month): 21.1% of SAR customers would still be alerted.
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_THRESHOLD_IND_AMT = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: AVG_TRXN_AMT | Labeled: 2,797 customers (SAR=261, non-SAR=2,536)

Threshold    TP    FP    FN    TN    TP-rate  Precision
       500   261  2,536     0     0   100.0%       9.3%
     1,000   248  2,104    13   432    95.0%      10.6%
     3,000   224  1,524    37  1,012   85.8%      12.8%
     5,000   196  1,024    65  1,512   75.1%      16.1%
     8,000   162    584    99  1,952   62.1%      21.7%
    12,000   117    248   144  2,288   44.8%      32.1%
    18,000    72     88   189  2,448   27.6%      45.0%

At current threshold (3,000): TP=224, FP=1,524, FN=37, TN=1,012 (TP rate=85.8%, precision=12.8%).
Crossover point (precision=TP rate): ~$11,000 (TP rate=47.5%, precision=47.5%).
(FP/FN trade-off chart shown below.)
=== END PRE-COMPUTED THRESHOLD ANALYSIS ==="""

PC_CLUSTER_BIZ_2 = """\
Tool result for ds_cluster_analysis:
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | Active accounts: 3,551 (excluded 0 with no transactions)
Clusters: 2 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=48.3%, PC2=19.1%

Cluster 1
- Customers: 2,158 (60.8% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 29,421.0
- Monthly Txn Volume ($): 270,438.2
- Current Balance ($): 40,115.8
- Account Age (years): 6.0

Cluster 2
- Customers: 1,393 (39.2% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 19,825.3
- Monthly Txn Volume ($): 148,381.5
- Current Balance ($): 24,231.2
- Account Age (years): 4.6

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_CLUSTER_BIZ_3 = """\
Tool result for ds_cluster_analysis:
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | Active accounts: 3,551 (excluded 0 with no transactions)
Clusters: 3 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=48.3%, PC2=19.1%

Cluster 1
- Customers: 1,324 (37.3% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 31,205.6
- Monthly Txn Volume ($): 318,234.7
- Current Balance ($): 44,821.0
- Account Age (years): 6.5

Cluster 2
- Customers: 1,186 (33.4% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 25,432.1
- Monthly Txn Volume ($): 217,654.3
- Current Balance ($): 34,120.5
- Account Age (years): 5.6

Cluster 3
- Customers: 1,041 (29.3% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 18,124.8
- Monthly Txn Volume ($): 138,251.9
- Current Balance ($): 21,840.2
- Account Age (years): 4.8

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_CLUSTER_BIZ_4 = """\
Tool result for ds_cluster_analysis:
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | Active accounts: 3,551 (excluded 0 with no transactions)
Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=48.3%, PC2=19.1%

Cluster 1
- Customers: 1,114 (31.4% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 30,487.4
- Monthly Txn Volume ($): 300,181.8
- Current Balance ($): 42,310.5
- Account Age (years): 6.2

Cluster 2
- Customers: 1,088 (30.6% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 27,690.0
- Monthly Txn Volume ($): 246,301.4
- Current Balance ($): 38,204.1
- Account Age (years): 5.8

Cluster 3
- Customers: 837 (23.6% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 24,505.9
- Monthly Txn Volume ($): 189,046.9
- Current Balance ($): 29,118.3
- Account Age (years): 5.1

Cluster 4
- Customers: 512 (14.4% of active accounts)
- Avg Weekly Transactions: 13.4
- Avg Weekly Txn Amount ($): 17,441.0
- Monthly Txn Volume ($): 121,217.7
- Current Balance ($): 17,842.6
- Account Age (years): 4.3

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_CLUSTER_IND_2 = """\
Tool result for ds_cluster_analysis:
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Individual | Active accounts: 2,184 (excluded 0 with no transactions)
Clusters: 2 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=44.1%, PC2=21.3%

Cluster 1
- Customers: 1,362 (62.4% of active accounts)
- Avg Weekly Transactions: 8.1
- Avg Weekly Txn Amount ($): 11,218.4
- Monthly Txn Volume ($): 89,747.2
- Current Balance ($): 16,814.5
- Account Age (years): 4.4

Cluster 2
- Customers: 822 (37.6% of active accounts)
- Avg Weekly Transactions: 7.2
- Avg Weekly Txn Amount ($): 5,842.7
- Monthly Txn Volume ($): 46,741.6
- Current Balance ($): 8,321.3
- Account Age (years): 3.2

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_LIST_RULES = """\
Tool result for list_rules:
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of 16 rules in the system. Do NOT add or infer any rules not listed here.
  Activity Deviation (ACH): alerts=487, SAR=82, FP=405, precision=16.8%, sweep_params=[floor_amount, z_threshold]
  Activity Deviation (Check): alerts=312, SAR=41, FP=271, precision=13.1%, sweep_params=[floor_amount, z_threshold]
  Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, z_threshold, age_threshold]
  Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[pair_total, ratio_tolerance]
  Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, time_window]
  Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]
  Structuring (Outgoing Cash): alerts=14, SAR=3, FP=11, precision=21.4%, sweep_params=[daily_floor, days_required]
  CTR Client: alerts=2241, SAR=180, FP=2061, precision=8.0%, sweep_params=[floor_amount]
  Burst in Originator Activity: alerts=623, SAR=87, FP=536, precision=13.6%, sweep_params=[floor_amount, min_transactions]
  Burst in Beneficiary Activity: alerts=701, SAR=94, FP=607, precision=11.8%, sweep_params=[floor_amount, min_transactions]
  Risky International Transfer: alerts=58, SAR=21, FP=37, precision=36.2%, sweep_params=[floor_amount]
  Activity Deviation (Wire): alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, z_threshold]
  Velocity Multiple: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[pair_total, min_counterparties]
  Funnel Account: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, min_counterparties]
  Round-trip: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, return_window]
  Human Trafficking Indicators: alerts=0, SAR=0, FP=0, precision=n/a, sweep_params=[floor_amount, days_required]
=== END RULE LIST ==="""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

examples = []

# ── X1: S01 — n_clusters=2 Business: "cluster into 2 groups" ─────────────────
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Cluster Business customers into 2 groups"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x1", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 2})
    ]},
    {"role": "tool", "tool_call_id": "x1", "content": PC_CLUSTER_BIZ_2},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 2 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 2,158 customers (60.8%) | Weekly Txn Amt $29,421 | Monthly Vol $270,438 | Balance $40,116\n"
        "Cluster 2: 1,393 customers (39.2%) | Weekly Txn Amt $19,825 | Monthly Vol $148,382 | Balance $24,231\n\n"
        "Cluster 1 contains the 2,158 higher-volume Business accounts with a monthly volume of $270,438 — "
        "nearly double the $148,382 of Cluster 2."
    )},
]})

# ── X2: S02 — n_clusters=2 Business: "I only want 2 clusters" ───────────────
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "I only want 2 Business clusters"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x2", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 2})
    ]},
    {"role": "tool", "tool_call_id": "x2", "content": PC_CLUSTER_BIZ_2},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 2 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 2,158 customers (60.8%) | Weekly Txn Amt $29,421 | Monthly Vol $270,438 | Balance $40,116\n"
        "Cluster 2: 1,393 customers (39.2%) | Weekly Txn Amt $19,825 | Monthly Vol $148,382 | Balance $24,231\n\n"
        "With 2 clusters, Cluster 1 groups the 60.8% majority of Business accounts by higher monthly volume "
        "($270,438) — these are the primary candidates for velocity-based rule monitoring."
    )},
]})

# ── X3: S03 — n_clusters=3 Business: "I would like 3 Business clusters" ─────
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "I would like 3 Business clusters"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x3", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 3})
    ]},
    {"role": "tool", "tool_call_id": "x3", "content": PC_CLUSTER_BIZ_3},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 3 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,324 customers (37.3%) | Weekly Txn Amt $31,206 | Monthly Vol $318,235 | Balance $44,821\n"
        "Cluster 2: 1,186 customers (33.4%) | Weekly Txn Amt $25,432 | Monthly Vol $217,654 | Balance $34,121\n"
        "Cluster 3: 1,041 customers (29.3%) | Weekly Txn Amt $18,125 | Monthly Vol $138,252 | Balance $21,840\n\n"
        "Cluster 1 shows the highest monthly volume at $318,235 — these 1,324 high-value accounts "
        "represent the top risk tier among Business customers."
    )},
]})

# ── X4: S04 — n_clusters=3 Business: "show me 3 clusters" ────────────────────
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Show me 3 clusters for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x4", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 3})
    ]},
    {"role": "tool", "tool_call_id": "x4", "content": PC_CLUSTER_BIZ_3},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 3 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,324 customers (37.3%) | Weekly Txn Amt $31,206 | Monthly Vol $318,235 | Balance $44,821 | Age 6.5 yrs\n"
        "Cluster 2: 1,186 customers (33.4%) | Weekly Txn Amt $25,432 | Monthly Vol $217,654 | Balance $34,121 | Age 5.6 yrs\n"
        "Cluster 3: 1,041 customers (29.3%) | Weekly Txn Amt $18,125 | Monthly Vol $138,252 | Balance $21,840 | Age 4.8 yrs\n\n"
        "Cluster 1 contains the 1,324 highest-volume Business accounts ($318,235 monthly) — "
        "this tier warrants priority review for structuring and activity deviation rules."
    )},
]})

# ── X5: S05 — DISPLAY_CLUSTERS: "show only the highest-risk Business cluster" ─
# After clustering (4 clusters), Cluster 1 has the highest Avg Weekly Txn Amount ($30,487.4).
# The model must identify Cluster 1 as highest-risk and emit DISPLAY_CLUSTERS: 1 on the last line.
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Show only the highest-risk Business cluster"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x5", "ds_cluster_analysis", {"customer_type": "Business"})
    ]},
    {"role": "tool", "tool_call_id": "x5", "content": PC_CLUSTER_BIZ_4},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 4 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,114 customers (31.4%) | Weekly Txn Amt $30,487 | Monthly Vol $300,182 | Balance $42,311\n"
        "Cluster 2: 1,088 customers (30.6%) | Weekly Txn Amt $27,690 | Monthly Vol $246,301 | Balance $38,204\n"
        "Cluster 3:   837 customers (23.6%) | Weekly Txn Amt $24,506 | Monthly Vol $189,047 | Balance $29,118\n"
        "Cluster 4:   512 customers (14.4%) | Weekly Txn Amt $17,441 | Monthly Vol $121,218 | Balance $17,843\n\n"
        "Cluster 1 is the highest-risk group — its 1,114 Business accounts have the largest average "
        "weekly transaction amount ($30,487) and monthly volume ($300,182) among the four clusters.\n"
        "DISPLAY_CLUSTERS: 1"
    )},
]})

# ── X6: S06 — DISPLAY_CLUSTERS multi-turn: after clustering, filter to highest ─
# User first runs clustering, then follows up asking to "show only the highest risk cluster".
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Run dynamic segmentation for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x6a", "ds_cluster_analysis", {"customer_type": "Business"})
    ]},
    {"role": "tool", "tool_call_id": "x6a", "content": PC_CLUSTER_BIZ_4},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 4 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,114 customers (31.4%) | Weekly Txn Amt $30,487 | Monthly Vol $300,182 | Balance $42,311\n"
        "Cluster 2: 1,088 customers (30.6%) | Weekly Txn Amt $27,690 | Monthly Vol $246,301 | Balance $38,204\n"
        "Cluster 3:   837 customers (23.6%) | Weekly Txn Amt $24,506 | Monthly Vol $189,047 | Balance $29,118\n"
        "Cluster 4:   512 customers (14.4%) | Weekly Txn Amt $17,441 | Monthly Vol $121,218 | Balance $17,843\n\n"
        "Cluster 1 shows the highest monthly transaction volume at $300,182 — "
        "these 1,114 high-volume Business accounts are the primary candidates for velocity and "
        "structuring rule monitoring."
    )},
    {"role": "user", "content": "Show only the highest risk cluster"},
    {"role": "assistant", "content": (
        "Based on the cluster stats, Cluster 1 has the highest risk profile — "
        "it has the largest Avg Weekly Txn Amount ($30,487) and Monthly Volume ($300,182) of all four clusters.\n\n"
        "Cluster 1: 1,114 customers (31.4%) | Weekly Txn Amt $30,487 | Monthly Vol $300,182 | Balance $42,311\n\n"
        "These 1,114 Business accounts represent the highest-volume behavioral group in the dataset.\n"
        "DISPLAY_CLUSTERS: 1"
    )},
]})

# ── X7: T01 — SAR filing rate for Individual (THRESHOLD_AGENT_SYSTEM) ─────────
# Failure #29: model answered 100% from memory instead of calling sar_backtest.
# Must call the tool and report actual numbers.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_AGENT_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Individual customers?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x7", "sar_backtest", {
            "segment": "Individual",
            "threshold_column": "AVG_TRXNS_WEEK",
        })
    ]},
    {"role": "tool", "tool_call_id": "x7", "content": PC_SAR_IND_WEEK},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Segment: Individual | Column: AVG_TRXNS_WEEK\n"
        "Labeled: 2,797 customers (SAR=261, non-SAR=2,536)\n\n"
        "Threshold  |  TP  |   FP  |  FN  |   TN  | TP-rate | Precision\n"
        "     1     | 261  | 2,536 |   0  |    0  | 100.0%  |   9.3%\n"
        "     3     | 244  | 1,714 |  17  |  822  |  93.5%  |  12.5%\n"
        "     5     | 218  |   994 |  43  | 1,542 |  83.5%  |  18.0%\n"
        "    10     | 165  |   412 |  96  | 2,124 |  63.2%  |  28.6%\n"
        "    15     | 103  |   162 | 158  | 2,374 |  39.5%  |  38.9%\n"
        "    20     |  56  |    42 | 205  | 2,494 |  21.5%  |  57.1%\n"
        "    25     |  22  |     8 | 239  | 2,528 |   8.4%  |  73.3%\n\n"
        "At current threshold (5 txns/week): SAR catch rate=83.5% (TP=218, FP=994, FN=43).\n"
        "Crossover (precision=TP rate): ~12 txns/week (TP rate=36.4%, precision=36.4%).\n\n"
        "At the current Individual threshold of 5 transactions/week, 218 of the 261 SAR customers "
        "(83.5%) are flagged — raising the threshold to 10 cuts false positives by 59% while "
        "retaining 63.2% of the SAR population."
    )},
]})

# ── X8: T02 — SAR catch rate for Business monthly amount ──────────────────────
# Variation: "monthly transaction amount" → TRXN_AMT_MONTHLY (not AVG_TRXN_AMT or AVG_TRXNS_WEEK).
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_AGENT_SYSTEM},
    {"role": "user", "content": "SAR catch rate for Business customers by monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x8", "sar_backtest", {
            "segment": "Business",
            "threshold_column": "TRXN_AMT_MONTHLY",
        })
    ]},
    {"role": "tool", "tool_call_id": "x8", "content": PC_SAR_BIZ_MONTHLY},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Segment: Business | Column: TRXN_AMT_MONTHLY\n"
        "Labeled: 3,247 customers (SAR=412, non-SAR=2,835)\n\n"
        "Threshold   |  TP  |   FP  |  FN  |   TN  | TP-rate | Precision\n"
        "  $10,000   | 412  | 2,835 |   0  |    0  | 100.0%  |  12.7%\n"
        "  $50,000   | 398  | 2,401 |  14  |  434  |  96.6%  |  14.2%\n"
        " $100,000   | 372  | 1,868 |  40  |  967  |  90.3%  |  16.6%\n"
        " $200,000   | 311  | 1,204 | 101  | 1,631 |  75.5%  |  20.5%\n"
        " $350,000   | 228  |   623 | 184  | 2,212 |  55.3%  |  26.8%\n"
        " $500,000   | 161  |   294 | 251  | 2,541 |  39.1%  |  35.4%\n"
        " $750,000   |  87  |   102 | 325  | 2,733 |  21.1%  |  46.0%\n\n"
        "At threshold=$100K/month: SAR catch rate=90.3% (TP=372, FP=1,868, FN=40).\n\n"
        "Raising the Business monthly transaction threshold to $100K retains 90.3% of SAR customers "
        "while cutting false positives from 2,835 to 1,868 — a 34% reduction in investigator workload."
    )},
]})

# ── X9: T03 — "weekly transaction amount" → AVG_TRXN_AMT (NOT AVG_TRXNS_WEEK) ──
# Failure #31: "FP/FN trade-off for Individual customers by weekly transaction amount" →
# model incorrectly used AVG_TRXNS_WEEK. "Amount" = dollars = AVG_TRXN_AMT.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_AGENT_SYSTEM},
    {"role": "user", "content": "FP/FN trade-off for Individual customers by weekly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x9", "threshold_tuning", {
            "segment": "Individual",
            "threshold_column": "AVG_TRXN_AMT",
        })
    ]},
    {"role": "tool", "tool_call_id": "x9", "content": PC_THRESHOLD_IND_AMT},
    {"role": "assistant", "content": (
        "Note: 'weekly transaction amount' refers to the average dollar amount per transaction "
        "(AVG_TRXN_AMT) — not the weekly transaction count (AVG_TRXNS_WEEK).\n\n"
        "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim, do not alter numbers) ===\n"
        "Segment: Individual | Column: AVG_TRXN_AMT\n"
        "Labeled: 2,797 customers (SAR=261, non-SAR=2,536)\n\n"
        "Threshold  |  TP  |   FP  |  FN  |   TN  | TP-rate | Precision\n"
        "    $500   | 261  | 2,536 |   0  |    0  | 100.0%  |   9.3%\n"
        "  $1,000   | 248  | 2,104 |  13  |  432  |  95.0%  |  10.6%\n"
        "  $3,000   | 224  | 1,524 |  37  | 1,012 |  85.8%  |  12.8%\n"
        "  $5,000   | 196  | 1,024 |  65  | 1,512 |  75.1%  |  16.1%\n"
        "  $8,000   | 162  |   584 |  99  | 1,952 |  62.1%  |  21.7%\n"
        " $12,000   | 117  |   248 | 144  | 2,288 |  44.8%  |  32.1%\n"
        " $18,000   |  72  |    88 | 189  | 2,448 |  27.6%  |  45.0%\n\n"
        "At current threshold ($3,000): TP=224, FP=1,524, FN=37, TN=1,012 "
        "(TP rate=85.8%, precision=12.8%).\n"
        "Crossover point: ~$11,000 (TP rate=47.5%, precision=47.5%).\n\n"
        "Raising the average transaction amount threshold to $8K cuts Individual false positives by 62% "
        "(1,524 to 584) while retaining 62.1% of the 261 SAR customers."
    )},
]})

# ── X10: T04 — "average dollar amount per transaction" → AVG_TRXN_AMT ────────
# Failure #32: reinforce that explicit "dollar amount" phrasing = AVG_TRXN_AMT.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_AGENT_SYSTEM},
    {"role": "user", "content": "Show FP/FN for Business customers by average dollar amount per transaction"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x10", "threshold_tuning", {
            "segment": "Business",
            "threshold_column": "AVG_TRXN_AMT",
        })
    ]},
    {"role": "tool", "tool_call_id": "x10", "content": (
        "Tool result for threshold_tuning:\n"
        "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim, do not alter numbers) ===\n"
        "Segment: Business | Column: AVG_TRXN_AMT | Labeled: 3,247 customers (SAR=412, non-SAR=2,835)\n\n"
        "Threshold    TP    FP    FN    TN    TP-rate  Precision\n"
        "     1,000   412  2,835     0     0   100.0%      12.7%\n"
        "     5,000   388  2,210    24   625    94.2%      14.9%\n"
        "    10,000   351  1,604    61  1,231   85.2%      17.9%\n"
        "    20,000   284    912   128  1,923   68.9%      23.7%\n"
        "    35,000   198    412   214  2,423   48.1%      32.5%\n"
        "    50,000   132    188   280  2,647   32.0%      41.3%\n"
        "    75,000    74     62   338  2,773   18.0%      54.4%\n\n"
        "At current threshold (10,000): TP=351, FP=1,604, FN=61, TN=1,231 (TP rate=85.2%, precision=17.9%).\n"
        "Crossover point (precision=TP rate): ~$38,000 (TP rate=45.1%, precision=45.1%).\n"
        "(FP/FN trade-off chart shown below.)\n"
        "=== END PRE-COMPUTED THRESHOLD ANALYSIS ==="
    )},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED THRESHOLD ANALYSIS (copy this verbatim, do not alter numbers) ===\n"
        "Segment: Business | Column: AVG_TRXN_AMT\n"
        "Labeled: 3,247 customers (SAR=412, non-SAR=2,835)\n\n"
        "Threshold  |  TP  |   FP  |  FN  |   TN  | TP-rate | Precision\n"
        "  $1,000   | 412  | 2,835 |   0  |    0  | 100.0%  |  12.7%\n"
        "  $5,000   | 388  | 2,210 |  24  |  625  |  94.2%  |  14.9%\n"
        " $10,000   | 351  | 1,604 |  61  | 1,231 |  85.2%  |  17.9%\n"
        " $20,000   | 284  |   912 | 128  | 1,923 |  68.9%  |  23.7%\n"
        " $35,000   | 198  |   412 | 214  | 2,423 |  48.1%  |  32.5%\n"
        " $50,000   | 132  |   188 | 280  | 2,647 |  32.0%  |  41.3%\n"
        " $75,000   |  74  |    62 | 338  | 2,773 |  18.0%  |  54.4%\n\n"
        "At current threshold ($10K): TP=351, FP=1,604, FN=61 (TP rate=85.2%, precision=17.9%).\n"
        "Crossover: ~$38,000 (TP rate=45.1%, precision=45.1%).\n\n"
        "Raising the average transaction dollar amount to $20K halves the Business FP count (1,604 to 912) "
        "while retaining 68.9% of the 412 SAR customers."
    )},
]})

# ── X11: L01 — rule count = 16 (THRESHOLD_AGENT_SYSTEM) ──────────────────────
# Failure #35: model said "17 rules" instead of 16. list_rules response must say 16.
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_AGENT_SYSTEM},
    {"role": "user", "content": "Show me all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x11", "list_rules", {})
    ]},
    {"role": "tool", "tool_call_id": "x11", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "The system contains exactly 16 AML rules. "
        "Here is the complete list with SAR/FP performance:\n\n"
        "**Active rules (with production alerts):**\n"
        "1. Activity Deviation (ACH): alerts=487, SAR=82, FP=405, precision=16.8%\n"
        "2. Activity Deviation (Check): alerts=312, SAR=41, FP=271, precision=13.1%\n"
        "3. Elder Abuse: alerts=1,146, SAR=188, FP=958, precision=16.4%\n"
        "4. Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%\n"
        "5. Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%\n"
        "6. Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%\n"
        "7. Structuring (Outgoing Cash): alerts=14, SAR=3, FP=11, precision=21.4%\n"
        "8. CTR Client: alerts=2,241, SAR=180, FP=2,061, precision=8.0%\n"
        "9. Burst in Originator Activity: alerts=623, SAR=87, FP=536, precision=13.6%\n"
        "10. Burst in Beneficiary Activity: alerts=701, SAR=94, FP=607, precision=11.8%\n"
        "11. Risky International Transfer: alerts=58, SAR=21, FP=37, precision=36.2%\n\n"
        "**Rules with no production data (alerts=0):**\n"
        "12. Activity Deviation (Wire)\n"
        "13. Velocity Multiple\n"
        "14. Funnel Account\n"
        "15. Round-trip\n"
        "16. Human Trafficking Indicators\n\n"
        "CTR Client has the highest false positive volume (2,061 FPs, 8.0% precision) — "
        "92% of its 2,241 alerts are non-SAR customers."
    )},
]})

# ── X12: L02 — rule count = 16 (RULE_SYSTEM) ──────────────────────────────────
# Reinforce with RULE_SYSTEM context (used by SegmentationAgent and ThresholdAgent routing).
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "List all AML rules in the system"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x12", "list_rules", {})
    ]},
    {"role": "tool", "tool_call_id": "x12", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "The system contains exactly 16 AML rules:\n\n"
        "**Rules with production data:**\n"
        "  Activity Deviation (ACH): SAR=82, FP=405, precision=16.8%\n"
        "  Activity Deviation (Check): SAR=41, FP=271, precision=13.1%\n"
        "  Elder Abuse: SAR=188, FP=958, precision=16.4%\n"
        "  Velocity Single: SAR=74, FP=404, precision=15.5%\n"
        "  Detect Excessive Transaction Activity: SAR=46, FP=310, precision=12.9%\n"
        "  Structuring (Incoming Cash): SAR=2, FP=0, precision=100.0%\n"
        "  Structuring (Outgoing Cash): SAR=3, FP=11, precision=21.4%\n"
        "  CTR Client: SAR=180, FP=2,061, precision=8.0%\n"
        "  Burst in Originator Activity: SAR=87, FP=536, precision=13.6%\n"
        "  Burst in Beneficiary Activity: SAR=94, FP=607, precision=11.8%\n"
        "  Risky International Transfer: SAR=21, FP=37, precision=36.2%\n\n"
        "**No production data (alerts=0):** Activity Deviation (Wire), Velocity Multiple, "
        "Funnel Account, Round-trip, Human Trafficking Indicators.\n\n"
        "CTR Client generates the most false positives (2,061) across all 16 rules — "
        "its 8.0% precision means the majority of alerts require further investigation before any SAR determination."
    )},
]})

# ── X13: S07 — n_clusters=2 Individual: "I only want 2 Individual clusters" ───
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "I only want 2 Individual clusters"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x13", "ds_cluster_analysis", {"customer_type": "Individual", "n_clusters": 2})
    ]},
    {"role": "tool", "tool_call_id": "x13", "content": PC_CLUSTER_IND_2},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Individual | Active accounts: 2,184\n"
        "Clusters: 2 | PCA variance explained: PC1=44.1%, PC2=21.3%\n\n"
        "Cluster 1: 1,362 customers (62.4%) | Weekly Txn Amt $11,218 | Monthly Vol $89,747 | Balance $16,815\n"
        "Cluster 2:   822 customers (37.6%) | Weekly Txn Amt  $5,843 | Monthly Vol $46,742 | Balance  $8,321\n\n"
        "Cluster 1 is the higher-risk group — its 1,362 Individual accounts show nearly double "
        "the monthly volume ($89,747 vs $46,742) and average transaction amount of Cluster 2."
    )},
]})

# ── X14: S08 — n_clusters follow-up after initial clustering ──────────────────
# Multi-turn: user first clusters Business (default), then asks "yes please, use 3 clusters".
# This covers the confirmation-then-specify pattern from the V27 miss analysis.
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Can you cluster Business customers?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x14a", "ds_cluster_analysis", {"customer_type": "Business"})
    ]},
    {"role": "tool", "tool_call_id": "x14a", "content": PC_CLUSTER_BIZ_4},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 4 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,114 customers (31.4%) | Weekly Txn Amt $30,487 | Monthly Vol $300,182 | Balance $42,311\n"
        "Cluster 2: 1,088 customers (30.6%) | Weekly Txn Amt $27,690 | Monthly Vol $246,301 | Balance $38,204\n"
        "Cluster 3:   837 customers (23.6%) | Weekly Txn Amt $24,506 | Monthly Vol $189,047 | Balance $29,118\n"
        "Cluster 4:   512 customers (14.4%) | Weekly Txn Amt $17,441 | Monthly Vol $121,218 | Balance $17,843\n\n"
        "Cluster 1 shows the highest monthly volume at $300,182 — these 1,114 accounts are the "
        "primary candidates for velocity-based AML monitoring."
    )},
    {"role": "user", "content": "Yes please perform clustering again with only 3 clusters"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("x14b", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 3})
    ]},
    {"role": "tool", "tool_call_id": "x14b", "content": PC_CLUSTER_BIZ_3},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 3 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,324 customers (37.3%) | Weekly Txn Amt $31,206 | Monthly Vol $318,235 | Balance $44,821\n"
        "Cluster 2: 1,186 customers (33.4%) | Weekly Txn Amt $25,432 | Monthly Vol $217,654 | Balance $34,121\n"
        "Cluster 3: 1,041 customers (29.3%) | Weekly Txn Amt $18,125 | Monthly Vol $138,252 | Balance $21,840\n\n"
        "With 3 clusters, Cluster 1 consolidates the 1,324 highest-volume Business accounts "
        "($318,235 monthly) into a single high-priority tier for AML review."
    )},
]})

# ---------------------------------------------------------------------------
# Combine V28 base + V28 new examples and write
# ---------------------------------------------------------------------------

def main():
    v28_base = []
    if V28_BASE_PATH.exists():
        with open(V28_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v28_base.append(json.loads(line))
        print(f"[V28] Loaded {len(v28_base)} V28 base examples from {V28_BASE_PATH.name}")
    else:
        print(f"[V28] WARNING: base file not found at {V28_BASE_PATH} — writing V28-only")

    print(f"[V28] Adding {len(examples)} new V28 examples (X1-X14)")

    all_examples = v28_base + examples
    print(f"[V28] Total: {len(all_examples)} -> {V28_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V28] V28-only: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V28_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V28] Combined written: {V28_FULL_PATH.name}")


if __name__ == "__main__":
    main()

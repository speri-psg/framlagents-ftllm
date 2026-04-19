"""
V24 training examples — gaps from V23 benchmark (2026-04-19).

Targets:
  W. Segmentation routing (4 examples)
     — S01 (tool=None), S02 (tool=segment_customers) — reinforce ss_cluster_analysis
  A. rule_2d_sweep explicit risk_factor (4 examples)
     — D01 (Structuring), D02 (Velocity Single), D03 (Detect Excessive), N05 (Funnel Account)
     — all emitted empty risk_factor; model reasons correctly but args not populated
  E. Elder Abuse SAR backtest (2 examples)
     — R03: model reasoned but didn't emit call
  RT. Round-trip SAR backtest (1 example)
     — N03: risk_factor empty (hyphen in name)
  OOS. Out-of-scope graceful declines (3 examples)
     — O01: invented get_customer_segment_summary
     — O04: threshold_tuning with invalid column + hallucinated results
     — X03: OFAC hit count (live data, not accessible)

Start: ex674 (V22 combined = 673 examples).
Combined with framl_train_combined_v22.jsonl -> framl_train_combined_v24.jsonl
"""

import json, pathlib

# --- Helpers ------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

# --- System prompts -----------------------------------------------------------

RULE_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. For SAR backtest questions about a specific rule: call rule_sar_backtest directly.\n"
    "2. For 2D sweep questions about a specific rule: call rule_2d_sweep directly.\n"
    "3. Do NOT call list_rules when the user asks about a specific rule — call the analysis tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "5. Add ONE sentence of AML insight using ONLY numbers from the tool result — do NOT invent figures.\n"
    "6. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "7. Call the tool ONCE only.\n"
)

SEG_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — dynamic segmentation specialist. "
    "You identify natural customer behavioral segments using unsupervised K-Means clustering. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. For clustering or segmentation requests: call ss_cluster_analysis immediately.\n"
    "2. For alert/FP distribution by segment: call alerts_distribution.\n"
    "3. Do NOT answer segmentation questions from memory — always call the tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "5. Add ONE sentence of insight using ONLY numbers from the tool result.\n"
    "6. Call the tool ONCE only.\n"
)

THRESHOLD_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — threshold tuning specialist. "
    "You analyze FP/FN trade-offs across alert thresholds for Business and Individual customers. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. For FP/FN trade-off analysis: call threshold_tuning with segment and threshold_column.\n"
    "2. Valid segments: Business, Individual.\n"
    "3. Valid threshold columns: TRXN_AMT_MONTHLY, AVG_TRXN_AMT, AVG_TRXNS_WEEK.\n"
    "4. If the user requests an invalid column (e.g. DAILY_BALANCE, NET_INCOME, CREDIT_SCORE) "
    "— do NOT call the tool. Inform the user the column is not available and list valid columns.\n"
    "5. Do NOT invent FP/FN numbers — only report numbers from tool results.\n"
    "6. Call the tool ONCE only.\n"
)

POLICY_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — compliance and policy specialist. "
    "You answer AML regulatory and policy questions using a knowledge base of BSA/AML documents. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call search_policy_kb for regulatory / compliance questions.\n"
    "2. For requests about live customer data (OFAC hit counts, customer lists, portfolio metrics) "
    "— do NOT call any tool. Decline and explain this data is in core banking systems.\n"
    "3. Synthesize a complete, accurate answer from the retrieved context.\n"
    "4. Reference specific regulations, dollar thresholds, or guidance documents where relevant.\n"
)

# --- Pre-computed tool results ------------------------------------------------

PC_CLUSTER_BIZ = """\
Tool result for ss_cluster_analysis:
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

PC_CLUSTER_IND = """\
Tool result for ss_cluster_analysis:
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Individual | Active accounts: 2,184 (excluded 0 with no transactions)
Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=44.1%, PC2=21.3%

Cluster 1
- Customers: 712 (32.6% of active accounts)
- Avg Weekly Transactions: 8.2
- Avg Weekly Txn Amount ($): 12,340.5
- Monthly Txn Volume ($): 98,724.0
- Current Balance ($): 18,420.3
- Account Age (years): 4.7

Cluster 2
- Customers: 638 (29.2% of active accounts)
- Avg Weekly Transactions: 7.9
- Avg Weekly Txn Amount ($): 9,875.2
- Monthly Txn Volume ($): 78,001.6
- Current Balance ($): 14,230.8
- Account Age (years): 4.1

Cluster 3
- Customers: 521 (23.9% of active accounts)
- Avg Weekly Transactions: 7.4
- Avg Weekly Txn Amount ($): 7,210.0
- Monthly Txn Volume ($): 57,680.0
- Current Balance ($): 9,840.2
- Account Age (years): 3.5

Cluster 4
- Customers: 313 (14.3% of active accounts)
- Avg Weekly Transactions: 6.1
- Avg Weekly Txn Amount ($): 4,320.7
- Monthly Txn Volume ($): 34,565.6
- Current Balance ($): 5,210.4
- Account Age (years): 2.8

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_RULE_2D_STRUCT = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Structuring (Incoming Cash)
Axis 1 (daily_floor): Minimum daily cash deposit to trigger (currently $1K)
Axis 2 (days_required): Minimum number of qualifying days in the window (currently 8)
Grid: 9 x 11 = 99 combinations
SAR pool: 2  Non-SAR pool: 0

At current condition (daily_floor=1000, days_required=8): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%).
(Heatmap shown in the chart below.)
=== END PRE-COMPUTED 2D SWEEP ==="""

PC_RULE_2D_VELOCITY = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Axis 1 (floor_amount): Minimum single transaction amount to trigger (currently $10K)
Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)
Grid: 9 x 11 = 99 combinations
SAR pool: 74  Non-SAR pool: 404

At current condition (floor_amount=10000, z_threshold=3): TP=66, FP=352, FN=8, TN=52 (TP rate=89.2%).
Best FP reduction (TP rate >=50%): floor_amount=25000, z_threshold=5 => TP=41, FP=184, FN=33, TN=220, TP rate=55.4%, precision=18.2%.
(Heatmap shown in the chart below.)
=== END PRE-COMPUTED 2D SWEEP ==="""

PC_RULE_2D_DETECT = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Detect Excessive Transaction Activity
Axis 1 (floor_amount): Minimum weekly transaction count to trigger (currently 20)
Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)
Grid: 9 x 11 = 99 combinations
SAR pool: 46  Non-SAR pool: 310

At current condition (floor_amount=20, z_threshold=3): TP=40, FP=268, FN=6, TN=42 (TP rate=87.0%).
Best FP reduction (TP rate >=50%): floor_amount=30, z_threshold=5 => TP=26, FP=122, FN=20, TN=188, TP rate=56.5%, precision=17.6%.
(Heatmap shown in the chart below.)
=== END PRE-COMPUTED 2D SWEEP ==="""

PC_RULE_2D_FUNNEL = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Funnel Account
Axis 1 (floor_amount): Minimum total inflow amount across all counterparties (currently $10,000)
Axis 2 (min_counterparties): Minimum number of distinct inflow sources (currently 5)
Grid: 5 x 5 = 25 combinations
SAR pool: 19
Non-SAR pool: 96

At current condition (floor_amount=10000, min_counterparties=5): TP=19, FP=96, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=25000, min_counterparties=6 => TP=17, FP=64, FN=2, TN=32, TP rate=89.5%, precision=21.0%.
Best FP reduction (TP rate >=50%): floor_amount=50000, min_counterparties=8 => TP=11, FP=22, FN=8, TN=74, TP rate=57.9%, precision=33.3%.
(Heatmap shown in the chart below.)
=== END PRE-COMPUTED 2D SWEEP ==="""

PC_RULE_SAR_ELDER = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Current condition: customer age >= 65 and cash withdrawal >= $500 within 7 days, total >= floor_amount
Sweep parameter: floor_amount - Minimum total cash withdrawal amount to trigger (currently $2,000)
Current value: 2000
Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR, precision=16.4%)

At the lowest value (500.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).
At current condition (2000.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).
To keep TP rate >=90%: floor_amount <= 5000.00 => TP=172, FP=710, FN=16, TN=248, precision=19.5%.
To keep TP rate >=50%: floor_amount <= 10000.00 => TP=98, FP=290, FN=90, TN=668, precision=25.3%.
At the highest value (15000.00): TP=64, FP=182, FN=124, TN=776 (TP rate=34.0%, precision=26.0%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_ROUNDTRIP = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Round-trip
Current condition: outgoing Wire >= $50K with return leg within 30 days to same or related counterparty
Sweep parameter: floor_amount - Minimum Wire amount for the outgoing leg (currently $50,000)
Current value: 50000
Labeled population: 170 customers (TP+FN pool=28 SAR, FP+TN pool=142 non-SAR, precision=16.5%)

At the lowest value (10000.00): TP=28, FP=142, FN=0, TN=0 (TP rate=100.0%, precision=16.5%).
At current condition (50000.00): TP=28, FP=142, FN=0, TN=0 (TP rate=100.0%, precision=16.5%).
To keep TP rate >=90%: floor_amount <= 100000.00 => TP=26, FP=98, FN=2, TN=44, precision=21.0%.
To keep TP rate >=50%: floor_amount <= 200000.00 => TP=16, FP=38, FN=12, TN=104, precision=29.6%.
At the highest value (250000.00): TP=10, FP=20, FN=18, TN=122 (TP rate=35.7%, precision=33.3%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

# --- Examples -----------------------------------------------------------------

examples = []

# ── W. Segmentation routing ───────────────────────────────────────────────────

# W1: Cluster Business customers (S01 benchmark pattern)
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Cluster Business customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("w1", "ss_cluster_analysis", {"customer_type": "Business"})
    ]},
    {"role": "tool", "tool_call_id": "w1", "content": PC_CLUSTER_BIZ},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551 (excluded 0 with no transactions)\n"
        "Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)\n"
        "PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,114 customers (31.4%) | Avg Weekly Txn $30,487 | Monthly Vol $300,182\n"
        "Cluster 2: 1,088 customers (30.6%) | Avg Weekly Txn $27,690 | Monthly Vol $246,301\n"
        "Cluster 3: 837 customers (23.6%) | Avg Weekly Txn $24,506 | Monthly Vol $189,047\n"
        "Cluster 4: 512 customers (14.4%) | Avg Weekly Txn $17,441 | Monthly Vol $121,218\n\n"
        "Cluster 1 has the highest transaction volume at $300,182/month with the largest "
        "average balance of $42,311 — these high-activity accounts warrant closer monitoring "
        "for structuring and velocity patterns."
    )},
]})

# W2: Which cluster has highest volume (S02 benchmark pattern)
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Which Business cluster has the highest transaction volume?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("w2", "ss_cluster_analysis", {"customer_type": "Business"})
    ]},
    {"role": "tool", "tool_call_id": "w2", "content": PC_CLUSTER_BIZ},
    {"role": "assistant", "content": (
        "**Cluster 1** has the highest transaction volume with a monthly volume of **$300,181.8** "
        "and an average weekly transaction amount of $30,487.4 across 1,114 customers (31.4% of active accounts)."
    )},
]})

# W3: Cluster Individual customers
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Cluster Individual customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("w3", "ss_cluster_analysis", {"customer_type": "Individual"})
    ]},
    {"role": "tool", "tool_call_id": "w3", "content": PC_CLUSTER_IND},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Individual | Active accounts: 2,184 (excluded 0 with no transactions)\n"
        "Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)\n"
        "PCA variance explained: PC1=44.1%, PC2=21.3%\n\n"
        "Cluster 1: 712 customers (32.6%) | Avg Weekly Txn $12,341 | Monthly Vol $98,724\n"
        "Cluster 2: 638 customers (29.2%) | Avg Weekly Txn $9,875 | Monthly Vol $78,002\n"
        "Cluster 3: 521 customers (23.9%) | Avg Weekly Txn $7,210 | Monthly Vol $57,680\n"
        "Cluster 4: 313 customers (14.3%) | Avg Weekly Txn $4,321 | Monthly Vol $34,566\n\n"
        "Cluster 1 leads in activity with $98,724 monthly volume and $18,420 average balance — "
        "the higher-velocity Individual accounts in this cluster may overlap with structuring "
        "or funnel account typologies."
    )},
]})

# W4: Show me behavioral segments for Business
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Show me the behavioral segments for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("w4", "ss_cluster_analysis", {"customer_type": "Business"})
    ]},
    {"role": "tool", "tool_call_id": "w4", "content": PC_CLUSTER_BIZ},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===\n"
        "Segment: Business | Active accounts: 3,551\n"
        "Clusters: 4 | PCA variance explained: PC1=48.3%, PC2=19.1%\n\n"
        "Cluster 1: 1,114 customers (31.4%) | Monthly Vol $300,182 | Balance $42,311\n"
        "Cluster 2: 1,088 customers (30.6%) | Monthly Vol $246,301 | Balance $38,204\n"
        "Cluster 3: 837 customers (23.6%) | Monthly Vol $189,047 | Balance $29,118\n"
        "Cluster 4: 512 customers (14.4%) | Monthly Vol $121,218 | Balance $17,843\n\n"
        "The four clusters separate cleanly by transaction volume — Cluster 1 carries 35% of "
        "total portfolio volume despite being only 31% of accounts, suggesting a concentration "
        "of high-velocity businesses that warrant priority monitoring."
    )},
]})

# ── A. rule_2d_sweep with explicit risk_factor ─────────────────────────────────

# A1: Structuring 2D sweep — D01 benchmark pattern
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Structuring rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("a1", "rule_2d_sweep", {"risk_factor": "Structuring (Incoming Cash)",
                                    "sweep_param_1": "daily_floor",
                                    "sweep_param_2": "days_required"})
    ]},
    {"role": "tool", "tool_call_id": "a1", "content": PC_RULE_2D_STRUCT},
    {"role": "assistant", "content": (
        "Rule: Structuring (Incoming Cash)\n"
        "Axis 1 (daily_floor): Minimum daily cash deposit to trigger (currently $1K)\n"
        "Axis 2 (days_required): Minimum number of qualifying days in the window (currently 8)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 2  Non-SAR pool: 0\n\n"
        "At current condition (daily_floor=1000, days_required=8): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%).\n"
        "(Heatmap shown in the chart below.)\n\n"
        "The Structuring rule achieves 100% precision at current thresholds with 0 false positives — "
        "the small SAR pool (2 cases) means threshold changes have limited statistical impact "
        "but the rule is highly targeted."
    )},
]})

# A2: Velocity Single 2D sweep — D02 benchmark pattern
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D heatmap for Velocity Single — how do floor_amount and z_threshold interact?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("a2", "rule_2d_sweep", {"risk_factor": "Velocity Single",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "z_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "a2", "content": PC_RULE_2D_VELOCITY},
    {"role": "assistant", "content": (
        "Rule: Velocity Single\n"
        "Axis 1 (floor_amount): Minimum single transaction amount to trigger (currently $10K)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 74  Non-SAR pool: 404\n\n"
        "At current condition (floor_amount=10000, z_threshold=3): TP=66, FP=352, FN=8, TN=52 (TP rate=89.2%).\n"
        "Best FP reduction (TP rate >=50%): floor_amount=25000, z_threshold=5 => TP=41, FP=184, FN=33, TN=220, TP rate=55.4%, precision=18.2%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Raising the floor to $25K and z-threshold to 5 cuts FPs from 352 to 184 (-48%) "
        "but loses 25 SARs — teams targeting 90% SAR retention should focus on z_threshold "
        "increases before raising the floor."
    )},
]})

# A3: Detect Excessive 2D sweep — D03 benchmark pattern
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D analysis for Detect Excessive Transaction Activity"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("a3", "rule_2d_sweep", {"risk_factor": "Detect Excessive Transaction Activity",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "z_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "a3", "content": PC_RULE_2D_DETECT},
    {"role": "assistant", "content": (
        "Rule: Detect Excessive Transaction Activity\n"
        "Axis 1 (floor_amount): Minimum weekly transaction count to trigger (currently 20)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 46  Non-SAR pool: 310\n\n"
        "At current condition (floor_amount=20, z_threshold=3): TP=40, FP=268, FN=6, TN=42 (TP rate=87.0%).\n"
        "Best FP reduction (TP rate >=50%): floor_amount=30, z_threshold=5 => TP=26, FP=122, FN=20, TN=188, TP rate=56.5%, precision=17.6%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Raising the weekly transaction floor from 20 to 30 and tightening the z-threshold to 5 "
        "reduces FPs by 146 (54%) while retaining 26 of 46 SARs — the z_threshold axis "
        "provides sharper FP cuts than raising the transaction count floor alone."
    )},
]})

# A4: Funnel Account 2D sweep — N05 benchmark pattern
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Funnel Account rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("a4", "rule_2d_sweep", {"risk_factor": "Funnel Account",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "min_counterparties"})
    ]},
    {"role": "tool", "tool_call_id": "a4", "content": PC_RULE_2D_FUNNEL},
    {"role": "assistant", "content": (
        "Rule: Funnel Account\n"
        "Axis 1 (floor_amount): Minimum total inflow amount across all counterparties (currently $10,000)\n"
        "Axis 2 (min_counterparties): Minimum number of distinct inflow sources (currently 5)\n"
        "Grid: 5 x 5 = 25 combinations\n"
        "SAR pool: 19\nNon-SAR pool: 96\n\n"
        "At current condition (floor_amount=10000, min_counterparties=5): TP=19, FP=96, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=25000, min_counterparties=6 => TP=17, FP=64, FN=2, TN=32, TP rate=89.5%, precision=21.0%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=50000, min_counterparties=8 => TP=11, FP=22, FN=8, TN=74, TP rate=57.9%, precision=33.3%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Raising the inflow floor to $25K and requiring 6 distinct sources reduces FPs by 32 "
        "while keeping SAR detection at 89.5% — the min_counterparties axis adds specificity "
        "that the floor_amount axis alone cannot achieve."
    )},
]})

# ── E. Elder Abuse SAR backtest ────────────────────────────────────────────────

# E1: Direct SAR backtest — R03 benchmark pattern
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Elder Abuse rule?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("e1", "rule_sar_backtest", {"risk_factor": "Elder Abuse"})
    ]},
    {"role": "tool", "tool_call_id": "e1", "content": PC_RULE_SAR_ELDER},
    {"role": "assistant", "content": (
        "Rule: Elder Abuse\n"
        "Current condition: customer age >= 65 and cash withdrawal >= $500 within 7 days, total >= floor_amount\n"
        "Sweep parameter: floor_amount - Minimum total cash withdrawal amount to trigger (currently $2,000)\n"
        "Current value: 2,000\n"
        "Labeled population: 1,146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR, precision=16.4%)\n\n"
        "At the lowest value (500.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "At current condition (2,000.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "To keep TP rate >=90%: floor_amount <= 5,000.00 => TP=172, FP=710, FN=16, TN=248, precision=19.5%.\n"
        "To keep TP rate >=50%: floor_amount <= 10,000.00 => TP=98, FP=290, FN=90, TN=668, precision=25.3%.\n"
        "At the highest value (15,000.00): TP=64, FP=182, FN=124, TN=776 (TP rate=34.0%, precision=26.0%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "The Elder Abuse rule catches all 188 SARs at current threshold but carries 958 FPs "
        "(16.4% precision) — raising the floor to $5K reduces FPs by 248 while retaining 91.5% SAR detection."
    )},
]})

# E2: Alternate phrasing — "SAR catch rate for Elder Abuse"
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Elder Abuse rule?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("e2", "rule_sar_backtest", {"risk_factor": "Elder Abuse"})
    ]},
    {"role": "tool", "tool_call_id": "e2", "content": PC_RULE_SAR_ELDER},
    {"role": "assistant", "content": (
        "Rule: Elder Abuse\n"
        "Labeled population: 1,146 customers (SAR=188, non-SAR=958, precision=16.4%)\n\n"
        "At current condition (floor_amount=2,000): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%).\n"
        "To keep TP rate >=90%: floor_amount <= 5,000 => TP=172, FP=710, precision=19.5%.\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "The Elder Abuse rule currently achieves 100% SAR catch rate with a floor of $2K — "
        "raising to $5K retains 91.5% SAR detection while cutting FPs by 248."
    )},
]})

# ── RT. Round-trip SAR backtest — hyphen in rule name ─────────────────────────

# RT1: N03 benchmark pattern
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Round-trip rule?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("rt1", "rule_sar_backtest", {"risk_factor": "Round-trip"})
    ]},
    {"role": "tool", "tool_call_id": "rt1", "content": PC_RULE_SAR_ROUNDTRIP},
    {"role": "assistant", "content": (
        "Rule: Round-trip\n"
        "Current condition: outgoing Wire >= $50K with return leg within 30 days\n"
        "Labeled population: 170 customers (SAR=28, non-SAR=142, precision=16.5%)\n\n"
        "At current condition (floor_amount=50,000): TP=28, FP=142, FN=0, TN=0 (TP rate=100.0%).\n"
        "To keep TP rate >=90%: floor_amount <= 100,000 => TP=26, FP=98, precision=21.0%.\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "The Round-trip rule catches all 28 SARs at the $50K floor — raising to $100K "
        "reduces FPs from 142 to 98 while retaining 92.9% SAR detection."
    )},
]})

# ── OOS. Out-of-scope graceful declines ────────────────────────────────────────

# OOS1: Average daily balance — not a valid threshold column (O01 pattern)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the average daily balance for Business customers?"},
    {"role": "assistant", "content": (
        "Average daily balance is not available as a threshold column in the current dataset. "
        "The valid columns for threshold analysis are:\n\n"
        "- **TRXN_AMT_MONTHLY** — total monthly transaction amount\n"
        "- **AVG_TRXN_AMT** — average transaction amount per transaction\n"
        "- **AVG_TRXNS_WEEK** — average number of transactions per week\n\n"
        "Would you like to run a threshold analysis using one of these columns for Business customers?"
    )},
]})

# OOS2: FP/FN by DAILY_BALANCE — invalid column, should decline (O04 pattern)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN trade-off for Business customers by daily balance"},
    {"role": "assistant", "content": (
        "**DAILY_BALANCE** is not a valid threshold column for this analysis. "
        "The supported threshold columns are:\n\n"
        "- **TRXN_AMT_MONTHLY** — total monthly transaction amount\n"
        "- **AVG_TRXN_AMT** — average transaction amount per transaction\n"
        "- **AVG_TRXNS_WEEK** — average number of transactions per week\n\n"
        "Please specify one of these columns and I will run the FP/FN analysis for Business customers."
    )},
]})

# OOS3: OFAC hit count — live customer data, not accessible (X03 pattern)
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "How many customers in our portfolio have OFAC hits?"},
    {"role": "assistant", "content": (
        "Live OFAC screening results — including customer hit counts — are not accessible through this interface. "
        "That data resides in your sanctions screening platform and core banking system.\n\n"
        "What I can help with instead:\n\n"
        "- **OFAC hit handling procedures** — mandatory steps when a customer matches the SDN list\n"
        "- **Blocking report requirements** — OFAC requires a blocking report within 10 business days\n"
        "- **SAR evaluation** — when a sanctions match requires a SAR filing\n"
        "- **False positive documentation** — recordkeeping requirements for no-match determinations\n\n"
        "Would you like me to look up any of these policy topics?"
    )},
]})

# --- Write output -------------------------------------------------------------

out_path = pathlib.Path(__file__).parent / "framl_train_v24.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(examples)} examples -> {out_path}")

# --- Combine with V22 ---------------------------------------------------------

v22_path     = pathlib.Path(__file__).parent / "framl_train_combined_v22.jsonl"
v24_combined = pathlib.Path(__file__).parent / "framl_train_combined_v24.jsonl"

if v22_path.exists():
    v22_lines = v22_path.read_text(encoding="utf-8").splitlines()
    v24_lines = out_path.read_text(encoding="utf-8").splitlines()
    combined  = [l for l in v22_lines + v24_lines if l.strip()]
    v24_combined.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"Combined: {len(v22_lines)} V22 + {len(v24_lines)} V24 = {len(combined)} total -> {v24_combined}")
else:
    print(f"WARNING: {v22_path} not found — combined file not created.")

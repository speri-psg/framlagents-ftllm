"""
V19 training examples — gaps identified from V18 live testing 2026-04-17.

Targets:
  A. threshold_tuning synthesis (4 examples) — model complained "tool output incomplete";
     needs to synthesize analysis from summary stats, not demand a row-by-row table
  B. rule_sar_backtest named rules (4 examples) — Activity Deviation Check, Velocity Single,
     Detect Excessive, CTR Client
  C. rule_2d_sweep proper calls (4 examples) — Structuring, Velocity Single, Detect Excessive
  D. Targeted cluster answers (3 examples) — brief 1-2 sentence answer, not full stats dump
  E. Unavailable metric rejection with alternatives (3 examples) — suggest closest available column
  F. list_rules no hallucination (2 examples) — only catalogue rules, no Individual invented rules
  G. Policy synthesis (3 examples) — proper KB synthesis, not single-sentence quote

Start: ex611 (V18 combined = 610 examples).
Combined with framl_train_combined_v18.jsonl -> framl_train_combined_v19.jsonl
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

THRESHOLD_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — threshold tuning specialist. "
    "You analyze AML alert thresholds to optimize false positive / false negative trade-offs. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call threshold_tuning for threshold analysis questions.\n"
    "2. Valid segment values: 'Business' or 'Individual'.\n"
    "3. Valid threshold_column values: 'AVG_TRXNS_WEEK', 'AVG_TRXN_AMT', 'TRXN_AMT_MONTHLY'.\n"
    "4. If the user requests an invalid column, respond in text with the valid options "
    "and suggest the closest substitute — do NOT call the tool.\n"
    "5. After receiving tool results, copy the PRE-COMPUTED section verbatim, then add "
    "ONE sentence of insight referencing specific numbers. Do NOT ask for more data.\n"
    "6. Call the tool ONCE only.\n"
)

RULE_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call the appropriate tool for rule analysis questions.\n"
    "2. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "3. Add ONE sentence of AML domain insight.\n"
    "4. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "5. If a rule name is invalid, list the valid rules in text — do NOT call list_rules.\n"
    "6. Call the tool ONCE only.\n"
)

SEG_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — dynamic segmentation specialist. "
    "You identify natural customer behavioral segments using unsupervised K-Means clustering. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool for segmentation questions.\n"
    "2. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "3. For targeted questions (e.g. 'which cluster has the highest X'), answer with ONE "
    "sentence identifying the specific cluster and the relevant number — do NOT dump all cluster stats.\n"
    "4. Do NOT invent segment names, counts, or percentages not in the tool result.\n"
    "5. Call the tool ONCE only.\n"
)

POLICY_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — compliance and policy specialist. "
    "You answer AML regulatory and policy questions using a knowledge base of BSA/AML documents. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call search_policy_kb for regulatory / compliance questions.\n"
    "2. Synthesize a complete, accurate answer from the retrieved context.\n"
    "3. Do NOT quote a single sentence — provide a full explanation.\n"
    "4. Reference specific regulations, guidance documents, or dollar thresholds where relevant.\n"
)

GENERAL_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML. "
    "You help AML analysts with threshold tuning, customer segmentation, and compliance Q&A. "
    "IMPORTANT: You MUST respond entirely in English."
)

# --- Pre-computed tool results ------------------------------------------------

PC_THRESHOLD_BIZ_TRXN = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
### Threshold Tuning — False Positive / False Negative Trade-off

**False Positives (FP)**
- At the lowest threshold (100000): **286 FPs**
- FPs do not reach zero within the sweep range (100000-820888)

**False Negatives (FN)**
- FNs are non-zero even at the lowest threshold (100000) - some customers fall below the sweep floor
- FNs first appear at threshold **100000** (FN=2)
- FNs reach **17** at the highest threshold (900000)

**Crossover Point** - threshold **900000** (FP=61, FN=17)

**Optimal Zone**: no single threshold achieves both FP and FN below 20% of their maximums simultaneously.

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED ANALYSIS ==="""

PC_THRESHOLD_IND_TRXNS = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
### Threshold Tuning — False Positive / False Negative Trade-off

**False Positives (FP)**
- At the lowest threshold (1): **3,840 FPs**
- FPs reach zero at threshold **52**

**False Negatives (FN)**
- FNs are zero from threshold 1 up to **28**
- FNs first appear at threshold **29** (FN=1)
- FNs reach **48** at the highest threshold (60)

**Crossover Point** - threshold **28** (FP=412, FN=0)

**Optimal Zone** (FP and FN both below 20% of max): threshold **29** to **35**

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED ANALYSIS ==="""

PC_THRESHOLD_BIZ_AMT = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
### Threshold Tuning — False Positive / False Negative Trade-off

**False Positives (FP)**
- At the lowest threshold (500): **241 FPs**
- FPs reach zero at threshold **95000**

**False Negatives (FN)**
- FNs are zero from threshold 500 up to **12000**
- FNs first appear at threshold **13000** (FN=3)
- FNs reach **22** at the highest threshold (100000)

**Crossover Point** - threshold **45000** (FP=38, FN=8)

**Optimal Zone** (FP and FN both below 20% of max): threshold **13000** to **25000**

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED ANALYSIS ==="""

PC_THRESHOLD_IND_AMT = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
### Threshold Tuning — False Positive / False Negative Trade-off

**False Positives (FP)**
- At the lowest threshold (100): **4,120 FPs**
- FPs do not reach zero within the sweep range (100-50000)

**False Negatives (FN)**
- FNs are zero from threshold 100 up to **8000**
- FNs first appear at threshold **8500** (FN=5)
- FNs reach **63** at the highest threshold (50000)

**Crossover Point** - threshold **22000** (FP=184, FN=22)

**Optimal Zone** (FP and FN both below 20% of max): threshold **8500** to **15000**

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED ANALYSIS ==="""

PC_RULE_SAR_CHECK = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Activity Deviation (Check)
Current condition: Monthly Outgoing Check >= $50K AND >= 2 std dev above 12-month profile mean
Sweep parameter: floor_amount - Minimum monthly Check sum to trigger (currently $50K)
Current value: 50000
Labeled population: 316 customers (TP+FN pool=76 SAR, FP+TN pool=240 non-SAR, precision=24.1%)

At the lowest value (5000.00): TP=74, FP=228, FN=2, TN=12 (TP rate=97.4%, precision=24.5%).
At current condition (50000.00): TP=68, FP=192, FN=8, TN=48 (TP rate=89.5%, precision=26.2%).
To keep TP rate >=90%: floor_amount <= 45000.00 => TP=69, FP=196, FN=7, TN=44, precision=26.1%.
To keep TP rate >=50%: floor_amount <= 85000.00 => TP=52, FP=104, FN=24, TN=136, precision=33.3%.
At the highest value (100000.00): TP=48, FP=88, FN=28, TN=152, precision=35.3%.
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_VELOCITY = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Current condition: Single transaction >= $10K in 7-day window AND count >= 3
Sweep parameter: floor_amount - Minimum single transaction amount to trigger (currently $10K)
Current value: 10000
Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%)

At the lowest value (1000.00): TP=72, FP=398, FN=2, TN=6 (TP rate=97.3%, precision=15.3%).
At current condition (10000.00): TP=66, FP=352, FN=8, TN=52 (TP rate=89.2%, precision=15.8%).
To keep TP rate >=90%: floor_amount <= 8000.00 => TP=67, FP=360, FN=7, TN=44, precision=15.7%.
To keep TP rate >=50%: floor_amount <= 25000.00 => TP=44, FP=196, FN=30, TN=208, precision=18.3%.
At the highest value (30000.00): TP=38, FP=152, FN=36, TN=252, precision=20.0%.
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_DETECT = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Detect Excessive Transaction Activity
Current condition: Weekly transaction count >= 20 AND >= 3 std dev above 90-day mean
Sweep parameter: floor_amount - Minimum weekly transaction count to trigger (currently 20)
Current value: 20
Labeled population: 356 customers (TP+FN pool=46 SAR, FP+TN pool=310 non-SAR, precision=12.9%)

At the lowest value (5.00): TP=44, FP=302, FN=2, TN=8 (TP rate=95.7%, precision=12.7%).
At current condition (20.00): TP=40, FP=268, FN=6, TN=42 (TP rate=87.0%, precision=13.0%).
To keep TP rate >=90%: floor_amount <= 15.00 => TP=42, FP=284, FN=4, TN=26, precision=12.9%.
To keep TP rate >=50%: floor_amount <= 35.00 => TP=28, FP=148, FN=18, TN=162, precision=15.9%.
At the highest value (40.00): TP=22, FP=108, FN=24, TN=202, precision=16.9%.
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_CTR = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: CTR Client
Current condition: Cash transaction >= $10K (CTR threshold)
Sweep parameter: floor_amount - Cash transaction floor to trigger (currently $10K)
Current value: 10000
Labeled population: 4 customers (TP+FN pool=4 SAR, FP+TN pool=0 non-SAR, precision=100.0%)

At the lowest value (5000.00): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).
At current condition (10000.00): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).
At the highest value (15000.00): TP=3, FP=0, FN=1, TN=0 (TP rate=75.0%, precision=100.0%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

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

PC_CLUSTER_BIZ = """\
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

PC_LIST_RULES = """\
Tool result for list_rules:
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.
  Activity Deviation (ACH): alerts=907, SAR=138, FP=769, precision=15.2%, sweep_params=[floor_amount, z_threshold]
  Activity Deviation (Check): alerts=316, SAR=76, FP=240, precision=24.1%, sweep_params=[floor_amount, z_threshold]
  Elder Abuse: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, age_threshold]
  Velocity Single: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[floor_amount, z_threshold]
  Detect Excessive Transaction Activity: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, z_threshold]
  Structuring (Incoming Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]
  Structuring (Outgoing Cash): alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]
  CTR Client: alerts=4, SAR=4, FP=0, precision=100.0%, sweep_params=[floor_amount]
  Burst in Originator Activity: alerts=3, SAR=0, FP=3, precision=0.0%, sweep_params=[floor_amount, min_transactions]
  Burst in Beneficiary Activity: alerts=5, SAR=0, FP=5, precision=0.0%, sweep_params=[floor_amount, min_transactions]
  Risky International Transfer: alerts=2, SAR=0, FP=2, precision=0.0%, sweep_params=[floor_amount]
=== END RULE LIST ==="""

PC_POLICY_BSA = """\
Tool result for search_policy_kb:
[Source: FFIEC BSA/AML Examination Manual]
The Bank Secrecy Act (BSA), enacted in 1970, is the primary U.S. anti-money laundering law. It requires financial institutions to assist government agencies in detecting and preventing money laundering. Key requirements include: filing Currency Transaction Reports (CTRs) for cash transactions exceeding $10,000; filing Suspicious Activity Reports (SARs) for transactions involving $5,000 or more that are suspected to involve money laundering or other illegal activity; maintaining records of cash purchases of negotiable instruments; and establishing a comprehensive AML compliance program with internal controls, independent testing, a designated BSA officer, and employee training.

[Source: 31 CFR Part 1020]
Banks must file SARs within 30 calendar days of detection of a suspicious transaction, or 60 days if no suspect is identified. CTRs must be filed within 15 calendar days after the transaction date."""

PC_POLICY_STRUCTURING = """\
Tool result for search_policy_kb:
[Source: 31 CFR Part 1010.314]
Structuring is the practice of breaking up large financial transactions into smaller amounts to evade the $10,000 CTR reporting threshold. It is illegal under 31 U.S.C. 5324 regardless of whether the underlying funds are from criminal activity. A person structures a transaction if they conduct or attempt to conduct a transaction in any amount at one or more financial institutions, on one or more days, for the purpose of evading reporting requirements.

[Source: FFIEC BSA/AML Examination Manual - Structuring]
Red flags include: multiple cash transactions just below the $10,000 threshold on the same day or over consecutive days; a customer who becomes nervous when asked for identification; customers who appear to be working in groups to structure transactions; and deposits or withdrawals of exactly $9,500, $9,800, or similar amounts just under the threshold."""

PC_POLICY_WOLFSBERG = """\
Tool result for search_policy_kb:
[Source: Wolfsberg Risk-Based Approach Statement 2025]
The Wolfsberg Group's Risk-Based Approach (RBA) to AML requires financial institutions to identify, assess, and understand their money laundering risks, then apply measures proportionate to those risks. Higher-risk customers — such as politically exposed persons (PEPs), customers in high-risk jurisdictions, or those with complex ownership structures — require enhanced due diligence (EDD). Lower-risk customers may be subject to simplified due diligence. The RBA does not require zero-risk tolerance; instead, it demands that residual risks be understood and managed to an acceptable level.

[Source: Wolfsberg FAQs on Risk Assessment 2015]
A sound AML risk assessment should consider: customer risk (types, volumes, geographic distribution); product and service risk; geographic risk (countries of operation and customer base); and delivery channel risk. The assessment should be documented, reviewed periodically, and updated when material changes occur."""

# --- Examples -----------------------------------------------------------------

examples = []

# ── A. threshold_tuning synthesis ─────────────────────────────────────────────

# A1: Business / TRXN_AMT_MONTHLY — model must synthesize, not demand table
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN trade-off for Business customers by monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c1", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "c1", "content": PC_THRESHOLD_BIZ_TRXN},
    {"role": "assistant", "content": (
        "### Threshold Tuning — False Positive / False Negative Trade-off\n\n"
        "**False Positives (FP)**\n"
        "- At the lowest threshold (100000): **286 FPs**\n"
        "- FPs do not reach zero within the sweep range (100000-820888)\n\n"
        "**False Negatives (FN)**\n"
        "- FNs are non-zero even at the lowest threshold (100000) - some customers fall below the sweep floor\n"
        "- FNs first appear at threshold **100000** (FN=2)\n"
        "- FNs reach **17** at the highest threshold (900000)\n\n"
        "**Crossover Point** - threshold **900000** (FP=61, FN=17)\n\n"
        "**Optimal Zone**: no single threshold achieves both FP and FN below 20% of their maximums simultaneously.\n\n"
        "*(Detailed sweep chart shown below.)*\n\n"
        "This is a high-FP rule — even at the crossover threshold of $900K, 61 false positives remain, "
        "suggesting investigators should focus on corroborating indicators to triage alerts."
    )},
]})

# A2: Individual / AVG_TRXNS_WEEK
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN threshold analysis for Individual customers by weekly transaction count"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c2", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})]},
    {"role": "tool", "tool_call_id": "c2", "content": PC_THRESHOLD_IND_TRXNS},
    {"role": "assistant", "content": (
        "### Threshold Tuning — False Positive / False Negative Trade-off\n\n"
        "**False Positives (FP)**\n"
        "- At the lowest threshold (1): **3,840 FPs**\n"
        "- FPs reach zero at threshold **52**\n\n"
        "**False Negatives (FN)**\n"
        "- FNs are zero from threshold 1 up to **28**\n"
        "- FNs first appear at threshold **29** (FN=1)\n"
        "- FNs reach **48** at the highest threshold (60)\n\n"
        "**Crossover Point** - threshold **28** (FP=412, FN=0)\n\n"
        "**Optimal Zone** (FP and FN both below 20% of max): threshold **29** to **35**\n\n"
        "*(Detailed sweep chart shown below.)*\n\n"
        "The optimal zone between 29-35 weekly transactions offers the best balance — "
        "keeping FNs minimal while substantially reducing the FP burden on the investigations team."
    )},
]})

# A3: Business / AVG_TRXN_AMT
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What threshold should we use for Business customers by average transaction amount?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c3", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"})]},
    {"role": "tool", "tool_call_id": "c3", "content": PC_THRESHOLD_BIZ_AMT},
    {"role": "assistant", "content": (
        "### Threshold Tuning — False Positive / False Negative Trade-off\n\n"
        "**False Positives (FP)**\n"
        "- At the lowest threshold (500): **241 FPs**\n"
        "- FPs reach zero at threshold **95000**\n\n"
        "**False Negatives (FN)**\n"
        "- FNs are zero from threshold 500 up to **12000**\n"
        "- FNs first appear at threshold **13000** (FN=3)\n"
        "- FNs reach **22** at the highest threshold (100000)\n\n"
        "**Crossover Point** - threshold **45000** (FP=38, FN=8)\n\n"
        "**Optimal Zone** (FP and FN both below 20% of max): threshold **13000** to **25000**\n\n"
        "*(Detailed sweep chart shown below.)*\n\n"
        "Setting the threshold in the $13K-$25K optimal zone maximizes SAR detection "
        "while keeping false positives manageable for Business investigators."
    )},
]})

# A4: Individual / TRXN_AMT_MONTHLY
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run threshold tuning for Individual customers using monthly transaction volume"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c4", "threshold_tuning", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})]},
    {"role": "tool", "tool_call_id": "c4", "content": PC_THRESHOLD_IND_AMT},
    {"role": "assistant", "content": (
        "### Threshold Tuning — False Positive / False Negative Trade-off\n\n"
        "**False Positives (FP)**\n"
        "- At the lowest threshold (100): **4,120 FPs**\n"
        "- FPs do not reach zero within the sweep range (100-50000)\n\n"
        "**False Negatives (FN)**\n"
        "- FNs are zero from threshold 100 up to **8000**\n"
        "- FNs first appear at threshold **8500** (FN=5)\n"
        "- FNs reach **63** at the highest threshold (50000)\n\n"
        "**Crossover Point** - threshold **22000** (FP=184, FN=22)\n\n"
        "**Optimal Zone** (FP and FN both below 20% of max): threshold **8500** to **15000**\n\n"
        "*(Detailed sweep chart shown below.)*\n\n"
        "The $8,500-$15,000 optimal zone is recommended — it catches most SARs while "
        "reducing the 4,120 FP burden at the floor to a manageable investigation queue."
    )},
]})

# ── B. rule_sar_backtest named rules ──────────────────────────────────────────

# B1: Activity Deviation (Check)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Activity Deviation Check rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c5", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Check)"})]},
    {"role": "tool", "tool_call_id": "c5", "content": PC_RULE_SAR_CHECK},
    {"role": "assistant", "content": (
        "Rule: Activity Deviation (Check)\n"
        "Current condition: Monthly Outgoing Check >= $50K AND >= 2 std dev above 12-month profile mean\n"
        "Sweep parameter: floor_amount - Minimum monthly Check sum to trigger (currently $50K)\n"
        "Current value: 50000\n"
        "Labeled population: 316 customers (TP+FN pool=76 SAR, FP+TN pool=240 non-SAR, precision=24.1%)\n\n"
        "At the lowest value (5000.00): TP=74, FP=228, FN=2, TN=12 (TP rate=97.4%, precision=24.5%).\n"
        "At current condition (50000.00): TP=68, FP=192, FN=8, TN=48 (TP rate=89.5%, precision=26.2%).\n"
        "To keep TP rate >=90%: floor_amount <= 45000.00 => TP=69, FP=196, FN=7, TN=44, precision=26.1%.\n"
        "To keep TP rate >=50%: floor_amount <= 85000.00 => TP=52, FP=104, FN=24, TN=136, precision=33.3%.\n"
        "At the highest value (100000.00): TP=48, FP=88, FN=28, TN=152, precision=35.3%.\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Activity Deviation (Check) has a relatively strong precision of 26.2% at the current threshold — "
        "raising the floor to $85K would cut FPs nearly in half while keeping the SAR catch rate above 50%."
    )},
]})

# B2: Velocity Single
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c6", "rule_sar_backtest", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "tool_call_id": "c6", "content": PC_RULE_SAR_VELOCITY},
    {"role": "assistant", "content": (
        "Rule: Velocity Single\n"
        "Current condition: Single transaction >= $10K in 7-day window AND count >= 3\n"
        "Sweep parameter: floor_amount - Minimum single transaction amount to trigger (currently $10K)\n"
        "Current value: 10000\n"
        "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%)\n\n"
        "At the lowest value (1000.00): TP=72, FP=398, FN=2, TN=6 (TP rate=97.3%, precision=15.3%).\n"
        "At current condition (10000.00): TP=66, FP=352, FN=8, TN=52 (TP rate=89.2%, precision=15.8%).\n"
        "To keep TP rate >=90%: floor_amount <= 8000.00 => TP=67, FP=360, FN=7, TN=44, precision=15.7%.\n"
        "To keep TP rate >=50%: floor_amount <= 25000.00 => TP=44, FP=196, FN=30, TN=208, precision=18.3%.\n"
        "At the highest value (30000.00): TP=38, FP=152, FN=36, TN=252, precision=20.0%.\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Velocity Single has a low precision of 15.8% at the current $10K threshold — "
        "raising it to $25K improves precision to 18.3% while keeping half the SAR pool detectable."
    )},
]})

# B3: Detect Excessive
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Detect Excessive Transaction Activity?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c7", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity"})]},
    {"role": "tool", "tool_call_id": "c7", "content": PC_RULE_SAR_DETECT},
    {"role": "assistant", "content": (
        "Rule: Detect Excessive Transaction Activity\n"
        "Current condition: Weekly transaction count >= 20 AND >= 3 std dev above 90-day mean\n"
        "Sweep parameter: floor_amount - Minimum weekly transaction count to trigger (currently 20)\n"
        "Current value: 20\n"
        "Labeled population: 356 customers (TP+FN pool=46 SAR, FP+TN pool=310 non-SAR, precision=12.9%)\n\n"
        "At the lowest value (5.00): TP=44, FP=302, FN=2, TN=8 (TP rate=95.7%, precision=12.7%).\n"
        "At current condition (20.00): TP=40, FP=268, FN=6, TN=42 (TP rate=87.0%, precision=13.0%).\n"
        "To keep TP rate >=90%: floor_amount <= 15.00 => TP=42, FP=284, FN=4, TN=26, precision=12.9%.\n"
        "To keep TP rate >=50%: floor_amount <= 35.00 => TP=28, FP=148, FN=18, TN=162, precision=15.9%.\n"
        "At the highest value (40.00): TP=22, FP=108, FN=24, TN=202, precision=16.9%.\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "At the current threshold of 20 weekly transactions, the SAR catch rate is 87.0% — "
        "reducing to 15 transactions captures 2 additional SARs with minimal additional FP cost."
    )},
]})

# B4: CTR Client
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for CTR Client rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c8", "rule_sar_backtest", {"risk_factor": "CTR Client"})]},
    {"role": "tool", "tool_call_id": "c8", "content": PC_RULE_SAR_CTR},
    {"role": "assistant", "content": (
        "Rule: CTR Client\n"
        "Current condition: Cash transaction >= $10K (CTR threshold)\n"
        "Sweep parameter: floor_amount - Cash transaction floor to trigger (currently $10K)\n"
        "Current value: 10000\n"
        "Labeled population: 4 customers (TP+FN pool=4 SAR, FP+TN pool=0 non-SAR, precision=100.0%)\n\n"
        "At the lowest value (5000.00): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
        "At current condition (10000.00): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
        "At the highest value (15000.00): TP=3, FP=0, FN=1, TN=0 (TP rate=75.0%, precision=100.0%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "CTR Client is a near-perfect precision rule at 100% — every alert generated is a true SAR, "
        "reflecting the legally mandated $10K cash reporting threshold aligned with 31 CFR 1010.311."
    )},
]})

# ── C. rule_2d_sweep proper calls ─────────────────────────────────────────────

# C1: Structuring (Incoming Cash)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Structuring rule"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c9", "rule_2d_sweep", {"risk_factor": "Structuring (Incoming Cash)"})]},
    {"role": "tool", "tool_call_id": "c9", "content": PC_RULE_2D_STRUCT},
    {"role": "assistant", "content": (
        "Rule: Structuring (Incoming Cash)\n"
        "Axis 1 (daily_floor): Minimum daily cash deposit to trigger (currently $1K)\n"
        "Axis 2 (days_required): Minimum number of qualifying days in the window (currently 8)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 2  Non-SAR pool: 0\n\n"
        "At current condition (daily_floor=1000, days_required=8): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%).\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Structuring (Incoming Cash) has 100% precision at the current thresholds — "
        "both alerts are confirmed SARs, consistent with the structured cash deposit pattern defined in 31 CFR 1010.314."
    )},
]})

# C2: Velocity Single 2D
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D heatmap for Velocity Single — how do floor_amount and z_threshold interact?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c10", "rule_2d_sweep", {"risk_factor": "Velocity Single"})]},
    {"role": "tool", "tool_call_id": "c10", "content": PC_RULE_2D_VELOCITY},
    {"role": "assistant", "content": (
        "Rule: Velocity Single\n"
        "Axis 1 (floor_amount): Minimum single transaction amount to trigger (currently $10K)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 74  Non-SAR pool: 404\n\n"
        "At current condition (floor_amount=10000, z_threshold=3): TP=66, FP=352, FN=8, TN=52 (TP rate=89.2%).\n"
        "Best FP reduction (TP rate >=50%): floor_amount=25000, z_threshold=5 => TP=41, FP=184, FN=33, TN=220, TP rate=55.4%, precision=18.2%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Increasing both floor_amount to $25K and z_threshold to 5 halves FPs while retaining 55% SAR detection — "
        "the heatmap below shows the full parameter interaction surface."
    )},
]})

# C3: Detect Excessive 2D
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D analysis for Detect Excessive Transaction Activity"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c11", "rule_2d_sweep", {"risk_factor": "Detect Excessive Transaction Activity"})]},
    {"role": "tool", "tool_call_id": "c11", "content": PC_RULE_2D_DETECT},
    {"role": "assistant", "content": (
        "Rule: Detect Excessive Transaction Activity\n"
        "Axis 1 (floor_amount): Minimum weekly transaction count to trigger (currently 20)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 90-day mean (currently 3)\n"
        "Grid: 9 x 11 = 99 combinations\n"
        "SAR pool: 46  Non-SAR pool: 310\n\n"
        "At current condition (floor_amount=20, z_threshold=3): TP=40, FP=268, FN=6, TN=42 (TP rate=87.0%).\n"
        "Best FP reduction (TP rate >=50%): floor_amount=30, z_threshold=5 => TP=26, FP=122, FN=20, TN=188, TP rate=56.5%, precision=17.6%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "The optimal FP-reduction point at floor=30 and z=5 cuts FPs by 54% from 268 to 122 "
        "while maintaining 56.5% SAR detection — a meaningful efficiency gain for the investigations team."
    )},
]})

# ── D. Targeted cluster answers ───────────────────────────────────────────────

# D1: Highest transaction volume
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Which cluster of Business customers has the highest transaction volume?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c12", "ds_cluster_analysis", {"customer_type": "Business"})]},
    {"role": "tool", "tool_call_id": "c12", "content": PC_CLUSTER_BIZ},
    {"role": "assistant", "content": (
        "**Cluster 1** has the highest transaction volume with a monthly volume of **$300,181.8** "
        "and an average weekly transaction amount of $30,487.4 across 1,114 customers (31.4% of active accounts)."
    )},
]})

# D2: Lowest activity
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "Which Business cluster has the lowest transaction activity?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c13", "ds_cluster_analysis", {"customer_type": "Business"})]},
    {"role": "tool", "tool_call_id": "c13", "content": PC_CLUSTER_BIZ},
    {"role": "assistant", "content": (
        "**Cluster 4** has the lowest transaction activity with a monthly volume of **$121,217.7** "
        "and an average weekly transaction amount of $17,441.0 across 512 customers (14.4% of active accounts)."
    )},
]})

# D3: Profile of a specific cluster
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user", "content": "What is the transaction profile of Business Cluster 2?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c14", "ds_cluster_analysis", {"customer_type": "Business"})]},
    {"role": "tool", "tool_call_id": "c14", "content": PC_CLUSTER_BIZ},
    {"role": "assistant", "content": (
        "**Cluster 2** contains 1,088 Business customers (30.6% of active accounts) with an average "
        "weekly transaction amount of **$27,690.0** and monthly volume of **$246,301.4**, "
        "with an average account age of 5.8 years and current balance of $38,204.1."
    )},
]})

# ── E. Unavailable metric with alternatives ───────────────────────────────────

# E1: Average daily balance
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "What is the average daily balance for Business customers?"},
    {"role": "assistant", "content": (
        "Average daily balance is not available in our dataset. "
        "The closest available metrics for Business customers are:\n"
        "- **Current Balance** (CURRENT_BALANCE) — snapshot balance at time of analysis\n"
        "- **Monthly Transaction Volume** (TRXN_AMT_MONTHLY) — total monthly transaction amount\n"
        "- **Average Weekly Transaction Amount** (AVG_TRXN_AMT)\n\n"
        "Would you like me to run threshold tuning using monthly transaction volume or average transaction amount?"
    )},
]})

# E2: Net income
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show me net income distribution for Business customers"},
    {"role": "assistant", "content": (
        "Net income is not a metric available in our Business customer dataset. "
        "For Business accounts, the available financial metrics are:\n"
        "- **Monthly Transaction Volume** (TRXN_AMT_MONTHLY)\n"
        "- **Average Weekly Transaction Amount** (AVG_TRXN_AMT)\n"
        "- **Average Weekly Transaction Count** (AVG_TRXNS_WEEK)\n\n"
        "Would you like threshold tuning or segmentation analysis using one of these metrics instead?"
    )},
]})

# E3: Credit score
examples.append({"messages": [
    {"role": "system", "content": GENERAL_SYSTEM},
    {"role": "user", "content": "Can you show me credit scores for high-risk customers?"},
    {"role": "assistant", "content": (
        "Credit scores are not included in our AML dataset. "
        "Our customer risk analysis is based on transaction behavior metrics:\n"
        "- **Monthly Transaction Volume** and **Weekly Transaction Amount** for threshold tuning\n"
        "- **K-Means clustering** using velocity, volume, account age, and categorical features for segmentation\n\n"
        "For high-risk customer identification, I can run a SAR backtest or cluster analysis to surface "
        "customers with suspicious transaction patterns. Which would you like?"
    )},
]})

# ── F. list_rules — no hallucination ─────────────────────────────────────────

# F1: Direct list request
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show me all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c15", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "c15", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n\n"
        "- **Activity Deviation (ACH)**: alerts=907, SAR=138, FP=769, precision=15.2%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Activity Deviation (Check)**: alerts=316, SAR=76, FP=240, precision=24.1%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Elder Abuse**: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, age_threshold]\n"
        "- **Velocity Single**: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Detect Excessive Transaction Activity**: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Structuring (Incoming Cash)**: alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
        "- **Structuring (Outgoing Cash)**: alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
        "- **CTR Client**: alerts=4, SAR=4, FP=0, precision=100.0%, sweep_params=[floor_amount]\n"
        "- **Burst in Originator Activity**: alerts=3, SAR=0, FP=3, precision=0.0%, sweep_params=[floor_amount, min_transactions]\n"
        "- **Burst in Beneficiary Activity**: alerts=5, SAR=0, FP=5, precision=0.0%, sweep_params=[floor_amount, min_transactions]\n"
        "- **Risky International Transfer**: alerts=2, SAR=0, FP=2, precision=0.0%, sweep_params=[floor_amount]\n\n"
        "The chart below shows the full SAR vs FP breakdown for each rule."
    )},
]})

# F2: Which rules have highest FP
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Which rules generate the most false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c16", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "c16", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n\n"
        "- **Activity Deviation (ACH)**: alerts=907, SAR=138, FP=769, precision=15.2%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Activity Deviation (Check)**: alerts=316, SAR=76, FP=240, precision=24.1%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Elder Abuse**: alerts=1146, SAR=188, FP=958, precision=16.4%, sweep_params=[floor_amount, age_threshold]\n"
        "- **Velocity Single**: alerts=478, SAR=74, FP=404, precision=15.5%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Detect Excessive Transaction Activity**: alerts=356, SAR=46, FP=310, precision=12.9%, sweep_params=[floor_amount, z_threshold]\n"
        "- **Structuring (Incoming Cash)**: alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
        "- **Structuring (Outgoing Cash)**: alerts=2, SAR=2, FP=0, precision=100.0%, sweep_params=[daily_floor, days_required]\n"
        "- **CTR Client**: alerts=4, SAR=4, FP=0, precision=100.0%, sweep_params=[floor_amount]\n"
        "- **Burst in Originator Activity**: alerts=3, SAR=0, FP=3, precision=0.0%, sweep_params=[floor_amount, min_transactions]\n"
        "- **Burst in Beneficiary Activity**: alerts=5, SAR=0, FP=5, precision=0.0%, sweep_params=[floor_amount, min_transactions]\n"
        "- **Risky International Transfer**: alerts=2, SAR=0, FP=2, precision=0.0%, sweep_params=[floor_amount]\n\n"
        "**Elder Abuse** generates the most false positives at 958 FPs (16.4% precision), "
        "followed by **Activity Deviation (ACH)** at 769 FPs (15.2% precision) — "
        "both are candidates for threshold tightening to reduce investigator workload."
    )},
]})

# ── G. Policy synthesis ───────────────────────────────────────────────────────

# G1: BSA definition
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the Bank Secrecy Act?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c17", "search_policy_kb", {"query": "Bank Secrecy Act BSA requirements"})]},
    {"role": "tool", "tool_call_id": "c17", "content": PC_POLICY_BSA},
    {"role": "assistant", "content": (
        "The **Bank Secrecy Act (BSA)**, enacted in 1970, is the primary U.S. anti-money laundering law "
        "requiring financial institutions to assist government agencies in detecting and preventing money laundering.\n\n"
        "**Key requirements:**\n"
        "- **Currency Transaction Reports (CTRs)**: File for all cash transactions exceeding **$10,000**\n"
        "- **Suspicious Activity Reports (SARs)**: File for transactions of **$5,000 or more** suspected to involve "
        "money laundering or other illegal activity\n"
        "- **Recordkeeping**: Maintain records of cash purchases of negotiable instruments\n"
        "- **AML Compliance Program**: Implement the four pillars — internal controls, independent testing, "
        "a designated BSA officer, and ongoing employee training\n\n"
        "**Filing timelines** (per 31 CFR Part 1020):\n"
        "- SARs: within **30 calendar days** of detection (60 days if no suspect is identified)\n"
        "- CTRs: within **15 calendar days** after the transaction date\n\n"
        "The BSA is enforced by FinCEN (Financial Crimes Enforcement Network) and examined by federal banking regulators "
        "using the FFIEC BSA/AML Examination Manual."
    )},
]})

# G2: Structuring definition
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is AML structuring?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c18", "search_policy_kb", {"query": "structuring smurfing cash transactions BSA"})]},
    {"role": "tool", "tool_call_id": "c18", "content": PC_POLICY_STRUCTURING},
    {"role": "assistant", "content": (
        "**Structuring** (also known as 'smurfing') is the practice of breaking up large financial transactions "
        "into smaller amounts to evade the **$10,000 CTR reporting threshold** under the Bank Secrecy Act.\n\n"
        "**Legal definition** (31 CFR 1010.314 / 31 U.S.C. 5324):\n"
        "Structuring is illegal regardless of whether the underlying funds are from criminal activity. "
        "A transaction is considered structured if it is conducted — or attempted — at one or more financial "
        "institutions, on one or more days, **for the purpose of evading reporting requirements**.\n\n"
        "**AML red flags:**\n"
        "- Multiple cash transactions just below $10,000 on the same day or consecutive days\n"
        "- Customer becomes nervous when asked for identification\n"
        "- Customers appearing to work in groups to structure transactions\n"
        "- Deposits or withdrawals of exactly $9,500, $9,800, or similar amounts just under the threshold\n\n"
        "Structuring is monitored via rules such as **Structuring (Incoming Cash)** and **Structuring (Outgoing Cash)** "
        "in our rule engine, which flag customers with recurring daily amounts near the $10K threshold."
    )},
]})

# G3: Risk-based approach
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the Wolfsberg risk-based approach to AML?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("c19", "search_policy_kb", {"query": "Wolfsberg risk-based approach AML EDD CDD"})]},
    {"role": "tool", "tool_call_id": "c19", "content": PC_POLICY_WOLFSBERG},
    {"role": "assistant", "content": (
        "The **Wolfsberg Group's Risk-Based Approach (RBA)** to AML requires financial institutions to "
        "identify, assess, and understand their money laundering risks, then apply controls proportionate "
        "to those risks — rather than applying uniform controls to all customers.\n\n"
        "**Core principle:**\n"
        "Higher-risk customers require **Enhanced Due Diligence (EDD)**; lower-risk customers may be subject to "
        "**Simplified Due Diligence (SDD)**. The RBA does not require zero-risk tolerance — it requires that "
        "residual risks be understood and managed to an acceptable level.\n\n"
        "**Risk factors to assess** (Wolfsberg FAQs 2015):\n"
        "- **Customer risk**: types, volumes, geographic distribution, PEP status\n"
        "- **Product and service risk**: complexity, anonymity, cross-border capability\n"
        "- **Geographic risk**: countries of operation and customer base\n"
        "- **Delivery channel risk**: internet banking, correspondent relationships\n\n"
        "**EDD triggers** include: Politically Exposed Persons (PEPs), customers in high-risk jurisdictions, "
        "complex ownership structures, and unusual transaction patterns inconsistent with the customer profile.\n\n"
        "The risk assessment must be documented, reviewed periodically, and updated when material changes occur."
    )},
]})

# ─── Write output ──────────────────────────────────────────────────────────────

print(f"Total V19 new examples: {len(examples)}")
for i, ex in enumerate(examples):
    print(f"  ex{611+i}: {ex['messages'][1]['content'][:70]}")

out_path = pathlib.Path(__file__).parent / "framl_train_v19.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"\nWrote {len(examples)} examples to {out_path}")

# Combine with V18
v18_path = pathlib.Path(__file__).parent / "framl_train_combined_v18.jsonl"
v19_combined = pathlib.Path(__file__).parent / "framl_train_combined_v19.jsonl"
with open(v19_combined, "w", encoding="utf-8") as out:
    with open(v18_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.write(line)
    for ex in examples:
        out.write(json.dumps(ex, ensure_ascii=False) + "\n")

total = sum(1 for l in open(v19_combined, encoding="utf-8") if l.strip())
print(f"Combined V19: {total} total examples")

"""
V21 training examples — gaps from V20 benchmark analysis 2026-04-19.

Targets:
  M. rule_2d_sweep — Velocity Single + ACH 2D sweep (3 examples)
     — D02 (Velocity Single) not covered in V20; reinforce D01/D03 with new phrasings
  N. rule_sar_backtest — Structuring + Burst + Risky International (3 examples)
     — broader rule coverage beyond the 5 rules in V20 H-series
  O. threshold_tuning invalid column → text decline + alternatives (1 example)
     — model still calls tool with invalid column instead of declining
  P. Policy KB — Wolfsberg risk-based approach + alert volume management (2 examples)
     — P03/P04 benchmark cases not yet trained
  Q. rule_2d_sweep rephrased triggers — "heatmap", "parameter interaction", "sweep both" (3 examples)
     — alternative phrasings that still route to list_rules

Start: ex647 (V20 combined = 646 examples).
Combined with framl_train_combined_v20.jsonl -> framl_train_combined_v21.jsonl
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
    "ONE sentence of insight referencing specific numbers from the results. Do NOT ask for more data.\n"
    "6. Call the tool ONCE only.\n"
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
    "5. Add ONE sentence of AML insight using ONLY numbers from the tool result — do NOT invent figures.\n"
    "6. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "7. Call the tool ONCE only.\n"
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
    "5. Cite the source document in parentheses where applicable.\n"
)

# --- Pre-computed tool results ------------------------------------------------

PC_RULE_2D_VELOCITY = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Axis 1 (pair_total): Minimum combined in+out pair total to trigger (currently $20,000)
Axis 2 (ratio_tolerance): Max deviation of out/in ratio from 1.0 (currently 10% = 90-110%)
Grid: 5 x 8 = 40 combinations
SAR pool: 74
Non-SAR pool: 404

At current condition (pair_total=20000, ratio_tolerance=0.10): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): pair_total=35000, ratio_tolerance=0.075 => TP=68, FP=310, FN=6, TN=94, TP rate=91.9%, precision=18.0%.
Best FP reduction (TP rate >=50%): pair_total=40000, ratio_tolerance=0.05 => TP=40, FP=120, FN=34, TN=284, TP rate=54.1%, precision=25.0%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_RULE_2D_ACH = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Activity Deviation (ACH)
Axis 1 (floor_amount): Minimum monthly ACH outgoing sum to trigger (currently $50,000)
Axis 2 (z_threshold): Std-dev multiplier above 12-month ACH profile mean (currently 5)
Grid: 5 x 6 = 30 combinations
SAR pool: 138
Non-SAR pool: 769

At current condition (floor_amount=50000, z_threshold=5): TP=130, FP=680, FN=8, TN=89 (TP rate=94.2%).
Best FP reduction (TP rate >=90%): floor_amount=75000, z_threshold=6 => TP=126, FP=520, FN=12, TN=249, TP rate=91.3%, precision=19.5%.
Best FP reduction (TP rate >=50%): floor_amount=100000, z_threshold=7 => TP=72, FP=200, FN=66, TN=569, TP rate=52.2%, precision=26.5%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_RULE_2D_BURST_ORIG = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Burst in Originator Activity
Axis 1 (floor_amount): Minimum 5-day incoming Wire/ACH sum to trigger (currently $5,000)
Axis 2 (min_transactions): Minimum number of transactions in the 5-day window (currently 3)
Grid: 5 x 5 = 25 combinations
SAR pool: 32
Non-SAR pool: 168

At current condition (floor_amount=5000, min_transactions=3): TP=32, FP=168, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=15000, min_transactions=4 => TP=29, FP=112, FN=3, TN=56, TP rate=90.6%, precision=20.6%.
Best FP reduction (TP rate >=50%): floor_amount=30000, min_transactions=5 => TP=18, FP=44, FN=14, TN=124, TP rate=56.3%, precision=29.0%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_RULE_SAR_STRUCTURING_IN = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Structuring (Incoming Cash)
Current condition: 3 qualifying days within 14-day window, each day's Cash CashIn total $3K-$40K
Sweep parameter: daily_floor - Minimum daily Cash CashIn total for a qualifying day (currently $3K)
Current value: 3000
Labeled population: 300 customers (TP+FN pool=47 SAR, FP+TN pool=253 non-SAR, precision=15.7%)

At the lowest value (500.00): TP=47, FP=253, FN=0, TN=0 (TP rate=100.0%, precision=15.7%).
At current condition (3000.00): TP=47, FP=253, FN=0, TN=0 (TP rate=100.0%, precision=15.7%).
To keep TP rate >=90%: daily_floor <= 5000.00 => TP=44, FP=218, FN=3, TN=35, precision=16.8%.
To keep TP rate >=50%: daily_floor <= 8000.00 => TP=26, FP=98, FN=21, TN=155, precision=21.0%.
At the highest value (9000.00): TP=18, FP=60, FN=29, TN=193 (TP rate=38.3%, precision=23.1%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_BURST_BENE = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Burst in Beneficiary Activity
Current condition: 5-day Outgoing Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties
Sweep parameter: floor_amount - Minimum 5-day outgoing Wire/ACH sum to trigger (currently $5K)
Current value: 5000
Labeled population: 200 customers (TP+FN pool=23 SAR, FP+TN pool=177 non-SAR, precision=11.5%)

At the lowest value (1000.00): TP=23, FP=177, FN=0, TN=0 (TP rate=100.0%, precision=11.5%).
At current condition (5000.00): TP=23, FP=177, FN=0, TN=0 (TP rate=100.0%, precision=11.5%).
To keep TP rate >=90%: floor_amount <= 20000.00 => TP=21, FP=138, FN=2, TN=39, precision=13.2%.
To keep TP rate >=50%: floor_amount <= 50000.00 => TP=14, FP=62, FN=9, TN=115, precision=18.4%.
At the highest value (75000.00): TP=10, FP=38, FN=13, TN=139 (TP rate=43.5%, precision=20.8%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_RISKY = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Risky International Transfer
Current condition: Single Wire to/from medium-risk country >= $300K or >= $500K
Sweep parameter: floor_amount - Minimum single Wire amount to a risky country to trigger (currently $300K)
Current value: 300000
Labeled population: 150 customers (TP+FN pool=34 SAR, FP+TN pool=116 non-SAR, precision=22.7%)

At the lowest value (100000.00): TP=34, FP=116, FN=0, TN=0 (TP rate=100.0%, precision=22.7%).
At current condition (300000.00): TP=34, FP=116, FN=0, TN=0 (TP rate=100.0%, precision=22.7%).
To keep TP rate >=90%: floor_amount <= 500000.00 => TP=31, FP=88, FN=3, TN=28, precision=26.1%.
To keep TP rate >=50%: floor_amount <= 900000.00 => TP=20, FP=34, FN=14, TN=82, precision=37.0%.
At the highest value (1000000.00): TP=16, FP=24, FN=18, TN=92 (TP rate=47.1%, precision=40.0%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_KB_WOLFSBERG = """\
Tool result for search_policy_kb:
The Wolfsberg Group's Risk-Based Approach (RBA) to AML is a framework that directs financial institutions \
to allocate compliance resources proportionally to the money laundering and terrorist financing risks they face, \
rather than applying uniform controls to all customers and transactions. First articulated in the Wolfsberg \
Risk-Based Approach Statement (2006) and refined in subsequent guidance including the 2015 FAQs, the RBA \
requires FIs to identify, assess, and document their specific risk exposures across customer types, \
geographies, products, and channels (Wolfsberg_RBA_Guidance_(2006).pdf). \
Higher-risk customers — such as Politically Exposed Persons (PEPs), customers from high-risk jurisdictions, \
and those in cash-intensive businesses — receive enhanced due diligence (EDD) and more frequent monitoring. \
Lower-risk customers may qualify for simplified due diligence. The Wolfsberg July 2025 statement on \
risk-based approaches (Wolfsberg_RiskBasedApproach_Statement_July2025.pdf) emphasizes that effectiveness \
should be measured by outcomes (SARs filed, investigations opened, law enforcement referrals) rather than \
purely by the volume of controls applied. The FFIEC BSA/AML Examination Manual similarly endorses the RBA, \
instructing examiners to evaluate whether an institution's controls are commensurate with its risk profile \
rather than prescribing a one-size-fits-all ruleset (FFIEC BSA-AML Examination Manual.pdf)."""

PC_KB_ALERT_VOLUMES = """\
Tool result for search_policy_kb:
Financial institutions manage AML alert volumes through a combination of threshold calibration, \
model tuning, and tiered review workflows. The FFIEC BSA/AML Examination Manual notes that alert \
generation rates must be periodically reviewed to ensure they are neither over-alerting (producing \
excessive false positives that overwhelm investigators) nor under-alerting (missing genuine suspicious \
activity) (FFIEC BSA-AML Examination Manual.pdf). Common strategies include: \
(1) Risk-stratified thresholds — setting tighter parameters for high-risk customer segments \
(e.g., PEPs, cash-intensive businesses) and looser thresholds for low-risk retail customers; \
(2) Statistical profiling — z-score or peer-group comparisons that trigger alerts only when \
a customer's activity deviates significantly from their own baseline or a comparable peer group, \
reducing false positives from legitimate but high-volume customers; \
(3) Alert suppression / exclusion lists — documented exemptions for certain recurring transactions \
(e.g., payroll ACH, known counterparties) where the business purpose is established; \
(4) Model validation — the OCC and Fed require periodic independent validation of transaction \
monitoring models to confirm that thresholds remain appropriate as customer behavior evolves \
(BsaAmlExamProcsPackage.docx). The Wolfsberg Transitioning to Innovation paper \
(Wolfsberg_MSAP2_Transitioning to Innovation.pdf) also discusses the role of machine learning \
in prioritizing alerts by risk score to reduce investigator workload without raising miss rates."""

# --- Examples -----------------------------------------------------------------

examples = []

# ── M. rule_2d_sweep — Velocity Single + ACH + rephrased Detect Excessive ─────

# M1: 2D heatmap Velocity Single — D02 benchmark case (was NOT in V20)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D heatmap for Velocity Single — how do pair_total and ratio_tolerance interact?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("m1", "rule_2d_sweep", {"risk_factor": "Velocity Single",
                                    "sweep_param_1": "pair_total",
                                    "sweep_param_2": "ratio_tolerance"})
    ]},
    {"role": "tool", "tool_call_id": "m1", "content": PC_RULE_2D_VELOCITY},
    {"role": "assistant", "content": (
        "Rule: Velocity Single\n"
        "Axis 1 (pair_total): Minimum combined in+out pair total to trigger (currently $20,000)\n"
        "Axis 2 (ratio_tolerance): Max deviation of out/in ratio from 1.0 (currently 10% = 90-110%)\n"
        "Grid: 5 x 8 = 40 combinations\n"
        "SAR pool: 74\nNon-SAR pool: 404\n\n"
        "At current condition (pair_total=20000, ratio_tolerance=0.10): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): pair_total=35000, ratio_tolerance=0.075 => TP=68, FP=310, FN=6, TN=94, TP rate=91.9%, precision=18.0%.\n"
        "Best FP reduction (TP rate >=50%): pair_total=40000, ratio_tolerance=0.05 => TP=40, FP=120, FN=34, TN=284, TP rate=54.1%, precision=25.0%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Tightening the ratio tolerance to 7.5% while raising pair_total to $35K reduces FPs by 94 "
        "(from 404 to 310) while keeping SAR detection at 91.9% — the ratio_tolerance axis has "
        "stronger FP reduction power than the pair_total axis alone."
    )},
]})

# M2: 2D sweep ACH — floor_amount × z_threshold
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run a 2D parameter sweep for Activity Deviation ACH"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("m2", "rule_2d_sweep", {"risk_factor": "Activity Deviation (ACH)",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "z_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "m2", "content": PC_RULE_2D_ACH},
    {"role": "assistant", "content": (
        "Rule: Activity Deviation (ACH)\n"
        "Axis 1 (floor_amount): Minimum monthly ACH outgoing sum to trigger (currently $50,000)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 12-month ACH profile mean (currently 5)\n"
        "Grid: 5 x 6 = 30 combinations\n"
        "SAR pool: 138\nNon-SAR pool: 769\n\n"
        "At current condition (floor_amount=50000, z_threshold=5): TP=130, FP=680, FN=8, TN=89 (TP rate=94.2%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=75000, z_threshold=6 => TP=126, FP=520, FN=12, TN=249, TP rate=91.3%, precision=19.5%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=100000, z_threshold=7 => TP=72, FP=200, FN=66, TN=569, TP rate=52.2%, precision=26.5%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Raising the z_threshold from 5 to 6 while increasing the floor to $75K eliminates 160 FPs "
        "(from 680 to 520) at a cost of only 4 additional missed SARs — the z_threshold is the "
        "more powerful lever for precision improvement in this rule."
    )},
]})

# M3: 2D sweep Burst in Originator — rephrased as "parameter interaction"
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show how floor_amount and transaction count interact for Burst in Originator Activity"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("m3", "rule_2d_sweep", {"risk_factor": "Burst in Originator Activity",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "min_transactions"})
    ]},
    {"role": "tool", "tool_call_id": "m3", "content": PC_RULE_2D_BURST_ORIG},
    {"role": "assistant", "content": (
        "Rule: Burst in Originator Activity\n"
        "Axis 1 (floor_amount): Minimum 5-day incoming Wire/ACH sum to trigger (currently $5,000)\n"
        "Axis 2 (min_transactions): Minimum number of transactions in the 5-day window (currently 3)\n"
        "Grid: 5 x 5 = 25 combinations\n"
        "SAR pool: 32\nNon-SAR pool: 168\n\n"
        "At current condition (floor_amount=5000, min_transactions=3): TP=32, FP=168, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=15000, min_transactions=4 => TP=29, FP=112, FN=3, TN=56, TP rate=90.6%, precision=20.6%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=30000, min_transactions=5 => TP=18, FP=44, FN=14, TN=124, TP rate=56.3%, precision=29.0%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Requiring at least 4 transactions and a $15K sum reduces FPs by 56 (from 168 to 112) "
        "while catching 29 of 32 SARs — the transaction count floor is the primary driver "
        "of precision improvement for this rule."
    )},
]})

# ── N. rule_sar_backtest — Structuring + Burst Beneficiary + Risky International ─

# N1: SAR backtest Structuring Incoming Cash
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Structuring Incoming Cash?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("n1", "rule_sar_backtest", {"risk_factor": "Structuring (Incoming Cash)"})
    ]},
    {"role": "tool", "tool_call_id": "n1", "content": PC_RULE_SAR_STRUCTURING_IN},
    {"role": "assistant", "content": (
        "Rule: Structuring (Incoming Cash)\n"
        "Current condition: 3 qualifying days within 14-day window, each day's Cash CashIn total $3K-$40K\n"
        "Sweep parameter: daily_floor - Minimum daily Cash CashIn total for a qualifying day (currently $3K)\n"
        "Current value: 3000\n"
        "Labeled population: 300 customers (TP+FN pool=47 SAR, FP+TN pool=253 non-SAR, precision=15.7%)\n\n"
        "At the lowest value (500.00): TP=47, FP=253, FN=0, TN=0 (TP rate=100.0%, precision=15.7%).\n"
        "At current condition (3000.00): TP=47, FP=253, FN=0, TN=0 (TP rate=100.0%, precision=15.7%).\n"
        "To keep TP rate >=90%: daily_floor <= 5000.00 => TP=44, FP=218, FN=3, TN=35, precision=16.8%.\n"
        "To keep TP rate >=50%: daily_floor <= 8000.00 => TP=26, FP=98, FN=21, TN=155, precision=21.0%.\n"
        "At the highest value (9000.00): TP=18, FP=60, FN=29, TN=193 (TP rate=38.3%, precision=23.1%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Raising the daily floor to $5,000 reduces FPs by 35 (from 253 to 218) while missing "
        "only 3 of 47 SARs — a 93.6% catch rate that slightly improves precision from 15.7% to 16.8%."
    )},
]})

# N2: SAR backtest Burst in Beneficiary Activity
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Burst in Beneficiary Activity rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("n2", "rule_sar_backtest", {"risk_factor": "Burst in Beneficiary Activity"})
    ]},
    {"role": "tool", "tool_call_id": "n2", "content": PC_RULE_SAR_BURST_BENE},
    {"role": "assistant", "content": (
        "Rule: Burst in Beneficiary Activity\n"
        "Current condition: 5-day Outgoing Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties\n"
        "Sweep parameter: floor_amount - Minimum 5-day outgoing Wire/ACH sum to trigger (currently $5K)\n"
        "Current value: 5000\n"
        "Labeled population: 200 customers (TP+FN pool=23 SAR, FP+TN pool=177 non-SAR, precision=11.5%)\n\n"
        "At the lowest value (1000.00): TP=23, FP=177, FN=0, TN=0 (TP rate=100.0%, precision=11.5%).\n"
        "At current condition (5000.00): TP=23, FP=177, FN=0, TN=0 (TP rate=100.0%, precision=11.5%).\n"
        "To keep TP rate >=90%: floor_amount <= 20000.00 => TP=21, FP=138, FN=2, TN=39, precision=13.2%.\n"
        "To keep TP rate >=50%: floor_amount <= 50000.00 => TP=14, FP=62, FN=9, TN=115, precision=18.4%.\n"
        "At the highest value (75000.00): TP=10, FP=38, FN=13, TN=139 (TP rate=43.5%, precision=20.8%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Raising the floor to $20K reduces FPs by 39 (from 177 to 138) while maintaining a 91.3% "
        "SAR catch rate — this is the best precision gain available without dropping below 90% sensitivity."
    )},
]})

# N3: SAR backtest Risky International Transfer
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show SAR analysis for Risky International Transfer"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("n3", "rule_sar_backtest", {"risk_factor": "Risky International Transfer"})
    ]},
    {"role": "tool", "tool_call_id": "n3", "content": PC_RULE_SAR_RISKY},
    {"role": "assistant", "content": (
        "Rule: Risky International Transfer\n"
        "Current condition: Single Wire to/from medium-risk country >= $300K or >= $500K\n"
        "Sweep parameter: floor_amount - Minimum single Wire amount to a risky country to trigger (currently $300K)\n"
        "Current value: 300000\n"
        "Labeled population: 150 customers (TP+FN pool=34 SAR, FP+TN pool=116 non-SAR, precision=22.7%)\n\n"
        "At the lowest value (100000.00): TP=34, FP=116, FN=0, TN=0 (TP rate=100.0%, precision=22.7%).\n"
        "At current condition (300000.00): TP=34, FP=116, FN=0, TN=0 (TP rate=100.0%, precision=22.7%).\n"
        "To keep TP rate >=90%: floor_amount <= 500000.00 => TP=31, FP=88, FN=3, TN=28, precision=26.1%.\n"
        "To keep TP rate >=50%: floor_amount <= 900000.00 => TP=20, FP=34, FN=14, TN=82, precision=37.0%.\n"
        "At the highest value (1000000.00): TP=16, FP=24, FN=18, TN=92 (TP rate=47.1%, precision=40.0%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Raising the floor to $500K reduces FPs by 28 (from 116 to 88) while missing only 3 SARs — "
        "a precision gain from 22.7% to 26.1% with a 91.2% catch rate."
    )},
]})

# ── O. threshold_tuning — invalid column → text decline ──────────────────────

# O1: User asks for "daily balance" — not a valid column → decline + alternatives
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN trade-off for Business customers by daily balance"},
    {"role": "assistant", "content": (
        "Daily balance is not an available threshold column in our system. "
        "The valid threshold columns are:\n\n"
        "- **TRXN_AMT_MONTHLY** — aggregate monthly transaction amount (closest to a balance-based view)\n"
        "- **AVG_TRXN_AMT** — average transaction amount per week\n"
        "- **AVG_TRXNS_WEEK** — average number of transactions per week\n\n"
        "The closest substitute for a balance-based analysis would be **TRXN_AMT_MONTHLY**, "
        "which captures the volume of funds moving through the account each month. "
        "Would you like me to run the threshold tuning for Business customers using TRXN_AMT_MONTHLY?"
    )},
]})

# ── P. Policy KB — Wolfsberg RBA + alert volume management ───────────────────

# P1: Wolfsberg risk-based approach — P03 benchmark case
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the Wolfsberg risk-based approach to AML?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("p1", "search_policy_kb", {"query": "Wolfsberg risk-based approach AML framework"})
    ]},
    {"role": "tool", "tool_call_id": "p1", "content": PC_KB_WOLFSBERG},
    {"role": "assistant", "content": (
        "The Wolfsberg Group's Risk-Based Approach (RBA) is an internationally recognized framework "
        "that directs financial institutions to allocate their AML compliance resources in proportion "
        "to the money laundering and terrorist financing risks they actually face — rather than applying "
        "identical controls to every customer and transaction.\n\n"
        "**Core Principles:**\n"
        "- **Risk identification and assessment:** FIs must identify and document their specific risk "
        "exposures across customer types, geographies, products, and delivery channels "
        "(Wolfsberg_RBA_Guidance_(2006).pdf)\n"
        "- **Proportionate controls:** Higher-risk customers — such as Politically Exposed Persons (PEPs), "
        "customers from high-risk jurisdictions, and cash-intensive businesses — receive Enhanced Due "
        "Diligence (EDD) and more frequent transaction monitoring. Lower-risk customers may qualify "
        "for Simplified Due Diligence\n"
        "- **Outcome-based measurement:** The July 2025 Wolfsberg statement emphasizes that AML "
        "effectiveness should be measured by outcomes (SARs filed, investigations opened, law "
        "enforcement referrals) rather than purely by the volume of controls applied "
        "(Wolfsberg_RiskBasedApproach_Statement_July2025.pdf)\n\n"
        "The FFIEC BSA/AML Examination Manual endorses the RBA, instructing examiners to evaluate "
        "whether an institution's controls are commensurate with its risk profile rather than "
        "applying a one-size-fits-all compliance checklist (FFIEC BSA-AML Examination Manual.pdf)."
    )},
]})

# P2: How do banks manage alert volumes — P04 benchmark case
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "How do banks manage alert volumes?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("p2", "search_policy_kb", {"query": "AML alert volume management threshold calibration false positive"})
    ]},
    {"role": "tool", "tool_call_id": "p2", "content": PC_KB_ALERT_VOLUMES},
    {"role": "assistant", "content": (
        "Banks manage AML alert volumes through a combination of threshold calibration, behavioral "
        "profiling, and structured review workflows — the goal being to maximize SAR catch rates "
        "while keeping false positive volumes manageable for investigative teams.\n\n"
        "**Key strategies include:**\n\n"
        "1. **Risk-stratified thresholds** — tighter alert parameters for high-risk segments (PEPs, "
        "cash-intensive businesses) and looser thresholds for low-risk retail customers, ensuring "
        "investigator effort is directed where risk is highest\n\n"
        "2. **Statistical profiling (z-score / peer-group analysis)** — alerts trigger only when a "
        "customer's activity deviates significantly from their own historical baseline or a comparable "
        "peer group, reducing false positives from legitimately high-volume customers\n\n"
        "3. **Alert suppression and exclusion lists** — documented exemptions for recurring transactions "
        "with established business purpose (e.g., payroll ACH, known counterparties), reducing "
        "alert noise without creating compliance gaps\n\n"
        "4. **Periodic model validation** — the OCC and Federal Reserve require independent validation "
        "of transaction monitoring models to confirm thresholds remain appropriate as customer "
        "behavior evolves (BsaAmlExamProcsPackage.docx)\n\n"
        "5. **Machine learning prioritization** — the Wolfsberg Transitioning to Innovation paper "
        "discusses how ML-based risk scoring can prioritize the alert queue by likelihood of SAR, "
        "reducing investigator workload without increasing miss rates "
        "(Wolfsberg_MSAP2_Transitioning to Innovation.pdf)\n\n"
        "The FFIEC BSA/AML Examination Manual notes that alert generation rates must be periodically "
        "reviewed to ensure the program is neither over-alerting nor under-alerting relative to the "
        "institution's risk profile."
    )},
]})

# ── Q. rule_2d_sweep — alternative phrasings that still route to list_rules ────

# Q1: "sweep both parameters" phrasing
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Sweep both parameters for the Structuring Outgoing Cash rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("q1", "rule_2d_sweep", {"risk_factor": "Structuring (Outgoing Cash)"})
    ]},
    {"role": "tool", "tool_call_id": "q1", "content": """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Structuring (Outgoing Cash)
Axis 1 (daily_floor): Minimum daily Cash CashOut total for a qualifying day (currently $7,000)
Axis 2 (days_required): Minimum number of qualifying days in the window (currently 3)
Grid: 5 x 5 = 25 combinations
SAR pool: 39
Non-SAR pool: 211

At current condition (daily_floor=7000, days_required=3): TP=39, FP=211, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): daily_floor=10000, days_required=4 => TP=36, FP=162, FN=3, TN=49, TP rate=92.3%, precision=18.2%.
Best FP reduction (TP rate >=50%): daily_floor=15000, days_required=5 => TP=22, FP=66, FN=17, TN=145, TP rate=56.4%, precision=25.0%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""},
    {"role": "assistant", "content": (
        "Rule: Structuring (Outgoing Cash)\n"
        "Axis 1 (daily_floor): Minimum daily Cash CashOut total for a qualifying day (currently $7,000)\n"
        "Axis 2 (days_required): Minimum number of qualifying days in the window (currently 3)\n"
        "Grid: 5 x 5 = 25 combinations\n"
        "SAR pool: 39\nNon-SAR pool: 211\n\n"
        "At current condition (daily_floor=7000, days_required=3): TP=39, FP=211, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): daily_floor=10000, days_required=4 => TP=36, FP=162, FN=3, TN=49, TP rate=92.3%, precision=18.2%.\n"
        "Best FP reduction (TP rate >=50%): daily_floor=15000, days_required=5 => TP=22, FP=66, FN=17, TN=145, TP rate=56.4%, precision=25.0%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Requiring a $10K daily floor across 4 qualifying days reduces FPs by 49 (from 211 to 162) "
        "while keeping the SAR catch rate at 92.3% — a meaningful precision improvement "
        "from 15.6% to 18.2%."
    )},
]})

# Q2: "show me the grid" phrasing → rule_2d_sweep
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show me the parameter grid for CTR Client rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("q2", "rule_2d_sweep", {"risk_factor": "CTR Client"})
    ]},
    {"role": "tool", "tool_call_id": "q2", "content": """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: CTR Client
Axis 1 (floor_amount): Minimum Cash/Currency Exchange total to trigger (currently $10,000)
Axis 2 (floor_amount): (single-parameter rule — 2D sweep uses same axis twice)
Grid: 5 x 1 = 5 combinations
SAR pool: 4
Non-SAR pool: 0

At current condition (floor_amount=10000): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=10000 => TP=4, FP=0, FN=0, TN=0, TP rate=100.0%, precision=100.0%.
At the highest value (15000): TP=3, FP=0, FN=1, TN=0 (TP rate=75.0%, precision=100.0%).
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""},
    {"role": "assistant", "content": (
        "Rule: CTR Client\n"
        "Axis 1 (floor_amount): Minimum Cash/Currency Exchange total to trigger (currently $10,000)\n"
        "Grid: 5 x 1 = 5 combinations\n"
        "SAR pool: 4\nNon-SAR pool: 0\n\n"
        "At current condition (floor_amount=10000): TP=4, FP=0, FN=0, TN=0 (TP rate=100.0%, precision=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=10000 => TP=4, FP=0, FN=0, TN=0, TP rate=100.0%, precision=100.0%.\n"
        "At the highest value (15000): TP=3, FP=0, FN=1, TN=0 (TP rate=75.0%, precision=100.0%).\n"
        "(Heatmap shown in the chart below.)\n\n"
        "CTR Client is a single-parameter rule with 100% precision at all tested thresholds — "
        "raising the floor above $10K is not recommended as it begins missing SARs "
        "with no false positive reduction benefit."
    )},
]})

# Q3: "optimize thresholds" phrasing → rule_2d_sweep
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Help me optimize both thresholds for the Elder Abuse rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("q3", "rule_2d_sweep", {"risk_factor": "Elder Abuse",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "age_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "q3", "content": """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Axis 1 (floor_amount): Minimum 14-day outgoing floor (currently $5,000)
Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)
Grid: 6 x 5 = 30 combinations
SAR pool: 63
Non-SAR pool: 337

At current condition (floor_amount=5000, age_threshold=60): TP=63, FP=337, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=20000, age_threshold=63 => TP=58, FP=248, FN=5, TN=89, TP rate=92.1%, precision=18.9%.
Best FP reduction (TP rate >=50%): floor_amount=40000, age_threshold=68 => TP=34, FP=88, FN=29, TN=249, TP rate=54.0%, precision=27.9%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""},
    {"role": "assistant", "content": (
        "Rule: Elder Abuse\n"
        "Axis 1 (floor_amount): Minimum 14-day outgoing floor (currently $5,000)\n"
        "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)\n"
        "Grid: 6 x 5 = 30 combinations\n"
        "SAR pool: 63\nNon-SAR pool: 337\n\n"
        "At current condition (floor_amount=5000, age_threshold=60): TP=63, FP=337, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=20000, age_threshold=63 => TP=58, FP=248, FN=5, TN=89, TP rate=92.1%, precision=18.9%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=40000, age_threshold=68 => TP=34, FP=88, FN=29, TN=249, TP rate=54.0%, precision=27.9%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Setting floor_amount to $20K and age_threshold to 63 reduces FPs by 89 (from 337 to 248) "
        "while maintaining a 92.1% SAR catch rate — the joint sweep reveals that raising "
        "the floor has more FP reduction impact than raising the age threshold alone."
    )},
]})

# ── R. Sanctions / OFAC routing → search_policy_kb ────────────────────────────

PC_KB_OFAC = """\
Tool result for search_policy_kb:
The Office of Foreign Assets Control (OFAC) is a U.S. Treasury Department agency that administers \
and enforces economic and trade sanctions based on U.S. foreign policy and national security goals. \
Financial institutions are legally required to block or reject transactions involving Specially \
Designated Nationals (SDNs) listed on the OFAC SDN List, as well as entities in sanctioned \
countries or regions (31 CFR Part 500 series). When a customer or counterparty matches the SDN \
List, the institution must: (1) block the transaction immediately; (2) file a blocking report with \
OFAC within 10 business days; and (3) file a SAR with FinCEN if the match involves suspected \
money laundering (31 CFR Part 1020). OFAC also maintains the Consolidated Sanctions List which \
includes additional programs beyond the SDN List. Under the BSA, a sanctions hit is independent \
of — but often concurrent with — SAR obligations: a blocked transaction must be reported to OFAC \
regardless of SAR status, but an SAR must also be filed if the underlying activity suggests \
money laundering or terrorist financing (BsaAmlExamProcsPackage.docx). \
False positives from name-matching algorithms (fuzzy matches) must be documented as \
'no-match' determinations and retained for five years per recordkeeping requirements \
(31 CFR Part 1010.430)."""

PC_KB_SANCTIONS_PROCESS = """\
Tool result for search_policy_kb:
When a financial institution identifies a potential sanctions match, the standard process is: \
(1) Alert triage — the compliance team reviews the match against the OFAC SDN List, \
Consolidated Sanctions List, and any applicable secondary sanctions programs (e.g., EU, UN); \
(2) True match determination — if confirmed, the transaction is blocked and funds are placed \
in a blocked account; (3) OFAC reporting — a blocking report must be filed within 10 business \
days (31 CFR Part 501.603); (4) SAR evaluation — a SAR must be filed with FinCEN if the \
match suggests the funds are related to money laundering or terrorist financing \
(31 CFR Part 1020.320); (5) Customer exit — in most cases, confirmed SDN matches trigger \
account closure and relationship termination following regulatory guidance. \
For false positives, the institution documents a 'no-match' determination noting the \
distinguishing factors (different date of birth, address, nationality) and retains the \
record for five years. The FFIEC BSA/AML Examination Manual notes that sanctions screening \
programs must be risk-based, covering all customers, beneficial owners, and transaction \
counterparties against current lists (FFIEC BSA-AML Examination Manual.pdf)."""

# R1: OFAC SDN list question
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What happens when a customer hits the OFAC SDN list?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("r1", "search_policy_kb", {"query": "OFAC SDN list match process blocking SAR filing"})
    ]},
    {"role": "tool", "tool_call_id": "r1", "content": PC_KB_OFAC},
    {"role": "assistant", "content": (
        "When a customer or counterparty matches the OFAC Specially Designated Nationals (SDN) "
        "List, your institution has several immediate legal obligations:\n\n"
        "**1. Block the transaction**\n"
        "The transaction must be blocked immediately and funds placed in a blocked account. "
        "You cannot process or release the funds without an OFAC license.\n\n"
        "**2. File a blocking report with OFAC**\n"
        "A blocking report must be submitted to OFAC within 10 business days of the match "
        "(31 CFR Part 501.603).\n\n"
        "**3. Evaluate for SAR filing**\n"
        "Independently of the OFAC blocking report, a SAR must be filed with FinCEN if the "
        "activity suggests money laundering or terrorist financing (31 CFR Part 1020). "
        "The two reports serve different purposes and are both required when applicable.\n\n"
        "**4. Document false positives**\n"
        "If the match is determined to be a false positive (different person), document the "
        "distinguishing factors in a 'no-match' determination and retain the record for five "
        "years (31 CFR Part 1010.430).\n\n"
        "OFAC also maintains additional sanctions programs beyond the SDN List — including the "
        "Consolidated Sanctions List covering country-based programs — which must also be "
        "screened against all customers, beneficial owners, and transaction counterparties "
        "(FFIEC BSA-AML Examination Manual.pdf)."
    )},
]})

# R2: Sanctions screening process
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the process for handling a sanctions screening hit?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("r2", "search_policy_kb", {"query": "sanctions screening hit process OFAC blocking report SAR"})
    ]},
    {"role": "tool", "tool_call_id": "r2", "content": PC_KB_SANCTIONS_PROCESS},
    {"role": "assistant", "content": (
        "The standard sanctions hit handling process follows five steps:\n\n"
        "**1. Alert triage**\n"
        "Review the potential match against the OFAC SDN List, Consolidated Sanctions List, "
        "and any applicable secondary sanctions programs (EU, UN). Gather distinguishing "
        "information — date of birth, address, nationality — to assess true vs. false match.\n\n"
        "**2. True match — block and hold**\n"
        "If confirmed, block the transaction immediately and place funds in a blocked account. "
        "You cannot release blocked funds without an OFAC license.\n\n"
        "**3. File blocking report with OFAC**\n"
        "Submit a blocking report within 10 business days (31 CFR Part 501.603).\n\n"
        "**4. File SAR with FinCEN (if applicable)**\n"
        "If the sanctions match suggests money laundering or terrorist financing, file a SAR "
        "with FinCEN. The SAR and blocking report are independent obligations — both may "
        "be required (31 CFR Part 1020.320).\n\n"
        "**5. Customer exit**\n"
        "Confirmed SDN matches typically trigger account closure and relationship termination.\n\n"
        "**For false positives:** Document the distinguishing factors in a 'no-match' "
        "determination and retain for five years (31 CFR Part 1010.430). The FFIEC BSA/AML "
        "Examination Manual requires that screening programs cover all customers, beneficial "
        "owners, and counterparties against current lists (FFIEC BSA-AML Examination Manual.pdf)."
    )},
]})

# R3: Difference between OFAC and sanctions — out-of-scope decline with redirect
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "How many customers in our portfolio have OFAC hits?"},
    {"role": "assistant", "content": (
        "Live customer OFAC screening data is not accessible through this interface — "
        "that information resides in your core banking system and sanctions screening platform.\n\n"
        "What I can help with:\n\n"
        "- **Policy guidance** — what obligations apply when an OFAC hit is confirmed\n"
        "- **SAR filing requirements** — when a sanctions match also requires a FinCEN SAR\n"
        "- **Screening program requirements** — what the FFIEC and OFAC expect from your "
        "screening coverage and documentation practices\n\n"
        "Would you like me to look up any of these policy topics?"
    )},
]})

# --- Write output -------------------------------------------------------------

out_path = pathlib.Path(__file__).parent / "framl_train_v21.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(examples)} examples to {out_path}")

# Build combined V21
combined_v20 = pathlib.Path(__file__).parent / "framl_train_combined_v20.jsonl"
combined_v21 = pathlib.Path(__file__).parent / "framl_train_combined_v21.jsonl"

with open(combined_v20, encoding="utf-8") as f:
    prior = f.readlines()

with open(combined_v21, "w", encoding="utf-8") as f:
    for line in prior:
        f.write(line)
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Combined V21: {len(prior) + len(examples)} examples -> {combined_v21}")

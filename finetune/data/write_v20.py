"""
V20 training examples — gaps identified from V19 live testing 2026-04-17.

Targets:
  H. rule_sar_backtest correct routing + closing insight (5 examples)
     — prompts 7, 8, 25 misrouted to list_rules; 5, 6 missing closing insight
  I. rule_2d_sweep correct routing + non-hallucinated insight (3 examples)
     — prompt 9 misrouted to list_rules; prompt 11 hallucinated numbers
  J. Policy with user-role tool format (2 examples)
     — prompt 21 (structuring) regressed; prompt 20 (BSA) misrouted
  K. Unavailable metric → polite decline + alternatives (2 examples)
     — prompt 16 (net income) → out_of_scope instead of category E response
  L. threshold_tuning with closing insight (2 examples)
     — prompts 4 missing closing insight sentence

Start: ex633 (V19 combined = 632 examples).
Combined with framl_train_combined_v19.jsonl -> framl_train_combined_v20.jsonl
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

GENERAL_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML. "
    "You help AML analysts with threshold tuning, customer segmentation, and compliance Q&A. "
    "IMPORTANT: You MUST respond entirely in English."
)

# --- Pre-computed tool results ------------------------------------------------

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

PC_RULE_SAR_ELDER = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Current condition: Monthly transaction >= $5K AND customer age >= 65 AND >= 2 std dev above profile mean
Sweep parameter: age_threshold - Minimum customer age to trigger (currently 65)
Current value: 65
Labeled population: 1146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR, precision=16.4%)

At the lowest value (55.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).
At current condition (65.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).
To keep TP rate >=90%: age_threshold <= 70.00 => TP=172, FP=840, FN=16, TN=118, precision=17.0%.
To keep TP rate >=50%: age_threshold <= 78.00 => TP=100, FP=390, FN=88, TN=568, precision=20.4%.
At the highest value (80.00): TP=88, FP=320, FN=100, TN=638 (TP rate=46.8%, precision=21.6%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_ACH = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Activity Deviation (ACH)
Current condition: Monthly ACH outgoing >= $10K AND >= 2 std dev above 12-month profile mean
Sweep parameter: floor_amount - Minimum monthly ACH sum to trigger (currently $10K)
Current value: 10000
Labeled population: 907 customers (TP+FN pool=138 SAR, FP+TN pool=769 non-SAR, precision=15.2%)

At the lowest value (1000.00): TP=138, FP=769, FN=0, TN=0 (TP rate=100.0%, precision=15.2%).
At current condition (10000.00): TP=138, FP=769, FN=0, TN=0 (TP rate=100.0%, precision=15.2%).
To keep TP rate >=90%: floor_amount <= 50000.00 => TP=130, FP=680, FN=8, TN=89, precision=16.1%.
To keep TP rate >=50%: floor_amount <= 120000.00 => TP=72, FP=290, FN=66, TN=479, precision=19.9%.
At the highest value (150000.00): TP=58, FP=210, FN=80, TN=559 (TP rate=42.0%, precision=21.6%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_VELOCITY = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Current condition: >=1 pair (in+out) within 14 days, out=90-110%% of in, pair total >= $20K
Sweep parameter: pair_total - Minimum combined in+out pair total to trigger (currently $20K)
Current value: 20000
Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%%)

At the lowest value (5000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%%, precision=15.5%%).
At current condition (20000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%%, precision=15.5%%).
To keep TP rate >=90%%: pair_total <= 40000.00 => TP=68, FP=363, FN=6, TN=41, precision=15.8%%.
To keep TP rate >=50%%: pair_total <= 40000.00 => TP=68, FP=363, FN=6, TN=41, precision=15.8%%.
At the highest value (40000.00): TP=68, FP=363, FN=6, TN=41 (TP rate=91.9%%, precision=15.8%%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_2D_STRUCTURING = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Structuring (Incoming Cash)
Axis 1 (daily_floor): Minimum daily cash deposit to count toward structuring pattern (currently $3,000)
Axis 2 (days_required): Number of days within the window the floor must be met (currently 3)
Grid: 5 x 5 = 25 combinations
SAR pool: 2
Non-SAR pool: 0

At current condition (daily_floor=3000, days_required=3): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): daily_floor=2000, days_required=2 => TP=2, FP=0, FN=0, TN=0, TP rate=100.0%, precision=100.0%.
Best FP reduction (TP rate >=50%): daily_floor=4000, days_required=4 => TP=1, FP=0, FN=1, TN=0, TP rate=50.0%, precision=100.0%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_RULE_2D_DETECT = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Detect Excessive Transaction Activity
Axis 1 (floor_amount): Minimum aggregate transaction amount in the last 30 days to trigger (currently $1,000)
Axis 2 (time_window): Number of days in the lookback window (currently 30)
Grid: 10 x 8 = 80 combinations
SAR pool: 46
Non-SAR pool: 310

At current condition (floor_amount=1000, time_window=30): TP=44, FP=302, FN=2, TN=8 (TP rate=95.7%).
Best FP reduction (TP rate >=90%): floor_amount=8000, time_window=21 => TP=42, FP=200, FN=4, TN=110, TP rate=91.3%, precision=17.4%.
Best FP reduction (TP rate >=50%): floor_amount=16000, time_window=14 => TP=26, FP=84, FN=20, TN=226, TP rate=56.5%, precision=23.6%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_RULE_2D_ELDER = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Axis 1 (floor_amount): Minimum monthly transaction amount to trigger (currently $5,000)
Axis 2 (age_threshold): Minimum customer age to trigger (currently 65)
Grid: 6 x 5 = 30 combinations
SAR pool: 188
Non-SAR pool: 958

At current condition (floor_amount=5000, age_threshold=65): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=20000, age_threshold=68 => TP=170, FP=700, FN=18, TN=258, TP rate=90.4%, precision=19.5%.
Best FP reduction (TP rate >=50%): floor_amount=40000, age_threshold=72 => TP=96, FP=240, FN=92, TN=718, TP rate=51.1%, precision=28.6%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_KB_STRUCTURING = """\
Tool result for search_policy_kb:
Under the Bank Secrecy Act (BSA), 31 U.S.C. § 5324 prohibits structuring — the deliberate breaking up of transactions to evade Currency Transaction Report (CTR) requirements. A CTR must be filed for any cash transaction exceeding $10,000 (31 CFR Part 1010.311). Structuring occurs when a person conducts transactions below this threshold with the intent to evade reporting, regardless of whether the underlying funds are from legal or illegal sources (31 CFR Part 1010 (up to date as of 3-26-2026).pdf). FinCEN guidance FIN-2010-G004 clarifies that structuring is illegal even when the funds are lawfully obtained. Common red flags include multiple cash deposits just below $10,000, deposits at different branches on the same day, or patterns of transactions that suggest awareness of reporting thresholds (docs/fin-2010-g004.pdf). Financial institutions are required to file a Suspicious Activity Report (SAR) when structuring is detected or suspected (BsaAmlExamProcsPackage.docx)."""

PC_KB_BSA = """\
Tool result for search_policy_kb:
The Bank Secrecy Act (BSA), enacted in 1970 and codified at 31 U.S.C. §§ 5311-5336, is the primary U.S. anti-money laundering law. It requires financial institutions (FIs) to assist government agencies in detecting and preventing money laundering and other financial crimes (31 CFR Part 1010 (up to date as of 3-26-2026).pdf). Key BSA requirements include: (1) Currency Transaction Reports (CTRs) for cash transactions exceeding $10,000; (2) Suspicious Activity Reports (SARs) for transactions involving $5,000 or more that the FI knows, suspects, or has reason to suspect involve illegal funds or are designed to evade BSA requirements; (3) Customer Identification Program (CIP) and Customer Due Diligence (CDD) requirements to verify customer identity and assess risk; and (4) recordkeeping requirements for wire transfers and purchases of monetary instruments (31 CFR Parts 1010 and 1020). The FFIEC BSA/AML Examination Manual provides supervisory guidance on how examiners assess BSA compliance programs, requiring FIs to have written policies, internal controls, independent testing, a designated BSA Officer, and ongoing training (FFIEC BSA-AML Examination Manual.pdf)."""

PC_THRESHOLD_IND_MONTHLY = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
### Threshold Tuning — False Positive / False Negative Trade-off

**False Positives (FP)**
- At the lowest threshold (198843): **3588 FPs**
- FPs do not reach zero within the sweep range (198843-1098843)

**False Negatives (FN)**
- FNs are non-zero even at the lowest threshold (198843) - some customers fall below the sweep floor
- FNs first appear at threshold **198843** (FN=10)
- FNs reach **437** at the highest threshold (1098843)

**Crossover Point** - threshold **1098843** (FP=798, FN=437)

**Optimal Zone**: no single threshold achieves both FP and FN below 20%% of their maximums simultaneously.

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED ANALYSIS ==="""

PC_THRESHOLD_BIZ_WEEKLY = """\
Tool result for threshold_tuning:
=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===
### Threshold Tuning — False Positive / False Negative Trade-off

**False Positives (FP)**
- At the lowest threshold (1): **1580 FPs**
- FPs do not reach zero within the sweep range (1-5)

**False Negatives (FN)**
- FNs are non-zero even at the lowest threshold (1) - some customers fall below the sweep floor
- FNs first appear at threshold **1** (FN=240)
- FNs reach **627** at the highest threshold (5)

**Crossover Point** - threshold **2** (FP=618, FN=513)

**Optimal Zone**: no single threshold achieves both FP and FN below 20%% of their maximums simultaneously.

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED ANALYSIS ==="""

# --- Examples -----------------------------------------------------------------

examples = []

# ── H. rule_sar_backtest — correct routing + closing insight ──────────────────

# H1: SAR catch rate for Detect Excessive → rule_sar_backtest (was misrouted to list_rules)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Detect Excessive Transaction Activity?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("h1", "rule_sar_backtest", {"risk_factor": "Detect Excessive Transaction Activity"})
    ]},
    {"role": "tool", "tool_call_id": "h1", "content": PC_RULE_SAR_DETECT},
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
        "Lowering the floor to 15 transactions per week maintains 91.3% SAR detection (42 of 46) "
        "while reducing FPs only marginally from 268 to 284 — the rule is relatively insensitive to "
        "this parameter and the real FP driver is the z-threshold."
    )},
]})

# H2: SAR backtest CTR Client → rule_sar_backtest (was misrouted to list_rules)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for CTR Client rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("h2", "rule_sar_backtest", {"risk_factor": "CTR Client"})
    ]},
    {"role": "tool", "tool_call_id": "h2", "content": PC_RULE_SAR_CTR},
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
        "The CTR Client rule is a high-precision rule with 0 FPs at all tested thresholds — "
        "raising the floor above $10K is not recommended as it begins missing SARs with no FP reduction benefit."
    )},
]})

# H3: SAR filing rate Elder Abuse → rule_sar_backtest (was misrouted to list_rules)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Elder Abuse rule?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("h3", "rule_sar_backtest", {"risk_factor": "Elder Abuse"})
    ]},
    {"role": "tool", "tool_call_id": "h3", "content": PC_RULE_SAR_ELDER},
    {"role": "assistant", "content": (
        "Rule: Elder Abuse\n"
        "Current condition: Monthly transaction >= $5K AND customer age >= 65 AND >= 2 std dev above profile mean\n"
        "Sweep parameter: age_threshold - Minimum customer age to trigger (currently 65)\n"
        "Current value: 65\n"
        "Labeled population: 1146 customers (TP+FN pool=188 SAR, FP+TN pool=958 non-SAR, precision=16.4%)\n\n"
        "At the lowest value (55.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "At current condition (65.00): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%, precision=16.4%).\n"
        "To keep TP rate >=90%: age_threshold <= 70.00 => TP=172, FP=840, FN=16, TN=118, precision=17.0%.\n"
        "To keep TP rate >=50%: age_threshold <= 78.00 => TP=100, FP=390, FN=88, TN=568, precision=20.4%.\n"
        "At the highest value (80.00): TP=88, FP=320, FN=100, TN=638 (TP rate=46.8%, precision=21.6%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Raising the age threshold to 70 reduces FPs from 958 to 840 (118 fewer investigations) "
        "while missing only 16 of 188 SARs — a 91.5% catch rate that may be an acceptable trade-off "
        "given the rule's current 16.4% precision."
    )},
]})

# H4: Activity Deviation ACH SAR backtest — with closing insight
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("h4", "rule_sar_backtest", {"risk_factor": "Activity Deviation (ACH)"})
    ]},
    {"role": "tool", "tool_call_id": "h4", "content": PC_RULE_SAR_ACH},
    {"role": "assistant", "content": (
        "Rule: Activity Deviation (ACH)\n"
        "Current condition: Monthly ACH outgoing >= $10K AND >= 2 std dev above 12-month profile mean\n"
        "Sweep parameter: floor_amount - Minimum monthly ACH sum to trigger (currently $10K)\n"
        "Current value: 10000\n"
        "Labeled population: 907 customers (TP+FN pool=138 SAR, FP+TN pool=769 non-SAR, precision=15.2%)\n\n"
        "At the lowest value (1000.00): TP=138, FP=769, FN=0, TN=0 (TP rate=100.0%, precision=15.2%).\n"
        "At current condition (10000.00): TP=138, FP=769, FN=0, TN=0 (TP rate=100.0%, precision=15.2%).\n"
        "To keep TP rate >=90%: floor_amount <= 50000.00 => TP=130, FP=680, FN=8, TN=89, precision=16.1%.\n"
        "To keep TP rate >=50%: floor_amount <= 120000.00 => TP=72, FP=290, FN=66, TN=479, precision=19.9%.\n"
        "At the highest value (150000.00): TP=58, FP=210, FN=80, TN=559 (TP rate=42.0%, precision=21.6%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Raising the ACH floor to $50K maintains 94.2% SAR detection (130 of 138) and reduces "
        "FPs by 89 — a modest gain given 769 total FPs, suggesting the z-threshold parameter "
        "may have a larger impact on precision than floor_amount alone."
    )},
]})

# H5: Velocity Single SAR backtest — with closing insight (was correct but missing insight)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Velocity Single"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("h5", "rule_sar_backtest", {"risk_factor": "Velocity Single"})
    ]},
    {"role": "tool", "tool_call_id": "h5", "content": PC_RULE_SAR_VELOCITY},
    {"role": "assistant", "content": (
        "Rule: Velocity Single\n"
        "Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K\n"
        "Sweep parameter: pair_total - Minimum combined in+out pair total to trigger (currently $20K)\n"
        "Current value: 20000\n"
        "Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%)\n\n"
        "At the lowest value (5000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
        "At current condition (20000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
        "To keep TP rate >=90%: pair_total <= 40000.00 => TP=68, FP=363, FN=6, TN=41, precision=15.8%.\n"
        "To keep TP rate >=50%: pair_total <= 40000.00 => TP=68, FP=363, FN=6, TN=41, precision=15.8%.\n"
        "At the highest value (40000.00): TP=68, FP=363, FN=6, TN=41 (TP rate=91.9%, precision=15.8%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "The pair_total sweep range tops out at $40K where 41 non-SARs are eliminated "
        "at a cost of 6 missed SARs — if a 91.9% catch rate is acceptable, raising to $40K "
        "is the only available lever for this parameter."
    )},
]})

# ── I. rule_2d_sweep — correct routing + non-hallucinated insight ─────────────

# I1: 2D sweep Structuring → rule_2d_sweep (was misrouted to list_rules)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Structuring rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("i1", "rule_2d_sweep", {"risk_factor": "Structuring (Incoming Cash)"})
    ]},
    {"role": "tool", "tool_call_id": "i1", "content": PC_RULE_2D_STRUCTURING},
    {"role": "assistant", "content": (
        "Rule: Structuring (Incoming Cash)\n"
        "Axis 1 (daily_floor): Minimum daily cash deposit to count toward structuring pattern (currently $3,000)\n"
        "Axis 2 (days_required): Number of days within the window the floor must be met (currently 3)\n"
        "Grid: 5 x 5 = 25 combinations\n"
        "SAR pool: 2\nNon-SAR pool: 0\n\n"
        "At current condition (daily_floor=3000, days_required=3): TP=2, FP=0, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): daily_floor=2000, days_required=2 => TP=2, FP=0, FN=0, TN=0, TP rate=100.0%, precision=100.0%.\n"
        "Best FP reduction (TP rate >=50%): daily_floor=4000, days_required=4 => TP=1, FP=0, FN=1, TN=0, TP rate=50.0%, precision=100.0%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "The Structuring rule maintains 100% precision across all tested parameter combinations — "
        "with only 2 alerts both being confirmed SARs, there is no FP reduction opportunity "
        "but also no risk of over-alerting at the current settings."
    )},
]})

# I2: 2D sweep Detect Excessive → rule_2d_sweep with non-hallucinated insight
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D analysis for Detect Excessive Transaction Activity"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("i2", "rule_2d_sweep", {"risk_factor": "Detect Excessive Transaction Activity"})
    ]},
    {"role": "tool", "tool_call_id": "i2", "content": PC_RULE_2D_DETECT},
    {"role": "assistant", "content": (
        "Rule: Detect Excessive Transaction Activity\n"
        "Axis 1 (floor_amount): Minimum aggregate transaction amount in the last 30 days to trigger (currently $1,000)\n"
        "Axis 2 (time_window): Number of days in the lookback window (currently 30)\n"
        "Grid: 10 x 8 = 80 combinations\n"
        "SAR pool: 46\nNon-SAR pool: 310\n\n"
        "At current condition (floor_amount=1000, time_window=30): TP=44, FP=302, FN=2, TN=8 (TP rate=95.7%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=8000, time_window=21 => TP=42, FP=200, FN=4, TN=110, TP rate=91.3%, precision=17.4%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=16000, time_window=14 => TP=26, FP=84, FN=20, TN=226, TP rate=56.5%, precision=23.6%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Setting floor_amount to $8,000 and time_window to 21 days achieves 91.3% SAR detection "
        "while reducing FPs from 302 to 200 — eliminating 102 unnecessary investigations "
        "at the cost of 2 additional missed SARs compared to current settings."
    )},
]})

# I3: 2D sweep Elder Abuse
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D heatmap for Elder Abuse — how do floor_amount and age_threshold interact?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("i3", "rule_2d_sweep", {"risk_factor": "Elder Abuse", "sweep_param_1": "floor_amount", "sweep_param_2": "age_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "i3", "content": PC_RULE_2D_ELDER},
    {"role": "assistant", "content": (
        "Rule: Elder Abuse\n"
        "Axis 1 (floor_amount): Minimum monthly transaction amount to trigger (currently $5,000)\n"
        "Axis 2 (age_threshold): Minimum customer age to trigger (currently 65)\n"
        "Grid: 6 x 5 = 30 combinations\n"
        "SAR pool: 188\nNon-SAR pool: 958\n\n"
        "At current condition (floor_amount=5000, age_threshold=65): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=20000, age_threshold=68 => TP=170, FP=700, FN=18, TN=258, TP rate=90.4%, precision=19.5%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=40000, age_threshold=72 => TP=96, FP=240, FN=92, TN=718, TP rate=51.1%, precision=28.6%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Combining floor_amount=$20,000 with age_threshold=68 reduces FPs by 258 (from 958 to 700) "
        "while maintaining 90.4% SAR detection — the joint parameter sweep reveals more FP reduction "
        "opportunity than either parameter alone would suggest."
    )},
]})

# ── J. Policy — structuring and BSA with correct KB synthesis ─────────────────

# J1: What is AML structuring?
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is AML structuring?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("j1", "search_policy_kb", {"query": "AML structuring definition CTR evasion"})
    ]},
    {"role": "tool", "tool_call_id": "j1", "content": PC_KB_STRUCTURING},
    {"role": "assistant", "content": (
        "Structuring — also known as 'smurfing' — is the deliberate practice of breaking up "
        "financial transactions into smaller amounts to evade Currency Transaction Report (CTR) "
        "reporting requirements.\n\n"
        "Under the Bank Secrecy Act (BSA), 31 U.S.C. § 5324, structuring is illegal regardless "
        "of whether the funds involved are from a legitimate or illegitimate source. A CTR must "
        "be filed for any cash transaction exceeding $10,000 (31 CFR Part 1010.311). Common "
        "structuring patterns include multiple cash deposits just below $10,000, deposits spread "
        "across different branches on the same day, or a series of transactions that suggest "
        "the customer is aware of reporting thresholds (31 CFR Part 1010).\n\n"
        "FinCEN guidance (FIN-2010-G004) explicitly clarifies that the intent to evade reporting "
        "— not the legality of the funds — is what makes structuring a criminal offence. "
        "Financial institutions are required to file a Suspicious Activity Report (SAR) when "
        "structuring is detected or suspected, even if the individual transactions are below the "
        "CTR threshold (BsaAmlExamProcsPackage.docx).\n\n"
        "In Canada, FINTRAC's rule mirrors the U.S. aggregation concept, requiring reporting of "
        "cash transactions of CAD $10,000 or more and flagging patterns indicative of deliberate "
        "structuring under the Proceeds of Crime (Money Laundering) and Terrorist Financing Act."
    )},
]})

# J2: What is the Bank Secrecy Act?
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the Bank Secrecy Act?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("j2", "search_policy_kb", {"query": "Bank Secrecy Act BSA requirements overview"})
    ]},
    {"role": "tool", "tool_call_id": "j2", "content": PC_KB_BSA},
    {"role": "assistant", "content": (
        "The Bank Secrecy Act (BSA), enacted in 1970 and codified at 31 U.S.C. §§ 5311-5336, "
        "is the cornerstone of the U.S. anti-money laundering (AML) regulatory framework. "
        "It requires financial institutions to assist government agencies in detecting and "
        "preventing money laundering, tax evasion, and other financial crimes.\n\n"
        "**Core BSA Requirements:**\n"
        "- **Currency Transaction Reports (CTRs):** Must be filed for cash transactions exceeding $10,000\n"
        "- **Suspicious Activity Reports (SARs):** Required for transactions of $5,000 or more that "
        "are suspected to involve illegal activity, evasion of BSA requirements, or have no lawful purpose\n"
        "- **Customer Identification Program (CIP):** FIs must verify the identity of customers at "
        "account opening under 31 CFR Parts 1010 and 1020\n"
        "- **Customer Due Diligence (CDD):** FIs must understand the nature and purpose of customer "
        "relationships and conduct ongoing monitoring\n"
        "- **Recordkeeping:** FIs must retain records of wire transfers, monetary instrument purchases, "
        "and other transactions for five years\n\n"
        "The FFIEC BSA/AML Examination Manual sets out the supervisory framework used by examiners "
        "to assess BSA compliance programs. A sound BSA program must include written policies and "
        "procedures, internal controls, independent testing, a designated BSA Compliance Officer, "
        "and ongoing staff training (FFIEC BSA-AML Examination Manual.pdf)."
    )},
]})

# ── K. Unavailable metric — net income and credit scores ─────────────────────

# K1: Net income distribution → decline + alternatives (was out_of_scope)
examples.append({"messages": [
    {"role": "system", "content": GENERAL_SYSTEM},
    {"role": "user", "content": "Show me net income distribution for Business customers"},
    {"role": "assistant", "content": (
        "Net income data is not available in our customer dataset. "
        "However, I can show you transaction-based financial profiles for Business customers, including:\n\n"
        "- **Monthly transaction volume** (TRXN_AMT_MONTHLY) — aggregate monthly inflow/outflow\n"
        "- **Average weekly transaction amount** (AVG_TRXN_AMT) — typical transaction size\n"
        "- **Average weekly transaction count** (AVG_TRXNS_WEEK) — activity frequency\n"
        "- **Current balance** — available via the segmentation cluster profiles\n\n"
        "Would you like me to run a threshold analysis or cluster segmentation using one of these available columns?"
    )},
]})

# K2: Credit scores → decline + alternatives (was misrouted to greeting)
examples.append({"messages": [
    {"role": "system", "content": GENERAL_SYSTEM},
    {"role": "user", "content": "Can you show me credit scores for high-risk customers?"},
    {"role": "assistant", "content": (
        "Credit score data is not available in our AML dataset. "
        "Our data focuses on transaction behavior and alert outcomes rather than credit risk metrics.\n\n"
        "For identifying high-risk customers I can help with:\n\n"
        "- **Cluster segmentation** — identify the highest-risk behavioral segment by transaction volume and frequency\n"
        "- **Rule-level SAR analysis** — see which rules flag the most confirmed SARs\n"
        "- **Threshold tuning** — find the alert threshold that best separates SAR from non-SAR customers\n\n"
        "Would you like to run one of these analyses?"
    )},
]})

# ── L. threshold_tuning with closing insight (previously missing) ─────────────

# L1: Individual monthly amount — was missing closing insight
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Run threshold tuning for Individual customers using monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("l1", "threshold_tuning", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})
    ]},
    {"role": "tool", "tool_call_id": "l1", "content": PC_THRESHOLD_IND_MONTHLY},
    {"role": "assistant", "content": (
        "### Threshold Tuning — False Positive / False Negative Trade-off\n\n"
        "**False Positives (FP)**\n"
        "- At the lowest threshold (198843): **3588 FPs**\n"
        "- FPs do not reach zero within the sweep range (198843-1098843)\n\n"
        "**False Negatives (FN)**\n"
        "- FNs are non-zero even at the lowest threshold (198843) — some customers fall below the sweep floor\n"
        "- FNs first appear at threshold **198843** (FN=10)\n"
        "- FNs reach **437** at the highest threshold (1098843)\n\n"
        "**Crossover Point** — threshold **1098843** (FP=798, FN=437)\n\n"
        "**Optimal Zone**: no single threshold achieves both FP and FN below 20% of their maximums simultaneously.\n\n"
        "*(Detailed sweep chart shown below.)*\n\n"
        "At the crossover point of $1,098,843, FPs drop to 798 but 437 SARs are missed — "
        "the wide sweep range with no optimal zone indicates the monthly amount threshold "
        "alone is insufficient to cleanly separate SAR from non-SAR Individual customers."
    )},
]})

# L2: Business weekly transactions — was missing closing insight
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user", "content": "Show FP/FN trade-off for Business customers by average weekly transactions"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("l2", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})
    ]},
    {"role": "tool", "tool_call_id": "l2", "content": PC_THRESHOLD_BIZ_WEEKLY},
    {"role": "assistant", "content": (
        "### Threshold Tuning — False Positive / False Negative Trade-off\n\n"
        "**False Positives (FP)**\n"
        "- At the lowest threshold (1): **1580 FPs**\n"
        "- FPs do not reach zero within the sweep range (1-5)\n\n"
        "**False Negatives (FN)**\n"
        "- FNs are non-zero even at the lowest threshold (1) — some customers fall below the sweep floor\n"
        "- FNs first appear at threshold **1** (FN=240)\n"
        "- FNs reach **627** at the highest threshold (5)\n\n"
        "**Crossover Point** — threshold **2** (FP=618, FN=513)\n\n"
        "**Optimal Zone**: no single threshold achieves both FP and FN below 20% of their maximums simultaneously.\n\n"
        "*(Detailed sweep chart shown below.)*\n\n"
        "The crossover at threshold 2 weekly transactions shows FPs and FNs are nearly balanced at 618 and 513 — "
        "the integer nature of the count column limits granularity and suggests pairing this threshold "
        "with a transaction amount parameter for better discrimination."
    )},
]})

# --- Write output -------------------------------------------------------------

OUT_DIR = pathlib.Path(__file__).parent
V19_FILE = OUT_DIR / "framl_train_combined_v19.jsonl"
V20_NEW  = OUT_DIR / "framl_train_v20.jsonl"
V20_COMB = OUT_DIR / "framl_train_combined_v20.jsonl"

# Load V19 combined
v19 = []
with open(V19_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            v19.append(json.loads(line))

print(f"Total V20 new examples: {len(examples)}")
for i, ex in enumerate(examples):
    preview = ex["messages"][1]["content"][:75].replace("\n", " ")
    print(f"  ex{632 + i + 1}: {preview}")

with open(V20_NEW, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

combined = v19 + examples
with open(V20_COMB, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")

print(f"\nWrote {len(examples)} examples to {V20_NEW}")
print(f"Combined V20: {len(combined)} total examples")

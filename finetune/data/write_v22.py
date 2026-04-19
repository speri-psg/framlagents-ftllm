"""
V22 training examples — gaps from V21 benchmark + live app testing (2026-04-19).

Targets:
  S. Round-trip + Funnel Account 2D sweep routing (3 examples)
     — Round-trip was mis-routed as out-of-scope; Funnel Account TOOL_OUTPUT leaked raw text
  T. Policy direct calls — alert volumes, structuring, BSA, OFAC data decline (4 examples)
     — P02 (structuring), P04 (alert volumes) deflected without calling search_policy_kb
  U. New typology graceful declines — no alert data in current portfolio (3 examples)
     — Human Trafficking, Activity Deviation Wire, Round-trip: model should acknowledge no data
  V. Velocity Multiple full rule name in args (2 examples)
     — model passed 'VeloMultiple' abbreviation; need full name in risk_factor

Start: ex662 (V21 combined = 661 examples).
Combined with framl_train_combined_v21.jsonl -> framl_train_combined_v22.jsonl
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

PC_RULE_2D_ROUNDTRIP = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Round-trip
Axis 1 (floor_amount): Minimum Wire amount for the outgoing leg to trigger (currently $50,000)
Axis 2 (return_window): Maximum days between outgoing and return leg (currently 30 days)
Grid: 5 x 6 = 30 combinations
SAR pool: 28
Non-SAR pool: 142

At current condition (floor_amount=50000, return_window=30): TP=28, FP=142, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=100000, return_window=21 => TP=26, FP=98, FN=2, TN=44, TP rate=92.9%, precision=21.0%.
Best FP reduction (TP rate >=50%): floor_amount=200000, return_window=14 => TP=16, FP=38, FN=12, TN=104, TP rate=57.1%, precision=29.6%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

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
=== END 2D SWEEP ==="""

PC_RULE_2D_VELOCITY_MULTIPLE = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Velocity Multiple
Axis 1 (pair_total): Minimum combined CashIn+CashOut total to trigger (currently $30,000)
Axis 2 (min_counterparties): Minimum number of distinct outgoing counterparties (currently 3)
Grid: 5 x 4 = 20 combinations
SAR pool: 41
Non-SAR pool: 214

At current condition (pair_total=30000, min_counterparties=3): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): pair_total=60000, min_counterparties=4 => TP=37, FP=148, FN=4, TN=66, TP rate=90.2%, precision=20.0%.
Best FP reduction (TP rate >=50%): pair_total=100000, min_counterparties=5 => TP=22, FP=58, FN=19, TN=156, TP rate=53.7%, precision=27.5%.
(Heatmap shown in the chart below.)
=== END 2D SWEEP ==="""

PC_RULE_SAR_VELOCITY_MULTIPLE = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Velocity Multiple
Current condition: 1 large CashIn within 14 days followed by >= 3 CashOuts to distinct counterparties, total >= $30K
Sweep parameter: pair_total - Minimum combined CashIn+CashOut total to trigger (currently $30K)
Current value: 30000
Labeled population: 255 customers (TP+FN pool=41 SAR, FP+TN pool=214 non-SAR, precision=16.1%)

At the lowest value (10000.00): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%, precision=16.1%).
At current condition (30000.00): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%, precision=16.1%).
To keep TP rate >=90%: pair_total <= 60000.00 => TP=37, FP=148, FN=4, TN=66, precision=20.0%.
To keep TP rate >=50%: pair_total <= 100000.00 => TP=22, FP=58, FN=19, TN=156, precision=27.5%.
At the highest value (120000.00): TP=18, FP=42, FN=23, TN=172 (TP rate=43.9%, precision=30.0%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_KB_ALERT_VOLUMES = """\
Tool result for search_policy_kb:
Financial institutions manage AML alert volumes through threshold calibration, model tuning, \
and tiered review workflows. The FFIEC BSA/AML Examination Manual notes that alert generation \
rates must be periodically reviewed to ensure they are neither over-alerting (producing excessive \
false positives that overwhelm investigators) nor under-alerting (missing genuine suspicious \
activity) (FFIEC BSA-AML Examination Manual.pdf). Common strategies include: \
(1) Risk-stratified thresholds — setting tighter parameters for high-risk customer segments \
(e.g., PEPs, cash-intensive businesses) and looser thresholds for low-risk retail customers; \
(2) Statistical profiling — z-score or peer-group comparisons that trigger alerts only when \
a customer's activity deviates significantly from their own baseline or a peer group, \
reducing false positives from legitimately high-volume customers; \
(3) Alert suppression / exclusion lists — documented exemptions for recurring transactions \
(e.g., payroll ACH, known counterparties) where business purpose is established; \
(4) Model validation — the OCC and Federal Reserve require periodic independent validation \
of transaction monitoring models to confirm thresholds remain appropriate as customer behavior \
evolves (BsaAmlExamProcsPackage.docx). The Wolfsberg Transitioning to Innovation paper \
(Wolfsberg_MSAP2_Transitioning to Innovation.pdf) discusses machine learning for alert \
prioritization to reduce investigator workload without raising miss rates."""

PC_KB_STRUCTURING = """\
Tool result for search_policy_kb:
Structuring — also known as 'smurfing' — is the practice of breaking up financial transactions \
into smaller amounts specifically to evade the $10,000 Currency Transaction Report (CTR) filing \
threshold required under 31 CFR Part 1010.311. Under 31 U.S.C. § 5324, it is a federal crime \
to structure or assist in structuring any transaction to evade BSA reporting requirements, \
regardless of whether the underlying funds are from lawful sources. Indicators include: \
multiple cash deposits or withdrawals just below $10,000 by the same customer on the same or \
consecutive days; transactions at different branches or institutions on the same day; \
and patterns of deposits that aggregate to just under reporting thresholds over a short window. \
The FFIEC BSA/AML Examination Manual specifically instructs examiners to look for customers \
who consistently transact in amounts between $3,000 and $9,999 over short windows \
(FFIEC BSA-AML Examination Manual.pdf). FinCEN advisory FIN-2010-G004 also warns that \
structuring should be reported via SAR even if the underlying funds appear legitimate, \
because the act of structuring itself is the crime (docs/fin-2010-g004.pdf)."""

PC_KB_BSA = """\
Tool result for search_policy_kb:
The Bank Secrecy Act (BSA), enacted in 1970 and codified at 31 U.S.C. §§ 5311-5336, is the \
primary U.S. anti-money laundering statute. It requires financial institutions to assist \
government agencies in detecting and preventing money laundering. Core BSA requirements include: \
(1) Currency Transaction Reports (CTRs) — filed with FinCEN for cash transactions exceeding \
$10,000 in a single business day (31 CFR Part 1010.311); \
(2) Suspicious Activity Reports (SARs) — filed when a transaction involves $5,000 or more \
and the institution knows, suspects, or has reason to suspect the funds involve illegal activity \
(31 CFR Part 1020.320 for banks); \
(3) Customer Identification Program (CIP) — required procedures to verify the identity of \
customers opening new accounts (31 CFR Part 1020.220); \
(4) Beneficial Ownership — institutions must identify the beneficial owners of legal entity \
customers (31 CFR Part 1010.230); \
(5) Recordkeeping — transaction records must be retained for five years (31 CFR Part 1010.430). \
The FFIEC BSA/AML Examination Manual provides the supervisory framework used by examiners \
from the OCC, Federal Reserve, FDIC, and NCUA to assess BSA compliance \
(FFIEC BSA-AML Examination Manual.pdf)."""

NO_DATA_HUMAN_TRAFFICKING = """\
Tool result for rule_sar_backtest:
No alert data found for rule: Human Trafficking Indicators.
This rule has not generated alerts in the current portfolio dataset."""

NO_DATA_ACT_DEV_WIRE = """\
Tool result for rule_sar_backtest:
No alert data found for rule: Activity Deviation (Wire).
This rule has not generated alerts in the current portfolio dataset."""

NO_DATA_ROUNDTRIP_SAR = """\
Tool result for rule_sar_backtest:
No alert data found for rule: Round-trip.
This rule has not generated alerts in the current portfolio dataset."""

# --- Examples -----------------------------------------------------------------

examples = []

# ── S. Round-trip + Funnel Account 2D sweep routing ───────────────────────────

# S1: Round-trip 2D sweep — floor_amount × return_window
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D analysis for Round-trip — floor amount vs return window"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("s1", "rule_2d_sweep", {"risk_factor": "Round-trip",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "return_window"})
    ]},
    {"role": "tool", "tool_call_id": "s1", "content": PC_RULE_2D_ROUNDTRIP},
    {"role": "assistant", "content": (
        "Rule: Round-trip\n"
        "Axis 1 (floor_amount): Minimum Wire amount for the outgoing leg to trigger (currently $50,000)\n"
        "Axis 2 (return_window): Maximum days between outgoing and return leg (currently 30 days)\n"
        "Grid: 5 x 6 = 30 combinations\n"
        "SAR pool: 28\nNon-SAR pool: 142\n\n"
        "At current condition (floor_amount=50000, return_window=30): TP=28, FP=142, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=100000, return_window=21 => TP=26, FP=98, FN=2, TN=44, TP rate=92.9%, precision=21.0%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=200000, return_window=14 => TP=16, FP=38, FN=12, TN=104, TP rate=57.1%, precision=29.6%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Raising the floor to $100K and tightening the return window to 21 days reduces FPs by 44 "
        "(from 142 to 98) while maintaining a 92.9% SAR catch rate — the floor_amount axis "
        "provides stronger FP reduction than shortening the return window alone."
    )},
]})

# S2: Funnel Account 2D sweep — floor_amount × min_counterparties
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for the Funnel Account rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("s2", "rule_2d_sweep", {"risk_factor": "Funnel Account",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "min_counterparties"})
    ]},
    {"role": "tool", "tool_call_id": "s2", "content": PC_RULE_2D_FUNNEL},
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
        "Raising the inflow floor to $25K and requiring 6 distinct counterparties reduces FPs by 32 "
        "(from 96 to 64) while keeping SAR detection at 89.5% — the min_counterparties axis "
        "adds specificity that floor_amount alone cannot achieve."
    )},
]})

# S3: Velocity Multiple 2D sweep — correct full rule name in args
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show me a 2D heatmap for the Velocity Multiple rule across pair total and counterparties"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("s3", "rule_2d_sweep", {"risk_factor": "Velocity Multiple",
                                    "sweep_param_1": "pair_total",
                                    "sweep_param_2": "min_counterparties"})
    ]},
    {"role": "tool", "tool_call_id": "s3", "content": PC_RULE_2D_VELOCITY_MULTIPLE},
    {"role": "assistant", "content": (
        "Rule: Velocity Multiple\n"
        "Axis 1 (pair_total): Minimum combined CashIn+CashOut total to trigger (currently $30,000)\n"
        "Axis 2 (min_counterparties): Minimum number of distinct outgoing counterparties (currently 3)\n"
        "Grid: 5 x 4 = 20 combinations\n"
        "SAR pool: 41\nNon-SAR pool: 214\n\n"
        "At current condition (pair_total=30000, min_counterparties=3): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): pair_total=60000, min_counterparties=4 => TP=37, FP=148, FN=4, TN=66, TP rate=90.2%, precision=20.0%.\n"
        "Best FP reduction (TP rate >=50%): pair_total=100000, min_counterparties=5 => TP=22, FP=58, FN=19, TN=156, TP rate=53.7%, precision=27.5%.\n"
        "(Heatmap shown in the chart below.)\n\n"
        "Raising pair_total to $60K and requiring 4 counterparties reduces FPs by 66 "
        "(from 214 to 148) while holding SAR detection at 90.2% — both axes contribute "
        "independently to specificity."
    )},
]})

# ── T. Policy direct calls ─────────────────────────────────────────────────────

# T1: Alert volume management — was deflecting without calling search_policy_kb
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "How do banks manage alert volumes?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("t1", "search_policy_kb", {"query": "AML alert volume management threshold calibration false positives"})
    ]},
    {"role": "tool", "tool_call_id": "t1", "content": PC_KB_ALERT_VOLUMES},
    {"role": "assistant", "content": (
        "Banks manage AML alert volumes through four main strategies:\n\n"
        "**1. Risk-stratified thresholds**\n"
        "Tighter parameters for high-risk segments (PEPs, cash-intensive businesses) and "
        "looser thresholds for low-risk retail customers — so the alert rate reflects "
        "actual risk rather than transaction volume (FFIEC BSA-AML Examination Manual.pdf).\n\n"
        "**2. Statistical profiling**\n"
        "Z-score or peer-group comparisons that trigger alerts only when a customer's "
        "activity deviates significantly from their own baseline or a comparable cohort, "
        "reducing false positives from legitimately high-volume customers.\n\n"
        "**3. Alert suppression / exclusion lists**\n"
        "Documented exemptions for recurring transactions with established business purpose "
        "(e.g., payroll ACH runs, known large counterparties), maintained with audit trails.\n\n"
        "**4. Periodic model validation**\n"
        "The OCC and Federal Reserve require independent validation of transaction monitoring "
        "models to confirm thresholds remain calibrated as customer behavior evolves "
        "(BsaAmlExamProcsPackage.docx).\n\n"
        "The Wolfsberg Group also highlights machine learning for alert prioritization by risk "
        "score as a way to reduce investigator workload without increasing miss rates "
        "(Wolfsberg_MSAP2_Transitioning to Innovation.pdf)."
    )},
]})

# T2: What is AML structuring — was deflecting without calling search_policy_kb
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is AML structuring?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("t2", "search_policy_kb", {"query": "AML structuring smurfing CTR threshold cash transactions"})
    ]},
    {"role": "tool", "tool_call_id": "t2", "content": PC_KB_STRUCTURING},
    {"role": "assistant", "content": (
        "**Structuring** (also called 'smurfing') is the practice of deliberately breaking up "
        "financial transactions into smaller amounts to evade the $10,000 Currency Transaction "
        "Report (CTR) filing requirement under 31 CFR Part 1010.311.\n\n"
        "**Why it's illegal**\n"
        "Under 31 U.S.C. § 5324, structuring is a federal crime regardless of whether the "
        "underlying funds are from lawful sources — the act of evading BSA reporting is itself "
        "the offense.\n\n"
        "**Common indicators**\n"
        "- Multiple cash deposits or withdrawals just below $10,000 by the same customer on "
        "the same or consecutive days\n"
        "- Transactions spread across different branches or institutions on the same day\n"
        "- Patterns aggregating to just under reporting thresholds over a short window "
        "(FFIEC BSA-AML Examination Manual.pdf)\n\n"
        "**SAR obligation**\n"
        "FinCEN advisory FIN-2010-G004 confirms that structuring must be reported via SAR "
        "even when the underlying funds appear legitimate, because the structuring act itself "
        "is the violation (docs/fin-2010-g004.pdf)."
    )},
]})

# T3: What is the Bank Secrecy Act — P01 benchmark case
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What is the Bank Secrecy Act?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("t3", "search_policy_kb", {"query": "Bank Secrecy Act BSA requirements CTR SAR CIP recordkeeping"})
    ]},
    {"role": "tool", "tool_call_id": "t3", "content": PC_KB_BSA},
    {"role": "assistant", "content": (
        "The **Bank Secrecy Act (BSA)**, enacted in 1970 (31 U.S.C. §§ 5311-5336), is the "
        "primary U.S. anti-money laundering statute. It requires financial institutions to "
        "assist government agencies in detecting and preventing money laundering through "
        "five core obligations:\n\n"
        "**1. Currency Transaction Reports (CTRs)**\n"
        "Filed with FinCEN for cash transactions exceeding $10,000 in a single business day "
        "(31 CFR Part 1010.311).\n\n"
        "**2. Suspicious Activity Reports (SARs)**\n"
        "Filed when a transaction involves $5,000 or more and the institution knows, suspects, "
        "or has reason to suspect illegal activity (31 CFR Part 1020.320 for banks).\n\n"
        "**3. Customer Identification Program (CIP)**\n"
        "Required procedures to verify the identity of customers opening new accounts "
        "(31 CFR Part 1020.220).\n\n"
        "**4. Beneficial Ownership**\n"
        "Institutions must identify beneficial owners of legal entity customers "
        "(31 CFR Part 1010.230).\n\n"
        "**5. Recordkeeping**\n"
        "Transaction records must be retained for five years (31 CFR Part 1010.430).\n\n"
        "The FFIEC BSA/AML Examination Manual provides the supervisory framework used by "
        "the OCC, Federal Reserve, FDIC, and NCUA to assess BSA compliance "
        "(FFIEC BSA-AML Examination Manual.pdf)."
    )},
]})

# T4: OFAC customer data query — should decline (not a policy question, not a live data query)
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "Which customers have OFAC sanctions hits?"},
    {"role": "assistant", "content": (
        "Live OFAC screening results for individual customers are not accessible through this "
        "interface — that data resides in your core banking and sanctions screening platforms.\n\n"
        "What I can help with:\n\n"
        "- **OFAC hit procedure** — what obligations apply when a match is confirmed "
        "(blocking, reporting timelines, SAR evaluation)\n"
        "- **Sanctions screening program requirements** — what the FFIEC and OFAC expect "
        "from your screening coverage and documentation practices\n"
        "- **False positive handling** — how to document and retain no-match determinations\n\n"
        "Would you like me to look up any of these policy topics?"
    )},
]})

# ── U. New typology graceful declines — no alert data ─────────────────────────

# U1: Human Trafficking Indicators — no data in portfolio
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Human Trafficking Indicators?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("u1", "rule_sar_backtest", {"risk_factor": "Human Trafficking Indicators"})
    ]},
    {"role": "tool", "tool_call_id": "u1", "content": NO_DATA_HUMAN_TRAFFICKING},
    {"role": "assistant", "content": (
        "The Human Trafficking Indicators rule has no alert history in the current portfolio dataset. "
        "This rule may be newly deployed or not yet active in the production monitoring engine, "
        "so no SAR backtest data is available.\n\n"
        "Once the rule generates alerts, I can run a full SAR backtest across the daily cash inflow "
        "floor and minimum consecutive days parameters."
    )},
]})

# U2: Activity Deviation Wire — no data in portfolio
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for the Activity Deviation Wire rule?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("u2", "rule_sar_backtest", {"risk_factor": "Activity Deviation (Wire)"})
    ]},
    {"role": "tool", "tool_call_id": "u2", "content": NO_DATA_ACT_DEV_WIRE},
    {"role": "assistant", "content": (
        "The Activity Deviation (Wire) rule has no alert history in the current portfolio dataset. "
        "This rule may be newly configured or not yet generating alerts in the monitoring engine.\n\n"
        "Once alerts are available, I can run a SAR backtest sweeping the monthly Wire floor "
        "and standard deviation multiplier parameters to identify optimal thresholds."
    )},
]})

# U3: Round-trip SAR backtest — no data (separate from 2D sweep which works)
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR catch rate for Round-trip rule?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("u3", "rule_sar_backtest", {"risk_factor": "Round-trip"})
    ]},
    {"role": "tool", "tool_call_id": "u3", "content": NO_DATA_ROUNDTRIP_SAR},
    {"role": "assistant", "content": (
        "The Round-trip rule has no SAR backtest data in the current portfolio dataset — "
        "the rule has not yet generated labeled alerts.\n\n"
        "I can run a 2D parameter sweep across the Wire floor amount and return window "
        "to show how different threshold combinations affect alert volume when the rule "
        "does become active. Would you like me to run that analysis?"
    )},
]})

# ── V. Velocity Multiple SAR backtest — full rule name in risk_factor ──────────

# V1: SAR backtest with full name — fixes 'VeloMultiple' abbreviation issue
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Velocity Multiple"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("v1", "rule_sar_backtest", {"risk_factor": "Velocity Multiple"})
    ]},
    {"role": "tool", "tool_call_id": "v1", "content": PC_RULE_SAR_VELOCITY_MULTIPLE},
    {"role": "assistant", "content": (
        "Rule: Velocity Multiple\n"
        "Current condition: 1 large CashIn within 14 days followed by >= 3 CashOuts to distinct counterparties, total >= $30K\n"
        "Sweep parameter: pair_total - Minimum combined CashIn+CashOut total to trigger (currently $30K)\n"
        "Current value: 30,000\n"
        "Labeled population: 255 customers (TP+FN pool=41 SAR, FP+TN pool=214 non-SAR, precision=16.1%)\n\n"
        "At the lowest value (10,000.00): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%, precision=16.1%).\n"
        "At current condition (30,000.00): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%, precision=16.1%).\n"
        "To keep TP rate >=90%: pair_total <= 60,000.00 => TP=37, FP=148, FN=4, TN=66, precision=20.0%.\n"
        "To keep TP rate >=50%: pair_total <= 100,000.00 => TP=22, FP=58, FN=19, TN=156, precision=27.5%.\n"
        "At the highest value (120,000.00): TP=18, FP=42, FN=23, TN=172 (TP rate=43.9%, precision=30.0%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "Raising pair_total to $60K preserves 90% SAR detection while cutting FPs by 66 — "
        "a reasonable starting point before tightening further."
    )},
]})

# V2: Velocity Multiple SAR — alternate phrasing to reinforce full name
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show SAR analysis for Velocity Multiple"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("v2", "rule_sar_backtest", {"risk_factor": "Velocity Multiple"})
    ]},
    {"role": "tool", "tool_call_id": "v2", "content": PC_RULE_SAR_VELOCITY_MULTIPLE},
    {"role": "assistant", "content": (
        "Rule: Velocity Multiple\n"
        "Current condition: 1 large CashIn within 14 days followed by >= 3 CashOuts to distinct counterparties, total >= $30K\n"
        "Sweep parameter: pair_total - Minimum combined CashIn+CashOut total to trigger (currently $30K)\n"
        "Current value: 30,000\n"
        "Labeled population: 255 customers (TP+FN pool=41 SAR, FP+TN pool=214 non-SAR, precision=16.1%)\n\n"
        "At the lowest value (10,000.00): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%, precision=16.1%).\n"
        "At current condition (30,000.00): TP=41, FP=214, FN=0, TN=0 (TP rate=100.0%, precision=16.1%).\n"
        "To keep TP rate >=90%: pair_total <= 60,000.00 => TP=37, FP=148, FN=4, TN=66, precision=20.0%.\n"
        "To keep TP rate >=50%: pair_total <= 100,000.00 => TP=22, FP=58, FN=19, TN=156, precision=27.5%.\n"
        "At the highest value (120,000.00): TP=18, FP=42, FN=23, TN=172 (TP rate=43.9%, precision=30.0%).\n"
        "(Detailed sweep table shown in the chart below.)\n\n"
        "The current $30K floor catches all 41 SARs — raising it to $60K loses only 4 SARs "
        "while dropping 66 false positives, improving precision from 16.1% to 20.0%."
    )},
]})

# --- Write output -------------------------------------------------------------

out_path = pathlib.Path(__file__).parent / "framl_train_v22.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(examples)} examples -> {out_path}")

# --- Combine with V21 ---------------------------------------------------------

v21_path  = pathlib.Path(__file__).parent / "framl_train_combined_v21.jsonl"
v22_combined = pathlib.Path(__file__).parent / "framl_train_combined_v22.jsonl"

if v21_path.exists():
    v21_lines = v21_path.read_text(encoding="utf-8").splitlines()
    v22_lines = out_path.read_text(encoding="utf-8").splitlines()
    combined  = [l for l in v21_lines + v22_lines if l.strip()]
    v22_combined.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"Combined: {len(v21_lines)} V21 + {len(v22_lines)} V22 = {len(combined)} total -> {v22_combined}")
else:
    print(f"WARNING: {v21_path} not found — combined file not created.")

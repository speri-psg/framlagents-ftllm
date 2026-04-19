"""
V25 training examples (2026-04-19).

Targets:
  F. Fake-rule graceful declines (4 examples)
     — Funnel Account, Round-trip, Velocity Multiple, Human Trafficking Indicators
     — model must call list_rules -> see alerts=0 -> decline, suggest real rules
  C. Cluster-filtered rule analysis (2 examples)
     — Elder Abuse Cluster 4 SAR backtest (no-text synthesis gap)
     — Velocity Single Cluster 2 SAR backtest
  R. Real rule coverage (2 examples)
     — Activity Deviation (Check) 2D sweep
     — Burst in Originator Activity SAR backtest
  P. Policy KB call (1 example)
     — Alert volume thresholds (model answered from memory instead of KB)
  Q. Clean synthesis fixes (3 examples)
     — Velocity Single pair_total (no "REETING" artifact)
     — Elder Abuse 2D sweep (no "per Abuse" artifact)
     — Activity Deviation ACH SAR backtest (no "RATE" prefix artifact)

Combined with framl_train_combined_v24.jsonl, MINUS 7 fake-rule examples.
  V24 combined: 687 examples
  Minus 7 fake: 680 clean base
  Plus 12 V25:  692 total -> framl_train_combined_v25.jsonl
"""

import json, pathlib, sys

DATA_DIR  = pathlib.Path(__file__).parent
V24_PATH  = DATA_DIR / "framl_train_combined_v24.jsonl"
V25_PATH  = DATA_DIR / "framl_train_combined_v25.jsonl"
THIS_PATH = DATA_DIR / "framl_train_v25.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _contains_fake_rule(example):
    """Return True if this example has pre-computed data for a non-production rule."""
    FAKE_MARKERS = [
        "Rule: Funnel Account",
        "Rule: Round-trip",
        "Rule: Velocity Multiple",
        '"risk_factor": "Funnel Account"',
        '"risk_factor": "Round-trip"',
        '"risk_factor": "Velocity Multiple"',
        "Funnel Account\nAxis",
        "Round-trip\nAxis",
        "Velocity Multiple\nAxis",
    ]
    text = json.dumps(example)
    return any(m in text for m in FAKE_MARKERS)


# ---------------------------------------------------------------------------
# System prompts (same as V24 for consistency)
# ---------------------------------------------------------------------------

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
    "8. When the user asks about a specific cluster (e.g. 'Cluster 4'), pass cluster as an integer "
    "to rule_sar_backtest or rule_2d_sweep.\n"
    "9. If list_rules shows alerts=0 for a requested rule, that rule has no production data. "
    "State this clearly and suggest the nearest active rule.\n"
    "10. The system contains exactly 11 active AML rules. Rules showing alerts=0 in list_rules "
    "are not deployed in production.\n"
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

# ---------------------------------------------------------------------------
# Pre-computed list_rules output (reflects production data: 11 active rules)
# ---------------------------------------------------------------------------

PC_LIST_RULES = """\
Tool result for list_rules:
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.
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
# Pre-computed rule analysis data
# ---------------------------------------------------------------------------

PC_RULE_SAR_ELDER_C4 = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Cluster filter: Cluster 4 (lowest-volume segment, 512 Business customers)
Current condition: Age >= 60 AND 14-day outgoing >= $5K AND >= 3 std dev above 90-day mean
Sweep parameter: z_threshold - Std-dev multiplier above 90-day mean (currently 3)
Current value: 3.0
Labeled population: 142 customers in Cluster 4 (TP+FN pool=21 SAR, FP+TN pool=121 non-SAR, precision=14.8%)

At the lowest value (0.00): TP=21, FP=121, FN=0, TN=0 (TP rate=100.0%, precision=14.8%).
At current condition (3.00): TP=21, FP=121, FN=0, TN=0 (TP rate=100.0%, precision=14.8%).
To keep TP rate >=90%: z_threshold <= 5.00 => TP=19, FP=84, FN=2, TN=37, precision=18.4%.
To keep TP rate >=50%: z_threshold <= 8.00 => TP=12, FP=41, FN=9, TN=80, precision=22.6%.
At the highest value (10.00): TP=7, FP=21, FN=14, TN=100 (TP rate=33.3%, precision=25.0%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_VS_C2 = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Cluster filter: Cluster 2 (30.6% of Business customers)
Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K
Sweep parameter: pair_total - Minimum combined in+out pair total to trigger (currently $20K)
Current value: 20000
Labeled population: 168 customers in Cluster 2 (TP+FN pool=28 SAR, FP+TN pool=140 non-SAR, precision=16.7%)

At the lowest value (5000.00): TP=28, FP=140, FN=0, TN=0 (TP rate=100.0%, precision=16.7%).
At current condition (20000.00): TP=28, FP=140, FN=0, TN=0 (TP rate=100.0%, precision=16.7%).
To keep TP rate >=90%: pair_total <= 35000.00 => TP=26, FP=98, FN=2, TN=42, precision=21.0%.
To keep TP rate >=50%: pair_total <= 65000.00 => TP=15, FP=44, FN=13, TN=96, precision=25.4%.
At the highest value (100000.00): TP=9, FP=18, FN=19, TN=122 (TP rate=32.1%, precision=33.3%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_2D_ACH_CHECK = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Activity Deviation (Check)
Axis 1 (floor_amount): Minimum monthly Outgoing Check sum to trigger (currently $50K)
Axis 2 (z_threshold): Std-dev multiplier above 12-month Check profile mean (currently 2)
Grid: 9 x 11 = 99 combinations
SAR pool: 41  Non-SAR pool: 271

At current condition (floor_amount=50000, z_threshold=2): TP=38, FP=205, FN=3, TN=66 (TP rate=92.7%).
Best FP reduction (TP rate >=90%): floor_amount=75000, z_threshold=3 => TP=37, FP=168, FN=4, TN=103, TP rate=90.2%, precision=18.0%.
Best FP reduction (TP rate >=50%): floor_amount=100000, z_threshold=4 => TP=22, FP=89, FN=19, TN=182, TP rate=53.7%, precision=19.8%.
(Heatmap shown in the chart below.)
=== END PRE-COMPUTED 2D SWEEP ==="""

PC_RULE_SAR_BURST_ORIG = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Burst in Originator Activity
Current condition: 5-day Incoming Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties
Sweep parameter: floor_amount - Minimum 5-day incoming Wire/ACH sum to trigger (currently $5K)
Current value: 5000
Labeled population: 623 customers (TP+FN pool=87 SAR, FP+TN pool=536 non-SAR, precision=13.6%)

At the lowest value (1000.00): TP=87, FP=536, FN=0, TN=0 (TP rate=100.0%, precision=13.6%).
At current condition (5000.00): TP=87, FP=536, FN=0, TN=0 (TP rate=100.0%, precision=13.6%).
To keep TP rate >=90%: floor_amount <= 20000.00 => TP=79, FP=382, FN=8, TN=154, precision=17.1%.
To keep TP rate >=50%: floor_amount <= 50000.00 => TP=46, FP=172, FN=41, TN=364, precision=21.1%.
At the highest value (100000.00): TP=22, FP=68, FN=65, TN=468 (TP rate=25.3%, precision=24.4%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_SAR_VS_PAIR = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Current condition: >=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K
Sweep parameter: pair_total - Minimum combined in+out pair total to trigger (currently $20K)
Current value: 20000
Labeled population: 478 customers (TP+FN pool=74 SAR, FP+TN pool=404 non-SAR, precision=15.5%)

At the lowest value (5000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).
At current condition (20000.00): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).
To keep TP rate >=90%: pair_total <= 35000.00 => TP=67, FP=274, FN=7, TN=130, precision=19.6%.
To keep TP rate >=50%: pair_total <= 65000.00 => TP=39, FP=122, FN=35, TN=282, precision=24.2%.
At the highest value (100000.00): TP=20, FP=62, FN=54, TN=342 (TP rate=27.0%, precision=24.4%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_RULE_2D_ELDER = """\
Tool result for rule_2d_sweep:
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Axis 1 (floor_amount): Minimum 14-day aggregated outgoing to trigger (currently $5K)
Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)
Grid: 9 x 9 = 81 combinations
SAR pool: 188  Non-SAR pool: 958

At current condition (floor_amount=5000, age_threshold=60): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%).
Best FP reduction (TP rate >=90%): floor_amount=8000, age_threshold=62 => TP=171, FP=712, FN=17, TN=246, TP rate=91.0%, precision=19.4%.
Best FP reduction (TP rate >=50%): floor_amount=15000, age_threshold=65 => TP=96, FP=278, FN=92, TN=680, TP rate=51.1%, precision=25.7%.
(Heatmap shown in the chart below.)
=== END PRE-COMPUTED 2D SWEEP ==="""

PC_RULE_SAR_ACH = """\
Tool result for rule_sar_backtest:
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Rule: Activity Deviation (ACH)
Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean
Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50K)
Current value: 50000
Labeled population: 487 customers (TP+FN pool=82 SAR, FP+TN pool=405 non-SAR, precision=16.8%)

At the lowest value (10000.00): TP=82, FP=405, FN=0, TN=0 (TP rate=100.0%, precision=16.8%).
At current condition (50000.00): TP=82, FP=405, FN=0, TN=0 (TP rate=100.0%, precision=16.8%).
To keep TP rate >=90%: floor_amount <= 75000.00 => TP=74, FP=288, FN=8, TN=117, precision=20.4%.
To keep TP rate >=50%: floor_amount <= 150000.00 => TP=44, FP=118, FN=38, TN=287, precision=27.2%.
At the highest value (200000.00): TP=28, FP=62, FN=54, TN=343 (TP rate=34.1%, precision=31.1%).
(Detailed sweep table shown in the chart below.)
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_POLICY_KB_ALERT_VOL = """\
Tool result for search_policy_kb:
=== RETRIEVED POLICY CONTEXT ===
Source: FFIEC BSA/AML Examination Manual (Section: Transaction Monitoring / Alert Management)

Alert volume and investigation capacity management are addressed as risk-based program requirements. \
Key guidance includes:

1. Alert Thresholds & Tuning: Banks must document the rationale for alert thresholds and review them \
at least annually. FFIEC examiners expect banks to demonstrate that thresholds balance SAR catch rates \
against investigator capacity — excessively high false positive rates indicate weak controls.

2. Reasonable Investigation Timelines: FinCEN guidance (FIN-2012-G002) requires that once a decision \
is made to file a SAR, filing must occur within 30 calendar days. Suspicious activity investigations \
should be completed within 60 days of initial detection.

3. Alert Volume as a Risk Indicator: Examiners flag both extremely low and extremely high alert volumes \
as potential red flags — low volumes may indicate undertriggering; excessive FP volumes signal poor \
calibration and can mask real risk by overwhelming investigators.

4. Wolfsberg Group (2016 Guidance on RBA): Recommends segmenting the customer base to calibrate \
thresholds by risk tier, ensuring high-risk segments receive tighter thresholds regardless of volume.

Source: FinCEN SAR Activity Review (various editions), FFIEC BSA/AML Examination Manual 2014
=== END RETRIEVED CONTEXT ==="""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

examples = []

# ── F. Fake-rule graceful declines ────────────────────────────────────────────

# F1: Funnel Account SAR backtest
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Funnel Account"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("f1", "list_rules", {})
    ]},
    {"role": "tool", "tool_call_id": "f1", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "The **Funnel Account** rule shows **alerts=0** in the current monitoring system — "
        "this rule has no production alert data and is not deployed.\n\n"
        "The 11 active rules with data are: Activity Deviation (ACH), Activity Deviation (Check), "
        "Elder Abuse, Velocity Single, Detect Excessive Transaction Activity, "
        "Structuring (Incoming Cash), Structuring (Outgoing Cash), CTR Client, "
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer.\n\n"
        "Did you mean **Burst in Originator Activity**, which monitors inflow concentration from multiple "
        "counterparties (87 SARs, 536 FPs, precision=13.6%)?"
    )},
]})

# F2: Round-trip 2D sweep
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Round-trip varying floor amount and return window"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("f2", "list_rules", {})
    ]},
    {"role": "tool", "tool_call_id": "f2", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "The **Round-trip** rule shows **alerts=0** — it has no production alert data and is not currently deployed.\n\n"
        "The 11 active rules available for 2D sweep analysis are: Activity Deviation (ACH), "
        "Activity Deviation (Check), Elder Abuse, Velocity Single, Detect Excessive Transaction Activity, "
        "Structuring (Incoming Cash), Structuring (Outgoing Cash), CTR Client, "
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer.\n\n"
        "For wire-transfer structuring analysis, consider **Risky International Transfer** "
        "(21 SARs, precision=36.2%) or **Velocity Single** (74 SARs, 404 FPs)."
    )},
]})

# F3: Velocity Multiple SAR backtest
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Velocity Multiple"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("f3", "list_rules", {})
    ]},
    {"role": "tool", "tool_call_id": "f3", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "The **Velocity Multiple** rule shows **alerts=0** — it has no production alert data "
        "and is not currently deployed.\n\n"
        "The closest active rule is **Velocity Single**, which monitors in/out pairs within "
        "14 days (74 SARs, 404 FPs, precision=15.5%). "
        "Would you like a SAR backtest for Velocity Single instead?"
    )},
]})

# F4: Human Trafficking Indicators SAR backtest
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What is the SAR filing rate for Human Trafficking Indicators?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("f4", "list_rules", {})
    ]},
    {"role": "tool", "tool_call_id": "f4", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        "The **Human Trafficking Indicators** rule shows **alerts=0** in the current monitoring system — "
        "this rule has no production alert data.\n\n"
        "The 11 active rules with SAR data are: Activity Deviation (ACH/Check), Elder Abuse, "
        "Velocity Single, Detect Excessive Transaction Activity, Structuring (Incoming/Outgoing Cash), "
        "CTR Client, Burst in Originator Activity, Burst in Beneficiary Activity, "
        "and Risky International Transfer.\n\n"
        "For cash-pattern rules with similar typology focus, see **Elder Abuse** (188 SARs, 16.4% precision) "
        "or **CTR Client** (180 SARs, 8.0% precision)."
    )},
]})

# ── C. Cluster-filtered rule analysis ─────────────────────────────────────────

# C1: Elder Abuse SAR backtest for Cluster 4
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show Elder Abuse SAR backtest for Cluster 4"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("c1", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "cluster": 4})
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": PC_RULE_SAR_ELDER_C4},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse | Cluster filter: Cluster 4\n"
        "Labeled population: 142 customers (SAR=21, FP=121, precision=14.8%)\n\n"
        "At current condition (z_threshold=3.00): TP=21, FP=121, FN=0, TN=0 (TP rate=100.0%).\n"
        "To keep TP rate >=90%: z_threshold <= 5.00 => TP=19, FP=84, FN=2, TN=37, precision=18.4%.\n"
        "To keep TP rate >=50%: z_threshold <= 8.00 => TP=12, FP=41, FN=9, TN=80, precision=22.6%.\n"
        "At the highest value (10.00): TP=7, FP=21, FN=14, TN=100 (TP rate=33.3%, precision=25.0%).\n\n"
        "Raising the z_threshold to 5 in Cluster 4 cuts FPs by 31% (121 to 84) "
        "while retaining 90% of the 21 SAR customers."
    )},
]})

# C2: Velocity Single SAR backtest for Cluster 2
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run SAR backtest for Velocity Single in Cluster 2"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("c2", "rule_sar_backtest", {"risk_factor": "Velocity Single", "cluster": 2})
    ]},
    {"role": "tool", "tool_call_id": "c2", "content": PC_RULE_SAR_VS_C2},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Velocity Single | Cluster filter: Cluster 2\n"
        "Labeled population: 168 customers (SAR=28, FP=140, precision=16.7%)\n\n"
        "At current condition (pair_total=20000): TP=28, FP=140, FN=0, TN=0 (TP rate=100.0%).\n"
        "To keep TP rate >=90%: pair_total <= 35000 => TP=26, FP=98, FN=2, TN=42, precision=21.0%.\n"
        "To keep TP rate >=50%: pair_total <= 65000 => TP=15, FP=44, FN=13, TN=96, precision=25.4%.\n"
        "At the highest value (100000): TP=9, FP=18, FN=19, TN=122 (TP rate=32.1%, precision=33.3%).\n\n"
        "In Cluster 2, raising the pair_total to $35K retains 92.9% of SARs "
        "while reducing FPs by 30% (140 to 98)."
    )},
]})

# ── R. Real rule coverage ──────────────────────────────────────────────────────

# R1: Activity Deviation (Check) 2D sweep
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show 2D sweep for Activity Deviation Check varying floor amount and sigma"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("r1", "rule_2d_sweep", {
            "risk_factor": "Activity Deviation (Check)",
            "sweep_param_1": "floor_amount",
            "sweep_param_2": "z_threshold",
        })
    ]},
    {"role": "tool", "tool_call_id": "r1", "content": PC_RULE_2D_ACH_CHECK},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (Check)\n"
        "Axis 1 (floor_amount): Minimum monthly Outgoing Check sum to trigger (currently $50K)\n"
        "Axis 2 (z_threshold): Std-dev multiplier above 12-month Check profile mean (currently 2)\n"
        "Grid: 9 x 11 = 99 combinations | SAR pool: 41 | Non-SAR pool: 271\n\n"
        "At current (floor_amount=50000, z_threshold=2): TP=38, FP=205, FN=3, TN=66 (TP rate=92.7%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=75000, z_threshold=3 => "
        "TP=37, FP=168, FN=4, TN=103, precision=18.0%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=100000, z_threshold=4 => "
        "TP=22, FP=89, FN=19, TN=182, precision=19.8%.\n\n"
        "Raising the floor to $75K and the z-threshold to 3 reduces FPs by 18% (205 to 168) "
        "while keeping 90.2% of the 41 SAR customers."
    )},
]})

# R2: Burst in Originator Activity SAR backtest
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Burst in Originator Activity"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("r2", "rule_sar_backtest", {"risk_factor": "Burst in Originator Activity"})
    ]},
    {"role": "tool", "tool_call_id": "r2", "content": PC_RULE_SAR_BURST_ORIG},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Burst in Originator Activity\n"
        "Current condition: 5-day Incoming Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties\n"
        "Sweep parameter: floor_amount (currently $5K)\n"
        "Labeled population: 623 customers (SAR=87, FP=536, precision=13.6%)\n\n"
        "At current condition (5000): TP=87, FP=536, FN=0, TN=0 (TP rate=100.0%, precision=13.6%).\n"
        "To keep TP rate >=90%: floor_amount <= 20000 => TP=79, FP=382, FN=8, TN=154, precision=17.1%.\n"
        "To keep TP rate >=50%: floor_amount <= 50000 => TP=46, FP=172, FN=41, TN=364, precision=21.1%.\n"
        "At highest value (100000): TP=22, FP=68, FN=65, TN=468 (TP rate=25.3%, precision=24.4%).\n\n"
        "Raising the floor to $20K cuts FPs by 29% (536 to 382) while retaining 90.8% of the 87 SAR customers."
    )},
]})

# ── P. Policy KB call ──────────────────────────────────────────────────────────

# P1: Alert volume thresholds — must call KB, not answer from memory
examples.append({"messages": [
    {"role": "system", "content": POLICY_SYSTEM},
    {"role": "user", "content": "What are the alert volume thresholds under BSA/AML policy?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("p1", "search_policy_kb", {"query": "alert volume thresholds investigation capacity AML"})
    ]},
    {"role": "tool", "tool_call_id": "p1", "content": PC_POLICY_KB_ALERT_VOL},
    {"role": "assistant", "content": (
        "BSA/AML policy does not prescribe specific numeric alert volume thresholds — "
        "instead, it establishes a **risk-based framework** for managing alert volumes:\n\n"
        "1. **Threshold documentation**: Banks must document the rationale for alert thresholds "
        "and review them at least annually (FFIEC BSA/AML Examination Manual).\n\n"
        "2. **SAR filing timeline**: Once a filing decision is made, SARs must be filed within "
        "**30 calendar days** (FinCEN FIN-2012-G002). Investigations should close within 60 days.\n\n"
        "3. **FP rate as a control indicator**: Examiners flag both extremely low and extremely high "
        "alert volumes — very high FP rates indicate poor calibration and may mask real risk.\n\n"
        "4. **Risk-based segmentation**: The Wolfsberg Group (2016) recommends segmenting the "
        "customer base to calibrate thresholds by risk tier, so high-risk segments receive tighter thresholds.\n\n"
        "In practice, banks typically target **SAR catch rates above 80%** and **precision above 15-20%** "
        "as internal benchmarks, though these are not mandated figures."
    )},
]})

# ── Q. Clean synthesis fixes ───────────────────────────────────────────────────

# Q1: Velocity Single pair_total sweep — clean response, no "REETING" artifact
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What happens to FP if I lower the pair total for Velocity Single?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("q1", "rule_sar_backtest", {
            "risk_factor": "Velocity Single",
            "sweep_param": "pair_total",
        })
    ]},
    {"role": "tool", "tool_call_id": "q1", "content": PC_RULE_SAR_VS_PAIR},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Velocity Single | Sweep parameter: pair_total (currently $20K)\n"
        "Labeled population: 478 customers (SAR=74, FP=404, precision=15.5%)\n\n"
        "At current condition (20000): TP=74, FP=404, FN=0, TN=0 (TP rate=100.0%, precision=15.5%).\n"
        "To keep TP rate >=90%: pair_total <= 35000 => TP=67, FP=274, FN=7, TN=130, precision=19.6%.\n"
        "To keep TP rate >=50%: pair_total <= 65000 => TP=39, FP=122, FN=35, TN=282, precision=24.2%.\n"
        "At highest value (100000): TP=20, FP=62, FN=54, TN=342 (TP rate=27.0%, precision=24.4%).\n\n"
        "Lowering pair_total below the current $20K will not reduce FPs — "
        "it will only add more FPs. Raising pair_total to $35K reduces FPs by 32% (404 to 274) "
        "while retaining 90.5% of the 74 SAR customers."
    )},
]})

# Q2: Elder Abuse 2D sweep — clean response, no "per Abuse" artifact
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Run a 2D sweep for Elder Abuse varying floor amount and age threshold"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("q2", "rule_2d_sweep", {
            "risk_factor": "Elder Abuse",
            "sweep_param_1": "floor_amount",
            "sweep_param_2": "age_threshold",
        })
    ]},
    {"role": "tool", "tool_call_id": "q2", "content": PC_RULE_2D_ELDER},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Elder Abuse\n"
        "Axis 1 (floor_amount): Minimum 14-day aggregated outgoing to trigger (currently $5K)\n"
        "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60)\n"
        "Grid: 9 x 9 = 81 combinations | SAR pool: 188 | Non-SAR pool: 958\n\n"
        "At current (floor_amount=5000, age_threshold=60): TP=188, FP=958, FN=0, TN=0 (TP rate=100.0%).\n"
        "Best FP reduction (TP rate >=90%): floor_amount=8000, age_threshold=62 => "
        "TP=171, FP=712, FN=17, TN=246, TP rate=91.0%, precision=19.4%.\n"
        "Best FP reduction (TP rate >=50%): floor_amount=15000, age_threshold=65 => "
        "TP=96, FP=278, FN=92, TN=680, TP rate=51.1%, precision=25.7%.\n\n"
        "Raising both the floor to $8K and the age cutoff to 62 achieves 91% SAR retention "
        "while reducing false positives by 26% (958 to 712)."
    )},
]})

# Q3: Activity Deviation ACH SAR backtest — clean response, no "RATE" prefix artifact
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("q3", "rule_sar_backtest", {
            "risk_factor": "Activity Deviation (ACH)",
            "sweep_param": "floor_amount",
        })
    ]},
    {"role": "tool", "tool_call_id": "q3", "content": PC_RULE_SAR_ACH},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
        "Rule: Activity Deviation (ACH)\n"
        "Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean\n"
        "Sweep parameter: floor_amount (currently $50K)\n"
        "Labeled population: 487 customers (SAR=82, FP=405, precision=16.8%)\n\n"
        "At current condition (50000): TP=82, FP=405, FN=0, TN=0 (TP rate=100.0%, precision=16.8%).\n"
        "To keep TP rate >=90%: floor_amount <= 75000 => TP=74, FP=288, FN=8, TN=117, precision=20.4%.\n"
        "To keep TP rate >=50%: floor_amount <= 150000 => TP=44, FP=118, FN=38, TN=287, precision=27.2%.\n"
        "At highest value (200000): TP=28, FP=62, FN=54, TN=343 (TP rate=34.1%, precision=31.1%).\n\n"
        "Raising the ACH floor to $75K retains 90.2% of the 82 SAR customers "
        "while reducing false positives by 29% (405 to 288)."
    )},
]})

# ---------------------------------------------------------------------------
# Combine with V24, filter fake examples, write V25
# ---------------------------------------------------------------------------

def main():
    # Load V24 combined
    v24_examples = []
    fake_count   = 0
    kept_count   = 0

    with open(V24_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if _contains_fake_rule(ex):
                fake_count += 1
                print(f"[V25] Filtered fake-rule example #{fake_count}")
            else:
                v24_examples.append(ex)
                kept_count += 1

    print(f"[V25] Kept {kept_count} / {kept_count + fake_count} V24 examples ({fake_count} fake-rule removed)")
    print(f"[V25] Adding {len(examples)} new V25 examples")

    all_examples = v24_examples + examples
    print(f"[V25] Total: {len(all_examples)} examples -> {V25_PATH.name}")

    # Write V25-only JSONL
    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V25] V25-only: {THIS_PATH.name} ({len(examples)} examples)")

    # Write combined V25 JSONL
    with open(V25_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V25] Combined written: {V25_PATH.name}")


if __name__ == "__main__":
    main()

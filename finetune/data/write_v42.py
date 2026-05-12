"""
V42 training examples (2026-05-12).

Targets:

  AA1–AA5  Precision-ranking queries — updated to use actual aria_alerts.csv numbers
           AND include ### header line (new Rule 9a).

  AF1–AF2  Precision-ranking — two MORE rule list datasets where a DIFFERENT rule
           wins precision so the model learns to READ from context, not memorise.

  AD1–AD3  Elder Abuse 2D sweep — three datasets with different optimal points.
           Teaches the model the header + verbatim-copy + insight pattern with
           diverse data so it cannot memorise a single answer.

  AE1–AE2  Rule SAR backtest — two different rules / params with headers.

  AG1–AG2  SAR catch rate (segment-level) — updated with ### header.

  AB1–AB6  Classification misroutes — conceptual queries → policy; OOS social.

  AC1–AC5  Multi-turn cluster comparison — verbatim stats, no invented fields.

Combined: aria_train_combined_v41_full.jsonl (base) + V42 new = aria_train_combined_v42_full.jsonl
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V41_BASE_PATH = DATA_DIR / "aria_train_combined_v41_full.jsonl"
V42_ONLY_PATH = DATA_DIR / "aria_train_v42.jsonl"
V42_FULL_PATH = DATA_DIR / "aria_train_combined_v42_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

# CLASSIFY_SYSTEM sourced from live orchestrator — training and runtime always in sync
from agents.orchestrator import _CLASSIFY_SYSTEM as CLASSIFY_SYSTEM  # noqa: E402
from write_v41 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM           # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}

def prev_context(stats):
    return f"[PREVIOUS CLUSTERING RESULT]\n{stats}\n[END PREVIOUS RESULT]"

examples = []

# ═══════════════════════════════════════════════════════════════════════════
# RULE LIST — three diverse datasets
# Rule 9a: ### header BEFORE the PRE-COMPUTED block
# ═══════════════════════════════════════════════════════════════════════════

# ── Dataset A: actual aria_alerts.csv numbers (Round-trip wins at 35.0%) ──

_LIST_A = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=300, SAR=46, FP=254, precision=15.3%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=200, SAR=26, FP=174, precision=13.0%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=400, SAR=59, FP=341, precision=14.8%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=250, SAR=49, FP=201, precision=19.6%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=350, SAR=49, FP=301, precision=14.0%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=300, SAR=46, FP=254, precision=15.3%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=250, SAR=37, FP=213, precision=14.8%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=400, SAR=52, FP=348, precision=13.0%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=200, SAR=34, FP=166, precision=17.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=200, SAR=33, FP=167, precision=16.5%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=150, SAR=40, FP=110, precision=26.7%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=150, SAR=25, FP=125, precision=16.7%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=200, SAR=35, FP=165, precision=17.5%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=150, SAR=39, FP=111, precision=26.0%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=100, SAR=35, FP=65, precision=35.0%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=100, SAR=23, FP=77, precision=23.0%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_A = "Tool result for list_rules:\n" + _LIST_A

# ── Dataset B: Elder Abuse wins precision (28.7%) ──

_LIST_B = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=520, SAR=98, FP=422, precision=18.8%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=190, SAR=33, FP=157, precision=17.4%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=310, SAR=89, FP=221, precision=28.7%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=290, SAR=56, FP=234, precision=19.3%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=420, SAR=71, FP=349, precision=16.9%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=280, SAR=48, FP=232, precision=17.1%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=200, SAR=36, FP=164, precision=18.0%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=450, SAR=62, FP=388, precision=13.8%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=210, SAR=40, FP=170, precision=19.0%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=220, SAR=38, FP=182, precision=17.3%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=160, SAR=35, FP=125, precision=21.9%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=130, SAR=21, FP=109, precision=16.2%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=180, SAR=32, FP=148, precision=17.8%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=140, SAR=27, FP=113, precision=19.3%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=90, SAR=18, FP=72, precision=20.0%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=80, SAR=14, FP=66, precision=17.5%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_B = "Tool result for list_rules:\n" + _LIST_B

# ── Dataset C: Velocity Single wins precision (41.2%) ──

_LIST_C = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR/FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=480, SAR=84, FP=396, precision=17.5%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=220, SAR=41, FP=179, precision=18.6%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=380, SAR=65, FP=315, precision=17.1%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=260, SAR=107, FP=153, precision=41.2%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=400, SAR=62, FP=338, precision=15.5%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=270, SAR=43, FP=227, precision=15.9%, sweep_params=[daily_floor, days_required]\n"
    "  Structuring (Outgoing Cash): alerts=230, SAR=39, FP=191, precision=16.9%, sweep_params=[daily_floor, days_required]\n"
    "  CTR Client: alerts=430, SAR=58, FP=372, precision=13.5%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=195, SAR=32, FP=163, precision=16.4%, sweep_params=[floor_amount, min_transactions]\n"
    "  Burst in Beneficiary Activity: alerts=205, SAR=31, FP=174, precision=15.1%, sweep_params=[floor_amount, min_transactions]\n"
    "  Risky International Transfer: alerts=140, SAR=33, FP=107, precision=23.6%, sweep_params=[floor_amount]\n"
    "  Activity Deviation (Wire): alerts=145, SAR=24, FP=121, precision=16.6%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=190, SAR=35, FP=155, precision=18.4%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=135, SAR=28, FP=107, precision=20.7%, sweep_params=[floor_amount, min_counterparties]\n"
    "  Round-trip: alerts=95, SAR=22, FP=73, precision=23.2%, sweep_params=[floor_amount, return_window]\n"
    "  Human Trafficking Indicators: alerts=85, SAR=18, FP=67, precision=21.2%, sweep_params=[floor_amount, days_required]\n"
    "=== END RULE LIST ==="
)
PC_LIST_C = "Tool result for list_rules:\n" + _LIST_C

# ═══════════════════════════════════════════════════════════════════════════
# AA1–AA5  Rule list / precision queries — Dataset A, with ### header
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What are the rules with the highest precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("aa1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa1", "content": PC_LIST_A},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_A + "\n\n"
        "Round-trip has the highest precision at 35.0% (35 SAR, 65 FP), "
        "followed by Risky International Transfer (26.7%) and Funnel Account (26.0%)."
    )},
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the best precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("aa2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa2", "content": PC_LIST_A},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_A + "\n\n"
        "Round-trip has the best precision at 35.0% — "
        "35 of its 100 alerts are confirmed SAR customers."
    )},
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Rank all AML rules by precision from highest to lowest."},
    {"role": "assistant", "content": None, "tool_calls": [tc("aa3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa3", "content": PC_LIST_A},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_A + "\n\n"
        "Ranked by precision descending: Round-trip (35.0%) > Risky International Transfer (26.7%) > "
        "Funnel Account (26.0%) > Human Trafficking Indicators (23.0%) > Velocity Single (19.6%) > "
        "Velocity Multiple (17.5%) > Burst in Originator Activity (17.0%) > Activity Deviation Wire (16.7%) > "
        "Burst in Beneficiary Activity (16.5%) > Structuring Incoming Cash / AD ACH (15.3%) > "
        "Elder Abuse / Structuring Outgoing Cash (14.8%) > DETA (14.0%) > CTR Client / AD Check (13.0%)."
    )},
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which AML rule wastes the fewest investigator hours?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("aa4", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa4", "content": PC_LIST_A},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_A + "\n\n"
        "Round-trip wastes the fewest investigator hours at 35.0% precision — "
        "65 FP out of 100 alerts, versus 348 FP for CTR Client (13.0%)."
    )},
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rules have the lowest precision — worst SAR-to-FP ratio?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("aa5", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "aa5", "content": PC_LIST_A},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_A + "\n\n"
        "CTR Client and Activity Deviation (Check) have the lowest precision at 13.0% each — "
        "both are the highest-priority candidates for threshold tightening."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AF1–AF2  Rule list — Datasets B and C (different precision winners)
# Teaches model to READ precision= field from tool output, not memorise winner
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Which rule has the highest precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("af1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "af1", "content": PC_LIST_B},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_B + "\n\n"
        "Elder Abuse has the highest precision at 28.7% (89 SAR, 221 FP) — "
        "more than twice the precision of CTR Client (13.8%), the lowest in this dataset."
    )},
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "What are the top rules for precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("af2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "af2", "content": PC_LIST_C},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        + _LIST_C + "\n\n"
        "Velocity Single has the highest precision at 41.2% (107 SAR, 153 FP), "
        "followed by Risky International Transfer (23.6%) and Round-trip (23.2%)."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AD1–AD3  Elder Abuse 2D sweep — three diverse datasets
# AD1: optimal at floor=$8K / age=66  (original training data)
# AD2: optimal at floor=$9K / age=68  (dataset B — higher threshold)
# AD3: optimal at floor=$10K / age=70 (dataset C — highest threshold)
# ═══════════════════════════════════════════════════════════════════════════

_2D_ELDER_A = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Axis 1 (floor_amount): Minimum 14-day outgoing floor to trigger (currently $5K)\n"
    "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60 years)\n"
    "Grid: 9 x 15 = 135 combinations\n"
    "SAR pool: 59  Non-SAR pool: 341\n\n"
    "At current condition (floor_amount=5000, age_threshold=60): TP=59, FP=341, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=90%): floor_amount=6000, age_threshold=62 => TP=54, FP=261, FN=5, TN=80, TP rate=91.5%, precision=17.1%.\n"
    "Best FP reduction (TP rate >=50%): floor_amount=8000, age_threshold=66 => TP=30, FP=163, FN=29, TN=178, TP rate=50.8%, precision=15.5%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END PRE-COMPUTED 2D SWEEP ==="
)
PC_2D_ELDER_A = "Tool result for rule_2d_sweep:\n" + _2D_ELDER_A

_2D_ELDER_B = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Axis 1 (floor_amount): Minimum 14-day outgoing floor to trigger (currently $5K)\n"
    "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60 years)\n"
    "Grid: 9 x 15 = 135 combinations\n"
    "SAR pool: 89  Non-SAR pool: 221\n\n"
    "At current condition (floor_amount=5000, age_threshold=60): TP=89, FP=221, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=90%): floor_amount=6000, age_threshold=62 => TP=81, FP=167, FN=8, TN=54, TP rate=91.0%, precision=32.7%.\n"
    "Best FP reduction (TP rate >=50%): floor_amount=9000, age_threshold=68 => TP=45, FP=89, FN=44, TN=132, TP rate=50.6%, precision=33.6%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END PRE-COMPUTED 2D SWEEP ==="
)
PC_2D_ELDER_B = "Tool result for rule_2d_sweep:\n" + _2D_ELDER_B

_2D_ELDER_C = (
    "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Axis 1 (floor_amount): Minimum 14-day outgoing floor to trigger (currently $5K)\n"
    "Axis 2 (age_threshold): Minimum customer age to trigger (currently 60 years)\n"
    "Grid: 9 x 15 = 135 combinations\n"
    "SAR pool: 59  Non-SAR pool: 341\n\n"
    "At current condition (floor_amount=5000, age_threshold=60): TP=59, FP=341, FN=0, TN=0 (TP rate=100.0%).\n"
    "Best FP reduction (TP rate >=90%): floor_amount=7000, age_threshold=63 => TP=54, FP=248, FN=5, TN=93, TP rate=91.5%, precision=17.9%.\n"
    "Best FP reduction (TP rate >=50%): floor_amount=10000, age_threshold=70 => TP=30, FP=121, FN=29, TN=220, TP rate=50.8%, precision=19.9%.\n"
    "(Heatmap shown in the chart below.)\n"
    "=== END PRE-COMPUTED 2D SWEEP ==="
)
PC_2D_ELDER_C = "Tool result for rule_2d_sweep:\n" + _2D_ELDER_C

# AD1 — Dataset A (floor=$8K, age=66)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run a 2D sweep for Elder Abuse varying floor amount and age threshold"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ad1", "rule_2d_sweep", {"risk_factor": "Elder Abuse",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "age_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "ad1", "content": PC_2D_ELDER_A},
    {"role": "assistant", "content": (
        "### 2D Sweep — Elder Abuse | Floor Amount × Age Threshold\n\n"
        + _2D_ELDER_A + "\n\n"
        "Maintaining a 90% SAR catch rate requires holding the floor at $6,000 with age ≥ 62 — "
        "tightening to the 50% threshold at $8,000 / age 66 cuts FP by 52% but misses 29 SARs."
    )},
]})

# AD2 — Dataset B (floor=$9K, age=68)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show the 2D grid for Elder Abuse with floor amount and age threshold"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ad2", "rule_2d_sweep", {"risk_factor": "Elder Abuse",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "age_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "ad2", "content": PC_2D_ELDER_B},
    {"role": "assistant", "content": (
        "### 2D Sweep — Elder Abuse | Floor Amount × Age Threshold\n\n"
        + _2D_ELDER_B + "\n\n"
        "The 90% SAR threshold at floor_amount $6,000 / age 62 already achieves 32.7% precision — "
        "raising to $9,000 / age 68 keeps 50.6% of SARs while cutting FP to just 89."
    )},
]})

# AD3 — Dataset C (floor=$10K, age=70)
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "2D analysis for Elder Abuse"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ad3", "rule_2d_sweep", {"risk_factor": "Elder Abuse",
                                    "sweep_param_1": "floor_amount",
                                    "sweep_param_2": "age_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "ad3", "content": PC_2D_ELDER_C},
    {"role": "assistant", "content": (
        "### 2D Sweep — Elder Abuse | Floor Amount × Age Threshold\n\n"
        + _2D_ELDER_C + "\n\n"
        "The 90% SAR operating point at $7,000 / age 63 reduces FP from 341 to 248 — "
        "tightening further to $10,000 / age 70 keeps 50.8% SAR catch at 19.9% precision."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AE1–AE2  Rule SAR backtest — two rules, with ### header
# ═══════════════════════════════════════════════════════════════════════════

_SAR_ELDER_AGE = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Elder Abuse\n"
    "Current condition: Monthly outgoing >= floor_amount AND customer age >= age_threshold (currently 60)\n"
    "Sweep parameter: age_threshold - Minimum customer age to trigger rule (currently 60)\n"
    "Current value: 60\n"
    "Labeled population: 310 customers (TP+FN pool=89 SAR, FP+TN pool=221 non-SAR, precision=28.7%)\n\n"
    "At the lowest value (55): TP=89, FP=221, FN=0, TN=0 (TP rate=100.0%, precision=28.7%).\n"
    "At current condition (60): TP=89, FP=221, FN=0, TN=0 (TP rate=100.0%, precision=28.7%).\n"
    "To keep TP rate >=90%: age_threshold <= 63 => TP=81, FP=167, FN=8, TN=54, precision=32.7%.\n"
    "To keep TP rate >=50%: age_threshold <= 68 => TP=45, FP=89, FN=44, TN=132, precision=33.6%.\n"
    "At the highest value (75): TP=12, FP=18, FN=77, TN=203 (TP rate=13.5%, precision=40.0%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_ELDER_AGE = "Tool result for rule_sar_backtest:\n" + _SAR_ELDER_AGE

_SAR_VS_PAIR = (
    "=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===\n"
    "Rule: Velocity Single\n"
    "Current condition: Combined in+out pair total >= $20K AND ratio deviation <= 10%\n"
    "Sweep parameter: pair_total - Minimum combined in+out pair total (currently $20K)\n"
    "Current value: 20000\n"
    "Labeled population: 260 customers (TP+FN pool=107 SAR, FP+TN pool=153 non-SAR, precision=41.2%)\n\n"
    "At the lowest value (5000): TP=107, FP=153, FN=0, TN=0 (TP rate=100.0%, precision=41.2%).\n"
    "At current condition (20000): TP=107, FP=153, FN=0, TN=0 (TP rate=100.0%, precision=41.2%).\n"
    "To keep TP rate >=90%: pair_total <= 38000 => TP=97, FP=103, FN=10, TN=50, precision=48.5%.\n"
    "To keep TP rate >=50%: pair_total <= 72000 => TP=55, FP=41, FN=52, TN=112, precision=57.3%.\n"
    "At the highest value (95000): TP=31, FP=18, FN=76, TN=135 (TP rate=29.0%, precision=63.3%).\n"
    "(Detailed sweep table shown in the chart below.)\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_VS_PAIR = "Tool result for rule_sar_backtest:\n" + _SAR_VS_PAIR

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show SAR backtest for Elder Abuse sweeping age threshold"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ae1", "rule_sar_backtest", {"risk_factor": "Elder Abuse", "sweep_param": "age_threshold"})
    ]},
    {"role": "tool", "tool_call_id": "ae1", "content": PC_SAR_ELDER_AGE},
    {"role": "assistant", "content": (
        "### SAR Backtest — Elder Abuse | age_threshold\n\n"
        + _SAR_ELDER_AGE + "\n\n"
        "Raising the age threshold to 63 keeps 90% SAR catch rate while cutting FP from 221 to 167 "
        "and lifting precision from 28.7% to 32.7%."
    )},
]})

examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "SAR backtest for Velocity Single by pair total"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ae2", "rule_sar_backtest", {"risk_factor": "Velocity Single", "sweep_param": "pair_total"})
    ]},
    {"role": "tool", "tool_call_id": "ae2", "content": PC_SAR_VS_PAIR},
    {"role": "assistant", "content": (
        "### SAR Backtest — Velocity Single | pair_total\n\n"
        + _SAR_VS_PAIR + "\n\n"
        "Raising pair_total to $38,000 keeps 90% of SARs while cutting FP from 153 to 103 "
        "and pushing precision from 41.2% to 48.5%."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AG1–AG2  SAR catch rate (segment-level) — updated with ### header
# ═══════════════════════════════════════════════════════════════════════════

# Reuse the SAR backtest bodies from write_v41 (imported indirectly via the
# combined base JSONL). We just need NEW examples that include the header so
# the model learns the pattern for this tool type too.

_SAR_BT_BIZ_WK = (
    "=== PRE-COMPUTED SAR BACKTEST ===\n"
    "### SAR Catch Rate — Business / AVG_TRXNS_WEEK\n\n"
    "**Population:** 1,384 alerted customers | **SARs:** 312 (22.5% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold (0.54): **312 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **1.54** (1 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **2.62** (281 of 312 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **3.62** (250 of 312 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **5.62** (157 of 312 caught)\n"
    "- At highest threshold (15.54): **11 caught**, 301 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_BT_BIZ_WK = "Tool result for sar_backtest:\n" + _SAR_BT_BIZ_WK

_SAR_BT_IND_AMT2 = (
    "=== PRE-COMPUTED SAR BACKTEST ===\n"
    "### SAR Catch Rate — Individual / TRXN_AMT_MONTHLY\n\n"
    "**Population:** 4,515 alerted customers | **SARs:** 839 (18.6% SAR filing rate)\n\n"
    "**Sweep Results**\n"
    "- At lowest threshold ($120.0): **839 SARs caught** (100%), 0 missed\n"
    "- SARs first missed at threshold **$620.0** (3 missed)\n"
    "- To keep ≥90% SAR catch rate: threshold ≤ **$120.0** (755 of 839 caught)\n"
    "- To keep ≥80% SAR catch rate: threshold ≤ **$2,120.0** (671 of 839 caught)\n"
    "- To keep ≥50% SAR catch rate: threshold ≤ **$8,120.0** (420 of 839 caught)\n"
    "- At highest threshold ($32,120.0): **9 caught**, 830 missed\n\n"
    "*(Detailed sweep chart shown below.)*\n"
    "=== END PRE-COMPUTED SAR BACKTEST ==="
)
PC_SAR_BT_IND_AMT2 = "Tool result for sar_backtest:\n" + _SAR_BT_IND_AMT2

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "SAR catch rate for Business customers by weekly transactions"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ag1", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})
    ]},
    {"role": "tool", "tool_call_id": "ag1", "content": PC_SAR_BT_BIZ_WK},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Business | Avg Weekly Transactions\n\n"
        + _SAR_BT_BIZ_WK + "\n\n"
        "Business SAR customers concentrate below 2.62 weekly transactions — "
        "holding that threshold preserves 90% SAR catch rate for this segment."
    )},
]})

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "What is the SAR filing rate for Individual customers by monthly amount?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ag2", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})
    ]},
    {"role": "tool", "tool_call_id": "ag2", "content": PC_SAR_BT_IND_AMT2},
    {"role": "assistant", "content": (
        "### SAR Catch Rate — Individual | Monthly Transaction Volume\n\n"
        + _SAR_BT_IND_AMT2 + "\n\n"
        "Monthly transaction amount is a weak separator for Individual customers — "
        "keeping ≥90% SAR catch requires holding the threshold at just $120.0, "
        "meaning most SAR customers transact at very low monthly volumes."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# AB1–AB6  Classification misroutes → correct labels
# ═══════════════════════════════════════════════════════════════════════════

for query, label in [
    ("How does ARIA help with SAR backtesting",                        "policy"),
    ("What is SAR backtesting",                                        "policy"),
    ("How does ARIA help with threshold tuning",                       "policy"),
    ("What is OFAC",                                                   "policy"),
    ("What are the main differences between US and Canada in AML regulations", "policy"),
    ("My Dog OFAC met a cat the other day",                            "out_of_scope"),
]:
    examples.append({"messages": [
        {"role": "system",    "content": CLASSIFY_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": label},
    ]})

# ═══════════════════════════════════════════════════════════════════════════
# AC1–AC5  Multi-turn cluster comparison — verbatim stats, no invented fields
# ═══════════════════════════════════════════════════════════════════════════

_CLUSTER_STATS = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | 4 clusters

Cluster 1 (n=121):
- Avg Weekly Transactions: **3.2**
- Avg Weekly Txn Amount: **$1,205**
- Monthly Txn Volume: **$4,820**
- Income: **$72,000**
- Current Balance: **$18,500**
- Account Age (years): **4.1**

Cluster 2 (n=98):
- Avg Weekly Transactions: **8.7**
- Avg Weekly Txn Amount: **$4,310**
- Monthly Txn Volume: **$17,240**
- Income: **$145,000**
- Current Balance: **$62,300**
- Account Age (years): **7.3**

Cluster 3 (n=156):
- Avg Weekly Transactions: **14.2**
- Avg Weekly Txn Amount: **$2,914**
- Monthly Txn Volume: **$41,380**
- Income: **$98,000**
- Current Balance: **$31,100**
- Account Age (years): **5.8**

Cluster 4 (n=110):
- Avg Weekly Transactions: **22.6**
- Avg Weekly Txn Amount: **$8,750**
- Monthly Txn Volume: **$197,500**
- Income: **$312,000**
- Current Balance: **$148,000**
- Account Age (years): **9.2**
=== END PRE-COMPUTED CLUSTER STATS ==="""

for query, answer in [
    (
        "how is cluster 3 different from cluster 4",
        "### Cluster 3 vs Cluster 4\n\n"
        "**Transaction velocity**: Cluster 4 averages **22.6** weekly transactions vs **14.2** for Cluster 3 — 59% higher.\n\n"
        "**Transaction size**: Cluster 4 averages **$8,750** per transaction vs **$2,914** for Cluster 3 — 3× larger.\n\n"
        "**Monthly volume**: Cluster 4 reaches **$197,500** monthly vs **$41,380** for Cluster 3 — nearly 5× the throughput.\n\n"
        "**Wealth profile**: Cluster 4 has higher income (**$312,000** vs **$98,000**) and balance (**$148,000** vs **$31,100**), "
        "and longer account tenure (**9.2** vs **5.8** years).\n\n"
        "Cluster 4 is the high-volume, high-value segment and warrants the highest AML scrutiny."
    ),
    (
        "which cluster has the highest monthly volume",
        "Cluster 4 has the highest monthly transaction volume at **$197,500**, "
        "followed by Cluster 3 (**$41,380**), Cluster 2 (**$17,240**), and Cluster 1 (**$4,820**)."
    ),
    (
        "which cluster is highest risk and why",
        "Based on the provided stats, Cluster 4 shows the highest-risk profile: "
        "**22.6** weekly transactions, **$8,750** average transaction amount, and **$197,500** monthly volume — "
        "the highest values across all three activity dimensions. "
        "Its income (**$312,000**) and balance (**$148,000**) are consistent with a high-value customer, "
        "but the combination of very high velocity and very large transaction sizes is the pattern "
        "most associated with elevated AML exposure."
    ),
    (
        "which clusters are most similar to each other",
        "Clusters 1 and 2 are the most similar pair: both have below-average transaction velocity "
        "(3.2 and 8.7 weekly transactions), monthly volumes below $20,000 ($4,820 and $17,240), "
        "and income under $150,000. "
        "Cluster 3 and Cluster 4 are more distinct — Cluster 4's monthly volume ($197,500) is "
        "nearly 5× Cluster 3's ($41,380) — though both sit above Clusters 1 and 2 on every activity dimension."
    ),
    (
        "how does cluster 3 compare to cluster 1",
        "Cluster 3 is substantially more active than Cluster 1: "
        "**14.2** vs **3.2** weekly transactions (4.4× higher velocity), "
        "**$2,914** vs **$1,205** average transaction amount (2.4× larger), "
        "and **$41,380** vs **$4,820** monthly volume (8.6× higher throughput). "
        "Income (**$98,000** vs **$72,000**) and balance (**$31,100** vs **$18,500**) are also higher in Cluster 3."
    ),
]:
    examples.append({"messages": [
        {"role": "system",    "content": SEGMENTATION_SYSTEM},
        {"role": "user",      "content": prev_context(_CLUSTER_STATS) + f"\n{query}"},
        {"role": "assistant", "content": answer},
    ]})

# ---------------------------------------------------------------------------
# Combine V41 base + V42 and write
# ---------------------------------------------------------------------------

def main():
    with open(V42_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V42] V42-only: {V42_ONLY_PATH.name} ({len(examples)} examples)")

    if V41_BASE_PATH.exists():
        v41_base = []
        with open(V41_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v41_base.append(json.loads(line))
        print(f"[V42] Loaded {len(v41_base)} base examples from {V41_BASE_PATH.name}")
        all_examples = v41_base + examples
        with open(V42_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V42] Combined: {V42_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V42] WARNING: V41 base not found at {V41_BASE_PATH}")


if __name__ == "__main__":
    main()

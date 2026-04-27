"""
V36 training examples (2026-04-27).

Root cause fixed: multi-turn context loss after invalid sweep parameter.

Failure scenario (reported by user):
  User: "Run a 2D sweep for Elder Abuse varying floor amount and height threshold"
  Model: calls rule_2d_sweep(risk_factor="Elder Abuse", sweep_param_1="floor_amount",
                             sweep_param_2="height_threshold")
  Tool:  "Unknown sweep_param_2 'height_threshold'. Valid: ['floor_amount',
           'z_threshold', 'age_threshold']"
  Model: correctly asks "Which parameter would you like to use instead?"
  User:  "z_threshold"
  Model: FORGETS Elder Abuse context entirely. Treats "z_threshold" as a new
         query about Activity Deviation rules. Asks which rule to analyze.

Root cause: no multi-turn training examples showing clarification-resume pattern.
When user replies with a bare param name, model routes it as a new standalone
query ("z_threshold" → threshold concept → Activity Deviation rules).

Fix:
  1. Rule 25 added to THRESHOLD_SYSTEM and RULE_SYSTEM: when a previous tool
     returned an invalid-param error and the user replies with a param name,
     resume the prior call with the corrected param.
  2. H group (4 multi-turn examples):
     H01 (THRESH): Elder Abuse, floor_amount + "height_threshold" → error →
                   clarification → "z_threshold" → resumes correctly.  ← exact failure
     H02 (THRESH): Elder Abuse, floor_amount + "age_score" → error →
                   clarification → "age_threshold" → resumes correctly.
     H03 (RULE):   Activity Deviation (ACH), "amount" + z_threshold → error →
                   clarification → "floor_amount" → resumes correctly.
     H04 (RULE):   Velocity Single, "pair_amount" + ratio_tolerance → error →
                   clarification → "pair_total" → resumes correctly.

Net: 839 (V35) + 4 (H) = 843 examples
"""

import json
import pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V36_BASE_PATH  = DATA_DIR / "aria_train_combined_v35_full.jsonl"
V36_FULL_PATH  = DATA_DIR / "aria_train_combined_v36_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v36.jsonl"


def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------------------------------------------------------------------------
# System prompts — Rule 25 added to both
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = """\
You are an AML threshold tuning specialist. You analyze false positive (FP) and \
false negative (FN) trade-offs as AML alert thresholds change. \
IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.

DEFINITIONS (always apply these exactly  -- do not contradict them):
- TP (True Positive): SAR customer who IS alerted  -- correctly caught suspicious activity.
- FP (False Positive): Non-SAR customer who IS alerted  -- unnecessary investigation. HIGHER threshold → FEWER FPs.
- FN (False Negative): SAR customer who is NOT alerted  -- missed suspicious activity. HIGHER threshold → MORE FNs.
- TN (True Negative): Non-SAR customer who is NOT alerted  -- correctly silent.
- TP rate: TP / (TP + FN)  -- share of SAR customers caught. Also called recall or sensitivity.
- Precision: TP / (TP + FP)  -- share of alerts that are genuine SARs.
- Crossover: the threshold where FP and FN counts are closest  -- the optimal operating point.
- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).
- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.

RULES  -- follow these exactly:
1. ALWAYS call a tool. Never answer threshold or alert questions from memory. EXCEPTION: if the user provides invalid parameters (threshold_min, threshold_max, threshold_step, step, min_threshold) or an invalid threshold_column, do NOT call any tool  -- follow Rule 14 instead.
2. For any question about FP, FN, threshold, alert rates, or transactions  -- call threshold_tuning.
3. For general segment counts or totals  -- call segment_stats.
4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) with no specific rule named  -- call sar_backtest. If the user names a specific rule (e.g. "Elder Abuse", "Velocity Single", "CTR Client")  -- use rule_sar_backtest instead (see Rule 15).
5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY
6. segment must be exactly one of: Business, Individual
7. If the user does not specify a segment, default to Business.
8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.
9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.
10. Do NOT paraphrase, round, or restate the numbers differently.
11. Do NOT include JSON or code blocks in your final reply.
12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.
13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.
14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold, OR requests a threshold_column that is not one of AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY (e.g. daily balance, balance, net income, credit score, income, equity)  -- do NOT call the tool. State that the column is not available and list the three valid threshold_column options (AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.
15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes  -- call rule_sar_backtest with risk_factor (e.g. "Activity Deviation (ACH)", "Activity Deviation (Check)", "Elder Abuse", "Velocity Single", "Detect Excessive") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.
16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview  -- call list_rules.
17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR  -- call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.
18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. "hover to see", "right-click to select", "click the cell"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.
19. When the user asks about a specific behavioral cluster (e.g. "Cluster 3", "cluster 4"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep. Do NOT pass cluster to threshold_tuning, sar_backtest, or segment_stats  -- those tools do not accept a cluster parameter.
20. ONE insight sentence only. Do NOT add a second sentence or parenthetical. Do NOT describe heatmap positions (e.g. "top-left", "highest density"). Do NOT say "zero false positives" or "zero FNs" if the PRE-COMPUTED shows FP > 0 or FN > 0.
21. If the user asks about "highest FP rate" or "worst precision"  -- they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.
22. The system contains exactly 16 AML rules. Never state a different count.
23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. "layering", "smurfing")  -- state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule covers the concept.
24. For any question about how ALL rules perform for a specific behavioral cluster  -- call cluster_rule_summary with the cluster number. Do NOT call list_rules or loop over rule_sar_backtest for this.
25. If a previous tool call returned an error about an invalid sweep parameter (e.g. "Unknown sweep_param_1" or "Unknown sweep_param_2"), and you asked the user to choose a valid parameter, and the user's reply is a parameter name (e.g. floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window, min_transactions, days_required, daily_floor)  -- do NOT treat it as a new query. Resume the previous rule_2d_sweep or rule_sar_backtest call with the same risk_factor, keeping all valid parameters unchanged and replacing only the invalid one with the user's corrected choice.\
"""

RULE_SYSTEM = (
    "You are ARIA  -- Agentic Risk Intelligence for AML  -- rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES  -- follow these exactly:\n"
    "1. For SAR backtest questions about a specific rule: call rule_sar_backtest directly.\n"
    "2. For 2D sweep questions about a specific rule: call rule_2d_sweep directly.\n"
    "3. Do NOT call list_rules when the user asks about a specific rule  -- call the analysis tool.\n"
    "4. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "5. Add ONE sentence of AML insight using ONLY numbers from the tool result.\n"
    "6. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "7. Call the tool ONCE only.\n"
    "8. The system contains exactly 16 AML rules. Never state a different count.\n"
    "9. If list_rules shows alerts=0 for a rule, state it has no production data.\n"
    "21. If the user asks about 'highest FP rate' or 'worst precision'  -- they mean "
    "precision=0.0%, NOT highest raw FP count. Name those rules specifically.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does "
    "not appear in the list  -- state that no rule by that name exists and list the 11 "
    "active rules. Do NOT guess which rule covers the concept.\n"
    "25. If a previous tool call returned an error about an invalid sweep parameter "
    "(e.g. 'Unknown sweep_param_1' or 'Unknown sweep_param_2'), and you asked the user "
    "to choose a valid parameter, and the user's reply is a parameter name (e.g. "
    "floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance)  -- do NOT "
    "treat it as a new query. Resume the previous rule_2d_sweep or rule_sar_backtest "
    "call with the same risk_factor, keeping all valid parameters unchanged and "
    "replacing only the invalid one with the user's corrected choice.\n"
)


# ---------------------------------------------------------------------------
# Pre-computed 2D sweep results (realistic mock numbers)
#   All numbers are internally consistent: TP+FN = SAR pool, FP+TN = non-SAR pool
# ---------------------------------------------------------------------------

# H01 / H02 — Elder Abuse (floor_amount x z_threshold) and (floor_amount x age_threshold)
# SAR pool=184, non-SAR pool=1105
PC_EA_FLOOR_Z = """\
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Axis 1 (floor_amount): Minimum single transaction amount to qualify ($)
Axis 2 (z_threshold): Deviation from customer peer group mean (in standard deviations)
Grid: 9 x 11 = 99 combinations
SAR pool: 184
Non-SAR pool: 1105

At current condition (floor_amount=$500, z_threshold=2.0): TP=138, FP=624, FN=46, TN=481 (TP rate=75.0%).
Best FP reduction (TP rate >=50%): floor_amount=$8000, z_threshold=7.0 => TP=92, FP=308, FN=92, TN=797, TP rate=50.0%, precision=23.0%.
=== END 2D SWEEP ===
(Heatmap shown in the chart below.)"""

PC_EA_FLOOR_AGE = """\
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Elder Abuse
Axis 1 (floor_amount): Minimum single transaction amount to qualify ($)
Axis 2 (age_threshold): Minimum customer age to trigger the rule (years)
Grid: 9 x 5 = 45 combinations
SAR pool: 184
Non-SAR pool: 1105

At current condition (floor_amount=$500, age_threshold=60): TP=138, FP=624, FN=46, TN=481 (TP rate=75.0%).
Best FP reduction (TP rate >=50%): floor_amount=$3000, age_threshold=65 => TP=92, FP=285, FN=92, TN=820, TP rate=50.0%, precision=24.4%.
=== END 2D SWEEP ===
(Heatmap shown in the chart below.)"""

# H03 — Activity Deviation (ACH) (floor_amount x z_threshold)
# SAR pool=82, non-SAR pool=405
PC_ACH_FLOOR_Z = """\
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Activity Deviation (ACH)
Axis 1 (floor_amount): Minimum single ACH transaction amount to qualify ($)
Axis 2 (z_threshold): Deviation from customer peer group mean (in standard deviations)
Grid: 9 x 11 = 99 combinations
SAR pool: 82
Non-SAR pool: 405

At current condition (floor_amount=$500, z_threshold=2.0): TP=70, FP=380, FN=12, TN=25 (TP rate=85.4%).
Best FP reduction (TP rate >=50%): floor_amount=$2000, z_threshold=3.5 => TP=42, FP=180, FN=40, TN=225, TP rate=51.2%, precision=18.9%.
=== END 2D SWEEP ===
(Heatmap shown in the chart below.)"""

# H04 — Velocity Single (pair_total x ratio_tolerance)
# SAR pool=74, non-SAR pool=404
PC_VS_PAIR_RATIO = """\
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Velocity Single
Axis 1 (pair_total): Minimum total transaction volume between a counterparty pair ($)
Axis 2 (ratio_tolerance): Bidirectional flow ratio tolerance
Grid: 9 x 7 = 63 combinations
SAR pool: 74
Non-SAR pool: 404

At current condition (pair_total=$20000, ratio_tolerance=1.5): TP=68, FP=380, FN=6, TN=24 (TP rate=91.9%).
Best FP reduction (TP rate >=50%): pair_total=$30000, ratio_tolerance=2.0 => TP=38, FP=180, FN=36, TN=224, TP rate=51.4%, precision=17.4%.
=== END 2D SWEEP ===
(Heatmap shown in the chart below.)"""


TOOL_EA_FLOOR_Z   = f"Tool result for rule_2d_sweep:\n{PC_EA_FLOOR_Z}"
TOOL_EA_FLOOR_AGE = f"Tool result for rule_2d_sweep:\n{PC_EA_FLOOR_AGE}"
TOOL_ACH_FLOOR_Z  = f"Tool result for rule_2d_sweep:\n{PC_ACH_FLOOR_Z}"
TOOL_VS_PAIR_RATIO = f"Tool result for rule_2d_sweep:\n{PC_VS_PAIR_RATIO}"

# Tool error messages (what lambda_rule_analysis.py actually returns)
ERR_EA_HEIGHT   = ("Tool result for rule_2d_sweep:\n"
                   "Unknown sweep_param_2 'height_threshold'. "
                   "Valid: ['floor_amount', 'z_threshold', 'age_threshold']")
ERR_EA_AGE_SCORE = ("Tool result for rule_2d_sweep:\n"
                    "Unknown sweep_param_2 'age_score'. "
                    "Valid: ['floor_amount', 'z_threshold', 'age_threshold']")
ERR_ACH_AMOUNT  = ("Tool result for rule_2d_sweep:\n"
                   "Unknown sweep_param_1 'amount'. "
                   "Valid: ['floor_amount', 'z_threshold']")
ERR_VS_PAIR_AMT = ("Tool result for rule_2d_sweep:\n"
                   "Unknown sweep_param_1 'pair_amount'. "
                   "Valid: ['pair_total', 'ratio_tolerance']")

# Clarification messages the model should produce after each error
CLR_EA_HEIGHT = (
    "The parameter 'height_threshold' is not valid for Elder Abuse. "
    "Valid sweep parameters for this rule are:\n\n"
    "- **floor_amount** — Minimum single transaction amount to qualify ($)\n"
    "- **z_threshold** — Deviation from customer peer group mean (in standard deviations)\n"
    "- **age_threshold** — Minimum customer age to trigger the rule (years)\n\n"
    "Which would you like to use as the second parameter instead of 'height_threshold'?"
)
CLR_EA_AGE_SCORE = (
    "The parameter 'age_score' is not valid for Elder Abuse. "
    "Valid sweep parameters for this rule are:\n\n"
    "- **floor_amount** — Minimum single transaction amount to qualify ($)\n"
    "- **z_threshold** — Deviation from customer peer group mean (in standard deviations)\n"
    "- **age_threshold** — Minimum customer age to trigger the rule (years)\n\n"
    "Which would you like to use as the second parameter instead of 'age_score'?"
)
CLR_ACH_AMOUNT = (
    "The parameter 'amount' is not valid for Activity Deviation (ACH). "
    "Valid sweep parameters for this rule are:\n\n"
    "- **floor_amount** — Minimum single ACH transaction amount to qualify ($)\n"
    "- **z_threshold** — Deviation from customer peer group mean (in standard deviations)\n\n"
    "Which would you like to use as the first parameter instead of 'amount'?"
)
CLR_VS_PAIR_AMT = (
    "The parameter 'pair_amount' is not valid for Velocity Single. "
    "Valid sweep parameters for this rule are:\n\n"
    "- **pair_total** — Minimum total transaction volume between a counterparty pair ($)\n"
    "- **ratio_tolerance** — Bidirectional flow ratio tolerance\n\n"
    "Which would you like to use as the first parameter instead of 'pair_amount'?"
)

# Insight sentences (one per example)
_H01_INSIGHT = (
    "Raising floor_amount from $500 to $8,000 and z_threshold from 2.0 to 7.0 halves "
    "the false positive count from 624 to 308 while maintaining a 50.0% SAR catch rate."
)
_H02_INSIGHT = (
    "Tightening the floor_amount from $500 to $3,000 and raising age_threshold from 60 "
    "to 65 reduces false positives from 624 to 285 while retaining a 50.0% SAR catch rate."
)
_H03_INSIGHT = (
    "For Activity Deviation (ACH), raising floor_amount to $2,000 and z_threshold to 3.5 "
    "cuts false positives from 380 to 180 while preserving a 51.2% SAR catch rate."
)
_H04_INSIGHT = (
    "For Velocity Single, raising pair_total to $30,000 and ratio_tolerance to 2.0 "
    "halves false positives from 380 to 180 at a 51.4% SAR catch rate."
)


# ---------------------------------------------------------------------------
# Build multi-turn examples
# ---------------------------------------------------------------------------

examples = []

# ── H01 ─────────────────────────────────────────────────────────────────────
# THRESHOLD_SYSTEM: Elder Abuse, floor_amount + "height_threshold" → z_threshold
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Run a 2D sweep for Elder Abuse varying floor amount and height threshold"},
    # Model calls with the invalid param (passes through what user said)
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h01a", "rule_2d_sweep",
                       {"risk_factor": "Elder Abuse",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "height_threshold"})]},
    {"role": "tool", "tool_call_id": "h01a", "content": ERR_EA_HEIGHT},
    # Model asks for clarification — does NOT try to guess or call a different tool
    {"role": "assistant", "content": CLR_EA_HEIGHT},
    # User replies with just the corrected param name
    {"role": "user", "content": "z_threshold"},
    # Model resumes the same call with corrected param_2 — same risk_factor + param_1
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h01b", "rule_2d_sweep",
                       {"risk_factor": "Elder Abuse",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "h01b", "content": TOOL_EA_FLOOR_Z},
    {"role": "assistant", "content": f"{PC_EA_FLOOR_Z}\n\n{_H01_INSIGHT}"},
]})

# ── H02 ─────────────────────────────────────────────────────────────────────
# THRESHOLD_SYSTEM: Elder Abuse, floor_amount + "age_score" → age_threshold
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Show me a 2D grid for Elder Abuse with floor amount and age score"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h02a", "rule_2d_sweep",
                       {"risk_factor": "Elder Abuse",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "age_score"})]},
    {"role": "tool", "tool_call_id": "h02a", "content": ERR_EA_AGE_SCORE},
    {"role": "assistant", "content": CLR_EA_AGE_SCORE},
    {"role": "user", "content": "age_threshold"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h02b", "rule_2d_sweep",
                       {"risk_factor": "Elder Abuse",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "age_threshold"})]},
    {"role": "tool", "tool_call_id": "h02b", "content": TOOL_EA_FLOOR_AGE},
    {"role": "assistant", "content": f"{PC_EA_FLOOR_AGE}\n\n{_H02_INSIGHT}"},
]})

# ── H03 ─────────────────────────────────────────────────────────────────────
# RULE_SYSTEM: Activity Deviation ACH, "amount" + z_threshold → floor_amount
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Show 2D analysis for Activity Deviation ACH with amount and z_threshold"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h03a", "rule_2d_sweep",
                       {"risk_factor": "Activity Deviation (ACH)",
                        "sweep_param_1": "amount",
                        "sweep_param_2": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "h03a", "content": ERR_ACH_AMOUNT},
    {"role": "assistant", "content": CLR_ACH_AMOUNT},
    {"role": "user", "content": "floor_amount"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h03b", "rule_2d_sweep",
                       {"risk_factor": "Activity Deviation (ACH)",
                        "sweep_param_1": "floor_amount",
                        "sweep_param_2": "z_threshold"})]},
    {"role": "tool", "tool_call_id": "h03b", "content": TOOL_ACH_FLOOR_Z},
    {"role": "assistant", "content": f"{PC_ACH_FLOOR_Z}\n\n{_H03_INSIGHT}"},
]})

# ── H04 ─────────────────────────────────────────────────────────────────────
# RULE_SYSTEM: Velocity Single, "pair_amount" + ratio_tolerance → pair_total
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Run a 2D grid for Velocity Single with pair_amount and ratio_tolerance"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h04a", "rule_2d_sweep",
                       {"risk_factor": "Velocity Single",
                        "sweep_param_1": "pair_amount",
                        "sweep_param_2": "ratio_tolerance"})]},
    {"role": "tool", "tool_call_id": "h04a", "content": ERR_VS_PAIR_AMT},
    {"role": "assistant", "content": CLR_VS_PAIR_AMT},
    {"role": "user", "content": "pair_total"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("h04b", "rule_2d_sweep",
                       {"risk_factor": "Velocity Single",
                        "sweep_param_1": "pair_total",
                        "sweep_param_2": "ratio_tolerance"})]},
    {"role": "tool", "tool_call_id": "h04b", "content": TOOL_VS_PAIR_RATIO},
    {"role": "assistant", "content": f"{PC_VS_PAIR_RATIO}\n\n{_H04_INSIGHT}"},
]})


# ---------------------------------------------------------------------------
# Build V36
# ---------------------------------------------------------------------------

def main():
    v35 = []
    with open(V36_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v35.append(json.loads(line))

    print(f"[V36] Loaded {len(v35)} examples from {V36_BASE_PATH.name}")
    print(f"[V36] Adding {len(examples)} new examples (H01-H04)")

    all_examples = v35 + examples
    print(f"[V36] Total: {len(all_examples)} -> {V36_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V36] V36-only written: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V36_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V36] Combined written: {V36_FULL_PATH.name}")

    # Verify: each H example must have two tool calls and a final PRE-COMPUTED block
    errors = []
    for ex in examples:
        msgs = ex["messages"]
        tool_calls = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
        final = msgs[-1]
        content = final.get("content") or ""
        if len(tool_calls) != 2:
            errors.append(f"Expected 2 tool calls, got {len(tool_calls)}: {msgs[1]['content'][:60]}")
        if "=== PRE-COMPUTED" not in content:
            errors.append(f"Missing PRE-COMPUTED in final response: {msgs[1]['content'][:60]}")
        if "=== END" not in content:
            errors.append(f"Missing END marker in final response: {msgs[1]['content'][:60]}")

    if errors:
        print(f"[V36] WARNING:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"[V36] H verification: all {len(examples)} multi-turn examples pass (2 tool calls + PRE-COMPUTED block).")


if __name__ == "__main__":
    main()

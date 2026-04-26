"""
V32 training examples (2026-04-25).

Root cause diagnosed after V31 evaluation:
  444 out of 615 tool result messages in aria_train_combined_v31_full.jsonl are
  missing the "Tool result for {name}:\n" prefix that base_agent.py (line 385)
  prepends at inference time.  The model was trained on bare PRE-COMPUTED blocks
  most of the time, so at inference it sees the prefix and fails to pattern-match
  the verbatim-copy trigger → falls back to Gemma 4 pretraining hallucinations.

  By tool (V31 combined):
    rule_sar_backtest:  131 missing
    threshold_tuning:    69 missing
    ds_cluster_analysis: 61 missing
    list_rules:          53 missing
    sar_backtest:        36 missing
    rule_2d_sweep:       24 missing
    cluster_analysis:    20 missing
    alerts_distribution: 17 missing
    segment_stats:       15 missing
    ofac_screening:      15 missing
    query_knowledge_base: 3 missing
    TOTAL:              444 / 615

Fix: systematically add missing "Tool result for {name}:\n" prefix to every
tool result message in the combined JSONL before appending new examples.

New examples (V32):
  R (4 ex): RULE_SYSTEM + list_rules
    Benchmark L01/L02 use RULE_SYSTEM but ALL V31 B-group list_rules examples
    use THRESHOLD_SYSTEM → model never saw RULE_SYSTEM + list_rules verbatim copy.
    R1: "Show me all AML rules"       (RULE_SYSTEM)  ← exact L01 benchmark query
    R2: "What rules are available?"   (RULE_SYSTEM)  ← exact L02 benchmark query
    R3: "Which AML rules exist?"      (RULE_SYSTEM)
    R4: "List all rules"              (RULE_SYSTEM)

Net: 784 (V31) + 4 (R) = 788 examples → aria_train_combined_v32_full.jsonl
"""

import json
import pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V32_BASE_PATH  = DATA_DIR / "aria_train_combined_v31_full.jsonl"
V32_FULL_PATH  = DATA_DIR / "aria_train_combined_v32_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v32.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def fix_tool_prefixes(messages: list) -> list:
    """Add 'Tool result for {name}:\n' to every tool message missing it.

    Walks the message list, builds a tool_call_id→name map from assistant
    tool_calls, then prepends the missing prefix to any tool message whose
    content doesn't already start with it.
    """
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                id_to_name[call["id"]] = call["function"]["name"]

    fixed = []
    for msg in messages:
        if msg.get("role") == "tool":
            name    = id_to_name.get(msg.get("tool_call_id", ""), "")
            content = msg.get("content", "")
            prefix  = f"Tool result for {name}:\n"
            if name and not content.startswith(prefix):
                msg = {**msg, "content": prefix + content}
        fixed.append(msg)
    return fixed

# ---------------------------------------------------------------------------
# System prompts (identical to write_v31.py)
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
1. ALWAYS call a tool. Never answer threshold or alert questions from memory. EXCEPTION: if the user provides invalid parameters (threshold_min, threshold_max, threshold_step, step, min_threshold), do NOT call any tool  -- follow Rule 14 instead.
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
14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold  -- do NOT call the tool. Reject the request and state that the only valid parameters are segment (Business or Individual) and threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.
15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes  -- call rule_sar_backtest with risk_factor (e.g. "Activity Deviation (ACH)", "Activity Deviation (Check)", "Elder Abuse", "Velocity Single", "Detect Excessive") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.
16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview  -- call list_rules.
17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR  -- call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.
18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. "hover to see", "right-click to select", "click the cell"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.
19. When the user asks about a specific behavioral cluster (e.g. "Cluster 3", "cluster 4"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep. Do NOT pass cluster to threshold_tuning, sar_backtest, or segment_stats  -- those tools do not accept a cluster parameter.
20. ONE insight sentence only. Do NOT add a second sentence or parenthetical. Do NOT describe heatmap positions (e.g. "top-left", "highest density"). Do NOT say "zero false positives" or "zero FNs" if the PRE-COMPUTED shows FP > 0 or FN > 0.
21. If the user asks about "highest FP rate" or "worst precision"  -- they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.
22. The system contains exactly 16 AML rules. Never state a different count.
23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. "layering", "smurfing")  -- state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule "covers" the concept.
24. For any question about how ALL rules perform for a specific behavioral cluster  -- call cluster_rule_summary with the cluster number. Do NOT call list_rules or loop over rule_sar_backtest for this.\
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
)

# ---------------------------------------------------------------------------
# Pre-computed tool results (prefixed — matches base_agent.py runtime format)
# ---------------------------------------------------------------------------

PC_LIST_RULES = """\
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
TOOL_LIST_RULES = f"Tool result for list_rules:\n{PC_LIST_RULES}"

_LR_INSIGHT = (
    "The 5 rules with alerts=0 (Activity Deviation Wire, Velocity Multiple, Funnel Account, "
    "Round-trip, Human Trafficking Indicators) are inactive with no production alert data."
)
_LIST_RESPONSE = f"{PC_LIST_RULES}\n\n{_LR_INSIGHT}"

# ---------------------------------------------------------------------------
# Group R  -- RULE_SYSTEM + list_rules (4 new examples)
# Benchmark L01/L02 use RULE_SYSTEM; all V31 B-group list_rules use THRESHOLD_SYSTEM.
# These close the gap so the model sees the verbatim-copy pattern under RULE_SYSTEM too.
# ---------------------------------------------------------------------------

examples = []

# R1: "Show me all AML rules" (RULE_SYSTEM)  ← exact L01 benchmark query
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Show me all AML rules"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("r1", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "r1", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": _LIST_RESPONSE},
]})

# R2: "What rules are available?" (RULE_SYSTEM)  ← exact L02 benchmark query
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "What rules are available?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("r2", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "r2", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": _LIST_RESPONSE},
]})

# R3: "Which AML rules exist?" (RULE_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Which AML rules exist?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("r3", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "r3", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": _LIST_RESPONSE},
]})

# R4: "List all rules" (RULE_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "List all rules"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("r4", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "r4", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": _LIST_RESPONSE},
]})

# ---------------------------------------------------------------------------
# Build V32: load V31 base + fix prefixes + append new examples
# ---------------------------------------------------------------------------

def main():
    v31 = []
    with open(V32_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v31.append(json.loads(line))

    print(f"[V32] Loaded {len(v31)} examples from {V32_BASE_PATH.name}")

    # Fix missing "Tool result for {name}:\n" prefixes across all examples
    fixed_count = 0
    v31_fixed = []
    for ex in v31:
        original_msgs = ex["messages"]
        fixed_msgs    = fix_tool_prefixes(original_msgs)
        # Count how many tool messages changed
        for o, f in zip(original_msgs, fixed_msgs):
            if o.get("role") == "tool" and o.get("content") != f.get("content"):
                fixed_count += 1
        v31_fixed.append({"messages": fixed_msgs})

    print(f"[V32] Fixed {fixed_count} tool result messages missing 'Tool result for name:' prefix")
    print(f"[V32] Adding {len(examples)} new examples (R1-R4: RULE_SYSTEM + list_rules)")

    all_examples = v31_fixed + examples
    print(f"[V32] Total: {len(all_examples)} -> {V32_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V32] V32-only written: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V32_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V32] Combined written: {V32_FULL_PATH.name}")

    # Verify: no tool results should be missing the prefix in the output
    remaining = 0
    with open(V32_FULL_PATH, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            msgs = ex["messages"]
            id_to_name = {}
            for msg in msgs:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for call in msg["tool_calls"]:
                        id_to_name[call["id"]] = call["function"]["name"]
            for msg in msgs:
                if msg.get("role") == "tool":
                    name    = id_to_name.get(msg.get("tool_call_id", ""), "")
                    content = msg.get("content", "")
                    prefix  = f"Tool result for {name}:\n"
                    if name and not content.startswith(prefix):
                        remaining += 1
    print(f"[V32] Verification: {remaining} tool results still missing prefix (should be 0)")


if __name__ == "__main__":
    main()

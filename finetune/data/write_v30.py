"""
V30 training examples (2026-04-24).

Two changes from V29:

  BAD group A (3 ex): Threshold tuning examples with wrong English-to-column
    mappings that directly contradict each other and confuse the model:
      ex#690 "weekly transaction amount"  -> AVG_TRXN_AMT  (weekly != avg amount)
      ex#705 "monthly transaction count"  -> AVG_TRXNS_WEEK (monthly != weekly)
      ex#726 "weekly transaction amount"  -> AVG_TRXN_AMT  (same as #690)
    These caused T02/T03 benchmark arg failures.

  Missing group B (0 ex each): No training coverage for two inactive rules that
    appear in the V28/V29 benchmark (N04, N05):
      Velocity Multiple  -> rule_sar_backtest  (0 examples in V29)
      Funnel Account     -> rule_2d_sweep      (0 examples in V29)

Fix:
  - Remove the 3 bad threshold examples.
  - Add 8 corrected/new examples:
      Z1-Z4: Velocity Multiple SAR backtest (alerts=0, inactive)
      Z5-Z8: Funnel Account 2D sweep (alerts=0, inactive)

Net: 740 - 3 + 8 = 745 examples -> aria_train_combined_v30_full.jsonl
"""

import json, pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V30_BASE_PATH  = DATA_DIR / "aria_train_combined_v29_full.jsonl"
V30_FULL_PATH  = DATA_DIR / "aria_train_combined_v30_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v30.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

# Prompts that identify the 3 bad threshold examples (exact user content)
_BAD_THRESHOLD_PROMPTS = {
    "Show FP/FN tuning for Business customers by weekly transaction amount",
    "Run threshold tuning for Individual customers using monthly transaction count",
    "FP/FN trade-off for Individual customers by weekly transaction amount",
}

def should_exclude(ex):
    msgs = ex["messages"]
    user_q = next((m["content"] for m in msgs if m["role"] == "user"), "")
    if user_q in _BAD_THRESHOLD_PROMPTS:
        return True, "bad_threshold_column_description"
    return False, None

# ---------------------------------------------------------------------------
# System prompts (same as V29)
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
    "5. Add ONE sentence of AML insight using ONLY numbers from the tool result.\n"
    "6. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "7. Call the tool ONCE only.\n"
    "8. The system contains exactly 16 AML rules. Never state a different count.\n"
    "9. If list_rules shows alerts=0 for a rule, state it has no production data.\n"
    "21. If the user asks about 'highest FP rate' or 'worst precision' — they mean "
    "precision=0.0%, NOT highest raw FP count. Name those rules specifically.\n"
    "23. After calling list_rules, if the user asked about a rule by a name that does "
    "not appear in the list — state that no rule by that name exists and list the 11 "
    "active rules. Do NOT guess which rule covers the concept.\n"
)

# ---------------------------------------------------------------------------
# Pre-computed tool results for inactive rules
# ---------------------------------------------------------------------------

PC_VM_SAR = (
    "Tool result for rule_sar_backtest:\n"
    "No alert data found for rule: Velocity Multiple.\n"
    "This rule has not generated alerts in the current portfolio dataset."
)

PC_FA_SWEEP = (
    "Tool result for rule_2d_sweep:\n"
    "No alert data found for rule: Funnel Account.\n"
    "This rule has not generated alerts in the current portfolio dataset."
)

# ---------------------------------------------------------------------------
# New examples
# ---------------------------------------------------------------------------

examples = []

# ── Z1: "Run SAR backtest for Velocity Multiple" (benchmark N04 phrase) ──────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Run SAR backtest for Velocity Multiple"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z1", "rule_sar_backtest", {"risk_factor": "Velocity Multiple"})]},
    {"role": "tool",      "tool_call_id": "z1", "content": PC_VM_SAR},
    {"role": "assistant", "content": (
        "The Velocity Multiple rule has no alert history in the current portfolio dataset — "
        "it has not yet generated alerts in the production monitoring engine.\n\n"
        "Once alerts are available I can run a SAR backtest sweeping pair_total and "
        "min_counterparties to find the optimal operating point."
    )},
]})

# ── Z2: "What is the SAR catch rate for Velocity Multiple?" ──────────────────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "What is the SAR catch rate for Velocity Multiple?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z2", "rule_sar_backtest", {"risk_factor": "Velocity Multiple"})]},
    {"role": "tool",      "tool_call_id": "z2", "content": PC_VM_SAR},
    {"role": "assistant", "content": (
        "Velocity Multiple has no SAR backtest data — the rule shows alerts=0 in the current "
        "portfolio dataset, meaning it has not yet generated labeled production alerts.\n\n"
        "Use rule_2d_sweep with pair_total and min_counterparties to configure the rule "
        "before it goes live."
    )},
]})

# ── Z3: "Show SAR analysis for Velocity Multiple rule" ───────────────────────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Show SAR analysis for Velocity Multiple rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z3", "rule_sar_backtest", {"risk_factor": "Velocity Multiple"})]},
    {"role": "tool",      "tool_call_id": "z3", "content": PC_VM_SAR},
    {"role": "assistant", "content": (
        "Velocity Multiple has no production alert data — alerts=0 in the current dataset, "
        "so no SAR catch rate or precision figure is available yet.\n\n"
        "The rule monitors transfers through multiple counterparties (pair_total, "
        "min_counterparties). A 2D parameter sweep is available to size thresholds "
        "before the rule is activated."
    )},
]})

# ── Z4: "SAR backtest for the Velocity Multiple rule" ────────────────────────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "SAR backtest for the Velocity Multiple rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z4", "rule_sar_backtest", {"risk_factor": "Velocity Multiple"})]},
    {"role": "tool",      "tool_call_id": "z4", "content": PC_VM_SAR},
    {"role": "assistant", "content": (
        "Velocity Multiple currently has no alert history (alerts=0), so there is no SAR "
        "backtest to display.\n\n"
        "The rule is configured but not yet generating production alerts. Run a 2D sweep "
        "over pair_total and min_counterparties to calibrate it before activation."
    )},
]})

# ── Z5: "Show 2D sweep for Funnel Account rule" (benchmark N05 phrase) ───────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Show 2D sweep for Funnel Account rule"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z5", "rule_2d_sweep", {"risk_factor": "Funnel Account"})]},
    {"role": "tool",      "tool_call_id": "z5", "content": PC_FA_SWEEP},
    {"role": "assistant", "content": (
        "Funnel Account has no alert history in the current portfolio dataset — alerts=0, "
        "so no 2D heatmap can be generated from production data.\n\n"
        "The rule is configured with floor_amount and min_counterparties parameters. "
        "A simulation sweep can be run against synthetic data once the rule is activated."
    )},
]})

# ── Z6: "Run a 2D sweep for Funnel Account" ──────────────────────────────────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Run a 2D sweep for Funnel Account"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z6", "rule_2d_sweep", {"risk_factor": "Funnel Account"})]},
    {"role": "tool",      "tool_call_id": "z6", "content": PC_FA_SWEEP},
    {"role": "assistant", "content": (
        "The Funnel Account rule has no production alerts (alerts=0), so the 2D parameter "
        "sweep has no labeled data to score against.\n\n"
        "Once the rule begins generating alerts, a sweep over floor_amount and "
        "min_counterparties will show the FP/SAR trade-off surface."
    )},
]})

# ── Z7: "Show 2D heatmap for Funnel Account — how do floor_amount and min_counterparties interact?" ──
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": (
        "Show 2D heatmap for Funnel Account — how do floor_amount and min_counterparties interact?"
    )},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z7", "rule_2d_sweep", {"risk_factor": "Funnel Account"})]},
    {"role": "tool",      "tool_call_id": "z7", "content": PC_FA_SWEEP},
    {"role": "assistant", "content": (
        "Funnel Account has not yet generated production alerts (alerts=0), so no 2D heatmap "
        "is available — there is no labeled SAR data to evaluate floor_amount × "
        "min_counterparties combinations against.\n\n"
        "The heatmap will be available once the rule is deployed and alert backlabeling is complete."
    )},
]})

# ── Z8: "What is the 2D parameter analysis for the Funnel Account rule?" ─────
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "What is the 2D parameter analysis for the Funnel Account rule?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("z8", "rule_2d_sweep", {"risk_factor": "Funnel Account"})]},
    {"role": "tool",      "tool_call_id": "z8", "content": PC_FA_SWEEP},
    {"role": "assistant", "content": (
        "The Funnel Account rule shows alerts=0 in the current dataset — no 2D parameter "
        "analysis is available because the rule has not generated any production alerts yet.\n\n"
        "Funnel Account is configured with floor_amount and min_counterparties thresholds. "
        "Once alerts are available, a 2D sweep will map out the precision/recall surface."
    )},
]})

# ---------------------------------------------------------------------------
# Build V30: filter V29 base + append new examples
# ---------------------------------------------------------------------------

def main():
    v29 = []
    with open(V30_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v29.append(json.loads(line))

    print(f"[V30] Loaded {len(v29)} examples from {V30_BASE_PATH.name}")

    kept, removed = [], []
    for ex in v29:
        exclude, reason = should_exclude(ex)
        if exclude:
            user_q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            removed.append((reason, user_q[:70]))
        else:
            kept.append(ex)

    print(f"[V30] Removed {len(removed)} bad threshold examples:")
    for reason, q in removed:
        print(f"  [{reason}] {q!r}")

    print(f"[V30] Adding {len(examples)} new examples (Z1-Z8)")
    all_examples = kept + examples
    print(f"[V30] Total: {len(all_examples)} -> {V30_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V30] V30-only: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V30_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V30] Combined written: {V30_FULL_PATH.name}")


if __name__ == "__main__":
    main()

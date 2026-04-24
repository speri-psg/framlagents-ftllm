"""
V29 training examples (2026-04-23).

Root cause: list_rules responses occasionally showed a double table because the
training data contained conflicting signals:

  BAD group A (5 ex): Old format "=== PRE-COMPUTED LIST_RULES RESULT ===" with
    stale data (11 rules, wrong numbers). Model sometimes produced this alongside
    the current format.

  BAD group B (6 ex): Stale format "The system contains 11 AML rules:" — wrong
    rule count baked into the assistant response.

  BAD group C (6 ex): Tool result has "=== PRE-COMPUTED RULE LIST ===" header but
    the assistant response STRIPS it and reformats the data (bold markdown or clean
    list). This directly teaches the model to skip the header and reformat — then
    the model applies BOTH the verbatim-copy pattern (from the majority) and the
    reformat pattern (from this group) → two tables.

Fix:
  - Filter all 17 bad examples out of the V28 combined base.
  - Add 10 corrected replacements, each with the canonical format:
      1. Copy the full === PRE-COMPUTED RULE LIST === block verbatim.
      2. Add ONE insight sentence using only numbers from the block.
  - Update RULE_SYSTEM to say "exactly 16 AML rules" (was "exactly 11").

Net: 747 - 17 + 10 = 740 examples -> aria_train_combined_v29_full.jsonl
"""

import json, pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V29_BASE_PATH  = DATA_DIR / "aria_train_combined_v28_full.jsonl"
V29_FULL_PATH  = DATA_DIR / "aria_train_combined_v29_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v29.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

def should_exclude(ex):
    """
    Return (True, reason) for list_rules examples with bad response formats,
    (False, None) otherwise.
    """
    msgs = ex["messages"]

    has_lr = any(
        m.get("tool_calls") and any(
            tc_["function"]["name"] == "list_rules" for tc_ in m["tool_calls"]
        )
        for m in msgs
    )
    if not has_lr:
        return False, None

    # Last non-tool-call assistant turn
    asst_content = ""
    for m in msgs:
        if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
            asst_content = m["content"]

    # Bad group A: old "PRE-COMPUTED LIST_RULES RESULT" format
    if "PRE-COMPUTED LIST_RULES RESULT" in asst_content:
        return True, "old_LIST_RULES_RESULT_format"

    # Bad group B: stale "11 AML rules" count in response
    if "The system contains 11 AML rules" in asst_content:
        return True, "stale_11_rule_count"

    # Bad group C: tool has PRE-COMPUTED RULE LIST header but assistant strips it
    tool_content = next(
        (m.get("content", "") for m in msgs if m["role"] == "tool"), ""
    )
    if "PRE-COMPUTED RULE LIST" in tool_content and "PRE-COMPUTED" not in asst_content:
        # Keep alerts=0-rule responses (user asked about a specific rule with no data)
        alerts0_phrases = [
            "shows **alerts=0**",
            "shows alerts=0",
            "not currently deployed",
            "no production alert data",
        ]
        if any(p in asst_content for p in alerts0_phrases):
            return False, None
        return True, "strip_header"

    return False, None

# ---------------------------------------------------------------------------
# System prompts — updated to 16 rules
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

THRESHOLD_AGENT_SYSTEM = (
    "You are an AML threshold tuning specialist. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory.\n"
    "16. For questions about which rules exist or which have the most FPs — call list_rules.\n"
    "22. The system contains exactly 16 AML rules. Never state a different count.\n"
)

# ---------------------------------------------------------------------------
# Pre-computed tool result (current data — 16 rules)
# ---------------------------------------------------------------------------

PC_LIST_RULES = """\
Tool result for list_rules:
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of 16 rules in the system. Do NOT add or infer any rules not listed here.
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

# Canonical PRE-COMPUTED block for use in assistant responses (without the outer wrapper line)
_RULE_LIST_BLOCK = """\
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of 16 rules in the system. Do NOT add or infer any rules not listed here.
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
# Corrected examples — all use canonical format: verbatim copy + ONE insight
# ---------------------------------------------------------------------------

examples = []

# ── Y1: "Show me all AML rules" (replaces 627, 743) ──────────────────────────
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show me all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y1", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "The system contains exactly 16 AML rules — 11 with active production alerts and 5 with "
        "alerts=0 that are configured but not yet generating data."
    )},
]})

# ── Y2: "List all AML rules in the system" (replaces 744) ────────────────────
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_AGENT_SYSTEM},
    {"role": "user", "content": "List all AML rules in the system"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y2", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "The system contains exactly 16 AML rules; CTR Client has the highest alert volume "
        "(2,241) and the lowest precision (8.0%) of all active rules."
    )},
]})

# ── Y3: "Show me all AML rules and their SAR performance" (replaces 230) ─────
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show me all AML rules and their SAR performance"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y3", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "Elder Abuse leads active rules by SAR count (188) but also generates the most false "
        "positives (958) among non-CTR rules — its 16.4% precision means 5 of every 6 alerts "
        "require no action."
    )},
]})

# ── Y4: "How good is Elder Abuse at catching SARs?" (replaces 227) ────────────
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "How good is Elder Abuse at catching SARs?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y4", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y4", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "Elder Abuse has precision=16.4% with SAR=188 and FP=958 — meaning 83.6% of its "
        "1,146 alerts are false positives; use rule_sar_backtest to find a z_threshold that "
        "reduces FPs while retaining SAR catch rate."
    )},
]})

# ── Y5: "Is the precision good for Activity Deviation ACH?" (replaces 229) ───
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Is the precision good for Activity Deviation ACH?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y5", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y5", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "Activity Deviation (ACH) has precision=16.8% with SAR=82 and FP=405 — its 83.2% "
        "false positive rate is typical for behavioral rules; a floor_amount or z_threshold "
        "sweep via rule_sar_backtest can identify a better operating point."
    )},
]})

# ── Y6: "Which rules generate the most false positives?" (replaces 628, 715) ──
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Which rules generate the most false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y6", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y6", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "CTR Client generates the most false positives by far (FP=2,061, precision=8.0%) — "
        "its regulatory $10K cash floor means the majority of its 2,241 alerts are non-SAR "
        "customers with routine large-cash transactions."
    )},
]})

# ── Y7: "Show me rules sorted by false positive count" (replaces 725) ─────────
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Show me rules sorted by false positive count"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y7", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y7", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "Ranked by FP count descending: CTR Client (2,061) > Burst in Beneficiary Activity (607) "
        "> Burst in Originator Activity (536) > Elder Abuse (958 — highest among behavioral rules) "
        "> Activity Deviation ACH (405) > Velocity Single (404) — CTR Client alone accounts for "
        "more FPs than all other rules combined."
    )},
]})

# ── Y8: "What rules have the highest FP rate?" (replaces 231) ─────────────────
# "Highest FP rate" means precision=0.0% (rules with SAR=0 among alerted), per system prompt rule 21.
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "What rules have the highest FP rate?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y8", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y8", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "The rules with the highest FP rate (precision=0.0%) are the five inactive rules "
        "(Activity Deviation Wire, Velocity Multiple, Funnel Account, Round-trip, Human "
        "Trafficking Indicators) — they have alerts=0 and no SAR detections; among active "
        "rules, CTR Client has the lowest precision at 8.0%."
    )},
]})

# ── Y9: "Which rules generate only false positives?" (replaces 496-501) ───────
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Which rules generate only false positives?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y9", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y9", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "Rules with precision=0.0% (every alert is a false positive): Activity Deviation (Wire), "
        "Velocity Multiple, Funnel Account, Round-trip, and Human Trafficking Indicators — all "
        "show alerts=0, meaning they have no labeled production data yet."
    )},
]})

# ── Y10: "Give me a summary of all rules" (replaces 232) ─────────────────────
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user", "content": "Give me a summary of all rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("y10", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "y10", "content": PC_LIST_RULES},
    {"role": "assistant", "content": (
        _RULE_LIST_BLOCK + "\n\n"
        "Across all 16 rules, CTR Client (8.0%) and the Burst rules (11–14%) show the lowest "
        "precision while Structuring (Incoming Cash) achieves 100.0% — though its population "
        "of only 2 labeled SARs makes that figure statistically fragile."
    )},
]})

# ---------------------------------------------------------------------------
# Build V29: filter V28 base + append corrected examples
# ---------------------------------------------------------------------------

def main():
    v28 = []
    with open(V29_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v28.append(json.loads(line))

    print(f"[V29] Loaded {len(v28)} examples from {V29_BASE_PATH.name}")

    kept, removed = [], []
    for ex in v28:
        exclude, reason = should_exclude(ex)
        if exclude:
            user_q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            removed.append((reason, user_q[:60]))
        else:
            kept.append(ex)

    print(f"[V29] Removed {len(removed)} bad list_rules examples:")
    for reason, q in removed:
        print(f"  [{reason}] {q!r}")

    print(f"[V29] Adding {len(examples)} corrected examples (Y1-Y10)")
    all_examples = kept + examples
    print(f"[V29] Total: {len(all_examples)} -> {V29_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V29] V29-only: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V29_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V29] Combined written: {V29_FULL_PATH.name}")


if __name__ == "__main__":
    main()

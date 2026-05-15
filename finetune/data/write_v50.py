"""
V50 training examples (2026-05-15).

Targets:

  ARL_V50_1-4  Rule count queries — no hallucinated number in response

    Root cause: Gemma 4's prefill pattern commits to a count before the
    list_rules tool result arrives.  V49's _ARL_COUNT_16 response hardcodes
    "The system monitors **16** AML rules" — training the model to emit a
    memorized number in the prefill, which can be wrong (e.g., "12").

    Fix: retrain the four count phrasings from V49 to respond with the
    standard insight phrase (no explicit count) — the chart table shows the
    complete list and the user can count rows there.

    V49's ARL_V49_9-12 examples are filtered out; these reinforced the
    prefill-with-number pattern and are superseded by the examples below.

    Phrasings:
      ARL_V50_1 "count the number of rules from list of AML rules"  (exact failure)
      ARL_V50_2 "How many rules does the system monitor?"
      ARL_V50_3 "what is the total number of rules in the system"
      ARL_V50_4 "How many rules are there in the system"

Base: aria_train_combined_v49_full.jsonl (594)
"""

import json, pathlib, re, sys

DATA_DIR      = pathlib.Path(__file__).parent
V49_FULL_PATH = DATA_DIR / "aria_train_combined_v49_full.jsonl"
V50_ONLY_PATH = DATA_DIR / "aria_train_v50.jsonl"
V50_FULL_PATH = DATA_DIR / "aria_train_combined_v50_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from write_v41 import THRESHOLD_SYSTEM          # noqa: E402
from write_v42 import tc                        # noqa: E402
from write_v45 import PC_LIST_H                 # noqa: E402

examples = []

# ===========================================================================
# Standard list_rules insight — no explicit count.
# The chart table is authoritative; the model must not commit a number it
# cannot verify in the prefill (generated before the tool result arrives).
# ===========================================================================

_ARL_RULE_INSIGHT = (
    "Rule performance summary — detailed table shown below.\n\n"
    "Rules with the most false positives: CTR Client (348 FP), "
    "Elder Abuse (341 FP), Detect Excessive Transaction Activity (301 FP)."
)


# ===========================================================================
# ARL_V50_1-4  Rule count queries → standard insight (no hardcoded number)
#
# All four phrasings that misrouted in the session (aria_session_20260515_1221)
# or in local testing.  Each calls list_rules once; the assistant response
# never includes a rule count — it uses the insight phrase above.
# ===========================================================================

# ARL_V50_1: exact local-test phrasing that returned "12 AML detection rules"
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "count the number of rules from list of AML rules"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the active monitoring rules.",
     "tool_calls": [tc("arl_v50_1a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_1a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_RULE_INSIGHT},
]})

# ARL_V50_2: "How many rules does the system monitor?" — exact session phrasing
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How many rules does the system monitor?"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the active monitoring rules.",
     "tool_calls": [tc("arl_v50_2a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_2a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_RULE_INSIGHT},
]})

# ARL_V50_3: "what is the total number of rules in the system" — exact session phrasing
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "what is the total number of rules in the system"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the active monitoring rules.",
     "tool_calls": [tc("arl_v50_3a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_3a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_RULE_INSIGHT},
]})

# ARL_V50_4: "How many rules are there in the system" — exact session phrasing
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "How many rules are there in the system"},
    {"role": "assistant",
     "content": "Calling list_rules to retrieve the active monitoring rules.",
     "tool_calls": [tc("arl_v50_4a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl_v50_4a", "content": PC_LIST_H},
    {"role": "assistant", "content": _ARL_RULE_INSIGHT},
]})


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

_COUNT_PATTERN = re.compile(
    r'The system monitors \*\*\d+\*\* AML rules',
    re.IGNORECASE,
)


def _has_prefill_rule_count(ex):
    """Remove V49's ARL_V49_9-12 style examples where the assistant response
    for a count/list query embeds a hardcoded number ('The system monitors
    **16** AML rules...').  These reinforce prefill-with-number behaviour
    and are superseded by the V50 insight-phrase examples above."""
    msgs = ex["messages"]
    has_list_rules = any(
        call.get("function", {}).get("name") == "list_rules"
        for m in msgs if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    )
    if not has_list_rules:
        return False
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            if _COUNT_PATTERN.search(m.get("content") or ""):
                return True
    return False


# ---------------------------------------------------------------------------
# Combine V49 full + V50 examples and write
# ---------------------------------------------------------------------------

def main():
    with open(V50_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V50] V50-only: {V50_ONLY_PATH.name} ({len(examples)} examples)")

    if V49_FULL_PATH.exists():
        v49_base = []
        with open(V49_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v49_base.append(json.loads(line))
        print(f"[V50] Loaded {len(v49_base)} base examples from {V49_FULL_PATH.name}")

        filtered = v49_base
        before = len(filtered)
        filtered = [ex for ex in filtered if not _has_prefill_rule_count(ex)]
        print(f"[V50] Removed {before - len(filtered)} prefill-rule-count examples (V49 ARL_V49_9-12 superseded)")

        all_examples = filtered + examples
        with open(V50_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V50] Combined: {V50_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V50] WARNING: V49 base not found at {V49_FULL_PATH}")


if __name__ == "__main__":
    main()

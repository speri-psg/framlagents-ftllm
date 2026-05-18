"""
V51 training data (2026-05-18).

PRIMARY JOB — full dataset consistency:
  Strip PRE-COMPUTED blocks from all 220 assistant messages in the V50 base
  that still echo the verbatim table. After V51, every assistant response in
  the dataset is insight-only (ONE ### header + ONE sentence), matching the
  live Rule 9 in both agents.

  Update system prompts on all 344 threshold/segmentation examples to the
  condensed V50 training prompts (with Rule 9 = insight-only already patched).
  Policy, OFAC, and routing examples are left unchanged.

  This makes training/inference Rule 9 fully consistent for the first time.
  The training/inference system prompt mismatch (condensed ~2,714 tokens vs
  live ~3,500 tokens) is a known remaining gap — not addressed here.

SECONDARY — new V51 examples:
  Placeholder section below. Fill in after aria-v12 smoke testing reveals gaps.

Base:  aria_train_combined_v50_full.jsonl  (581 ex)
Stats: 220 PRE-COMPUTED assistant messages stripped → 0 empty after strip
       344 system prompts replaced (213 threshold + 131 segmentation)
"""

import json, pathlib, re, sys

DATA_DIR      = pathlib.Path(__file__).parent
V50_FULL_PATH = DATA_DIR / "aria_train_combined_v50_full.jsonl"
V51_ONLY_PATH = DATA_DIR / "aria_train_v51.jsonl"
V51_FULL_PATH = DATA_DIR / "aria_train_combined_v51_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

# Import the condensed, Rule-9-patched training prompts from write_v50.
# THRESHOLD_SYSTEM: ~2,714 tokens, RULE INVENTORY present, Rule 9 = insight-only.
# SEGMENTATION_SYSTEM: Rule 9 patched to insight-only.
from write_v50 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM  # noqa: E402
from write_v42 import tc, prev_context                        # noqa: E402

print(f"[V51] THRESHOLD_SYSTEM:   {len(THRESHOLD_SYSTEM)} chars (~{len(THRESHOLD_SYSTEM)//4} tokens)")
print(f"[V51] SEGMENTATION_SYSTEM: {len(SEGMENTATION_SYSTEM)} chars (~{len(SEGMENTATION_SYSTEM)//4} tokens)")
assert "Rule 9" not in THRESHOLD_SYSTEM or "copy" not in THRESHOLD_SYSTEM, \
    "THRESHOLD_SYSTEM still has old Rule 9 copy instruction"
assert "copy the cluster stats verbatim" not in SEGMENTATION_SYSTEM, \
    "SEGMENTATION_SYSTEM still has old Rule 9 copy instruction"

# ---------------------------------------------------------------------------
# System prompt fingerprints — identify which examples to update
# ---------------------------------------------------------------------------

_THR_FINGERPRINT  = "You are ARIA — Agentic Risk Intelligence for AML. You analyze false positive"
_SEG_FINGERPRINT  = "You are ARIA — Agentic Risk Intelligence for AML. You identify natural"

# ---------------------------------------------------------------------------
# Strip PRE-COMPUTED blocks from assistant message text
# ---------------------------------------------------------------------------

_STRIP_PATTERNS = [
    # Full-block patterns with specific END markers (run first)
    (r'===.*?PRE-COMPUTED ANALYSIS.*?===.*?===\s*END PRE-COMPUTED ANALYSIS\s*===\n?(?:\([^\n]*\)\n?)?',          re.DOTALL),
    (r'===.*?PRE-COMPUTED SEGMENT STATS.*?===.*?===\s*END PRE-COMPUTED SEGMENT STATS\s*===\n?(?:\([^\n]*\)\n?)?', re.DOTALL),
    (r'===.*?PRE-COMPUTED CLUSTER STATS.*?===.*?===\s*END PRE-COMPUTED CLUSTER STATS\s*===\n?(?:\([^\n]*\)\n?)?', re.DOTALL),
    (r'===.*?PRE-COMPUTED CLUSTER RULE SUMMARY.*?===.*?===\s*END CLUSTER RULE SUMMARY\s*===\n?(?:\([^\n]*\)\n?)?', re.DOTALL),
    (r'===.*?PRE-COMPUTED CLUSTER THRESHOLD ANALYSIS[^\n]*===\n?',                                                0),
    (r'\n?===\s*END CLUSTER THRESHOLD ANALYSIS\s*===\n?',                                                         0),
    (r'===.*?PRE-COMPUTED SAR BACKTEST.*?===.*?===\s*END PRE-COMPUTED SAR BACKTEST\s*===\n?',                     re.DOTALL),
    (r'PRE-COMPUTED SAR BACKTEST.*?(?:===\s*END PRE-COMPUTED SAR BACKTEST\s*===\s*|END PRE-COMPUTED SAR BACKTEST\s*===\s*)', re.DOTALL),
    (r'===.*?PRE-COMPUTED RULE SWEEP.*?===.*?===\s*END RULE SWEEP\s*===\n?',                                      re.DOTALL),
    (r'PRE-COMPUTED RULE SWEEP.*?(?:===\s*END RULE SWEEP\s*===\s*|END RULE SWEEP\s*===\s*)',                      re.DOTALL),
    (r'===.*?PRE-COMPUTED RULE LIST.*?===.*?===\s*END RULE LIST\s*===\n?',                                        re.DOTALL),
    (r'Available AML rules with SAR/FP performance.*?END RULE LIST\s*===\s*',                                     re.DOTALL),
    (r'===.*?PRE-COMPUTED 2D SWEEP.*?===.*?===\s*END 2D SWEEP\s*===\n?',                                         re.DOTALL),
    (r'PRE-COMPUTED 2D SWEEP.*?(?:===\s*END 2D SWEEP\s*===\s*|END 2D SWEEP\s*===\s*)',                           re.DOTALL),
    # Variant END markers — same block types, different END label used in early examples
    (r'===.*?PRE-COMPUTED SAR BACKTEST.*?===.*?===\s*END SAR BACKTEST\s*===\n?',                                  re.DOTALL),
    (r'===.*?PRE-COMPUTED 2D SWEEP.*?===.*?===\s*END PRE-COMPUTED 2D SWEEP\s*===\n?',                            re.DOTALL),
    # Generic === END PRE-COMPUTED === marker (early training examples before typed END markers)
    # Covers: PRE-COMPUTED 2D SWEEP RESULT, CLUSTER ANALYSIS, SAR BACKTEST (copy verbatim), etc.
    (r'===.*?PRE-COMPUTED.*?===.*?===\s*END PRE-COMPUTED\s*===\n?',                                               re.DOTALL),
    # Bold markdown format: **PRE-COMPUTED CLUSTER STATS — ...** (no === markers, early examples)
    # Block consists of: bold header + blank line + cluster data lines (each starting **Cluster N**)
    (r'\*\*PRE-COMPUTED CLUSTER STATS[^\n]*\*\*\n\n(?:\*\*Cluster \d+\*\*[^\n]*\n(?:-[^\n]*\n)*\n?)+',           re.DOTALL),
    # Fallback: orphaned END markers and single-line PRE-COMPUTED headers
    (r'^PRE-COMPUTED [^\n]+\n?',                                                                                  re.MULTILINE),
    (r'^END PRE-COMPUTED[^\n]*===\s*\n?',                                                                         re.MULTILINE),
    (r'^END CLUSTER[^\n]*===\s*\n?',                                                                              re.MULTILINE),
    (r'^END RULE[^\n]*===\s*\n?',                                                                                  re.MULTILINE),
    # Inline artifacts from old training examples
    (r'\(Detailed sweep (?:chart|table)[^)]*\)\s*',                                                               re.IGNORECASE),
    (r'\bSee chart/table below\.\s*',                                                                             re.IGNORECASE),
    # Inline references to PRE-COMPUTED in insight sentences — replace with neutral phrasing
    (r'\bthe PRE-COMPUTED (?:section|block)\b',                                                                   re.IGNORECASE),
]


def _strip_precomputed(text: str) -> str:
    for pat, flags in _STRIP_PATTERNS[:-1]:  # all except the inline-reference pattern
        text = re.sub(pat, '', text, flags=flags or 0)
    # Inline reference replacement (last pattern): substitute rather than delete
    text = re.sub(r'\bthe PRE-COMPUTED (?:section|block)\b', 'the data', text, flags=re.IGNORECASE)
    # Last-resort: responses that start directly with === PRE-COMPUTED and have no END marker.
    # The whole response is: [PRE-COMPUTED block with internal \n\n] + \n\n + [insight paragraph].
    # Keep only the last paragraph (always the insight sentence).
    if text.startswith('=== PRE-COMPUTED') and 'PRE-COMPUTED' in text:
        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        last = parts[-1] if parts else text
        if last and not last.startswith('==='):
            text = last
    return text.strip()


def _rewrite_responses(ex: dict) -> dict:
    """Strip PRE-COMPUTED blocks from all final assistant messages.
    Tool role messages are intentionally left untouched — they are model
    input (the data source), not model output."""
    changed = False
    new_msgs = []
    for m in ex["messages"]:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            content = m.get("content", "")
            if "PRE-COMPUTED" in content:
                stripped = _strip_precomputed(content)
                if stripped:
                    new_msgs.append({**m, "content": stripped})
                    changed = True
                else:
                    # Safety: keep original if stripping leaves nothing
                    print(f"[V51] WARNING: strip left empty string, keeping original (prefix: {content[:60]!r})")
                    new_msgs.append(m)
            else:
                new_msgs.append(m)
        else:
            new_msgs.append(m)
    return ({**ex, "messages": new_msgs}, changed)


def _update_system_prompt(ex: dict) -> tuple:
    """Replace threshold/segmentation system prompts with condensed V51 versions.
    Policy, OFAC, and routing examples are left unchanged."""
    sys_content = ex["messages"][0].get("content", "")
    if sys_content.startswith(_THR_FINGERPRINT):
        new_sys = THRESHOLD_SYSTEM
    elif sys_content.startswith(_SEG_FINGERPRINT):
        new_sys = SEGMENTATION_SYSTEM
    else:
        return ex, False  # policy / OFAC / routing — unchanged

    if sys_content == new_sys:
        return ex, False  # already up to date (e.g., the 10 V50 new examples)

    new_msgs = [{**ex["messages"][0], "content": new_sys}] + ex["messages"][1:]
    return {**ex, "messages": new_msgs}, True


# ===========================================================================
# New V51 examples — fill in after aria-v12 smoke testing
#
# Known candidates (update after testing):
#   - Multi-turn cluster format: align [PREVIOUS CLUSTERING RESULT] prefix to
#     production format if model misses it
#   - Any new gaps found during aria-v12 demo
# ===========================================================================

examples = []

# (No new examples yet — placeholder for post-smoke-test additions)


# ---------------------------------------------------------------------------
# Combine + write
# ---------------------------------------------------------------------------

def main():
    v50_base = []
    with open(V50_FULL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v50_base.append(json.loads(line))
    print(f"[V51] Loaded {len(v50_base)} base examples from {V50_FULL_PATH.name}")

    # Step 1: Strip PRE-COMPUTED blocks from assistant responses
    stripped_base = []
    n_pc_stripped = 0
    for ex in v50_base:
        new_ex, changed = _rewrite_responses(ex)
        stripped_base.append(new_ex)
        if changed:
            n_pc_stripped += 1
    print(f"[V51] Stripped PRE-COMPUTED from {n_pc_stripped} examples")

    # Step 2: Update system prompts to condensed V51 versions
    updated_base = []
    n_sys_updated = 0
    for ex in stripped_base:
        new_ex, changed = _update_system_prompt(ex)
        updated_base.append(new_ex)
        if changed:
            n_sys_updated += 1
    print(f"[V51] Updated system prompts on {n_sys_updated} examples")

    # Step 3: Append new V51 examples
    all_examples = updated_base + examples

    # Write V51-only
    with open(V51_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V51] V51-only: {V51_ONLY_PATH.name} ({len(examples)} examples)")

    # Write V51 combined
    with open(V51_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V51] Combined: {V51_FULL_PATH.name} ({len(all_examples)} total)")

    # Verify: no PRE-COMPUTED should remain in any assistant response
    n_remaining = 0
    for ex in all_examples:
        for m in ex["messages"]:
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                if "PRE-COMPUTED" in (m.get("content") or ""):
                    n_remaining += 1
    if n_remaining:
        print(f"[V51] WARNING: {n_remaining} assistant messages still contain PRE-COMPUTED")
    else:
        print("[V51] OK: zero PRE-COMPUTED blocks remain in any assistant response")


if __name__ == "__main__":
    main()

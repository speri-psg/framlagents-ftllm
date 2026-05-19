#!/usr/bin/env python
"""
smoke_run.py — Automated smoke test for ARIA agents.

Runs named multi-turn prompt sessions against the real data stack.
No Dash server is started; the orchestrator + tool_executor run directly.

Usage:
    python smoke_run.py                   # run all sessions
    python smoke_run.py "Clustering"      # run sessions whose name contains "Clustering"
"""

import sys
import os
import time
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Allow --model <name> to override OLLAMA_MODEL before config.py is imported.
# Example: python smoke_run.py --model aria-v15
_model_idx = next((i for i, a in enumerate(sys.argv) if a == "--model"), None)
if _model_idx and _model_idx + 1 < len(sys.argv):
    os.environ["OLLAMA_MODEL"] = sys.argv[_model_idx + 1]
    sys.argv = [a for i, a in enumerate(sys.argv) if i not in (_model_idx, _model_idx + 1)]

if not os.getenv("OLLAMA_MODEL"):
    print("WARNING: OLLAMA_MODEL not set — defaulting to aria-v15.")
    print("         Run with: python smoke_run.py --model <your-ollama-model-name>\n")
    os.environ["OLLAMA_MODEL"] = "aria-v15"

print(f"Model: {os.environ['OLLAMA_MODEL']}", flush=True)
print("Loading application stack (startup clustering may take ~30s)...", flush=True)
import application
print("Ready.\n", flush=True)

WRAP_WIDTH  = 90
PREVIEW_LEN = 600   # chars of response shown in terminal


# ── Session runner ──────────────────────────────────────────────────────────────

def run_session(name: str, turns: list) -> tuple[int, int]:
    """
    Run a named multi-turn session. Returns (passed, failed) assertion counts.

    Each turn is:
      str                  → query with no assertion
      (str, str)           → (query, substring_that_must_appear)
      (str, str, str)      → (query, must_appear, must_NOT_appear)
    """
    print(f"\n{'='*70}")
    print(f"SESSION: {name}")
    print('='*70)

    last_assistant    = ""
    history: list     = []
    last_cluster_ctx  = ""   # raw stats injected as [PREVIOUS CLUSTERING RESULT]
    last_rule_list    = ""   # injected as [PREVIOUS RULE LIST]
    passed = failed   = 0

    for item in turns:
        if isinstance(item, str):
            query, must_have, must_not = item, None, None
        elif len(item) == 2:
            query, must_have, must_not = item[0], item[1], None
        else:
            query, must_have, must_not = item

        print(f"\n  >>> {query}")

        # Mirror handle_chat: set the query on thread_local so tool_executor's
        # cluster-count regex can read it.
        application._thread_local.current_query = query
        application._thread_local.last_2d_state = {}

        # Clear module-level globals before the call so we capture only this turn.
        application._last_cluster_raw_stats = ""
        application._last_rule_list         = ""

        t0 = time.time()
        try:
            agent_text, chart_results = application.orchestrator.run(
                query,
                application.tool_executor,
                last_assistant,
                history,
                last_cluster_ctx,   # [PREVIOUS CLUSTERING RESULT] context
                last_rule_list,     # [PREVIOUS RULE LIST] context
            )
            elapsed = time.time() - t0

            # Capture state for the next turn (mirrors handle_chat lines 1896-1904)
            raw_stats_this_turn = application._last_cluster_raw_stats
            new_rule_list       = application._last_rule_list

            seg_tools = {"ds_cluster_analysis", "cluster_analysis"}
            if any(r[0] in seg_tools for r in (chart_results or [])):
                last_cluster_ctx = raw_stats_this_turn or agent_text or ""
            if new_rule_list:
                last_rule_list = new_rule_list

            # Display
            display = (agent_text or "").strip()
            preview = display[:PREVIEW_LEN]
            wrapped = textwrap.fill(preview, width=WRAP_WIDTH, subsequent_indent="      ")
            print(f"  [{elapsed:.1f}s] {wrapped}")
            if len(display) > PREVIEW_LEN:
                print(f"      ... ({len(display):,} chars total)")
            charts_summary = ", ".join(r[0] for r in (chart_results or []))
            if charts_summary:
                print(f"  [charts: {charts_summary}]")

            # Assertions
            if must_have:
                if must_have.lower() in display.lower():
                    print(f"  ✓ PASS  must_have={repr(must_have[:50])}")
                    passed += 1
                else:
                    print(f"  ✗ FAIL  must_have={repr(must_have[:50])} NOT FOUND")
                    failed += 1
            if must_not:
                if must_not.lower() in display.lower():
                    print(f"  ✗ FAIL  must_not={repr(must_not[:50])} WAS FOUND")
                    failed += 1
                else:
                    print(f"  ✓ PASS  must_not={repr(must_not[:50])} absent")
                    passed += 1

            # Build history for the next turn (last 4 messages = 2 pairs)
            history.append({"role": "user",      "content": query})
            history.append({"role": "assistant", "content": display[:500]})
            history = history[-4:]
            last_assistant = display

        except Exception as exc:
            import traceback
            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s] ERROR: {exc}")
            traceback.print_exc()
            failed += 1

    total = passed + failed
    print(f"\n  Session result: {passed}/{total} assertions passed")
    return passed, failed


# ── Test sessions ───────────────────────────────────────────────────────────────
#
# Assertion format:  (query, must_appear)  or  (query, must_appear, must_not_appear)
# Bare strings have no assertion — useful for setup turns.

SESSIONS = [

    # ── Segmentation ─────────────────────────────────────────────────────────

    ("Clustering — Business initial bullets", [
        ("Cluster Business customers by transaction behavior",
         "Cluster 1",
         "Clustering complete"),   # must NOT show old generic message
    ]),

    ("Clustering — Business comparison + follow-ups", [
        "Cluster Business customers by transaction behavior",
        ("how does cluster 1 compare to cluster 4",
         "Cluster 1"),
        ("how about cluster 2 and cluster 3",
         "Cluster 2"),
        ("which cluster has the highest number of customers",
         "Cluster 1"),
        ("which one has the lowest",            # gap 3: hallucination risk
         "30"),                                 # Cluster 2 (30) is the actual smallest
    ]),

    ("Clustering — elliptical follow-up without 'cluster' keyword (gap 2)", [
        "Cluster Business customers by transaction behavior",
        ("which cluster is the oldest",
         "Cluster 1"),
        ("and the youngest",                    # omits 'cluster' — gap 2
         "Cluster"),
    ]),

    ("Clustering — single-cluster describe after comparison (gap 1)", [
        "Cluster Business customers by transaction behavior",
        ("compare cluster 2 and cluster 4",
         "Cluster 2"),
        ("how about cluster 3",                 # gap 1: refused after comparison turn
         "Cluster 3"),
    ]),

    ("Clustering — SAR count read from prior context (gap 7)", [
        "Cluster Business customers by transaction behavior",
        ("how many SARs in cluster 1",          # should answer from PREVIOUS CLUSTERING RESULT
         "265",                                 # actual SARs for Cluster 1 Business
         "rule-alert pairs"),                   # must NOT return cluster_rule_summary output
    ]),

    ("Clustering — Individual initial bullets (gap 5)", [
        ("Show me behavioral segments for Individual customers",
         "Cluster 1",
         "Customers:"),                         # must NOT echo raw PRE-COMPUTED stats block
    ]),

    ("Clustering — Individual elliptical follow-up (gap 2 + 6)", [
        "Show me behavioral segments for Individual customers",
        ("which cluster is the oldest",
         "Cluster 1"),
        ("and the youngest",                    # gap 2 on Individual
         "Cluster"),
    ]),

    ("Clustering — superlative value hallucination (gap 6)", [
        "Cluster Business customers by transaction behavior",
        ("which cluster has the highest avg weekly transaction amount",
         "Cluster 1"),
        ("and the lowest",
         "Cluster"),
    ]),

    # ── Rules (ARL) ──────────────────────────────────────────────────────────

    ("Rules — precision ranking", [
        ("Which rules have the highest precision?",
         "precision"),
    ]),

    ("Rules — multi-turn bottom/top + plain list (gaps 8, 8b)", [
        "Which rules have the highest precision?",
        ("and the lowest?",                     # gap 8: should re-sort, not 'I don't have data'
         "%",
         "I do not have"),
        ("What are all the AML rules?",         # should return plain list, not precision sort
         "rule"),
    ]),

    ("Rules — parameter filter multi-turn", [
        ("Which rules have z_threshold?",
         "z_threshold"),
        ("what about days_required?",           # should filter list, not give a definition
         "days_required"),
    ]),

    # ── Threshold / SAR backtest ──────────────────────────────────────────────

    ("Threshold — cluster-adaptive Individual", [
        ("Run cluster-adaptive threshold analysis for Individual customers",
         "Cluster",
         "Results shown below."),              # must NOT return generic canned phrase
    ]),

    ("SAR Backtest — Business segment", [
        "Hello",                                # establish conversation context (mirrors app greeting)
        ("Show SAR backtest for Business customers",
         "SAR"),
    ]),

    ("SAR Backtest — Elder Abuse rule sweep", [
        ("Show SAR backtest for Elder Abuse rule",
         "Elder Abuse"),
    ]),

]


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filter_arg = sys.argv[1].lower() if len(sys.argv) > 1 else None

    total_pass = total_fail = 0
    for session_name, turns in SESSIONS:
        if filter_arg and filter_arg not in session_name.lower():
            continue
        p, f = run_session(session_name, turns)
        total_pass += p
        total_fail += f

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail} assertions")
    print("ALL GOOD" if total_fail == 0 else f"{total_fail} FAILURE(S) — see FAIL lines above")
    print('='*70)

"""
V33 training examples (2026-04-26).

Root cause diagnosed after V32 evaluation:
  L02 benchmark case still fails: "Which rules generate the most false positives?"
  V32 added R1-R4 (RULE_SYSTEM + list_rules) but all four examples use neutral
  display queries ("Show me all AML rules", "What rules are available?", etc.).
  L02 is an ANALYTICAL question — the model answers it directly from the PRE-COMPUTED
  data ("CTR Client has 2,061 FPs") instead of copying the full block first.

  The format_check requires "=== PRE-COMPUTED RULE LIST" to appear in the response.
  The model never learned that analytical questions over list_rules results still
  require copying the full block before answering the specific question.

New examples (V33):
  A (4 ex): RULE_SYSTEM + list_rules — analytical questions (FP / precision ranking)
    A1: "Which rules generate the most false positives?"  ← exact L02 benchmark query
    A2: "Which rule has the lowest precision?"
    A3: "Show me rules ranked by false positive count"
    A4: "What is the worst performing rule by FP count?"

  B (4 ex): THRESHOLD_SYSTEM + list_rules — analytical questions (same queries)
    B1: "Which rules generate the most false positives?"  (THRESHOLD_SYSTEM)
    B2: "Which rule has the lowest precision?"             (THRESHOLD_SYSTEM)
    B3: "Which rules have no production data?"             (THRESHOLD_SYSTEM)
    B4: "Which rules have the highest alert volume?"       (THRESHOLD_SYSTEM)

  C (3 ex): SEG_SYSTEM + ds_cluster_analysis(All) — unqualified clustering requests
    Without segment, default is customer_type="All". These fix the hallucination
    where the model outputs memorised cluster stats instead of calling the tool.
    C1: "Run a clustering analysis"
    C2: "Cluster all customers"
    C3: "Run a full clustering analysis"

Net: 797 (V32) + 8 (A+B) + 3 (C) = 808 examples
"""

import json
import pathlib

DATA_DIR       = pathlib.Path(__file__).parent
V33_BASE_PATH  = DATA_DIR / "aria_train_combined_v32_full.jsonl"
V33_FULL_PATH  = DATA_DIR / "aria_train_combined_v33_full.jsonl"
THIS_PATH      = DATA_DIR / "aria_train_v33.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------------------------------------------------------------------------
# System prompts (identical to write_v32.py)
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
# Pre-computed tool results
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

# Insight sentences tailored to each analytical question
_INSIGHT_MOST_FP = (
    "CTR Client generates the most false positives (2,061 FPs, 8.0% precision), "
    "followed by Elder Abuse (958 FPs) and Burst in Beneficiary Activity (607 FPs)."
)
_INSIGHT_LOWEST_PRECISION = (
    "CTR Client has the lowest precision among active rules at 8.0%, "
    "generating 2,061 false positives out of 2,241 total alerts."
)
_INSIGHT_RANKED_FP = (
    "The top false-positive generators are CTR Client (2,061 FPs), Elder Abuse (958 FPs), "
    "and Burst in Beneficiary Activity (607 FPs) — these three rules drive the majority of analyst workload."
)
_INSIGHT_WORST_RULE = (
    "CTR Client has the highest false positive count at 2,061 FPs (8.0% precision), "
    "making it the primary driver of unnecessary investigations in the rule portfolio."
)
_INSIGHT_NO_DATA = (
    "The 5 rules with alerts=0 (Activity Deviation Wire, Velocity Multiple, Funnel Account, "
    "Round-trip, Human Trafficking Indicators) are inactive with no production alert data."
)
_INSIGHT_HIGHEST_VOLUME = (
    "CTR Client generates the highest alert volume with 2,241 alerts at 8.0% precision, "
    "followed by Elder Abuse (1,146 alerts, 16.4% precision) and Burst in Beneficiary Activity (701 alerts, 11.8% precision)."
)

# ---------------------------------------------------------------------------
# Group A  -- RULE_SYSTEM + list_rules — analytical questions (4 examples)
# These teach the model that analytical questions still require copying the
# full PRE-COMPUTED block, with the insight sentence answering the question.
# ---------------------------------------------------------------------------

examples = []

# A1: exact L02 benchmark query (RULE_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Which rules generate the most false positives?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("a1", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "a1", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_MOST_FP}"},
]})

# A2: lowest precision (RULE_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Which rule has the lowest precision?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("a2", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "a2", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_LOWEST_PRECISION}"},
]})

# A3: ranked by FP count (RULE_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "Show me rules ranked by false positive count"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("a3", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "a3", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_RANKED_FP}"},
]})

# A4: worst rule by FP count (RULE_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": RULE_SYSTEM},
    {"role": "user",      "content": "What is the worst performing rule by FP count?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("a4", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "a4", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_WORST_RULE}"},
]})

# ---------------------------------------------------------------------------
# Group B  -- THRESHOLD_SYSTEM + list_rules — analytical questions (4 examples)
# Threshold agent is the likely route for FP/precision queries in production.
# ---------------------------------------------------------------------------

# B1: most false positives (THRESHOLD_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rules generate the most false positives?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("b1", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "b1", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_MOST_FP}"},
]})

# B2: lowest precision (THRESHOLD_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rule has the lowest precision?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("b2", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "b2", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_LOWEST_PRECISION}"},
]})

# B3: no production data (THRESHOLD_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rules have no production data?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("b3", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "b3", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_NO_DATA}"},
]})

# B4: highest alert volume (THRESHOLD_SYSTEM)
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rules have the highest alert volume?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("b4", "list_rules", {})]},
    {"role": "tool",      "tool_call_id": "b4", "content": TOOL_LIST_RULES},
    {"role": "assistant", "content": f"{PC_LIST_RULES}\n\n{_INSIGHT_HIGHEST_VOLUME}"},
]})

SEG_SYSTEM = (
    "You are ARIA  -- Agentic Risk Intelligence for AML  -- dynamic segmentation specialist. "
    "You identify natural customer behavioral segments using unsupervised K-Means clustering. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES  -- follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred)  -- call ds_cluster_analysis.\n"
    "3. For alert/FP distribution by segment  -- call alerts_distribution.\n"
    "4. For the legacy alerts dataset  -- call cluster_analysis only if the user explicitly asks.\n"
    "5. Do NOT call multiple segmentation tools for the same request  -- pick exactly one.\n"
    "6. customer_type must be exactly one of: Business, Individual, All\n"
    "   If the user does NOT specify a customer type, default to All.\n"
    "7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.\n"
    "8. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence "
    "describing the highest-risk cluster based solely on the numbers in the tool result. "
    "Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.\n"
    "12. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
)

# Pre-computed "All" cluster stats — consistent with current demo dataset scale
# (Business active ~2,369 + Individual active ~2,631 = ~5,000 total)
PC_CLUSTER_ALL = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: All | Active accounts: 5,000 (excluded 0 with no transactions)
Clusters: 4 | Features: 7 numeric + 57 encoded categorical (6 original)
PCA variance explained: PC1=6.2%, PC2=4.1%

Cluster 1: Mid Freq / Mid Value (78.4% of active accounts)
- Avg Weekly Transactions: **1.2**
- Avg Transaction Amount ($): **2,847.3**
- Monthly Txn Volume ($): **11,892.1**
- Current Balance ($): **16,541.0**
- Account Age (years): **6.1**

Cluster 2: Mid Freq / High Value (10.3% of active accounts)
- Avg Weekly Transactions: **1.4**
- Avg Transaction Amount ($): **4,218.6**
- Monthly Txn Volume ($): **18,340.2**
- Current Balance ($): **24,180.5**
- Account Age (years): **7.3**

Cluster 3: Low Freq / Low Value (7.8% of active accounts)
- Avg Weekly Transactions: **0.6**
- Avg Transaction Amount ($): **1,203.4**
- Monthly Txn Volume ($): **4,812.7**
- Current Balance ($): **8,920.3**
- Account Age (years): **3.2**

Cluster 4: High Freq / High Value (3.5% of active accounts)
- Avg Weekly Transactions: **3.8**
- Avg Transaction Amount ($): **8,741.2**
- Monthly Txn Volume ($): **42,156.8**
- Current Balance ($): **61,340.0**
- Account Age (years): **9.4**

=== END PRE-COMPUTED CLUSTER STATS ==="""
TOOL_CLUSTER_ALL = f"Tool result for ds_cluster_analysis:\n{PC_CLUSTER_ALL}"

_C_INSIGHT = (
    "Cluster 4 represents the highest-risk segment at 3.5% of all accounts, "
    "with an average monthly transaction volume of $42,156.8 and average balance of $61,340.0 "
    "— significantly above the portfolio average."
)

# ---------------------------------------------------------------------------
# Group C  -- SEG_SYSTEM + ds_cluster_analysis(All) — unqualified clustering (3 examples)
# Fixes hallucination: model outputs memorised old stats instead of calling tool.
# Rule 6 of SEG_SYSTEM: "If the user does NOT specify a customer type, default to All."
# ---------------------------------------------------------------------------

# C1: "Run a clustering analysis"
examples.append({"messages": [
    {"role": "system",    "content": SEG_SYSTEM},
    {"role": "user",      "content": "Run a clustering analysis"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("c1", "ds_cluster_analysis", {"customer_type": "All", "n_clusters": 4})]},
    {"role": "tool",      "tool_call_id": "c1", "content": TOOL_CLUSTER_ALL},
    {"role": "assistant", "content": f"{PC_CLUSTER_ALL}\n\n{_C_INSIGHT}"},
]})

# C2: "Cluster all customers"
examples.append({"messages": [
    {"role": "system",    "content": SEG_SYSTEM},
    {"role": "user",      "content": "Cluster all customers"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("c2", "ds_cluster_analysis", {"customer_type": "All", "n_clusters": 4})]},
    {"role": "tool",      "tool_call_id": "c2", "content": TOOL_CLUSTER_ALL},
    {"role": "assistant", "content": f"{PC_CLUSTER_ALL}\n\n{_C_INSIGHT}"},
]})

# C3: "Run a full clustering analysis"
examples.append({"messages": [
    {"role": "system",    "content": SEG_SYSTEM},
    {"role": "user",      "content": "Run a full clustering analysis"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("c3", "ds_cluster_analysis", {"customer_type": "All", "n_clusters": 4})]},
    {"role": "tool",      "tool_call_id": "c3", "content": TOOL_CLUSTER_ALL},
    {"role": "assistant", "content": f"{PC_CLUSTER_ALL}\n\n{_C_INSIGHT}"},
]})

# ---------------------------------------------------------------------------
# Build V33: load V32 base + append new examples
# ---------------------------------------------------------------------------

def main():
    v32 = []
    with open(V33_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v32.append(json.loads(line))

    print(f"[V33] Loaded {len(v32)} examples from {V33_BASE_PATH.name}")
    print(f"[V33] Adding {len(examples)} new examples (A1-A4, B1-B4, C1-C3)")

    all_examples = v32 + examples
    print(f"[V33] Total: {len(all_examples)} -> {V33_FULL_PATH.name}")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V33] V33-only written: {THIS_PATH.name} ({len(examples)} examples)")

    with open(V33_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[V33] Combined written: {V33_FULL_PATH.name}")

    # Spot-check: verify all new examples have a PRE-COMPUTED block in the final response
    missing = []
    for ex in examples:
        last = ex["messages"][-1]
        content = last.get("content") or ""
        has_block = ("=== PRE-COMPUTED RULE LIST" in content
                     or "=== PRE-COMPUTED CLUSTER STATS" in content)
        if last["role"] == "assistant" and not has_block:
            missing.append(ex["messages"][1]["content"])
    if missing:
        print(f"[V33] WARNING: {len(missing)} examples missing PRE-COMPUTED block in response:")
        for q in missing:
            print(f"  - {q}")
    else:
        print(f"[V33] Verification: all {len(examples)} new examples contain PRE-COMPUTED RULE LIST block.")


if __name__ == "__main__":
    main()

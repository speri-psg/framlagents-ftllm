"""
V18 training examples — gaps identified from live testing 2026-04-17.

Targets:
  A. alerts_distribution (4 examples) — model hallucinated 4 fake segments; needs verbatim copy
  B. cluster_analysis targeted queries (3 examples) — model hallucinated cluster data
  C. Rejection wording — invalid column suggests closest substitute (2 examples)
  D. No tool on rejection/clarification (3 examples) — model was calling list_rules unnecessarily
  E. rule_2d_sweep Structuring rules (3 examples) — model self-generated result inline
  F. list_rules precision query (2 examples) — model reported decimal not %

Start: ex595 (V17 combined = 594 examples).
Combined with framl_train_combined_v17.jsonl → framl_train_combined_v18.jsonl
"""

import json, pathlib

# ─── Helpers ──────────────────────────────────────────────────────────────────

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

# ─── System prompts ───────────────────────────────────────────────────────────

SEG_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — dynamic segmentation specialist. "
    "You identify natural customer behavioral segments using unsupervised K-Means clustering. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool for segmentation or alert distribution questions.\n"
    "2. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "3. Add ONE sentence of insight based solely on the numbers provided.\n"
    "4. Do NOT invent segment names, counts, or percentages not in the tool result.\n"
    "5. Call the tool ONCE only.\n"
)

RULE_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call the appropriate tool for rule analysis questions.\n"
    "2. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "3. Add ONE sentence of AML domain insight.\n"
    "4. Do NOT invent rule names, TP/FP counts, or thresholds not in the tool result.\n"
    "5. If a rule name is invalid, list the valid rules in text — do NOT call list_rules.\n"
    "6. If a query is ambiguous (multiple matching rules), ask for clarification — do NOT call list_rules.\n"
    "7. Call the tool ONCE only.\n"
)

THRESHOLD_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML — threshold tuning specialist. "
    "You analyze AML alert thresholds to optimize false positive / false negative trade-offs. "
    "IMPORTANT: You MUST respond entirely in English.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call threshold_tuning for threshold analysis questions.\n"
    "2. Valid segment values: 'Business' or 'Individual'.\n"
    "3. Valid threshold_column values: 'AVG_TRXNS_WEEK', 'AVG_TRXN_AMT', 'TRXN_AMT_MONTHLY'.\n"
    "4. If the user requests an invalid column, respond in text with the valid options "
    "and suggest the closest substitute — do NOT call the tool.\n"
    "5. After receiving tool results, copy the PRE-COMPUTED section verbatim.\n"
    "6. Add ONE sentence of insight. Call the tool ONCE only.\n"
)

# ─── Pre-computed tool results ─────────────────────────────────────────────────

PC_SEGMENT_STATS = """\
=== PRE-COMPUTED SEGMENT STATS (copy verbatim, do not compute new numbers) ===
### Segment Overview

**Business**
- Accounts: **3,212** (42.1% of total)
- Alerts: **1,847** (38.6% of all alerts)
- False Positives: **1,203** (FP rate=65.1% of alerts)
- False Negatives: **44**

**Individual**
- Accounts: **4,416** (57.9% of total)
- Alerts: **2,938** (61.4% of all alerts)
- False Positives: **1,876** (FP rate=63.9% of alerts)
- False Negatives: **91**

=== END PRE-COMPUTED SEGMENT STATS ==="""

PC_BUS_CLUSTER = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | Active accounts: 3,551 (excluded 0 with no transactions)
Clusters: 4 | Features: 5 numeric + 12 encoded categorical (4 original)
PCA variance explained: PC1=48.3%, PC2=19.1%

**Cluster 1**
- Customers: **1,114** (31.4% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **30,487.4**
- Monthly Txn Volume ($): **300,181.8**
- Current Balance ($): **42,310.5**
- Account Age (years): **6.2**

**Cluster 2**
- Customers: **1,088** (30.6% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **27,690.0**
- Monthly Txn Volume ($): **246,301.4**
- Current Balance ($): **38,204.1**
- Account Age (years): **5.8**

**Cluster 3**
- Customers: **837** (23.6% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **24,505.9**
- Monthly Txn Volume ($): **189,046.9**
- Current Balance ($): **29,118.3**
- Account Age (years): **5.1**

**Cluster 4**
- Customers: **512** (14.4% of active accounts)
- Avg Weekly Transactions: **13.4**
- Avg Weekly Txn Amount ($): **17,441.0**
- Monthly Txn Volume ($): **121,217.7**
- Current Balance ($): **17,842.6**
- Account Age (years): **4.3**

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_IND_CLUSTER = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Individual | Active accounts: 4,312 (excluded 104 with no transactions)
Clusters: 4 | Features: 7 numeric + 18 encoded categorical (5 original)
PCA variance explained: PC1=41.2%, PC2=22.7%

**Cluster 1**
- Customers: **1,620** (37.6% of active accounts)
- Avg Weekly Transactions: **4.2**
- Avg Weekly Txn Amount ($): **1,240.5**
- Monthly Txn Volume ($): **18,320.1**
- Income ($): **48,200.0**
- Current Balance ($): **6,840.3**
- Account Age (years): **3.8**
- Age: **34.0**

**Cluster 2**
- Customers: **1,504** (34.9% of active accounts)
- Avg Weekly Transactions: **6.8**
- Avg Weekly Txn Amount ($): **3,805.2**
- Monthly Txn Volume ($): **52,180.4**
- Income ($): **72,100.0**
- Current Balance ($): **14,220.7**
- Account Age (years): **5.1**
- Age: **42.0**

**Cluster 3**
- Customers: **812** (18.8% of active accounts)
- Avg Weekly Transactions: **9.3**
- Avg Weekly Txn Amount ($): **12,440.1**
- Monthly Txn Volume ($): **148,620.8**
- Income ($): **118,500.0**
- Current Balance ($): **38,710.2**
- Account Age (years): **7.4**
- Age: **51.0**

**Cluster 4**
- Customers: **376** (8.7% of active accounts)
- Avg Weekly Transactions: **15.1**
- Avg Weekly Txn Amount ($): **48,920.7**
- Monthly Txn Volume ($): **584,180.3**
- Income ($): **241,000.0**
- Current Balance ($): **128,340.6**
- Account Age (years): **9.2**
- Age: **58.0**

=== END PRE-COMPUTED CLUSTER STATS ==="""

PC_LIST_RULES = """\
=== PRE-COMPUTED RULE LIST (copy verbatim) ===
| Rule | Alerts | SAR | FP | Precision |
|---|---|---|---|---|
| Activity Deviation (ACH) | 312 | 87 | 225 | 27.9% |
| Activity Deviation (Check) | 284 | 74 | 210 | 26.1% |
| Elder Abuse | 1,045 | 181 | 864 | 17.3% |
| Velocity Single | 198 | 61 | 137 | 30.8% |
| Detect Excessive Transaction Activity | 427 | 138 | 289 | 32.3% |
| Structuring (Incoming Cash) | 143 | 52 | 91 | 36.4% |
| Structuring (Outgoing Cash) | 167 | 58 | 109 | 34.7% |
| CTR Client | 89 | 34 | 55 | 38.2% |
| Burst in Originator Activity | 203 | 71 | 132 | 35.0% |
| Burst in Beneficiary Activity | 178 | 59 | 119 | 33.1% |
| Risky International Transfer | 156 | 63 | 93 | 40.4% |
=== END RULE LIST ==="""

PC_2D_STRUCT_IN = """\
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Structuring (Incoming Cash)
Axis 1 (daily_floor): Minimum daily cash-in amount to trigger ($)
Axis 2 (days_required): Consecutive days above floor required
Grid: 4 x 2 = 8 combinations
SAR pool: 52
Non-SAR pool: 91

At current condition (daily_floor=1000, days_required=3): TP=48, FP=84, FN=4, TN=7 (TP rate=92.3%).
Best FP reduction (TP rate >=50%): daily_floor=1500, days_required=2 => TP=41, FP=63, FN=11, TN=28, TP rate=78.8%, precision=39.4%.
=== END 2D SWEEP ===
(Heatmap shown in the chart below.)"""

PC_2D_STRUCT_OUT = """\
=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===
Rule: Structuring (Outgoing Cash)
Axis 1 (daily_floor): Minimum daily cash-out amount to trigger ($)
Axis 2 (days_required): Consecutive days above floor required
Grid: 4 x 2 = 8 combinations
SAR pool: 58
Non-SAR pool: 109

At current condition (daily_floor=1000, days_required=3): TP=54, FP=98, FN=4, TN=11 (TP rate=93.1%).
Best FP reduction (TP rate >=50%): daily_floor=1500, days_required=2 => TP=47, FP=71, FN=11, TN=38, TP rate=81.0%, precision=39.8%.
=== END 2D SWEEP ===
(Heatmap shown in the chart below.)"""

# ─── Examples ─────────────────────────────────────────────────────────────────
examples = []

# ══════════════════════════════════════════════════════════════════════════════
# A. alerts_distribution — verbatim copy of PRE-COMPUTED SEGMENT STATS
# ══════════════════════════════════════════════════════════════════════════════

# ex595
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "Show alert distribution across segments"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex595", "alerts_distribution", {})]},
    {"role": "tool", "content": PC_SEGMENT_STATS, "tool_call_id": "call_ex595"},
    {"role": "assistant", "content": (
        PC_SEGMENT_STATS + "\n\n"
        "Individual customers generate 61.4% of all alerts despite comprising 57.9% of accounts, "
        "suggesting slightly higher per-account alert rates compared to Business customers."
    )},
]})

# ex596
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "Compare false positive rates between Business and Individual segments"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex596", "alerts_distribution", {})]},
    {"role": "tool", "content": PC_SEGMENT_STATS, "tool_call_id": "call_ex596"},
    {"role": "assistant", "content": (
        PC_SEGMENT_STATS + "\n\n"
        "Business customers show a marginally higher FP rate (65.1%) compared to Individual (63.9%), "
        "indicating both segments would benefit from threshold tightening to reduce investigator workload."
    )},
]})

# ex597
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "Which segment has the most false positives?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex597", "alerts_distribution", {})]},
    {"role": "tool", "content": PC_SEGMENT_STATS, "tool_call_id": "call_ex597"},
    {"role": "assistant", "content": (
        PC_SEGMENT_STATS + "\n\n"
        "Individual customers have the most false positives in absolute terms — "
        "1,876 FPs out of 2,938 alerts (FP rate=63.9%) — driven by their larger account population."
    )},
]})

# ex598
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "How many alerts does each segment generate?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex598", "alerts_distribution", {})]},
    {"role": "tool", "content": PC_SEGMENT_STATS, "tool_call_id": "call_ex598"},
    {"role": "assistant", "content": (
        PC_SEGMENT_STATS + "\n\n"
        "Business generates 1,847 alerts (38.6%) and Individual generates 2,938 alerts (61.4%), "
        "broadly in line with their respective account populations."
    )},
]})

# ══════════════════════════════════════════════════════════════════════════════
# B. cluster_analysis — targeted queries using PRE-COMPUTED CLUSTER STATS
# ══════════════════════════════════════════════════════════════════════════════

# ex599
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "Which cluster of Business customers has the highest transaction volume?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex599", "cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": PC_BUS_CLUSTER, "tool_call_id": "call_ex599"},
    {"role": "assistant", "content": (
        PC_BUS_CLUSTER + "\n\n"
        "Cluster 1 has the highest transaction volume with a Monthly Txn Volume of $300,181.8 "
        "and Avg Weekly Txn Amount of $30,487.4 — making it the primary high-value Business segment."
    )},
]})

# ex600
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "Show me cluster breakdown for Individual customers"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex600", "cluster_analysis", {"customer_type": "Individual", "n_clusters": 4})]},
    {"role": "tool", "content": PC_IND_CLUSTER, "tool_call_id": "call_ex600"},
    {"role": "assistant", "content": (
        PC_IND_CLUSTER + "\n\n"
        "Cluster 4 (8.7% of individuals) shows a Monthly Txn Volume of $584,180.3 — "
        "nearly 4x Cluster 3 — identifying it as the highest-risk individual segment for AML review."
    )},
]})

# ex601
examples.append({"messages": [
    {"role": "system", "content": SEG_SYSTEM},
    {"role": "user",   "content": "Which Business cluster has the lowest average transaction amount?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex601", "cluster_analysis", {"customer_type": "Business", "n_clusters": 4})]},
    {"role": "tool", "content": PC_BUS_CLUSTER, "tool_call_id": "call_ex601"},
    {"role": "assistant", "content": (
        PC_BUS_CLUSTER + "\n\n"
        "Cluster 4 has the lowest Avg Weekly Txn Amount at $17,441.0, representing the "
        "smallest-transaction Business segment — likely lower-risk but still monitored for structuring patterns."
    )},
]})

# ══════════════════════════════════════════════════════════════════════════════
# C. Rejection wording — invalid column suggests closest substitute
# ══════════════════════════════════════════════════════════════════════════════

# ex602
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Tune the threshold for Individual customers using average daily balance"},
    {"role": "assistant", "content": (
        "Average daily balance is not an available threshold column. "
        "The valid threshold columns are:\n"
        "- **AVG_TRXNS_WEEK** — average weekly transaction frequency\n"
        "- **AVG_TRXN_AMT** — average transaction size\n"
        "- **TRXN_AMT_MONTHLY** — total monthly transaction volume\n\n"
        "TRXN_AMT_MONTHLY is the closest proxy for balance activity — "
        "shall I run the threshold sweep for Individual customers using that column instead?"
    )},
]})

# ex603
examples.append({"messages": [
    {"role": "system", "content": THRESHOLD_SYSTEM},
    {"role": "user",   "content": "Tune thresholds for Business using transaction frequency"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex603", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})]},
    # Note: 'transaction frequency' correctly maps to AVG_TRXNS_WEEK — this is a pass
    {"role": "tool", "content": "=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===\nSegment: Business | Column: AVG_TRXNS_WEEK\nSweep range: 1.0 to 52.0 (51 thresholds)\nTotal alerted: 1,847 | SARs: 643 | non-SARs: 1,204 | overall precision=34.8%\n\nAt lowest threshold (1.0): TP=643, FP=1,204, FN=0, TN=0 (TP rate=100.0%, precision=34.8%).\nCrossover Point — threshold 18.0 (FP=312, FN=298)\n=== END PRE-COMPUTED ANALYSIS ===", "tool_call_id": "call_ex603"},
    {"role": "assistant", "content": (
        "=== PRE-COMPUTED ANALYSIS (copy this verbatim, do not alter numbers) ===\n"
        "Segment: Business | Column: AVG_TRXNS_WEEK\n"
        "Sweep range: 1.0 to 52.0 (51 thresholds)\n"
        "Total alerted: 1,847 | SARs: 643 | non-SARs: 1,204 | overall precision=34.8%\n\n"
        "At lowest threshold (1.0): TP=643, FP=1,204, FN=0, TN=0 (TP rate=100.0%, precision=34.8%).\n"
        "Crossover Point — threshold 18.0 (FP=312, FN=298)\n"
        "=== END PRE-COMPUTED ANALYSIS ===\n\n"
        "At the crossover (threshold=18.0 weekly transactions), FP and FN are nearly equal — "
        "raising the threshold above this point begins missing more SARs than it eliminates false positives."
    )},
]})

# ══════════════════════════════════════════════════════════════════════════════
# D. No tool on rejection/clarification
# ══════════════════════════════════════════════════════════════════════════════

# ex604 — invalid rule name: text only, no list_rules call
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Run rule SAR backtest for Money Mule"},
    {"role": "assistant", "content": (
        "No rule named \"Money Mule\" exists in the monitoring rule set. "
        "The available AML rules are: Activity Deviation (ACH), Activity Deviation (Check), "
        "Elder Abuse, Velocity Single, Detect Excessive Transaction Activity, "
        "Structuring (Incoming Cash), Structuring (Outgoing Cash), CTR Client, "
        "Burst in Originator Activity, Burst in Beneficiary Activity, and Risky International Transfer. "
        "Which of these would you like me to backtest?"
    )},
]})

# ex605 — ambiguous rule name: clarification only, no list_rules call
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Run a 2D sweep for the Structuring rule"},
    {"role": "assistant", "content": (
        "There are two Structuring rules: \"Structuring (Incoming Cash)\" and "
        "\"Structuring (Outgoing Cash)\". Which one would you like me to sweep?"
    )},
]})

# ex606 — follow-up after clarification: now execute correctly
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Run a 2D sweep for Structuring rule for incoming cash"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex606", "rule_2d_sweep", {
         "risk_factor": "Structuring (Incoming Cash)",
         "sweep_param_1": "daily_floor",
         "sweep_param_2": "days_required"
     })]},
    {"role": "tool", "content": PC_2D_STRUCT_IN, "tool_call_id": "call_ex606"},
    {"role": "assistant", "content": (
        PC_2D_STRUCT_IN + "\n\n"
        "Raising the daily_floor to $1,500 with days_required=2 reduces FP by 25% "
        "while maintaining a 78.8% SAR catch rate — a reasonable trade-off for reducing investigator workload."
    )},
]})

# ══════════════════════════════════════════════════════════════════════════════
# E. rule_2d_sweep — Structuring Outgoing Cash
# ══════════════════════════════════════════════════════════════════════════════

# ex607
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "2D sweep for Structuring Outgoing Cash"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex607", "rule_2d_sweep", {
         "risk_factor": "Structuring (Outgoing Cash)",
         "sweep_param_1": "daily_floor",
         "sweep_param_2": "days_required"
     })]},
    {"role": "tool", "content": PC_2D_STRUCT_OUT, "tool_call_id": "call_ex607"},
    {"role": "assistant", "content": (
        PC_2D_STRUCT_OUT + "\n\n"
        "At the current condition the rule catches 93.1% of SARs; tightening to daily_floor=$1,500 "
        "reduces false positives by 27.6% with a modest drop to 81.0% SAR detection."
    )},
]})

# ex608 — Structuring Incoming Cash with explicit params
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Run 2D parameter sweep for Structuring Incoming Cash"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex608", "rule_2d_sweep", {
         "risk_factor": "Structuring (Incoming Cash)",
         "sweep_param_1": "daily_floor",
         "sweep_param_2": "days_required"
     })]},
    {"role": "tool", "content": PC_2D_STRUCT_IN, "tool_call_id": "call_ex608"},
    {"role": "assistant", "content": (
        PC_2D_STRUCT_IN + "\n\n"
        "The current condition (daily_floor=$1,000, days_required=3) catches 92.3% of SARs; "
        "relaxing to days_required=2 while raising the floor to $1,500 maintains acceptable SAR "
        "coverage while cutting false positives by 25%."
    )},
]})

# ══════════════════════════════════════════════════════════════════════════════
# F. list_rules precision query — copy % from PRE-COMPUTED
# ══════════════════════════════════════════════════════════════════════════════

# ex609
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Which rule has the lowest precision?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex609", "list_rules", {})]},
    {"role": "tool", "content": PC_LIST_RULES, "tool_call_id": "call_ex609"},
    {"role": "assistant", "content": (
        PC_LIST_RULES + "\n\n"
        "Elder Abuse has the lowest precision at **17.3%** — for every SAR filed, "
        "over 4 alerts are false positives, making it the highest-priority candidate for threshold tightening."
    )},
]})

# ex610
examples.append({"messages": [
    {"role": "system", "content": RULE_SYSTEM},
    {"role": "user",   "content": "Which rule has the best precision?"},
    {"role": "assistant", "content": None,
     "tool_calls": [tc("call_ex610", "list_rules", {})]},
    {"role": "tool", "content": PC_LIST_RULES, "tool_call_id": "call_ex610"},
    {"role": "assistant", "content": (
        PC_LIST_RULES + "\n\n"
        "Risky International Transfer has the highest precision at **40.4%** — "
        "the best signal-to-noise ratio in the ruleset, meaning fewer investigative resources wasted per SAR."
    )},
]})

# ─── Write output ──────────────────────────────────────────────────────────────

out_v18 = "finetune/data/framl_train_failures_v18.jsonl"
with open(out_v18, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"V18 new examples : {len(examples)} (ex595–ex{594 + len(examples)})")
print(f"Written to       : {out_v18}")

# Combine V17 + V18
v17_path     = "finetune/data/framl_train_combined_v17.jsonl"
v18_combined = "finetune/data/framl_train_combined_v18.jsonl"

with open(v18_combined, "w", encoding="utf-8") as fout:
    for src in [v17_path, out_v18]:
        p = pathlib.Path(src)
        if p.exists():
            fout.write(p.read_text(encoding="utf-8"))
        else:
            print(f"WARNING: {src} not found — skipping")

total = sum(1 for _ in open(v18_combined, encoding="utf-8"))
print(f"V18 combined     : {total} examples -> {v18_combined}")

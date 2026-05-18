"""
V52 training examples (2026-05-18).

Targets 9 gaps found during aria-v13 (V51) smoke testing:

  ACS_V52_1   Gap 1 — Single-cluster follow-up after comparison turn
              3-turn: (compare C2&C4) → "how about cluster 3" must describe C3
              NOT refuse with "I cannot characterize a specific cluster".

  ACS_V52_2   Gap 2 — Elliptical "and the youngest" without "cluster" keyword
              Business: "oldest cluster" → "and the youngest" → correct value

  ACS_V52_3   Gap 2 — Elliptical "how about the oldest" without "cluster" keyword
              Business: "youngest cluster" → "how about the oldest" → correct value

  ACS_V52_4   Gap 3 — "which one has lowest" pulls exact count from PREVIOUS RESULT
              Business: "highest number → which has lowest" → exact n, not hallucinated

  ACS_V52_5   Gap 4 — Small cluster characterization reads exact n from stats
              Business: "how about cluster 2" → n=30, not plausible-sounding large number

  BUS_V52_7   Initial clustering: per-cluster bullets for Business
              segment_customers(Business) → ### + one bullet per cluster

  IND_V52_1   Gap 5 — Individual segmentation initial call: per-cluster bullets, no stats copy
              segment_customers(Individual) → ### + one bullet per cluster, no PRE-COMPUTED echo

  IND_V52_2   Gap 5 — Individual segmentation multi-turn: initial bullets + follow-up one-sentence
              Tool call → per-cluster bullets; follow-up → one sentence from context

  IND_V52_3   Gap 6 — Superlative response includes exact value from PREVIOUS RESULT
              Individual: "oldest" → correct value; "and the youngest" → correct value

  ACS_V52_6   Gap 7 — SAR count read from PREVIOUS CLUSTERING RESULT, no tool call
              Individual: "how many SARs in cluster 1" → 717, read from injected stats

  ARL_V52_1   Gap 8 — Rules multi-turn: "how about the bottom 3" infers precision from context
              top 3 by precision → "how about the bottom 3" → bottom 3 by precision, no clarification

  ARL_V52_2   Gap 8b — Bottom N by precision: all 3 names + % values, no "shown below"
              Explicit "bottom 3 rules by precision" → 3 rules named with %, no table reference

  AT_V52_1    Gap 9 — cluster_threshold_analysis: per-cluster bullets in response
              Response must list recommended threshold per cluster, not just aggregate summary

Base:  aria_train_combined_v51_full.jsonl
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V51_FULL_PATH = DATA_DIR / "aria_train_combined_v51_full.jsonl"
V52_ONLY_PATH = DATA_DIR / "aria_train_v52.jsonl"
V52_FULL_PATH = DATA_DIR / "aria_train_combined_v52_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from write_v50 import THRESHOLD_SYSTEM, SEGMENTATION_SYSTEM as _SEG_V50  # noqa: E402
from write_v42 import tc, prev_context                                    # noqa: E402

# Patch Rule 9 to distinguish initial segment_customers call from follow-ups.
# Initial call: per-cluster bullet list (one bullet per cluster, key stats).
# Follow-up with [PREVIOUS CLUSTERING RESULT]: ONE sentence answer.
SEGMENTATION_SYSTEM = _SEG_V50.replace(
    '9. After receiving tool results, output ONE sentence describing the highest-risk cluster '
    'based solely on the numbers in the tool result. Do NOT copy or re-echo cluster stats — '
    'the chart panel renders them automatically. Do NOT suggest thresholds, dollar cutoffs, '
    'or monitoring actions.',
    '9. After receiving segment_customers tool results: output ONE ### header, then ONE '
    'per-cluster bullet summarising key stats (customer count, avg weekly transactions, '
    'monthly volume, account age, alerts, SARs). Do NOT copy the raw PRE-COMPUTED block '
    'verbatim — write compact human-readable bullets. After follow-up queries that include '
    '[PREVIOUS CLUSTERING RESULT]: answer in ONE sentence using exact values from that block. '
    'Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.',
)
assert "per-cluster bullet" in SEGMENTATION_SYSTEM, "Rule 9 patch failed"

print(f"[V52] THRESHOLD_SYSTEM:    {len(THRESHOLD_SYSTEM)} chars (~{len(THRESHOLD_SYSTEM)//4} tokens)")
print(f"[V52] SEGMENTATION_SYSTEM: {len(SEGMENTATION_SYSTEM)} chars (~{len(SEGMENTATION_SYSTEM)//4} tokens)")

examples = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def prev_rule_list(rule_list_content):
    return f"[PREVIOUS RULE LIST]\n{rule_list_content}\n[END RULE LIST]"


# ═══════════════════════════════════════════════════════════════════════════
# Cluster stats blocks
# ═══════════════════════════════════════════════════════════════════════════

# Business segmentation — matches smoketest cluster sizes (Cluster 1 dominates)
_BUS_STATS = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | 4 clusters

Cluster 1 (n=2,271):
- Avg Weekly Transactions: **1.2**
- Avg Weekly Txn Amount: **$3,450**
- Monthly Txn Volume: **$14,967**
- Account Age (years): **6.8**

Cluster 2 (n=30):
- Avg Weekly Transactions: **8.4**
- Avg Weekly Txn Amount: **$12,200**
- Monthly Txn Volume: **$51,600**
- Account Age (years): **4.2**

Cluster 3 (n=37):
- Avg Weekly Transactions: **3.1**
- Avg Weekly Txn Amount: **$2,800**
- Monthly Txn Volume: **$11,200**
- Account Age (years): **3.1**

Cluster 4 (n=31):
- Avg Weekly Transactions: **6.5**
- Avg Weekly Txn Amount: **$8,900**
- Monthly Txn Volume: **$37,800**
- Account Age (years): **9.7**
=== END PRE-COMPUTED CLUSTER STATS ===

Alert/SAR/FP counts per cluster (customer-level, matches treemap):
Cluster 1: Alerts: 2,105 | SARs: 717 | FPs: 1,388
Cluster 2: Alerts: 28 | SARs: 9 | FPs: 19
Cluster 3: Alerts: 34 | SARs: 7 | FPs: 27
Cluster 4: Alerts: 29 | SARs: 8 | FPs: 21"""

_BUS_TOOL_RESULT = "Tool result for segment_customers:\n" + _BUS_STATS

# Individual segmentation — ages from smoketest (Cluster 2 oldest=54.5, Cluster 3 youngest=44.1)
_IND_STATS = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Individual | 4 clusters

Cluster 1 (n=6,674):
- Avg Weekly Transactions: **1.2**
- Avg Weekly Txn Amount: **$890**
- Monthly Txn Volume: **$3,780**
- Account Age (years): **38.2**

Cluster 2 (n=1,842):
- Avg Weekly Transactions: **4.3**
- Avg Weekly Txn Amount: **$2,150**
- Monthly Txn Volume: **$9,450**
- Account Age (years): **54.5**

Cluster 3 (n=1,127):
- Avg Weekly Transactions: **2.8**
- Avg Weekly Txn Amount: **$1,340**
- Monthly Txn Volume: **$5,720**
- Account Age (years): **44.1**

Cluster 4 (n=798):
- Avg Weekly Transactions: **7.6**
- Avg Weekly Txn Amount: **$4,230**
- Monthly Txn Volume: **$18,900**
- Account Age (years): **47.3**
=== END PRE-COMPUTED CLUSTER STATS ===

Alert/SAR/FP counts per cluster (customer-level, matches treemap):
Cluster 1: Alerts: 3,317 | SARs: 717 | FPs: 2,600
Cluster 2: Alerts: 1,205 | SARs: 382 | FPs: 823
Cluster 3: Alerts: 743 | SARs: 201 | FPs: 542
Cluster 4: Alerts: 651 | SARs: 198 | FPs: 453"""

_IND_TOOL_RESULT = "Tool result for segment_customers:\n" + _IND_STATS


# ═══════════════════════════════════════════════════════════════════════════
# Rule list blocks (new === RULE LIST === header, matching runtime)
# Dataset A — Round-trip wins precision (35.0%)
# ═══════════════════════════════════════════════════════════════════════════

_LIST_A_V52 = (
    "=== RULE LIST ===\n"
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
_PC_LIST_A_V52 = "Tool result for list_rules:\n" + _LIST_A_V52


# ═══════════════════════════════════════════════════════════════════════════
# ACS_V52_1 — Gap 1: single-cluster follow-up after comparison turn
# ═══════════════════════════════════════════════════════════════════════════
# Teach: after a comparison response, a "how about cluster N" query must
# describe that cluster from PREVIOUS CLUSTERING RESULT — not refuse.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    # Turn 1: compare two clusters
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\ncompare cluster 2 and cluster 4"},
    {"role": "assistant", "content": (
        "### Cluster 2 vs Cluster 4 — Business\n\n"
        "**Transaction velocity**: Cluster 4 averages **6.5** weekly transactions vs **8.4** for Cluster 2 — "
        "Cluster 2 is slightly more active on a weekly basis.\n\n"
        "**Transaction size**: Cluster 4 averages **$8,900** per transaction vs **$12,200** for Cluster 2 — "
        "Cluster 2 handles larger individual transactions.\n\n"
        "**Monthly volume**: Cluster 4 reaches **$37,800** monthly vs **$51,600** for Cluster 2 — "
        "Cluster 2 generates more monthly throughput despite having fewer customers.\n\n"
        "**Account tenure**: Cluster 4 is significantly older (**9.7** vs **4.2** years), "
        "suggesting more established relationship history.\n\n"
        "Both clusters are small (30 and 31 customers respectively) but show above-average activity "
        "compared to the dominant Cluster 1 — Cluster 2 warrants particular attention given its "
        "high monthly volume relative to its size."
    )},
    # Turn 2: ask about a third cluster — must describe it, not refuse
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nhow about cluster 3"},
    {"role": "assistant", "content": (
        "### Cluster 3 — Business\n\n"
        "Cluster 3 contains **37 customers** with moderate weekly transaction activity (**3.1** transactions, "
        "**$2,800** average amount, **$11,200** monthly volume) and the youngest account age of any cluster "
        "(**3.1 years**), suggesting recently onboarded customers. "
        "With **34 alerted** customers, **7 SARs**, and **27 FPs**, its SAR rate is lower than Clusters 2 and 4 "
        "but its young tenure warrants continued monitoring for emerging patterns."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ACS_V52_2 — Gap 2: elliptical "and the youngest" (Business)
# ═══════════════════════════════════════════════════════════════════════════
# Teach: "and the youngest" with no "cluster" keyword must infer context
# and answer with the exact account age from PREVIOUS CLUSTERING RESULT.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nwhich cluster is the oldest"},
    {"role": "assistant", "content": (
        "Cluster 4 has the oldest accounts with an average age of **9.7 years**."
    )},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nand the youngest"},
    {"role": "assistant", "content": (
        "Cluster 3 has the youngest accounts with an average age of **3.1 years**."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ACS_V52_3 — Gap 2: elliptical "how about the oldest" (Business)
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nwhat is the youngest cluster"},
    {"role": "assistant", "content": (
        "Cluster 3 is the youngest with an average account age of **3.1 years**."
    )},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nhow about the oldest"},
    {"role": "assistant", "content": (
        "Cluster 4 is the oldest with an average account age of **9.7 years**."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ACS_V52_4 — Gap 3: "which one has lowest" pulls exact count (Business)
# ═══════════════════════════════════════════════════════════════════════════
# Teach: "which one has lowest" after a highest-customer-count question must
# read the exact n from PREVIOUS CLUSTERING RESULT, not hallucinate a count.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nwhich cluster has the highest number of customers"},
    {"role": "assistant", "content": (
        "Cluster 1 has the highest number of customers with **2,271 accounts** (95.9% of the Business segment)."
    )},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nwhich one has lowest"},
    {"role": "assistant", "content": (
        "Cluster 2 has the lowest number of customers with **30 accounts** (1.3% of the Business segment)."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ACS_V52_5 — Gap 4: small cluster characterization reads exact n (Business)
# ═══════════════════════════════════════════════════════════════════════════
# Teach: "how about cluster 2" with n=30 must echo 30, not generate a
# plausible-sounding large number like 1,890.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nhow do you characterize cluster 1"},
    {"role": "assistant", "content": (
        "### Cluster 1 — Business\n\n"
        "Cluster 1 is the dominant segment with **2,271 customers** (95.9%) — "
        "low-activity accounts averaging just **1.2** weekly transactions and **$14,967** monthly volume, "
        "consistent with standard business banking customers with limited AML exposure."
    )},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nhow about cluster 4"},
    {"role": "assistant", "content": (
        "### Cluster 4 — Business\n\n"
        "Cluster 4 is a small group of **31 customers** with the longest account tenure (**9.7 years**) "
        "and elevated activity — **6.5** weekly transactions and **$37,800** monthly volume — "
        "suggesting established high-value business relationships that warrant enhanced due diligence."
    )},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nhow about cluster 2"},
    {"role": "assistant", "content": (
        "### Cluster 2 — Business\n\n"
        "Cluster 2 is the smallest segment with **30 customers** (1.3%) but the highest per-customer activity: "
        "**8.4** weekly transactions, **$12,200** average transaction amount, and **$51,600** monthly volume — "
        "the highest monthly throughput of any cluster despite being the smallest group, "
        "which makes these accounts the highest priority for transaction monitoring review."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# BUS_V52_7 — Initial Business clustering: per-cluster bullets
# ═══════════════════════════════════════════════════════════════════════════
# Teach: initial segment_customers(Business) → ### header + one bullet per
# cluster. No verbatim PRE-COMPUTED copy; no single-sentence summary.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Cluster Business customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("bus7", "segment_customers", {"segment": "Business", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "bus7", "content": _BUS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Business Customer Segments\n\n"
        "- **Cluster 1** (2,271 customers · 95.9%): 1.2 txns/wk · $14,967/mo · 6.8 yr tenure · "
        "2,105 alerted · 717 SARs · 1,388 FPs\n"
        "- **Cluster 2** (30 customers · 1.3%): 8.4 txns/wk · $51,600/mo · 4.2 yr · "
        "28 alerted · 9 SARs — highest per-customer volume\n"
        "- **Cluster 3** (37 customers · 1.6%): 3.1 txns/wk · $11,200/mo · 3.1 yr · "
        "34 alerted · 7 SARs — youngest accounts\n"
        "- **Cluster 4** (31 customers · 1.3%): 6.5 txns/wk · $37,800/mo · 9.7 yr · "
        "29 alerted · 8 SARs — most tenured, elevated activity"
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# IND_V52_1 — Gap 5: Individual segmentation initial call — per-cluster bullets
# ═══════════════════════════════════════════════════════════════════════════
# Teach: segment_customers(Individual) → ### + one bullet per cluster.
# No verbatim PRE-COMPUTED copy.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Show me behavioral segments for Individual customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ind1", "segment_customers", {"segment": "Individual", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ind1", "content": _IND_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Individual Customer Segments\n\n"
        "- **Cluster 1** (6,674 customers · 63.3%): 1.2 txns/wk · $3,780/mo · 38.2 yr · "
        "3,317 alerted · 717 SARs · 2,600 FPs — largest segment, highest raw alert volume\n"
        "- **Cluster 2** (1,842 customers · 17.5%): 4.3 txns/wk · $9,450/mo · 54.5 yr · "
        "1,205 alerted · 382 SARs — oldest accounts, elevated activity\n"
        "- **Cluster 3** (1,127 customers · 10.7%): 2.8 txns/wk · $5,720/mo · 44.1 yr · "
        "743 alerted · 201 SARs\n"
        "- **Cluster 4** (798 customers · 7.6%): 7.6 txns/wk · $18,900/mo · 47.3 yr · "
        "651 alerted · 198 SARs — highest per-customer volume and SAR density"
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# IND_V52_2 — Gap 5: Individual multi-turn — initial bullets + follow-up one-sentence
# ═══════════════════════════════════════════════════════════════════════════
# Teach: initial tool call → per-cluster bullets; follow-up with
# [PREVIOUS CLUSTERING RESULT] → one sentence from context.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "cluster individual customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ind2", "segment_customers", {"segment": "Individual", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ind2", "content": _IND_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Individual Customer Segments\n\n"
        "- **Cluster 1** (6,674 customers · 63.3%): 1.2 txns/wk · $3,780/mo · 38.2 yr · "
        "3,317 alerted · 717 SARs · 2,600 FPs\n"
        "- **Cluster 2** (1,842 customers · 17.5%): 4.3 txns/wk · $9,450/mo · 54.5 yr · "
        "1,205 alerted · 382 SARs — oldest, elevated activity\n"
        "- **Cluster 3** (1,127 customers · 10.7%): 2.8 txns/wk · $5,720/mo · 44.1 yr · "
        "743 alerted · 201 SARs\n"
        "- **Cluster 4** (798 customers · 7.6%): 7.6 txns/wk · $18,900/mo · 47.3 yr · "
        "651 alerted · 198 SARs — highest per-customer volume and SAR density"
    )},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nwhich cluster has the most alerts"},
    {"role": "assistant", "content": (
        "Cluster 1 has the most alerts with **3,317 alerted customers**, of which **717 are SARs** and "
        "**2,600 are false positives** — its large size (6,674 customers) drives the high alert volume "
        "even though its per-customer risk is lower than Cluster 4."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# IND_V52_3 — Gap 6: superlative includes exact value from PREVIOUS RESULT
# ═══════════════════════════════════════════════════════════════════════════
# Teach: "and the youngest" follow-up must give 44.1 (Cluster 3) from the
# injected stats — not hallucinate a plausible age like 31.2.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nwhich cluster is the oldest"},
    {"role": "assistant", "content": (
        "Cluster 2 is the oldest with an average account age of **54.5 years**."
    )},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nand the youngest"},
    {"role": "assistant", "content": (
        "Cluster 1 is the youngest with an average account age of **38.2 years**."
    )},
]})

# Contrastive pair: youngest first, then oldest
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nwhat is the youngest cluster"},
    {"role": "assistant", "content": (
        "Cluster 1 is the youngest with an average account age of **38.2 years**."
    )},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nhow about the oldest"},
    {"role": "assistant", "content": (
        "Cluster 2 is the oldest with an average account age of **54.5 years**."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ACS_V52_6 — Gap 7: SAR count read directly from PREVIOUS CLUSTERING RESULT
# ═══════════════════════════════════════════════════════════════════════════
# Teach: when the PREVIOUS CLUSTERING RESULT already contains "SARs: 717",
# the model must read it directly — no tool call to cluster_rule_summary.

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nhow many SARs in cluster 1"},
    {"role": "assistant", "content": (
        "Cluster 1 has **717 SAR customers** out of 3,317 alerted accounts, "
        "with the remaining 2,600 being false positives — a SAR rate of 21.6% among alerted customers."
    )},
]})

# Second variant: "how many alerts does cluster 4 have"
examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nhow many alerts does cluster 4 have"},
    {"role": "assistant", "content": (
        "Cluster 4 has **651 alerted customers**, of which **198 are SARs** and **453 are false positives**."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ARL_V52_1 — Gap 8: "how about the bottom 3" infers metric from context
# ═══════════════════════════════════════════════════════════════════════════
# Teach: after "top 3 by precision", "how about the bottom 3" must infer
# "by precision" and answer with 3 rule names + % values — no clarification,
# no "I do not have access to performance metrics".

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "top three highest precision rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("arl52_1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl52_1", "content": _PC_LIST_A_V52},
    {"role": "assistant", "content": (
        "### Top Rules by Precision\n\n"
        "The top three rules by precision are "
        "**Round-trip** (35.0%), **Risky International Transfer** (26.7%), and **Funnel Account** (26.0%)."
    )},
    {"role": "user",      "content": prev_rule_list(_LIST_A_V52) + "\n\nhow about the bottom 3"},
    {"role": "assistant", "content": (
        "### Lowest Precision Rules\n\n"
        "The bottom three rules by precision are "
        "**CTR Client** (13.0%), **Activity Deviation (Check)** (13.0%), and "
        "**Detect Excessive Transaction Activity** (14.0%)."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ARL_V52_2 — Gap 8b: explicit "bottom 3 by precision" — no "shown below",
#             all 3 rules named with % values
# ═══════════════════════════════════════════════════════════════════════════
# Teach: follow-up precision query uses _last_rule_list context and gives
# all 3 rule names + precision % — no "detailed table shown below" (table
# was already shown on the initial list_rules turn, not on follow-ups).

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "list all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("arl52_2a", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "arl52_2a", "content": _PC_LIST_A_V52},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        "Rule performance summary — detailed table shown below.\n\n"
        "Rules with the most false positives: CTR Client (348 FP), "
        "Elder Abuse (341 FP), Detect Excessive Transaction Activity (301 FP)."
    )},
    {"role": "user",      "content": prev_rule_list(_LIST_A_V52) + "\n\nwhat are the bottom 3 rules by precision"},
    {"role": "assistant", "content": (
        "### Lowest Precision Rules\n\n"
        "The three rules with the lowest precision are "
        "**CTR Client** (13.0%), **Activity Deviation (Check)** (13.0%), and "
        "**Detect Excessive Transaction Activity** (14.0%) — "
        "all three generate the highest investigator workload relative to confirmed SARs."
    )},
]})

# Contrastive: top 3 by SAR count → "how about the bottom 3" → bottom 3 by SAR count
_LIST_A_SAR_SORTED = _LIST_A_V52  # same data, model reads SAR= field

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": prev_rule_list(_LIST_A_V52) + "\n\nwhich rules have the highest SAR count"},
    {"role": "assistant", "content": (
        "### Highest SAR Count Rules\n\n"
        "The top three rules by raw SAR count are "
        "**Elder Abuse** (59 SARs), **CTR Client** (52 SARs), and "
        "**Risky International Transfer** (40 SARs)."
    )},
    {"role": "user",      "content": prev_rule_list(_LIST_A_V52) + "\n\nhow about the bottom 3"},
    {"role": "assistant", "content": (
        "### Lowest SAR Count Rules\n\n"
        "The three rules with the lowest SAR counts are "
        "**Human Trafficking Indicators** (23 SARs), **Activity Deviation (Wire)** (25 SARs), and "
        "**Activity Deviation (Check)** (26 SARs)."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# AT_V52_1 — Gap 9: cluster_threshold_analysis — per-cluster bullets in response
# ═══════════════════════════════════════════════════════════════════════════
# Teach: response to cluster_threshold_analysis must list the recommended
# threshold and FP change per cluster — not just the aggregate summary sentence.
# The ADAPTIVE SENSITIVITY SUMMARY numbers are a subset; per-cluster lines
# give the actionable recommendations.

_AT_BUS_TOOL_RESULT = """\
Tool result for cluster_threshold_analysis:
=== CLUSTER THRESHOLD ANALYSIS ===
Segment: **Business** | Column: AVG_TRXNS_WEEK (Avg Weekly Transactions) | Clusters: 4 | Target SAR catch: ≥90%

**Cluster 1 — High Volume** (48 customers, SAR pool: 21)
- Uniform 1.20: TP=20, FP=24, TP rate=95.2%, precision=45.5%
- Recommended 1.60: TP=19, FP=17, TP rate=90.5%, precision=52.8% (-7 FP, -1 SAR)

**Cluster 2 — Mid-High** (35 customers, SAR pool: 14)
- Uniform 1.20: TP=13, FP=18, TP rate=92.9%, precision=41.9%
- Recommended 1.00: TP=13, FP=14, TP rate=92.9%, precision=48.1% (-4 FP, +0 SAR)

**Cluster 3 — Mid-Low** (67 customers, SAR pool: 27)
- Uniform 1.20: TP=25, FP=36, TP rate=92.6%, precision=41.0%
- Recommended 1.00: TP=25, FP=31, TP rate=92.6%, precision=44.6% (-5 FP, +0 SAR)

**Cluster 4 — Low Volume** (154 customers, SAR pool: 63)
- Uniform 1.20: TP=59, FP=85, TP rate=93.7%, precision=41.0%
- Recommended 0.80: TP=59, FP=81, TP rate=93.7%, precision=42.1% (-4 FP, +0 SAR)

**ADAPTIVE SENSITIVITY SUMMARY**
- Uniform 1.20 applied to all Business: TP=117, FP=163, TP rate=93.6%
- Cluster-adaptive thresholds: TP=116, FP=143, TP rate=92.8%
- Net change: **-20 FP**, -1 SARs (92.8% SAR retention)
=== END CLUSTER THRESHOLD ANALYSIS ==="""

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show strategies to reduce false positives for Business customer segmentation"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("at52_1", "cluster_threshold_analysis", {
            "segment": "Business",
            "threshold_column": "AVG_TRXNS_WEEK",
            "n_clusters": 4,
            "target_sar_rate": 0.90
        })
    ]},
    {"role": "tool", "tool_call_id": "at52_1", "content": _AT_BUS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Adaptive Thresholds — Business | Avg Weekly Transactions\n\n"
        "- **Cluster 1 (High Volume):** raise to 1.60 → −7 FP, 90.5% SAR retention\n"
        "- **Cluster 2 (Mid-High):** lower to 1.00 → −4 FP, 92.9% SAR retention\n"
        "- **Cluster 3 (Mid-Low):** lower to 1.00 → −5 FP, 92.6% SAR retention\n"
        "- **Cluster 4 (Low Volume):** lower to 0.80 → −4 FP, 93.7% SAR retention\n\n"
        "Net: **−20 FP** across all clusters at 92.8% SAR retention — "
        "hover on the chart bars for full per-cluster FP breakdown."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# ARS_V52_1 — Gap 10: rule_sar_backtest cross-row TP rate contamination
# ═══════════════════════════════════════════════════════════════════════════
# Teach: the model must pair each threshold with its own exact TP rate from
# that row of the RULE SWEEP table.
# Confirmed from app: at $90K → 40 caught, 6 missed → TP rate = 87.0%.
# The model hallucinated 95.7% (which belongs to $70K) at the $90K row.
# The insight must name the specific threshold AND its correct TP rate,
# with caught + missed summing to the total SAR pool (46 for ACH).

_PC_ARS_ACH_FLOOR = """\
Tool result for rule_sar_backtest:
=== RULE SWEEP ===
Rule: Activity Deviation (ACH)
Current condition: Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean
Sweep parameter: floor_amount - Minimum monthly Outgoing ACH sum to trigger (currently $50K)
Current value: 50,000
Labeled population: 300 customers (TP+FN pool=46 SAR, FP+TN pool=254 non-SAR, precision=15.3%)

At the lowest value ($10,000): TP=46, FP=254, FN=0, TN=0 (TP rate=100.0%, precision=15.3%).
At current condition ($50,000): TP=46, FP=254, FN=0, TN=0 (TP rate=100.0%, precision=15.3%).
To keep TP rate >=90%: floor_amount <= $70,000 => TP=44, FP=230, FN=2, TN=24, precision=16.1%.
To keep TP rate >=50%: floor_amount <= $110,000 => TP=35, FP=183, FN=11, TN=71, precision=16.1%.
At $70,000: TP=44, FP=230, FN=2, TN=24 (TP rate=95.7%, precision=16.1%).
At $90,000: TP=40, FP=207, FN=6, TN=47 (TP rate=87.0%, precision=16.2%).
At $110,000: TP=35, FP=183, FN=11, TN=71 (TP rate=76.1%, precision=16.1%).
At the highest value ($130,000): TP=27, FP=152, FN=19, TN=102 (TP rate=58.7%, precision=15.1%).

*(Detailed sweep chart shown below.)*
=== END RULE SWEEP ==="""

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show SAR backtest for Activity Deviation ACH rule"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ars52_1", "rule_sar_backtest",
           {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})
    ]},
    {"role": "tool", "tool_call_id": "ars52_1", "content": _PC_ARS_ACH_FLOOR},
    {"role": "assistant", "content": (
        "### SAR Backtest — Activity Deviation (ACH)\n\n"
        "Raising floor_amount from $50K to $70K maintains **95.7% SAR catch rate** "
        "(44 of 46 SARs caught, 2 missed) while reducing false positives from 254 to 230 "
        "and improving precision from 15.3% to 16.1% — "
        "the recommended threshold; pushing to $90K drops the catch rate to **87.0%** "
        "(40 caught, 6 missed) with diminishing FP benefit."
    )},
]})

# Contrastive: question that directly asks about $90K → must give 87.0%, not 95.7%
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "What happens to the SAR catch rate if we raise the ACH floor to $90K?"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ars52_2", "rule_sar_backtest",
           {"risk_factor": "Activity Deviation (ACH)", "sweep_param": "floor_amount"})
    ]},
    {"role": "tool", "tool_call_id": "ars52_2", "content": _PC_ARS_ACH_FLOOR},
    {"role": "assistant", "content": (
        "### SAR Backtest — Activity Deviation (ACH) | $90K Floor\n\n"
        "At a $90K floor_amount: **40 of 46 SARs caught** (87.0% catch rate, 6 missed), "
        "with FPs reduced from 254 to 207 and precision improving slightly from 15.3% to 16.2% — "
        "however, the 6 missed SARs represent a meaningful detection gap; "
        "$70K is the highest threshold that keeps TP rate above 90% (44 caught, 95.7%)."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# Combine + write
# ═══════════════════════════════════════════════════════════════════════════

def main():
    v51_base = []
    with open(V51_FULL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v51_base.append(json.loads(line))
    print(f"[V52] Loaded {len(v51_base)} base examples from {V51_FULL_PATH.name}")

    all_examples = v51_base + examples

    # Write V52-only
    with open(V52_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V52] V52-only: {V52_ONLY_PATH.name} ({len(examples)} examples)")

    # Write V52 combined
    with open(V52_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V52] Combined: {V52_FULL_PATH.name} ({len(all_examples)} total)")


if __name__ == "__main__":
    main()

"""
V53 training examples (2026-05-19).

Root cause of V52 (aria-v15) 0-char regression:
  Training data uses tool name 'segment_customers'; runtime system prompt
  says 'call ds_cluster_analysis'. At inference the model calls ds_cluster_analysis,
  gets tool results, but has NO training signal to generate text afterward —
  training only showed responses to 'segment_customers' results. Same mismatch
  may contribute to list_rules 0-char (Rule 9 conflict: "no bullet lists" vs
  training examples that produce bullet lists).

Targets:

  DS_V53_1-5  Fix 0-char: ds_cluster_analysis (customer_type=) → per-cluster bullets
              Uses ds_cluster_analysis as tool name, customer_type as param —
              matching the actual runtime tool and segmentation_agent.py SYSTEM_PROMPT.
              Business (initial + multi-turn), Individual (initial + multi-turn), All.

  RL_V53_1-3  Fix 0-char: list_rules → ranked insight response
              More tool_call → text examples to re-anchor text generation after
              list_rules. Different entry phrasings to avoid memorisation.

  AT_V53_1    Gap 9: cluster_threshold_analysis Individual — per-cluster bullets
              The Business AT was already in AT_V52_1; this adds the Individual
              variant so the model sees the pattern across both segments.

  SAR_V53_1-2 SAR backtest segment — default to AVG_TRXNS_WEEK without asking
              Business and Individual variants: no threshold_column in query →
              model calls sar_backtest(segment=X, threshold_column=AVG_TRXNS_WEEK).

System prompt changes:
  SEGMENTATION_SYSTEM: Rule 9 updated from "segment_customers" → "ds_cluster_analysis"
  THRESHOLD_SYSTEM:    Rule 9 exception added for cluster_threshold_analysis (per-cluster bullets)

Base: aria_train_combined_v52_full.jsonl
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V52_FULL_PATH = DATA_DIR / "aria_train_combined_v52_full.jsonl"
V53_ONLY_PATH = DATA_DIR / "aria_train_v53.jsonl"
V53_FULL_PATH = DATA_DIR / "aria_train_combined_v53_full.jsonl"

_PROJECT_ROOT = str(DATA_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(DATA_DIR))

from write_v52 import THRESHOLD_SYSTEM as _THR_V52, SEGMENTATION_SYSTEM as _SEG_V52  # noqa: E402
from write_v42 import tc, prev_context                                                 # noqa: E402

# ── System prompt patches ─────────────────────────────────────────────────────

# Fix training/runtime mismatch: Rule 9 referenced "segment_customers" but the
# runtime segmentation_agent.py SYSTEM_PROMPT says "call ds_cluster_analysis".
SEGMENTATION_SYSTEM = _SEG_V52.replace(
    "9. After receiving segment_customers tool results:",
    "9. After receiving ds_cluster_analysis tool results:",
)
assert "ds_cluster_analysis tool results" in SEGMENTATION_SYSTEM, \
    "SEGMENTATION_SYSTEM Rule 9 patch failed"

# Add per-cluster bullets exception for cluster_threshold_analysis — Rule 9
# currently says "Do NOT include tables, bullet lists" which conflicts with
# the expected AT response format.
THRESHOLD_SYSTEM = _THR_V52.replace(
    "Do NOT include tables, bullet lists, or numerical summaries in your response.",
    "Do NOT include tables, bullet lists, or numerical summaries in your response. "
    "EXCEPTION: after cluster_threshold_analysis results — write ONE ### header, "
    "then ONE bullet per cluster (recommended threshold, FP change, SAR retention %) "
    "and a net summary line. The per-cluster bar chart is shown separately.",
)
assert "EXCEPTION: after cluster_threshold_analysis" in THRESHOLD_SYSTEM, \
    "THRESHOLD_SYSTEM Rule 9 exception patch failed"

print(f"[V53] SEGMENTATION_SYSTEM: {len(SEGMENTATION_SYSTEM)} chars "
      f"('ds_cluster_analysis tool results' present: "
      f"{'ds_cluster_analysis tool results' in SEGMENTATION_SYSTEM})")
print(f"[V53] THRESHOLD_SYSTEM:    {len(THRESHOLD_SYSTEM)} chars "
      f"(AT exception present: {'EXCEPTION: after cluster_threshold_analysis' in THRESHOLD_SYSTEM})")

examples = []


# ═══════════════════════════════════════════════════════════════════════════
# Cluster stats blocks (reuse from V52 — same actual production data)
# ═══════════════════════════════════════════════════════════════════════════

_BUS_STATS = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Business | 4 clusters

**Cluster 1** (n=2,271):
- Avg Weekly Transactions: **1.2**
- Avg Weekly Txn Amount: **$3,450**
- Monthly Txn Volume: **$14,967**
- Account Age (years): **6.8**

**Cluster 2** (n=30):
- Avg Weekly Transactions: **8.4**
- Avg Weekly Txn Amount: **$12,200**
- Monthly Txn Volume: **$51,600**
- Account Age (years): **4.2**

**Cluster 3** (n=37):
- Avg Weekly Transactions: **3.1**
- Avg Weekly Txn Amount: **$2,800**
- Monthly Txn Volume: **$11,200**
- Account Age (years): **3.1**

**Cluster 4** (n=31):
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

_IND_STATS = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: Individual | 4 clusters

**Cluster 1** (n=6,674):
- Avg Weekly Transactions: **1.2**
- Avg Weekly Txn Amount: **$890**
- Monthly Txn Volume: **$3,780**
- Account Age (years): **38.2**

**Cluster 2** (n=1,842):
- Avg Weekly Transactions: **4.3**
- Avg Weekly Txn Amount: **$2,150**
- Monthly Txn Volume: **$9,450**
- Account Age (years): **54.5**

**Cluster 3** (n=1,127):
- Avg Weekly Transactions: **2.8**
- Avg Weekly Txn Amount: **$1,340**
- Monthly Txn Volume: **$5,720**
- Account Age (years): **44.1**

**Cluster 4** (n=798):
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

_BUS_DS_TOOL_RESULT = "Tool result for ds_cluster_analysis:\n" + _BUS_STATS
_IND_DS_TOOL_RESULT = "Tool result for ds_cluster_analysis:\n" + _IND_STATS

# Rule list (same data as V52 Dataset A)
from write_v52 import _LIST_A_V52, _PC_LIST_A_V52  # noqa: E402

def prev_rule_list(content):
    return f"[PREVIOUS RULE LIST]\n{content}\n[END RULE LIST]"


# ═══════════════════════════════════════════════════════════════════════════
# DS_V53_1 — Business initial clustering via ds_cluster_analysis
# ═══════════════════════════════════════════════════════════════════════════
# Key fix: tool name is ds_cluster_analysis, param is customer_type.
# This matches what the model calls at inference (per runtime SYSTEM_PROMPT).

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Cluster Business customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ds53_1", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ds53_1", "content": _BUS_DS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Business Customer Segments\n\n"
        "- **Cluster 1** (2,271 customers · 95.9%): 1.2 txns/wk · $14,967/mo · 6.8 yr tenure · "
        "2,105 alerted · 717 SARs · 1,388 FPs — dominant low-activity segment\n"
        "- **Cluster 2** (30 customers · 1.3%): 8.4 txns/wk · $51,600/mo · 4.2 yr · "
        "28 alerted · 9 SARs — highest per-customer volume despite smallest count\n"
        "- **Cluster 3** (37 customers · 1.6%): 3.1 txns/wk · $11,200/mo · 3.1 yr · "
        "34 alerted · 7 SARs — youngest accounts\n"
        "- **Cluster 4** (31 customers · 1.3%): 6.5 txns/wk · $37,800/mo · 9.7 yr · "
        "29 alerted · 8 SARs — most tenured, elevated activity"
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# DS_V53_2 — Business multi-turn: ds_cluster_analysis then follow-up from context
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Show behavioral segments for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ds53_2", "ds_cluster_analysis", {"customer_type": "Business", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ds53_2", "content": _BUS_DS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Business Customer Segments\n\n"
        "- **Cluster 1** (2,271 customers · 95.9%): 1.2 txns/wk · $14,967/mo · 6.8 yr · "
        "2,105 alerted · 717 SARs · 1,388 FPs\n"
        "- **Cluster 2** (30 customers · 1.3%): 8.4 txns/wk · $51,600/mo · 4.2 yr · "
        "28 alerted · 9 SARs — highest per-customer monthly throughput\n"
        "- **Cluster 3** (37 customers · 1.6%): 3.1 txns/wk · $11,200/mo · 3.1 yr · "
        "34 alerted · 7 SARs — youngest accounts\n"
        "- **Cluster 4** (31 customers · 1.3%): 6.5 txns/wk · $37,800/mo · 9.7 yr · "
        "29 alerted · 8 SARs — most tenured"
    )},
    {"role": "user",      "content": prev_context(_BUS_STATS) + "\nwhich cluster has the most SARs"},
    {"role": "assistant", "content": (
        "Cluster 1 has the most SARs with **717 SAR customers** out of 2,105 alerted "
        "(34.0% SAR rate), driven by its dominant size of 2,271 customers."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# DS_V53_3 — Individual initial clustering via ds_cluster_analysis
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Cluster Individual customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ds53_3", "ds_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ds53_3", "content": _IND_DS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Individual Customer Segments\n\n"
        "- **Cluster 1** (6,674 customers · 63.3%): 1.2 txns/wk · $3,780/mo · 38.2 yr · "
        "3,317 alerted · 717 SARs · 2,600 FPs — largest segment\n"
        "- **Cluster 2** (1,842 customers · 17.5%): 4.3 txns/wk · $9,450/mo · 54.5 yr · "
        "1,205 alerted · 382 SARs — oldest, elevated activity\n"
        "- **Cluster 3** (1,127 customers · 10.7%): 2.8 txns/wk · $5,720/mo · 44.1 yr · "
        "743 alerted · 201 SARs\n"
        "- **Cluster 4** (798 customers · 7.6%): 7.6 txns/wk · $18,900/mo · 47.3 yr · "
        "651 alerted · 198 SARs — highest per-customer volume and SAR density"
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# DS_V53_4 — Individual multi-turn: ds_cluster_analysis + comparison follow-up
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Show me behavioral segments for Individual customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ds53_4", "ds_cluster_analysis", {"customer_type": "Individual", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ds53_4", "content": _IND_DS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Individual Customer Segments\n\n"
        "- **Cluster 1** (6,674 customers · 63.3%): 1.2 txns/wk · $3,780/mo · 38.2 yr · "
        "3,317 alerted · 717 SARs · 2,600 FPs\n"
        "- **Cluster 2** (1,842 customers · 17.5%): 4.3 txns/wk · $9,450/mo · 54.5 yr · "
        "1,205 alerted · 382 SARs — oldest accounts\n"
        "- **Cluster 3** (1,127 customers · 10.7%): 2.8 txns/wk · $5,720/mo · 44.1 yr · "
        "743 alerted · 201 SARs\n"
        "- **Cluster 4** (798 customers · 7.6%): 7.6 txns/wk · $18,900/mo · 47.3 yr · "
        "651 alerted · 198 SARs — highest activity and SAR density"
    )},
    {"role": "user",      "content": prev_context(_IND_STATS) + "\nhow does cluster 2 compare to cluster 4"},
    {"role": "assistant", "content": (
        "### Cluster 2 vs Cluster 4 — Individual\n\n"
        "**Transaction activity**: Cluster 4 is far more active — **7.6** weekly transactions "
        "vs **4.3** for Cluster 2, and **$18,900** monthly volume vs **$9,450**.\n\n"
        "**Account age**: Nearly identical — Cluster 2 averages **54.5 years**, Cluster 4 **47.3 years**.\n\n"
        "**AML risk**: Cluster 4 has higher per-customer risk — **198 SARs** across 798 customers "
        "(24.8% rate) vs **382 SARs** across 1,842 customers (20.7% rate) for Cluster 2. "
        "Cluster 4's high transaction frequency combined with elevated SAR density makes it "
        "the highest-risk individual segment."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# DS_V53_5 — All segment clustering via ds_cluster_analysis
# ═══════════════════════════════════════════════════════════════════════════

_ALL_STATS = """\
=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===
Segment: All | 4 clusters

**Cluster 1** (n=8,812):
- Avg Weekly Transactions: **1.2**
- Avg Weekly Txn Amount: **$1,240**
- Monthly Txn Volume: **$5,380**
- Account Age (years): **40.1**

**Cluster 2** (n=1,987):
- Avg Weekly Transactions: **4.1**
- Avg Weekly Txn Amount: **$2,980**
- Monthly Txn Volume: **$12,420**
- Account Age (years): **52.3**

**Cluster 3** (n=1,201):
- Avg Weekly Transactions: **2.9**
- Avg Weekly Txn Amount: **$1,780**
- Monthly Txn Volume: **$7,640**
- Account Age (years): **43.7**

**Cluster 4** (n=810):
- Avg Weekly Transactions: **7.4**
- Avg Weekly Txn Amount: **$9,650**
- Monthly Txn Volume: **$42,300**
- Account Age (years): **11.2**
=== END PRE-COMPUTED CLUSTER STATS ===

Alert/SAR/FP counts per cluster (customer-level, matches treemap):
Cluster 1: Alerts: 5,214 | SARs: 1,388 | FPs: 3,826
Cluster 2: Alerts: 2,148 | SARs: 584 | FPs: 1,564
Cluster 3: Alerts: 982 | SARs: 274 | FPs: 708
Cluster 4: Alerts: 368 | SARs: 112 | FPs: 256"""

_ALL_DS_TOOL_RESULT = "Tool result for ds_cluster_analysis:\n" + _ALL_STATS

examples.append({"messages": [
    {"role": "system",    "content": SEGMENTATION_SYSTEM},
    {"role": "user",      "content": "Cluster all customers by transaction behavior"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("ds53_5", "ds_cluster_analysis", {"customer_type": "All", "n_clusters": 4})
    ]},
    {"role": "tool", "tool_call_id": "ds53_5", "content": _ALL_DS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### All Customer Segments\n\n"
        "- **Cluster 1** (8,812 customers · 67.6%): 1.2 txns/wk · $5,380/mo · 40.1 yr · "
        "5,214 alerted · 1,388 SARs · 3,826 FPs — dominant low-activity segment\n"
        "- **Cluster 2** (1,987 customers · 15.2%): 4.1 txns/wk · $12,420/mo · 52.3 yr · "
        "2,148 alerted · 584 SARs — older, moderately active\n"
        "- **Cluster 3** (1,201 customers · 9.2%): 2.9 txns/wk · $7,640/mo · 43.7 yr · "
        "982 alerted · 274 SARs\n"
        "- **Cluster 4** (810 customers · 6.2%): 7.4 txns/wk · $42,300/mo · 11.2 yr · "
        "368 alerted · 112 SARs — highest-volume, youngest accounts"
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# RL_V53_1 — list_rules: top by precision (entry-point reinforcement)
# ═══════════════════════════════════════════════════════════════════════════
# Adds another list_rules → ranked text response to re-anchor text generation.

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rules have the highest precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("rl53_1", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "rl53_1", "content": _PC_LIST_A_V52},
    {"role": "assistant", "content": (
        "### Highest Precision Rules\n\n"
        "The top three rules by SAR precision are "
        "**Round-trip** (35.0%), **Risky International Transfer** (26.7%), and "
        "**Funnel Account** (26.0%) — "
        "these generate fewer false positives per SAR than any other rule in the system."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# RL_V53_2 — list_rules: FP-ranked overview (different entry phrasing)
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show me a performance overview of all AML rules"},
    {"role": "assistant", "content": None, "tool_calls": [tc("rl53_2", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "rl53_2", "content": _PC_LIST_A_V52},
    {"role": "assistant", "content": (
        "### AML Rule Performance Overview\n\n"
        "Rule performance summary — detailed table shown below.\n\n"
        "Highest false positive burden: **CTR Client** (348 FP), "
        "**Detect Excessive Transaction Activity** (301 FP), **Elder Abuse** (341 FP). "
        "Best precision: **Round-trip** (35.0%), **Risky International Transfer** (26.7%)."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# RL_V53_3 — list_rules: bottom 3 directly asked (no prior top-N turn)
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Which rules have the lowest precision?"},
    {"role": "assistant", "content": None, "tool_calls": [tc("rl53_3", "list_rules", {})]},
    {"role": "tool", "tool_call_id": "rl53_3", "content": _PC_LIST_A_V52},
    {"role": "assistant", "content": (
        "### Lowest Precision Rules\n\n"
        "The three rules with the lowest precision are "
        "**Activity Deviation (Check)** (13.0%), **CTR Client** (13.0%), and "
        "**Detect Excessive Transaction Activity** (14.0%) — "
        "these generate the highest investigator workload relative to confirmed SARs "
        "and are the primary candidates for threshold tightening."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# AT_V53_1 — cluster_threshold_analysis Individual — per-cluster bullets
# ═══════════════════════════════════════════════════════════════════════════
# AT_V52_1 covered Business. This adds Individual so the per-cluster bullets
# pattern is anchored across both segments.

_AT_IND_TOOL_RESULT = """\
Tool result for cluster_threshold_analysis:
=== CLUSTER THRESHOLD ANALYSIS ===
Segment: **Individual** | Column: AVG_TRXNS_WEEK (Avg Weekly Transactions) | Clusters: 4 | Target SAR catch: >=90%

**Cluster 1 — Low Activity** (156 customers, SAR pool: 52)
- Uniform 1.20: TP=48, FP=96, TP rate=92.3%, precision=33.3%
- Recommended 0.80: TP=48, FP=84, TP rate=92.3%, precision=36.4% (-12 FP, +0 SAR)

**Cluster 2 — Moderate** (89 customers, SAR pool: 35)
- Uniform 1.20: TP=32, FP=51, TP rate=91.4%, precision=38.6%
- Recommended 1.60: TP=32, FP=44, TP rate=91.4%, precision=42.1% (-7 FP, +0 SAR)

**Cluster 3 — Active** (72 customers, SAR pool: 28)
- Uniform 1.20: TP=26, FP=40, TP rate=92.9%, precision=39.4%
- Recommended 2.00: TP=25, FP=33, TP rate=89.3%, precision=43.1% (-7 FP, -1 SAR)

**Cluster 4 — High Volume** (47 customers, SAR pool: 19)
- Uniform 1.20: TP=18, FP=26, TP rate=94.7%, precision=40.9%
- Recommended 3.00: TP=18, FP=22, TP rate=94.7%, precision=45.0% (-4 FP, +0 SAR)

**ADAPTIVE SENSITIVITY SUMMARY**
- Uniform 1.20 applied to all Individual: TP=124, FP=213, TP rate=92.5%
- Cluster-adaptive thresholds: TP=123, FP=183, TP rate=91.8%
- Net change: **-30 FP**, -1 SARs (91.8% SAR retention)
=== END CLUSTER THRESHOLD ANALYSIS ===
"""[:-1]

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run cluster-adaptive threshold analysis for Individual customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("at53_1", "cluster_threshold_analysis", {
            "segment": "Individual",
            "threshold_column": "AVG_TRXNS_WEEK",
            "n_clusters": 4,
            "target_sar_rate": 0.90,
        })
    ]},
    {"role": "tool", "tool_call_id": "at53_1", "content": _AT_IND_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### Adaptive Thresholds — Individual | Avg Weekly Transactions\n\n"
        "- **Cluster 1 (Low Activity):** lower to 0.80 → −12 FP, 92.3% SAR retention\n"
        "- **Cluster 2 (Moderate):** raise to 1.60 → −7 FP, 91.4% SAR retention\n"
        "- **Cluster 3 (Active):** raise to 2.00 → −7 FP, 89.3% SAR retention\n"
        "- **Cluster 4 (High Volume):** raise to 3.00 → −4 FP, 94.7% SAR retention\n\n"
        "Net: **−30 FP** across all clusters at 91.8% SAR retention."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# SAR_V53_1 — sar_backtest Business: default to AVG_TRXNS_WEEK without asking
# ═══════════════════════════════════════════════════════════════════════════
# Teach: "Show SAR backtest for Business customers" with no column specified
# → call sar_backtest(segment="Business", threshold_column="AVG_TRXNS_WEEK").
# Do NOT ask for clarification.

_SAR_BUS_TOOL_RESULT = """\
Tool result for sar_backtest:
=== SAR BACKTEST ===
### SAR Backtest — Business / AVG_TRXNS_WEEK

**Population:** 304 alerted customers | **SARs:** 45 (14.8% SAR filing rate)

**Sweep Results**
- At lowest threshold (0.1): **45 SARs caught** (100%), 0 missed
- SARs first missed at threshold **1.8** (1 missed)
- To keep >=90% SAR catch rate: threshold <= **3.2** (41 of 45 caught)
- To keep >=80% SAR catch rate: threshold <= **5.5** (37 of 45 caught)
- To keep >=50% SAR catch rate: threshold <= **8.9** (23 of 45 caught)
- At highest threshold (14.2): **12 caught**, 33 missed

*(Detailed sweep chart shown below.)*
=== END SAR BACKTEST ==="""

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show SAR backtest for Business customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("sar53_1", "sar_backtest", {
            "segment": "Business",
            "threshold_column": "AVG_TRXNS_WEEK",
        })
    ]},
    {"role": "tool", "tool_call_id": "sar53_1", "content": _SAR_BUS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### SAR Backtest — Business | Avg Weekly Transactions\n\n"
        "At a threshold of **3.2** weekly transactions, **41 of 45 SARs** are caught (91.1% catch rate) "
        "with 14.8% precision — raising to 5.5 drops SAR coverage to 82.2% (8 missed), "
        "which exceeds the recommended 90% minimum; "
        "the current threshold of 1.8 captures all SARs but generates the maximum false positive load."
    )},
]})

# Variant: different phrasing, same default behavior
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run SAR backtest on Business segment"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("sar53_2", "sar_backtest", {
            "segment": "Business",
            "threshold_column": "AVG_TRXNS_WEEK",
        })
    ]},
    {"role": "tool", "tool_call_id": "sar53_2", "content": _SAR_BUS_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### SAR Backtest — Business | Avg Weekly Transactions\n\n"
        "The Business segment has **45 SAR customers** among 304 alerted accounts (14.8% filing rate). "
        "Keeping AVG_TRXNS_WEEK at or below **3.2** catches **41 SARs** (91.1%) — "
        "the recommended operating point before meaningful detection degradation occurs."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# SAR_V53_2 — sar_backtest Individual: same default pattern
# ═══════════════════════════════════════════════════════════════════════════

_SAR_IND_TOOL_RESULT = """\
Tool result for sar_backtest:
=== SAR BACKTEST ===
### SAR Backtest — Individual / AVG_TRXNS_WEEK

**Population:** 1,853 alerted customers | **SARs:** 317 (17.1% SAR filing rate)

**Sweep Results**
- At lowest threshold (0.1): **317 SARs caught** (100%), 0 missed
- SARs first missed at threshold **0.9** (1 missed)
- To keep >=90% SAR catch rate: threshold <= **2.4** (286 of 317 caught)
- To keep >=80% SAR catch rate: threshold <= **4.1** (254 of 317 caught)
- To keep >=50% SAR catch rate: threshold <= **7.3** (159 of 317 caught)
- At highest threshold (11.8): **94 caught**, 223 missed

*(Detailed sweep chart shown below.)*
=== END SAR BACKTEST ==="""

examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show SAR backtest for Individual customers"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("sar53_3", "sar_backtest", {
            "segment": "Individual",
            "threshold_column": "AVG_TRXNS_WEEK",
        })
    ]},
    {"role": "tool", "tool_call_id": "sar53_3", "content": _SAR_IND_TOOL_RESULT},
    {"role": "assistant", "content": (
        "### SAR Backtest — Individual | Avg Weekly Transactions\n\n"
        "The Individual segment has **317 SAR customers** among 1,853 alerted accounts (17.1% filing rate). "
        "At a threshold of **2.4** weekly transactions, **286 SARs** are caught (90.2%) — "
        "pushing to 4.1 drops coverage to 80.1% (254 caught, 63 missed); "
        "2.4 is the recommended ceiling to maintain >=90% detection."
    )},
]})


# ═══════════════════════════════════════════════════════════════════════════
# Combine + write
# ═══════════════════════════════════════════════════════════════════════════

def main():
    v52_base = []
    if V52_FULL_PATH.exists():
        with open(V52_FULL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v52_base.append(json.loads(line))
        print(f"[V53] Loaded {len(v52_base)} base examples from {V52_FULL_PATH.name}")
    else:
        print(f"[V53] WARNING: {V52_FULL_PATH.name} not found — writing V53-only")

    all_examples = v52_base + examples

    with open(V53_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V53] V53-only:  {V53_ONLY_PATH.name} ({len(examples)} examples)")

    with open(V53_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V53] Combined:  {V53_FULL_PATH.name} ({len(all_examples)} total)")


if __name__ == "__main__":
    main()

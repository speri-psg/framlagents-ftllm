"""
V41 training examples (2026-05-08).

Targets — gaps discovered during aria-v4 app testing:

  Y1–Y4   Gap 1: Conceptual threshold tuning questions routed to threshold agent
           (Rule 26 capped response at 2-3 sentences).
           Fix: classifier routes "what is threshold tuning" type → policy.

  Y5–Y8   Gap 2: Conceptual dynamic segmentation questions routed to segmentation
           agent (Rule 1 calls a tool instead of explaining).
           Fix: classifier routes "explain dynamic segmentation" type → policy.

  Y9–Y12  Gap 3: Multi-cluster comparison failed when [PREVIOUS CLUSTERING RESULT]
           only contained the most recently discussed single cluster.

  Z1–Z4   Gap 4: sar_backtest had no training examples — model improvised LaTeX
           math notation ($\\ge$, $\\le$) and leaked "Tool result for..." prefix.
           Fix: 4 sar_backtest examples showing correct verbatim copy with ≥/≤.
           Fix: app now injects the full original clustering result; model needs
           training to compare two clusters from a full 4-cluster context block.

Combined: aria_train_combined_v40_full.jsonl (base) + 16 V41 = aria_train_combined_v41_full.jsonl
"""

import json, pathlib, sys

DATA_DIR      = pathlib.Path(__file__).parent
V40_BASE_PATH = DATA_DIR / "aria_train_combined_v40_full.jsonl"
V41_ONLY_PATH = DATA_DIR / "aria_train_v41.jsonl"
V41_FULL_PATH = DATA_DIR / "aria_train_combined_v41_full.jsonl"

# Import THRESHOLD_SYSTEM verbatim from write_v40 (needed for sar_backtest examples)
sys.path.insert(0, str(DATA_DIR))
from write_v40 import THRESHOLD_SYSTEM  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }

def prev_context(stats):
    return f"[PREVIOUS CLUSTERING RESULT]\n{stats}\n[END PREVIOUS RESULT]"

# ---------------------------------------------------------------------------
# System prompts — copied verbatim from agents/*.py (do not abbreviate)
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
You are a routing classifier for ARIA. Given a user query, respond with \
one or more of these labels (comma-separated, no other text):

  threshold    — user wants to RUN threshold tuning analysis on OUR LOCAL DATA (FP/FN trade-off charts, sweep analysis)
  segmentation — user wants to RUN clustering/segmentation on OUR LOCAL DATA (K-Means, treemap, behavioral groups)
  ofac         — user wants to RUN OFAC sanctions screening on OUR LOCAL CUSTOMER DATA (SDN list hits, sanctioned country exposure)
  policy       — user is asking a GENERAL KNOWLEDGE question about AML, compliance, regulations, industry practices, or best practices — does NOT require running local data analysis
  greeting     — query is a greeting or social pleasantry (hello, hi, how are you, etc.)
  out_of_scope — query is not related to any of the above AML topics

Key distinction:
- "Show FP/FN tuning for Business customers" → threshold  (run local analysis)
- "Show FP/FN threshold tuning for Individual customers" → threshold
- "Run SAR backtest for Individual customers" → threshold  (SAR backtest is a threshold tool)
- "What threshold catches 90% of SARs?" → threshold
- "SAR catch rate for Business monthly transaction amount" → threshold
- "Run SAR backtest" → threshold
- "Show me a 2D grid for Activity Deviation ACH" → threshold  (2D sweep is a threshold tool)
- "How do floor amount and sigma interact for Activity Deviation?" → threshold
- "Show me the ACH deviation rule performance" → threshold
- "What is the SAR catch rate for Activity Deviation Check?" → threshold
- "Show the heatmap for Elder Abuse" → threshold
- "2D analysis for Velocity Single" → threshold
- "How does time window interact with floor amount for Detect Excessive?" → threshold
- "Show me the AML rule performance overview" → threshold  (list_rules is a threshold tool)
- "Which rules generate the most false positives?" → threshold
- "What is the SAR catch rate for the Activity Deviation rule?" → threshold
- "Show rule-level FP analysis" → threshold
- "What happens to FP if I raise the age threshold for Elder Abuse?" → threshold
- "How do banks manage alert volumes?" → policy  (general knowledge question)
- "What is AML?" → policy  (general knowledge question)
- "What is threshold tuning?" → policy  (conceptual overview of the practice — NOT a request to run analysis on local data)
- "Explain threshold tuning" → policy  (explanation request — NOT running local data)
- "How does threshold tuning work?" → policy
- "Can you explain what threshold tuning means?" → policy
- "What is dynamic segmentation?" → policy  (conceptual question — NOT a request to run clustering)
- "Explain dynamic segmentation" → policy
- "How does behavioral segmentation work?" → policy
- "What is K-Means clustering?" → policy
- "What is customer segmentation?" → policy
- "Cluster all customers" → segmentation  (run local analysis)
- "What does AML policy say about structuring?" → policy  (general knowledge + knowledge base)
- "Show alerts and false positive distribution across segments" → segmentation  (distribution chart, NOT threshold tuning)
- "Show alert distribution" → segmentation
- "How are alerts spread across segments?" → segmentation
- "Which segment has the most alerts?" → segmentation
- "What is the average transaction amount for Business customers?" → threshold  (segment_stats tool)
- "How many alerts does the Individual segment have?" → threshold  (segment_stats tool)
- "What are the transaction stats for Business customers?" → threshold
- "Show me Business customer stats" → threshold
- "Show me all AML rules" → threshold  (list_rules is a threshold tool — NOT policy)
- "What rules are in the system?" → threshold
- "List all the AML rules" → threshold
- "What transactions are flagged by the layering rule?" → threshold  (list_rules — 'layering' is not a KB topic)
- "Which rule covers layering?" → threshold  (list_rules)
- "Show rule sweep for xyz_column" → threshold  (rule sweep request, even with unknown param — NOT out_of_scope)
- "Show rule sweep for an invalid parameter" → threshold
- "What is the SAR filing rate for Individual?" → threshold  (sar_backtest is a threshold tool)
- "SAR filing rate for Business" → threshold
- "Which rule has the highest FP rate?" → threshold  (list_rules)
- "Which rules generate only false positives?" → threshold
- "Run a SAR backtest for the structuring rule" → threshold  (rule_sar_backtest — NOT out_of_scope)
- "SAR backtest for Elder Abuse" → threshold
- "Show Elder Abuse sweep for Cluster 4" → threshold  (cluster-filtered rule sweep)
- "Run SAR backtest for Activity Deviation ACH in Cluster 2" → threshold
- "Show 2D heatmap for Elder Abuse for Cluster 3" → threshold
- "Which cluster has the most false positives for Velocity Single?" → threshold
- "Which cluster of Business customers has the highest transaction volume?" → segmentation
- "Which Business cluster has the most activity?" → segmentation
- "Which cluster has the most transaction activity?" → segmentation
- "Show Business customer clusters by transaction behavior" → segmentation
- "Run OFAC screening" → ofac
- "Show OFAC sanctions exposure" → ofac
- "Which customers are on the sanctions list?" → ofac
- "How many customers are from sanctioned countries?" → ofac
- "Show me OFAC hits" → ofac
- "Screen customers against SDN list" → ofac
- "What is our Iran/North Korea customer exposure?" → ofac
- "Show comprehensive sanctions hits" → ofac
- "Show me a 2D grid for Elder Abuse" → threshold  (2D grid = 2D sweep, same tool)
- "Show 2D analysis for Detect Excessive Transaction Activity" → threshold  (2D analysis = 2D sweep)
- "Run a 2D grid analysis for Velocity Single" → threshold
- "Show grid analysis for Activity Deviation ACH" → threshold
- "What are Canada's suspicious transaction reporting requirements?" → policy
- "What are Canada's AML rules?" → policy
- "What does FINTRAC require?" → policy
- "What is AML structuring?" → policy  (prefix 'AML' does not change the topic — still a policy question)
- "What is tructuring?" → policy  (typo for 'structuring' — still an AML definition question)
- "What is smurfing?" → policy  (synonym for structuring — AML definition question)
- "What is AML layering?" → policy
- "What is AML typology?" → policy
- "cluster into 3 groups" → segmentation  (user specifying cluster count is still a segmentation request)
- "I only want 2 business clusters" → segmentation
- "show me 4 clusters for Individual customers" → segmentation
- "I want k-means with 3 clusters" → segmentation
- "What are the EU requirements for beneficial ownership registers?" → policy  (EU regulatory question)
- "What does the 4th AMLD require for customer due diligence?" → policy
- "What does the 5th AMLD say about virtual assets?" → policy
- "What are FATF recommendations for banks?" → policy
- "What does UN Security Council Resolution 1373 require of banks?" → policy
- "What are EBA guidelines on ML/TF risk factors?" → policy
- "What are the beneficial ownership disclosure requirements?" → policy
- "What does the EU AML Regulation require?" → policy
- "What is the AMLA?" → policy
- "Does UNODC have guidance on AML?" → policy
- "What are PEP requirements under AML regulations?" → policy
- "Thanks, that was helpful!" → greeting
- "Thanks, that's great" → greeting
- "Got it, thanks" → greeting
- "Thank you" → greeting
- "That was useful, thanks" → greeting
- "Can you send this to my compliance team?" → out_of_scope  (action request, not an AML analysis task)
- "Can you email this to someone?" → out_of_scope
- "Can you export this as a PDF?" → out_of_scope
- "What is a false positive?" → threshold  (FP/FN definition is a threshold concept, not policy)
- "What is a false negative?" → threshold
- "What is the difference between FP and FN?" → threshold
- "Explain false positives in AML monitoring" → threshold
- "What does FP mean?" → threshold
- "Can you explain false positives and false negatives?" → threshold
- "What is a 2D grid?" → threshold  (2D grid = rule_2d_sweep — a threshold tool concept)
- "What is a 2D sweep?" → threshold
- "How does a 2D grid work?" → threshold
- "Are you ARIA?" → greeting  (identity question — not an AML topic)
- "What is your name?" → greeting
- "Who are you?" → greeting
- "Ahoy!" → greeting
- "Ahoy matey!" → greeting
- "What are true positives in AML monitoring?" → threshold  (TP/TN definitions are threshold/confusion-matrix concepts)
- "What are true negatives?" → threshold
- "What is the difference between TP and TN?" → threshold
- "What is OFAC?" → policy  (definition question — NOT a screening request)
- "What does OFAC stand for?" → policy
- "My dog OFAC met a cat the other day" → out_of_scope  (OFAC here is a name, not AML topic)
- "OFAC said hello" → out_of_scope  (not an AML query)
- "Is OFAC the same as sanctions screening?" → policy  (terminology question — NOT a screening request)
- "What does OFAC stand for?" → policy  (terminology question)
- "What is OFAC?" → policy
- "What are the rules that have z_threshold as a parameter?" → threshold  (list_rules — filter by parameter)
- "Which rule shows the highest SAR count?" → threshold  (list_rules tool)

Rules:
- Output ONLY the label(s), comma-separated. No explanation, no punctuation other than commas.
- A query can map to multiple labels (e.g. threshold,segmentation).
- When in doubt between out_of_scope and a AML label, prefer the AML label.\
"""

SEGMENTATION_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML. You identify natural customer behavioral "
    "segments using unsupervised K-Means clustering and explain their AML risk profiles. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "RULES — follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer segmentation or cluster questions from memory.\n"
    "2. For clustering with rich demographics (preferred) — call ds_cluster_analysis.\n"
    "3. For alert/FP distribution by segment — call alerts_distribution.\n"
    "4. For the legacy alerts dataset — call cluster_analysis only if the user explicitly asks.\n"
    "5. Do NOT call multiple segmentation tools for the same request — pick exactly one.\n"
    "6. customer_type must be exactly one of: Business, Individual, All\n"
    "   If the user does NOT specify a customer type, default to All.\n"
    "7. n_clusters must be an integer 2-8, or 0 to auto-select. Default is 4.\n"
    "   If the user says \"N clusters\", \"into N\", \"only N\", or \"I want N\" (e.g. \"cluster into 3\", \"I only want 2 clusters\"),\n"
    "   set n_clusters=N exactly in the tool call. Do NOT ignore the user's requested count and do NOT default to 4.\n"
    "8. If the user asks to prepare or refresh the raw data — call prepare_segmentation_data first.\n"
    "9. After receiving tool results, copy the cluster stats verbatim, then add ONE sentence describing the highest-risk cluster based solely on the numbers in the tool result. Do NOT suggest thresholds, dollar cutoffs, or monitoring actions.\n"
    "10. If the user asks to show specific clusters (e.g. \"show only cluster 3\", \"highest risk\",\n"
    "    \"top 2 high risk\", \"low activity clusters\"):\n"
    "    - Identify which cluster number(s) match the request from the stats\n"
    "      (highest avg_trxn_amt = highest risk, lowest avg_num_trxns = lowest activity, etc.)\n"
    "    - On the VERY LAST LINE of your response, write EXACTLY this and nothing else:\n"
    "      DISPLAY_CLUSTERS: N\n"
    "      where N is a comma-separated list of cluster numbers (e.g. DISPLAY_CLUSTERS: 4  or  DISPLAY_CLUSTERS: 1,4)\n"
    "    - Do NOT mention this line in your text — it is a system directive, not for the user.\n"
    "    If the user does NOT ask to filter, do NOT include a DISPLAY_CLUSTERS line.\n"
    "11. Do NOT include JSON, code blocks, or raw data tables in your final reply.\n"
    "12. ONLY use numbers that appear in the tool result. Do NOT invent, estimate, or calculate new numbers.\n"
    "13. Do NOT invent threshold values, dollar amounts, or cutoffs. Only reference numbers explicitly present in the tool result. Do NOT suggest specific threshold values (e.g. \"$250K\", \"< 80,000\") unless they appear verbatim in the tool result.\n"
    "14. If the user asks which cluster to set a threshold for, or asks for threshold recommendations per cluster — do NOT invent values. Tell the user to use the threshold_tuning or sar_backtest tools with the relevant segment instead.\n"
    "15. If a [PREVIOUS CLUSTERING RESULT] block is provided in the context AND the user is asking to characterize, describe, compare, or explain a specific cluster — answer from that data WITHOUT calling any tool. Compare the named cluster's stats (avg_trxn_amt, monthly volume, balance, account age) against the other clusters to identify what makes it distinctive. Name the risk profile in one sentence. Do NOT re-run clustering."
)

# Business cluster stats block — matches what the model returns after ds_cluster_analysis
STATS_BIZ = """\
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
- Account Age (years): **4.3**"""

# ---------------------------------------------------------------------------
# V41 examples
# ---------------------------------------------------------------------------

examples = []

# ═══════════════════════════════════════════════════════════════════════════
# GAP 1 — "What is threshold tuning?" → policy, not threshold (Y1–Y4)
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "What is threshold tuning?"},
    {"role": "assistant", "content": "policy"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "Explain threshold tuning"},
    {"role": "assistant", "content": "policy"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "How does threshold tuning work?"},
    {"role": "assistant", "content": "policy"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "Can you explain what threshold tuning means in AML?"},
    {"role": "assistant", "content": "policy"},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 2 — "What is dynamic segmentation?" → policy, not segmentation (Y5–Y8)
# ═══════════════════════════════════════════════════════════════════════════

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "What is dynamic segmentation?"},
    {"role": "assistant", "content": "policy"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "Explain dynamic segmentation"},
    {"role": "assistant", "content": "policy"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "How does behavioral segmentation work in AML?"},
    {"role": "assistant", "content": "policy"},
]})

examples.append({"messages": [
    {"role": "system",    "content": CLASSIFY_SYSTEM},
    {"role": "user",      "content": "What is K-Means clustering and why is it used in AML?"},
    {"role": "assistant", "content": "policy"},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 3 — Multi-cluster comparison from full [PREVIOUS CLUSTERING RESULT] (Y9–Y12)
# ═══════════════════════════════════════════════════════════════════════════

# Y9: Compare Cluster 1 vs Cluster 4 (most extreme pair)
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\nAre Cluster 1 and Cluster 4 similar in risk profile?"},
    {"role": "assistant", "content": (
        "No — Cluster 1 and Cluster 4 have meaningfully different risk profiles.\n\n"
        "**Cluster 1** (1,114 customers, 31.4% of active accounts): avg weekly transaction amount $30,487.4, "
        "monthly volume $300,181.8, current balance $42,310.5, account age 6.2 years.\n\n"
        "**Cluster 4** (512 customers, 14.4% of active accounts): avg weekly transaction amount $17,441.0, "
        "monthly volume $121,217.7, current balance $17,842.6, account age 4.3 years.\n\n"
        "Cluster 1 is the highest-activity group with the largest balances and most established accounts; "
        "Cluster 4 is the smallest, lowest-activity group with the youngest accounts — making them the most "
        "distinct risk profiles in this segmentation."
    )},
]})

# Y10: Compare Cluster 2 vs Cluster 3 (adjacent mid-tier pair)
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\nHow does Cluster 2 compare to Cluster 3?"},
    {"role": "assistant", "content": (
        "Cluster 2 and Cluster 3 are adjacent mid-tier groups, but Cluster 2 shows consistently higher "
        "activity across every dimension.\n\n"
        "**Cluster 2** (1,088 customers, 30.6%): avg weekly transaction amount $27,690.0, monthly volume "
        "$246,301.4, current balance $38,204.1, account age 5.8 years.\n\n"
        "**Cluster 3** (837 customers, 23.6%): avg weekly transaction amount $24,505.9, monthly volume "
        "$189,046.9, current balance $29,118.3, account age 5.1 years.\n\n"
        "Cluster 2 carries higher AML risk — its transaction amounts, monthly volume, and balances are all "
        "consistently higher than Cluster 3's, and its accounts are slightly more established at 5.8 years average age."
    )},
]})

# Y11: Which clusters are most similar to each other?
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\nWhich clusters are most similar to each other?"},
    {"role": "assistant", "content": (
        "Based on the previous clustering result, Cluster 1 and Cluster 2 are the most similar pair.\n\n"
        "Both share the same avg weekly transaction count (13.4) and are the two largest groups by customer "
        "count (1,114 and 1,088 respectively). Their transaction amounts ($30,487.4 vs $27,690.0), monthly "
        "volumes ($300,181.8 vs $246,301.4), and balances ($42,310.5 vs $38,204.1) are the closest of any "
        "cluster pair in this segmentation.\n\n"
        "Cluster 3 and Cluster 4 are both lower-activity than Clusters 1 and 2, but differ from each other "
        "in transaction amounts, balance, and account maturity."
    )},
]})

# Y12: Difference between Cluster 1 and Cluster 3
examples.append({"messages": [
    {"role": "system", "content": SEGMENTATION_SYSTEM},
    {"role": "user",   "content": prev_context(STATS_BIZ) + "\n\nWhat is the difference between Cluster 1 and Cluster 3?"},
    {"role": "assistant", "content": (
        "Cluster 1 and Cluster 3 share the same avg weekly transaction count (13.4) but differ in "
        "transaction amounts, monthly volume, and account maturity.\n\n"
        "**Cluster 1** (1,114 customers, 31.4%): avg weekly transaction amount $30,487.4, monthly volume "
        "$300,181.8, current balance $42,310.5, account age 6.2 years.\n\n"
        "**Cluster 3** (837 customers, 23.6%): avg weekly transaction amount $24,505.9, monthly volume "
        "$189,046.9, current balance $29,118.3, account age 5.1 years.\n\n"
        "The key distinction is transaction value and account maturity — Cluster 1 processes higher-value "
        "transactions with larger balances and accounts averaging 6.2 years old, compared to Cluster 3's "
        "5.1-year average, placing Cluster 1 at materially higher AML risk."
    )},
]})

# ═══════════════════════════════════════════════════════════════════════════
# GAP 4 — sar_backtest format: verbatim copy with ≥/≤, no "Tool result" prefix (Z1–Z4)
# ═══════════════════════════════════════════════════════════════════════════

_SAR_BT_IND_TRXNS_BODY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
### SAR Backtest — Individual / AVG_TRXNS_WEEK

**Population:** 2,167 alerted customers | **SARs:** 466 (21.5% SAR filing rate)

**Sweep Results**
- At lowest threshold (0.54): **466 SARs caught** (100%), 0 missed
- SARs first missed at threshold **1.54** (92 missed)
- To keep ≥90% SAR catch rate: threshold ≤ **0.54** (420 of 466 caught)
- To keep ≥80% SAR catch rate: threshold ≤ **1.54** (373 of 466 caught)
- To keep ≥50% SAR catch rate: threshold ≤ **1.54** (234 of 466 caught)
- At highest threshold (12.54): **18 caught**, 448 missed

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED SAR BACKTEST ==="""

_SAR_BT_BIZ_TRXNS_BODY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
### SAR Backtest — Business / AVG_TRXNS_WEEK

**Population:** 1,384 alerted customers | **SARs:** 312 (22.5% SAR filing rate)

**Sweep Results**
- At lowest threshold (0.62): **312 SARs caught** (100%), 0 missed
- SARs first missed at threshold **2.62** (47 missed)
- To keep ≥90% SAR catch rate: threshold ≤ **0.62** (281 of 312 caught)
- To keep ≥80% SAR catch rate: threshold ≤ **2.62** (250 of 312 caught)
- To keep ≥50% SAR catch rate: threshold ≤ **4.62** (157 of 312 caught)
- At highest threshold (14.62): **22 caught**, 290 missed

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED SAR BACKTEST ==="""

_SAR_BT_IND_AMT_BODY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
### SAR Backtest — Individual / TRXN_AMT_MONTHLY

**Population:** 2,167 alerted customers | **SARs:** 466 (21.5% SAR filing rate)

**Sweep Results**
- At lowest threshold (120.0): **466 SARs caught** (100%), 0 missed
- SARs first missed at threshold **1,120.0** (38 missed)
- To keep ≥90% SAR catch rate: threshold ≤ **120.0** (420 of 466 caught)
- To keep ≥80% SAR catch rate: threshold ≤ **2,120.0** (374 of 466 caught)
- To keep ≥50% SAR catch rate: threshold ≤ **5,120.0** (234 of 466 caught)
- At highest threshold (24,120.0): **14 caught**, 452 missed

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED SAR BACKTEST ==="""

_SAR_BT_BIZ_AMT_BODY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
### SAR Backtest — Business / TRXN_AMT_MONTHLY

**Population:** 1,384 alerted customers | **SARs:** 312 (22.5% SAR filing rate)

**Sweep Results**
- At lowest threshold (850.0): **312 SARs caught** (100%), 0 missed
- SARs first missed at threshold **3,850.0** (29 missed)
- To keep ≥90% SAR catch rate: threshold ≤ **850.0** (281 of 312 caught)
- To keep ≥80% SAR catch rate: threshold ≤ **4,850.0** (250 of 312 caught)
- To keep ≥50% SAR catch rate: threshold ≤ **9,850.0** (157 of 312 caught)
- At highest threshold (32,850.0): **11 caught**, 301 missed

*(Detailed sweep chart shown below.)*
=== END PRE-COMPUTED SAR BACKTEST ==="""

PC_SAR_BT_IND_TRXNS = "Tool result for sar_backtest:\n" + _SAR_BT_IND_TRXNS_BODY
PC_SAR_BT_BIZ_TRXNS = "Tool result for sar_backtest:\n" + _SAR_BT_BIZ_TRXNS_BODY
PC_SAR_BT_IND_AMT   = "Tool result for sar_backtest:\n" + _SAR_BT_IND_AMT_BODY
PC_SAR_BT_BIZ_AMT   = "Tool result for sar_backtest:\n" + _SAR_BT_BIZ_AMT_BODY

# Z1: Individual / AVG_TRXNS_WEEK
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run SAR backtest for Individual customers using average weekly transactions"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("z1", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"})
    ]},
    {"role": "tool", "tool_call_id": "z1", "content": PC_SAR_BT_IND_TRXNS},
    {"role": "assistant", "content": (
        _SAR_BT_IND_TRXNS_BODY + "\n\n"
        "At the current threshold of 0.54 weekly transactions every alerted SAR customer "
        "is caught — raising beyond 1.54 causes the first meaningful SAR losses."
    )},
]})

# Z2: Business / AVG_TRXNS_WEEK
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Show SAR backtest for Business customers by average weekly transactions"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("z2", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"})
    ]},
    {"role": "tool", "tool_call_id": "z2", "content": PC_SAR_BT_BIZ_TRXNS},
    {"role": "assistant", "content": (
        _SAR_BT_BIZ_TRXNS_BODY + "\n\n"
        "Business customers show a 22.5% SAR filing rate — "
        "the threshold can be raised to 2.62 weekly transactions before SAR losses exceed 15%."
    )},
]})

# Z3: Individual / TRXN_AMT_MONTHLY
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "SAR catch rate for Individual customers using monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("z3", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"})
    ]},
    {"role": "tool", "tool_call_id": "z3", "content": PC_SAR_BT_IND_AMT},
    {"role": "assistant", "content": (
        _SAR_BT_IND_AMT_BODY + "\n\n"
        "Monthly transaction amount is a weaker separator than weekly frequency — "
        "the ≥90% catch rate requires holding the threshold at just $120.0, "
        "indicating SAR customers span a wide range of monthly volumes."
    )},
]})

# Z4: Business / TRXN_AMT_MONTHLY
examples.append({"messages": [
    {"role": "system",    "content": THRESHOLD_SYSTEM},
    {"role": "user",      "content": "Run SAR backtest for Business segment using monthly transaction amount"},
    {"role": "assistant", "content": None, "tool_calls": [
        tc("z4", "sar_backtest", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"})
    ]},
    {"role": "tool", "tool_call_id": "z4", "content": PC_SAR_BT_BIZ_AMT},
    {"role": "assistant", "content": (
        _SAR_BT_BIZ_AMT_BODY + "\n\n"
        "Business SAR customers are distributed across a wide monthly volume range — "
        "raising the threshold above $9,850.0 cuts the SAR catch rate below 50%, "
        "making conservative threshold setting critical for this segment."
    )},
]})

# ---------------------------------------------------------------------------
# Combine V40 base + V41 new examples and write
# ---------------------------------------------------------------------------

def main():
    # Write V41-only file
    with open(V41_ONLY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[V41] V41-only: {V41_ONLY_PATH.name} ({len(examples)} examples)")

    # Attempt to combine with V40 base
    if V40_BASE_PATH.exists():
        v40_base = []
        with open(V40_BASE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v40_base.append(json.loads(line))
        print(f"[V41] Loaded {len(v40_base)} base examples from {V40_BASE_PATH.name}")
        all_examples = v40_base + examples
        with open(V41_FULL_PATH, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[V41] Combined: {V41_FULL_PATH.name} ({len(all_examples)} total)")
    else:
        print(f"[V41] WARNING: V40 base not found at {V40_BASE_PATH}")
        print(f"[V41] V41-only file written — combine manually if needed.")


if __name__ == "__main__":
    main()

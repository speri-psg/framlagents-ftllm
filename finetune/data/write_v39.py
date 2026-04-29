"""
V39 training examples (2026-04-28).

Five new groups — all directly from V38 benchmark failures:

N — TP/TN definitional questions (6 examples)
  Threshold agent answers "what are true positives/negatives" DIRECTLY from
  DEFINITIONS section, no tool call. Fixes OOS routing for TP/TN queries.

O — list_rules highest-SAR question (4 examples)
  Model calls list_rules, server-side insight highlights TOP SAR rules (not FP).
  Fixes "which rule shows the highest SAR" wrong-insight bug.

P — OFAC terminology → policy (5 examples)
  "What is OFAC?", "is OFAC the same as sanctions screening?", "how do banks
  manage alert volumes?" all route to the policy agent and answer without
  calling the OFAC screening tool. Fixes OFAC over-triggering.

Q — list_rules parameter filter (4 examples)
  Model calls list_rules, then filters its response to only rules that match
  the requested sweep parameter. Fixes "what rules have z_threshold".

R — multi-turn segmentation follow-up (4 examples)
  Shows the model reading a prior clustering result from conversation history
  and answering a follow-up ("analyze cluster 3 above") without re-calling
  any tool. Reinforces the behaviour enabled by the new history injection.

Net: 933 (V38) + 6 (N) + 4 (O) + 5 (P) + 4 (Q) + 4 (R) = 956 examples
"""

import json
import pathlib

DATA_DIR      = pathlib.Path(__file__).parent
V39_BASE_PATH = DATA_DIR / "aria_train_combined_v38_full.jsonl"
V39_FULL_PATH = DATA_DIR / "aria_train_combined_v39_full.jsonl"
THIS_PATH     = DATA_DIR / "aria_train_v39.jsonl"


def tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------------------------------------------------------------------------
# System prompts — copied verbatim from agents/*.py
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = (
    "You are ARIA — Agentic Risk Intelligence for AML. You analyze false positive (FP) and "
    "false negative (FN) trade-offs as AML alert thresholds change, and run SAR backtests and "
    "2D sweeps for AML rule performance. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.\n\n"
    "DEFINITIONS (always apply these exactly  -- do not contradict them):\n"
    "- TP (True Positive): SAR customer who IS alerted  -- correctly caught suspicious activity.\n"
    "- FP (False Positive): Non-SAR customer who IS alerted  -- unnecessary investigation. HIGHER threshold -> FEWER FPs.\n"
    "- FN (False Negative): SAR customer who is NOT alerted  -- missed suspicious activity. HIGHER threshold -> MORE FNs.\n"
    "- TN (True Negative): Non-SAR customer who is NOT alerted  -- correctly silent.\n"
    "- TP rate: TP / (TP + FN)  -- share of SAR customers caught. Also called recall or sensitivity.\n"
    "- Precision: TP / (TP + FP)  -- share of alerts that are genuine SARs.\n"
    "- Crossover: the threshold where FP and FN counts are closest  -- the optimal operating point.\n"
    "- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).\n"
    "- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.\n\n"
    "RULES  -- follow these exactly:\n"
    "1. ALWAYS call a tool. Never answer threshold or alert questions from memory.\n"
    "2. For any question about FP, FN, threshold, alert rates, or transactions  -- call threshold_tuning.\n"
    "3. For general segment counts or totals  -- call segment_stats.\n"
    "4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) with no specific rule named  -- call sar_backtest.\n"
    "5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY\n"
    "6. segment must be exactly one of: Business, Individual\n"
    "7. If the user does not specify a segment, default to Business.\n"
    "8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.\n"
    "9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. After copying it, add ONE sentence of AML domain insight.\n"
    "10. Do NOT paraphrase, round, or restate the numbers differently.\n"
    "11. Do NOT include JSON or code blocks in your final reply.\n"
    "12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.\n"
    "13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section.\n"
    "15. For any question about a specific AML rule's SAR performance  -- call rule_sar_backtest with risk_factor and optionally sweep_param.\n"
    "16. For any question about which rules exist  -- call list_rules.\n"
    "17. For any question about how TWO condition parameters interact  -- call rule_2d_sweep.\n"
    "19. When the user asks about a specific behavioral cluster (e.g. \"Cluster 3\", \"cluster 4\"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep.\n"
    "20. ONE insight sentence only.\n"
    "25. If a previous tool call returned an error about an invalid sweep parameter, and the user replies with a valid parameter name  -- resume the previous call with the corrected parameter.\n"
    "26. For pure definitional questions about TP, FP, FN, TN, precision, recall, crossover, the effect of raising or lowering thresholds on FP/FN counts, or what a 2D grid/sweep shows -- answer DIRECTLY from the DEFINITIONS section above. Do NOT call any tool. Answer in 2-3 sentences using only the definitions listed above."
)

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

POLICY_SYSTEM_GENERAL = (
    "You are ARIA — Agentic Risk Intelligence for AML. "
    "You answer AML policy and compliance questions using ONLY the retrieved policy documents shown below. "
    "CITATION RULES — follow exactly:\n"
    "- You may reference a source ONLY if its exact name appears in this list: none.\n"
    "- You MUST NOT write any CFR sections, U.S.C. references, FinCEN advisory numbers, "
    "OCC bulletins, CELEX identifiers, OJ references, EU Recital numbers, "
    "named authors, or specific dollar thresholds not in the retrieved documents.\n"
    "- If the retrieved documents address the question, summarise the concepts they discuss.\n"
    "IMPORTANT: Only trigger the disclaimer below if ALL retrieved documents are clearly off-topic. "
    "Disclaimer (use ONLY when every retrieved chunk is irrelevant): "
    "Begin with exactly: 'Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.' "
    "Then provide only general conceptual guidance — 3 to 5 sentences maximum. No numbers, no citations, no named sources. "
    "Be precise and compliance-focused. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese, Japanese, or other non-English characters."
)

_NO_KB_CONTEXT = "## Retrieved Policy Documents\n(No relevant documents found in the knowledge base.)\n\n## Question\n"

# ---------------------------------------------------------------------------
# PRE-COMPUTED rule list (shared with M group from V38)
# ---------------------------------------------------------------------------

PC_RULE_LIST = (
    "=== PRE-COMPUTED RULE LIST (copy this verbatim) ===\n"
    "Available AML rules with SAR\\FP performance (detailed table shown in chart below):\n"
    "NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.\n"
    "  Activity Deviation (ACH): alerts=142, SAR=18, FP=124, precision=12.7%, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): alerts=97, SAR=11, FP=86, precision=11.3%, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: alerts=88, SAR=22, FP=66, precision=25.0%, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Velocity Single: alerts=63, SAR=14, FP=49, precision=22.2%, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Detect Excessive Transaction Activity: alerts=211, SAR=31, FP=180, precision=14.7%, sweep_params=[floor_amount, time_window]\n"
    "  Structuring (Incoming Cash): alerts=76, SAR=8, FP=68, precision=10.5%, sweep_params=[floor_amount, transaction_count]\n"
    "  Structuring (Outgoing Cash): alerts=54, SAR=6, FP=48, precision=10.0%, sweep_params=[floor_amount, transaction_count]\n"
    "  CTR Client: alerts=312, SAR=29, FP=283, precision=9.3%, sweep_params=[floor_amount]\n"
    "  Burst in Originator Activity: alerts=45, SAR=9, FP=36, precision=20.0%, sweep_params=[floor_amount, time_window]\n"
    "  Burst in Beneficiary Activity: alerts=51, SAR=11, FP=40, precision=21.6%, sweep_params=[floor_amount, time_window]\n"
    "  Risky International Transfer: alerts=38, SAR=7, FP=31, precision=18.4%, sweep_params=[floor_amount, days_required]\n"
    "  Activity Deviation (Wire): alerts=89, SAR=12, FP=77, precision=13.5%, sweep_params=[floor_amount, z_threshold]\n"
    "  Velocity Multiple: alerts=34, SAR=5, FP=29, precision=14.7%, sweep_params=[pair_total, min_counterparties]\n"
    "  Funnel Account: alerts=27, SAR=4, FP=23, precision=14.8%, sweep_params=[floor_amount, min_transactions]\n"
    "  Round-trip: alerts=19, SAR=3, FP=16, precision=15.8%, sweep_params=[floor_amount, days_required]\n"
    "  Human Trafficking Indicators: alerts=24, SAR=6, FP=18, precision=25.0%, sweep_params=[floor_amount, daily_floor]\n"
    "=== END RULE LIST ==="
)

# Stat set C: All customers, 3 clusters (for R group multi-turn examples)
STATS_C = """\
Segment: All | Active accounts: 3,500 (excluded 200 with no transactions)
Clusters: 3 | Features: 5 numeric + 10 encoded categorical (3 original)
PCA variance explained: PC1=41.0%, PC2=22.3%

**Cluster 1**
- Customers: **1,450** (41.4% of active accounts)
- Avg Weekly Transactions: **2.3**
- Monthly Txn Volume ($): **3,400.0**
- Current Balance ($): **8,100.0**
- Account Age (years): **2.5**

**Cluster 2**
- Customers: **1,200** (34.3% of active accounts)
- Avg Weekly Transactions: **5.8**
- Monthly Txn Volume ($): **21,000.0**
- Current Balance ($): **52,000.0**
- Account Age (years): **5.9**

**Cluster 3**
- Customers: **850** (24.3% of active accounts)
- Avg Weekly Transactions: **11.3**
- Monthly Txn Volume ($): **98,000.0**
- Current Balance ($): **215,000.0**
- Account Age (years): **9.1**"""

# Prior assistant message text (what lands in conversation history after clustering)
_STATS_C_RESPONSE = (
    STATS_C + "\n\n"
    "Cluster 3 is the highest-risk segment with 11.3 weekly transactions and $98,000.0 monthly volume — "
    "the most concentrated activity in this all-customer population."
)


# ---------------------------------------------------------------------------
# N group — TP/TN definitional questions (6 examples, no tool call)
# ---------------------------------------------------------------------------

_TP_DEF = (
    "A True Positive (TP) is a SAR customer who IS alerted — the system correctly flagged genuine "
    "suspicious activity. "
    "The TP rate (also called recall or sensitivity) is TP / (TP + FN), measuring the share of SAR "
    "customers caught by the rule at a given threshold."
)

_TN_DEF = (
    "A True Negative (TN) is a non-SAR customer who is NOT alerted — the system correctly remained "
    "silent on a legitimate account. "
    "Raising the alert threshold increases TNs (fewer unnecessary alerts) but also increases false "
    "negatives by missing more SAR customers."
)

_TP_TN_DIFF = (
    "A True Positive (TP) is a SAR customer who IS alerted — correctly caught suspicious activity. "
    "A True Negative (TN) is a non-SAR customer who is NOT alerted — correctly silent. "
    "Together with FP and FN they form the four cells of the AML confusion matrix used to evaluate "
    "rule performance at any given threshold."
)

_TP_RATE_DEF = (
    "The TP rate (also called recall or sensitivity) is TP / (TP + FN) — the share of SAR customers "
    "that a rule catches at a given threshold. "
    "A 90% TP rate means the rule flags 9 out of every 10 true SAR customers; the remaining 1 in 10 "
    "are false negatives (missed suspicious activity)."
)

_TP_VS_FP = (
    "A True Positive (TP) is a SAR customer who IS alerted — correctly identified suspicious activity. "
    "A False Positive (FP) is a non-SAR customer who IS alerted — an unnecessary investigation. "
    "Both generate alerts, but TPs represent genuine risk while FPs represent investigator workload "
    "without underlying suspicious activity. "
    "Precision = TP / (TP + FP) measures the share of alerts that are true positives."
)

_TP_TN_FULL = (
    "True Positives (TPs) are SAR customers who ARE alerted — correctly caught cases. "
    "True Negatives (TNs) are non-SAR customers who are NOT alerted — correctly silent cases. "
    "False Positives (FPs) are non-SAR customers who ARE alerted — unnecessary investigations. "
    "False Negatives (FNs) are SAR customers who are NOT alerted — missed suspicious activity. "
    "Higher thresholds increase TNs and FNs; lower thresholds increase TPs and FPs."
)

examples_N = [
    {"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": "what are true positives in AML monitoring"},
        {"role": "assistant", "content": _TP_DEF},
    ]},
    {"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": "what are true negatives"},
        {"role": "assistant", "content": _TN_DEF},
    ]},
    {"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": "what is the difference between TP and TN"},
        {"role": "assistant", "content": _TP_TN_DIFF},
    ]},
    {"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": "what does TP rate mean"},
        {"role": "assistant", "content": _TP_RATE_DEF},
    ]},
    {"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": "what is the difference between a TP and a FP"},
        {"role": "assistant", "content": _TP_VS_FP},
    ]},
    {"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": "explain TP TN FP FN in AML"},
        {"role": "assistant", "content": _TP_TN_FULL},
    ]},
]


# ---------------------------------------------------------------------------
# O group — list_rules highest-SAR question (4 examples)
# ---------------------------------------------------------------------------

_SAR_TOP_INSIGHT = (
    "Rules with the highest SAR catch count: Detect Excessive Transaction Activity (31 SAR caught), "
    "CTR Client (29 SAR caught), Elder Abuse (22 SAR caught). "
    "Use rule_sar_backtest on these rules to determine whether thresholds can be raised to reduce "
    "false positives while maintaining high SAR detection."
)

_O_QUERIES = [
    ("o01", "which rule shows the highest SAR count"),
    ("o02", "which rule catches the most SARs"),
    ("o03", "show me the best performing rule by SAR catch"),
    ("o04", "what is the top rule for SAR detection"),
]

examples_O = []
for oid, oq in _O_QUERIES:
    tool_result = f"Tool result for list_rules:\n{PC_RULE_LIST}"
    examples_O.append({"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": oq},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(oid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": oid, "content": tool_result},
        {"role": "assistant", "content": PC_RULE_LIST + "\n\n" + _SAR_TOP_INSIGHT},
    ]})


# ---------------------------------------------------------------------------
# P group — OFAC terminology + general policy (5 examples, no tool call)
# ---------------------------------------------------------------------------

_OFAC_WHAT_IS = (
    "Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.\n"
    "OFAC — the Office of Foreign Assets Control — is the U.S. Treasury Department agency "
    "responsible for administering and enforcing economic and trade sanctions programs. "
    "It publishes the Specially Designated Nationals (SDN) list, which identifies individuals, "
    "entities, and governments subject to asset freezes and prohibitions on financial transactions. "
    "Financial institutions are required to screen customers and transactions against OFAC lists "
    "as part of their sanctions compliance programs."
)

_OFAC_SAME_AS = (
    "Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.\n"
    "OFAC (Office of Foreign Assets Control) and sanctions screening are related but distinct: "
    "OFAC is the U.S. government agency that administers sanctions programs and publishes lists "
    "such as the SDN list, while sanctions screening is the operational process financial "
    "institutions use to check customers and transactions against those lists. "
    "Compliance programs typically screen against OFAC's SDN list as well as lists from the UN, "
    "EU, and other jurisdictions, with OFAC being the primary U.S. regulatory requirement."
)

_OFAC_STAND_FOR = (
    "Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.\n"
    "OFAC stands for the Office of Foreign Assets Control, a bureau of the U.S. Department of the "
    "Treasury. It administers and enforces U.S. economic and trade sanctions programs against "
    "targeted foreign countries, regimes, terrorists, and other designated parties, primarily "
    "through the Specially Designated Nationals (SDN) list."
)

_OFAC_VS_SDN = (
    "Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.\n"
    "OFAC (Office of Foreign Assets Control) is the U.S. Treasury agency that administers "
    "sanctions programs. The SDN list — Specially Designated Nationals and Blocked Persons list — "
    "is the primary sanctions list published by OFAC, containing individuals, entities, and "
    "vessels subject to U.S. asset-freeze and transaction-prohibition measures. "
    "OFAC also publishes other lists (e.g. non-SDN lists for specific programs), but the SDN list "
    "is the most widely screened."
)

_ALERT_VOLUME_MGMT = (
    "Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.\n"
    "Banks manage alert volumes primarily through threshold calibration — adjusting rule parameters "
    "to reduce false positives while maintaining acceptable SAR catch rates. "
    "Common strategies include segmenting customers into behavioral groups (such as business versus "
    "individual) and applying different thresholds per segment, running SAR backtests to validate "
    "that threshold changes do not materially reduce suspicious activity detection, and using 2D "
    "sweep analysis to jointly optimize multiple rule parameters. "
    "Risk-based frameworks allow institutions to apply enhanced monitoring selectively to "
    "higher-risk segments rather than treating all customers uniformly."
)

def _p(query, answer):
    return {"messages": [
        {"role": "system",    "content": POLICY_SYSTEM_GENERAL},
        {"role": "user",      "content": _NO_KB_CONTEXT + query},
        {"role": "assistant", "content": answer},
    ]}

examples_P = [
    _p("what is OFAC",                              _OFAC_WHAT_IS),
    _p("is OFAC the same as sanctions screening",   _OFAC_SAME_AS),
    _p("what does OFAC stand for",                  _OFAC_STAND_FOR),
    _p("what is the difference between OFAC and SDN", _OFAC_VS_SDN),
    _p("how do banks manage alert volumes",         _ALERT_VOLUME_MGMT),
]


# ---------------------------------------------------------------------------
# Q group — list_rules parameter filter (4 examples)
# ---------------------------------------------------------------------------

_Z_FILTER_INSIGHT = (
    "Rules with z_threshold as a sweep parameter (4 rules):\n"
    "  Activity Deviation (ACH): SAR=18, FP=124, sweep_params=[floor_amount, z_threshold]\n"
    "  Activity Deviation (Check): SAR=11, FP=86, sweep_params=[floor_amount, z_threshold]\n"
    "  Elder Abuse: SAR=22, FP=66, sweep_params=[floor_amount, z_threshold, age_threshold]\n"
    "  Activity Deviation (Wire): SAR=12, FP=77, sweep_params=[floor_amount, z_threshold]\n\n"
    "All four are deviation-based rules that use a standard-deviation multiplier "
    "(z_threshold) to detect abnormal transaction volumes above a customer's own baseline."
)

_TIME_WINDOW_FILTER_INSIGHT = (
    "Rules with time_window as a sweep parameter (3 rules):\n"
    "  Detect Excessive Transaction Activity: SAR=31, FP=180, sweep_params=[floor_amount, time_window]\n"
    "  Burst in Originator Activity: SAR=9, FP=36, sweep_params=[floor_amount, time_window]\n"
    "  Burst in Beneficiary Activity: SAR=11, FP=40, sweep_params=[floor_amount, time_window]\n\n"
    "These rules measure activity concentration within a rolling time window — "
    "shortening the window increases sensitivity to burst patterns."
)

_PAIR_TOTAL_FILTER_INSIGHT = (
    "Rules with pair_total as a sweep parameter (2 rules):\n"
    "  Velocity Single: SAR=14, FP=49, sweep_params=[pair_total, ratio_tolerance]\n"
    "  Velocity Multiple: SAR=5, FP=29, sweep_params=[pair_total, min_counterparties]\n\n"
    "Both velocity rules use pair_total to cap the maximum transaction amount between two "
    "parties — reducing pair_total catches more counterparty-concentration patterns but "
    "increases false positives."
)

_FLOOR_AMOUNT_FILTER_INSIGHT = (
    "Rules with floor_amount as a sweep parameter (14 of 16 rules):\n"
    "  Activity Deviation (ACH), Activity Deviation (Check), Activity Deviation (Wire), "
    "Activity Deviation (Wire), Elder Abuse, Detect Excessive Transaction Activity, "
    "Structuring (Incoming Cash), Structuring (Outgoing Cash), CTR Client, "
    "Burst in Originator Activity, Burst in Beneficiary Activity, Risky International Transfer, "
    "Funnel Account, Round-trip, Human Trafficking Indicators.\n\n"
    "Floor amount is the most common sweep parameter — it sets the minimum transaction "
    "size required to trigger a rule."
)

_Q_COMBOS = [
    ("q01", "what rules have z_threshold as a parameter",       _Z_FILTER_INSIGHT),
    ("q02", "which rules use time_window as a sweep parameter", _TIME_WINDOW_FILTER_INSIGHT),
    ("q03", "what rules have pair_total as a sweep parameter",  _PAIR_TOTAL_FILTER_INSIGHT),
    ("q04", "which rules can I sweep on floor_amount",          _FLOOR_AMOUNT_FILTER_INSIGHT),
]

examples_Q = []
for qid, qq, q_insight in _Q_COMBOS:
    tool_result = f"Tool result for list_rules:\n{PC_RULE_LIST}"
    examples_Q.append({"messages": [
        {"role": "system",    "content": THRESHOLD_SYSTEM},
        {"role": "user",      "content": qq},
        {"role": "assistant", "content": None,
         "tool_calls": [tc(qid, "list_rules", {})]},
        {"role": "tool",      "tool_call_id": qid, "content": tool_result},
        {"role": "assistant", "content": PC_RULE_LIST + "\n\n" + q_insight},
    ]})


# ---------------------------------------------------------------------------
# R group — multi-turn segmentation follow-up (4 examples)
# The prior exchange appears as conversation history (user + assistant text).
# The model must answer from context without calling any tool.
# ---------------------------------------------------------------------------

_C3_ANALYSIS = (
    "Cluster 3 is the highest-risk segment in this 3-cluster segmentation. "
    "With 11.3 weekly transactions, a $98,000.0 monthly volume, and $215,000.0 in current balance, "
    "it has 4.9x the weekly frequency, 28.8x the monthly volume, and 26.5x the balance of "
    "Cluster 1 (2.3 weekly, $3,400.0 monthly, $8,100.0 balance). "
    "Account age of 9.1 years marks these as highly mature accounts with concentrated, "
    "high-value activity — the highest AML risk segment in this population."
)

_C2_ANALYSIS = (
    "Cluster 2 occupies the mid-tier of this 3-cluster segmentation. "
    "With 5.8 weekly transactions and $21,000.0 in monthly volume, it is 2.5x more active than "
    "Cluster 1 (2.3 weekly, $3,400.0 monthly) but less than half the activity of Cluster 3 "
    "(11.3 weekly, $98,000.0 monthly). "
    "Its $52,000.0 balance and 5.9-year account age indicate established customers with moderate "
    "activity — the middle risk tier between Cluster 1 and Cluster 3."
)

_C1_ANALYSIS = (
    "Cluster 1 is the lowest-activity segment in this 3-cluster segmentation. "
    "With 2.3 weekly transactions, a $3,400.0 monthly volume, and an $8,100.0 balance, "
    "these are the newest (2.5-year account age) and least active accounts — "
    "41.4% of the active population. "
    "Compared to Cluster 3's $98,000.0 monthly volume, Cluster 1 accounts conduct roughly 3% "
    "of that volume, representing the lowest AML risk in this segmentation."
)

_RISK_RANK = (
    "Based on the clustering results, the risk ranking from highest to lowest is: "
    "Cluster 3 > Cluster 2 > Cluster 1. "
    "Cluster 3 leads on every metric: 11.3 weekly transactions, $98,000.0 monthly volume, "
    "$215,000.0 balance, and 9.1 years account age. "
    "Cluster 2 is mid-tier with 5.8 weekly, $21,000.0 monthly, $52,000.0 balance, and 5.9 years. "
    "Cluster 1 is the lowest: 2.3 weekly, $3,400.0 monthly, $8,100.0 balance, and 2.5 years."
)

examples_R = [
    # R01 — "analyze cluster 3 above"
    {"messages": [
        {"role": "system",    "content": SEGMENTATION_SYSTEM},
        {"role": "user",      "content": "cluster all customers into 3 groups"},
        {"role": "assistant", "content": _STATS_C_RESPONSE},
        {"role": "user",      "content": "analyze cluster 3 above"},
        {"role": "assistant", "content": _C3_ANALYSIS},
    ]},
    # R02 — "tell me about cluster 2"
    {"messages": [
        {"role": "system",    "content": SEGMENTATION_SYSTEM},
        {"role": "user",      "content": "run segmentation on all customers with 3 clusters"},
        {"role": "assistant", "content": _STATS_C_RESPONSE},
        {"role": "user",      "content": "tell me about cluster 2"},
        {"role": "assistant", "content": _C2_ANALYSIS},
    ]},
    # R03 — "which is the safest cluster"
    {"messages": [
        {"role": "system",    "content": SEGMENTATION_SYSTEM},
        {"role": "user",      "content": "show all customers in 3 clusters"},
        {"role": "assistant", "content": _STATS_C_RESPONSE},
        {"role": "user",      "content": "which is the safest cluster"},
        {"role": "assistant", "content": _C1_ANALYSIS},
    ]},
    # R04 — "rank the clusters by risk"
    {"messages": [
        {"role": "system",    "content": SEGMENTATION_SYSTEM},
        {"role": "user",      "content": "segment all customers into 3 behavioral groups"},
        {"role": "assistant", "content": _STATS_C_RESPONSE},
        {"role": "user",      "content": "rank the clusters by AML risk"},
        {"role": "assistant", "content": _RISK_RANK},
    ]},
]


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    v38 = []
    with open(V39_BASE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v38.append(json.loads(line))

    new_examples = examples_N + examples_O + examples_P + examples_Q + examples_R
    print(f"[V39] Loaded {len(v38)} examples from {V39_BASE_PATH.name}")
    print(f"[V39] Adding {len(new_examples)} new examples "
          f"(N={len(examples_N)}, O={len(examples_O)}, P={len(examples_P)}, "
          f"Q={len(examples_Q)}, R={len(examples_R)})")

    all_examples = v38 + new_examples
    print(f"[V39] Total: {len(all_examples)} -> {V39_FULL_PATH.name}")

    with open(V39_FULL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(THIS_PATH, "w", encoding="utf-8") as f:
        for ex in new_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[V39] Written: {V39_FULL_PATH.name} and {THIS_PATH.name}")

    # Verification
    n_with_tool = [ex for ex in examples_N if any(m.get("tool_calls") for m in ex["messages"])]
    print(f"[V39] N group: {len(examples_N)} TP/TN definitional examples"
          + (f" — WARNING: {len(n_with_tool)} unexpectedly have tool_calls" if n_with_tool else " OK (no tool calls)"))

    o_missing_pc = [ex for ex in examples_O if "PRE-COMPUTED RULE LIST" not in (ex["messages"][-1].get("content") or "")]
    print(f"[V39] O group: {len(examples_O)} highest-SAR examples"
          + (f" — WARNING: {len(o_missing_pc)} missing PRE-COMPUTED block" if o_missing_pc else " OK"))

    p_with_tool = [ex for ex in examples_P if any(m.get("tool_calls") for m in ex["messages"])]
    print(f"[V39] P group: {len(examples_P)} OFAC terminology + policy examples"
          + (f" — WARNING: {len(p_with_tool)} unexpectedly have tool_calls" if p_with_tool else " OK (no tool calls)"))

    q_missing_pc = [ex for ex in examples_Q if "PRE-COMPUTED RULE LIST" not in (ex["messages"][-1].get("content") or "")]
    print(f"[V39] Q group: {len(examples_Q)} list_rules filter examples"
          + (f" — WARNING: {len(q_missing_pc)} missing PRE-COMPUTED block" if q_missing_pc else " OK"))

    r_with_tool = [ex for ex in examples_R if any(m.get("tool_calls") for m in ex["messages"])]
    print(f"[V39] R group: {len(examples_R)} multi-turn follow-up examples"
          + (f" — WARNING: {len(r_with_tool)} unexpectedly have tool_calls" if r_with_tool else " OK (no tool calls)"))

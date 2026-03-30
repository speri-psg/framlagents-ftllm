"""Generate V3 training examples (Gaps 11–16) and append to framl_train_failures_v2.jsonl."""
import json, os

OUT = os.path.join(os.path.dirname(__file__), "data", "framl_train_failures_v2.jsonl")

SYS_GENERAL = (
    "You are a FRAML (Fraud + AML) analytics AI assistant. You analyze false positive/false negative "
    "trade-offs in AML alert thresholds, perform customer behavioral segmentation, and interpret "
    "clustering results. Use the available tools to retrieve data, then provide clear, analytical "
    "insights. Be concise and reference specific numbers when interpreting results. Respond only in "
    "English \u2014 never output Chinese or any non-English characters. Call each tool exactly once per request."
)

SYS_POLICY = (
    "You are a FRAML policy and compliance specialist. "
    "You answer questions by referencing AML policies, regulatory guidelines, and best practices "
    "retrieved from the knowledge base. "
    "Always cite the source document when referencing policy content. "
    "When the knowledge base contains relevant content, cite the source document. "
    "When the retrieved documents do not contain relevant content, you MUST: "
    "1. Begin your response with exactly: 'Note: The knowledge base does not contain specific guidance "
    "on this topic. The following is general AML knowledge only.' "
    "2. Do NOT cite any statute numbers (U.S.C., CFR, etc.), document codes, advisory numbers, or URLs. "
    "3. Provide only general conceptual guidance without specific legal references. "
    "Only cite sources when the exact source document name appears in the retrieved policy documents "
    "shown above. Be precise and compliance-focused."
)

def tool_msg(call_id, name, args_dict):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args_dict)}
        }]
    }

def tool_result(call_id, content):
    return {"role": "tool", "content": content, "tool_call_id": call_id}

def ex(system, messages):
    return {"messages": [{"role": "system", "content": system}] + messages}

# ── PRE-COMPUTED blocks (reuse realistic numbers from production) ─────────────

BIZ_MONTHLY = """\
PRE-COMPUTED ANALYSIS:
At the lowest threshold (52), there are 1,486 false positives.
False positives decrease as the threshold rises.
False negatives are zero for all thresholds from 52 up to and including 8,943.
False negatives first become non-zero at threshold 9,392 (FN=2).
False negatives increase as the threshold continues to rise.
The crossover point is at threshold 26,992 (FP=205, FN=181).
The optimal zone spans threshold 22,449 to 31,535.
At threshold 26,992: FP=205, FN=181.
At the highest threshold, FP=0, FN=181."""

IND_AVG_AMT = """\
PRE-COMPUTED ANALYSIS:
At the lowest threshold (50), there are 1,857 false positives.
False positives decrease as the threshold rises.
False negatives are zero for all thresholds from 50 up to and including 142.
False negatives first become non-zero at threshold 144 (FN=7).
False negatives increase as the threshold continues to rise.
The crossover point is at threshold 276 (FP=312, FN=298).
The optimal zone spans threshold 258 to 294.
At threshold 276: FP=312, FN=298.
At the highest threshold, FP=0, FN=542."""

IND_WEEKLY = """\
PRE-COMPUTED ANALYSIS:
At the lowest threshold (2), there are 2,341 false positives.
False positives decrease as the threshold rises.
False negatives are zero for all thresholds from 2 up to and including 8.
False negatives first become non-zero at threshold 9 (FN=3).
False negatives increase as the threshold continues to rise.
The crossover point is at threshold 18 (FP=187, FN=174).
The optimal zone spans threshold 15 to 21.
At threshold 18: FP=187, FN=174.
At the highest threshold, FP=0, FN=387."""

BIZ_AVG_AMT = """\
PRE-COMPUTED ANALYSIS:
At the lowest threshold (10), there are 1,124 false positives.
False positives decrease as the threshold rises.
False negatives are zero for all thresholds from 10 up to and including 287.
False negatives first become non-zero at threshold 301 (FN=4).
False negatives increase as the threshold continues to rise.
The crossover point is at threshold 892 (FP=143, FN=139).
The optimal zone spans threshold 741 to 1,043.
At threshold 892: FP=143, FN=139.
At the highest threshold, FP=0, FN=312."""

# ─────────────────────────────────────────────────────────────────────────────

examples = []

# ── Gap 11 (×2): FP/FN label swap ────────────────────────────────────────────

# 11a — Business TRXN_AMT_MONTHLY: highest threshold has FP=0, FN=181
# Correct response must NOT call FN=181 "false alarms" — false alarms = FPs
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Analyze threshold tuning for Business monthly transaction amount"},
    tool_msg("call_v3_ex01", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
    tool_result("call_v3_ex01", BIZ_MONTHLY),
    {"role": "assistant", "content": (
        BIZ_MONTHLY + "\n\n"
        "At the highest threshold, zero false positives means no unnecessary investigator reviews, "
        "but all 181 suspicious transactions remain undetected (FN=181) \u2014 a compliance risk "
        "acceptable only for institutions with strong compensating controls such as enhanced due "
        "diligence or peer monitoring programs."
    )}
]))

# 11b — Individual AVG_TRXN_AMT: highest threshold has FP=0, FN=542
# Correct: FP=0 means zero false alarms; FN=542 means 542 missed suspicious transactions
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "What\u2019s the threshold analysis for Individual average transaction amount?"},
    tool_msg("call_v3_ex02", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
    tool_result("call_v3_ex02", IND_AVG_AMT),
    {"role": "assistant", "content": (
        IND_AVG_AMT + "\n\n"
        "At the crossover threshold of 276, false positives (unnecessary investigator reviews) and "
        "false negatives (undetected suspicious transactions) are nearly balanced at 312 and 298 "
        "respectively; at the highest threshold FP=0 eliminates all false alarms but FN=542 means "
        "542 suspicious transactions go undetected \u2014 a material regulatory exposure."
    )}
]))

# ── Gap 12 (×2): Threshold direction error ────────────────────────────────────

# 12a — General policy: lowering threshold increases alert volume
examples.append(ex(SYS_POLICY, [
    {"role": "user", "content": "If I lower the alert threshold, will investigators have fewer alerts to review?"},
    {"role": "assistant", "content": (
        "No \u2014 lowering the alert threshold **increases** the number of alerts sent to investigators.\n\n"
        "**How threshold direction works:**\n"
        "- **Raising the threshold** \u2192 fewer transactions exceed the threshold \u2192 fewer alerts "
        "\u2192 fewer false positives \u2192 **reduced investigator workload** (but more false negatives \u2014 "
        "higher risk of missing suspicious activity).\n"
        "- **Lowering the threshold** \u2192 more transactions exceed the threshold \u2192 more alerts "
        "\u2192 more false positives \u2192 **increased investigator workload** (but fewer false negatives \u2014 "
        "lower risk of missing suspicious activity).\n\n"
        "To reduce investigator burden, the threshold should be **raised**, not lowered. "
        "The institution should document this risk-based decision, back-test the new threshold "
        "against historical SAR filings, and review it periodically to ensure suspicious activity "
        "is not being systematically missed."
    )}
]))

# 12b — Specific ask: we're overwhelmed with alerts
examples.append(ex(SYS_POLICY, [
    {"role": "user", "content": "We\u2019re overwhelmed with alerts. Should we raise or lower our transaction monitoring threshold to reduce the volume?"},
    {"role": "assistant", "content": (
        "**Raise the threshold.**\n\n"
        "**Why raising reduces alert volume:**\n"
        "A higher threshold means only transactions exceeding a larger value trigger alerts. "
        "Fewer transactions clear the bar \u2192 fewer alerts \u2192 fewer false positives \u2192 "
        "less investigator workload.\n\n"
        "**Why lowering makes it worse:**\n"
        "A lower threshold triggers alerts on more transactions, generating more false positives "
        "and compounding investigator burden.\n\n"
        "**AML risk trade-off:**\n"
        "Raising the threshold reduces false positives but increases false negatives \u2014 suspicious "
        "transactions below the new threshold will no longer generate alerts. Under a risk-based "
        "approach this is defensible if the institution documents the rationale, back-tests against "
        "SAR history, and maintains compensating controls (enhanced due diligence, peer group "
        "monitoring) for the segments where the threshold is raised.\n\n"
        "Use the threshold_tuning tool to identify the crossover point for your segment before "
        "finalizing the new threshold."
    )}
]))

# ── Gap 13 (×2): Model misreads PRE-COMPUTED FP/FN counts as transaction volumes ──

# 13a — Business TRXN_AMT_MONTHLY: FP=1,486 at lowest threshold
# Model must NOT say "1486 transactions" or "1486 accounts" — it is 1,486 false positive ALERTS
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Show me the Business monthly transaction amount threshold analysis"},
    tool_msg("call_v3_ex03", "threshold_tuning", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
    tool_result("call_v3_ex03", BIZ_MONTHLY),
    {"role": "assistant", "content": (
        BIZ_MONTHLY + "\n\n"
        "At the minimum threshold of $52/month, 1,486 false positive alerts would require "
        "investigator review \u2014 meaning 1,486 reviews of legitimate business activity; "
        "operating within the optimal zone of $22,449\u2013$31,535 monthly reduces the false "
        "positive alert count to the 181\u2013205 range, balancing review burden against "
        "missed detection risk."
    )}
]))

# 13b — Individual AVG_TRXNS_WEEK: FP=2,341 at lowest threshold
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Individual weekly transaction frequency threshold \u2014 give me the analysis"},
    tool_msg("call_v3_ex04", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"}),
    tool_result("call_v3_ex04", IND_WEEKLY),
    {"role": "assistant", "content": (
        IND_WEEKLY + "\n\n"
        "At the minimum threshold of 2 transactions per week, 2,341 false positive alerts "
        "would be generated \u2014 meaning 2,341 investigator reviews on accounts conducting "
        "legitimate activity; a threshold of 18 transactions per week reduces false positive "
        "alerts to 187 while missing 174 suspicious accounts, representing a symmetric "
        "operating point within the optimal zone."
    )}
]))

# ── Gap 14 (×2): Invented tool parameters ─────────────────────────────────────

# 14a — User asks for smaller step increments
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Can you re-run the analysis with smaller step increments between thresholds?"},
    {"role": "assistant", "content": (
        "The threshold_tuning tool does not accept parameters for step size or threshold range. "
        "The only available parameters are:\n"
        "- **segment**: `Business` or `Individual`\n"
        "- **threshold_column**: `AVG_TRXNS_WEEK`, `AVG_TRXN_AMT`, or `TRXN_AMT_MONTHLY`\n\n"
        "The step size and threshold range are determined by the underlying data. "
        "Finer-grained analysis would require changes to the data pipeline \u2014 outside the "
        "scope of this tool.\n\n"
        "If you want to re-run for a different segment or threshold column, specify one of the "
        "supported values above."
    )}
]))

# 14b — User asks to start from a specific dollar value
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Re-run the Business monthly amount analysis but start from $5,000 instead"},
    {"role": "assistant", "content": (
        "The threshold_tuning tool does not support a starting value or `threshold_min` parameter. "
        "The only parameters it accepts are:\n"
        "- **segment**: `Business` or `Individual`\n"
        "- **threshold_column**: `AVG_TRXNS_WEEK`, `AVG_TRXN_AMT`, or `TRXN_AMT_MONTHLY`\n\n"
        "The sweep always covers the full data-determined range. To re-run the standard Business "
        "monthly amount analysis call:\n\n"
        "`threshold_tuning(segment=\"Business\", threshold_column=\"TRXN_AMT_MONTHLY\")`\n\n"
        "A restricted-range analysis would require a change to the underlying data processing pipeline."
    )}
]))

# ── Gap 15 (×2): Verbatim copy reinforcement ──────────────────────────────────

# 15a — Business AVG_TRXN_AMT
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Run threshold tuning for Business average transaction amount"},
    tool_msg("call_v3_ex05", "threshold_tuning", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),
    tool_result("call_v3_ex05", BIZ_AVG_AMT),
    {"role": "assistant", "content": (
        BIZ_AVG_AMT + "\n\n"
        "Setting the Business AVG_TRXN_AMT threshold within the optimal zone of 741 to 1,043 "
        "keeps false positive alert volume well below the 1,124 maximum seen at the lowest "
        "threshold, and the crossover at 892 \u2014 where FP and FN are nearly equal at 143 and "
        "139 \u2014 represents a defensible risk-based starting point for AML program review."
    )}
]))

# 15b — Individual AVG_TRXNS_WEEK
examples.append(ex(SYS_GENERAL, [
    {"role": "user", "content": "Threshold analysis for Individual weekly transactions"},
    tool_msg("call_v3_ex06", "threshold_tuning", {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"}),
    tool_result("call_v3_ex06", IND_WEEKLY),
    {"role": "assistant", "content": (
        IND_WEEKLY + "\n\n"
        "For Individual customers, a weekly transaction frequency threshold of 18 is the crossover "
        "point where false positive alert reviews and missed suspicious transactions are nearly "
        "balanced at 187 and 174 respectively; institutions with limited investigative capacity "
        "may prefer thresholds above 18 to reduce alert volume, accepting the increased false "
        "negative risk."
    )}
]))

# ── Gap 16 (×2): Fabricated statute citations ─────────────────────────────────

# 16a — BSA penalty for senior management: KB unlikely to have this detail
examples.append(ex(SYS_POLICY, [
    {"role": "user", "content": "What are the penalties for willful BSA violations by senior management?"},
    {"role": "assistant", "content": (
        "Note: The knowledge base does not contain specific guidance on this topic. "
        "The following is general AML knowledge only.\n\n"
        "Willful violations of the Bank Secrecy Act by senior management can result in both "
        "civil and criminal consequences. At the civil level, financial regulators have applied "
        "substantial civil money penalties against both institutions and individual officers, "
        "with major enforcement actions reaching tens or hundreds of millions of dollars. "
        "Individual officers and employees can face personal civil liability separately from "
        "the institution.\n\n"
        "Criminal liability for willful BSA violations may result in fines and imprisonment for "
        "individuals. Senior managers who knowingly direct, participate in, or deliberately ignore "
        "patterns of non-compliance \u2014 particularly structuring, failure to file SARs, or "
        "operating without adequate AML controls \u2014 have faced personal criminal prosecution "
        "in notable enforcement cases.\n\n"
        "Penalties are assessed by FinCEN, federal bank regulators (OCC, Federal Reserve, FDIC), "
        "and in criminal matters, the Department of Justice. Severity depends on willfulness, "
        "duration of non-compliance, cooperation, and whether violations facilitated underlying "
        "financial crime."
    )}
]))

# 16b — SAR filing for cyber-enabled fraud: KB unlikely to have FinCEN cyber-specific guidance
examples.append(ex(SYS_POLICY, [
    {"role": "user", "content": "What are FinCEN\u2019s specific SAR filing requirements for cyber-enabled fraud?"},
    {"role": "assistant", "content": (
        "Note: The knowledge base does not contain specific guidance on this topic. "
        "The following is general AML knowledge only.\n\n"
        "FinCEN has indicated that cyber-enabled financial crime \u2014 including business email "
        "compromise, account takeover fraud, and ransomware-related transactions \u2014 should be "
        "reported via Suspicious Activity Reports when the activity meets the applicable filing "
        "threshold.\n\n"
        "For general SAR purposes, banks must file when a transaction involves at least $5,000 "
        "and the institution knows, suspects, or has reason to suspect the transaction involves "
        "funds from illegal activity, is designed to evade BSA requirements, or lacks a lawful "
        "purpose. Money services businesses have a lower $2,000 threshold.\n\n"
        "For cyber-related SARs, institutions are encouraged to use the cyber event indicators "
        "section of the SAR form and to include technical identifiers (IP addresses, device "
        "fingerprints, email headers) where available. FinCEN has encouraged sharing of "
        "cyber indicator information through information-sharing programs.\n\n"
        "When in doubt, if the institution cannot identify a lawful explanation for the "
        "transaction pattern, it should file."
    )}
]))

# ─────────────────────────────────────────────────────────────────────────────

with open(OUT, "a", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Appended {len(examples)} examples to {OUT}")

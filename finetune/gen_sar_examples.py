"""Generate SAR backtest training examples and append to framl_train_failures_v2.jsonl."""
import json, os

OUT = os.path.join(os.path.dirname(__file__), "data", "framl_train_failures_v2.jsonl")

SYS = (
    "You are a FRAML (Fraud + AML) analytics AI assistant. You analyze false positive/false negative "
    "trade-offs in AML alert thresholds, perform customer behavioral segmentation, and interpret "
    "clustering results. Use the available tools to retrieve data, then provide clear, analytical "
    "insights. Be concise and reference specific numbers when interpreting results. Respond only in "
    "English \u2014 never output Chinese or any non-English characters. Call each tool exactly once per request."
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

def ex(messages):
    return {"messages": [{"role": "system", "content": SYS}] + messages}

# ── PRE-COMPUTED SAR BACKTEST blocks (real numbers from compute_sar_backtest) ─

BIZ_MONTHLY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Business | Column: trxn_amt_monthly
Total simulated SARs: 44 out of 367 alerted customers (12.0% SAR filing rate).

At the lowest threshold (1397.0): 44 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 1151144.0 (27 missed).
To catch at least 90% of SARs, threshold must stay at or below 1397.0 (40 of 44 caught).
To catch at least 80% of SARs, threshold must stay at or below 1397.0 (36 of 44 caught).
To catch at least 50% of SARs, threshold must stay at or below 1397.0 (23 of 44 caught).
At the highest threshold (116125844.0): 0 SARs caught, 44 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

IND_MONTHLY = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: trxn_amt_monthly
Total simulated SARs: 640 out of 4668 alerted customers (13.7% SAR filing rate).

At the lowest threshold (485.0): 640 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 222054.0 (14 missed).
To catch at least 90% of SARs, threshold must stay at or below 222054.0 (577 of 640 caught).
To catch at least 80% of SARs, threshold must stay at or below 443623.0 (513 of 640 caught).
To catch at least 50% of SARs, threshold must stay at or below 886761.0 (321 of 640 caught).
At the highest threshold (22378954.0): 0 SARs caught, 640 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

IND_AMT = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Individual | Column: avg_weekly_trxn_amt
Total simulated SARs: 640 out of 4668 alerted customers (13.7% SAR filing rate).

At the lowest threshold (485.0): 640 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 93758.0 (127 missed).
To catch at least 90% of SARs, threshold must stay at or below 485.0 (577 of 640 caught).
To catch at least 80% of SARs, threshold must stay at or below 93758.0 (513 of 640 caught).
To catch at least 50% of SARs, threshold must stay at or below 93758.0 (321 of 640 caught).
At the highest threshold (9421058.0): 0 SARs caught, 640 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

BIZ_AMT = """\
=== PRE-COMPUTED SAR BACKTEST (copy this verbatim, do not alter numbers) ===
Segment: Business | Column: avg_weekly_trxn_amt
Total simulated SARs: 44 out of 367 alerted customers (12.0% SAR filing rate).

At the lowest threshold (266.38): 44 SARs caught (100.0%), 0 missed.
SARs first begin to be missed at threshold 452988.38 (23 missed).
To catch at least 90% of SARs, threshold must stay at or below 266.38 (40 of 44 caught).
To catch at least 80% of SARs, threshold must stay at or below 266.38 (36 of 44 caught).
To catch at least 50% of SARs, threshold must stay at or below 266.38 (23 of 44 caught).
At the highest threshold (45725188.38): 0 SARs caught, 44 missed (100.0% missed).
=== END PRE-COMPUTED SAR BACKTEST ==="""

examples = []

# ── ex33: Business monthly — "how many SARs would we catch" ──────────────────
examples.append(ex([
    {"role": "user", "content": "How many SARs would we catch for Business customers using monthly transaction amount as the threshold?"},
    tool_msg("call_v3_ex07", "sar_backtest", {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),
    tool_result("call_v3_ex07", BIZ_MONTHLY),
    {"role": "assistant", "content": (
        BIZ_MONTHLY + "\n\n"
        "For Business customers, all 44 simulated SARs are caught at the minimum threshold of $1,397/month "
        "and are not meaningfully affected until $1,151,144/month where 27 are suddenly missed \u2014 "
        "this cliff-edge pattern indicates the SAR-flagged Business accounts are tightly clustered at "
        "lower transaction volumes, and any threshold increase above $1.15M requires documented "
        "compensating controls."
    )}
]))

# ── ex34: Individual monthly — "SAR backtest" phrasing ───────────────────────
examples.append(ex([
    {"role": "user", "content": "Run a SAR backtest for Individual customers on the monthly amount threshold"},
    tool_msg("call_v3_ex08", "sar_backtest", {"segment": "Individual", "threshold_column": "TRXN_AMT_MONTHLY"}),
    tool_result("call_v3_ex08", IND_MONTHLY),
    {"role": "assistant", "content": (
        IND_MONTHLY + "\n\n"
        "For Individual customers, keeping the monthly amount threshold at or below $222,054 preserves "
        "detection of at least 90% of the 640 simulated SARs (577 caught) \u2014 institutions "
        "considering a threshold increase above this level should document the compensating controls "
        "that would offset the additional SAR miss risk."
    )}
]))

# ── ex35: Individual avg amount — "SAR detection rate" phrasing ───────────────
examples.append(ex([
    {"role": "user", "content": "What is the SAR detection rate for Individual customers by average transaction amount?"},
    tool_msg("call_v3_ex09", "sar_backtest", {"segment": "Individual", "threshold_column": "AVG_TRXN_AMT"}),
    tool_result("call_v3_ex09", IND_AMT),
    {"role": "assistant", "content": (
        IND_AMT + "\n\n"
        "For Individual customers, maintaining a 90% SAR catch rate requires keeping the average "
        "weekly transaction amount threshold at or below $485.00 \u2014 the sudden loss of 127 SARs "
        "at $93,758 reflects a cluster of SAR-flagged accounts with aggregate weekly amounts in that "
        "range, warranting closer attention if a threshold increase to that level is under consideration."
    )}
]))

# ── ex36: Business avg amount — "if I raise the threshold" phrasing ──────────
examples.append(ex([
    {"role": "user", "content": "If I raise the Business average transaction amount threshold, how does SAR catch rate change?"},
    tool_msg("call_v3_ex10", "sar_backtest", {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),
    tool_result("call_v3_ex10", BIZ_AMT),
    {"role": "assistant", "content": (
        BIZ_AMT + "\n\n"
        "For Business customers, 23 of the 44 SARs are first missed at $452,988/week \u2014 a single "
        "threshold step loses more than 50% of detections, indicating those SAR accounts are clustered "
        "near that value; given the small Business SAR population, each missed case represents a "
        "disproportionate compliance gap and the program should maintain thresholds well below $452,988."
    )}
]))

with open(OUT, "a", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Appended {len(examples)} SAR backtest examples to {OUT}")

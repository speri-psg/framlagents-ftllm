"""
ARIA Model Benchmark — evaluates fine-tuned model alignment with AML domain.

Scoring dimensions:
  1. Tool routing accuracy   — correct tool called? (automated)
  2. Parameter accuracy      — correct key args?   (automated)
  3. Number fidelity         — no hallucinated numbers in closing insight? (automated)
  4. Policy citation quality — human eval, scored 1-3 (prompted, not automated)

Usage:
  python benchmark_aria.py --base-url http://localhost:11434/v1 --model aria-v1
  python benchmark_aria.py --base-url https://<tunnel>.trycloudflare.com/v1 --model aria-v1
"""

import argparse
import io
import json
import re
import sys
import urllib.request

# Force UTF-8 output on Windows so ✓/✗ don't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL    = "aria-v1"
TIMEOUT          = 180

SYSTEM_RULE = (
    "You are ARIA — Agentic Risk Intelligence for AML — rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English. "
    "For SAR backtest questions about a specific rule: call rule_sar_backtest directly. "
    "For 2D sweep questions: call rule_2d_sweep directly. "
    "Do NOT call list_rules when the user asks about a specific rule."
)

SYSTEM_THRESHOLD = (
    "You are ARIA — Agentic Risk Intelligence for AML — threshold tuning specialist. "
    "You analyze AML alert thresholds to optimize false positive / false negative trade-offs. "
    "IMPORTANT: You MUST respond entirely in English. "
    "ALWAYS call threshold_tuning for threshold analysis questions. "
    "Valid threshold_column values: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY."
)

SYSTEM_SEG = (
    "You are ARIA — Agentic Risk Intelligence for AML — dynamic segmentation specialist. "
    "You identify natural customer behavioral segments using unsupervised K-Means clustering. "
    "IMPORTANT: You MUST respond entirely in English. "
    "ALWAYS call a tool for segmentation questions."
)

SYSTEM_POLICY = (
    "You are ARIA — Agentic Risk Intelligence for AML — compliance and policy specialist. "
    "You answer AML regulatory and policy questions using a knowledge base of BSA/AML documents. "
    "IMPORTANT: You MUST respond entirely in English. "
    "ALWAYS call search_policy_kb for regulatory / compliance questions."
)

SYSTEM_GENERAL = (
    "You are ARIA — Agentic Risk Intelligence for AML. "
    "You help AML analysts with threshold tuning, customer segmentation, and compliance Q&A. "
    "IMPORTANT: You MUST respond entirely in English."
)

# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------
# Each case: prompt, system, expected_tool, expected_args (subset check), category
# expected_tool=None means expect plain text (no tool call)
# expected_args: dict of key → expected value (substring match for strings)
# policy_score: if True, human will score 1-3 after seeing the response

@dataclass
class Case:
    id: str
    prompt: str
    system: str
    expected_tool: Optional[str]
    expected_args: dict = field(default_factory=dict)
    human_eval: bool = False
    note: str = ""

CASES = [
    # ── Category 1: Tool routing ─────────────────────────────────────────────

    # Threshold tuning
    Case("T01", "Show FP/FN trade-off for Business customers by monthly transaction amount",
         SYSTEM_THRESHOLD, "threshold_tuning",
         {"segment": "Business", "threshold_column": "TRXN_AMT_MONTHLY"}),

    Case("T02", "Show FP/FN trade-off for Individual customers by weekly transaction count",
         SYSTEM_THRESHOLD, "threshold_tuning",
         {"segment": "Individual", "threshold_column": "AVG_TRXNS_WEEK"}),

    Case("T03", "Run threshold tuning for Business customers by average transaction amount",
         SYSTEM_THRESHOLD, "threshold_tuning",
         {"segment": "Business", "threshold_column": "AVG_TRXN_AMT"}),

    # Rule SAR backtest
    Case("R01", "What is the SAR catch rate for Detect Excessive Transaction Activity?",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Detect Excessive"},
         note="risk_factor substring match"),

    Case("R02", "Run SAR backtest for CTR Client rule",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "CTR Client"}),

    Case("R03", "What is the SAR filing rate for Elder Abuse rule?",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Elder Abuse"}),

    Case("R04", "Run SAR backtest for Activity Deviation ACH rule",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Activity Deviation"}),

    Case("R05", "Show SAR backtest for Velocity Single",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Velocity Single"}),

    # Rule 2D sweep
    Case("D01", "Run a 2D sweep for Structuring rule",
         SYSTEM_RULE, "rule_2d_sweep",
         {"risk_factor": "Structuring"}),

    Case("D02", "Show 2D heatmap for Velocity Single — how do floor_amount and z_threshold interact?",
         SYSTEM_RULE, "rule_2d_sweep",
         {"risk_factor": "Velocity Single"}),

    Case("D03", "Show 2D analysis for Detect Excessive Transaction Activity",
         SYSTEM_RULE, "rule_2d_sweep",
         {"risk_factor": "Detect Excessive"}),

    # List rules
    Case("L01", "Show me all AML rules",
         SYSTEM_RULE, "list_rules", {}),

    Case("L02", "Which rules generate the most false positives?",
         SYSTEM_RULE, "list_rules", {}),

    # Segmentation
    Case("S01", "Cluster Business customers by transaction behavior",
         SYSTEM_SEG, None,   # accepts cluster_analysis or ss_cluster_analysis
         note="tool must contain 'cluster'"),

    Case("S02", "Which Business cluster has the highest transaction volume?",
         SYSTEM_SEG, None,
         note="tool must contain 'cluster'"),

    # Policy
    Case("P01", "What is the Bank Secrecy Act?",
         SYSTEM_POLICY, "search_policy_kb", {},
         human_eval=True),

    Case("P02", "What is AML structuring?",
         SYSTEM_POLICY, "search_policy_kb", {},
         human_eval=True),

    Case("P03", "What is the Wolfsberg risk-based approach to AML?",
         SYSTEM_POLICY, "search_policy_kb", {},
         human_eval=True),

    Case("P04", "How do banks manage alert volumes?",
         SYSTEM_POLICY, "search_policy_kb", {},
         human_eval=True),

    # Out-of-scope / unavailable metric
    Case("O01", "What is the average daily balance for Business customers?",
         SYSTEM_GENERAL, None,
         note="should be plain text decline"),

    Case("O02", "Show me net income distribution for Business customers",
         SYSTEM_GENERAL, None,
         note="should be plain text decline"),

    Case("O03", "Can you show me credit scores for high-risk customers?",
         SYSTEM_GENERAL, None,
         note="should be plain text decline"),

    Case("O04", "Show FP/FN trade-off for Business customers by daily balance",
         SYSTEM_THRESHOLD, None,
         note="invalid column — should be plain text decline with alternatives"),

    # Sanctions / OFAC routing
    Case("X01", "What happens when a customer hits the OFAC SDN list?",
         SYSTEM_POLICY, "search_policy_kb", {},
         human_eval=True),

    Case("X02", "What is the process for handling a sanctions screening hit?",
         SYSTEM_POLICY, "search_policy_kb", {},
         human_eval=True),

    Case("X03", "How many customers in our portfolio have OFAC hits?",
         SYSTEM_POLICY, None,
         note="data not in system — should be plain text decline with redirect"),

    # New rule typologies — SAR backtest routing
    Case("N01", "Run SAR backtest for Funnel Account rule",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Funnel"}),

    Case("N02", "Show SAR analysis for Human Trafficking Indicators",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Human Trafficking"}),

    Case("N03", "What is the SAR catch rate for Round-trip rule?",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Round"}),

    Case("N04", "Run SAR backtest for Velocity Multiple",
         SYSTEM_RULE, "rule_sar_backtest",
         {"risk_factor": "Velocity Multiple"}),

    Case("N05", "Show 2D sweep for Funnel Account rule",
         SYSTEM_RULE, "rule_2d_sweep",
         {"risk_factor": "Funnel"}),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat(base_url: str, model: str, system: str, user: str) -> str:
    url = base_url.rstrip("/").replace("/v1", "") + "/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        msg = json.loads(resp.read())["message"]
        # Gemma 4 separates reasoning from output — tool calls may appear in thinking
        thinking = msg.get("thinking") or ""
        content  = msg.get("content")  or ""
        return (thinking + "\n" + content).strip() if thinking else content


def extract_tool_call(content: str) -> Optional[tuple]:
    """Return (tool_name, args_dict) if content contains a JSON tool call, else None."""
    if not content:
        return None

    # Format 1: Gemma 4 native — call:tool_name\n{...} (with newline)
    m = re.search(
        r'call:(\w+)\s*\n\s*(\{(?:[^{}]|\{[^{}]*\})*\})',
        content, re.DOTALL
    )
    if m:
        try:
            args = json.loads(m.group(2))
            return m.group(1), args
        except Exception:
            pass

    # Format 6: Gemma 4 template native — <|tool_call>call:NAME{...}<tool_call|>
    # Template renders args as JSON string directly adjacent (no newline)
    m = re.search(r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>', content, re.DOTALL)
    if m:
        try:
            args = json.loads("{" + m.group(2) + "}")
            return m.group(1), args
        except Exception:
            # args may be key:value pairs not JSON — try bare call:name{} fallback
            pass

    # Format 6b: call:NAME{...} no delimiters, no newline
    m = re.search(r'call:(\w+)\{((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)
    if m:
        try:
            args = json.loads("{" + m.group(2) + "}")
            return m.group(1), args
        except Exception:
            pass

    # Format 2: Qwen style — <tool_call>{"name": ..., "arguments": ...}</tool_call>
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if "name" in obj and "arguments" in obj:
                return obj["name"], obj["arguments"] if isinstance(obj["arguments"], dict) else {}
        except Exception:
            pass

    # Format 3: OpenAI-style JSON — {"name": "...", "arguments": {...}}
    m = re.search(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}',
        content, re.DOTALL
    )
    if m:
        try:
            args = json.loads(m.group(2))
            return m.group(1), args
        except Exception:
            pass

    # Format 4: bare JSON object with "name" key
    m = re.search(r'\{[^{}]{10,}\}', content, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "name" in obj:
                return obj["name"], obj.get("arguments", {})
        except Exception:
            pass

    # Format 7: Gemma 4 native tool_code — <eos>tool_code print(func(kwargs))<unused##>
    m = re.search(
        r'<eos>tool_code\s+print\((\w+)\((.*?)\)\)',
        content, re.DOTALL
    )
    if m:
        func_name = m.group(1)
        raw_kwargs = m.group(2)
        args = {}
        # Parse Python-style kwargs: key="value" or key=number or key=True/False
        for km in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([\d.]+)|(True|False))', raw_kwargs):
            key = km.group(1)
            val = km.group(2) or km.group(3) or km.group(4)
            if km.group(3):
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
            elif km.group(4):
                val = val == 'True'
            args[key] = val
        return func_name, args

    # Format 8: backtick-wrapped function call — `func_name(kwargs)`
    m = re.search(r'`(\w+)\(([^`]{0,400})\)`', content, re.DOTALL)
    if m:
        func_name = m.group(1)
        raw_kwargs = m.group(2)
        # Only treat as tool call if func_name looks like a known tool
        _known = {"threshold_tuning", "rule_sar_backtest", "rule_2d_sweep",
                  "list_rules", "search_policy_kb", "cluster_analysis"}
        if func_name in _known:
            args = {}
            for km in re.finditer(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\d+(?:\.\d+)?))", raw_kwargs):
                key = km.group(1)
                val = km.group(2) or km.group(3) or km.group(4)
                if km.group(4):
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
                args[key] = val
            return func_name, args

    # Format 5: natural language thinking — "call/use/invoke [the] `tool_name`"
    _SKIP_WORDS = {"the", "a", "an", "this", "that", "it", "my", "our"}
    for m in re.finditer(
        r'(?:call|use|invoke|using)\s+(?:(?:the|a|an)\s+)?[`"]?(\w+)[`"]?',
        content, re.IGNORECASE
    ):
        tool_name = m.group(1)
        if tool_name.lower() in _SKIP_WORDS:
            continue
        args = {}
        for km in re.finditer(r'[`"]?(\w+)[`"]?\s*[=:]\s*[`\'"]([^`\'"]+)[`\'"]', content):
            k, v = km.group(1), km.group(2)
            if k.lower() not in {"call", "name"} | _SKIP_WORDS:
                args[k] = v
        return tool_name, args

    return None


def extract_numbers(text: str) -> set:
    """Extract all numeric values from text (int and float)."""
    return set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))


def check_routing(case: Case, tool_name: Optional[str]) -> tuple:
    """Returns (pass: bool, detail: str)"""
    if case.expected_tool is None:
        # Expect no tool call OR a cluster tool for segmentation
        if "cluster" in (case.note or "").lower():
            ok = tool_name is not None and "cluster" in (tool_name or "").lower()
            return ok, f"tool={tool_name}"
        # Expect plain text (no tool)
        ok = tool_name is None
        return ok, "no tool call" if ok else f"unexpected tool: {tool_name}"
    else:
        ok = tool_name == case.expected_tool
        return ok, f"tool={tool_name}" if ok else f"expected={case.expected_tool} got={tool_name}"


_ARG_ALIASES = {
    "segment":           ["segment", "customer_type", "customer_segment", "segment_type"],
    "threshold_column":  ["threshold_column", "transaction_amount", "amount_type", "column", "metric"],
    "risk_factor":       ["risk_factor", "rule", "rule_name", "risk_factor_name"],
}

def check_args(case: Case, args: dict) -> tuple:
    """Subset check — all expected_args keys must match (substring), with alias expansion."""
    if not case.expected_args or args is None:
        return True, "n/a"
    failures = []
    for k, v in case.expected_args.items():
        # Check canonical key and all aliases
        candidates = _ARG_ALIASES.get(k, [k])
        actual = ""
        for alias in candidates:
            if alias in args:
                actual = str(args[alias]).lower()
                break
        if v.lower() not in actual:
            failures.append(f"{k}: expected '{v}' in '{actual}'")
    ok = len(failures) == 0
    return ok, "ok" if ok else "; ".join(failures)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(base_url: str, model: str, verbose: bool):
    print(f"\n{'='*70}")
    print(f"  ARIA Benchmark — model={model}  url={base_url}")
    print(f"{'='*70}\n")

    results = []

    for case in CASES:
        print(f"[{case.id}] {case.prompt[:65]}...")
        try:
            content = chat(base_url, model, case.system, case.prompt)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({"id": case.id, "error": str(e)})
            continue

        tc = extract_tool_call(content)
        tool_name = tc[0] if tc else None
        tool_args = tc[1] if tc else {}

        route_ok, route_detail   = check_routing(case, tool_name)
        args_ok,   args_detail   = check_args(case, tool_args)

        result = {
            "id":           case.id,
            "route_ok":     route_ok,
            "route_detail": route_detail,
            "args_ok":      args_ok,
            "args_detail":  args_detail,
            "human_eval":   case.human_eval,
            "content":      content,
        }
        results.append(result)

        status = "✓" if route_ok else "✗"
        arg_st = "✓" if args_ok  else "✗"
        print(f"  Route {status} {route_detail}")
        print(f"  Args  {arg_st} {args_detail}")

        if verbose:
            print(f"  Response (first 1500): {content[:1500].replace(chr(10),' ')}")

        if case.human_eval:
            print(f"  ── Policy response (score 1=wrong 2=shallow 3=correct+cited) ──")
            print(f"  {content[:500]}")
            score = input("  Your score [1/2/3]: ").strip()
            result["human_score"] = int(score) if score.isdigit() else None

        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print("  SCORECARD")
    print(f"{'='*70}")

    valid      = [r for r in results if "error" not in r]
    non_policy = [r for r in valid if not r.get("human_eval")]
    policy     = [r for r in valid if r.get("human_eval")]

    route_pass = sum(1 for r in non_policy if r["route_ok"])
    route_tot  = len(non_policy)
    args_pass  = sum(1 for r in non_policy if r["route_ok"] and r["args_ok"])
    args_tot   = sum(1 for r in non_policy if r["route_ok"])  # only scored when routing correct

    print(f"\n  1. Tool routing accuracy : {route_pass}/{route_tot} = {100*route_pass//max(route_tot,1)}%")
    print(f"  2. Parameter accuracy    : {args_pass}/{args_tot}  = {100*args_pass//max(args_tot,1)}%  (of correctly routed)")

    if policy and any("human_score" in r for r in policy):
        scores = [r["human_score"] for r in policy if r.get("human_score") is not None]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  4. Policy quality (human): {avg:.1f}/3.0  ({len(scores)} questions scored)")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n  Errors: {len(errors)} — {[r['id'] for r in errors]}")

    # Failures detail
    failures = [r for r in non_policy if not r["route_ok"]]
    if failures:
        print(f"\n  Routing failures:")
        for r in failures:
            print(f"    [{r['id']}] {r['route_detail']}")

    arg_failures = [r for r in non_policy if r["route_ok"] and not r["args_ok"]]
    if arg_failures:
        print(f"\n  Arg failures:")
        for r in arg_failures:
            print(f"    [{r['id']}] {r['args_detail']}")

    print(f"\n{'='*70}\n")

    # Save results
    out_file = "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA model benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()
    run_benchmark(args.base_url, args.model, args.verbose)

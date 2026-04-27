"""
ARIA Model Benchmark — evaluates fine-tuned model alignment with AML domain.

Scoring dimensions:
  1. Tool routing accuracy   — correct tool called? (automated)
  2. Parameter accuracy      — correct key args?   (automated)
  3. Number fidelity         — no hallucinated numbers in closing insight? (automated)
  4. Policy citation quality — human eval, scored 1-3 (prompted, not automated)

Usage:
  python benchmark_aria.py --base-url http://localhost:11434/v1 --model aria-v2
  python benchmark_aria.py --base-url https://<tunnel>.trycloudflare.com/v1 --model aria-v2
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
DEFAULT_MODEL    = "aria-v2"
TIMEOUT          = 180

SYSTEM_RULE = (
    "You are ARIA — Agentic Risk Intelligence for AML — rule performance specialist. "
    "You analyze AML monitoring rules using SAR backtesting and 2D parameter sweeps. "
    "IMPORTANT: You MUST respond entirely in English. "
    "For SAR backtest questions about a specific rule: call rule_sar_backtest directly. "
    "For 2D sweep questions: call rule_2d_sweep directly. "
    "Do NOT call list_rules when the user asks about a specific rule. "
    "After receiving tool results, the tool result contains a PRE-COMPUTED section. "
    "You MUST copy that section word-for-word into your response. "
    "Do NOT change any numbers, thresholds, or directional statements. "
    "After copying it, add ONE sentence of AML domain insight."
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
    "ALWAYS call search_policy_kb for regulatory / compliance questions. "
    "For questions about data availability, portfolio metrics, or how many customers have X: "
    "respond directly without calling any tool — state that this operational data is not available "
    "in the knowledge base and suggest the user check their internal compliance system."
)

SYSTEM_GENERAL = (
    "You are ARIA — Agentic Risk Intelligence for AML. "
    "You help AML analysts with threshold tuning, customer segmentation, and compliance Q&A. "
    "IMPORTANT: You MUST respond entirely in English."
)

# V29 canonical tool result — injected in format-check cases
PC_LIST_RULES = """\
Tool result for list_rules:
=== PRE-COMPUTED RULE LIST (copy this verbatim) ===
Available AML rules with SAR/FP performance (detailed table shown in chart below):
NOTE: This is the COMPLETE list of 16 rules in the system. Do NOT add or infer any rules not listed here.
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

# ---------------------------------------------------------------------------
# Tool schemas — passed to Ollama so the model outputs native tool_calls
# rather than reasoning text. Mirrors the schemas in agents/threshold_agent.py.
# ---------------------------------------------------------------------------

_TOOL_threshold_tuning = {
    "type": "function",
    "function": {
        "name": "threshold_tuning",
        "description": (
            "Analyze false positive / false negative trade-offs as a threshold column is swept "
            "for a given customer segment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "enum": ["Business", "Individual"],
                    "description": "Customer segment to analyze.",
                },
                "threshold_column": {
                    "type": "string",
                    "enum": ["AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"],
                    "description": (
                        "Column to sweep as the alert threshold. "
                        "AVG_TRXNS_WEEK = average NUMBER of transactions per week (count/frequency). "
                        "AVG_TRXN_AMT = average DOLLAR AMOUNT per transaction. "
                        "TRXN_AMT_MONTHLY = average total monthly transaction DOLLAR VOLUME. "
                        "Use AVG_TRXN_AMT when the user says 'transaction amount', 'average amount', or 'dollar amount'. "
                        "Use AVG_TRXNS_WEEK when the user says 'transaction count', 'number of transactions', or 'frequency'. "
                        "Use TRXN_AMT_MONTHLY when the user says 'monthly amount' or 'monthly volume'."
                    ),
                },
            },
            "required": ["segment", "threshold_column"],
        },
    },
}

_TOOL_rule_sar_backtest = {
    "type": "function",
    "function": {
        "name": "rule_sar_backtest",
        "description": (
            "For a specific named AML rule, sweep a rule condition parameter and show "
            "how many SAR customers are caught vs. missed. Use when the user names a specific "
            "rule and asks about SAR filing rate, SAR catch rate, SAR detection, or rule backtest."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "risk_factor": {
                    "type": "string",
                    "description": (
                        "Rule name to analyze (e.g. 'Activity Deviation (ACH)', 'Elder Abuse', "
                        "'Velocity Single', 'CTR Client', 'Detect Excessive Transaction Activity', "
                        "'Velocity Multiple', 'Human Trafficking Indicators', 'Round-trip')."
                    ),
                },
            },
            "required": ["risk_factor"],
        },
    },
}

_TOOL_rule_2d_sweep = {
    "type": "function",
    "function": {
        "name": "rule_2d_sweep",
        "description": (
            "2D grid sweep: vary two condition parameters simultaneously for an AML rule "
            "and produce a heatmap. Use when the user asks how two parameters interact, "
            "wants a grid or heatmap, or wants to optimize two thresholds at once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "risk_factor": {
                    "type": "string",
                    "description": (
                        "Rule name (e.g. 'Activity Deviation (ACH)', 'Elder Abuse', "
                        "'Velocity Single', 'Detect Excessive Transaction Activity', "
                        "'Structuring (Incoming Cash)', 'Funnel Account')."
                    ),
                },
            },
            "required": ["risk_factor"],
        },
    },
}

_TOOL_list_rules = {
    "type": "function",
    "function": {
        "name": "list_rules",
        "description": (
            "List all available AML detection rules with their SAR count, false positive count, "
            "and precision. Use when no specific rule is named or the user wants an overview."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_search_policy_kb = {
    "type": "function",
    "function": {
        "name": "search_policy_kb",
        "description": "Search the AML policy knowledge base to answer regulatory and compliance questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    },
}

_TOOL_ds_cluster_analysis = {
    "type": "function",
    "function": {
        "name": "ds_cluster_analysis",
        "description": (
            "Cluster customers by behavioral features using K-Means. "
            "Use for segmentation questions about customer clusters or behavioral groups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "enum": ["Business", "Individual"],
                    "description": "Customer segment to cluster.",
                },
            },
            "required": ["segment"],
        },
    },
}

# Map system prompt → tool list sent to Ollama
TOOLS_BY_SYSTEM = {
    SYSTEM_RULE:      [_TOOL_rule_sar_backtest, _TOOL_rule_2d_sweep, _TOOL_list_rules],
    SYSTEM_THRESHOLD: [_TOOL_threshold_tuning, _TOOL_list_rules],
    SYSTEM_SEG:       [_TOOL_ds_cluster_analysis],
    SYSTEM_POLICY:    [_TOOL_search_policy_kb],
    SYSTEM_GENERAL:   [_TOOL_threshold_tuning, _TOOL_rule_sar_backtest,
                       _TOOL_rule_2d_sweep, _TOOL_list_rules, _TOOL_search_policy_kb,
                       _TOOL_ds_cluster_analysis],
}

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
    format_check: bool = False  # V29: simulate tool result and verify response format
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
         SYSTEM_RULE, "list_rules", {}, format_check=True),

    Case("L02", "Which rules generate the most false positives?",
         SYSTEM_RULE, "list_rules", {}, format_check=True),

    # Segmentation
    Case("S01", "Cluster Business customers by transaction behavior",
         SYSTEM_SEG, None,   # accepts cluster_analysis or ds_cluster_analysis
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

def _parse_msg(msg: dict, valid_names: Optional[set] = None) -> tuple:
    """Return (content_str, (tool_name, tool_args) | None) from an Ollama message dict.

    valid_names: if provided, native tool calls with names outside this set are
    discarded (the model hallucinated a non-existent tool name).
    """
    thinking = msg.get("thinking") or ""
    content  = msg.get("content")  or ""
    text = (thinking + "\n" + content).strip() if thinking else content

    # Native Ollama tool_calls take priority — reliable structured output
    native = msg.get("tool_calls")
    if native:
        fn = native[0].get("function", {})
        name = fn.get("name", "")
        if name and (valid_names is None or name in valid_names):
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return text, (name, args)

    return text, None


def chat(base_url: str, model: str, system: str, user: str,
         tools: Optional[list] = None) -> tuple:
    """Single-turn chat. Returns (content_str, (tool_name, args) | None).

    Passes tool schemas to Ollama when provided so the model emits native
    tool_calls instead of reasoning text.
    """
    url = base_url.rstrip("/").replace("/v1", "") + "/api/chat"
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    valid_names = {t["function"]["name"] for t in tools} if tools else None
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        msg = json.loads(resp.read())["message"]
        return _parse_msg(msg, valid_names)


def chat_with_result(base_url: str, model: str, system: str, user: str,
                     tool_name: str, tool_result: str,
                     tools: Optional[list] = None) -> str:
    """Two-turn chat: inject a pre-computed tool result and return the model's final response."""
    url = base_url.rstrip("/").replace("/v1", "") + "/api/chat"
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": tool_name, "arguments": {}}}]},
            {"role": "tool",      "content": tool_result},
        ],
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        msg = json.loads(resp.read())["message"]
        text, _ = _parse_msg(msg)
        return text


def check_format(content: str) -> tuple:
    """V29 regression: verify list_rules response has no double-table or stale data.

    Checks:
      - PRE-COMPUTED RULE LIST block appears exactly once
      - Old PRE-COMPUTED LIST_RULES RESULT format is absent
      - Stale '11 AML rules' count is absent
    """
    issues = []
    count = content.count("=== PRE-COMPUTED RULE LIST")
    if count == 0:
        issues.append("missing PRE-COMPUTED RULE LIST block")
    elif count > 1:
        issues.append(f"double-table: block appears {count}x")
    if "PRE-COMPUTED LIST_RULES RESULT" in content:
        issues.append("old LIST_RULES RESULT format present")
    if "11 AML rules" in content or "system contains 11" in content.lower():
        issues.append("stale 11-rule count")
    ok = len(issues) == 0
    return ok, "ok" if ok else "; ".join(issues)


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

    # Format 7: Gemma 4 native tool_code — [<eos>]tool_code print(func(kwargs))
    m = re.search(
        r'(?:<eos>)?tool_code\s+print\((\w+)\((.*?)\)\)',
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
        case_tools = TOOLS_BY_SYSTEM.get(case.system)
        try:
            content, native_tc = chat(base_url, model, case.system, case.prompt,
                                      tools=case_tools)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({"id": case.id, "error": str(e)})
            continue

        # Native tool_calls from Ollama take priority; fall back to regex parsing.
        # For regex fallback, discard any name not in valid_names — Format 5
        # (natural language scan) regularly false-positives on words like
        # "following", "knowledge", "available" from decline responses.
        valid_names = {t["function"]["name"] for t in case_tools} if case_tools else None
        if native_tc:
            tool_name, tool_args = native_tc
        else:
            parsed = extract_tool_call(content)
            tool_name = parsed[0] if parsed else None
            tool_args = parsed[1] if parsed else {}
            if valid_names and tool_name and tool_name not in valid_names:
                tool_name, tool_args = None, {}

        route_ok, route_detail   = check_routing(case, tool_name)
        args_ok,   args_detail   = check_args(case, tool_args)

        # V29 format check: simulate the tool result turn and verify response format
        format_ok, format_detail = True, "n/a"
        format_content = ""
        if case.format_check and route_ok:
            try:
                format_content = chat_with_result(
                    base_url, model, case.system, case.prompt,
                    tool_name, PC_LIST_RULES,
                    # No tools on the second turn — model must produce final text,
                    # not fire another tool call.
                )
                format_ok, format_detail = check_format(format_content)
            except Exception as e:
                format_ok, format_detail = False, f"error: {e}"

        result = {
            "id":            case.id,
            "route_ok":      route_ok,
            "route_detail":  route_detail,
            "args_ok":       args_ok,
            "args_detail":   args_detail,
            "format_ok":     format_ok,
            "format_detail": format_detail,
            "human_eval":    case.human_eval,
            "content":       content,
        }
        results.append(result)

        status = "✓" if route_ok else "✗"
        arg_st = "✓" if args_ok  else "✗"
        fmt_st = ("✓" if format_ok else "✗") if format_detail != "n/a" else "-"
        print(f"  Route {status} {route_detail}")
        print(f"  Args  {arg_st} {args_detail}")
        print(f"  Fmt   {fmt_st} {format_detail}")
        if verbose and format_content:
            print(f"  Format response (first 500): {format_content[:500].replace(chr(10), ' ')}")

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

    fmt_cases = [r for r in non_policy if r.get("format_detail") != "n/a"]
    fmt_pass  = sum(1 for r in fmt_cases if r.get("format_ok"))
    fmt_tot   = len(fmt_cases)

    print(f"\n  1. Tool routing accuracy : {route_pass}/{route_tot} = {100*route_pass//max(route_tot,1)}%")
    print(f"  2. Parameter accuracy    : {args_pass}/{args_tot}  = {100*args_pass//max(args_tot,1)}%  (of correctly routed)")
    if fmt_tot:
        print(f"  3. Format accuracy (V29) : {fmt_pass}/{fmt_tot}  = {100*fmt_pass//max(fmt_tot,1)}%  (list_rules double-table / stale-count check)")

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

    fmt_failures = [r for r in fmt_cases if not r.get("format_ok")]
    if fmt_failures:
        print(f"\n  Format failures (V29):")
        for r in fmt_failures:
            print(f"    [{r['id']}] {r['format_detail']}")

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

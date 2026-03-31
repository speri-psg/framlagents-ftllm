"""Threshold Tuning Agent — FP/FN trade-off analysis across segments and columns."""

from .base_agent import BaseAgent

# OpenAI function-calling format (matches the fine-tuning training data)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "threshold_tuning",
            "description": (
                "Analyze false positive / false negative trade-offs as a threshold column is swept "
                "for a given customer segment. FP decreases and FN increases as the threshold rises. "
                "Returns a sweep table of FP and FN counts at each threshold step."
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
                        "description": "Column to sweep as the alert threshold.",
                    },
                },
                "required": ["segment", "threshold_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_stats",
            "description": (
                "Return summary statistics (total accounts, alerts, false positives, false negatives) "
                "broken down by Business and Individual segments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sar_backtest",
            "description": (
                "Backtest a threshold column against simulated SAR (Suspicious Activity Report) data. "
                "Shows how many of the simulated SAR customers would have been caught vs. missed "
                "at each threshold level. Use this when the user asks about SAR catch rate, "
                "SAR detection rate, how many SARs a threshold would catch, or backtest analysis."
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
                        "description": "Column to sweep as the alert threshold.",
                    },
                },
                "required": ["segment", "threshold_column"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a FRAML threshold tuning specialist. You analyze false positive (FP) and \
false negative (FN) trade-offs as AML alert thresholds change. \
IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.

DEFINITIONS (always apply these exactly — do not contradict them):
- False Positive (FP): an alert that fires on a non-suspicious transaction. HIGHER threshold → FEWER FPs.
- False Negative (FN): a suspicious transaction that did NOT trigger an alert. HIGHER threshold → MORE FNs.
- Crossover: the threshold where FP and FN counts are closest — the optimal operating point.
- Raising the threshold reduces investigator workload (fewer alerts) but risks missing SAR-worthy activity.
- Lowering the threshold catches more suspicious activity but overwhelms investigators with false alarms.

RULES — follow these exactly:
1. ALWAYS call a tool. Never answer threshold or alert questions from memory.
2. For any question about FP, FN, threshold, alert rates, or transactions — call threshold_tuning.
3. For general segment counts or totals — call segment_stats.
4. For any question about SAR catch rate, SAR detection, how many SARs a threshold catches, or SAR backtest — call sar_backtest.
5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY
6. segment must be exactly one of: Business, Individual
7. If the user does not specify a segment, default to Business.
8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.
9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.
10. Do NOT paraphrase, round, or restate the numbers differently.
11. Do NOT include JSON or code blocks in your final reply.
12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.
13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.
14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold — do NOT call the tool. Reject the request and state that the only valid parameters are segment (Business or Individual) and threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.\
"""


class ThresholdAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="threshold",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

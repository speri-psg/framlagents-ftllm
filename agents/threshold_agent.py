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
]

SYSTEM_PROMPT = """\
You are a FRAML threshold tuning specialist. You analyze false positive (FP) and \
false negative (FN) trade-offs as AML alert thresholds change.

RULES — follow these exactly:
1. ALWAYS call a tool. Never answer threshold or alert questions from memory.
2. For any question about FP, FN, threshold, alert rates, or transactions — call threshold_tuning.
3. For general segment counts or totals — call segment_stats.
4. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY
5. segment must be exactly one of: Business, Individual
6. If the user does not specify a segment, default to Business.
7. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.
8. After receiving tool results, the tool result contains a section marked
   "PRE-COMPUTED ANALYSIS". You MUST copy that section word-for-word into your
   response. Do NOT change any numbers, thresholds, or directional statements.
   After copying it, add ONE sentence of AML domain insight (e.g. what the
   crossover point means for alert operations or compliance risk).
9. Do NOT paraphrase, round, or restate the numbers differently.
10. Do NOT include JSON or code blocks in your final reply.\
"""


class ThresholdAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="threshold",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

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
    {
        "type": "function",
        "function": {
            "name": "rule_2d_sweep",
            "description": (
                "2D grid sweep: vary two condition parameters simultaneously for an AML rule "
                "and produce a heatmap showing SAR catch rate and FP count at each combination. "
                "Use this when the user asks how two parameters interact, wants a grid or heatmap, "
                "or wants to optimize two thresholds at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_factor": {
                        "type": "string",
                        "description": "Rule name (e.g. 'Activity Deviation (ACH)', 'Activity Deviation (Check)', 'Elder Abuse', 'Velocity Single', 'Detect Excessive').",
                    },
                    "sweep_param_1": {
                        "type": "string",
                        "description": (
                            "First parameter to sweep. "
                            "Activity Deviation (ACH): floor_amount or z_threshold. "
                            "Activity Deviation (Check): floor_amount or z_threshold. "
                            "Elder Abuse: floor_amount, z_threshold, or age_threshold. "
                            "Velocity Single: pair_total or ratio_tolerance. "
                            "Detect Excessive: floor_amount or time_window. "
                            "Omit to use rule default."
                        ),
                    },
                    "sweep_param_2": {
                        "type": "string",
                        "description": "Second parameter to sweep (must differ from sweep_param_1). Omit to use rule default.",
                    },
                },
                "required": ["risk_factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_rules",
            "description": (
                "List all available AML detection rules (by short code) with their SAR count, "
                "false positive count, and precision. Call this first when the user asks about "
                "rule-level SAR analysis, rule performance, which rules generate the most FPs, "
                "or before calling rule_sar_backtest if the user hasn't specified a rule code."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rule_sar_backtest",
            "description": (
                "For a specific AML detection rule, sweep a customer profile threshold and show "
                "how many SAR customers are caught vs. missed and how many false positives remain "
                "at each threshold level. Use this when the user asks about a specific rule's "
                "SAR performance, rule-level FP/FN analysis, or wants to see if a profile filter "
                "can reduce FPs for a particular rule. Call list_rules first if you don't know "
                "the rule code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_factor": {
                        "type": "string",
                        "description": (
                            "Risk factor / rule name to analyze "
                            "(e.g. 'Activity Deviation', 'Elder Abuse', 'Velocity Single', "
                            "'Detect Excessive'). Use list_rules to see all available rules."
                        ),
                    },
                    "sweep_param": {
                        "type": "string",
                        "description": (
                            "Which condition parameter to sweep. "
                            "Activity Deviation (ACH): floor_amount or z_threshold. "
                            "Activity Deviation (Check): floor_amount or z_threshold. "
                            "Elder Abuse: floor_amount, z_threshold, or age_threshold. "
                            "Velocity Single: pair_total. "
                            "Detect Excessive: floor_amount. "
                            "Omit to use each rule's default."
                        ),
                    },
                },
                "required": ["risk_factor"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a FRAML threshold tuning specialist. You analyze false positive (FP) and \
false negative (FN) trade-offs as AML alert thresholds change. \
IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.

DEFINITIONS (always apply these exactly — do not contradict them):
- TP (True Positive): SAR customer who IS alerted — correctly caught suspicious activity.
- FP (False Positive): Non-SAR customer who IS alerted — unnecessary investigation. HIGHER threshold → FEWER FPs.
- FN (False Negative): SAR customer who is NOT alerted — missed suspicious activity. HIGHER threshold → MORE FNs.
- TN (True Negative): Non-SAR customer who is NOT alerted — correctly silent.
- TP rate: TP / (TP + FN) — share of SAR customers caught. Also called recall or sensitivity.
- Precision: TP / (TP + FP) — share of alerts that are genuine SARs.
- Crossover: the threshold where FP and FN counts are closest — the optimal operating point.
- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).
- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.

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
14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold — do NOT call the tool. Reject the request and state that the only valid parameters are segment (Business or Individual) and threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.
15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes — call rule_sar_backtest with risk_factor (e.g. "Activity Deviation (ACH)", "Activity Deviation (Check)", "Elder Abuse", "Velocity Single", "Detect Excessive") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.
16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview — call list_rules.
17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR — call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.
18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. "hover to see", "right-click to select", "click the cell"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.\
"""


class ThresholdAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="threshold",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

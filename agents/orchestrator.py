"""
Orchestrator Agent — routes user queries to specialist agents and runs them in parallel.

Routing is done locally via keyword matching (no API call needed):
  threshold    → ThresholdAgent    (FP/FN tuning, alert stats)
  segmentation → SegmentationAgent (clustering, smart segmentation, alerts distribution)
  policy       → PolicyAgent       (AML policy, regulatory questions)
  default      → ThresholdAgent + SegmentationAgent (both in parallel)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from .threshold_agent import ThresholdAgent
from .segmentation_agent import SegmentationAgent
from .policy_agent import PolicyAgent

THRESHOLD_KEYWORDS = {
    'threshold', 'tuning', 'false positive', 'false negative', 'fp', 'fn',
    'alert stats', 'avg_trxns', 'avg_trxn', 'trxn_amt', 'trxns_week',
    'trxn_amt_monthly', 'monthly', 'weekly', 'transactions',
}
SEGMENTATION_KEYWORDS = {
    'cluster', 'clustering', 'segment', 'segmentation', 'smart segment',
    'k-means', 'kmeans', 'pca', 'distribution', 'alerts distribution',
    'behavioral', 'group', 'groups',
}
POLICY_KEYWORDS = {
    'policy', 'policies', 'compliance', 'regulation', 'regulatory',
    'guideline', 'guidelines', 'best practice', 'aml policy', 'rule',
    'requirement', 'standard', 'fatf', 'bsa', 'fincen',
}


class OrchestratorAgent:

    def __init__(self):
        self.threshold_agent = ThresholdAgent()
        self.segmentation_agent = SegmentationAgent()
        self.policy_agent = PolicyAgent()
        self._agent_map = {
            "threshold": self.threshold_agent,
            "segmentation": self.segmentation_agent,
            "policy": self.policy_agent,
        }

    def _route(self, query: str) -> list:
        """Local keyword-based routing — no API call required."""
        q = query.lower()
        agents = []
        if any(kw in q for kw in THRESHOLD_KEYWORDS):
            agents.append("threshold")
        if any(kw in q for kw in SEGMENTATION_KEYWORDS):
            agents.append("segmentation")
        if any(kw in q for kw in POLICY_KEYWORDS):
            agents.append("policy")
        if not agents:
            agents = ["threshold", "segmentation"]
        print(f"[orchestrator] routing to: {agents}")
        return agents

    def run(self, query: str, tool_executor) -> tuple:
        """
        Route query, run required agents (in parallel if >1), merge results.
        Returns: (combined_text, all_chart_results)
        """
        agents_needed = self._route(query)
        to_run = [
            (name, self._agent_map[name])
            for name in agents_needed
            if name in self._agent_map
        ]
        if not to_run:
            to_run = [("threshold", self.threshold_agent)]

        if len(to_run) == 1:
            name, agent = to_run[0]
            return agent.run(query, tool_executor)

        results = {}
        with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
            futures = {
                executor.submit(agent.run, query, tool_executor): name
                for name, agent in to_run
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    print(f"[orchestrator] agent '{name}' error: {e}")
                    results[name] = (f"[{name} agent error: {e}]", [])

        all_charts = []
        text_parts = []
        for name in agents_needed:
            if name in results:
                text, charts = results[name]
                text_parts.append(f"**{name.capitalize()} Analysis:**\n{text}")
                all_charts.extend(charts)

        return "\n\n".join(text_parts), all_charts

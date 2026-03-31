"""
Orchestrator Agent — routes user queries to specialist agents and runs them in parallel.

Routing is done via LLM classification (single fast API call):
  threshold    → ThresholdAgent    (FP/FN tuning, alert stats)
  segmentation → SegmentationAgent (clustering, smart segmentation, alerts distribution)
  policy       → PolicyAgent       (AML policy, regulatory questions)
  greeting     → friendly greeting response (no agent run)
  out_of_scope → polite refusal (no agent run)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from .base_agent import OLLAMA_BASE_URL, OLLAMA_MODEL
from .threshold_agent import ThresholdAgent
from .segmentation_agent import SegmentationAgent
from .policy_agent import PolicyAgent

_CLASSIFY_SYSTEM = """\
You are a routing classifier for a FRAML AI Assistant. Given a user query, respond with \
one or more of these labels (comma-separated, no other text):

  threshold    — user wants to RUN threshold tuning analysis on OUR LOCAL DATA (FP/FN trade-off charts, sweep analysis)
  segmentation — user wants to RUN clustering/segmentation on OUR LOCAL DATA (K-Means, treemap, behavioral groups)
  policy       — user is asking a GENERAL KNOWLEDGE question about AML, compliance, regulations, industry practices, or best practices — does NOT require running local data analysis
  greeting     — query is a greeting or social pleasantry (hello, hi, how are you, etc.)
  out_of_scope — query is not related to any of the above FRAML topics

Key distinction:
- "Show FP/FN tuning for Business customers" → threshold  (run local analysis)
- "Show FP/FN threshold tuning for Individual customers" → threshold
- "Run SAR backtest for Individual customers" → threshold  (SAR backtest is a threshold tool)
- "What threshold catches 90% of SARs?" → threshold
- "SAR catch rate for Business monthly transaction amount" → threshold
- "Run SAR backtest" → threshold
- "How do banks manage alert volumes?" → policy  (general knowledge question)
- "What is AML?" → policy  (general knowledge question)
- "Cluster all customers" → segmentation  (run local analysis)
- "What does AML policy say about structuring?" → policy  (general knowledge + knowledge base)
- "Show alerts and false positive distribution across segments" → segmentation  (distribution chart, NOT threshold tuning)
- "Show alert distribution" → segmentation
- "How are alerts spread across segments?" → segmentation
- "Which segment has the most alerts?" → segmentation

Rules:
- Output ONLY the label(s), comma-separated. No explanation, no punctuation other than commas.
- A query can map to multiple labels (e.g. threshold,segmentation).
- When in doubt between out_of_scope and a FRAML label, prefer the FRAML label.\
"""


class OrchestratorAgent:

    _GREETING = (
        "Hello! I'm your FRAML AI Assistant. I can help you with:\n"
        "- **Threshold tuning** — FP/FN trade-off analysis\n"
        "- **Customer segmentation** — K-Means clustering\n"
        "- **AML policy Q&A** — compliance and regulatory questions\n\n"
        "Try one of the suggested prompts on the left, or ask me a question."
    )

    _OUT_OF_SCOPE = (
        "I can only help with FRAML-specific topics:\n"
        "- **Threshold tuning** — FP/FN trade-off analysis\n"
        "- **Customer segmentation** — K-Means clustering\n"
        "- **AML policy Q&A** — compliance and regulatory questions\n\n"
        "Please rephrase your question around one of these areas."
    )

    def __init__(self):
        self.threshold_agent    = ThresholdAgent()
        self.segmentation_agent = SegmentationAgent()
        self.policy_agent       = PolicyAgent()
        self._agent_map = {
            "threshold":    self.threshold_agent,
            "segmentation": self.segmentation_agent,
            "policy":       self.policy_agent,
        }
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def _route(self, query: str) -> list:
        """LLM-based routing — classify query into agent labels."""
        try:
            resp = self._client.chat.completions.create(
                model=OLLAMA_MODEL,
                max_tokens=20,
                temperature=0,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user",   "content": query},
                ],
            )
            raw = resp.choices[0].message.content or ""
            labels = [l.strip().lower() for l in raw.split(",") if l.strip()]
        except Exception as e:
            print(f"[orchestrator] classification error: {e} — defaulting to out_of_scope")
            labels = ["out_of_scope"]

        print(f"[orchestrator] routing to: {labels}")
        return labels

    def run(self, query: str, tool_executor) -> tuple:
        """
        Route query via LLM, run required agents (in parallel if >1), merge results.
        Returns: (combined_text, all_chart_results)
        """
        labels = self._route(query)

        if "greeting" in labels:
            return self._GREETING, []

        agent_labels = [l for l in labels if l in self._agent_map]

        if not agent_labels:
            return self._OUT_OF_SCOPE, []

        to_run = [(name, self._agent_map[name]) for name in agent_labels]

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
        for name in agent_labels:
            if name in results:
                text, charts = results[name]
                text_parts.append(f"**{name.capitalize()} Analysis:**\n{text}")
                all_charts.extend(charts)

        return "\n\n".join(text_parts), all_charts

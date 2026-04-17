"""
Orchestrator Agent — routes user queries to specialist agents and runs them in parallel.

Routing is done via LLM classification (single fast API call):
  threshold    → ThresholdAgent    (FP/FN tuning, alert stats)
  segmentation → SegmentationAgent (clustering, dynamic segmentation, alerts distribution)
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
You are a routing classifier for ARIA. Given a user query, respond with \
one or more of these labels (comma-separated, no other text):

  threshold    — user wants to RUN threshold tuning analysis on OUR LOCAL DATA (FP/FN trade-off charts, sweep analysis)
  segmentation — user wants to RUN clustering/segmentation on OUR LOCAL DATA (K-Means, treemap, behavioral groups)
  ofac         — user wants to RUN OFAC sanctions screening on OUR LOCAL CUSTOMER DATA (SDN list hits, sanctioned country exposure)
  policy       — user is asking a GENERAL KNOWLEDGE question about AML, compliance, regulations, industry practices, or best practices — does NOT require running local data analysis
  greeting     — query is a greeting or social pleasantry (hello, hi, how are you, etc.)
  out_of_scope — query is not related to any of the above AML topics

Key distinction:
- "Show FP/FN tuning for Business customers" → threshold  (run local analysis)
- "Show FP/FN threshold tuning for Individual customers" → threshold
- "Run SAR backtest for Individual customers" → threshold  (SAR backtest is a threshold tool)
- "What threshold catches 90% of SARs?" → threshold
- "SAR catch rate for Business monthly transaction amount" → threshold
- "Run SAR backtest" → threshold
- "Show me a 2D grid for Activity Deviation ACH" → threshold  (2D sweep is a threshold tool)
- "How do floor amount and sigma interact for Activity Deviation?" → threshold
- "Show me the ACH deviation rule performance" → threshold
- "What is the SAR catch rate for Activity Deviation Check?" → threshold
- "Show the heatmap for Elder Abuse" → threshold
- "2D analysis for Velocity Single" → threshold
- "How does time window interact with floor amount for Detect Excessive?" → threshold
- "Show me the AML rule performance overview" → threshold  (list_rules is a threshold tool)
- "Which rules generate the most false positives?" → threshold
- "What is the SAR catch rate for the Activity Deviation rule?" → threshold
- "Show rule-level FP analysis" → threshold
- "What happens to FP if I raise the age threshold for Elder Abuse?" → threshold
- "How do banks manage alert volumes?" → policy  (general knowledge question)
- "What is AML?" → policy  (general knowledge question)
- "Cluster all customers" → segmentation  (run local analysis)
- "What does AML policy say about structuring?" → policy  (general knowledge + knowledge base)
- "Show alerts and false positive distribution across segments" → segmentation  (distribution chart, NOT threshold tuning)
- "Show alert distribution" → segmentation
- "How are alerts spread across segments?" → segmentation
- "Which segment has the most alerts?" → segmentation
- "What is the average transaction amount for Business customers?" → threshold  (segment_stats tool)
- "How many alerts does the Individual segment have?" → threshold  (segment_stats tool)
- "What are the transaction stats for Business customers?" → threshold
- "Show me Business customer stats" → threshold
- "Show me all AML rules" → threshold  (list_rules is a threshold tool — NOT policy)
- "What rules are in the system?" → threshold
- "List all the AML rules" → threshold
- "What transactions are flagged by the layering rule?" → threshold  (list_rules — 'layering' is not a KB topic)
- "Which rule covers layering?" → threshold  (list_rules)
- "Show rule sweep for xyz_column" → threshold  (rule sweep request, even with unknown param — NOT out_of_scope)
- "Show rule sweep for an invalid parameter" → threshold
- "What is the SAR filing rate for Individual?" → threshold  (sar_backtest is a threshold tool)
- "SAR filing rate for Business" → threshold
- "Which rule has the highest FP rate?" → threshold  (list_rules)
- "Which rules generate only false positives?" → threshold
- "Run a SAR backtest for the structuring rule" → threshold  (rule_sar_backtest — NOT out_of_scope)
- "SAR backtest for Elder Abuse" → threshold
- "Show Elder Abuse sweep for Cluster 4" → threshold  (cluster-filtered rule sweep)
- "Run SAR backtest for Activity Deviation ACH in Cluster 2" → threshold
- "Show 2D heatmap for Elder Abuse for Cluster 3" → threshold
- "Which cluster has the most false positives for Velocity Single?" → threshold
- "Which cluster of Business customers has the highest transaction volume?" → segmentation
- "Which Business cluster has the most activity?" → segmentation
- "Which cluster has the most transaction activity?" → segmentation
- "Show Business customer clusters by transaction behavior" → segmentation
- "Run OFAC screening" → ofac
- "Show OFAC sanctions exposure" → ofac
- "Which customers are on the sanctions list?" → ofac
- "How many customers are from sanctioned countries?" → ofac
- "Show me OFAC hits" → ofac
- "Screen customers against SDN list" → ofac
- "What is our Iran/North Korea customer exposure?" → ofac
- "Show comprehensive sanctions hits" → ofac

Rules:
- Output ONLY the label(s), comma-separated. No explanation, no punctuation other than commas.
- A query can map to multiple labels (e.g. threshold,segmentation).
- When in doubt between out_of_scope and a AML label, prefer the AML label.\
"""


class OrchestratorAgent:

    _GREETING = (
        "Hello! I'm ARIA — Agentic Risk Intelligence for AML. I can help you with:\n"
        "- **Threshold tuning** — optimize your alert investigation budget by analyzing FP/FN trade-offs, SAR catch rates, and rule sweep performance across threshold parameters\n"
        "- **Customer segmentation** — identify behavioral risk clusters using K-Means across transaction velocity, volume, and account characteristics\n"
        "- **AML policy Q&A** — answer questions on BSA/AML regulations, FFIEC examination guidance, FinCEN advisories, and Wolfsberg Group best practices\n\n"
        "Try one of the suggested prompts on the left, or ask me a question."
    )

    _OUT_OF_SCOPE = (
        "I can only help with AML-specific topics:\n"
        "- **Threshold tuning** — FP/FN trade-off analysis, SAR catch rates, rule sweep optimization\n"
        "- **Customer segmentation** — K-Means behavioral clustering, alert distribution by segment\n"
        "- **AML policy Q&A** — BSA/AML regulations, FFIEC guidance, FinCEN advisories\n\n"
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

    def _route(self, query: str, last_assistant: str = "") -> list:
        """LLM-based routing — classify query into agent labels."""
        try:
            classify_messages = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
            if last_assistant:
                classify_messages.append({"role": "assistant", "content": last_assistant})
            classify_messages.append({"role": "user", "content": query})
            resp = self._client.chat.completions.create(
                model=OLLAMA_MODEL,
                max_tokens=20,
                temperature=0,
                messages=classify_messages,
            )
            raw = resp.choices[0].message.content or ""
            valid = {"threshold", "segmentation", "ofac", "policy", "greeting", "out_of_scope"}
            labels = [l.strip().lower() for l in raw.split(",") if l.strip().lower() in valid]
        except Exception as e:
            print(f"[orchestrator] classification error: {e} — defaulting to out_of_scope")
            labels = ["out_of_scope"]

        # Keyword override — correct obvious misrouting regardless of LLM output
        q_lower = query.lower()
        is_segmentation = any(w in q_lower for w in ["cluster", "segment", "k-means", "kmeans", "treemap"])
        is_threshold = any(w in q_lower for w in ["sweep", "fp", "fn", "sar", "heatmap", "backtest", "tuning", "threshold"])
        is_rule_query = any(w in q_lower for w in ["rule", "rules", "false positive", "false negative", "precision", "layering", "structuring"])
        # "cluster N" in a sweep/backtest query = filter, not segmentation request
        cluster_as_filter = is_threshold and is_segmentation
        if is_segmentation and not is_threshold:
            labels = ["segmentation"]
            print("[orchestrator] keyword override → segmentation")
        elif cluster_as_filter:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (cluster is a filter, not segmentation)")
        elif is_rule_query and "policy" in labels and "threshold" in labels:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (rule query, dropped policy)")
        elif "out_of_scope" in labels and (is_threshold or is_rule_query) and not is_segmentation:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (rescued from out_of_scope)")

        # OFAC keyword override — always catch sanctions/OFAC queries
        is_ofac = any(w in q_lower for w in [
            "ofac", "sdn", "sanctions", "sanctioned", "sanction list",
            "iran exposure", "north korea exposure", "dprk", "SDN list",
        ])
        if is_ofac:
            labels = ["ofac"]
            print("[orchestrator] keyword override → ofac")

        # Keyword fallback when fine-tuned model ignores classification prompt
        if not labels:
            q = query.lower()
            if any(w in q for w in ["ofac", "sdn", "sanction"]):
                labels = ["ofac"]
            elif any(w in q for w in ["threshold", "sweep", "fp", "fn", "sar", "heatmap", "rule", "alert", "tuning", "backtest"]):
                labels = ["threshold"]
            elif any(w in q for w in ["cluster", "segment", "k-means", "kmeans", "treemap"]):
                labels = ["segmentation"]
            elif any(w in q for w in ["policy", "compliance", "regulation", "bsa", "aml", "wolfsberg", "fincen", "structuring"]):
                labels = ["policy"]
            elif any(w in q for w in ["hello", "hi", "hey", "howdy", "greetings"]):
                labels = ["greeting"]
            else:
                labels = ["out_of_scope"]
            print(f"[orchestrator] keyword fallback labels: {labels}")

        print(f"[orchestrator] routing to: {labels}")
        return labels

    def run(self, query: str, tool_executor, last_assistant: str = "") -> tuple:
        """
        Route query via LLM, run required agents (in parallel if >1), merge results.
        Returns: (combined_text, all_chart_results)
        """
        labels = self._route(query, last_assistant)

        if "greeting" in labels:
            return self._GREETING, []

        # OFAC screening is handled directly via tool_executor (no specialist agent)
        if "ofac" in labels:
            # Detect name lookup: 2+ capitalised words that look like a person/entity name
            import re as _re
            _name_match = _re.search(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query
            )
            if _name_match:
                name = _name_match.group(1)
                text, fig = tool_executor("ofac_name_lookup", {"name": name})
            else:
                text, fig = tool_executor("ofac_screening", {})
                chart_results = [("ofac_screening", {}, fig)] if fig is not None else []
                return text, chart_results
            return text, []

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

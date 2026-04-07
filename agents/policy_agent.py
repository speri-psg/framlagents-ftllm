"""Policy Agent — ChromaDB RAG over AML policy PDFs (OpenAI-compatible client)."""

import os
import re
import sys
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from .base_agent import OLLAMA_BASE_URL, OLLAMA_MODEL

_AGENTS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CHROMA_PATH

COLLECTION_NAME = "framl_kb"
TOP_K = 7

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a FRAML policy and compliance specialist. "
    "You answer questions using ONLY the retrieved policy documents shown below. "
    "CITATION RULES — follow exactly:\n"
    "- You may reference a source ONLY if its exact name appears in this list: {source_list}.\n"
    "- You MUST NOT write any of the following anywhere in your response:\n"
    "  * CFR section or part numbers (e.g. 31 CFR 1010.314, 31 CFR Part 1020)\n"
    "  * U.S.C. section references (e.g. 31 U.S.C. § 5318)\n"
    "  * FinCEN advisory or guidance numbers (e.g. FIN-2020-A005, FIN-2014-A008)\n"
    "  * OCC bulletin or manual codes (e.g. BSA-04, OCC 2000-17)\n"
    "  * Named authors, law firm names, or individuals\n"
    "  * Any statute name not in the retrieved documents (e.g. USA PATRIOT Act, Bank Secrecy Act)\n"
    "  * Specific dollar thresholds or dates not explicitly stated in the retrieved documents\n"
    "- If the retrieved documents address the question, summarise the concepts they discuss and "
    "reference the document by name from the list above only.\n"
    "IMPORTANT: Only trigger the disclaimer below if ALL retrieved documents are clearly off-topic. "
    "Disclaimer (use ONLY when every retrieved chunk is irrelevant): "
    "Begin with exactly: 'Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.' "
    "Then provide only general conceptual guidance — 3 to 5 sentences maximum. No numbers, no citations, no named sources. "
    "Be precise and compliance-focused. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese, Japanese, or other non-English characters. "
)


class PolicyAgent:
    """Handles policy RAG — retrieval + generation, no tool-use loop needed."""

    def __init__(self):
        self.name = "policy"
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = OLLAMA_MODEL
        self.collection = None
        self._load_kb()

    def _load_kb(self):
        if not os.path.exists(CHROMA_PATH):
            print("PolicyAgent: No ChromaDB found — policy RAG disabled.")
            return
        try:
            ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = chroma.get_collection(COLLECTION_NAME, embedding_function=ef)
            count = self.collection.count()
            if count == 0:
                self.collection = None
                print("PolicyAgent: ChromaDB empty.")
            else:
                print(f"PolicyAgent: KB loaded ({count} chunks).")
        except Exception as e:
            print(f"PolicyAgent: ChromaDB error: {e}")

    def retrieve(self, query: str) -> tuple:
        """Returns (context_text, source_names_list)."""
        if self.collection is None:
            return "", []
        try:
            results = self.collection.query(query_texts=[query], n_results=TOP_K)
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            sources = list(dict.fromkeys(m["source"] for m in metas))  # unique, ordered
            sections = [f"[Policy: {m['source']}]\n{d}" for d, m in zip(docs, metas)]
            return "\n\n---\n\n".join(sections), sources
        except Exception as e:
            print(f"PolicyAgent: retrieval error: {e}")
            return "", []

    # Patterns that indicate fabricated inline citations
    _FABRICATED_PATTERNS = [
        re.compile(r"\bFIN-\d{4}-[A-Z]\d+\b"),                      # FIN-2020-A005
        re.compile(r"\b31\s+CFR\s+(Part\s+)?\d+(\.\d+)?\b"),        # 31 CFR 1010.314, 31 CFR Part 1020
        re.compile(r"\b(?:Title\s+)?\d+\s+U\.S\.C\.?\s*(?:§\s*\d+)?\b"),  # 31 U.S.C. § 5318, Title 31 U.S.C.
        re.compile(r"\bU\.S\.C\.?\s*§\s*\d+\b"),                         # U.S.C. § 5318 (standalone)
        re.compile(r"\bOCC\s+\d{4}-\d+\b", re.IGNORECASE),          # OCC 2000-17
        re.compile(r"\bBSA-\d+\b", re.IGNORECASE),                  # BSA-04
        re.compile(r"\(\s*[a-z]\s*\)\s*\(\s*\d+\s*\)\s*\(\s*[A-Z]\s*\)"),  # (a)(2)(A) orphan subsections
        re.compile(r"\(\s*[a-z]\s*\)\s*\(\s*\d+\s*\)"),             # (a)(1) orphan subsections
    ]

    def _strip_fabricated_citations(self, text: str, allowed_sources: list) -> str:
        """Remove fabricated inline citation patterns and unverified Source: lines."""
        allowed_lower = [s.lower() for s in allowed_sources]

        def _is_in_allowed_source(match_str: str) -> bool:
            """True if this matched string is a substring of an allowed source name."""
            m = match_str.lower()
            return any(m in src for src in allowed_lower)

        lines = text.splitlines()
        filtered = []
        for line in lines:
            # Drop Source: lines citing nothing in the retrieved set
            if re.match(r"^\s*source\s*:", line, re.IGNORECASE):
                cited = line.split(":", 1)[1].strip()
                if not any(s.lower() in cited.lower() for s in allowed_sources):
                    print(f"[policy] stripped fabricated source line: {line.strip()}")
                    continue

            # Strip inline fabricated citation tokens — skip if from an allowed source
            cleaned = line
            for pattern in self._FABRICATED_PATTERNS:
                def _replace(m, pat=pattern):
                    match_str = m.group(0)
                    if _is_in_allowed_source(match_str):
                        return match_str  # keep — it's a legitimate retrieved source
                    print(f"[policy] stripped inline citation: {match_str}")
                    return ""
                cleaned = pattern.sub(_replace, cleaned)

            # Clean up artifacts left by stripping: empty parens, double spaces, leading punctuation
            cleaned = re.sub(r"\(\s*\)", "", cleaned)        # empty ()
            cleaned = re.sub(r"\[\s*\]", "", cleaned)        # empty []
            cleaned = re.sub(r"\s+([,\.;:])", r"\1", cleaned)  # space before punctuation
            cleaned = re.sub(r"  +", " ", cleaned).strip()
            if cleaned:
                filtered.append(cleaned)
        return "\n".join(filtered)

    def run(self, query: str, tool_executor=None, policy_context: str = "") -> tuple:
        """Returns (response_text, []) — policy agent produces no charts."""
        context, sources = self.retrieve(query)
        source_list = ", ".join(sources) if sources else "none"
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(source_list=source_list)

        if context:
            user_content = f"## Retrieved Policy Documents\n{context}\n\n## Question\n{query}"
        else:
            user_content = f"## Retrieved Policy Documents\n(No relevant documents found in the knowledge base.)\n\n## Question\n{query}"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content or ""
        text = self._strip_fabricated_citations(text, sources)
        return text, []

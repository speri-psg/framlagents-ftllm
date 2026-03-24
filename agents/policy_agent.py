"""Policy Agent — ChromaDB RAG over AML policy PDFs (OpenAI-compatible client)."""

import os
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
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a FRAML policy and compliance specialist. "
    "You answer questions by referencing AML policies, regulatory guidelines, and best practices "
    "retrieved from the knowledge base. "
    "Always cite the source document when referencing policy content. "
    "If no relevant policy is found, say so clearly rather than guessing. "
    "Be precise and compliance-focused."
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

    def retrieve(self, query: str) -> str:
        if self.collection is None:
            return ""
        try:
            results = self.collection.query(query_texts=[query], n_results=TOP_K)
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            sections = [f"[Policy: {m['source']}]\n{d}" for d, m in zip(docs, metas)]
            return "\n\n---\n\n".join(sections)
        except Exception as e:
            print(f"PolicyAgent: retrieval error: {e}")
            return ""

    def run(self, query: str, tool_executor=None, policy_context: str = "") -> tuple:
        """Returns (response_text, []) — policy agent produces no charts."""
        context = self.retrieve(query)
        if not context:
            return "No relevant policy documents found in the knowledge base.", []

        user_content = f"## Retrieved Policy Documents\n{context}\n\n## Question\n{query}"
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or "", []

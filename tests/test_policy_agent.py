"""Tests for agents/policy_agent.py — retrieval logic (no LLM/ChromaDB required)."""
import pytest
from unittest.mock import patch, MagicMock


def _make_agent():
    """Create a PolicyAgent with LLM mocked."""
    with patch("openai.OpenAI", return_value=MagicMock()):
        from agents.policy_agent import PolicyAgent
        agent = PolicyAgent()
    return agent


# ── Retrieval logic ────────────────────────────────────────────────────────────

class TestPolicyAgentRetrieval:

    def test_retrieve_no_collection_returns_empty(self):
        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch("agents.policy_agent._upload_kb.retrieve", return_value=("", [])):
            from agents.policy_agent import PolicyAgent
            agent = PolicyAgent()
            ctx, sources = agent.retrieve("What is AML?")
        assert ctx == ""
        assert sources == []

    def test_retrieve_merges_upload_context(self):
        upload_ctx = "Uploaded document content about AML."
        upload_sources = ["my_document.pdf"]
        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch("agents.policy_agent._upload_kb.retrieve", return_value=(upload_ctx, upload_sources)):
            from agents.policy_agent import PolicyAgent
            agent = PolicyAgent()
            ctx, sources = agent.retrieve("What is AML?")
        assert upload_ctx in ctx
        assert "my_document.pdf" in sources

    def test_retrieve_deduplicates_sources(self):
        """Sources from uploads that would overlap shouldn't appear twice."""
        upload_ctx = "Some context."
        upload_sources = ["shared_source.pdf"]
        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch("agents.policy_agent._upload_kb.retrieve", return_value=(upload_ctx, upload_sources)):
            from agents.policy_agent import PolicyAgent
            agent = PolicyAgent()
            _, sources = agent.retrieve("What is AML?")
        assert sources.count("shared_source.pdf") == 1

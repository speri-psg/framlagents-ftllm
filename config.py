"""
config.py — Central configuration for FRAML AI Assistant.

Override LLM endpoint via environment variables:
    set OLLAMA_BASE_URL=http://localhost:11434/v1   (default — Ollama)
    set OLLAMA_MODEL=qwen2.5:7b                     (default)

To use a fine-tuned model via vLLM after training:
    set OLLAMA_BASE_URL=http://localhost:8000/v1
    set OLLAMA_MODEL=qwen-framl
"""

import os

# ── LLM endpoint ──────────────────────────────────────────────────────────────
# Honour legacy VLLM_* names too so existing docs still work.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",
                            os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1"))
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",
                            os.getenv("VLLM_MODEL", "qwen2.5:7b"))

# ── Local data paths (all relative to this file's directory) ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

ALERTS_CSV   = os.path.join(_HERE, "docs", "custs_accts_txns_alerts.csv")
SS_CSV       = os.path.join(_HERE, "docs", "ss_segmentation_data.csv")
SAR_CSV      = os.path.join(_HERE, "docs", "sar_simulation.csv")
SS_FILES_DIR = os.path.join(_HERE, "ss_files")
CHROMA_PATH  = os.path.join(_HERE, "chroma_db")
DOCS_DIR     = os.path.join(_HERE, "docs")
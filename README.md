---
title: Agentic AML Demo
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: other
short_description: "Agentic AML demo: threshold tuning and segmentation"
---

# Agentic AML Demo

> **If you see a "Restart the Space" prompt, click it to launch the demo — startup takes ~2 minutes.**

An AI-powered Anti-Money Laundering (AML) analytics assistant built with a fine-tuned **Qwen 2.5 7B** model and an agentic tool-calling architecture.

## What it does

Chat with the assistant to:

- **Threshold tuning** — sweep FP/FN trade-offs as alert thresholds change across transaction metrics
- **SAR backtest** — test how many true SARs a threshold configuration would have caught
- **2D rule sweep** — optimize two parameters simultaneously with an interactive heatmap + drill-down
- **Dynamic Segmentation** — cluster customers into behavioral segments using K-Means
- **AML policy Q&A** — ask compliance questions answered from a regulatory knowledge base (FFIEC, Wolfsberg, FinCEN)

## Model

Fine-tuned from `Qwen/Qwen2.5-7B-Instruct` on ~450 domain-specific AML analytics examples using supervised fine-tuning (SFT). The model learns to:

- Classify user intent and route to the correct analytics tool
- Call Python analytics functions with the right parameters
- Interpret results and provide AML-relevant insights without hallucinating numbers

## Architecture

```
User prompt
    │
    ▼
OrchestratorAgent  (intent classification + routing)
    │
    ├── ThresholdAgent     (FP/FN sweep, SAR backtest, 2D sweep)
    ├── SegmentationAgent  (K-Means clustering, cluster stats)
    └── PolicyAgent        (ChromaDB RAG over regulatory docs)
```

Tools are called via Ollama's OpenAI-compatible API with structured tool schemas. All analytics are pre-computed in Python — the model copies verbatim numbers to eliminate hallucination.

## Dataset

Synthetic dataset of 5,000 customer accounts with transaction behavior metrics, alert flags, false positive/negative labels, and simulated SAR outcomes across Business and Individual segments.

## Tech stack

- **LLM**: Qwen 2.5 7B (Q4_K_M GGUF) via Ollama
- **UI**: Plotly Dash + dash-bootstrap-components
- **Analytics**: pandas, scikit-learn, Plotly
- **Knowledge base**: ChromaDB + sentence-transformers
- **Training**: Unsloth SFT on vast.ai (RTX 3090)

## License

This repository uses a dual-licensing structure:

**Source code** (`agents/`, `application.py`, tools, scripts) — [Apache 2.0](LICENSE)
Free to use, modify, and distribute for any purpose.

**Model weights and training data** (`finetune/data/`) — [Modified OpenRAIL-M](LICENSE_MODEL)
Free for personal use, academic research, and organizations with annual revenue
and total funding each below **USD $2 million**. Commercial use above that threshold
requires a separate license. The model may not be used to build a competing
AML transaction monitoring or financial crime analytics product or service.

For commercial licensing enquiries, open an issue in this repository.

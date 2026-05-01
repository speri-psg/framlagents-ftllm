#!/bin/bash
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
HF_REPO="speri/aria-v2"
GGUF_FILE="aria-v2-q4km.gguf"          # Q4_K_M ~2.5 GB; matches what was uploaded to HF
GGUF_PATH="/data/${GGUF_FILE}"
MODEL_NAME="aria-v2"

# ── 1. Start Ollama daemon ────────────────────────────────────────────────────
ollama serve &
echo "[startup] Ollama starting..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[startup] Ollama ready."

# ── 2. Download GGUF if not cached in persistent storage ─────────────────────
if [ ! -f "$GGUF_PATH" ]; then
    echo "[startup] Downloading ${GGUF_FILE} from HuggingFace Hub (${HF_REPO})..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${GGUF_FILE}',
    local_dir='/data',
    token=os.environ.get('HF_TOKEN'),
)
print('[startup] Download complete.')
"
else
    echo "[startup] GGUF already cached at ${GGUF_PATH} — skipping download."
fi

# ── 3. Write Modelfile via Python (avoids heredoc escaping issues) ─────────────
python3 << 'PYEOF'
import os

SYSTEM_PROMPT = (
    "You are ARIA, an AML (Anti-Money Laundering) analytics AI assistant built by Xceed. "
    "You analyze false positive/false negative trade-offs in AML alert thresholds, "
    "perform customer behavioral segmentation, and interpret clustering results. "
    "Use the available tools to retrieve data, then provide clear, analytical insights. "
    "IMPORTANT: You MUST respond entirely in English. "
    "Be concise and reference specific numbers when interpreting results."
)

lines = [
    "FROM /data/aria-v2-q4km.gguf",
    "",
    "PARAMETER num_ctx 8192",
    "PARAMETER temperature 0.1",
    "PARAMETER top_p 0.9",
    "PARAMETER stop <turn|>",
    "PARAMETER stop <eos>",
    "",
    'TEMPLATE """',
    "{{ if .System }}<|turn>user",
    "{{ .System }}<turn|>",
    "{{ end }}",
    "{{ range .Messages }}",
    '{{ if eq .Role "user" }}<|turn>user',
    "{{ .Content }}<turn|>",
    "<|turn>model",
    '{{ else if eq .Role "assistant" }}{{ .Content }}<turn|>',
    '{{ else if eq .Role "tool" }}<|turn>tool',
    "{{ .Content }}<turn|>",
    "<|turn>model",
    "{{ end }}",
    "{{ end }}",
    '"""',
    "",
    f'SYSTEM "{SYSTEM_PROMPT}"',
]

content = "\n".join(lines)
with open("/tmp/Modelfile.aml", "w") as f:
    f.write(content)
print("[startup] Modelfile written.")
PYEOF

# ── 4. Register model with Ollama (skip if already registered) ───────────────
if ollama list | grep -q "^${MODEL_NAME}"; then
    echo "[startup] Model ${MODEL_NAME} already registered — skipping create."
else
    echo "[startup] Registering model as ${MODEL_NAME}..."
    ollama create "${MODEL_NAME}" -f /tmp/Modelfile.aml
    echo "[startup] Model registered."
fi

# ── 5. Launch Dash app ────────────────────────────────────────────────────────
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL="${MODEL_NAME}"
echo "[startup] Starting Dash app on port 7860..."
exec python3 /app/application.py

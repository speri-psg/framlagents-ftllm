#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AWS Demo Setup — qwen-framl on g4dn.xlarge (T4 16GB)
# Run once after launching instance:
#   chmod +x aws_demo_setup.sh && ./aws_demo_setup.sh
#
# Prerequisites:
#   - AWS g4dn.xlarge with Deep Learning AMI (CUDA pre-installed)
#   - Port 11434 open in the instance's Security Group (inbound TCP)
#   - HF_TOKEN env var set if the HF repo is private:
#       export HF_TOKEN=hf_xxx
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HF_REPO="speri420/qwen-framl-v10"
GGUF_FILENAME="qwen-framl-q4km.gguf"
GGUF_PATH="/home/ubuntu/${GGUF_FILENAME}"
OLLAMA_MODEL_NAME="qwen-framl"
OLLAMA_PORT=11434

echo "=== Step 1/5 — Install Ollama ==="
if command -v ollama &>/dev/null; then
    echo "Ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed."
fi

echo ""
echo "=== Step 2/5 — Download GGUF from HuggingFace ==="
if [ -f "${GGUF_PATH}" ] && [ "$(stat -c%s "${GGUF_PATH}")" -gt 1000000000 ]; then
    echo "GGUF already present: ${GGUF_PATH}"
else
    HF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILENAME}"
    HEADERS=""
    if [ -n "${HF_TOKEN:-}" ]; then
        HEADERS="--header Authorization: Bearer ${HF_TOKEN}"
    fi
    echo "Downloading from ${HF_URL} ..."
    wget --progress=bar:force -O "${GGUF_PATH}" \
        ${HF_TOKEN:+--header "Authorization: Bearer ${HF_TOKEN}"} \
        "${HF_URL}"
    echo "Download complete: $(du -h "${GGUF_PATH}" | cut -f1)"
fi

echo ""
echo "=== Step 3/5 — Write Modelfile ==="
MODELFILE_PATH="/tmp/Modelfile.framl"
cat > "${MODELFILE_PATH}" <<'MODELFILE'
FROM /home/ubuntu/qwen-framl-q4km.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

SYSTEM "You are a FRAML (Fraud + AML) analytics AI assistant. You analyze false positive/false negative trade-offs in AML alert thresholds, perform customer behavioral segmentation, and interpret clustering results. Use the available tools to retrieve data, then provide clear, analytical insights. Be concise and reference specific numbers when interpreting results."
MODELFILE

echo "Modelfile written to ${MODELFILE_PATH}"

echo ""
echo "=== Step 4/5 — Start Ollama server ==="
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
nohup ollama serve > /tmp/ollama.log 2>&1 &
echo "Waiting for Ollama to start..."
sleep 5

# Verify it's up
if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null; then
    echo "Ollama is running on port ${OLLAMA_PORT}"
else
    echo "ERROR: Ollama did not start. Check /tmp/ollama.log"
    tail -20 /tmp/ollama.log
    exit 1
fi

echo ""
echo "=== Step 5/5 — Register model with Ollama ==="
ollama create "${OLLAMA_MODEL_NAME}" -f "${MODELFILE_PATH}"
echo "Model registered: ${OLLAMA_MODEL_NAME}"

echo ""
echo "=== Smoke test ==="
ollama list

echo ""
echo "────────────────────────────────────────────────────────"
echo "Setup complete."
echo ""
echo "On your local machine, set these env vars before starting the Dash app:"
echo ""
echo "  PowerShell:"
INSTANCE_IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<your-aws-public-ip>")
echo "    \$env:OLLAMA_BASE_URL = \"http://${INSTANCE_IP}:${OLLAMA_PORT}/v1\""
echo "    \$env:OLLAMA_MODEL    = \"${OLLAMA_MODEL_NAME}\""
echo "    .venv\Scripts\python.exe application.py"
echo ""
echo "  CMD:"
echo "    set OLLAMA_BASE_URL=http://${INSTANCE_IP}:${OLLAMA_PORT}/v1"
echo "    set OLLAMA_MODEL=${OLLAMA_MODEL_NAME}"
echo "    .venv\Scripts\python.exe application.py"
echo "────────────────────────────────────────────────────────"

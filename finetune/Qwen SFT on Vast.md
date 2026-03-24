# Qwen SFT on Vast.ai — Session Notes

**Date:** 2026-03-24
**Goal:** Fine-tune Qwen2.5-7B-Instruct on FRAML tool-calling dataset, export to GGUF, serve via Ollama
**Instance:** vast.ai GPU instance (RTX 3090 or similar, 24GB VRAM)
**Repo:** https://github.com/speri-psg/framlagents-ftllm

---

## 1. Setup

### Instance Type
- Use **PyTorch template** (not bare CUDA) — comes with PyTorch + CUDA + cuDNN pre-installed
- RTX 3090 (24GB) recommended at ~$0.30/hr, RTX 4090 (24GB) ~$0.50/hr
- **CRITICAL: Must be CUDA 12.x — do NOT use CUDA 13.x (RTX 5090)** — Ollama 0.18.2 does not support it
- **Set disk to 50GB+** — default 16GB root disk fills up immediately (Ollama install + model files)
- **Expose port 11434** when creating the instance (for Ollama)

### Get the code
In the Jupyter terminal:
```bash
git clone https://github.com/speri-psg/framlagents-ftllm.git
cd framlagents-ftllm
```
Then open `finetune/qwen_framl_vastai.ipynb` in Jupyter.

---

## 2. Notebook Cell Execution

### Cell 1 — Install Dependencies
```python
%pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
%pip install -q torch>=2.1.0 transformers>=4.46.0 trl>=0.12.0 peft>=0.13.0 \
              datasets>=3.0.0 bitsandbytes>=0.44.0 accelerate>=0.34.0 \
              sentencepiece huggingface_hub
```

### Cell 2 — Environment & Imports
**Important:** Move `from unsloth import FastLanguageModel` to the top, before trl/transformers/peft imports. Otherwise you get:
```
UserWarning: Unsloth should be imported before [trl, transformers, peft]
```

Correct order:
```python
import os
os.environ["HF_HOME"] = "/dev/shm/hf_cache"

# Unsloth must come before trl/transformers/peft
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import json, torch
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
```

### Cell 3 — Configuration
Key settings used:
```python
BASE_MODEL   = "unsloth/Qwen2.5-7B-Instruct"
EPOCHS       = 5
BATCH_SIZE   = 2
GRAD_ACCUM   = 4      # effective batch = 8
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
```
Paths set to `/dev/shm/` (RAM disk) — fast but lost on restart.

### Cell 4 — Model Download
Downloads `unsloth/Qwen2.5-7B-Instruct` (4-bit BNB format) to `/dev/shm/hf_cache`.
**Size:** ~7.15GB
**Time:** ~8 min at 33MB/s

Expected output:
```
Download complete: 100%  7.15G/7.15G [08:24<00:00, 33.3MB/s]
Fetching 2 files: 100%  2/2
Loading weights: 100%  339/339 [00:01<00:00, 208.17it/s]
unsloth/qwen2.5-7b-instruct does not have a padding token — using <|PAD_TOKEN|>
```
All warnings are harmless.

### Cell 5 — LoRA Adapters
Expected output:
```
Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = 0.05.
Unsloth will patch all other layers, except LoRA matrices, causing a performance hit.
Unsloth 2026.3.11 patched 28 layers with 0 QKV layers, 0 O layers and 0 MLP layers.
Trainable params: 40,370,176  (0.82% of 4,931,917,312)
```
- The dropout warning is expected — set `LORA_DROPOUT=0` for max speed if overfitting is not a concern
- Only 0.82% of parameters are trained (LoRA efficiency)

### Cell 6 — Load Dataset
Loads `finetune/data/framl_train.jsonl` — 60 tool-calling examples.

### Cell 7 — Training
Training loss curve (5 epochs, 40 steps):

| Step | Loss |
|------|------|
| 5 | 2.5704 |
| 10 | 1.3535 |
| 15 | 0.6478 |
| 20 | 0.4013 |
| 25 | 0.2965 |
| 30 | 0.2040 |
| 35 | 0.1637 |
| 40 | 0.2031 |

**Final loss: 0.73** (epoch average across all 5 epochs)
Sharp drop from 2.57 → 0.65 in first 15 steps — model quickly learned FRAML tool-calling pattern.

---

## 3. GGUF Export — Issues & Fixes

### Problem: Unsloth's `save_pretrained_gguf` kept failing
`capture_output=True` in Unsloth hid the real error. Diagnosis required running the converter manually.

### Disk Space Situation
```
/dev/shm    31GB total  — RAM-backed, fast
/           16GB total  — root overlay, 95% full (885MB free)
/workspace  —           — no space available
```

**Key lesson:** `/dev/shm` is the only usable large disk on this instance. Must manage space carefully.

### Fix: Manual GGUF conversion (bypass Unsloth's wrapper)

**Step 1 — Clear HF cache (no longer needed after model loaded):**
```bash
rm -rf /dev/shm/hf_cache
```
Freed ~21GB.

**Step 2 — Convert HF safetensors → f16 GGUF:**
```bash
/venv/main/bin/python /root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py \
    --outfile /dev/shm/qwen-framl-f16.gguf \
    --outtype f16 \
    /dev/shm/qwen-framl-gguf
```
Output: `Model successfully exported to /dev/shm/qwen-framl-f16.gguf` (~14GB)

Note: Numpy `RuntimeWarning: overflow encountered in divide` warnings are harmless.

**Step 3 — Delete safetensors to free space:**
```bash
rm /dev/shm/qwen-framl-gguf/model-*.safetensors
```
Freed ~15GB.

**Step 4 — Quantize f16 → q4_k_m:**
```bash
/root/.unsloth/llama.cpp/llama-quantize \
    /dev/shm/qwen-framl-f16.gguf \
    /dev/shm/qwen-framl-q4km.gguf \
    Q4_K_M
```
Output: `qwen-framl-q4km.gguf` — **4.68GB**

**Important:** Cannot quantize from q8_0 → q4_k_m directly.
Must go: `HF safetensors → f16 GGUF → q4_k_m GGUF`

**Step 5 — Clean up f16:**
```bash
rm /dev/shm/qwen-framl-f16.gguf
```

### Space management summary
| Stage | /dev/shm free |
|-------|--------------|
| After training | ~11GB |
| After rm hf_cache | ~32GB |
| After f16 written | ~3GB |
| After rm safetensors | ~17GB |
| After q4_k_m written | ~12GB |
| After rm f16 | ~26GB |

---

## 4. Ollama Setup — Issues & Fixes

### Problem: RTX 5090 + CUDA 13 not supported
This session used an RTX 5090 (CUDA 13.0). Ollama 0.18.2 failed GPU discovery and fell back to CPU inference (~3+ min per response).

**Root cause:** Ollama's runner uses `/tmp` for a temp binary during CUDA init. `/tmp` was full (on root disk), so CUDA init failed silently.

**Fix attempted:** `TMPDIR=/dev/shm` and `OLLAMA_LLM_LIBRARY=cuda_v13` — still fell back to CPU. RTX 5090 is too new for Ollama 0.18.2.

**Solution for next session: Use RTX 3090 or RTX 4090 (CUDA 12.x)**. Check CUDA version shown on the instance — must be **12.x not 13.x**.

---

### Problem: Ollama install.sh failed — root disk full
```
tar: lib/ollama/vulkan: Cannot mkdir: No space left on device
```
Install script extracts to `/usr/local/` which is on the 16GB root overlay (full).

**Fix:** Download tarball and extract to `/dev/shm`, run binary from there:
```bash
curl -L "https://github.com/ollama/ollama/releases/download/v0.18.2/ollama-linux-amd64.tar.zst" -o /dev/shm/ollama.tar.zst
tar --use-compress-program=unzstd -xf /dev/shm/ollama.tar.zst -C /dev/shm
```
Binary lands at `/dev/shm/bin/ollama`.

**Problem:** `/dev/shm` is mounted `noexec` — binaries can't run from there.

**Fix:** Copy to `/dev` (not noexec):
```bash
cp /dev/shm/bin/ollama /dev/ollama
chmod +x /dev/ollama
```

### Required env vars to run Ollama with no root disk space
All of these must be set — Ollama tries to write to root disk by default:
```bash
LD_LIBRARY_PATH=/dev/shm/lib/ollama/cuda_v13:/dev/shm/lib/ollama
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_MODELS=/dev/shm/ollama_models
HOME=/dev/shm                    # prevents ~/.ollama key generation on root disk
TMPDIR=/dev/shm                  # prevents /tmp usage during CUDA init
OLLAMA_LLM_LIBRARY=cuda_v13      # force CUDA 13 library
```

### Start server (one line for vast.ai terminal)
```bash
LD_LIBRARY_PATH=/dev/shm/lib/ollama/cuda_v13:/dev/shm/lib/ollama OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=/dev/shm/ollama_models HOME=/dev/shm TMPDIR=/dev/shm OLLAMA_LLM_LIBRARY=cuda_v13 nohup /dev/ollama serve > /dev/shm/ollama.log 2>&1 &
```

### Write Modelfile (use /dev/shm not /tmp)
```bash
printf 'FROM /dev/shm/qwen-framl-q4km.gguf\nPARAMETER num_ctx 4096\nPARAMETER temperature 0.1\nPARAMETER top_p 0.9\nPARAMETER stop "<|im_end|>"\nSYSTEM "You are a FRAML (Fraud + AML) analytics AI assistant. Use the available tools to retrieve data, then provide clear, analytical insights."\n' > /dev/shm/Modelfile.framl
```

### Register model
```bash
LD_LIBRARY_PATH=/dev/shm/lib/ollama OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=/dev/shm/ollama_models HOME=/dev/shm /dev/ollama create qwen-framl -f /dev/shm/Modelfile.framl
```

### Verify GPU is being used
```bash
grep "inference compute" /dev/shm/ollama.log
```
Must show `library=cuda` not `library=cpu`. If CPU, CUDA init failed — check TMPDIR and disk space.

### Start server (on CUDA 12 instance with enough disk)

---

## 5. Connect Dash App

On your local Windows machine, set env vars before starting `application.py`:

**CMD:**
```bat
set OLLAMA_BASE_URL=http://<vast-ip>:<mapped-port>/v1
set OLLAMA_MODEL=qwen-framl
.venv\Scripts\python.exe application.py
```

**PowerShell:**
```powershell
$env:OLLAMA_BASE_URL = "http://<vast-ip>:<mapped-port>/v1"
$env:OLLAMA_MODEL    = "qwen-framl"
.venv\Scripts\python.exe application.py
```

Find `<vast-ip>` and `<mapped-port>` in the vast.ai instance dashboard under **Open Ports**.

---

## 6. Key Lessons Learned

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Unsloth GGUF export fails silently | `capture_output=True` hides converter error | Run converter manually in terminal |
| Cannot quantize q8_0 → q4_k_m | llama-quantize restriction | Must go HF → f16 GGUF → q4_k_m |
| `/dev/shm` space tight | HF cache still present after model loaded | `rm -rf /dev/shm/hf_cache` after Cell 4 |
| `/` root disk full | 16GB overlay, 95% used | Use 50GB+ disk on next instance |
| Unsloth import warning | trl imported before unsloth | Move unsloth import to top of Cell 2 |
| Ollama install fails | Root disk full, tar can't extract | Download tarball to /dev/shm, copy binary to /dev/ |
| `/dev/shm` is noexec | Linux mount flag | Copy binary to `/dev/` which allows execution |
| Ollama writes to `~/.ollama` | Default HOME is /root on root disk | Set `HOME=/dev/shm` |
| Ollama uses `/tmp` for CUDA init | Default TMPDIR on root disk | Set `TMPDIR=/dev/shm` |
| Ollama falls back to CPU | RTX 5090 + CUDA 13 not supported | Use RTX 3090/4090 with CUDA 12.x |
| VRAM not released after kernel restart | PyTorch holds GPU memory until process exits | `kill -9 <ipykernel PID>` from terminal |

---

## 7. Unsloth vs llama.cpp vs Ollama — Quick Reference

| Tool | Role |
|------|------|
| **Unsloth** | Fast QLoRA training (2x speed, 60% less VRAM vs HuggingFace) |
| **llama.cpp** | C++ inference engine that runs GGUF models; also does quantization |
| **Ollama** | User-friendly wrapper around llama.cpp — model management + OpenAI API |
| **GGUF** | Single-file model format consumed by Ollama/llama.cpp |
| **q4_k_m** | 4-bit quantization — ~4.5GB for 7B model, sweet spot for quality vs size |

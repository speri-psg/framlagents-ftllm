#!/bin/bash
# vastai_setup.sh
# Run this once after SSHing into your vast.ai instance to set up the fine-tuning environment.
# Tested on PyTorch 2.1+ images (CUDA 12.1).

set -e

echo "=== 1. Install Unsloth ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "=== 2. Install remaining dependencies ==="
pip install -r requirements_finetune.txt

echo "=== 3. (Optional) Login to HuggingFace Hub ==="
echo "Run: huggingface-cli login"

echo "=== 4. (Optional) Login to Weights & Biases ==="
echo "Run: wandb login"

echo ""
echo "Setup complete. Next steps:"
echo "  1. Upload data:   scp -P <port> data/framl_train.jsonl root@<host>:~/finetune/data/"
echo "  2. Run training:  python train_unsloth.py"
echo "  3. Merge model:   python train_unsloth.py --merge_and_save (adds ~20 min)"
echo ""
echo "To serve with vLLM after merging:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "      --model outputs/qwen-framl-threshold-merged \\"
echo "      --enable-auto-tool-choice \\"
echo "      --tool-call-parser hermes \\"
echo "      --port 8000"

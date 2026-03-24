"""
train_unsloth.py

QLoRA fine-tuning of Qwen2.5-7B-Instruct on FRAML tool-calling dataset.
Uses Unsloth for 2x faster training and ~60% less VRAM vs standard HuggingFace.

Tested on:
  - RTX 4090 (24 GB VRAM) — fits comfortably with QLoRA 4-bit
  - A100 40 GB               — comfortable, can increase batch size

Run on vast.ai instance:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install -r requirements_finetune.txt
    python train_unsloth.py

After training, push the LoRA adapter to HuggingFace Hub:
    huggingface-cli login
    python train_unsloth.py --push_to_hub your-hf-username/qwen-framl-threshold

Serve the merged model with vLLM (OpenAI-compatible endpoint):
    python -m vllm.entrypoints.openai.api_server \
        --model outputs/qwen-framl-threshold-merged \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Unsloth must be imported before transformers / trl
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"   # 4-bit quantised via Unsloth
MAX_SEQ_LEN = 4096
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = str(_THIS_DIR / "data" / "framl_train.jsonl")
DEFAULT_OUT = str(_THIS_DIR / "outputs" / "qwen-framl-threshold")


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} training examples from {path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Format: apply Qwen2.5 chat template to each example
# ---------------------------------------------------------------------------
def format_example(example, tokenizer):
    """Apply the Qwen2.5 chat template to a messages list."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME, help="Base model (HuggingFace ID or local path)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to framl_train.jsonl")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--push_to_hub", default=None, help="HuggingFace repo ID to push adapter (optional)")
    parser.add_argument("--merge_and_save", action="store_true", help="Merge LoRA into base and save full model")
    args = parser.parse_args()

    # ── Load base model with Unsloth 4-bit QLoRA ──────────────────────────
    print(f"\nLoading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,       # QLoRA — use False for full fine-tune on A100 80GB
        dtype=None,              # auto-detect (bfloat16 on Ampere+)
    )

    # Apply Qwen2.5 chat template (includes tool-call tokens)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ── Add LoRA adapters ──────────────────────────────────────────────────
    print(f"Attaching LoRA (r={args.lora_r}, alpha={LORA_ALPHA}) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves VRAM
        random_state=42,
    )

    # ── Load & format dataset ──────────────────────────────────────────────
    raw_ds = load_jsonl(args.data)
    formatted_ds = raw_ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=raw_ds.column_names,
    )
    print(f"Dataset formatted. Sample:\n{formatted_ds[0]['text'][:400]}\n...")

    # ── SFT Trainer config ─────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",        # set to "wandb" if you want W&B logging
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=True,            # pack short examples for efficiency
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_ds,
        args=sft_config,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    trainer_stats = trainer.train()
    print(f"\nTraining complete. Loss: {trainer_stats.training_loss:.4f}")

    # ── Save LoRA adapter ──────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"LoRA adapter saved → {out_dir}")

    # ── Optional: merge LoRA into base weights and save full model ─────────
    if args.merge_and_save:
        merged_dir = str(out_dir) + "-merged"
        print(f"\nMerging LoRA into base model → {merged_dir} ...")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print("Merge complete.")

    # ── Optional: push adapter to HuggingFace Hub ─────────────────────────
    if args.push_to_hub:
        print(f"\nPushing adapter to HuggingFace Hub: {args.push_to_hub} ...")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print("Push complete.")

    print("\nDone.")
    print(f"\nTo serve with vLLM (after merging):")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {str(out_dir)}-merged \\")
    print(f"      --enable-auto-tool-choice \\")
    print(f"      --tool-call-parser hermes")


if __name__ == "__main__":
    main()

"""
test_ollama.py — verify Ollama is reachable and tool calling works.

Usage:
    python test_ollama.py
"""

import json
import sys
from openai import OpenAI

BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:7b"

client = OpenAI(base_url=BASE_URL, api_key="ollama")

# ── 1. Basic connectivity ─────────────────────────────────────────────────────
print("1. Checking Ollama connectivity ...")
try:
    models = client.models.list()
    names = [m.id for m in models.data]
    print(f"   OK — models available: {names}")
    if MODEL not in names:
        print(f"   WARNING: '{MODEL}' not found. Run:  ollama pull {MODEL}")
        sys.exit(1)
except Exception as e:
    print(f"   FAILED: {e}")
    print("   Is Ollama running? Start it with:  ollama serve")
    sys.exit(1)

# ── 2. Plain chat ─────────────────────────────────────────────────────────────
print("\n2. Plain chat response ...")
resp = client.chat.completions.create(
    model=MODEL,
    max_tokens=64,
    messages=[{"role": "user", "content": "Say 'FRAML ready' and nothing else."}],
)
print(f"   Response: {resp.choices[0].message.content!r}")

# ── 3. Tool calling ───────────────────────────────────────────────────────────
print("\n3. Tool calling test ...")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "threshold_tuning",
            "description": "Analyze FP/FN trade-offs for a customer segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {"type": "string", "enum": ["Business", "Individual"]},
                    "threshold_column": {
                        "type": "string",
                        "enum": ["AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"],
                    },
                },
                "required": ["segment", "threshold_column"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model=MODEL,
    max_tokens=256,
    tools=TOOLS,
    tool_choice="auto",
    messages=[
        {
            "role": "system",
            "content": "You are a FRAML analyst. Use the threshold_tuning tool when asked about FP/FN thresholds.",
        },
        {
            "role": "user",
            "content": "Show me FP/FN trade-off for business customers using weekly transaction count.",
        },
    ],
)

msg = resp.choices[0].message
if msg.tool_calls:
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    print(f"   Tool called: {tc.function.name}")
    print(f"   Arguments:   {args}")
    if args.get("segment") == "Business" and args.get("threshold_column") == "AVG_TRXNS_WEEK":
        print("   PASS — correct tool and parameters.")
    else:
        print("   PARTIAL — tool called but wrong parameters (expected Business / AVG_TRXNS_WEEK).")
        print("   This is normal for the base model; fine-tuning will fix it.")
else:
    print(f"   No tool call — model responded with text: {msg.content!r}")
    print("   Tool calling may not be reliable on the base model.")
    print("   Fine-tuning on framl_train.jsonl will fix this.")

print("\nDone. If steps 1-2 passed, run: python application.py")

"""Base agent — shared agentic tool-use loop using OpenAI-compatible API (Ollama / vLLM)."""

import json
import sys
import os
from openai import OpenAI

# Add project root to path so config is importable regardless of working dir
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Keep legacy names so any external code that imported these still works
VLLM_BASE_URL = OLLAMA_BASE_URL
MODEL_NAME    = OLLAMA_MODEL

MAX_TOOL_ITERATIONS = 6  # prevent infinite loops if model misfires


class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: list):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools  # OpenAI function-calling format
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = OLLAMA_MODEL

    def run(self, query: str, tool_executor, policy_context: str = "") -> tuple:
        """
        Agentic loop. Calls tools until the model returns a final text response.
        Returns: (response_text, [(tool_name, tool_input, fig), ...])
        """
        user_content = f"{policy_context}\n\n{query}".strip() if policy_context else query

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        chart_results = []

        create_kwargs = dict(
            model=self.model,
            max_tokens=1024,
            messages=messages,
        )
        if self.tools:
            create_kwargs["tools"] = self.tools
            create_kwargs["tool_choice"] = "auto"

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(**create_kwargs)
            msg = response.choices[0].message

            if msg.tool_calls:
                # Append assistant turn with tool_calls for context
                messages.append({
                    "role": "assistant",
                    "content": msg.content,  # may be None
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool call; append individual tool result messages
                for tc in msg.tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}
                    print(f"[{self.name}] tool call: {tc.function.name}({tool_input})")
                    result_text, fig = tool_executor(tc.function.name, tool_input)
                    if fig is not None:
                        chart_results.append((tc.function.name, tool_input, fig))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

                create_kwargs["messages"] = messages

            else:
                # Final text response (or base model answered without tool call)
                return msg.content or "", chart_results

        # Exceeded max iterations — return whatever text we have
        print(f"[{self.name}] WARNING: hit MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS})")
        return msg.content or "[No response after max tool iterations]", chart_results

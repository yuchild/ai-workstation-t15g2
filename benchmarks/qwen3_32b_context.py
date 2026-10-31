from ollama import chat

MODEL = "qwen3:32b"
CONTEXTS = [8192, 16384, 32768]

PROMPT = """
Design a Python coding agent with:
- Read-only filesystem access restricted to a workspace
- Structured audit logging
- Human approval before file modifications
- A maximum of five tool calls per task

Explain the architecture and provide concise Python pseudocode.
"""

for ctx in CONTEXTS:
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        options={
            "num_ctx": ctx,
            "temperature": 0,
            "num_predict": 700,
        },
        keep_alive="5m",
    )

    generated = response.eval_count or 0
    generation_s = (response.eval_duration or 0) / 1e9
    prompt_s = (response.prompt_eval_duration or 0) / 1e9
    total_s = (response.total_duration or 0) / 1e9

    print(f"\nContext: {ctx:,}")
    print(f"Generated tokens: {generated}")
    print(f"Generation speed: {generated / generation_s:.2f} tok/s"
          if generation_s else "Generation speed: N/A")
    print(f"Prompt processing: {prompt_s:.2f}s")
    print(f"Total duration: {total_s:.2f}s")

import json
import urllib.request

MODEL = "qwen2.5-coder:14b"

for context in [16384, 32768]:
    print(f"\nTesting context: {context}")

    payload = {
        "model": MODEL,
        "prompt": (
            "Write a Python FastAPI application with "
            "a healthcheck endpoint, Pydantic validation, "
            "and pytest tests."
        ),
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": context,
            "num_predict": 512,
            "temperature": 0,
        },
    }

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)

        seconds = result["eval_duration"] / 1e9
        tokens = result["eval_count"]

        print("Generated tokens:", tokens)
        print("Tokens/sec:", round(tokens / seconds, 2))
        print("Total seconds:", round(result["total_duration"] / 1e9, 2))

    except Exception as error:
        print("Error:", error)

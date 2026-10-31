from pathlib import Path
from ollama import chat

WORKSPACE = Path("agent_workspace").resolve()
MODEL = "qwen3-agent"


def read_file(filename: str) -> str:
    target = (WORKSPACE / filename).resolve()

    if not target.is_relative_to(WORKSPACE):
        raise PermissionError("Access outside workspace denied")

    if not target.is_file():
        raise FileNotFoundError(filename)

    if target.stat().st_size > 100_000:
        raise ValueError("File exceeds size limit")

    return target.read_text(encoding="utf-8")


def analyze_file(filename: str) -> str:
    content = read_file(filename)

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code review assistant. "
                    "Identify bugs, maintainability issues, "
                    "and missing tests. "
                    "Provide suggestions only. "
                    "Do not claim to modify files."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Review the following file: {filename}\n\n"
                    f"```python\n{content}\n```"
                ),
            },
        ],
        options={"num_ctx": 16384},
    )

    return response.message.content


if __name__ == "__main__":
    print(analyze_file("example.py"))

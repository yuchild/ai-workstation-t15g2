import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from agents.langgraph.read_only_graph import tool_node, MAX_TOOL_CALLS

calls = [
    {
        "name": "read_file",
        "args": {"filename": "example.py"},
        "id": f"test-{i}",
        "type": "tool_call",
    }
    for i in range(6)
]

state = {
    "messages": [AIMessage(content="", tool_calls=calls)],
    "tool_calls_used": 0,
}

result = tool_node(state)

assert result["tool_calls_used"] == MAX_TOOL_CALLS
assert len(result["messages"]) == 6
assert "limit reached" in result["messages"][-1].content.lower()

print("PASS: Five tool executions allowed; sixth request rejected.")

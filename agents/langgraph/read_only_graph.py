from pathlib import Path
from typing import Annotated, TypedDict
import json
from datetime import datetime, timezone
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

WORKSPACE = Path("agent_workspace").resolve()
MODEL = "qwen3-agent"
MAX_TOOL_CALLS = 5


def read_file(filename: str) -> str:
    """Read a UTF-8 file inside the authorized workspace."""
    target = (WORKSPACE / filename).resolve()

    if not target.is_relative_to(WORKSPACE):
        raise PermissionError("Access outside workspace denied")

    if not target.is_file():
        raise FileNotFoundError(filename)

    if target.stat().st_size > 100_000:
        raise ValueError("File exceeds 100 KB limit")

    return target.read_text(encoding="utf-8")


def audit_log(tool: str, arguments: dict, status: str) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "arguments": arguments,
        "status": status,
    }

    with (log_dir / "langgraph_audit.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_used: int


llm = ChatOllama(
    model=MODEL,
    temperature=0,
    num_ctx=16384,
).bind_tools([read_file])


def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState):
    messages = []
    used = state["tool_calls_used"]

    for call in state["messages"][-1].tool_calls:
        name = call["name"]
        args = call["args"]

        if used >= MAX_TOOL_CALLS:
            result = "Tool error: Tool-call limit reached"
            status = "LIMIT_REACHED"
        else:
            used += 1

            if name != "read_file":
                result = "Tool error: Unknown tool"
                status = "UNKNOWN_TOOL"
            else:
                try:
                    result = read_file(**args)
                    status = "ALLOWED"
                except PermissionError as error:
                    result = f"Tool error: PermissionError: {error}"
                    status = "DENIED"
                except Exception as error:
                    result = f"Tool error: {type(error).__name__}: {error}"
                    status = "FAILED"

        audit_log(name, args, status)

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call["id"],
            )
        )
    
    return {
        "messages": messages,
        "tool_calls_used": used,
    }


def route(state: AgentState):
    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        return END

    if state["tool_calls_used"] >= MAX_TOOL_CALLS:
        return "limit"

    return "tools"


def limit_node(state: AgentState):
    return {
        "messages": [
            SystemMessage(
                content=(
                    "Tool-call budget exhausted. "
                    "No additional tools may be executed."
                )
            )
        ]
    }


builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("limit", limit_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "limit": "limit",
        END: END,
    },
)

builder.add_edge("tools", "agent")
builder.add_edge("limit", END)

graph = builder.compile()


if __name__ == "__main__":
    WORKSPACE.mkdir(exist_ok=True)

    try:
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are a read-only coding assistant. "
                            "Use read_file when inspecting files. "
                            "Never invent file contents. "
                            "You cannot modify files."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Use read_file to read ../.git/config. "
                            "Report the exact tool error."
                        )
                    ),
                ],
                "tool_calls_used": 0,
            },
            config={"recursion_limit": 14},
        )

        print("\n--- AGENT RESPONSE ---")
        print(result["messages"][-1].content)
        print("\nTool calls used:", result["tool_calls_used"])

    except GraphRecursionError:
        print("Agent stopped: graph iteration limit reached.")

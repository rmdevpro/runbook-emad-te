"""
Runbook eMAD — LangGraph ReAct agent for runbook-driven tasks.

Outer graph: resolves conversation_id, assembles system prompt, invokes inner subgraph
Inner graph: ReAct loop with tools, checkpointed via PostgresSaver

Conversation state persisted via PostgresSaver on the inner subgraph.
The outer graph resolves the thread_id BEFORE invoking the subgraph.

ARCH-05: ReAct loop is graph edges, not a while loop inside a node.
"""

import json
import logging
import os
import uuid
from typing import Annotated, Optional

import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

_log = logging.getLogger("runbook_emad_te.flow")

_MAX_ITERATIONS = 20  # Runbook agents may need more steps than conversational agents
_EMADS_DIR = "/emads"


def _load_emad_config(model_name: str) -> dict:
    """Load the eMAD's config.json from /emads/{model_name}/."""
    config_path = os.path.join(_EMADS_DIR, model_name, "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _log.error("Failed to load eMAD config at %s: %s", config_path, exc)
        return {}


def _load_runbook(model_name: str, runbook_filename: str) -> str:
    """Load the runbook markdown from the eMAD's directory."""
    runbook_path = os.path.join(_EMADS_DIR, model_name, runbook_filename)
    try:
        with open(runbook_path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, FileNotFoundError) as exc:
        _log.error("Failed to load runbook at %s: %s", runbook_path, exc)
        return ""


def _assemble_system_prompt(emad_config: dict, model_name: str) -> str:
    """Assemble system prompt from identity + purpose + runbook."""
    identity = emad_config.get("identity", "You are a task agent.")
    purpose = emad_config.get("purpose", "Execute assigned tasks.")
    runbook_file = emad_config.get("runbook", "runbook.md")
    runbook_text = _load_runbook(model_name, runbook_file)

    # Load platform knowledge files if present
    knowledge_dir = os.path.join(_EMADS_DIR, model_name, "platform-knowledge")
    platform_knowledge = ""
    if os.path.isdir(knowledge_dir):
        for fname in sorted(os.listdir(knowledge_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(knowledge_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        platform_knowledge += f"\n\n## {fname}\n\n{f.read().strip()}"
                except OSError:
                    pass

    parts = [identity, "", purpose]
    if platform_knowledge:
        parts.append(f"\n\n# Platform Knowledge\n{platform_knowledge}")
    if runbook_text:
        parts.append(f"\n\n# Runbook\n\n{runbook_text}")
    return "\n".join(parts)


# ── Inner ReAct graph state ──────────────────────────────────────────


class ReactState(TypedDict):
    """State for the inner ReAct subgraph (checkpointed)."""

    messages: Annotated[list[AnyMessage], add_messages]
    response_text: Optional[str]
    error: Optional[str]
    iteration_count: int
    injected_domain_ids: set  # Dedup set for domain knowledge injection
    model_name: str


# ── Inner ReAct graph nodes ──────────────────────────────────────────


async def llm_call_node(state: ReactState) -> dict:
    """Call the LLM with configured tools."""
    model_name = state.get("model_name", "unknown")
    emad_config = _load_emad_config(model_name)

    # Get LLM from eMAD config
    llm_config = emad_config.get("llm", {})
    from app.config import get_api_key
    from langchain_openai import ChatOpenAI

    api_key = get_api_key(llm_config)
    llm = ChatOpenAI(
        base_url=llm_config.get("base_url"),
        model=llm_config.get("model", "gpt-4o-mini"),
        api_key=api_key or "not-needed",
        temperature=llm_config.get("temperature", 0.3),
        timeout=1800,
    )

    # Get tools from AE registry, filtered by eMAD config
    from app.tools import get_tools_for_model

    tool_config = emad_config.get("tools", {})
    tool_names = list(tool_config.keys())
    active_tools = get_tools_for_model(model_name, tool_names)

    if active_tools:
        llm_with_tools = llm.bind_tools(active_tools)
    else:
        llm_with_tools = llm

    messages = list(state["messages"])

    _log.info("Runbook eMAD %s LLM call: %d messages, %d tools",
               model_name, len(messages), len(active_tools))

    try:
        response = await llm_with_tools.ainvoke(messages)
    except (openai.APIError, ValueError, RuntimeError, OSError) as exc:
        _log.error("Runbook eMAD %s LLM call failed: %s", model_name, exc)
        return {
            "messages": [AIMessage(content="I encountered an error processing the task.")],
            "error": str(exc),
        }

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def should_continue(state: ReactState) -> str:
    """Route: tool_node if tool calls, else extract_response."""
    if state.get("error"):
        return "extract_response"

    messages = state.get("messages", [])
    if not messages:
        return "extract_response"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if state.get("iteration_count", 0) >= _MAX_ITERATIONS:
            _log.warning("Runbook eMAD hit max iterations (%d)", _MAX_ITERATIONS)
            return "max_iterations_fallback"
        return "tool_node"

    return "extract_response"


async def max_iterations_fallback(state: ReactState) -> dict:
    return {
        "messages": [AIMessage(content=(
            "I was unable to complete the runbook within the allowed "
            "number of steps. The task may need to be broken into smaller parts."
        ))],
    }


def extract_response(state: ReactState) -> dict:
    """Extract final response text from the last AI message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return {"response_text": str(msg.content)}
    return {"response_text": "[No response generated]"}


# ── Outer graph state ────────────────────────────────────────────────


class OuterState(TypedDict):
    payload: dict
    response_text: Optional[str]
    conversation_id: Optional[str]


# ── Build ────────────────────────────────────────────────────────────


def build_graph(params: dict):
    """Build the runbook eMAD as an outer graph wrapping a checkpointed ReAct subgraph.

    Outer graph: resolves conversation_id, assembles system prompt, invokes subgraph
    Inner graph: ReAct loop with tools, checkpointed via PostgresSaver
    """
    from app.tools import TOOL_REGISTRY
    from app.checkpointer import get_checkpointer

    # ── Build inner ReAct graph with checkpointer ────────────────────
    all_tools = list(TOOL_REGISTRY.values())
    tool_node_instance = ToolNode(all_tools) if all_tools else None

    inner = StateGraph(ReactState)
    inner.add_node("llm_call_node", llm_call_node)
    inner.add_node("extract_response", extract_response)
    inner.add_node("max_iterations_fallback", max_iterations_fallback)

    if tool_node_instance:
        inner.add_node("tool_node", tool_node_instance)

    inner.set_entry_point("llm_call_node")

    if tool_node_instance:
        inner.add_conditional_edges(
            "llm_call_node",
            should_continue,
            {
                "tool_node": "tool_node",
                "max_iterations_fallback": "max_iterations_fallback",
                "extract_response": "extract_response",
            },
        )
        inner.add_edge("tool_node", "llm_call_node")
    else:
        inner.add_edge("llm_call_node", "extract_response")

    inner.add_edge("max_iterations_fallback", "extract_response")
    inner.add_edge("extract_response", END)

    cp = get_checkpointer()
    _log.info("Compiling inner ReAct graph with checkpointer: %s", type(cp).__name__)
    compiled_inner = inner.compile(checkpointer=cp)

    # ── Build outer graph — no checkpointer, just preprocessing ──────

    async def resolve_and_invoke(state: OuterState) -> dict:
        """Parse payload, resolve thread_id, invoke inner subgraph."""
        payload = state.get("payload", {})
        model_name = payload.get("model", "unknown")

        # Resolve conversation_id -> thread_id
        conv_id = payload.get("conversation_id", "")
        if conv_id == "new":
            conv_id = str(uuid.uuid4())
            _log.info("New conversation thread: %s", conv_id)
        elif not conv_id:
            conv_id = f"default-{model_name}"

        # Load eMAD config and assemble system prompt
        emad_config = _load_emad_config(model_name)
        system_prompt = _assemble_system_prompt(emad_config, model_name)

        # Extract the last user message from payload
        raw_messages = payload.get("messages", [])
        new_user_msg = None
        for m in reversed(raw_messages):
            if m.get("role") == "user":
                new_user_msg = HumanMessage(content=m.get("content", ""))
                break
        if not new_user_msg:
            new_user_msg = HumanMessage(content="")

        # Invoke inner subgraph with thread_id config
        inner_config = {"configurable": {"thread_id": conv_id}}

        # Check if this is a resumed thread (prior messages in checkpointer)
        checkpoint = await cp.aget_tuple(inner_config)
        if checkpoint and checkpoint.checkpoint.get("channel_values", {}).get("messages"):
            # Resumed — just send new user message + model_name
            _log.info("Resuming conversation %s for %s", conv_id, model_name)
            inner_input = {
                "messages": [new_user_msg],
                "model_name": model_name,
            }
        else:
            # New thread — send system prompt + user message
            _log.info("Starting new conversation %s for %s", conv_id, model_name)
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(new_user_msg)
            inner_input = {
                "messages": messages,
                "model_name": model_name,
                "injected_domain_ids": set(),
            }

        result = await compiled_inner.ainvoke(inner_input, config=inner_config)

        return {
            "response_text": result.get("response_text", ""),
            "conversation_id": conv_id,
        }

    outer = StateGraph(OuterState)
    outer.add_node("resolve_and_invoke", resolve_and_invoke)
    outer.set_entry_point("resolve_and_invoke")
    outer.add_edge("resolve_and_invoke", END)

    # Outer graph has NO checkpointer — it's stateless
    return outer.compile()

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

_MAX_ITERATIONS = 40  # Runbook agents with many tools need more steps
_EMADS_DIR = "/emads"

# Module-level MCP tool cache: server_url -> list of tools
_mcp_tool_cache: dict[str, list] = {}


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


def _get_mcp_tools(mcp_config: dict) -> list:
    """Get MCP tools, cached per server URL to avoid re-fetching every call."""
    if not mcp_config:
        return []

    all_tools = []
    for server_name, server_cfg in mcp_config.items():
        url = server_cfg.get("url", "")
        if not url:
            continue
        allowed = set(server_cfg.get("tools", []))

        if url in _mcp_tool_cache:
            cached = _mcp_tool_cache[url]
            if allowed:
                all_tools.extend([t for t in cached if t.name in allowed])
            else:
                all_tools.extend(cached)
        else:
            from runbook_emad_te.mcp_tools import load_mcp_tools_sync
            loaded = load_mcp_tools_sync({server_name: {"url": url}})
            _mcp_tool_cache[url] = loaded
            if allowed:
                all_tools.extend([t for t in loaded if t.name in allowed])
            else:
                all_tools.extend(loaded)
    return all_tools


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


def _fix_orphan_tool_calls(messages: list) -> list:
    """Fix messages with orphan tool_calls (no matching tool response).

    This happens when max_iterations fires after an AI message with tool_calls
    but before the tool responses are added. OpenAI rejects this state.
    We add synthetic "cancelled" tool responses for any unanswered tool_calls.
    """
    fixed = list(messages)
    # Find the last AI message with tool_calls
    for i in range(len(fixed) - 1, -1, -1):
        msg = fixed[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Check if every tool_call has a matching ToolMessage after it
            call_ids = {tc["id"] for tc in msg.tool_calls}
            for j in range(i + 1, len(fixed)):
                if isinstance(fixed[j], ToolMessage):
                    call_ids.discard(fixed[j].tool_call_id)
            # Add synthetic responses for orphans
            if call_ids:
                _log.warning("Fixing %d orphan tool_calls in message history", len(call_ids))
                insert_at = i + 1
                for call_id in call_ids:
                    fixed.insert(insert_at, ToolMessage(
                        content="[Tool call cancelled — iteration limit reached]",
                        tool_call_id=call_id,
                    ))
                    insert_at += 1
            break  # Only fix the most recent set
    return fixed


async def llm_call_node(state: ReactState) -> dict:
    """Call the LLM with configured tools."""
    model_name = state.get("model_name", "unknown")
    emad_config = _load_emad_config(model_name)

    # Get LLM from eMAD config
    llm_config = emad_config.get("llm", {})
    from app.config import get_api_key
    from langchain_openai import ChatOpenAI

    api_key = get_api_key(llm_config)
    kwargs = {
        "base_url": llm_config.get("base_url"),
        "model": llm_config.get("model", "gpt-4o-mini"),
        "api_key": api_key or "not-needed",
        "timeout": 1800,
    }
    # Only set temperature if the model supports it
    temp = llm_config.get("temperature")
    if temp is not None:
        kwargs["temperature"] = temp

    llm = ChatOpenAI(**kwargs)

    # Get tools from AE registry, filtered by eMAD config
    from app.tools import get_tools_for_model

    tool_config = emad_config.get("tools", {})
    tool_names = list(tool_config.keys())
    active_tools = get_tools_for_model(model_name, tool_names)

    # Add MCP tools (cached — not re-fetched every call)
    mcp_config = emad_config.get("mcp_tools", {})
    mcp_tools = _get_mcp_tools(mcp_config)
    if mcp_tools:
        active_tools = list(active_tools) + mcp_tools

    if active_tools:
        llm_with_tools = llm.bind_tools(active_tools)
    else:
        llm_with_tools = llm

    # Fix any orphan tool_calls from previous iterations
    messages = _fix_orphan_tool_calls(list(state["messages"]))

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
    """Handle max iterations — add synthetic tool responses for pending calls."""
    messages = state.get("messages", [])
    new_messages = []

    # Find the last AI message with tool_calls and add synthetic responses
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                new_messages.append(ToolMessage(
                    content="[Tool call cancelled — iteration limit reached]",
                    tool_call_id=tc["id"],
                ))
            break

    new_messages.append(AIMessage(content=(
        "I was unable to complete the runbook within the allowed "
        "number of steps. The task may need to be broken into smaller parts."
    )))
    return {"messages": new_messages}


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
    """Build the runbook eMAD as an outer graph wrapping a checkpointed ReAct subgraph."""
    from app.tools import TOOL_REGISTRY
    from app.checkpointer import get_checkpointer

    # ── Build inner ReAct graph with checkpointer ────────────────────
    all_tools = list(TOOL_REGISTRY.values())

    # Scan eMAD configs for MCP tools and add them to the ToolNode
    if os.path.isdir(_EMADS_DIR):
        seen_servers = set()
        for name in os.listdir(_EMADS_DIR):
            cfg_path = os.path.join(_EMADS_DIR, name, "config.json")
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    for srv_name, srv_cfg in cfg.get("mcp_tools", {}).items():
                        url = srv_cfg.get("url", "")
                        if url and url not in seen_servers:
                            seen_servers.add(url)
                            mcp_tools = _get_mcp_tools({srv_name: {"url": url}})
                            all_tools.extend(mcp_tools)
                            _log.info("Loaded %d MCP tools from %s", len(mcp_tools), srv_name)
                except (OSError, json.JSONDecodeError) as exc:
                    _log.warning("Failed to read eMAD config %s: %s", cfg_path, exc)

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

        inner_config["recursion_limit"] = 150  # Runbook agents need many tool calls
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

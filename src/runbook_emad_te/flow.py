"""
Runbook eMAD — LangGraph ReAct agent for runbook-driven tasks.

Outer graph: resolves conversation_id, assembles system prompt, invokes inner subgraph
Inner graph: ReAct loop with tools, checkpointed via PostgresSaver

TE packages receive a TEContext from the AE — no direct app.* imports.
ARCH-05: ReAct loop is graph edges, not a while loop inside a node.
"""

import json
import logging
import os
import time
import uuid
from typing import Annotated, Any, Optional

import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

_log = logging.getLogger("runbook_emad_te.flow")

_MAX_ITERATIONS = 40
_EMADS_DIR = "/emads"

# Module-level MCP tool cache: server_url -> list of tools
_mcp_tool_cache: dict[str, list] = {}


def _load_emad_config(model_name: str) -> dict:
    config_path = os.path.join(_EMADS_DIR, model_name, "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _log.error("Failed to load eMAD config at %s: %s", config_path, exc)
        return {}


def _load_runbook(model_name: str, runbook_filename: str) -> str:
    runbook_path = os.path.join(_EMADS_DIR, model_name, runbook_filename)
    try:
        with open(runbook_path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, FileNotFoundError) as exc:
        _log.error("Failed to load runbook at %s: %s", runbook_path, exc)
        return ""


def _assemble_system_prompt(emad_config: dict, model_name: str) -> str:
    identity = emad_config.get("identity", "You are a task agent.")
    purpose = emad_config.get("purpose", "Execute assigned tasks.")
    runbook_file = emad_config.get("runbook", "runbook.md")
    runbook_text = _load_runbook(model_name, runbook_file)

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


def _resolve_tools_for_model(context: Any, model_name: str, _cache: dict = {}) -> list:
    """Single source of truth for tool resolution."""
    if model_name in _cache:
        return _cache[model_name]

    emad_config = _load_emad_config(model_name)

    # Get AE tools via context (no app.* import)
    tool_config = emad_config.get("tools", {})
    tool_names = list(tool_config.keys())
    active_tools = list(context.get_tools_for_model(model_name, tool_names))

    # Add MCP tools
    mcp_config = emad_config.get("mcp_tools", {})
    mcp_tools = _get_mcp_tools(mcp_config)
    if mcp_tools:
        active_tools.extend(mcp_tools)

    _cache[model_name] = active_tools
    _log.info("Resolved %d tools for model %s", len(active_tools), model_name)
    return active_tools


# ── State types ──────────────────────────────────────────────────────


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    final_response: Optional[str]  # ERQ-002 §13.2: was response_text
    error: Optional[str]
    iteration_count: int
    injected_domain_ids: set
    model_name: str


class OuterState(TypedDict):
    payload: dict
    final_response: Optional[str]
    conversation_id: Optional[str]


# ── Build ────────────────────────────────────────────────────────────


def build_graph(context):
    """Build the runbook eMAD stategraph.

    Args:
        context: TEContext provided by the AE. Gives access to tools,
                 checkpointer, config, and inference without importing app.*.
                 May also be a plain dict for backward compatibility.
    """
    # Support both TEContext objects and plain dicts (backward compat)
    if isinstance(context, dict):
        # Legacy: try to import from app.* as fallback
        class _LegacyContext:
            def get_tools_for_model(self, model_name, tool_names):
                from app.tools import get_tools_for_model
                return get_tools_for_model(model_name, tool_names)
            def get_all_tools(self):
                from app.tools import get_tool_registry
                return get_tool_registry()
            def get_checkpointer(self):
                from app.checkpointer import get_checkpointer
                return get_checkpointer()
            def get_api_key(self, llm_config):
                import os
                env = llm_config.get("api_key_env", "")
                return os.environ.get(env) if env else None
            def get_chat_model(self, llm_config):
                from langchain_openai import ChatOpenAI
                api_key = self.get_api_key(llm_config)
                kwargs = {
                    "base_url": llm_config.get("base_url"),
                    "model": llm_config.get("model", "gpt-4o-mini"),
                    "api_key": api_key or "not-needed",
                    "timeout": llm_config.get("timeout", 1800),
                }
                temp = llm_config.get("temperature")
                if temp is not None:
                    kwargs["temperature"] = temp
                return ChatOpenAI(**kwargs)
        context = _LegacyContext()
        _log.warning("build_graph received dict — using legacy app.* imports")

    # ── Node functions (closures over context) ───────────────────

    async def llm_call_node(state: ReactState) -> dict:
        model_name = state.get("model_name", "unknown")
        emad_config = _load_emad_config(model_name)
        verbose = emad_config.get("verbose_tracing", False)

        llm_config = emad_config.get("llm", {})
        llm = context.get_chat_model(llm_config)

        active_tools = _resolve_tools_for_model(context, model_name)

        if active_tools:
            llm_with_tools = llm.bind_tools(active_tools)
        else:
            llm_with_tools = llm

        # Fix orphan tool_calls and persist fixes
        messages = list(state["messages"])
        orphan_fixes = []
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                call_ids = {tc["id"] for tc in msg.tool_calls}
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], ToolMessage):
                        call_ids.discard(messages[j].tool_call_id)
                if call_ids:
                    _log.warning("Fixing %d orphan tool_calls", len(call_ids))
                    for call_id in call_ids:
                        fix_msg = ToolMessage(
                            content="[Tool call cancelled — previous turn limit reached]",
                            tool_call_id=call_id,
                        )
                        orphan_fixes.append(fix_msg)
                        messages.insert(i + 1, fix_msg)
                break

        iteration = state.get("iteration_count", 0)
        _log.info("Runbook eMAD %s LLM call: %d messages, %d tools, iteration %d/%d",
                   model_name, len(messages), len(active_tools), iteration, _MAX_ITERATIONS)

        t0 = time.monotonic()
        try:
            response = await llm_with_tools.ainvoke(messages)
        except (openai.APIError, ValueError, RuntimeError, OSError) as exc:
            _log.error("Runbook eMAD %s LLM call failed: %s", model_name, exc)
            return {
                "messages": orphan_fixes + [AIMessage(content=f"I encountered an error: {exc}")],
                "error": str(exc),
            }
        llm_ms = int((time.monotonic() - t0) * 1000)

        if verbose:
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
            tool_names = [tc.get("name", "?") for tc in tool_calls] if tool_calls else []
            _log.info("TRACE %s iter=%d llm_ms=%d tool_calls=%s content_len=%d",
                       model_name, iteration, llm_ms,
                       tool_names or "none",
                       len(response.content) if response.content else 0)

        return {
            "messages": orphan_fixes + [response],
            "iteration_count": iteration + 1,
        }

    def should_continue(state: ReactState) -> str:
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
        messages = state.get("messages", [])
        new_messages = []
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

    async def dynamic_tool_node(state: ReactState) -> dict:
        model_name = state.get("model_name", "unknown")
        emad_config = _load_emad_config(model_name)
        verbose = emad_config.get("verbose_tracing", False)

        active_tools = _resolve_tools_for_model(context, model_name)
        tool_map = {t.name: t for t in active_tools}

        messages = state.get("messages", [])
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        result_messages = []
        for tc in last.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            call_id = tc["id"]

            t0 = time.monotonic()
            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                content = f"ERROR: Tool '{tool_name}' not available. Available: {sorted(tool_map.keys())}"
                _log.warning("Tool not found: %s", tool_name)
            else:
                try:
                    content = await tool_fn.ainvoke(tool_args)
                    if not isinstance(content, str):
                        content = str(content)
                except Exception as exc:
                    content = f"ERROR: Tool '{tool_name}' failed: {exc}. Do not retry with same arguments."
                    _log.error("Tool %s failed: %s", tool_name, exc)

            tool_ms = int((time.monotonic() - t0) * 1000)
            if verbose:
                _log.info("TRACE %s tool=%s ms=%d args=%s result_len=%d",
                           model_name, tool_name, tool_ms,
                           json.dumps(tool_args)[:200], len(content))

            result_messages.append(ToolMessage(content=content, tool_call_id=call_id))

        return {"messages": result_messages}

    def extract_response(state: ReactState) -> dict:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return {"final_response": str(msg.content)}
        return {"final_response": "[No response generated]"}

    # ── Compile inner graph ──────────────────────────────────────

    cp = context.get_checkpointer()

    inner = StateGraph(ReactState)
    inner.add_node("llm_call_node", llm_call_node)
    inner.add_node("tool_node", dynamic_tool_node)
    inner.add_node("extract_response", extract_response)
    inner.add_node("max_iterations_fallback", max_iterations_fallback)

    inner.set_entry_point("llm_call_node")
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
    inner.add_edge("max_iterations_fallback", "extract_response")
    inner.add_edge("extract_response", END)

    _log.info("Compiling inner ReAct graph with checkpointer: %s", type(cp).__name__)
    compiled_inner = inner.compile(checkpointer=cp)

    # ── Outer graph ──────────────────────────────────────────────

    async def resolve_and_invoke(state: OuterState) -> dict:
        payload = state.get("payload", {})
        model_name = payload.get("model", "unknown")

        conv_id = payload.get("conversation_id", "")
        if conv_id == "new":
            conv_id = str(uuid.uuid4())
            _log.info("New conversation thread: %s", conv_id)
        elif not conv_id:
            conv_id = f"default-{model_name}"

        emad_config = _load_emad_config(model_name)
        system_prompt = _assemble_system_prompt(emad_config, model_name)

        raw_messages = payload.get("messages", [])
        new_user_msg = None
        for m in reversed(raw_messages):
            if m.get("role") == "user":
                new_user_msg = HumanMessage(content=m.get("content", ""))
                break
        if not new_user_msg:
            new_user_msg = HumanMessage(content="")

        inner_config = {"configurable": {"thread_id": conv_id}}

        checkpoint = await cp.aget_tuple(inner_config)
        if checkpoint and checkpoint.checkpoint.get("channel_values", {}).get("messages"):
            _log.info("Resuming conversation %s for %s", conv_id, model_name)
            inner_input = {
                "messages": [new_user_msg],
                "model_name": model_name,
                "iteration_count": 0,
                "error": None,
            }
        else:
            _log.info("Starting new conversation %s for %s", conv_id, model_name)
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(new_user_msg)
            inner_input = {
                "messages": messages,
                "model_name": model_name,
                "iteration_count": 0,
                "error": None,
                "injected_domain_ids": set(),
            }

        inner_config["recursion_limit"] = 150
        result = await compiled_inner.ainvoke(inner_input, config=inner_config)

        return {
            "final_response": result.get("final_response", ""),
            "conversation_id": conv_id,
        }

    outer = StateGraph(OuterState)
    outer.add_node("resolve_and_invoke", resolve_and_invoke)
    outer.set_entry_point("resolve_and_invoke")
    outer.add_edge("resolve_and_invoke", END)

    return outer.compile()

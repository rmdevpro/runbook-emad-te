"""
Runbook eMAD — LangGraph ReAct agent for runbook-driven tasks.

Receives the full OpenAI payload. Reads its config from /emads/{model-name}/config.json.
Assembles system prompt from identity + purpose + runbook text.
Runs a ReAct loop with configurable AE tools.
Returns response_text and conversation_id.

Conversation state persisted via LangGraph checkpointer (thread_id from payload).
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


class RunbookState(TypedDict):
    """State for the runbook eMAD agent."""

    payload: dict
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: Optional[str]
    response_text: Optional[str]
    error: Optional[str]
    iteration_count: int
    injected_domain_ids: set  # Dedup set for domain knowledge injection


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


def build_graph(params: dict):
    """Build the runbook eMAD stategraph.

    This is called once when the eMAD is first dispatched. The graph
    is cached by the router. The model name comes from the payload
    at runtime, not from params.
    """

    async def init_node(state: RunbookState) -> dict:
        """Parse payload, load eMAD config, assemble system prompt."""
        payload = state.get("payload", {})
        model_name = payload.get("model", "unknown")

        # Load this eMAD's config
        emad_config = _load_emad_config(model_name)

        # Resolve conversation_id
        conv_id = payload.get("conversation_id")
        if conv_id == "new":
            conv_id = str(uuid.uuid4())
        elif not conv_id:
            conv_id = f"default-{model_name}"

        # Parse messages
        raw_messages = payload.get("messages", [])
        lc_messages = []
        for m in raw_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "unknown")))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Assemble and prepend system prompt
        has_system = any(isinstance(m, SystemMessage) for m in lc_messages)
        if not has_system:
            system_prompt = _assemble_system_prompt(emad_config, model_name)
            if system_prompt:
                lc_messages = [SystemMessage(content=system_prompt)] + lc_messages

        return {
            "messages": lc_messages,
            "conversation_id": conv_id,
            "iteration_count": 0,
            "injected_domain_ids": set(),
        }

    async def llm_call_node(state: RunbookState) -> dict:
        """Call the LLM with configured tools."""
        payload = state.get("payload", {})
        model_name = payload.get("model", "unknown")
        emad_config = _load_emad_config(model_name)

        # Get LLM from eMAD config
        llm_config = emad_config.get("llm", {})
        from app.config import get_chat_model, get_api_key

        # Build kwargs for ChatOpenAI
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

    def should_continue(state: RunbookState) -> str:
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

    async def max_iterations_fallback(state: RunbookState) -> dict:
        return {
            "messages": [AIMessage(content=(
                "I was unable to complete the runbook within the allowed "
                "number of steps. The task may need to be broken into smaller parts."
            ))],
        }

    def extract_response(state: RunbookState) -> dict:
        """Extract final response and conversation_id."""
        response_text = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = str(msg.content)
                break

        if not response_text:
            response_text = "[No response generated]"

        return {
            "response_text": response_text,
            "conversation_id": state.get("conversation_id"),
        }

    # Build the graph
    # We need to get tools at build time for the ToolNode
    # But we don't know the model name yet — use a dynamic approach
    # For now, create ToolNode with all available tools
    from app.tools import TOOL_REGISTRY

    all_tools = list(TOOL_REGISTRY.values())
    tool_node_instance = ToolNode(all_tools) if all_tools else None

    g = StateGraph(RunbookState)
    g.add_node("init_node", init_node)
    g.add_node("llm_call_node", llm_call_node)
    g.add_node("extract_response", extract_response)
    g.add_node("max_iterations_fallback", max_iterations_fallback)

    if tool_node_instance:
        g.add_node("tool_node", tool_node_instance)

    g.set_entry_point("init_node")
    g.add_edge("init_node", "llm_call_node")

    if tool_node_instance:
        g.add_conditional_edges(
            "llm_call_node",
            should_continue,
            {
                "tool_node": "tool_node",
                "max_iterations_fallback": "max_iterations_fallback",
                "extract_response": "extract_response",
            },
        )
        g.add_edge("tool_node", "llm_call_node")
    else:
        g.add_edge("llm_call_node", "extract_response")

    g.add_edge("max_iterations_fallback", "extract_response")
    g.add_edge("extract_response", END)

    from langgraph.checkpoint.memory import MemorySaver

    # TODO: Switch to PostgresSaver for production persistence
    return g.compile(checkpointer=MemorySaver())

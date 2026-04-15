"""
MCP tool wrapper — creates LangChain tools from a remote MCP server.

eMAD configs can declare mcp_tools to pull tools from external MCP servers
(e.g., Hymie for desktop automation). These tools are NOT in the AE registry —
they're TE-specific and only loaded for eMADs that need them.

Config format in config.json:
    "mcp_tools": {
        "hymie2": {
            "url": "http://192.168.1.132:9223/mcp",
            "tools": ["desktop_click", "desktop_type", "desktop_screenshot"]
        }
    }

If "tools" is empty or absent, all tools from the server are loaded.
"""

import asyncio
import json
import logging
from typing import Any, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

_log = logging.getLogger("runbook_emad_te.mcp_tools")

_ACCEPT_HEADER = "application/json, text/event-stream"

# JSON Schema type -> Python type mapping
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic(name: str, schema: dict) -> type[BaseModel]:
    """Convert a JSON Schema to a Pydantic model for LangChain tool binding."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = {}

    for prop_name, prop_def in properties.items():
        python_type = _TYPE_MAP.get(prop_def.get("type", "string"), str)
        description = prop_def.get("description", "")
        default = prop_def.get("default")

        if prop_name in required:
            fields[prop_name] = (python_type, Field(description=description))
        else:
            fields[prop_name] = (
                Optional[python_type],
                Field(default=default, description=description),
            )

    model_name = f"MCP_{name}_Args"
    return create_model(model_name, **fields)


async def _mcp_call(url: str, method: str, params: dict | None = None) -> dict:
    """Call an MCP server method and return the result."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": _ACCEPT_HEADER,
            },
        )
        resp.raise_for_status()

        # Handle SSE response format
        text = resp.text
        for line in text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])

        # Try plain JSON
        return resp.json()


async def _list_mcp_tools(url: str) -> list[dict]:
    """Get tool definitions from an MCP server."""
    await _mcp_call(url, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "runbook-emad-te", "version": "1.0"},
    })
    result = await _mcp_call(url, "tools/list")
    return result.get("result", {}).get("tools", [])


def _make_mcp_tool(server_url: str, tool_def: dict) -> StructuredTool:
    """Create a LangChain StructuredTool from an MCP tool definition."""
    name = tool_def["name"]
    description = tool_def.get("description", f"MCP tool: {name}")
    input_schema = tool_def.get("inputSchema", {"type": "object", "properties": {}})

    # Build Pydantic model from the MCP input schema so the LLM
    # knows exactly what parameters the tool accepts
    args_model = _schema_to_pydantic(name, input_schema)

    async def _call_tool(**kwargs: Any) -> str:
        # Remove None values (optional params not provided)
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            result = await _mcp_call(server_url, "tools/call", {
                "name": name,
                "arguments": clean_kwargs,
            })
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            # FIX bug #4: Structured error instead of opaque failure
            return f"ERROR: MCP tool '{name}' call failed: {exc}. Do not retry with the same arguments."

        # Check for JSON-RPC error
        if "error" in result:
            err = result["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return f"ERROR: MCP tool '{name}' returned error: {msg}. Do not retry with the same arguments."

        mcp_result = result.get("result", {})

        # Check if the tool itself reported an error
        if mcp_result.get("isError"):
            content = mcp_result.get("content", [])
            err_text = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
            return f"ERROR: Tool '{name}' failed: {err_text}. Do not retry with the same arguments."

        content = mcp_result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append(f"[image: {item.get('mimeType', 'image/png')}]")
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "OK (no output)"

    return StructuredTool.from_function(
        coroutine=_call_tool,
        name=name,
        description=description,
        args_schema=args_model,
    )


def load_mcp_tools_sync(mcp_config: dict) -> list[StructuredTool]:
    """Load MCP tools from config. Called at graph build time.

    Args:
        mcp_config: Dict of server_name -> {url, tools} from eMAD config.

    Returns:
        List of LangChain StructuredTool instances.
    """
    tools = []

    for server_name, server_cfg in mcp_config.items():
        url = server_cfg.get("url", "")
        if not url:
            _log.warning("MCP server '%s' has no URL, skipping", server_name)
            continue

        allowed = set(server_cfg.get("tools", []))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    tool_defs = pool.submit(
                        asyncio.run, _list_mcp_tools(url)
                    ).result(timeout=30)
            else:
                tool_defs = asyncio.run(_list_mcp_tools(url))
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            _log.error("Failed to list tools from MCP server '%s' at %s: %s",
                       server_name, url, exc)
            continue

        for tool_def in tool_defs:
            name = tool_def["name"]
            if allowed and name not in allowed:
                continue
            try:
                tool = _make_mcp_tool(url, tool_def)
                tools.append(tool)
                _log.info("Loaded MCP tool: %s from %s", name, server_name)
            except (ValueError, TypeError) as exc:
                _log.warning("Failed to create tool '%s' from %s: %s",
                             name, server_name, exc)

    return tools

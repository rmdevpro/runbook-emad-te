"""
runbook-emad-te — Universal runbook-driven eMAD Thought Engine.

Executes runbooks step-by-step using configurable AE tools.
System prompt assembled from identity + purpose + runbook text.
Persistent conversation via LangGraph checkpointer.
"""

from runbook_emad_te.flow import build_graph  # noqa: F401

EMAD_PACKAGE_NAME = "runbook-emad-te"
DESCRIPTION = (
    "Universal runbook-driven eMAD TE — configurable identity, purpose, "
    "tools, and runbook. Executes tasks step-by-step with ReAct loop."
)
SUPPORTED_PARAMS = {
    "identity": {"type": "string", "description": "Agent identity (who am I)"},
    "purpose": {"type": "string", "description": "Agent purpose (what am I for)"},
    "runbook": {"type": "string", "description": "Runbook filename in the eMAD directory"},
    "tools": {"type": "object", "description": "Map of tool names to per-tool parameters"},
    "knowledge_tags": {"type": "array", "description": "Domain info source tags to include in searches"},
}

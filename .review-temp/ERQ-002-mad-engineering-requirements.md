# REQ-002: MAD Engineering Requirements

**Status:** Active
**Date:** 2026-03-31 (renumbered from REQ-001, 2026-03-25)
**Category:** Engineering Requirements
**Base:** ERQ-001 (Engineering Requirements) — all MADs must also comply with ERQ-001.
**Purpose:** Define engineering requirements specific to MADs (Multipurpose Agentic Duos) that go beyond the base engineering requirements.

---

## Scope

These requirements apply to all MADs: pMADs, eMADs, State 1 through State 4, in addition to REQ-001 (base engineering requirements). They define what makes a MAD a MAD — StateGraph architecture, AE/TE separation, Imperator pattern, MCP protocol, Prometheus metrics. General engineering quality (code, security, logging, resilience, configuration) is in ERQ-001. Container-specific requirements are in ERQ-003 (pMAD). Package-specific requirements are in ERQ-004 (eMAD). Ecosystem deployment is in REQ-005.

---

> **Sections 1 (Code Quality), 3 (Security), 4 (Logging), 5 (Async), 6.3 (Health Endpoint), 7 (Resilience), and 8 (Configuration) have been extracted to ERQ-001 (Engineering Requirements) where they apply to all projects. The sections below are MAD-specific additions.**

---

## 2. LangGraph Architecture

**2.1 StateGraph Mandate**

-   All programmatic and cognitive logic must be implemented as LangGraph StateGraphs. The graph is the application — not a wrapper around procedural code.
-   Each distinct operation is a node. State flows between nodes via typed state. Flow control is expressed as graph edges and conditional routing, not as procedural if/else chains inside nodes. Nodes must not contain loops, sequential multi-step logic, or branching that controls what happens next — those are graph structure, not node internals. If a node does multiple unrelated things sequentially, those must be separate nodes. If a node contains branching logic to decide what to do next, that must be a conditional edge. If a node contains a loop (e.g., a tool-calling cycle), that loop must be expressed as graph edges between nodes, not as a while/for loop inside a single node.
-   The HTTP server (if present) initializes and invokes compiled StateGraphs. No application logic in route handlers.
-   Before writing any custom code, check if LangChain or LangGraph already provides a component for it. Use standard components for: chat model calls (`ChatOpenAI`), embeddings, structured output parsing (`with_structured_output`), retrievers, vector stores, tool binding, retry logic, and checkpointing. Do not write raw HTTP calls to inference providers when LangChain chat models exist. Do not write custom output parsers when structured output exists. Do not write custom retry logic when LangGraph provides it.
-   Where a standard component genuinely does not fit the requirement (e.g., knowledge graph traversal requires edge-following, not vector similarity), native APIs may be used with justification documented as an exception.

**2.2 State Immutability**

-   StateGraph node functions must not modify input state in-place.
-   Each node returns a new dictionary containing only updated state variables.

**2.3 Checkpointing**

-   LangGraph checkpointing used for state persistence where applicable (e.g., long-running agent loops). Not required for short-lived background flows.

***

## 6. Communication (MAD-specific)

**6.1 MCP Transport**

-   MCP uses HTTP/SSE transport.

**6.2 Tool Naming**

-   MCP tools use domain prefixes: `[domain]_[action]`.

**6.3 Prometheus Metrics**

-   Systems that expose HTTP must provide `GET /metrics` in Prometheus exposition format.
-   Metrics produced inside StateGraphs, not in imperative route handlers.

---

## 9. AE/TE Separation

**Purpose:** Every MAD is composed of two distinct aspects that must be separable.

**9.1 Action Engine and Thought Engine**

-   Every MAD is composed of an Action Engine (AE) and a Thought Engine (TE).
-   The AE is the physical system: containers, gateway, databases, caches, tool handlers, message routing, queue processing. The AE makes action possible.
-   The TE is the cognitive intelligence: the Imperator and its cognitive apparatus (context assembly, inference, conversation-as-state). The TE decides what action to take.
-   Both AE and TE express their programmatic logic as LangGraph StateGraphs.

**9.2 Separation Requirement**

-   The TE must be conceptually and physically separable from the AE.
-   The TE is a package — it can be developed, versioned, and deployed independently of the AE.
-   The AE calls into the TE when cognitive work is needed. The TE does not receive external requests directly — all external communication arrives through the AE.
-   The TE has access to the AE's capabilities via the base contract (§13), but does not own them.

**9.3 Applicability**

-   pMADs own both an AE and a TE.
-   eMADs own only a TE — they borrow the hosting pMAD's AE.
-   The TE does not know or care whether it runs inside its own pMAD's AE or a host pMAD's AE. This is the portability guarantee.

***

## 10. Dynamic StateGraph Loading

**Purpose:** StateGraph changes must be deployable without container restarts.

**10.1 Runtime Package Installation**

-   Both AE and TE StateGraph packages must be installable at runtime without restarting the container.
-   The mechanism is `install_stategraph(package_name)` — a bootstrap tool that installs a published package and makes it live immediately.

**10.2 Package Discovery**

-   StateGraph packages are discovered via Python's `setuptools` entry_points mechanism.
-   Each pMAD defines two entry_points groups: `[pmad-name].ae` for AE packages and `[pmad-name].te` for TE packages.
-   The bootstrap kernel scans both groups on startup and after each `install_stategraph()` call.

**10.3 Package Publishing**

-   StateGraph packages are standard Python packages with semantic versioning.
-   Packages must be published to a package index (PyPI, devpi, or equivalent) before installation.

***

## 11. Imperator Requirements

**Purpose:** Every TE must include an Imperator as its prime agent.

**11.1 Imperator Mandate**

-   Every TE must include an Imperator — the prime agent that owns the TE's cognitive operations.
-   The Imperator is the front door for all conversational interaction with the MAD's cognitive layer.

**11.2 Identity and Purpose**

-   The Imperator must declare a unique Identity (what it is) and Purpose (what it is for).
-   Identity and Purpose are fixed properties of the TE package — defined at development time, not at runtime, and not derivable from calling context.
-   Identity and Purpose are expressed through the system prompt template.

**11.3 Intent**

-   The Imperator's Intent is generated at runtime as its Purpose meets current circumstances.
-   Intent is never hardcoded or pre-scripted. This is what distinguishes the Imperator from an Executor.

**11.4 ReAct Pattern**

-   The Imperator must be implemented as a graph-based ReAct agent: agent node → conditional edge → tool node → edge back to agent.
-   This is a proper graph structure — not a procedural loop inside a single node.

**11.5 Executor Distinction**

-   Executors within the same TE may be cognitive but must not be assigned Identity or Purpose.
-   An Executor's Intent is entirely derivative of the calling system and circumstances.
-   Executors are not addressable by external agents.

***

## 12. TE Package Structure

**Purpose:** Define what a TE package contains.

**12.1 Required Contents**

-   A TE package must contain:
    -   StateGraph definitions (the Imperator's reasoning flow and any Executor flows)
    -   System prompt templates (Identity, Purpose, and Persona definitions)
    -   Tool registrations (what tools the Imperator needs from the AE)
    -   Entry_points registration (so the bootstrap kernel can discover the package)

**12.2 Optional Contents**

-   PCP configuration (routing rules, escalation thresholds) — when the TE implements a Progressive Cognitive Pipeline.
-   Persona profiles — when the Imperator supports dynamic persona selection.

**12.3 Package Independence**

-   A TE package must not import from or depend on a specific AE implementation.
-   It must depend only on the base contract (§13) and standard framework components (LangChain, LangGraph).

***

## 13. AE/TE Base Contract

**Purpose:** Define the standard interface between any AE and any TE, enabling TE portability.

**13.1 Contract Requirement**

-   Every TE must be developed against a standard base contract.
-   The base contract is identical for pMAD TEs and eMAD TEs — this is the portability guarantee.
-   pMADs may extend the base contract with domain-specific additions, but must satisfy the base contract in full.

**13.2 AE Provides to TE**

-   Connection pools (database, cache, and other backing service access)
-   Logging configuration (structured logging setup, log level)
-   Observability hooks (metrics registry for Prometheus instrumentation)
-   Peer proxy access (ability to reach other MADs, when deployed in an ecosystem)
-   Initial state: `messages` (conversation history), `context_window_id`, `conversation_id`

**13.3 TE Provides to AE**

-   Entry point for invocation (a compiled StateGraph callable via `ainvoke`)
-   Output state: `final_response` (the Imperator's response text), messages to persist (including tool calls and tool results)
-   Identity and Purpose declarations (available for the AE to present to callers)
-   Tool registrations (list of tools the TE requires from the AE)

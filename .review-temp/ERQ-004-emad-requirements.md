# REQ-003: State 4 eMAD Engineering Requirements

**Status:** Active
**Date:** 2026-03-25 (promoted from draft 2026-03-21)
**Category:** Engineering Requirements
**Purpose:** Define engineering requirements specific to eMADs (ephemeral MADs) — intelligence-only packages hosted by a pMAD. These are additive to ERQ-002 (MAD Engineering Requirements).

---

## Scope

These requirements apply to eMADs — MADs that have no containers of their own. An eMAD is a Python package containing StateGraph flows, published to a package index (Alexandria or PyPI), and loaded at runtime by a host pMAD (typically Kaiser). The eMAD runs inside the host's process space.

All ERQ-001 (base) and ERQ-002 (MAD requirements) also apply. This document adds package-specific requirements only.

---

## 1. Package Structure

**1.1 Python Package**
- An eMAD is a standard Python package with a `pyproject.toml`.
- Built as a wheel and published to a package index.
- Installable via `pip install` without additional build steps.

**1.2 Entry Points**
- The eMAD registers itself via the host pMAD's entry_points group.
- Entry point form: `{name} = "{module}.register:build_graph"`
- The `build_graph(params: dict) -> StateGraph` function returns a compiled StateGraph.

**1.3 Package Data**
- Non-Python files needed at runtime (prompt templates, JSON defaults, etc.) must be declared in `pyproject.toml` under `[tool.setuptools.package-data]`.
- Files are accessed via `Path(__file__).resolve().parent` relative paths, not hardcoded absolute paths.

---

## 2. Host Contract

**2.1 Input State**
- The host pMAD provides initial state to the eMAD's StateGraph. The minimum contract is defined by the host (e.g., Kaiser provides: `messages`, `rogers_conversation_id`, `rogers_context_window_id`).
- The eMAD must document what initial state it expects.

**2.2 Output State**
- The eMAD's StateGraph must include `final_response` in its output state.
- The host uses `final_response` as the eMAD's reply.

**2.3 No Container Assumptions**
- The eMAD does not own containers, volumes, or network configuration.
- It accesses backing services through the host pMAD's infrastructure (peer proxy, connection pools).
- It does not assume specific filesystem paths except those provided by the host (/workspace, /storage).

---

## 3. Imperator

**3.1 Imperator Required**
- Every eMAD TE must include an Imperator as its prime agent.
- The Imperator declares Identity (what it is) and Purpose (what it is for).
- These are fixed properties of the package, not runtime configuration.

**3.2 Naming Convention**
- Main flow file: `flows/imperator.py`
- State class: `{Name}ImperatorState`
- Graph builder: `build_imperator_graph()`
- Registered via `register.py`

---

## 4. Peer Access

**4.1 Ecosystem Services**
- The eMAD accesses ecosystem services (Rogers, Sutherland, Starret) through the host pMAD's peer proxy.
- No direct network calls to ecosystem services.
- The peer proxy pattern is transparent — the eMAD calls a function that routes through the host.

**4.2 Inference**
- All LLM calls route through the configured inference path (Sutherland in ecosystem, configurable provider in State 4).
- The eMAD does not directly instantiate LLM clients — it uses the provided inference interface.

---

## 5. Statelessness

**5.1 No Internal State**
- The eMAD stores nothing internally between invocations.
- All persistent state is owned by ecosystem services (Rogers for conversation, the host pMAD for operational state).
- Quorum recordings and other artifacts go to shared storage, not to the eMAD's package directory.

---

## 6. Versioning

**6.1 Semantic Versioning**
- eMAD packages follow semantic versioning.
- Version bumped on every publish.

**6.2 Hot Install**
- The host pMAD's `install_stategraph()` installs and activates the new version without container restart.
- The eMAD must be safe to hot-swap — no global state, no import-time side effects that persist across versions.

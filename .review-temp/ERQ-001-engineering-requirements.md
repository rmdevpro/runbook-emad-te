# REQ-001: Engineering Requirements

**Status:** Active
**Date:** 2026-03-31
**Category:** Engineering Requirements
**Derived from:** Extracted from former REQ-001 (State 4 MAD Engineering Requirements), now REQ-002.
**Purpose:** Define engineering requirements that apply to ALL software projects — standalone applications, MADs, and any other system built in this ecosystem.

---

## Scope

These requirements apply universally to all projects. They define baseline engineering quality, security, observability, resilience, and configuration standards. MAD-specific requirements (StateGraph, AE/TE, Imperator, MCP protocol, Prometheus) are in REQ-002. pMAD container requirements are in REQ-003. eMAD package requirements are in REQ-004. Ecosystem deployment requirements are in REQ-005.

---

## 1. Code Quality

**1.1 Code Clarity**

- All code must be clear, readable, and maintainable.
- Descriptive names for variables, functions, and classes.
- Small, focused functions — each does one thing well.
- Comments explain why, not what.

**1.2 Code Formatting**

- All code must be formatted with the standard formatter for its language (Python: `black`, JavaScript: project-defined).
- Verification: formatter check passes without errors.

**1.3 Code Linting**

- All code must pass the standard linter for its language (Python: `ruff`, JavaScript: `eslint` or equivalent) without errors.

**1.4 Unit Testing**

- All programmatic logic must have corresponding tests covering the primary success path and common error conditions.

**1.5 Version Pinning**

- All dependencies locked to exact versions.
- Python: `==` in requirements.txt. Node: exact versions in package.json.
- The most recent stable version of all items should be used unless there is a very solid overriding concern.

---

## 2. Security Posture

**2.1 No Hardcoded Secrets**

- No credentials in code, Dockerfiles, compose files, or committed files.
- Credentials loaded from environment variables or credential files at runtime.
- Repository ships example/template credential files. Real credentials are gitignored.

**2.2 Input Validation**

- All data from external sources (user inputs, API responses, tool inputs) must be validated before use.

**2.3 Null/None Checking**

- Variables that could be None/null/undefined must be explicitly checked before attribute access.

---

## 3. Logging and Observability

**3.1 Logging to stdout/stderr**

- All logs go to stdout (normal) or stderr (errors). No log files inside containers or processes.

**3.2 Structured Logging**

- JSON format, one object per line.
- Fields: timestamp (ISO 8601), level (DEBUG/INFO/WARN/ERROR), message, context fields.

**3.3 Log Levels**

- DEBUG, INFO, WARN, ERROR. Default: INFO. Configurable.

**3.4 Log Content**

- Do log: lifecycle events, errors with context, performance metrics.
- Do not log: secrets, full request/response bodies, health check successes (only state changes).

**3.5 Specific Exception Handling**

- No blanket catch-all exception handlers. Catch specific, anticipated exceptions.

**3.6 Resource Management**

- All external resources (file handles, database connections) must be reliably closed using context managers, try/finally, or equivalent.

**3.7 Error Context**

- All logged errors and raised exceptions must include sufficient context to debug.

**3.8 Pipeline Observability**

- Processing pipelines (multi-stage flows) must support a verbose logging mode that reports what is happening at each stage, including intermediate outputs and performance measurements.
- This mode must be togglable via configuration — not always on, not requiring code changes to enable.

---

## 4. Async Correctness

**4.1 No Blocking I/O**

- No blocking I/O in async functions. Use async libraries appropriate to the language/runtime.
- Synchronous blocking calls in async context are forbidden.

---

## 5. Communication

**5.1 Health Endpoint**

- Systems that expose HTTP must provide `GET /health` returning 200 when healthy, 503 when unhealthy, with per-dependency status.

---

## 6. Resilience

**6.1 Graceful Degradation**

- Failure of optional components causes degraded operation, not a crash.
- Core operations continue with reduced capability.
- Health endpoint reports degraded status.

**6.2 Independent Startup**

- Components start and bind ports without waiting for dependencies.
- Dependency unavailability handled at request time.

**6.3 Idempotency**

- Operations that may be retried must be safe to execute more than once with the same input without causing unintended side effects.

**6.4 Fail Fast**

- If the system detects invalid configuration, missing required dependencies, or corrupt state, it must fail immediately with a clear error rather than proceeding and producing wrong results silently.

---

## 7. Configuration

**7.1 Configurable External Dependencies**

- Inference providers, model selection, and similar external dependencies must be configurable via config file or environment variables.

**7.2 Externalized Configuration**

- Configuration, parameters, and content that may change between deployments or over time must be externalized from application code.
- This includes: prompt templates, model parameters, retry counts, timeouts, thresholds, file paths, URL patterns, and any value that someone might reasonably want to change without changing code.

**7.3 Hot-Reload vs Startup Config**

- Runtime-changeable settings (models, tuning parameters): read per operation, no restart needed.
- Infrastructure settings (database connections, ports): read at startup, restart required.

## 8. Deployment

**8.1 Compose Self-Sufficiency**

- `docker compose up -d --build` must produce a fully working deployment with no wrapper scripts, manual steps, or external tooling.
- All init logic, dependency ordering, healthchecks, and permissions must be handled by the Dockerfile, entrypoint, and compose file.
- Do not use wrapper scripts (`deploy.sh`, `start.sh`) to work around gaps in the compose configuration. If compose alone doesn't work, fix the compose file and Dockerfile.

**8.2 Environment Separation**

- Environment-specific values (ports, volume paths, container names, env vars) must be the only difference between deployment targets.
- Application code, Dockerfile, and entrypoint must be identical across environments.

---

## Related Documents

- [REQ-002 MAD Engineering Requirements](ERQ-002-mad-engineering-requirements.md) — MAD-specific additions (StateGraph, AE/TE, Imperator, MCP, Prometheus)
- [REQ-003 pMAD Requirements](ERQ-003-pmad-requirements.md) — pMAD container requirements
- [REQ-004 eMAD Requirements](ERQ-004-emad-requirements.md) — eMAD package requirements
- [REQ-005 Ecosystem Requirements](ERQ-005-ecosystem-requirements.md) — Joshua26 ecosystem deployment

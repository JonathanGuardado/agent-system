# CLAUDE.md — Agent System v1

## Project identity

This repo is an autonomous software development agent system.

The system receives human requests from Slack, turns approved work into Jira
tickets, detects `ai-ready` tickets, locks them, executes them through a
LangGraph pipeline, opens a PR, and reports back to Slack.

## Architecture in one line

Slack → ai-model-selector → IntakeHandler → Jira source of truth →
DetectionComponent → SQLite lock → LangGraph StateGraph →
ImplementationComponent → internal ModelRouter → tests → PR → Slack alert

## Current implementation focus

We are building the system incrementally.

Current phase:

- P0: ✅ Infrastructure
- P1: ✅ Intake handler foundation
- P2: ✅ Detection + Locking foundation
- P3: ✅ Tool adapters
- P4: ✅ Internal ModelRouter + provider clients (smoke validated)
- P5: 🚧 LangGraph pipeline — skeleton + services built, wiring pending

Current active task:

Wire the remaining full end-to-end pipeline pieces: the real coding-agent loop
and the process entrypoint that boots Slack intake + Detection +
ExecutionWorker together.

P5 work already done:
- LangGraph StateGraph (graph.py) with full node sequence and routing
- TicketState, service Protocols, TicketNodeRunner (DI surface)
- LocalImplementationService (worktree prep), AdapterTestService
- ModelRouterPlannerService, ModelRouterImplementationService, ModelRouterReviewService
- GitService (commit/push) + GhPullRequestOpener
- JiraEscalationService, JiraLabelApprovalService, AutoApprovalService
- OrchestratorRunner (lock+heartbeat+claim+graph+release)
- ExecutionWorker + JiraExecutionCoordinator
- config/repos/agent-system.yaml repo contract
- SQLiteCheckpointer wired into persistent graph compilation
- OrchestratorRunner stale-checkpoint guard after fresh lock acquisition
- Slack-driven execution-approval interrupt + SQLite approval persistence
- 352 unit tests passing

P5 remaining gaps (in priority order):
1. Coding-agent loop — LocalImplementationService defaults to _prepare_only_step (no writes); ModelRouterImplementationService is single-shot, no tool-use iteration
2. Process entrypoint — no app.py that boots Slack listener + Detection + ExecutionWorker together

Do not claim Slack, Jira, the full orchestrator, or the coding-agent loop are
done until the coding-agent loop is implemented.

## Runtime model

The LLM routing layer is implemented inside this repo as an internal Python
module.

Router package:

```txt
src/ticket_agent/router/
├── __init__.py
├── factory.py
├── model_router.py
├── selector_config.py
└── providers/
    ├── __init__.py
    ├── base.py
    ├── config.py
    ├── deepseek.py
    ├── gemini.py
    ├── http.py
    ├── ollama.py
    └── stubs.py
```

Components call the internal router directly:

```python
await model_router.invoke(
    capability="code.implement",
    messages=messages,
    ticket_id=ticket_key,
)
```

The internal ModelRouter owns:

- ai-model-selector integration
- endpoint execution for selector-selected providers
- fallback chain execution
- provider lookup
- response normalization
- attempt tracking and timeout handling

The router is internal Python code. It does not expose an HTTP server or an
OpenAI-compatible API.

## Environment assumptions

These are available on the HP:

- Ollama + Qwen 3.5 9B
  - Ollama runs at `localhost:11434`
  - Ollama/Qwen is optional local/simple fallback only.

- API keys
  - Stored in `~/config/agent-system.env`
  - Includes DeepSeek and Gemini keys.
  - Load them from environment/config.
  - Do not duplicate keys into this repo.
  - Do not print secrets in logs or test output.

- Python environment
  - The agent system may have its own venv.
  - Keep runtime dependencies scoped to this repo.

## Package layout

Main package:

```txt
src/ticket_agent/
```

Tests:

```txt
tests/unit/
```

Expected high-level areas:

```txt
src/ticket_agent/
├── detection/
├── intake/
├── locks/
├── models/
├── orchestrator/
├── router/
├── tools/
└── execution/
```

Configuration:

```txt
config/
├── system.yaml
├── capabilities.yaml
├── models.yaml
├── task_profiles.yaml
├── budgets.yaml
└── repos/
```

## Key architecture decisions

- `ai-model-selector` is deterministic.
  - No LLM call.
  - No network call.
  - Used for intent/capability resolution and model tier selection.

- ModelRouter is internal.
  - It lives inside `src/ticket_agent/router/`.
  - It is imported and called directly by Python components.
  - It is not an HTTP service.
  - It should not expose an OpenAI-compatible API in v1.
  - Do not add an external `router.py`.
  - Do not add FastAPI or a `localhost:8080` router service.

- Components should not call provider APIs directly.
  - They call `ModelRouter.invoke(...)`.
  - Provider-specific logic belongs under `router/providers/`.
  - The providers package does not export `httpx` directly; tests patch
    `ticket_agent.router.providers.http.httpx`.

- Components should not know provider API keys.
  - API keys are loaded by provider clients.
  - Secrets must never be passed into LLM prompts, tool adapters, logs,
    or LangGraph state.

- LangGraph is workflow runtime only.
  - It owns node sequencing, state transitions, retries, checkpoints,
    and human interrupts.
  - It does not write code.
  - It does not reason about implementation details.

- `ImplementationComponent` is the coding agent.
  - It runs inside the LangGraph `implement` node.
  - It uses internal ModelRouter for LLM calls.
  - It uses local tool adapters for file, shell, test, and git operations.

- SQLite WAL is used for both:
  - distributed ticket locks via `ticket_locks`
  - LangGraph checkpoints via `SqliteSaver`

- A LangGraph checkpoint without a valid SQLite lock is stale.
  - Never resume stale checkpoints.
  - Reconciler must clean Jira and expired locks before work resumes.

- Jira is the execution source of truth.
  - Detection reads from Jira.
  - Orchestrator works from Jira tickets.
  - Slack is only the human-facing interface.

- Slack has two separate approval moments:
  1. Intake approval: approve the proposal before Jira is written.
  2. Execution approval: approve the LangGraph execution before code is changed.

## Internal ModelRouter contract

The internal router exposes one primary async method:

```python
class ModelRouter:
    async def invoke(
        self,
        capability: str,
        messages: list[dict],
        ticket_id: str | None = None,
        metadata: dict | None = None,
    ) -> ModelResponse:
        ...
```

`ModelResponse` should include:

```python
class ModelResponse(BaseModel):
    content: str
    model: str
    provider: str
    capability: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    fallback_used: bool = False
    attempts: list[ModelAttempt] = []
```

`ModelAttempt` should include:

```python
class ModelAttempt(BaseModel):
    model: str
    provider: str
    success: bool
    error: str | None = None
    latency_ms: int | None = None
```

Provider interface:

```python
class ProviderClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        ...
```

Provider implementations for v1:

- `DeepSeekProvider`
- `GeminiProvider`
- `OllamaProvider`
- fake provider clients for tests

The router should try:

1. `decision.primary`
2. each model in `decision.fallbacks`
3. raise `AllBackendsFailedError` if all fail

## ai-model-selector usage

The internal ModelRouter should use:

```python
IntentResolver.resolve(...)
build_request_context(...)
DeterministicSelector.select(...)
```

Flow:

```python
resolution = resolver.resolve(capability)
context = build_request_context(resolution)
decision = selector.select(context)
```

Then the internal router maps `decision.primary` to a configured provider/model.

## Model policy

- DeepSeek V4 Pro: primary coding and implementation model.
- Gemini: verification, structured checks, planning/design, and future
  browsing/research-style tasks.
- Ollama/Qwen: optional local/simple fallback only.

MiniMax and GLM are intentionally not part of v1 for now.

Example:

```yaml
# config/models.yaml
models:
  deepseek-v4-pro:
    provider: deepseek
    model_id: deepseek-v4-pro
  gemini-flash:
    provider: gemini
    model_id: gemini-2.5-flash
  qwen-local:
    provider: ollama
    model_id: qwen3.5:9b
```

## Cost metadata

Cost-aware routing is out of scope for v1. Estimated cost fields may remain on
response models for future logging, but estimated cost must not affect model
choice. Do not add BudgetGuard yet.

## Non-obvious rules

- Test commands are never auto-detected.
  - Always load them from `config/repos/{repo}.yaml`.

- Tool adapters are direct Python calls in v1.
  - Do not implement MCP yet.
  - Adapter method signatures should remain MCP-compatible for future migration.

- OpenClaw is post-MVP.
  - Do not add OpenClaw integration in v1.

- Do not merge PRs automatically.
  - The system opens PRs only.
  - Human review and merge stay manual.

## Security rules enforced in code

### FileAdapter

FileAdapter must enforce all of the following before every read/write/list:

- Use `Path.resolve()` before boundary checks.
- Resolved path must stay inside the worktree.
- Symlink escapes must be blocked.
- Writes outside allowed source directories must be rejected.
- Protected files must be rejected.

Protected paths include:

```txt
.github/
Dockerfile
docker-compose.yml
.env
secrets/
```

File operation policy:

- Create/modify source files: allowed inside `source_dirs`
- Delete files: not allowed, escalate
- Rename/move files: not allowed, escalate
- CI/CD files: not allowed, escalate
- Config files: only allowed if listed in `config_paths_allowed`

### ShellAdapter

ShellAdapter must enforce:

- Command allowlist before execution.
- Denylist for dangerous commands.
- No shell interpolation when avoidable.
- Run from the worktree only.
- Strip environment variables.
- Keep only:
  - `PATH`
  - `HOME`
  - `VIRTUAL_ENV`
- Never expose API keys or secrets.
- Enforce timeout and kill process on timeout.

### TestAdapter

TestAdapter must:

- Load test command from repo contract.
- Never infer test command from package files.
- Return structured test result.
- Include stdout/stderr summary.
- Mark timeout explicitly.

### GitAdapter

GitAdapter must:

- Use isolated worktrees.
- Branch format:

```txt
agent/{TICKET-KEY}/{short-lock-id}
```

- Never push to `main`.
- Never force-push.
- Clean up worktrees after PR or escalation.
- Open PRs only. Do not merge.

## Repo contract rules

Every repo that agents work on must have:

```txt
config/repos/{repo}.yaml
```

The system must escalate if the repo contract is missing.

Required contract fields:

```yaml
repo:
  name: my-project
  root: ~/repos/my-project
  default_branch: main

language:
  primary: python
  package_manager: poetry

commands:
  test:
    command: ["python", "-m", "pytest", "tests/", "-x", "-q"]
    timeout_seconds: 120
    working_directory: "."
  lint:
    command: ["python", "-m", "ruff", "check", "src/"]
    timeout_seconds: 120
    working_directory: "."
  install: null

policy:
  dependency_install_allowed: false
  config_paths_allowed: []
  protected_paths:
    - .github/
    - Dockerfile
    - docker-compose.yml
    - .env
    - secrets/

source_dirs:
  - src/

test_dirs:
  - tests/
```

## Coding expectations

When implementing code:

- Prefer small, focused modules.
- Keep side effects isolated.
- Add unit tests for happy paths and failure modes.
- Use typed dataclasses or Pydantic models for structured results.
- Avoid broad exception swallowing.
- Use explicit custom errors for policy violations.
- Keep security-sensitive behavior easy to inspect.
- Do not introduce network calls outside provider clients.
- Do not add new dependencies unless needed.

## Testing expectations

For every new component, add tests under:

```txt
tests/unit/
```

Security-sensitive components must include negative tests.

Required adapter test coverage:

- Path escape is blocked.
- Symlink escape is blocked.
- Protected path write is blocked.
- Unauthorized config write is blocked.
- Denied shell command is blocked.
- Unknown shell command is blocked.
- Shell timeout is handled.
- Test command is loaded from repo contract.
- Missing repo contract escalates.
- Git branch name follows `agent/{TICKET-KEY}/{id}`.

Required router test coverage:

- Primary model succeeds.
- Primary fails and fallback succeeds.
- All providers fail.
- Provider API keys are not logged.
- Ollama provider sends `think: false` when calling Qwen.
- Components call ModelRouter, not provider clients directly.

## Current P3 implementation checklist

- [x] FileAdapter path boundary enforcement
- [x] FileAdapter protected path policy
- [x] ShellAdapter allowlist and denylist
- [x] ShellAdapter env isolation
- [x] ShellAdapter timeout handling
- [x] RepoContract
- [x] TestAdapter using repo contract test command
- [x] GitAdapter worktree creation
- [x] GitAdapter commit/push/cleanup
- [x] Unit tests for adapter failure modes

## Current P4 implementation checklist

- [x] Internal ModelRouter foundation
- [x] ProviderClient interface
- [x] ai-model-selector adapter/config wrapper
- [x] fallback chain
- [x] router tests
- [x] DeepSeekProvider
- [x] GeminiProvider
- [x] OllamaProvider
- [x] provider tests
- [x] no-secret logging/error behavior verified
- [x] router factory

## Important references

Architecture spec:

```txt
Agent_System_Architecture_v1.html
```

Implementation guide:

```txt
Agent_System_Implementation_Guide.html
```

## What not to do

Do not:

- Replace `ai-model-selector`
- Add an external `router.py`
- Add FastAPI or a `localhost:8080` router service
- Add MCP or OpenClaw yet
- Auto-detect test commands
- Add MiniMax or GLM to v1
- Add cost-aware routing in v1
- Push to main
- Force-push
- Merge PRs automatically
- Store secrets in this repo
- Let tools operate outside the worktree
- Let the LLM decide security policy
- Claim LangGraph or orchestration is done before it is implemented

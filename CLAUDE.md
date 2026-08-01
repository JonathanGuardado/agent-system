# CLAUDE.md — Agent System v1

## Project identity

This repo is an autonomous software development agent system.

Today it receives human requests from Slack, turns approved work into Jira
tickets, executes `ai-ready` tickets through a LangGraph pipeline, and opens a
pull request that a human reviews and merges. The goal-pursuit architecture it
is being built toward is described in
[`docs/autonomous-delivery-roadmap.md`](docs/autonomous-delivery-roadmap.md);
see "Roadmap and current status" below for the line between the two.

**Target repositories are external inputs to this runtime.** A target is
described by a `config/repos/*.yaml` contract; the application changes the
system produces inside it are the runtime's *outputs*. Nothing in `src/` may be
shaped around any particular target. `ofertas-sv` is one such target from an
earlier iteration — see the configuration-debt notes in the roadmap for what
currently names it and why that is a naming problem rather than an
architectural one.

## Architecture in one line

**Current:**

Slack → ai-model-selector → IntakeHandler → authorized GoalContract / Jira
projection → durable spine → DetectionComponent → authorization+sandbox
preflight → SQLite lock/action journal → LangGraph StateGraph →
ImplementationComponent → internal ModelRouter → worktree tests →
summary-based review → commit/push → PR → Slack alert

**Remaining target after P14–P18** (not current behavior):

Slack → authorized signed `GoalContract` → durable goal spine → Jira projection
→ implementation → candidate commit → isolated verification of that SHA →
independent complete-diff review → attestation/outbox → trusted CI →
human-reviewed promotion PR → bounded rework

## Roadmap and current status

**The canonical roadmap is
[`docs/autonomous-delivery-roadmap.md`](docs/autonomous-delivery-roadmap.md).**
It holds the detailed P6–P18 specifications, checklists, exit criteria, design
rationale, and the current-versus-target architecture. This section is a
summary; when the two differ, the roadmap document is correct.

### What the runtime does today

A bounded, human-reviewed ticket-processing pipeline — **not** unattended
autonomous delivery:

```txt
Slack request
  → model-assisted proposal
  → human intake approval
  → authorized signed GoalContract + durable goal spine
  → Jira epic/tasks as the goal projection
  → ai-ready detection
  → authorization/sandbox preflight
  → SQLite ticket lock/checkpoint + action journal
  → LangGraph planning
  → execution approval (automatic or Slack, depending on policy)
  → implementation
  → worktree tests
  → summary-based model review
  → commit/push
  → pull request
  → Jira/Slack reporting
```

Important current boundaries:

- **The authorized `GoalContract` is scope authority; Jira is its work-queue
  projection.** Detection reads Jira, but shared preflight rejects missing,
  invalid, revoked, out-of-scope, or insufficient-autonomy authority before
  execution mutation.
- **Goal identity and recovery state are durable.** `goal_id` reaches graph
  state and observability records, while the action journal applies concrete
  operation policy and budget reservation at side-effect boundaries.
- **Verification is not bound to the committed SHA.** Tests run in the
  worktree; the commit happens later, inside `open_pull_request`.
- **Review does not consume the real diff.** It reads the implementing model's
  own summary and result.
- **Production contract commands are sandboxed.** A lazy enforcing preflight
  refuses execution before locks, claims, worktrees, or resume, both test paths
  use one runtime shell factory, and every wrapper launch emits an attestation.
  P13 remains partial only because context assembly / the knowledge map is open.

**The system never merges pull requests automatically.** The runtime opens
PRs; it does not merge them. Any review or merge is performed by a human, as
required by current policy. This is a behavioral boundary enforced by the code,
not a claim that every generated PR has been reviewed and merged in practice.

### Remaining target architecture (P14–P18)

The contract, spine, journal, and Jira projection at the start of this flow are
current foundations. Do not describe the remaining verification, review, and
delivery steps as current behavior:

```txt
Slack → authorized signed GoalContract → durable goal spine / action journal
  → Jira projection → implementation → candidate commit
  → isolated verification of that SHA → independent complete-diff review
  → attestation / outbox → trusted CI / integration delivery
  → human-reviewed promotion PR → bounded review-comment rework
```

### Phase status

Legend: ✅ complete (implemented, wired, enforced, tested) · 🔶 partial
(components exist, invariant not enforced) · ⬜ not started · ⏸️ deferred by
dependency · ↪ absorbed into another phase.

| Phase | State | Reality |
|---|---|---|
| P0–P7 | ✅ | Bounded ticket-processing MVP: infrastructure, intake, detection, locking, adapters, router, implementation loop, acceptance criteria |
| P8 | ⏸️ | Bug-fix work profile. Deferred until P15/P16 provide trustworthy feedback |
| P9 | ↪ | Diff-based review + lint gate. Absorbed into P14/P15; never scheduled separately |
| P10 | 🔶 | Observability foundations exist; later producers and operational evidence missing |
| P11 | ✅ | This repo owns selector config; resolution is pinned and the library copy is example-only |
| P12 | 🔶 | Durable authority, revocation, autonomy, and the action journal are enforced; non-convergence and live recovery evidence remain |
| P13 | 🔶 | Sandbox enforcement and attestation are wired; context assembly / knowledge map remains |
| P14 | ⬜ | Immutable-SHA verification |
| P15 | ⬜ | Independent complete-diff review |
| P16 | ⬜ | Repeatable evaluation and demonstration evidence |
| P17 | ⬜ | Durable delivery, outbox, attestation, trusted CI, promotion |
| P18 | ⬜ | Bounded rework from promotion-PR change requests |

**A capability is never ✅ because its types exist.** The sandbox and durable
goal authority are enforced, but P13 still lacks context assembly and P12 still
lacks non-convergence policy plus reviewed live recovery evidence; both are 🔶.

**P0–P7 are implemented and unit-tested, not operationally validated.**
External Slack/Jira/GitHub end-to-end verification remains manual.

### Recommended execution order

Phase numbers are **stable identities, not execution order** — they are cited
by commits and tests, so they are never renumbered. Dependencies dictate:

```txt
P11 → P13.3a → P12.2a → (P12.2b + P12.2c) → P12.2d → P12.2e
    → P14 → P15 → P16 → P17 → P18 → revisit P8
```

P11 (configuration ownership) first because everything downstream reads
selector and policy config. P13.3a then makes sandbox refusal happen before
locks, claims, worktrees, or resume. P12.2b and P12.2c are one safe release
unit: durable authorization publication must ship with consumption at every
entry point. P12 precedes P14–P17 because those all record evidence against the
goal that now reaches the graph. Per-step rationale and operation-level
recovery policy are in
[the roadmap](docs/autonomous-delivery-roadmap.md#recommended-execution-order).

One phase per PR; multiple PRs per phase is fine. When a phase lands, update
the status table there and here — and only mark ✅ once it is wired into the
production path, not merely implemented.

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

- Ollama + Qwen 3.6 27B
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

- **Scope authority — `GoalContract` vs Jira.**

  **Current behavior:** the authorized `GoalContract` is scope authority and
  Jira is a projection and work queue written *from* the plan. Detection reads
  Jira and the orchestrator works from Jira tickets, but shared preflight
  revalidates current durable authority before execution mutation.
  - The contract records what the human authorized: objective, acceptance
    criteria, non-goals, scope, budgets. Agents plan freely *underneath* it
    and may never widen it.
  - Durable intent and action state are retained independently of Jira's role
    as the human-visible work queue.
  - Slack is only the human-facing interface, before and after.

  The migration is runtime-enforced and unit-tested. P12 remains partial for
  non-convergence policy and reviewed live process-kill recovery evidence.

- **Approval moments.**

  **Current behavior:** two moments. Intake approval — a human approves the
  proposal before Jira is written. Then execution approval, which is
  **automatic or Slack-driven depending on configuration**: the default
  `AutoApprovalService` (`orchestrator/local_services.py:492`, wired at
  `app.py:1295`) returns `True` unconditionally on the reasoning that the human
  already approved the whole plan at intake; `JiraLabelApprovalService` and the
  Slack-driven execution-approval interrupt are the alternatives.

  **P12-enforced behavior:** at autonomy level
  `autonomous`, **one** moment — an allowlisted user in an allowlisted channel
  authorizes ordinary in-policy work at intake, and the only remaining human
  action is reviewing the promotion PR. Re-approval would be required only for
  elevated risk, ambiguity, scope expansion, a declared human-required
  boundary, or a semantic-check disagreement. The shared execution preflight
  persists the effective mode and every concrete action boundary enforces it;
  later merge/promotion executors remain P17 work.

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
    model_id: qwen3.6:27b
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
    writable_paths: [".pytest_cache"]
    network: none
  lint:
    command: ["python", "-m", "ruff", "check", "src/"]
    timeout_seconds: 120
    working_directory: "."
    writable_paths: [".ruff_cache"]
    network: none
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
- Give the model shell or git tool actions. `run_tests` (P6) is the only
  model-callable command, and it must run the repo-contract test command
  through TestAdapter — nothing else.
- Let the model bypass FileAdapter boundaries via `search` or `edit_file`
- Implement more than one roadmap phase in a single PR
- Add MiniMax or GLM to v1
- Add cost-aware routing in v1
- Push to main
- Force-push
- Merge anything into `main`, ever, automatically. (P17 merges into
  `integration/*` only, and only on genuine CI evidence. `main` is reached
  solely through a promotion PR that a human reviews.)
- Use `gh pr merge --auto`. Once armed, GitHub merges even after a halt.
- Weaken or modify the evaluator, the verification trust root, or the
  authorization policy during the same goal run in order to make that goal
  pass
- Run unattended repository commands without a working sandbox
- Store secrets in this repo
- Let tools operate outside the worktree
- Let the LLM decide security policy
- Claim LangGraph or orchestration is done before it is implemented

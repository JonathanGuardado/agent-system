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
- P1: ✅ Slack intake + proposal approvals
- P2: ✅ Detection + SQLite locking
- P3: ✅ Local tool adapters
- P4: ✅ Internal ModelRouter + provider clients
- P5: ✅ MVP runtime wiring
- P6: ✅ Implementation loop upgrade (search, edit, paginated reads, in-loop tests)
- P7: ⬜ Acceptance criteria pipeline (intake → plan → review)
- P8: ⬜ Bug-fix work profile (reproduce-first)
- P9: ⬜ Diff-based review + lint gate
- P10: ⬜ Run transcripts + funnel metrics
- P11: ⬜ Config single-source-of-truth cleanup

P6–P11 are specified in the "Roadmap" section below. Work them in order:
P6 raises the capability ceiling of the whole system and unblocks the value
of every later phase. One phase per PR. When a phase lands, mark it ✅ here
and check its boxes in the roadmap section.

MVP work now includes:
- LangGraph StateGraph (graph.py) with full node sequence and routing
- TicketState, service Protocols, TicketNodeRunner (DI surface)
- LocalImplementationService (worktree prep), AdapterTestService
- IterativeImplementationService with file-only JSON tool calls
- ModelRouterProposalGenerator with deterministic fallback
- JiraWriter support for multi-ticket Epic creation in existing projects
- ModelRouterPlannerService, ModelRouterReviewService
- GitService (commit/push) + GhPullRequestOpener
- JiraEscalationService, JiraLabelApprovalService, AutoApprovalService
- OrchestratorRunner (lock+heartbeat+claim+graph+release)
- ExecutionWorker + JiraExecutionCoordinator
- config/repos/agent-system.yaml repo contract
- SQLiteCheckpointer wired into persistent graph compilation
- OrchestratorRunner stale-checkpoint guard after fresh lock acquisition
- Slack-driven execution-approval interrupt + SQLite approval persistence
- app.py process entrypoint for Slack listener + Detection + ExecutionWorker
- runtime smoke check for Slack/Jira/GitHub/config prerequisites

Deferred after MVP:
- brand-new Jira project creation
- Slack Block Kit buttons
- model-callable shell/git tools (a model-callable `run_tests` action is
  planned in P6; it may only run the repo-contract test command through
  TestAdapter)
- MCP/OpenClaw/multi-host execution
- auto-filing bug tickets from CI failures (revisit after P11)

## Roadmap: capability phases P6–P11

These phases close the gap between "the pipeline works" and "the system can
autonomously turn ideas into merged-quality PRs." Each phase below is a
self-contained spec: an implementing session should be able to complete a
phase from this section plus the referenced files, without new design
decisions. Every phase ships with unit tests under `tests/unit/` and updates
the phase list at the top of this file.

### P6 — Implementation loop upgrade (do this first)

Goal: `IterativeImplementationService` in
`src/ticket_agent/orchestrator/model_services.py` currently exposes only
`read_file`, `list_dir`, `write_file`, `finish`. That caps success at small
changes in small repos. Extend the JSON tool-call contract to:

```txt
read_file:  {path, offset?, limit?}   # offset = 1-based start line, limit = line count
list_dir:   {path}
search:     {pattern, path?, max_results?}
edit_file:  {path, old_string, new_string, replace_all?}
write_file: {path, content}
run_tests:  {}
finish:     {summary, notes?}
```

Behavior rules:

- `read_file` without `offset`/`limit` keeps current behavior (truncated at
  `tool_result_max_chars`). With `offset`/`limit` it returns only those
  lines. The result must state whether the returned view is complete.
- `search` is read-only regex search over worktree text files. It must only
  visit paths the FileAdapter would allow, skip binary files, cap results
  (default 50 matches), and return `path:line:matched-line` entries,
  truncated to `tool_result_max_chars`.
- `edit_file` performs exact-string replacement. Fail with error code
  `edit_target_not_found` when `old_string` is absent, and
  `edit_target_ambiguous` when it matches more than once without
  `replace_all: true`. Successful edits append the path to `changed_files`.
- Truncated-write guard: track, per path, within one loop run, whether the
  model has ever seen a complete view of the file (untruncated read, or the
  file did not previously exist). If not, reject `write_file` for that path
  with error code `truncated_write_rejected` and instruct the model to use
  `edit_file`. This prevents silent destruction of unseen file content.
- `run_tests` executes the repo-contract test command through the existing
  TestAdapter inside the worktree. Never auto-detect the command. Budget:
  max 5 runs per implementation attempt; further calls fail with
  `test_budget_exhausted`. Return the structured TestAdapter result
  (pass/fail, truncated stdout/stderr summary, timeout flag).
- Wiring: `LocalImplementationService` (orchestrator/local_services.py)
  already prepares the worktree context; extend the context it hands to
  `implement_context(...)` with a test runner so
  `IterativeImplementationService` never constructs adapters itself.
- Update the system prompt in `_implementation_loop_messages` to document
  every action, the edit-over-rewrite preference, and the run-tests-before-
  finish convention.

P6 checklist:

- [x] `search` action with boundary, binary-skip, and cap tests
- [x] `edit_file` action with not-found / ambiguous / replace_all tests
- [x] paginated `read_file` with complete-view flag
- [x] truncated-write guard with negative test
- [x] `run_tests` action via TestAdapter with budget test
- [x] prompt update in `_implementation_loop_messages`
- [x] no shell/git actions exposed to the model (negative test)

P6 landed. Notes for later phases: the model-callable `run_tests` action is
wired through `ImplementationContext.test_runner`, built by
`_make_contract_test_runner` in `orchestrator/local_services.py` (contract
test command via `LocalTestAdapter` only). The per-run loop state lives in
`_LoopContext` in `orchestrator/model_services.py`; `complete_view_paths`
tracks which files may be safely rewritten. P6 tests are in
`tests/unit/test_model_services_p6_actions.py`.

### P7 — Acceptance criteria pipeline

Goal: make testable acceptance criteria flow from intake through planning to
review, so review has something falsifiable to check.

- ProposalGenerator (`intake/proposal_generator.py`): each proposed ticket
  gains `acceptance_criteria: list[str]` (1–7 short, testable statements).
  Write them into the Jira description under an `Acceptance Criteria`
  heading via JiraWriter.
- Clarifying questions: when the intake model judges a request too ambiguous
  to propose, it may return a clarifying-question payload instead of a
  proposal. The Slack listener posts the question in the intake thread and
  treats the human reply as a revision input (reuse the existing proposal
  revision flow in `intake/approval_flow.py`). One clarifying round max,
  then propose with stated assumptions.
- Planning: `_planning_messages` includes the criteria; the plan payload
  must include a `criteria_coverage` mapping of criterion → planned step.
- Review: `_review_messages` includes the criteria; the review payload must
  include a per-criterion verdict (`met: bool` plus one-line evidence). Any
  unmet criterion routes to `rejected`.

P7 checklist:

- [ ] proposal schema + Jira description rendering
- [ ] clarifying-question round in Slack intake (single round)
- [ ] plan prompt + `criteria_coverage` validation
- [ ] review per-criterion verdicts drive accept/reject routing

### P8 — Bug-fix work profile

Goal: bugs get a reproduce-first workflow instead of the generic feature
flow.

- Profile detection: Jira issue type `Bug` or label `bug` sets
  `work_profile: "bug"` on `TicketState` (default `"feature"`).
- Bug planning: the plan must include a reproduction section — expected
  behavior, actual behavior, and where the regression test will live.
- Bug implementation prompt: write the failing test first, call `run_tests`
  to confirm it fails for the expected reason, then fix, then `run_tests`
  green. The `finish` summary must name the regression test.
- Review for bugs additionally checks that a regression test exists in
  `changed_files`.
- Out of scope for P8: auto-filing bugs from CI or PR feedback (deferred).

P8 checklist:

- [ ] `work_profile` on TicketState + detection from Jira fields
- [ ] bug-specific plan and implementation prompts
- [ ] review requires regression test for bug profile
- [ ] tests for profile routing and prompt selection

### P9 — Diff-based review + lint gate

Goal: review judges the real diff, and lint stops style/static regressions
before a PR opens.

- GitAdapter (`adapters/local/git_adapter.py`) gains read-only
  `diff_stat()` and `diff_text(max_chars_per_file)` against the base branch
  inside the worktree. No new write operations.
- The review node input includes the real diff (per-file truncation, keep
  the `changed_files` list) instead of only model-produced summaries.
- Lint gate: after tests pass in the RUN_TESTS node, run the repo-contract
  `lint` command when the contract defines one (it is currently dead
  config). Lint failure routes exactly like test failure
  (`retry → IMPLEMENT`). Reuse the AdapterTestService pattern; keep lint
  output in state for the implementation retry prompt.

P9 checklist:

- [ ] GitAdapter diff methods with truncation tests
- [ ] review prompt consumes real diff
- [ ] lint execution from repo contract + routing test
- [ ] contract without lint command skips the gate cleanly

### P10 — Run transcripts + funnel metrics

Goal: make runs debuggable and improvement measurable.

- TranscriptRecorder: append-only JSONL per ticket run at
  `.agent-system-data/transcripts/{TICKET-KEY}-{lock8}.jsonl`. Record node
  enter/exit, every ModelRouter attempt (capability, provider, model,
  tokens, latency, success/error), and every implementation tool call and
  truncated result. All recorded content must pass through the existing
  `redaction` module. Inject the recorder through the node runner and
  router factory; default to a no-op recorder so tests stay silent.
- Funnel metrics: SQLite table `ticket_funnel` (ticket_key, claimed_at,
  implemented_at, tests_passed_at, pr_opened_at, merged_at, escalated_at,
  escalation_reason). Written from orchestrator nodes and the merged-
  delivery poller. Add `scripts/report_funnel.py` to print stage counts and
  conversion rates.

P10 checklist:

- [ ] TranscriptRecorder + redaction coverage test
- [ ] recorder wired into node runner and router (no-op default)
- [ ] `ticket_funnel` writes at each stage
- [ ] `scripts/report_funnel.py`

### P11 — Config single source of truth

Goal: stop `capabilities.yaml` / `models.yaml` / `task_profiles.yaml` from
drifting between this repo and `ai-model-selector`.

- This repo's `config/` is canonical. Verify
  `router/selector_config.py` always loads selector config from explicit
  paths in this repo — never from the `ai-model-selector` package's bundled
  `config/`.
- In the `ai-model-selector` repo, mark its `config/` as examples only
  (README note). No selector code changes.

P11 checklist:

- [ ] selector config paths verified/enforced from this repo
- [ ] ai-model-selector README marks bundled config as example-only

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
- Merge PRs automatically
- Store secrets in this repo
- Let tools operate outside the worktree
- Let the LLM decide security policy
- Claim LangGraph or orchestration is done before it is implemented

# Ticket Agent

Ticket Agent is a Slack-first, Jira-backed software development agent system.
It is designed to turn approved human requests into tracked tickets, claim work
safely, execute changes in a controlled local environment, open pull requests,
and report progress back to the team.

The project is organized around a simple principle: external systems own human
coordination and source-of-truth state, while this repo owns the deterministic
execution pipeline that moves a ticket from "ready" to "pull request opened."

## Architecture

At a high level, the system flows through these stages:

```txt
Slack request
  -> model-assisted proposal
  -> human intake approval
  -> Jira epic/tasks
  -> ai-ready detection
  -> SQLite ticket lock/checkpoint
  -> LangGraph planning
  -> execution approval (automatic or Slack, depending on policy)
  -> implementation
  -> worktree tests
  -> summary-based model review
  -> commit/push
  -> pull request
  -> Jira and Slack reporting
```

> **This is a bounded, human-supervised ticket-processing system. Unattended
> autonomous delivery is not ready.** The system never merges pull requests
> automatically: the runtime opens PRs and does not merge them, and any review
> or merge is performed by a human as required by current policy. See the
> [Roadmap](#roadmap) for what is missing and in what order it lands.

Slack is the human-facing interface, and **Jira is the execution source of
truth today** — detection reads from Jira and the orchestrator works from Jira
tickets. (After P12 the authorized `GoalContract` becomes the scope authority
and Jira becomes a projection; that migration has not happened.) SQLite
provides local coordination for locks and workflow checkpoints. LangGraph
orchestrates the execution workflow, while the actual coding, file, shell,
test, and git operations stay in explicit Python components.

**Target repositories are external inputs** to the runtime, each described by a
`config/repos/*.yaml` contract; the application changes produced inside them
are the runtime's outputs.

## Slack Usage

Post actionable work in the configured intake channel. The bot replies with a
proposal first; Jira tickets are created only after someone replies `approve`
in that proposal thread. Reply `cancel` to discard a proposal, or describe
edits to revise it.

The Slack app must subscribe to channel message events for this to work:
`message.channels` for public channels, `message.groups` for private channels,
and `message.im` for direct-message Q&A.

Non-ticket questions can be asked in the intake channel or in a direct message
with the app. Use natural question phrasing or an explicit prefix:

```txt
is there a ticket for OAuth login?
how is AGENT-123 going?
ask: what is blocking the Slack Q&A feature?
```

The Q&A path gathers Jira context, then asks the internal model router to
compose the reply. Direct messages never create proposals or Jira tickets;
post actionable work in the intake channel so approvals and execution updates
stay visible.

## Core Components

The main package lives under `src/ticket_agent/`.

```txt
src/ticket_agent/
├── detection/
├── intake/
├── locks/
├── models/
├── router/
├── tools/
└── execution/
```

The important MVP boundaries are:

- Intake turns a human request into a proposed Jira Epic/Task set before
  approval. Multi-ticket proposals create an Epic in an existing Jira project;
  single-ticket proposals stay as Tasks. Multi-ticket initiatives run
  sequentially through `integration/<epic-key>` and conclude with a
  cumulative promotion PR when GitHub feedback polling is enabled.
- Detection finds Jira tickets that are ready for agent execution.
- Locks prevent multiple workers from claiming the same ticket.
- Tool adapters provide constrained local file, shell, test, and git access.
- The model router centralizes all LLM calls behind one internal interface.
- LangGraph sequences planning, optional plain-text execution approval, file-only
  implementation, tests, review, pull request creation, escalation, and
  reporting.

## Model Routing

LLM access is handled by an internal Python router, not an HTTP service.
Components call `ModelRouter.invoke(...)` directly and never call provider APIs
on their own.

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

The router uses `ai-model-selector` for deterministic capability selection,
then maps the selected model to a configured provider client. It owns fallback
execution, attempt tracking, response normalization, timeouts, and provider
lookup.

Provider roles in v1:

- DeepSeek V4 Pro is the primary coding and implementation model.
- Gemini is used for verification, structured checks, planning/design, and
  future research-style tasks.
- Ollama/Qwen is an optional local fallback for simple work.

MiniMax, GLM, cost-aware routing, MCP, OpenClaw, and an external router service
are outside the v1 scope.

## Tooling And Safety

Execution tools are intentionally narrow. They are direct Python adapters with
policy checks at the boundary:

- `FileAdapter` enforces worktree boundaries, protected paths, and symlink
  escape protection.
- `ShellAdapter` runs only allowlisted commands with a stripped environment and
  explicit timeouts. Production repo-contract commands require Bubblewrap
  before locks, claims, worktrees, or resume and return per-command sandbox
  attestations.
- `TestAdapter` loads test commands from repo contracts instead of inferring
  them from project files.
- `GitAdapter` works through isolated branches/worktrees and opens pull
  requests without merging them.

Secrets are not passed into prompts, LangGraph state, tool calls, logs, or test
output. Provider clients load their own credentials from environment/config.

## Repo Contracts

Each target repository needs a contract under `config/repos/`. The contract
describes the repo root, source directories, protected paths, and allowed test
commands. This keeps agent execution predictable and prevents the system from
guessing how to run a project.

## Runtime Configuration

The production entrypoint is:

```bash
ticket-agent
```

It reads `.env` by default, or a path provided via `AGENT_SYSTEM_ENV_PATH` /
`--env-path`. Production deployments can point `AGENT_SYSTEM_ENV_PATH` at a
host-managed secret file such as `~/config/agent-system.env`.

### Runtime wiring

`ticket-agent` loads validated environment config, builds the Jira and Slack
adapters, then starts four long-running loops together:

- Slack intake listener
- Jira detection polling
- Execution worker
- Lock reconciler

`ticket-agent-smoke-runtime` is the non-mutating preflight for that runtime.

- `--skip-network` checks startup config, repo contracts, Jira field-map
  wiring, model-provider env vars, and local GitHub CLI auth.
- Without `--skip-network`, it also checks Slack `auth.test`, Jira `/myself`,
  each Jira project listed in `AGENT_SYSTEM_JIRA_TARGET_PROJECTS`, required
  Epic/Task issue types, and the configured Epic Link field when
  `JIRA_FIELD_EPIC_LINK` is set.

### Required local runtime configuration

Copy [.env.example](.env.example) to `.env` and fill in real values.
The primary variables for local Slack/Jira runs are:

- Slack:
  - `SLACK_BOT_TOKEN`
  - `SLACK_APP_TOKEN`
  - `AGENT_SYSTEM_INTAKE_CHANNEL`
  - `AGENT_SYSTEM_EXECUTION_APPROVAL_CHANNEL`
- Jira:
  - `JIRA_BASE_URL`
  - `JIRA_USER_EMAIL`
  - `JIRA_API_KEY`
  - `AGENT_SYSTEM_JIRA_TARGET_PROJECTS`
- Jira field map:
  - `JIRA_FIELD_AGENT_ASSIGNED_COMPONENT`
  - `JIRA_FIELD_AGENT_RETRY_COUNT`
  - `JIRA_FIELD_AGENT_CAPABILITIES_NEEDED`
  - `JIRA_FIELD_REPOSITORY`
  - `JIRA_FIELD_REPO_PATH`
  - `JIRA_FIELD_SLACK_THREAD_TS`
  - `JIRA_FIELD_SLACK_CHANNEL`
  - `JIRA_FIELD_MAX_ATTEMPTS`
  - `JIRA_FIELD_EPIC_LINK` when the Jira project still requires an Epic Link
    custom field
- Model providers:
  - `DEEPSEEK_API_KEY`
  - `GEMINI_API_KEY`
- GitHub identities:
  - `GH_ADMIN_TOKEN` for the admin account that creates repositories and adds
    collaborators
  - `GH_BOT_TOKEN` for the bot account that pushes code, opens PRs, and
    processes PR feedback
- Repo contract path:
  - `AGENT_SYSTEM_REPO_CONFIG_PATH` (defaults to `config/repos`)
- Intake proposal generation:
  - `AGENT_SYSTEM_INTAKE_MODEL_TIMEOUT_SECONDS` (defaults to `30`)
- Execution mode:
  - `AGENT_SYSTEM_EXECUTION_MODE=dry_run` for the first Slack/Jira slice so
    execution approval records Jira/Slack state without attempting code changes
  - `AGENT_SYSTEM_EXECUTION_APPROVAL_POLICY=slack` if you want a second
    per-ticket approval after planning; otherwise proposal approval is enough
    in execute mode

For personal GitHub repositories, configure `GH_ADMIN_TOKEN` and
`GH_BOT_TOKEN` as personal access tokens (classic) with the `repo` scope.
When a repository must be created, the admin account owns it and invites the
bot account as a collaborator; implementation branches and PRs then use the
bot identity. Keep both tokens only in `.env`. See
[docs/setup-guide.md](docs/setup-guide.md#3-github-accounts) for creation and
verification steps. Code-writing GitHub operations fail fast when
`GH_BOT_TOKEN` is missing so the runtime does not fall back to a developer's
local `gh` login.

Local prerequisites that are not environment variables:

- `gh` must be installed. The smoke check validates each configured GitHub
  token, or falls back to local `gh` authentication when role tokens are not
  provided.
- The repo contract path must contain valid YAML contracts for each target repo.

### Local integration checklist

1. Create a local env file:

   ```bash
   cp .env.example .env
   ```

2. Confirm GitHub CLI auth before any runtime smoke check:

   ```bash
   gh auth status
   ```

3. Run unit tests:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" python -m pytest tests/unit/ -q
   ```

4. Run smoke without network calls:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" ticket-agent-smoke-runtime --skip-network
   ```

5. Run smoke with Slack/Jira network checks:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" ticket-agent-smoke-runtime
   ```

6. Print manual Slack/Jira vertical-slice steps:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" ticket-agent-smoke-e2e
   ```

7. Start the app locally:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" ticket-agent
   ```

## Roadmap

**P0–P7 are complete: a bounded, human-reviewed ticket-processing pipeline.**
Slack intake, Jira tickets, detection, locking, the LangGraph workflow,
implementation, tests, PR creation, and feedback polling are implemented and
unit-tested. External Slack/Jira/GitHub end-to-end verification remains manual.

Everything after that exists to make the pipeline trustworthy enough to run
*unattended*, which **it is not today**.

📖 **The canonical, detailed roadmap** — per-phase specifications, current
wiring status, checklists, exit criteria, and known limitations — is
[`docs/autonomous-delivery-roadmap.md`](docs/autonomous-delivery-roadmap.md).
The table below is a summary; the roadmap document is authoritative.

Status marks: ✅ complete (implemented, wired, enforced, tested) · 🔶 partial
(components exist, invariant not enforced) · ⬜ not started · ⏸️ deferred by
dependency · ↪ absorbed into another phase.

| Phase | | Summary |
|---|---|---|
| P0–P7 | ✅ | Bounded ticket-processing MVP: infrastructure, intake, detection, locking, adapters, router, implementation loop, acceptance criteria |
| P8 | ⏸️ | Bug-fix work profile. Blocked until P15/P16 can tell whether a strategy helped |
| P9 | ↪ | Real-diff review and the lint gate. Absorbed into P15 and P14; never scheduled separately |
| P10 | 🔶 | Observability foundations exist; later producers and operational evidence are missing |
| P11 | ✅ | This repo owns selector config; resolution is pinned and the library copy is example-only |
| P12 | 🔶 | Goal contracts exist but durable authority, revocation, spine, journal, and execution enforcement do not |
| P13 | 🔶 | Sandbox enforcement and per-command evidence are wired; context assembly / knowledge map remains |
| P14 | ⬜ | Verify an immutable committed SHA in an isolated checkout |
| P15 | ⬜ | Independent review of the complete real diff, maker ≠ checker across fallbacks |
| P16 | ⬜ | Repeatable evaluation corpus and demonstration evidence |
| P17 | ⬜ | Durable delivery: outbox, attestation, trusted CI, promotion |
| P18 | ⬜ | Bounded rework from promotion-PR change requests |

**Phase numbers are stable identities, not execution order.** They are cited by
commits and tests, so they are never renumbered — but dependencies dictate a
different working order:

> P11 → P13.3a → P12.2a → (P12.2b + P12.2c) → P12.2d → P12.2e
> → P14 → P15 → P16 → P17 → P18, then revisit P8.

P13.3a comes before P12 because sandbox refusal must protect every execution
entry point before later authorization plumbing relies on it. P12.2b and
P12.2c ship together so durable authorization publication and consumption
cannot diverge. P12 comes before P14–P17 because those phases all record
evidence against a goal that does not yet reach the graph. P18 follows P17
because it acts on comments left on the promotion PR that P17 opens. Per-step
rationale is in
[the roadmap](docs/autonomous-delivery-roadmap.md#recommended-execution-order).

Two things are worth knowing before trusting a status mark anywhere:

- **A capability is not complete because its types exist.** The sandbox and the
  goal contract are both fully implemented and neither is enforced.
- **Opt-in is not operational proof.** Transcripts default to off; a feature
  that *can* be enabled has demonstrated nothing.

## Tests

Unit tests live under `tests/unit/`, and run through the repo virtual
environment:

```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest tests/unit/ -q
```

Use `python -m pytest`, not the bare `pytest` console script — `python -m` puts
the working directory on `sys.path`, which the test suite's imports require.
The bare console script fails during collection.

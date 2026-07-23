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
  -> intake approval
  -> Jira epic/tasks
  -> ai-ready detection
  -> SQLite ticket lock
  -> LangGraph workflow
  -> execution approval
  -> implementation tools
  -> tests
  -> pull request
  -> Jira and Slack update
```

Slack is the human-facing interface. Jira is the execution source of truth.
SQLite provides local coordination for locks and future workflow checkpoints.
LangGraph is intended to orchestrate the execution workflow, while the actual
coding, file, shell, test, and git operations stay in explicit Python
components.

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
  explicit timeouts.
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
   PATH="$PWD/.venv/bin:$PATH" pytest tests/unit/ -q
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

The MVP pipeline (P0–P5) is complete: Slack intake, Jira tickets, detection,
locking, the LangGraph workflow, implementation, tests, PR creation, and
feedback polling all run end to end. The next phases close the gap between
"the pipeline works" and "the system autonomously turns ideas into
merge-quality PRs." Full implementation specs for each phase live in
[CLAUDE.md](CLAUDE.md#roadmap-capability-phases-p6p11); work them in order,
one phase per PR.

- **P6 — Implementation loop upgrade. (done)** The coding loop gained
  `search` (regex over the worktree), `edit_file` (exact-string patching
  instead of full-file rewrites), paginated `read_file`, a guard that blocks
  blind rewrites of partially-read files, and a bounded `run_tests` action
  that runs the repo-contract test command inside the loop.
- **P7 — Acceptance criteria pipeline. (done)** Intake proposals now include
  testable acceptance criteria (with one clarifying-question round in Slack
  when a request is ambiguous), rendered into the Jira description under an
  `Acceptance Criteria` heading. Plans return a `criteria_coverage` map, and
  review returns a per-criterion verdict that routes to rejected when any
  criterion is unmet. Multi-ticket proposals are ordered foundation-first so
  the first ticket establishes the shared scaffold the rest build on.
- **P8 — Bug-fix work profile.** Tickets typed/labeled as bugs get a
  reproduce-first flow: write a failing regression test, confirm it fails,
  fix, confirm green. Review requires the regression test.
- **P9 — Diff-based review + lint gate.** Review reads the real git diff
  from the worktree, and the repo contract's `lint` command runs after
  tests, routing back to implementation on failure.
- **P10 — Transcripts + funnel metrics.** Every run writes a redacted JSONL
  transcript (model attempts, tool calls, node transitions), and a
  `ticket_funnel` table tracks claimed → implemented → tests passed → PR
  opened → merged, with a report script.
- **P11 — Config ownership.** This repo's `config/` becomes the single
  source of truth for selector configuration; `ai-model-selector`'s bundled
  configs are marked example-only.

## Tests

Unit tests live under `tests/unit/`.

If `pytest` is not available globally, use the repo virtual environment:

```bash
PATH="$PWD/.venv/bin:$PATH" pytest tests/unit/
```

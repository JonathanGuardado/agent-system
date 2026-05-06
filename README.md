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
  single-ticket proposals stay as Tasks.
- Detection finds Jira tickets that are ready for agent execution.
- Locks prevent multiple workers from claiming the same ticket.
- Tool adapters provide constrained local file, shell, test, and git access.
- The model router centralizes all LLM calls behind one internal interface.
- LangGraph sequences planning, plain-text execution approval, file-only
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

It reads `~/config/agent-system.env` by default. Required values include Slack
Socket Mode tokens, Jira Cloud credentials, DeepSeek/Gemini provider keys, and
the Jira custom field map used for execution metadata.

Important Jira field mappings:

- `JIRA_FIELD_AGENT_ASSIGNED_COMPONENT`
- `JIRA_FIELD_AGENT_RETRY_COUNT`
- `JIRA_FIELD_AGENT_CAPABILITIES_NEEDED`
- `JIRA_FIELD_REPOSITORY`
- `JIRA_FIELD_REPO_PATH`
- `JIRA_FIELD_SLACK_THREAD_TS`
- `JIRA_FIELD_SLACK_CHANNEL`
- `JIRA_FIELD_MAX_ATTEMPTS`
- `JIRA_FIELD_EPIC_LINK` when the Jira project requires an Epic Link custom
  field instead of the standard parent field.

Run the non-mutating runtime smoke check before starting the worker:

```bash
ticket-agent-smoke-runtime --skip-network
ticket-agent-smoke-runtime
```

The first command validates local config, repo contracts, provider env vars,
and GitHub CLI auth without network calls. The second also checks Slack
`auth.test` and Jira `/myself`.

## Tests

Unit tests live under `tests/unit/`.

If `pytest` is not available globally, use the repo virtual environment:

```bash
PATH="$PWD/.venv/bin:$PATH" pytest tests/unit/
```

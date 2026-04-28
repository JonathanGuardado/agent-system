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
  -> intake approval
  -> Jira ticket
  -> ai-ready detection
  -> SQLite ticket lock
  -> LangGraph workflow
  -> implementation tools
  -> tests
  -> pull request
  -> Slack update
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

The important architectural boundaries are:

- Intake turns a human request into a proposed Jira ticket after approval.
- Detection finds Jira tickets that are ready for agent execution.
- Locks prevent multiple workers from claiming the same ticket.
- Tool adapters provide constrained local file, shell, test, and git access.
- The model router centralizes all LLM calls behind one internal interface.
- The future LangGraph workflow will sequence planning, implementation,
  testing, review, escalation, and reporting.

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

## Tests

Unit tests live under `tests/unit/`.

If `pytest` is not available globally, use the repo virtual environment:

```bash
PATH="$PWD/.venv/bin:$PATH" pytest tests/unit/
```

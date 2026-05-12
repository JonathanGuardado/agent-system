# Setup Guide

This guide walks through every value the system needs to run locally,
where to obtain it, and how to verify the result. Pair it with
[.env.example](../.env.example) — copy that file to `.env` and fill in
the real values as you work through each section.

```bash
cp .env.example .env
```

The runtime loads `.env` by default. You can still override it with
`AGENT_SYSTEM_ENV_PATH` or `--env-path` when you want to use a different file.
Production deployments can point `AGENT_SYSTEM_ENV_PATH` at a host-managed
secret file such as `~/config/agent-system.env`.

> Secrets must never be committed. `.env` is gitignored. Provider
> clients read their own keys from the environment — do not pass keys into
> prompts, tool adapters, LangGraph state, or logs.

---

## 0. Local prerequisites (not env vars)

Before any value below matters, make sure the host has:

- **Python venv** at `.venv/` with the project installed.
- **`gh` CLI** installed and authenticated. The smoke check calls
  `gh auth status`. If it fails, no PR can be opened.

  ```bash
  gh auth status
  gh auth login   # if not authenticated
  ```

- **Repo contracts** under whatever path you set for
  `AGENT_SYSTEM_REPO_CONFIG_PATH`. Each target repo needs a YAML contract
  (see [Section 6](#6-repo-contracts) and the schema in [CLAUDE.md](../CLAUDE.md)).

- **Ollama** (optional, only if you want a local fallback). Should listen on
  `localhost:11434` with a Qwen-family model pulled.

---

## 1. Slack

Slack is the human-facing interface. The runtime uses **Socket Mode**, so the
app does not need a public URL.

### Create the Slack app

1. Go to <https://api.slack.com/apps> → **Create New App** → **From scratch**.
2. Pick a workspace you control.
3. Under **Socket Mode**, enable it and generate an **App-Level Token**
   with the `connections:write` scope. This is your `SLACK_APP_TOKEN`
   (starts with `xapp-`).
4. Under **OAuth & Permissions**, add the bot scopes the listener needs at a
   minimum:

   - `chat:write`
   - `channels:history`
   - `groups:history`
   - `im:history`
   - `app_mentions:read`
   - `reactions:read`

5. Install the app to the workspace. Copy the **Bot User OAuth Token**
   (starts with `xoxb-`). This is your `SLACK_BOT_TOKEN`.
6. To use the app as a direct-message assistant for non-ticket Jira
   questions, enable the app's messages tab / direct messages in Slack app
   configuration and subscribe the app to message events for DMs.
7. Invite the bot to the intake channel and the execution-approval channel.
8. Copy each channel's ID (Slack → channel → **View channel details** → copy
   the `C…` ID at the bottom).

### Variables

| Variable | What it is | How to get it |
|---|---|---|
| `SLACK_BOT_TOKEN` | Bot user OAuth token (`xoxb-…`). | Slack app → OAuth & Permissions. |
| `SLACK_APP_TOKEN` | App-level token (`xapp-…`) with `connections:write`. | Slack app → Basic Information → App-Level Tokens. |
| `AGENT_SYSTEM_INTAKE_CHANNEL` | Channel ID where humans request work. | Slack channel details. |
| `AGENT_SYSTEM_EXECUTION_APPROVAL_CHANNEL` | Channel ID where the bot asks for execution approval. | Slack channel details. May be the same channel as intake. |

---

## 2. Jira Cloud

Jira is the execution source of truth. Detection reads from Jira, the
orchestrator works from Jira tickets, and proposals are written here after
intake approval.

### Get the credentials

1. `JIRA_BASE_URL` is your tenant URL, e.g. `https://your-domain.atlassian.net`
   (no trailing slash).
2. `JIRA_USER_EMAIL` is the Atlassian account email used for API auth.
3. Create an API token at
   <https://id.atlassian.com/manage-profile/security/api-tokens> — that value
   is `JIRA_API_KEY`.
4. `JIRA_TIMEOUT_SECONDS` defaults to `30`. Increase if your Jira tenant is
   slow.

### Pick the target project(s)

`AGENT_SYSTEM_JIRA_TARGET_PROJECTS` is a comma-separated list of Jira
project keys the system is allowed to read and write (e.g. `AGENT,WEB`).
Each project must:

- Already exist (the v1 system does not create new projects).
- Have **Epic** and **Task** issue types available.
- Be accessible to the API user.

### Map the custom fields

Jira custom fields are tenant-specific; the system needs the `customfield_…`
ID for each logical field. Find IDs by opening any issue in your project and
calling:

```bash
curl -u "$JIRA_USER_EMAIL:$JIRA_API_KEY" \
  "$JIRA_BASE_URL/rest/api/3/field" \
  | jq '.[] | {id, name}'
```

Match each logical field below to a custom field ID in your tenant. If a
field does not yet exist, create it in Jira first.

| Variable | Logical purpose |
|---|---|
| `                                                                                                                                                                                             ` | Which agent component owns the ticket. |
| `JIRA_FIELD_AGENT_RETRY_COUNT` | Retry counter the orchestrator increments. |
| `JIRA_FIELD_AGENT_CAPABILITIES_NEEDED` | Capabilities the ticket requires. |
| `JIRA_FIELD_REPOSITORY` | Repo name the work targets. |
| `JIRA_FIELD_REPO_PATH` | Local clone path for the repo. |
| `JIRA_FIELD_SLACK_THREAD_TS` | Slack thread that originated the ticket. |
| `JIRA_FIELD_SLACK_CHANNEL` | Slack channel the originating thread is in. |
| `JIRA_FIELD_MAX_ATTEMPTS` | Optional ticket -specific attempt cap. |
| `JIRA_FIELD_EPIC_LINK` | Only set if your project still uses a custom Epic Link field (older Jira projects). Leave unset on next-gen projects that use the native parent. |

The required keys (everything except `JIRA_FIELD_MAX_ATTEMPTS` and
`JIRA_FIELD_EPIC_LINK`) are enforced by the smoke check; missing values fail
startup.

If you prefer to ship the whole map as JSON instead of individual env vars,
set `JIRA_FIELD_MAP_JSON` to a JSON object whose keys are the logical names
and values are the `customfield_…` IDs. Individual `JIRA_FIELD_*` vars take
precedence when both are set.

---

## 3. Model providers

The internal `ModelRouter` calls providers directly. Two API keys are
required for v1; Ollama is optional.

| Variable | Provider | Where to get it |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek (primary coding/implementation model). | <https://platform.deepseek.com> → API keys. |
| `GEMINI_API_KEY` | Google Gemini (verification, planning, structured checks). | <https://aistudio.google.com/app/apikey>. |

Ollama (optional fallback) is detected by HTTP probe at
`localhost:11434`. No env var is needed; pull a Qwen model first:

```bash
ollama pull qwen3.5:9b
```

> v1 deliberately excludes MiniMax and GLM. Do not add provider keys for
> those. Cost-aware routing is also out of scope; estimated cost fields are
> recorded for future telemetry but never affect model choice.

---

## 4. Local runtime options

These tune the long-running loops. Defaults are in `.env.example` and are
sane for local development.

| Variable | Default | Meaning |
|---|---|---|
| `AGENT_SYSTEM_REPO_CONFIG_PATH` | `config/repos` | Directory holding repo contract YAMLs. Resolved relative to the working directory. |
| `AGENT_SYSTEM_DATA_DIR` | `.agent-system-data` | Where SQLite lock and checkpoint DBs live. |
| `AGENT_SYSTEM_COMPONENT_ID` | `agent-system-local` | Identifier the lock manager writes when claiming tickets. Use a unique value per host. |
| `AGENT_SYSTEM_POLL_INTERVAL_SECONDS` | `30` | How often detection polls Jira for `ai-ready` tickets. |
| `AGENT_SYSTEM_MAX_BACKOFF_SECONDS` | `300` | Upper bound for poll backoff. Must be ≥ `POLL_INTERVAL_SECONDS`. |
| `AGENT_SYSTEM_HEARTBEAT_INTERVAL_SECONDS` | `600` | How often the active worker refreshes its lock heartbeat. |
| `AGENT_SYSTEM_RECONCILE_INTERVAL_SECONDS` | `300` | How often the reconciler clears stale locks/checkpoints. |
| `AGENT_SYSTEM_PULL_REQUEST_BASE_BRANCH` | `main` | Base branch every agent PR targets. |
| `AGENT_SYSTEM_EXECUTION_MODE` | `execute` | Use `dry_run` for the first vertical-slice test so approval stops before implementation. |

Optional:

- `AGENT_SYSTEM_RECONCILE_BATCH_SIZE` — caps how many stale locks the
  reconciler processes per cycle.
- `AGENT_SYSTEM_CONTRACT_DIR` — alternative spelling of
  `AGENT_SYSTEM_REPO_CONFIG_PATH`; the first one set wins.
- `AGENT_SYSTEM_ENV_PATH` — the entrypoint reads this if no `--env-path` is
  passed. Use it only when you want to override the default `.env`.

---

## 5. Repo contracts

Every target repo needs a contract under `AGENT_SYSTEM_REPO_CONFIG_PATH`,
named after the repo (e.g. `config/repos/agent-system.yaml`). The contract
tells the system how to test, lint, and stay inside safe directories. The
schema and required fields live in [CLAUDE.md](../CLAUDE.md). The smoke
check fails if the contract directory is missing or empty.

Minimum required keys per contract: `repo`, `language`, `commands.test`,
`policy`, `source_dirs`, `test_dirs`. Test commands are never auto-detected
— they must come from the contract.

---

## 6. Verifying the setup

Run these in order. Each step gates the next.

1. **Unit tests** (no env needed):

   ```bash
   PATH="$PWD/.venv/bin:$PATH" pytest tests/unit/ -q
   ```

2. **Smoke check, no network** — validates startup config, repo contracts,
   Jira field-map wiring, model-provider env vars, and `gh auth status`:

   ```bash
  PATH="$PWD/.venv/bin:$PATH" ticket-agent-smoke-runtime --skip-network
   ```

3. **Smoke check with network** — additionally hits Slack `auth.test`,
   Jira `/myself`, every project in `AGENT_SYSTEM_JIRA_TARGET_PROJECTS`,
   confirms Epic/Task issue types exist, and (if set) verifies
   `JIRA_FIELD_EPIC_LINK`:

   ```bash
  PATH="$PWD/.venv/bin:$PATH" ticket-agent-smoke-runtime
   ```

4. **Start the runtime**:

   ```bash
  PATH="$PWD/.venv/bin:$PATH" ticket-agent
   ```

   This runs the Slack listener, Jira detection poller, execution worker,
   and lock reconciler together.

5. **Print manual vertical-slice steps**:

   ```bash
  PATH="$PWD/.venv/bin:$PATH" ticket-agent-smoke-e2e
   ```

---

## 7. Common failure modes

- **Smoke fails on `repo_contracts`** — `AGENT_SYSTEM_REPO_CONFIG_PATH`
  points somewhere with no YAML files, or the YAML is invalid.
- **Smoke fails on `jira_field_map`** — a required `JIRA_FIELD_*` variable
  is missing or set to a `customfield_…` ID that does not exist in the
  tenant.
- **`jira_project_metadata` fails** — the API user can see the project but
  the project does not have both `Epic` and `Task` issue types.
- **`github_auth` fails** — install `gh` first, then re-run `gh auth login`.
  The orchestrator cannot
  open PRs without it.
- **Slack listener silent** — the bot was not invited to
  `AGENT_SYSTEM_INTAKE_CHANNEL`, or the app-level token is missing the
  `connections:write` scope.
- **Provider errors with no detail** — provider clients deliberately do not
  log keys or full request bodies. Re-check the key value out-of-band; do
  not add print statements that include the key.

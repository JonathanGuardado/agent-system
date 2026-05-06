from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from ticket_agent import runtime_smoke as runtime_smoke_module
from ticket_agent.runtime_smoke import collect_smoke_checks, main


def test_runtime_smoke_passes_non_mutating_checks_with_configured_env(tmp_path):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=True,
            run_command=_successful_command,
        )
    )

    by_name = {check.name: check for check in checks}
    assert by_name["startup_config"].status == "pass"
    assert by_name["repo_contracts"].status == "pass"
    assert by_name["jira_field_map"].status == "pass"
    assert by_name["github_auth"].status == "pass"
    assert by_name["slack_auth"].status == "skip"
    assert by_name["jira_auth"].status == "skip"


def test_runtime_smoke_reports_missing_jira_field_map(tmp_path):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    for key in tuple(env):
        if key.startswith("JIRA_FIELD_"):
            del env[key]

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=True,
            run_command=_successful_command,
        )
    )

    field_map = {check.name: check for check in checks}["jira_field_map"]
    assert field_map.status == "fail"
    assert "repository" in field_map.detail


def test_runtime_smoke_main_returns_failure_when_gh_auth_fails(tmp_path, capsys):
    contract_dir = _write_contract(tmp_path)

    def failing_command(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            ("gh", "auth", "status"),
            1,
            stdout="",
            stderr="not logged in",
        )

    exit_code = main(
        ["--skip-network", "--env-path", str(_empty_env_file(tmp_path))],
        env=_env(contract_dir),
        run_command=failing_command,
    )

    assert exit_code == 1
    assert "not logged in" in capsys.readouterr().out


def test_jira_project_metadata_skipped_when_no_network(tmp_path):
    contract_dir = _write_contract(tmp_path)
    checks = asyncio.run(
        collect_smoke_checks(
            env=_env(contract_dir),
            env_path=_empty_env_file(tmp_path),
            skip_network=True,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_project_metadata"].status == "skip"
    assert by_name["jira_epic_link_field"].status == "skip"


def test_jira_project_metadata_skipped_when_no_target_projects(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    # No AGENT_SYSTEM_JIRA_TARGET_PROJECTS in env
    monkeypatch.setattr(runtime_smoke_module, "httpx", _FakeHttpx([]))

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_project_metadata"].status == "skip"
    assert "AGENT_SYSTEM_JIRA_TARGET_PROJECTS" in by_name["jira_project_metadata"].detail
    assert by_name["jira_epic_link_field"].status == "skip"


def test_jira_project_exists_check_passes(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    env["AGENT_SYSTEM_JIRA_TARGET_PROJECTS"] = "AGENT"
    project_response = {
        "key": "AGENT",
        "issueTypes": [
            {"name": "Task"},
            {"name": "Bug"},
            {"name": "Epic"},
        ],
    }
    # _env() includes JIRA_FIELD_EPIC_LINK → field check runs and needs a response.
    field_response = [{"id": "customfield_epic_link"}]
    monkeypatch.setattr(
        runtime_smoke_module,
        "httpx",
        _FakeHttpx([project_response, field_response]),
    )

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_project_agent"].status == "pass"
    assert by_name["jira_issue_types_agent"].status == "pass"


def test_jira_project_not_found_fails(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    env["AGENT_SYSTEM_JIRA_TARGET_PROJECTS"] = "MISSING"
    # _env() includes JIRA_FIELD_EPIC_LINK → field check runs after the project check.
    field_response = [{"id": "customfield_epic_link"}]
    monkeypatch.setattr(
        runtime_smoke_module,
        "httpx",
        _FakeHttpx([_NotFoundResponse(), field_response]),
    )

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_project_missing"].status == "fail"
    assert "not found" in by_name["jira_project_missing"].detail
    assert by_name["jira_issue_types_missing"].status == "skip"


def test_jira_issue_types_check_fails_when_epic_missing(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    env["AGENT_SYSTEM_JIRA_TARGET_PROJECTS"] = "AGENT"
    project_response = {
        "key": "AGENT",
        "issueTypes": [{"name": "Task"}, {"name": "Bug"}],  # no Epic
    }
    # _env() includes JIRA_FIELD_EPIC_LINK → field check runs.
    field_response = [{"id": "customfield_epic_link"}]
    monkeypatch.setattr(
        runtime_smoke_module,
        "httpx",
        _FakeHttpx([project_response, field_response]),
    )

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_project_agent"].status == "pass"
    assert by_name["jira_issue_types_agent"].status == "fail"
    assert "Epic" in by_name["jira_issue_types_agent"].detail


def test_jira_epic_link_field_passes_when_configured_and_found(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    env["AGENT_SYSTEM_JIRA_TARGET_PROJECTS"] = "AGENT"
    env["JIRA_FIELD_EPIC_LINK"] = "customfield_10014"
    project_response = {
        "key": "AGENT",
        "issueTypes": [{"name": "Task"}, {"name": "Epic"}],
    }
    fields_response = [{"id": "customfield_10014", "name": "Epic Link"}]
    monkeypatch.setattr(
        runtime_smoke_module,
        "httpx",
        _FakeHttpx([project_response, fields_response]),
    )

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_epic_link_field"].status == "pass"
    assert "customfield_10014" in by_name["jira_epic_link_field"].detail


def test_jira_epic_link_field_fails_when_field_not_in_jira(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    env["AGENT_SYSTEM_JIRA_TARGET_PROJECTS"] = "AGENT"
    env["JIRA_FIELD_EPIC_LINK"] = "customfield_99999"
    project_response = {
        "key": "AGENT",
        "issueTypes": [{"name": "Task"}, {"name": "Epic"}],
    }
    fields_response = [{"id": "customfield_10014", "name": "Epic Link"}]
    monkeypatch.setattr(
        runtime_smoke_module,
        "httpx",
        _FakeHttpx([project_response, fields_response]),
    )

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_epic_link_field"].status == "fail"
    assert "customfield_99999" in by_name["jira_epic_link_field"].detail


def test_jira_epic_link_field_skipped_when_not_configured(tmp_path, monkeypatch):
    contract_dir = _write_contract(tmp_path)
    env = _env(contract_dir)
    env["AGENT_SYSTEM_JIRA_TARGET_PROJECTS"] = "AGENT"
    # Remove the EPIC_LINK mapping so the field check is skipped (no GET /field call).
    del env["JIRA_FIELD_EPIC_LINK"]
    project_response = {
        "key": "AGENT",
        "issueTypes": [{"name": "Task"}, {"name": "Epic"}],
    }
    monkeypatch.setattr(
        runtime_smoke_module,
        "httpx",
        _FakeHttpx([project_response]),
    )

    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=_empty_env_file(tmp_path),
            skip_network=False,
            run_command=_successful_command,
        )
    )
    by_name = {check.name: check for check in checks}
    assert by_name["jira_epic_link_field"].status == "skip"
    assert "parent key" in by_name["jira_epic_link_field"].detail


# ---------------------------------------------------------------------------
# Fake httpx infrastructure for network-isolated smoke check tests
# ---------------------------------------------------------------------------
#
# _FakeHttpx uses a shared response queue consumed across AsyncClient instances.
# URL /myself is handled automatically (jira auth) without consuming from the
# queue. Slack auth uses post() which always returns {"ok": True}.
# All other GET calls consume the next item from the queue in order.


class _NotFoundResponse:
    status_code = 404

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return {}


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeAsyncClient:
    """One AsyncClient instance, backed by the shared _FakeHttpx response queue."""

    def __init__(self, owner: "_FakeHttpx") -> None:
        self._owner = owner

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> Any:
        del kwargs
        if "/myself" in url:
            # Jira auth check — always succeed without consuming the queue.
            return _FakeResponse({})
        resp = self._owner._consume()
        return resp if isinstance(resp, _NotFoundResponse) else _FakeResponse(resp)

    async def post(self, url: str, **kwargs: Any) -> Any:
        del url, kwargs
        return _FakeResponse({"ok": True})


class _FakeHttpx:
    """Drop-in replacement for the httpx module in smoke check tests."""

    def __init__(self, responses: list[Any]) -> None:
        self._queue = list(responses)
        self._pos = 0

    def AsyncClient(self, **kwargs: Any) -> "_FakeAsyncClient":
        del kwargs
        return _FakeAsyncClient(self)

    def _consume(self) -> Any:
        item = self._queue[self._pos]
        self._pos += 1
        return item


def _write_contract(tmp_path: Path) -> Path:
    contract_dir = tmp_path / "repos"
    contract_dir.mkdir()
    (contract_dir / "agent-system.yaml").write_text(
        """
repo:
  name: agent-system
  root: /tmp/agent-system
  default_branch: main
language:
  primary: python
  package_manager: setuptools
commands:
  test:
    command: ["python", "-m", "pytest", "tests/", "-q"]
    timeout_seconds: 120
    working_directory: "."
  lint: null
  install: null
policy:
  dependency_install_allowed: false
  protected_paths: []
source_dirs:
  - src/
test_dirs:
  - tests/
""".strip(),
        encoding="utf-8",
    )
    return contract_dir


def _empty_env_file(tmp_path: Path) -> Path:
    env_path = tmp_path / "agent-system.env"
    env_path.write_text("", encoding="utf-8")
    return env_path


def _env(contract_dir: Path) -> dict[str, str]:
    env = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "JIRA_BASE_URL": "https://jira.example.test",
        "JIRA_USER_EMAIL": "agent@example.test",
        "JIRA_API_KEY": "jira-key",
        "DEEPSEEK_API_KEY": "deepseek-key",
        "GEMINI_API_KEY": "gemini-key",
        "AGENT_SYSTEM_REPO_CONFIG_PATH": str(contract_dir),
    }
    for logical in (
        "AGENT_ASSIGNED_COMPONENT",
        "AGENT_CAPABILITIES_NEEDED",
        "AGENT_RETRY_COUNT",
        "REPOSITORY",
        "REPO_PATH",
        "SLACK_CHANNEL",
        "SLACK_THREAD_TS",
        "MAX_ATTEMPTS",
        "EPIC_LINK",
    ):
        env[f"JIRA_FIELD_{logical}"] = f"customfield_{logical.lower()}"
    return env


def _successful_command(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    del args, kwargs
    return subprocess.CompletedProcess(
        ("gh", "auth", "status"),
        0,
        stdout="logged in",
        stderr="",
    )

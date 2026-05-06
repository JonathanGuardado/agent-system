from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

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

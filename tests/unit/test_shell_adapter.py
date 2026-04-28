from __future__ import annotations

import sys

import pytest

from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.domain.errors import CommandNotAllowedError, PathBoundaryError


def _worktree(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    return worktree


def test_shell_adapter_runs_allowlisted_command_inside_worktree(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
    )

    result = shell.run((sys.executable, "-c", "print('ok')"))

    assert result.ok
    assert not result.timed_out
    assert result.stdout == "ok\n"
    assert result.stderr == ""


def test_shell_adapter_rejects_command_outside_allowlist(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
    )

    with pytest.raises(CommandNotAllowedError):
        shell.run((sys.executable, "-m", "pytest"))


def test_shell_adapter_rejects_denylisted_command_even_if_allowlisted(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[("curl",)],
    )

    with pytest.raises(CommandNotAllowedError):
        shell.run(("curl", "https://example.com"))


def test_shell_adapter_rejects_dangerous_argv_containing_docker(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
    )

    with pytest.raises(CommandNotAllowedError):
        shell.run((sys.executable, "-c", "print('docker')"))


def test_shell_adapter_rejects_cwd_outside_worktree(tmp_path):
    worktree = _worktree(tmp_path)
    shell = LocalShellAdapter(worktree, allowed_commands=[(sys.executable, "-c")])

    with pytest.raises(PathBoundaryError):
        shell.run((sys.executable, "-c", "print('ok')"), cwd=tmp_path)


def test_shell_adapter_env_isolation_hides_parent_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("JIRA_API_KEY", "secret-token")
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
    )

    result = shell.run(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ.get('JIRA_API_KEY', '<missing>'))",
        )
    )

    assert result.ok
    assert result.stdout == "<missing>\n"


def test_shell_adapter_env_isolation_sets_home_to_tmp(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
    )

    result = shell.run((sys.executable, "-c", "import os; print(os.environ['HOME'])"))

    assert result.ok
    assert result.stdout == "/tmp\n"


def test_shell_adapter_timeout_returns_timed_out_result(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
    )

    result = shell.run(
        (sys.executable, "-c", "import time; time.sleep(5)"),
        timeout_seconds=1,
    )

    assert not result.ok
    assert result.timed_out
    assert result.returncode == 124
    assert "timed out" in result.stderr

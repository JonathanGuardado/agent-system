from __future__ import annotations

import sys

import pytest

from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.domain.errors import CommandNotAllowedError, PathBoundaryError


def test_shell_adapter_runs_allowlisted_command_inside_worktree(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    shell = LocalShellAdapter(worktree, allowed_commands=[(sys.executable, "-c")])

    result = shell.run((sys.executable, "-c", "print('ok')"))

    assert result.ok
    assert result.stdout == "ok\n"
    assert result.stderr == ""


def test_shell_adapter_rejects_command_outside_allowlist(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    shell = LocalShellAdapter(worktree, allowed_commands=[(sys.executable, "-c")])

    with pytest.raises(CommandNotAllowedError):
        shell.run((sys.executable, "-m", "pytest"))


def test_shell_adapter_rejects_cwd_outside_worktree(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    shell = LocalShellAdapter(worktree, allowed_commands=[(sys.executable, "-c")])

    with pytest.raises(PathBoundaryError):
        shell.run((sys.executable, "-c", "print('ok')"), cwd=tmp_path)

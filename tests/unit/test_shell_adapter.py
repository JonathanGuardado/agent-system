from __future__ import annotations

import subprocess
import sys
import time

import pytest

from ticket_agent.adapters.local.sandbox import BubblewrapSandbox, NullSandbox
from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.domain.errors import CommandNotAllowedError, PathBoundaryError
from ticket_agent.domain.execution import CommandExecutionPolicy

_POLICY = CommandExecutionPolicy()


def _worktree(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    return worktree


def test_shell_adapter_runs_allowlisted_command_inside_worktree(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
        sandbox=NullSandbox(),
    )

    result = shell.run((sys.executable, "-c", "print('ok')"), policy=_POLICY)

    assert result.ok
    assert not result.timed_out
    assert result.stdout == "ok\n"
    assert result.stderr == ""


def test_shell_adapter_rejects_command_outside_allowlist(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
        sandbox=NullSandbox(),
    )

    with pytest.raises(CommandNotAllowedError):
        shell.run((sys.executable, "-m", "pytest"), policy=_POLICY)


def test_shell_adapter_rejects_denylisted_command_even_if_allowlisted(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[("curl",)],
        sandbox=NullSandbox(),
    )

    with pytest.raises(CommandNotAllowedError):
        shell.run(("curl", "https://example.com"), policy=_POLICY)


def test_shell_adapter_rejects_dangerous_argv_containing_docker(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
        sandbox=NullSandbox(),
    )

    with pytest.raises(CommandNotAllowedError):
        shell.run((sys.executable, "-c", "print('docker')"), policy=_POLICY)


def test_shell_adapter_rejects_cwd_outside_worktree(tmp_path):
    worktree = _worktree(tmp_path)
    shell = LocalShellAdapter(worktree, allowed_commands=[(sys.executable, "-c")], sandbox=NullSandbox())

    with pytest.raises(PathBoundaryError):
        shell.run(
            (sys.executable, "-c", "print('ok')"),
            cwd=tmp_path,
            policy=_POLICY,
        )


def test_shell_adapter_env_isolation_hides_parent_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("JIRA_API_KEY", "secret-token")
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
        sandbox=NullSandbox(),
    )

    result = shell.run(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ.get('JIRA_API_KEY', '<missing>'))",
        ),
        policy=_POLICY,
    )

    assert result.ok
    assert result.stdout == "<missing>\n"


def test_shell_adapter_env_isolation_sets_home_to_tmp(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
        sandbox=NullSandbox(),
    )

    result = shell.run(
        (sys.executable, "-c", "import os; print(os.environ['HOME'])"),
        policy=_POLICY,
    )

    assert result.ok
    assert result.stdout == "/tmp\n"


def test_shell_adapter_timeout_returns_timed_out_result(tmp_path):
    shell = LocalShellAdapter(
        _worktree(tmp_path),
        allowed_commands=[(sys.executable, "-c")],
        sandbox=NullSandbox(),
    )

    result = shell.run(
        (sys.executable, "-c", "import time; time.sleep(5)"),
        timeout_seconds=1,
        policy=_POLICY,
    )

    assert not result.ok
    assert result.timed_out
    assert result.returncode == 124
    assert "timed out" in result.stderr


def test_timeout_reaps_the_whole_process_tree(tmp_path):
    """Regression: subprocess.run(timeout=) kills only the direct child.

    `npm` and `pytest` spawn trees, so a gate that times out used to leave
    grandchildren running -- holding the worktree and burning CPU with nothing
    left to reap them. The fix is start_new_session plus killpg on the group.
    """

    marker = "918273"
    command = ("/bin/sh", "-c", f"/bin/sleep {marker} & sleep 60")
    adapter = LocalShellAdapter(tmp_path, [command], sandbox=NullSandbox())

    def survivors() -> list[str]:
        found = subprocess.run(
            ["pgrep", "-f", f"^/bin/sleep {marker}$"],
            capture_output=True,
            text=True,
        )
        return found.stdout.split()

    try:
        result = adapter.run(command, timeout_seconds=2, policy=_POLICY)
        time.sleep(1.0)

        assert result.timed_out is True
        assert result.returncode == 124
        assert survivors() == []
    finally:
        subprocess.run(
            ["pkill", "-f", f"^/bin/sleep {marker}$"], capture_output=True
        )


def test_failure_to_start_is_reported_not_raised(tmp_path):
    command = ("/nonexistent/binary", "--flag")
    adapter = LocalShellAdapter(tmp_path, [command], sandbox=NullSandbox())

    result = adapter.run(command, policy=_POLICY)

    assert result.returncode == 127
    assert "failed to start" in result.stderr


@pytest.mark.skipif(
    not BubblewrapSandbox.available(),
    reason="bwrap cannot create a user namespace on this host",
)
def test_real_wrapper_attestation_carries_complete_command_evidence(tmp_path):
    root = _worktree(tmp_path)
    nested = root / "packages" / "app"
    nested.mkdir(parents=True)
    shell = LocalShellAdapter(
        root,
        allowed_commands=[("/bin/true",)],
        sandbox=BubblewrapSandbox(),
    )
    policy = CommandExecutionPolicy(
        network="none",
        writable_paths=(".cache",),
    )

    result = shell.run(("/bin/true",), cwd="packages/app", policy=policy)

    assert result.ok
    attestation = result.sandbox_attestation
    assert attestation is not None
    assert attestation.sandbox_profile == "bwrap"
    assert attestation.command_policy_digest == policy.digest
    assert attestation.repository_root == str(root.resolve())
    assert attestation.command_working_directory == str(nested.resolve())
    assert attestation.network_mode == "none"
    assert attestation.writable_mounts == (str((root / ".cache").resolve()),)
    assert len(attestation.launch_digest) == 64


def test_shell_adapter_requires_an_explicit_sandbox(tmp_path):
    """A caller that wants no boundary must say so, visibly.

    The previous default silently handed unsupervised code the weakest
    option, so the omission is now a construction error rather than a
    passthrough.
    """

    with pytest.raises(TypeError, match="sandbox"):
        LocalShellAdapter(
            _worktree(tmp_path),
            allowed_commands=[(sys.executable, "-c")],
        )

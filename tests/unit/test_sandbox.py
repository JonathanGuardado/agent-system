"""Tests for execution isolation.

Isolation tests that only inspect argv prove nothing about isolation. The
ones that matter here execute a real command and check what it could reach,
so they are skipped rather than faked when bwrap is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import socket
import subprocess
import threading

import pytest

from ticket_agent.adapters.local.sandbox import (
    BubblewrapSandbox,
    NullSandbox,
    SandboxPolicy,
    SandboxUnavailableError,
    build_sandbox,
    is_enforcing_sandbox,
)

_SECRET_RE = re.compile(
    r"(TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|COOKIE|SESSION)"
    r"|^(GH|JIRA|SLACK|DEEPSEEK|GEMINI|AWS)_",
    re.IGNORECASE,
)

requires_sandbox = pytest.mark.skipif(
    not BubblewrapSandbox.available(),
    reason="bwrap cannot create a user namespace on this host",
)


def _run(argv, env=None, timeout=30):
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


@pytest.fixture()
def worktree(tmp_path):
    (tmp_path / "src.py").write_text("original")
    return tmp_path


# -- argv shape ------------------------------------------------------------


def test_network_is_unshared_for_gates_and_shared_only_for_install(worktree):
    sandbox = BubblewrapSandbox()

    gate = sandbox.wrap(
        ["/bin/true"], root=worktree, cwd=worktree, policy=SandboxPolicy()
    )
    install = sandbox.wrap(
        ["/bin/true"],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(network="install"),
    )

    assert "--unshare-all" in gate
    assert "--share-net" not in gate
    assert "--share-net" in install
    # Resolver files are bound only when the network is actually usable.
    assert "/etc/resolv.conf" in install
    assert "/etc/resolv.conf" not in gate


def test_environment_is_cleared_not_inherited(worktree):
    """--unshare-all does not imply a clean environment; measured, not assumed."""

    argv = BubblewrapSandbox().wrap(
        ["/bin/true"], root=worktree, cwd=worktree, policy=SandboxPolicy()
    )

    assert "--clearenv" in argv


def test_rlimits_are_applied_in_child_argv_not_preexec(worktree):
    """preexec_fn is not async-signal-safe and this process runs threads."""

    argv = BubblewrapSandbox().wrap(
        ["/bin/true"],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(cpu_seconds=7, memory_bytes=123456, max_processes=9),
    )

    assert "prlimit" in " ".join(argv)
    assert "--cpu=7" in argv
    assert "--as=123456" in argv
    assert "--nproc=9" in argv


def test_writable_path_escaping_the_worktree_is_refused(worktree):
    with pytest.raises(SandboxUnavailableError):
        BubblewrapSandbox().wrap(
            ["/bin/true"],
            root=worktree,
            cwd=worktree,
            policy=SandboxPolicy(writable_paths=(Path("/etc"),)),
        )


def test_null_sandbox_is_a_passthrough(worktree):
    assert NullSandbox().wrap(
        ["/bin/true"],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(),
    ) == (
        "/bin/true",
    )


# -- behavior --------------------------------------------------------------


@requires_sandbox
def test_worktree_is_read_only_with_explicit_writable_mounts(worktree):
    sandbox = BubblewrapSandbox()
    policy = SandboxPolicy(writable_paths=(Path("node_modules"),))

    argv = sandbox.wrap(
        [
            "/bin/sh",
            "-c",
            "echo mutated > src.py 2>/dev/null && echo WROTE_SOURCE || echo ro; "
            "echo dep > node_modules/a.js && echo WROTE_MOUNT",
        ],
        root=worktree,
        cwd=worktree,
        policy=policy,
    )
    result = _run(argv)

    assert "WROTE_SOURCE" not in result.stdout
    assert "ro" in result.stdout
    assert "WROTE_MOUNT" in result.stdout
    assert (worktree / "src.py").read_text() == "original"


@requires_sandbox
def test_credentials_do_not_reach_the_child(worktree):
    polluted = dict(
        os.environ,
        GH_TOKEN="ghp_pollutedtoken0123456789ABCDEF",
        DEEPSEEK_API_KEY="sk-polluted-value-0123456789",
        AWS_SECRET_ACCESS_KEY="polluted",
        SOME_SESSION_COOKIE="polluted",
    )
    argv = BubblewrapSandbox().wrap(
        ["/usr/bin/env"], root=worktree, cwd=worktree, policy=SandboxPolicy()
    )

    result = _run(argv, env=polluted)

    # Match the variable name only. Matching the whole line makes any value
    # that happens to contain "key" or "credentials" -- a tmpdir path, for
    # instance -- look like a leak.
    names = [line.split("=", 1)[0] for line in result.stdout.splitlines() if "=" in line]
    leaked = [name for name in names if _SECRET_RE.search(name)]
    assert leaked == []
    assert "PATH" in names, "the child still needs its allowlisted variables"


@requires_sandbox
def test_install_reaches_controlled_local_endpoint_but_gate_cannot(worktree):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    accepted: list[bool] = []

    def accept_once():
        connection, _ = server.accept()
        accepted.append(True)
        connection.close()

    thread = threading.Thread(target=accept_once, daemon=True)
    thread.start()
    script = (
        "import socket; "
        f"socket.create_connection(('127.0.0.1', {port}), 1).close(); "
        "print('connected')"
    )
    sandbox = BubblewrapSandbox()
    install = sandbox.wrap(
        ["/usr/bin/python3", "-c", script],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(network="install"),
    )
    gate = sandbox.wrap(
        ["/usr/bin/python3", "-c", script],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(network="none"),
    )

    try:
        assert _run(install).returncode == 0
        thread.join(timeout=2)
        assert accepted == [True]
        assert _run(gate).returncode != 0
    finally:
        server.close()


@requires_sandbox
def test_nested_working_directory_binds_repository_root(worktree):
    nested = worktree / "packages" / "app"
    nested.mkdir(parents=True)
    sibling = worktree / "shared.txt"
    sibling.write_text("visible", encoding="utf-8")

    argv = BubblewrapSandbox().wrap(
        ["/bin/cat", "../../shared.txt"],
        root=worktree,
        cwd=nested,
        policy=SandboxPolicy(),
    )

    root_bind = ("--ro-bind", str(worktree.resolve()), str(worktree.resolve()))
    assert any(tuple(argv[index : index + 3]) == root_bind for index in range(len(argv)))
    chdir_index = argv.index("--chdir")
    assert argv[chdir_index + 1] == str(nested.resolve())
    assert _run(argv).stdout == "visible"


@requires_sandbox
def test_host_secrets_are_not_reachable_from_inside(worktree):
    probes = ["/var/run/docker.sock", "/etc/shadow", str(Path.home() / ".ssh")]
    script = "; ".join(
        f'[ -e "{p}" ] && echo "REACHABLE {p}" || true' for p in probes
    )
    argv = BubblewrapSandbox().wrap(
        ["/bin/sh", "-c", script + "; echo done"],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(),
    )

    result = _run(argv)

    assert "REACHABLE" not in result.stdout
    assert "done" in result.stdout


@requires_sandbox
def test_memory_limit_is_enforced(worktree):
    argv = BubblewrapSandbox().wrap(
        ["/usr/bin/python3", "-c", "b = bytearray(512 * 1024 * 1024); print('ALLOCATED')"],
        root=worktree,
        cwd=worktree,
        policy=SandboxPolicy(memory_bytes=64 * 1024 * 1024),
    )

    result = _run(argv)

    assert "ALLOCATED" not in result.stdout


@requires_sandbox
def test_availability_probe_uses_the_same_binds_as_wrap():
    """A probe that binds less than wrap() reports working hosts as broken.

    /lib64 is a symlink into /usr on Debian-family systems, so a probe that
    binds only /usr and /bin cannot exec even /bin/true, and the host looks
    sandbox-incapable when it is not.
    """

    assert BubblewrapSandbox.available() is True
    assert BubblewrapSandbox.available("no-such-binary-anywhere") is False


def test_build_sandbox_refuses_to_degrade_when_isolation_is_required(monkeypatch):
    monkeypatch.setattr(BubblewrapSandbox, "available", staticmethod(lambda *_: False))

    with pytest.raises(SandboxUnavailableError):
        build_sandbox(required=True)

    assert isinstance(build_sandbox(required=False), NullSandbox)


def test_null_sandbox_is_not_an_enforcing_boundary():
    assert is_enforcing_sandbox(NullSandbox()) is False


def test_bubblewrap_sandbox_is_an_enforcing_boundary():
    assert is_enforcing_sandbox(BubblewrapSandbox()) is True


def test_enforcement_is_decided_by_the_wrapper_not_a_configured_string():
    """The preflight and the autonomy resolver must agree on one definition."""

    class _ClaimsToBeSandboxed:
        profile = "bwrap-ish"

        def wrap(self, argv, *, root, cwd, policy):
            del root, cwd, policy
            return tuple(argv)

    assert is_enforcing_sandbox(_ClaimsToBeSandboxed()) is False

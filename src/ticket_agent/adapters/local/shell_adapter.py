"""Allowlisted local shell command adapter."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import subprocess
from typing import Sequence

from ticket_agent.adapters.local.sandbox import (
    NullSandbox,
    Sandbox,
    SandboxPolicy,
)
from ticket_agent.domain.errors import CommandNotAllowedError, PathBoundaryError
from ticket_agent.domain.execution import (
    CommandExecutionPolicy,
    SandboxAttestation,
)
from ticket_agent.ports.tools import CommandResult

#: Seconds to wait for a signalled process group before escalating.
_KILL_GRACE_SECONDS = 5


_DENYLISTED_COMMAND_NAMES = frozenset(
    {
        "curl",
        "wget",
        "ssh",
        "scp",
        "nc",
        "netcat",
        "sudo",
        "su",
        "kill",
        "pkill",
        "chmod",
        "chown",
    }
)

_DANGEROUS_ARGV_VALUES = frozenset(
    {"rm", "docker", "kubectl", "/etc/", "/var/run/docker.sock"}
)


class LocalShellAdapter:
    """Run explicitly allowlisted commands inside a worktree boundary."""

    def __init__(
        self,
        worktree_root: str | Path,
        allowed_commands: Sequence[Sequence[str]],
        *,
        default_timeout_seconds: int = 300,
        sandbox: Sandbox | None = None,
    ) -> None:
        self._root = Path(worktree_root).resolve(strict=True)
        self._allowed_commands = tuple(
            _normalize_command(command) for command in allowed_commands
        )
        self._default_timeout_seconds = default_timeout_seconds
        # The allowlist and denylist stay as defense in depth. They are
        # command filtering, not a boundary -- the sandbox is the boundary.
        self._sandbox = sandbox or NullSandbox()

        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def sandbox_profile(self) -> str:
        return self._sandbox.profile

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: int | None = None,
        policy: CommandExecutionPolicy,
    ) -> CommandResult:
        normalized = _normalize_command(command)
        if _is_blocked_command(normalized):
            raise CommandNotAllowedError(normalized)
        if not self._is_allowed(normalized):
            raise CommandNotAllowedError(normalized)

        resolved_cwd = self._resolve_cwd(cwd)
        timeout = timeout_seconds or self._default_timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        sandbox_policy = SandboxPolicy.from_execution_policy(policy)
        launch = self._sandbox.wrap(
            normalized,
            root=self._root,
            cwd=resolved_cwd,
            policy=sandbox_policy,
        )
        attestation = _sandbox_attestation(
            sandbox_profile=self._sandbox.profile,
            policy=policy,
            repository_root=self._root,
            command_cwd=resolved_cwd,
            launch=launch,
        )
        return self._run_process(
            launch,
            reported_command=normalized,
            cwd=resolved_cwd,
            timeout=timeout,
            attestation=attestation,
        )

    def _run_process(
        self,
        launch: Sequence[str],
        *,
        reported_command: tuple[str, ...],
        cwd: Path,
        timeout: int,
        attestation: SandboxAttestation,
    ) -> CommandResult:
        """Run a command in its own process group and reap the whole tree.

        ``subprocess.run(timeout=...)`` kills only the direct child. ``npm``
        and ``pytest`` spawn trees, so a timeout there orphans grandchildren
        that keep running, holding the worktree and burning CPU. Starting a
        new session and signalling the process *group* is what actually
        reclaims them.

        Note this is deliberately not ``preexec_fn``: ``start_new_session``
        is implemented as a ``setsid`` call in CPython's C fork path and is
        safe with threads, whereas an arbitrary Python callable there is not.
        """

        try:
            process = subprocess.Popen(  # noqa: S603 - argv is allowlisted above
                list(launch),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_isolated_environment(),
                start_new_session=True,
            )
        except OSError as exc:
            return CommandResult(
                command=reported_command,
                returncode=127,
                stdout="",
                stderr=f"failed to start command: {exc}",
                timed_out=False,
                sandbox_attestation=attestation,
            )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._terminate_group(process)
            message = f"command timed out after {timeout} seconds"
            return CommandResult(
                command=reported_command,
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\n{message}".strip(),
                timed_out=True,
                sandbox_attestation=attestation,
            )

        return CommandResult(
            command=reported_command,
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=False,
            sandbox_attestation=attestation,
        )

    def _terminate_group(self, process: subprocess.Popen) -> tuple[str, str]:
        """SIGTERM the group, then SIGKILL anything still alive."""

        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(process.pid), signal_number)
            except (ProcessLookupError, PermissionError):
                break
            try:
                stdout, stderr = process.communicate(timeout=_KILL_GRACE_SECONDS)
                return _coerce_output(stdout), _coerce_output(stderr)
            except subprocess.TimeoutExpired:
                continue

        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return "", ""
        return _coerce_output(stdout), _coerce_output(stderr)

    def _is_allowed(self, command: tuple[str, ...]) -> bool:
        return any(command[: len(prefix)] == prefix for prefix in self._allowed_commands)

    def _resolve_cwd(self, cwd: str | Path | None) -> Path:
        if cwd is None:
            return self._root

        candidate = Path(cwd)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise PathBoundaryError(resolved, self._root) from exc
        return resolved


def _normalize_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        raise ValueError("command must be a non-empty sequence of non-empty strings")
    normalized = tuple(command)
    if not normalized or not all(isinstance(part, str) and part for part in normalized):
        raise ValueError("command must be a non-empty sequence of non-empty strings")
    return normalized


def _is_blocked_command(command: tuple[str, ...]) -> bool:
    command_name = Path(command[0]).name
    if command_name in _DENYLISTED_COMMAND_NAMES:
        return True
    if _contains_dangerous_argv_value(command):
        return True
    return False


def _contains_dangerous_argv_value(command: tuple[str, ...]) -> bool:
    for index, value in enumerate(command):
        if any(dangerous in value for dangerous in _DANGEROUS_ARGV_VALUES):
            return True
        if "chmod 777" in value:
            return True
        if (
            value == "chmod"
            and index + 1 < len(command)
            and command[index + 1] == "777"
        ):
            return True
        if value == "777" and index > 0 and command[index - 1] == "chmod":
            return True
    return False


def _isolated_environment() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/tmp",
    }
    if "VIRTUAL_ENV" in os.environ:
        env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
    return env


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _sandbox_attestation(
    *,
    sandbox_profile: str,
    policy: CommandExecutionPolicy,
    repository_root: Path,
    command_cwd: Path,
    launch: Sequence[str],
) -> SandboxAttestation:
    writable_mounts = tuple(
        str((repository_root / path).resolve()) for path in policy.writable_paths
    )
    launch_digest = hashlib.sha256(
        b"\0".join(part.encode("utf-8") for part in launch)
    ).hexdigest()
    return SandboxAttestation(
        sandbox_profile=sandbox_profile,
        command_policy_digest=policy.digest,
        repository_root=str(repository_root),
        command_working_directory=str(command_cwd),
        network_mode=policy.network,
        writable_mounts=writable_mounts,
        launch_digest=launch_digest,
    )

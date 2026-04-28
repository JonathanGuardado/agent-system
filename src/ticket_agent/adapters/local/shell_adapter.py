"""Allowlisted local shell command adapter."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Sequence

from ticket_agent.domain.errors import CommandNotAllowedError, PathBoundaryError
from ticket_agent.ports.tools import CommandResult


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
    ) -> None:
        self._root = Path(worktree_root).resolve(strict=True)
        self._allowed_commands = tuple(
            _normalize_command(command) for command in allowed_commands
        )
        self._default_timeout_seconds = default_timeout_seconds

        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")

    @property
    def root(self) -> Path:
        return self._root

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: int | None = None,
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

        try:
            completed = subprocess.run(
                normalized,
                cwd=resolved_cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_isolated_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_output(exc.stdout)
            stderr = _coerce_output(exc.stderr)
            message = f"command timed out after {timeout} seconds"
            return CommandResult(
                command=normalized,
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\n{message}".strip(),
                timed_out=True,
            )

        return CommandResult(
            command=normalized,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )

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

"""Allowlisted local shell command adapter."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence

from ticket_agent.domain.errors import CommandNotAllowedError, PathBoundaryError
from ticket_agent.ports.tools import CommandResult


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
        self._allowed_commands = tuple(tuple(command) for command in allowed_commands)
        self._default_timeout_seconds = default_timeout_seconds

        if any(not command for command in self._allowed_commands):
            raise ValueError("allowed command prefixes must not be empty")
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
            )

        return CommandResult(
            command=normalized,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _is_allowed(self, command: tuple[str, ...]) -> bool:
        return any(command[: len(prefix)] == prefix for prefix in self._allowed_commands)

    def _resolve_cwd(self, cwd: str | Path | None) -> Path:
        if cwd is None:
            return self._root

        candidate = Path(cwd)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise PathBoundaryError(resolved, self._root) from exc
        return resolved


def _normalize_command(command: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(command)
    if not normalized or not all(isinstance(part, str) and part for part in normalized):
        raise ValueError("command must be a non-empty sequence of non-empty strings")
    return normalized


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value

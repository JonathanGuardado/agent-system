"""Tool adapter boundary interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Structured result returned by shell-like adapters."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class FilePort(Protocol):
    """Worktree-scoped file operations."""

    @property
    def root(self) -> Path:
        """Resolved worktree root."""

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path and enforce the worktree boundary."""

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        """Read text from a worktree-scoped path."""

    def write_text(
        self,
        path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write text to a worktree-scoped path."""

    def exists(self, path: str | Path) -> bool:
        """Return whether a worktree-scoped path exists."""


class ShellPort(Protocol):
    """Command execution boundary."""

    @property
    def root(self) -> Path:
        """Resolved worktree root."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        """Run an allowlisted command inside the worktree boundary."""


class TestPort(Protocol):
    """Repository test execution boundary."""

    def run_tests(self, suite: str = "default") -> CommandResult:
        """Run a named test suite from the repo contract."""

    def run_lint(self) -> CommandResult | None:
        """Run the repo contract lint command when one is configured."""

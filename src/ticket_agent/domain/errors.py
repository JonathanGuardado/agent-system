"""Domain-specific exceptions."""

from __future__ import annotations

from pathlib import Path


class AgentSystemError(Exception):
    """Base exception for expected ticket-agent failures."""


class PathBoundaryError(AgentSystemError):
    """Raised when a requested path escapes the configured worktree boundary."""

    def __init__(self, path: str | Path, root: str | Path) -> None:
        super().__init__(f"path escapes worktree boundary: {path!s} is outside {root!s}")
        self.path = Path(path)
        self.root = Path(root)


class CommandNotAllowedError(AgentSystemError):
    """Raised when a shell command is not covered by the configured allowlist."""

    def __init__(self, command: tuple[str, ...]) -> None:
        rendered = " ".join(command) if command else "<empty>"
        super().__init__(f"command is not allowed: {rendered}")
        self.command = command


class RepoContractError(AgentSystemError):
    """Raised when a repo contract file is missing required structure."""


class TicketLockError(AgentSystemError):
    """Raised when ticket lock configuration or usage is invalid."""


class GitAdapterError(AgentSystemError):
    """Base exception for expected local git adapter failures."""


class WorktreeCreationError(GitAdapterError):
    """Raised when a git worktree cannot be created."""


class NoChangesToCommitError(GitAdapterError):
    """Raised when commit is requested but there are no staged changes."""


class PushError(GitAdapterError):
    """Raised when pushing a branch fails."""


class WorktreeCleanupError(GitAdapterError):
    """Raised when a git worktree cannot be removed."""

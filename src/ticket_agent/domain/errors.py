"""Domain-specific exceptions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ticket_agent.domain.model import ModelAttempt
from ticket_agent.domain.model_router import ModelAttemptFailure


class AgentSystemError(Exception):
    """Base exception for expected ticket-agent failures."""


class PathBoundaryError(AgentSystemError):
    """Raised when a requested path escapes the configured worktree boundary."""

    def __init__(self, path: str | Path, root: str | Path) -> None:
        super().__init__(f"path escapes worktree boundary: {path!s} is outside {root!s}")
        self.path = Path(path)
        self.root = Path(root)


class PolicyViolationError(AgentSystemError):
    """Raised when a local adapter operation violates repository policy."""

    def __init__(self, path: str | Path, reason: str) -> None:
        super().__init__(f"policy violation for {path!s}: {reason}")
        self.path = Path(path)
        self.reason = reason


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


class ModelRouterError(AgentSystemError):
    """Base exception for expected model router failures."""


class AllBackendsFailedError(ModelRouterError):
    """Raised when every selected provider backend fails."""

    def __init__(self, attempts: Sequence[ModelAttempt]) -> None:
        self.attempts = tuple(attempts)
        detail = "; ".join(
            f"{attempt.provider}/{attempt.model}: {attempt.error or 'unknown error'}"
            for attempt in self.attempts
        )
        super().__init__(f"all model backends failed: {detail}")


class ModelCallError(ModelRouterError):
    """Raised when one model call fails."""

    def __init__(self, model: str, error: str) -> None:
        super().__init__(f"{model}: {error}")
        self.model = model
        self.error = error


class AllModelsFailedError(ModelRouterError):
    """Raised when the primary and every fallback model call fail."""

    def __init__(self, failures: list[ModelAttemptFailure]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{failure.model}: {failure.error}" for failure in self.failures
        )
        super().__init__(f"all model attempts failed: {detail}")

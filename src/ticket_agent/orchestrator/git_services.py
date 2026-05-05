"""Git-backed pull request service implementations."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Protocol

from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.domain.errors import PullRequestCreationError
from ticket_agent.orchestrator.state import TicketState


class GitPullRequestPort(Protocol):
    def commit(self, worktree_path: str | Path, message: str) -> str: ...

    def push(self, worktree_path: str | Path, branch_name: str) -> None: ...


class GitWorktreeCleanupPort(Protocol):
    def cleanup_worktree(
        self,
        repo_path: str | Path,
        worktree_path: str | Path,
    ) -> None: ...


class PullRequestOpener(Protocol):
    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str: ...


class GitService:
    """Commit, push, and open a pull request for completed ticket work."""

    def __init__(
        self,
        *,
        git: GitPullRequestPort | None = None,
        pull_request_opener: PullRequestOpener | None = None,
        base_branch: str = "main",
    ) -> None:
        self._git = git or GitAdapter()
        self._pull_request_opener = pull_request_opener or GhPullRequestOpener()
        self._base_branch = base_branch

    async def open_pull_request(self, state: TicketState) -> str:
        if state.pull_request_url:
            return state.pull_request_url

        worktree_path = _required_worktree_path(state)
        branch_name = _required_branch_name(state)

        commit_message = _commit_message(state)
        self._git.commit(worktree_path, commit_message)
        self._git.push(worktree_path, branch_name)
        return self._pull_request_opener.open_pull_request(
            worktree_path=worktree_path,
            branch_name=branch_name,
            base_branch=self._base_branch,
            title=_pull_request_title(state),
            body=_pull_request_body(state),
        )


class WorktreeCleanupService:
    """Remove terminal ticket worktrees from the local repository."""

    def __init__(self, *, git: GitWorktreeCleanupPort | None = None) -> None:
        self._git = git or GitAdapter()

    def cleanup(self, state: TicketState) -> None:
        repo_path = _worktree_cleanup_repo_path(state)
        worktree_path = _worktree_path(state)
        if repo_path is None or worktree_path is None:
            return
        self._git.cleanup_worktree(repo_path, worktree_path)


class GhPullRequestOpener:
    """Open pull requests through the GitHub CLI."""

    def __init__(self, *, timeout_seconds: int = 300) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds

    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str:
        existing_url = self._existing_pull_request_url(
            worktree_path=worktree_path,
            branch_name=branch_name,
            base_branch=base_branch,
        )
        if existing_url is not None:
            return existing_url

        command = (
            "gh",
            "pr",
            "create",
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--title",
            title,
            "--body",
            body,
        )
        try:
            result = subprocess.run(
                command,
                cwd=worktree_path,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise PullRequestCreationError(
                f"gh pr create timed out after {self._timeout_seconds} seconds"
            ) from exc

        if result.returncode != 0:
            raise PullRequestCreationError(_subprocess_failure_message(result))

        url = result.stdout.strip()
        if not url:
            raise PullRequestCreationError("gh pr create did not return a PR URL")
        return url

    def _existing_pull_request_url(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
    ) -> str | None:
        command = (
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--json",
            "url",
            "--jq",
            ".[0].url",
        )
        try:
            result = subprocess.run(
                command,
                cwd=worktree_path,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None
        url = result.stdout.strip()
        if not url or url == "null":
            return None
        return url


def _worktree_path(state: TicketState) -> Path | None:
    if not state.worktree_path:
        return None
    return Path(state.worktree_path)


def _worktree_cleanup_repo_path(state: TicketState) -> Path | None:
    if not state.repo_path:
        return None
    return Path(state.repo_path)


def _required_worktree_path(state: TicketState) -> Path:
    worktree_path = _worktree_path(state)
    if worktree_path is None:
        raise PullRequestCreationError(
            "worktree_path is required to open pull request"
        )
    return worktree_path


def _required_branch_name(state: TicketState) -> str:
    if not state.branch_name:
        raise PullRequestCreationError(
            "branch_name is required to open pull request"
        )
    return state.branch_name


def _commit_message(state: TicketState) -> str:
    return f"{state.ticket_key}: {state.summary}"


def _pull_request_title(state: TicketState) -> str:
    return _commit_message(state)


def _pull_request_body(state: TicketState) -> str:
    parts = [
        f"Ticket: {state.ticket_key}",
        f"Summary: {state.summary}",
    ]
    if state.description:
        parts.extend(("", state.description))
    return "\n".join(parts)


def _subprocess_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    return output or f"gh pr create exited with return code {result.returncode}"


__all__ = [
    "GhPullRequestOpener",
    "GitPullRequestPort",
    "GitWorktreeCleanupPort",
    "GitService",
    "PullRequestOpener",
    "WorktreeCleanupService",
]

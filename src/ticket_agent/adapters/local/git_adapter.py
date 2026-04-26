"""Local git adapter."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence

from ticket_agent.domain.errors import (
    GitAdapterError,
    NoChangesToCommitError,
    PushError,
    WorktreeCleanupError,
    WorktreeCreationError,
)
from ticket_agent.domain.git import WorktreeInfo


class GitAdapter:
    """Run local git operations for ticket worktrees."""

    def __init__(self, *, default_timeout_seconds: int = 300) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        self._default_timeout_seconds = default_timeout_seconds

    def create_worktree(
        self,
        repo_path: str | Path,
        ticket_key: str,
        lock_id: str,
    ) -> WorktreeInfo:
        repo = Path(repo_path).resolve(strict=True)
        branch_name = f"agent/{ticket_key}/{lock_id[:8]}"
        worktree_path = repo.parent / ".worktrees" / ticket_key
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        result = self._run_git(
            ("worktree", "add", "-b", branch_name, str(worktree_path)),
            cwd=repo,
        )
        if result.returncode != 0:
            raise WorktreeCreationError(_failure_message(result))

        return WorktreeInfo(
            repo_path=repo,
            worktree_path=worktree_path,
            branch_name=branch_name,
            ticket_key=ticket_key,
            lock_id=lock_id,
        )

    def commit(self, worktree_path: str | Path, message: str) -> str:
        worktree = Path(worktree_path).resolve(strict=True)

        add_result = self._run_git(("add", "-A"), cwd=worktree)
        if add_result.returncode != 0:
            raise GitAdapterError(_failure_message(add_result))

        diff_result = self._run_git(("diff", "--cached", "--quiet"), cwd=worktree)
        if diff_result.returncode == 0:
            raise NoChangesToCommitError("no changes to commit")
        if diff_result.returncode != 1:
            raise GitAdapterError(_failure_message(diff_result))

        commit_result = self._run_git(("commit", "-m", message), cwd=worktree)
        if commit_result.returncode != 0:
            raise GitAdapterError(_failure_message(commit_result))

        sha_result = self._run_git(("rev-parse", "HEAD"), cwd=worktree)
        if sha_result.returncode != 0:
            raise GitAdapterError(_failure_message(sha_result))
        return sha_result.stdout.strip()

    def push(self, worktree_path: str | Path, branch_name: str) -> None:
        worktree = Path(worktree_path).resolve(strict=True)

        result = self._run_git(("push", "origin", branch_name), cwd=worktree)
        if result.returncode != 0:
            raise PushError(_failure_message(result))

    def cleanup_worktree(self, repo_path: str | Path, worktree_path: str | Path) -> None:
        repo = Path(repo_path).resolve(strict=True)

        result = self._run_git(
            ("worktree", "remove", "--force", str(worktree_path)),
            cwd=repo,
        )
        if result.returncode != 0:
            raise WorktreeCleanupError(_failure_message(result))

    def _run_git(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self._default_timeout_seconds,
        )


def _failure_message(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    return output or f"git exited with return code {result.returncode}"

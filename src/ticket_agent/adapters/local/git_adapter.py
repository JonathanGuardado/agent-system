"""Local git adapter."""

from __future__ import annotations

from pathlib import Path
import re
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


_SAFE_REF_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_PROTECTED_BRANCHES = frozenset({"main", "master", "develop"})


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
        short_lock_id: str,
    ) -> WorktreeInfo:
        repo = Path(repo_path).resolve(strict=True)
        _validate_safe_ref_component(ticket_key, "ticket_key")
        _validate_safe_ref_component(short_lock_id, "short_lock_id")

        branch_name = f"agent/{ticket_key}/{short_lock_id}"
        worktree_path = repo / ".worktrees" / ticket_key / short_lock_id
        try:
            _validate_worktree_path(repo, worktree_path)
        except WorktreeCleanupError as exc:
            raise WorktreeCreationError(str(exc)) from exc
        if worktree_path.exists() or worktree_path.is_symlink():
            raise WorktreeCreationError(f"worktree already exists: {worktree_path}")

        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        result = self._add_worktree(repo, worktree_path, branch_name)
        if result.returncode != 0:
            raise WorktreeCreationError(_failure_message(result))

        return WorktreeInfo(
            repo_path=repo,
            worktree_path=worktree_path,
            branch_name=branch_name,
            ticket_key=ticket_key,
            lock_id=short_lock_id,
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
        _validate_push_branch(branch_name)

        result = self._run_git(("push", "origin", branch_name), cwd=worktree)
        if result.returncode != 0:
            raise PushError(_failure_message(result))

    def cleanup_worktree(self, repo_path: str | Path, worktree_path: str | Path) -> None:
        repo = Path(repo_path).resolve(strict=True)
        resolved_worktree = Path(worktree_path).resolve(strict=False)
        _validate_worktree_path(repo, resolved_worktree)
        if not resolved_worktree.exists() and not resolved_worktree.is_symlink():
            return

        result = self._run_git(
            ("worktree", "remove", "--force", str(resolved_worktree)),
            cwd=repo,
        )
        if result.returncode != 0:
            raise WorktreeCleanupError(_failure_message(result))
        _remove_empty_parents(resolved_worktree, repo / ".worktrees")

    def _add_worktree(
        self,
        repo: Path,
        worktree_path: Path,
        branch_name: str,
    ) -> subprocess.CompletedProcess[str]:
        branch_result = self._run_git(
            ("rev-parse", "--verify", f"refs/heads/{branch_name}"),
            cwd=repo,
        )
        if branch_result.returncode == 0:
            return self._run_git(
                ("worktree", "add", str(worktree_path), branch_name),
                cwd=repo,
            )
        return self._run_git(
            ("worktree", "add", "-b", branch_name, str(worktree_path)),
            cwd=repo,
        )

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


def _validate_safe_ref_component(value: str, label: str) -> None:
    if _SAFE_REF_COMPONENT.fullmatch(value) is None:
        raise WorktreeCreationError(f"unsafe {label}: {value}")


def _validate_push_branch(branch_name: str) -> None:
    if branch_name in _PROTECTED_BRANCHES:
        raise PushError(f"refusing to push protected branch: {branch_name}")
    if not branch_name.startswith("agent/"):
        raise PushError(f"refusing to push non-agent branch: {branch_name}")

    parts = branch_name.split("/")
    if len(parts) != 3 or parts[0] != "agent":
        raise PushError(f"unsafe agent branch name: {branch_name}")
    for label, value in (("ticket_key", parts[1]), ("short_lock_id", parts[2])):
        if _SAFE_REF_COMPONENT.fullmatch(value) is None:
            raise PushError(f"unsafe {label} in branch name: {branch_name}")


def _validate_worktree_path(repo: Path, worktree_path: Path) -> None:
    worktrees_root = repo / ".worktrees"
    if worktrees_root.is_symlink():
        raise WorktreeCleanupError(
            f"worktrees root must not be a symlink: {worktrees_root}"
        )
    resolved_worktrees_root = worktrees_root.resolve(strict=False)
    resolved_worktree_path = worktree_path.resolve(strict=False)
    try:
        relative = resolved_worktree_path.relative_to(resolved_worktrees_root)
    except ValueError as exc:
        raise WorktreeCleanupError(
            f"worktree path is outside {worktrees_root}: {worktree_path}"
        ) from exc
    if not relative.parts:
        raise WorktreeCleanupError(f"refusing to remove worktrees root: {worktree_path}")


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    stop = stop_at.resolve(strict=False)
    current = path.parent.resolve(strict=False)
    while current != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent

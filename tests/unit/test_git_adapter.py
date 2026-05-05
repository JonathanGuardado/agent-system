from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.domain.errors import (
    NoChangesToCommitError,
    PushError,
    WorktreeCleanupError,
    WorktreeCreationError,
)


def test_worktree_creation_creates_expected_branch_and_path(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()

    info = adapter.create_worktree(repo, "ABC-123", "12345678")

    assert info.repo_path == repo.resolve()
    assert info.worktree_path == repo / ".worktrees" / "ABC-123" / "12345678"
    assert info.branch_name == "agent/ABC-123/12345678"
    assert info.ticket_key == "ABC-123"
    assert info.lock_id == "12345678"
    assert info.worktree_path.exists()
    assert _git(("branch", "--show-current"), cwd=info.worktree_path) == info.branch_name


def test_worktree_creation_rejects_unsafe_ticket_key_with_path_traversal(tmp_path):
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(WorktreeCreationError, match="unsafe ticket_key"):
        GitAdapter().create_worktree(repo, "../ABC-123", "12345678")


def test_worktree_creation_rejects_unsafe_short_lock_id(tmp_path):
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(WorktreeCreationError, match="unsafe short_lock_id"):
        GitAdapter().create_worktree(repo, "ABC-123", "../12345678")


def test_worktree_creation_isolates_worktrees_by_lock_id(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    first = adapter.create_worktree(repo, "ABC-123", "12345678")
    second = adapter.create_worktree(repo, "ABC-123", "abcdef12")

    assert first.worktree_path == repo / ".worktrees" / "ABC-123" / "12345678"
    assert second.worktree_path == repo / ".worktrees" / "ABC-123" / "abcdef12"


def test_worktree_creation_does_not_reuse_existing_worktree_for_same_lock(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    adapter.create_worktree(repo, "ABC-123", "12345678")

    with pytest.raises(WorktreeCreationError, match="already exists"):
        adapter.create_worktree(repo, "ABC-123", "12345678")


def test_commit_returns_valid_sha_when_files_changed(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    info = adapter.create_worktree(repo, "ABC-123", "12345678")
    (info.worktree_path / "feature.txt").write_text("hello\n", encoding="utf-8")

    sha = adapter.commit(info.worktree_path, "Add feature")

    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert _git(("rev-parse", "HEAD"), cwd=info.worktree_path) == sha


def test_commit_raises_no_changes_to_commit_error_when_clean(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    info = adapter.create_worktree(repo, "ABC-123", "12345678")

    with pytest.raises(NoChangesToCommitError):
        adapter.commit(info.worktree_path, "No changes")


@pytest.mark.parametrize("branch_name", ("main", "master", "develop"))
def test_push_rejects_protected_branches(tmp_path, branch_name):
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(PushError, match="protected branch"):
        GitAdapter().push(repo, branch_name)


def test_push_rejects_branch_outside_agent_namespace(tmp_path):
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(PushError, match="non-agent branch"):
        GitAdapter().push(repo, "feature/ABC-123")


def test_push_does_not_use_force(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    GitAdapter().push(repo, "agent/ABC-123/12345678")

    assert calls == [("git", "push", "origin", "agent/ABC-123/12345678")]
    assert "--force" not in calls[0]
    assert "-f" not in calls[0]


def test_cleanup_rejects_path_outside_repo_worktrees(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(WorktreeCleanupError, match="outside"):
        GitAdapter().cleanup_worktree(repo, outside)


def test_cleanup_removes_valid_worktree(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    info = adapter.create_worktree(repo, "ABC-123", "12345678")

    adapter.cleanup_worktree(repo, info.worktree_path)

    assert not info.worktree_path.exists()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(("init",), cwd=path)
    _git(("config", "user.email", "agent@example.com"), cwd=path)
    _git(("config", "user.name", "Ticket Agent"), cwd=path)
    (path / "README.md").write_text("# Test repo\n", encoding="utf-8")
    _git(("add", "README.md"), cwd=path)
    _git(("commit", "-m", "Initial commit"), cwd=path)
    return path


def _git(args: tuple[str, ...], *, cwd: Path) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

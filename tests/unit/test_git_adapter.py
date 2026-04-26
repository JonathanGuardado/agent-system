from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.domain.errors import NoChangesToCommitError


def test_worktree_creation_creates_expected_branch_and_path(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()

    info = adapter.create_worktree(repo, "ABC-123", "1234567890abcdef")

    assert info.repo_path == repo.resolve()
    assert info.worktree_path == tmp_path / ".worktrees" / "ABC-123"
    assert info.branch_name == "agent/ABC-123/12345678"
    assert info.ticket_key == "ABC-123"
    assert info.lock_id == "1234567890abcdef"
    assert info.worktree_path.exists()
    assert _git(("branch", "--show-current"), cwd=info.worktree_path) == info.branch_name


def test_commit_returns_valid_sha_when_files_changed(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    info = adapter.create_worktree(repo, "ABC-123", "1234567890abcdef")
    (info.worktree_path / "feature.txt").write_text("hello\n", encoding="utf-8")

    sha = adapter.commit(info.worktree_path, "Add feature")

    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert _git(("rev-parse", "HEAD"), cwd=info.worktree_path) == sha


def test_commit_raises_no_changes_to_commit_error_when_clean(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    info = adapter.create_worktree(repo, "ABC-123", "1234567890abcdef")

    with pytest.raises(NoChangesToCommitError):
        adapter.commit(info.worktree_path, "No changes")


def test_cleanup_removes_the_worktree(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    adapter = GitAdapter()
    info = adapter.create_worktree(repo, "ABC-123", "1234567890abcdef")

    adapter.cleanup_worktree(repo, info.worktree_path)

    assert not info.worktree_path.exists()


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

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
from typing import Any

import pytest

from ticket_agent.domain.errors import (
    NoChangesToCommitError,
    PullRequestCreationError,
    PushError,
)
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.local_services import GhPullRequestOpener, GitService
from ticket_agent.orchestrator.state import TicketState


def test_git_service_commits_pushes_opens_pr_and_returns_url(tmp_path):
    git = _FakeGit()
    opener = _FakePullRequestOpener("https://github.test/acme/repo/pull/7")
    service = GitService(
        git=git,
        pull_request_opener=opener,
        base_branch="develop",
    )
    state = _state(tmp_path, description="Detailed ticket notes")

    result = asyncio.run(service.open_pull_request(state))

    assert result == "https://github.test/acme/repo/pull/7"
    assert git.calls == [
        ("commit", tmp_path, "AGENT-123: Add concrete PR service"),
        ("push", tmp_path, "agent/AGENT-123/12345678"),
    ]
    assert opener.calls == [
        {
            "worktree_path": tmp_path,
            "branch_name": "agent/AGENT-123/12345678",
            "base_branch": "develop",
            "title": "AGENT-123: Add concrete PR service",
            "body": (
                "Ticket: AGENT-123\n"
                "Summary: Add concrete PR service\n"
                "\n"
                "Detailed ticket notes"
            ),
        }
    ]


def test_git_service_requires_worktree_path():
    service = GitService(git=_FakeGit(), pull_request_opener=_FakePullRequestOpener())

    with pytest.raises(PullRequestCreationError, match="worktree_path is required"):
        asyncio.run(
            service.open_pull_request(
                TicketState(
                    ticket_key="AGENT-123",
                    summary="Missing worktree",
                    branch_name="agent/AGENT-123/12345678",
                )
            )
        )


def test_git_service_requires_branch_name(tmp_path):
    git = _FakeGit()
    opener = _FakePullRequestOpener()
    service = GitService(git=git, pull_request_opener=opener)

    with pytest.raises(PullRequestCreationError, match="branch_name is required"):
        asyncio.run(
            service.open_pull_request(
                TicketState(
                    ticket_key="AGENT-123",
                    summary="Missing branch",
                    worktree_path=str(tmp_path),
                )
            )
        )

    assert git.calls == []
    assert opener.calls == []


def test_git_service_does_not_push_or_open_pr_when_commit_fails(tmp_path):
    git = _FakeGit(commit_error=NoChangesToCommitError("no changes to commit"))
    opener = _FakePullRequestOpener()
    service = GitService(git=git, pull_request_opener=opener)

    with pytest.raises(NoChangesToCommitError, match="no changes"):
        asyncio.run(service.open_pull_request(_state(tmp_path)))

    assert git.calls == [
        ("commit", tmp_path, "AGENT-123: Add concrete PR service"),
    ]
    assert opener.calls == []


def test_git_service_does_not_open_pr_when_push_fails(tmp_path):
    git = _FakeGit(push_error=PushError("remote rejected branch"))
    opener = _FakePullRequestOpener()
    service = GitService(git=git, pull_request_opener=opener)

    with pytest.raises(PushError, match="remote rejected"):
        asyncio.run(service.open_pull_request(_state(tmp_path)))

    assert git.calls == [
        ("commit", tmp_path, "AGENT-123: Add concrete PR service"),
        ("push", tmp_path, "agent/AGENT-123/12345678"),
    ]
    assert opener.calls == []


def test_ticket_node_runner_open_pull_request_stores_git_service_url(tmp_path):
    service = GitService(
        git=_FakeGit(),
        pull_request_opener=_FakePullRequestOpener(
            "https://github.test/acme/repo/pull/8"
        ),
    )
    runner = _runner(pull_request=service)
    initial_state = _state(tmp_path)

    state = initial_state.model_copy(
        update=asyncio.run(runner.open_pull_request(initial_state))
    )

    assert state.current_node == "open_pull_request"
    assert state.workflow_status == "opening_pull_request"
    assert state.pull_request_url == "https://github.test/acme/repo/pull/8"


def test_gh_pull_request_opener_runs_gh_pr_create_without_shell(tmp_path, monkeypatch):
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(command, 0, "https://github.test/pr/9\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = GhPullRequestOpener().open_pull_request(
        worktree_path=tmp_path,
        branch_name="agent/AGENT-123/12345678",
        base_branch="main",
        title="AGENT-123: Add service",
        body="Ticket: AGENT-123",
    )

    assert result == "https://github.test/pr/9"
    assert calls == [
        (
            (
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                "agent/AGENT-123/12345678",
                "--title",
                "AGENT-123: Add service",
                "--body",
                "Ticket: AGENT-123",
            ),
            {
                "cwd": tmp_path,
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 300,
            },
        )
    ]


def test_gh_pull_request_opener_raises_when_gh_times_out(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(PullRequestCreationError, match="timed out after 1 seconds"):
        GhPullRequestOpener(timeout_seconds=1).open_pull_request(
            worktree_path=tmp_path,
            branch_name="agent/AGENT-123/12345678",
            base_branch="main",
            title="AGENT-123: Add service",
            body="Ticket: AGENT-123",
        )


def test_gh_pull_request_opener_raises_when_gh_fails(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "authentication required")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(PullRequestCreationError, match="authentication required"):
        GhPullRequestOpener().open_pull_request(
            worktree_path=tmp_path,
            branch_name="agent/AGENT-123/12345678",
            base_branch="main",
            title="AGENT-123: Add service",
            body="Ticket: AGENT-123",
        )


def _state(
    worktree_path: Path,
    *,
    description: str = "",
) -> TicketState:
    return TicketState(
        ticket_key="AGENT-123",
        summary="Add concrete PR service",
        description=description,
        worktree_path=str(worktree_path),
        branch_name="agent/AGENT-123/12345678",
    )


def _runner(**overrides: Any) -> TicketNodeRunner:
    defaults = {
        "planner": _Planner(),
        "approval": _Approval(),
        "implementation": _Implementation(),
        "tests": _Tests(),
        "review": _Review(),
        "pull_request": _PullRequest(),
        "escalation": _Escalation(),
    }
    defaults.update(overrides)
    return TicketNodeRunner(**defaults)


class _FakeGit:
    def __init__(
        self,
        *,
        commit_error: Exception | None = None,
        push_error: Exception | None = None,
    ) -> None:
        self._commit_error = commit_error
        self._push_error = push_error
        self.calls: list[tuple[str, Path, str]] = []

    def commit(self, worktree_path: str | Path, message: str) -> str:
        self.calls.append(("commit", Path(worktree_path), message))
        if self._commit_error is not None:
            raise self._commit_error
        return "abc123"

    def push(self, worktree_path: str | Path, branch_name: str) -> None:
        self.calls.append(("push", Path(worktree_path), branch_name))
        if self._push_error is not None:
            raise self._push_error


class _FakePullRequestOpener:
    def __init__(self, url: str = "https://github.test/acme/repo/pull/1") -> None:
        self._url = url
        self.calls: list[dict[str, Any]] = []

    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str:
        self.calls.append(
            {
                "worktree_path": worktree_path,
                "branch_name": branch_name,
                "base_branch": base_branch,
                "title": title,
                "body": body,
            }
        )
        return self._url


class _Planner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {}


class _Approval:
    async def request_approval(self, state: TicketState) -> bool:
        return True


class _Implementation:
    async def implement(self, state: TicketState) -> dict[str, Any]:
        return {}


class _Tests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed"}


class _Review:
    async def review(self, state: TicketState) -> dict[str, Any]:
        return {"status": "accepted"}


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/acme/repo/pull/1"


class _Escalation:
    async def escalate(self, state: TicketState, reason: str) -> None:
        return None

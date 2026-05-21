from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticket_agent.feedback.github import (
    FeedbackExecutionCoordinator,
    FeedbackItem,
    GitHubFeedbackPoller,
)
from ticket_agent.orchestrator.runner import TicketWorkItem
from ticket_agent.orchestrator.state import TicketState


def test_feedback_poller_enqueues_unseen_items_once():
    item = _feedback_item()
    queue: asyncio.Queue[FeedbackItem] = asyncio.Queue()
    store = _Store()
    poller = GitHubFeedbackPoller(
        client=_Client([item]),
        store=store,
        queue=queue,
        poll_interval_seconds=1,
    )

    assert asyncio.run(poller.poll_once()) == 1
    assert asyncio.run(poller.poll_once()) == 0

    assert queue.get_nowait() == item
    assert queue.empty()
    assert store.seen(item.fingerprint)


def test_feedback_execution_runs_existing_pr_branch(tmp_path):
    item = _feedback_item(repo_path=str(tmp_path), branch_name="agent/AGENT-123/pr")
    loader = _Loader(
        TicketWorkItem(
            ticket_key="AGENT-123",
            summary="Implement feedback",
            description="Original ticket",
            repository="repo",
            repo_path=str(tmp_path),
        )
    )
    runner = _Runner()
    worktree_factory = _WorktreeFactory(tmp_path / "wt")
    cleaner = _Cleaner()
    coordinator = FeedbackExecutionCoordinator(
        loader=loader,
        runner=runner,
        worktree_factory=worktree_factory,
        worktree_cleaner=cleaner,
    )

    asyncio.run(coordinator.run_feedback(item))

    assert worktree_factory.calls == [
        (str(tmp_path), "AGENT-123", "agent/AGENT-123/pr", item.fingerprint[:8])
    ]
    assert runner.work_items[0].branch_name == "agent/AGENT-123/pr"
    assert runner.work_items[0].worktree_path == str(tmp_path / "wt")
    assert "Pull request feedback to address" in runner.work_items[0].description
    assert "Please change the CTA copy" in runner.work_items[0].description
    assert cleaner.cleaned[0].worktree_path == str(tmp_path / "wt")


def _feedback_item(
    *,
    repo_path: str = "/repo",
    branch_name: str = "agent/AGENT-123/abc",
) -> FeedbackItem:
    return FeedbackItem(
        ticket_key="AGENT-123",
        repo_path=repo_path,
        branch_name=branch_name,
        pull_request_url="https://github.test/acme/repo/pull/1",
        feedback_text="Please change the CTA copy.",
        fingerprint="feedfacecafebeef",
    )


class _Client:
    def __init__(self, items: list[FeedbackItem]) -> None:
        self.items = items

    def find_feedback(self) -> list[FeedbackItem]:
        return list(self.items)


class _Store:
    def __init__(self) -> None:
        self.items: set[str] = set()

    def seen(self, fingerprint: str) -> bool:
        return fingerprint in self.items

    def mark_seen(self, fingerprint: str) -> None:
        self.items.add(fingerprint)


class _Loader:
    def __init__(self, work_item: TicketWorkItem) -> None:
        self.work_item = work_item

    async def load(self, ticket_key: str) -> TicketWorkItem:
        return self.work_item


class _Runner:
    def __init__(self) -> None:
        self.work_items: list[TicketWorkItem] = []

    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        self.work_items.append(work_item)
        return TicketState(ticket_key=work_item.ticket_key, summary=work_item.summary)


@dataclass(frozen=True)
class _Worktree:
    worktree_path: Path


class _WorktreeFactory:
    def __init__(self, worktree_path: Path) -> None:
        self.worktree_path = worktree_path
        self.calls: list[tuple[str, str, str, str]] = []

    def create_worktree_for_branch(
        self,
        repo_path: str | Path,
        ticket_key: str,
        branch_name: str,
        short_lock_id: str,
    ) -> _Worktree:
        self.calls.append((str(repo_path), ticket_key, branch_name, short_lock_id))
        return _Worktree(self.worktree_path)


class _Cleaner:
    def __init__(self) -> None:
        self.cleaned: list[TicketState] = []

    def cleanup(self, state: TicketState) -> Any:
        self.cleaned.append(state)

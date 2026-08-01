from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticket_agent.feedback.github import (
    GitHubMergedDeliveryPoller,
    FeedbackExecutionCoordinator,
    FeedbackItem,
    GitHubFeedbackPoller,
    MergedDeliveryItem,
    SequentialDeliveryAdvancer,
)
from ticket_agent.jira.constants import LABEL_AI_READY
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket
from ticket_agent.orchestrator.runner import TicketWorkItem
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.adapters.local.sandbox import SandboxUnavailableError
import pytest


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


def test_feedback_preflight_refuses_before_worktree_creation(tmp_path):
    item = _feedback_item(repo_path=str(tmp_path))
    loader = _Loader(
        TicketWorkItem(
            ticket_key="AGENT-123",
            summary="Feedback",
            description="Original",
            repository="repo",
        )
    )
    worktree_factory = _WorktreeFactory(tmp_path / "wt")
    coordinator = FeedbackExecutionCoordinator(
        loader=loader,
        runner=_Runner(),
        worktree_factory=worktree_factory,
        execution_preflight=_FailingPreflight(),
    )

    with pytest.raises(SandboxUnavailableError, match="sandbox unavailable"):
        asyncio.run(coordinator.run_feedback(item))

    assert worktree_factory.calls == []


def test_merged_delivery_releases_only_next_queued_ticket(tmp_path):
    jira = FakeJiraClient(
        [
            JiraTicket(
                key="LAB-31",
                summary="First",
                status="Done",
                labels=["ai-ready", "ai-sequence-lab-30", "ai-step-0001"],
            ),
            JiraTicket(
                key="LAB-32",
                summary="Second",
                status="To Do",
                labels=["ai-sequence-lab-30", "ai-step-0002"],
            ),
            JiraTicket(
                key="LAB-33",
                summary="Third",
                status="To Do",
                labels=["ai-sequence-lab-30", "ai-step-0003"],
            ),
        ]
    )
    item = _merged_item(repo_path=str(tmp_path), ticket_key="LAB-31")
    poller = GitHubMergedDeliveryPoller(
        client=_MergedClient([item]),
        store=_Store(),
        advancer=SequentialDeliveryAdvancer(jira),
        promotion_opener=_PromotionOpener(),
        poll_interval_seconds=1,
    )

    assert asyncio.run(poller.poll_once()) == 1
    assert LABEL_AI_READY in jira.ticket("LAB-32").labels
    assert LABEL_AI_READY not in jira.ticket("LAB-33").labels
    assert asyncio.run(poller.poll_once()) == 0


def test_last_merged_delivery_step_opens_cumulative_promotion_pr(tmp_path):
    jira = FakeJiraClient(
        JiraTicket(
            key="LAB-33",
            summary="Last",
            status="Done",
            labels=["ai-ready", "ai-sequence-lab-30", "ai-step-0003"],
        )
    )
    opener = _PromotionOpener()
    poller = GitHubMergedDeliveryPoller(
        client=_MergedClient(
            [_merged_item(repo_path=str(tmp_path), ticket_key="LAB-33")]
        ),
        store=_Store(),
        advancer=SequentialDeliveryAdvancer(jira),
        promotion_opener=opener,
        promotion_base_branch="main",
        poll_interval_seconds=1,
    )

    assert asyncio.run(poller.poll_once()) == 1
    assert opener.calls[0]["branch_name"] == "integration/lab-30"
    assert opener.calls[0]["base_branch"] == "main"


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


def _merged_item(*, repo_path: str, ticket_key: str) -> MergedDeliveryItem:
    return MergedDeliveryItem(
        ticket_key=ticket_key,
        repo_path=repo_path,
        integration_branch="integration/lab-30",
        pull_request_url=f"https://github.test/acme/repo/pull/{ticket_key}",
        fingerprint=f"merged-{ticket_key}",
    )


class _Client:
    def __init__(self, items: list[FeedbackItem]) -> None:
        self.items = items

    def find_feedback(self) -> list[FeedbackItem]:
        return list(self.items)


class _MergedClient:
    def __init__(self, items: list[MergedDeliveryItem]) -> None:
        self.items = items

    def find_merged_delivery_items(self) -> list[MergedDeliveryItem]:
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


class _PromotionOpener:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def open_pull_request(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "https://github.test/acme/repo/pull/promotion"


class _FailingPreflight:
    def check(self):
        raise SandboxUnavailableError("sandbox unavailable")

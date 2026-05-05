from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.jira.constants import (
    EVENT_JIRA_EXECUTION_COMPLETED,
    EVENT_JIRA_EXECUTION_FAILED,
    EVENT_JIRA_EXECUTION_FAILURE_REPORT_FAILED,
    EVENT_JIRA_EXECUTION_IN_REVIEW,
    EVENT_JIRA_EXECUTION_RELEASED_WITHOUT_PR,
    EVENT_JIRA_EXECUTION_SLACK_NOTIFICATION_FAILED,
    EVENT_JIRA_EXECUTION_STARTED,
    EVENT_JIRA_EXECUTION_WORKTREE_CLEANUP_FAILED,
)
from ticket_agent.jira.execution_coordinator import JiraExecutionCoordinator
from ticket_agent.jira.models import JiraExecutionError
from ticket_agent.orchestrator.runner import TicketClaimFailedError, TicketWorkItem
from ticket_agent.orchestrator.state import TicketState


def test_run_ticket_success_with_pr_marks_in_review_and_returns_state():
    work_item = _work_item()
    final_state = _state(pull_request_url="https://github.com/example/repo/pull/1")
    loader = _FakeLoader(work_item)
    execution_service = _FakeExecutionService()
    runner = _FakeRunner(final_state)
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert result is final_state
    assert loader.calls == ["AGENT-123"]
    assert execution_service.calls == [
        (
            "mark_in_review",
            "AGENT-123",
            "https://github.com/example/repo/pull/1",
        ),
    ]
    assert runner.calls == [work_item]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_IN_REVIEW,
        EVENT_JIRA_EXECUTION_COMPLETED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_IN_REVIEW] == {
        "ticket_key": "AGENT-123",
        "pull_request_url": "https://github.com/example/repo/pull/1",
    }


def test_run_ticket_success_without_pr_releases_claim_and_returns_state():
    work_item = _work_item()
    final_state = _state()
    loader = _FakeLoader(work_item)
    execution_service = _FakeExecutionService()
    runner = _FakeRunner(final_state)
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert result is final_state
    assert execution_service.calls == [
        ("mark_released", "AGENT-123", None),
    ]
    assert runner.calls == [work_item]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_RELEASED_WITHOUT_PR,
        EVENT_JIRA_EXECUTION_COMPLETED,
    ]


def test_run_ticket_success_with_pr_posts_slack_and_cleans_terminal_resources():
    work_item = _work_item(slack_channel="C-EXEC", slack_thread_ts="thread-1")
    final_state = _state(
        pull_request_url="https://github.com/example/repo/pull/1",
        repo_path="/repos/agent-system",
        worktree_path="/repos/agent-system/.worktrees/AGENT-123/12345678",
        slack_channel="C-EXEC",
        slack_thread_ts="thread-1",
    )
    execution_service = _FakeExecutionService()
    slack = _FakeSlack()
    cleaner = _FakeCleaner()
    checkpointer = _FakeCheckpointer()
    coordinator = JiraExecutionCoordinator(
        _FakeLoader(work_item),
        execution_service,
        _FakeRunner(final_state),
        slack=slack,
        worktree_cleaner=cleaner,
        checkpointer=checkpointer,
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert result is final_state
    assert execution_service.calls == [
        (
            "mark_in_review",
            "AGENT-123",
            "https://github.com/example/repo/pull/1",
        ),
    ]
    assert slack.messages == [
        (
            "C-EXEC",
            "thread-1",
            "ticket-agent",
            "AI execution opened a pull request for AGENT-123:\n\n"
            "https://github.com/example/repo/pull/1",
        )
    ]
    assert cleaner.states == [final_state]
    assert checkpointer.deleted_threads == ["AGENT-123"]


def test_escalated_final_state_marks_failed_posts_slack_and_cleans_worktree():
    work_item = _work_item(slack_thread_ts="thread-1")
    final_state = _state(
        workflow_status="escalated",
        escalation_reason="no changes to commit",
        repo_path="/repos/agent-system",
        worktree_path="/repos/agent-system/.worktrees/AGENT-123/12345678",
        slack_thread_ts="thread-1",
    )
    execution_service = _FakeExecutionService()
    slack = _FakeSlack()
    cleaner = _FakeCleaner()
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        _FakeLoader(work_item),
        execution_service,
        _FakeRunner(final_state),
        emit=events,
        slack=slack,
        worktree_cleaner=cleaner,
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert result is final_state
    assert execution_service.calls == [
        ("mark_failed", "AGENT-123", "no changes to commit")
    ]
    assert slack.messages == [
        (
            None,
            "thread-1",
            "ticket-agent",
            "AI execution escalated AGENT-123:\n\nno changes to commit",
        )
    ]
    assert cleaner.states == [final_state]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]


def test_slack_notification_failure_does_not_corrupt_jira_finalization():
    final_state = _state(
        pull_request_url="https://github.com/example/repo/pull/1",
        slack_thread_ts="thread-1",
    )
    execution_service = _FakeExecutionService()
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        _FakeLoader(_work_item(slack_thread_ts="thread-1")),
        execution_service,
        _FakeRunner(final_state),
        emit=events,
        slack=_FakeSlack(error=RuntimeError("slack exploded")),
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert result is final_state
    assert execution_service.calls == [
        (
            "mark_in_review",
            "AGENT-123",
            "https://github.com/example/repo/pull/1",
        ),
    ]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_SLACK_NOTIFICATION_FAILED,
        EVENT_JIRA_EXECUTION_IN_REVIEW,
        EVENT_JIRA_EXECUTION_COMPLETED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_SLACK_NOTIFICATION_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "slack exploded",
    }


def test_worktree_cleanup_failure_is_reported_without_masking_success():
    final_state = _state(
        pull_request_url="https://github.com/example/repo/pull/1",
        repo_path="/repos/agent-system",
        worktree_path="/repos/agent-system/.worktrees/AGENT-123/12345678",
    )
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        _FakeLoader(_work_item()),
        _FakeExecutionService(),
        _FakeRunner(final_state),
        emit=events,
        worktree_cleaner=_FakeCleaner(error=RuntimeError("cleanup exploded")),
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert result is final_state
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_IN_REVIEW,
        EVENT_JIRA_EXECUTION_WORKTREE_CLEANUP_FAILED,
        EVENT_JIRA_EXECUTION_COMPLETED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_WORKTREE_CLEANUP_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "cleanup exploded",
    }


def test_worktree_cleanup_runs_when_final_jira_update_fails():
    error = JiraExecutionError("review update exploded")
    final_state = _state(
        pull_request_url="https://github.com/example/repo/pull/1",
        repo_path="/repos/agent-system",
        worktree_path="/repos/agent-system/.worktrees/AGENT-123/12345678",
    )
    cleaner = _FakeCleaner()
    coordinator = JiraExecutionCoordinator(
        _FakeLoader(_work_item()),
        _FakeExecutionService(fail_on={"mark_in_review": error}),
        _FakeRunner(final_state),
        worktree_cleaner=cleaner,
    )

    with pytest.raises(JiraExecutionError):
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert cleaner.states == [final_state]


def test_loader_failure_does_not_claim_or_run_and_reraises_original_error():
    error = RuntimeError("load exploded")
    loader = _FakeLoader(error)
    execution_service = _FakeExecutionService()
    runner = _FakeRunner(_state())
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert exc_info.value is error
    assert loader.calls == ["AGENT-123"]
    assert execution_service.calls == []
    assert runner.calls == []
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "load exploded",
    }


def test_runner_claim_failure_does_not_mark_failed():
    error = TicketClaimFailedError("AGENT-123", JiraExecutionError("claim exploded"))
    loader = _FakeLoader(_work_item())
    execution_service = _FakeExecutionService()
    runner = _FakeRunner(error)
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(TicketClaimFailedError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert exc_info.value is error
    assert execution_service.calls == []
    assert runner.calls == [_work_item()]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "ticket claim failed for AGENT-123: claim exploded",
    }


def test_runner_failure_marks_failed_and_reraises_original_error():
    error = RuntimeError("runner exploded")
    loader = _FakeLoader(_work_item())
    execution_service = _FakeExecutionService()
    runner = _FakeRunner(error)
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert exc_info.value is error
    assert execution_service.calls == [
        ("mark_failed", "AGENT-123", "runner exploded"),
    ]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "runner exploded",
    }


def test_runner_failure_plus_mark_failed_failure_reraises_runner_error():
    runner_error = RuntimeError("runner exploded")
    mark_failed_error = JiraExecutionError("mark failed exploded")
    loader = _FakeLoader(_work_item())
    execution_service = _FakeExecutionService(
        fail_on={"mark_failed": mark_failed_error}
    )
    runner = _FakeRunner(runner_error)
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert exc_info.value is runner_error
    assert execution_service.calls == [
        ("mark_failed", "AGENT-123", "runner exploded"),
    ]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILURE_REPORT_FAILED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_FAILURE_REPORT_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "mark failed exploded",
    }


def test_runner_success_but_mark_in_review_failure_raises_jira_error():
    error = JiraExecutionError("review update exploded")
    loader = _FakeLoader(_work_item())
    execution_service = _FakeExecutionService(fail_on={"mark_in_review": error})
    runner = _FakeRunner(
        _state(pull_request_url="https://github.com/example/repo/pull/1")
    )
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert exc_info.value is error
    assert execution_service.calls == [
        (
            "mark_in_review",
            "AGENT-123",
            "https://github.com/example/repo/pull/1",
        ),
        ("mark_failed", "AGENT-123", "review update exploded"),
    ]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "review update exploded",
    }


def test_runner_success_without_pr_but_mark_released_failure_raises_jira_error():
    error = JiraExecutionError("release exploded")
    loader = _FakeLoader(_work_item())
    execution_service = _FakeExecutionService(fail_on={"mark_released": error})
    runner = _FakeRunner(_state())
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert exc_info.value is error
    assert execution_service.calls == [
        ("mark_released", "AGENT-123", None),
        ("mark_failed", "AGENT-123", "release exploded"),
    ]
    assert events.names == [
        EVENT_JIRA_EXECUTION_STARTED,
        EVENT_JIRA_EXECUTION_FAILED,
    ]
    assert events.payloads[EVENT_JIRA_EXECUTION_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "release exploded",
    }


def test_runner_cancelled_error_propagates_without_marking_failed():
    loader = _FakeLoader(_work_item())
    execution_service = _FakeExecutionService()
    runner = _FakeRunner(asyncio.CancelledError())
    events = _EventRecorder()
    coordinator = JiraExecutionCoordinator(
        loader,
        execution_service,
        runner,
        emit=events,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    assert execution_service.calls == []
    assert events.names == [EVENT_JIRA_EXECUTION_STARTED]


def _work_item(**updates: Any) -> TicketWorkItem:
    values = {
        "ticket_key": "AGENT-123",
        "summary": "Implement Jira execution",
        "description": "Wire Jira to the runner.",
        "repository": "agent-system",
    }
    values.update(updates)
    return TicketWorkItem(**values)


def _state(**updates: Any) -> TicketState:
    values = {
        "ticket_key": "AGENT-123",
        "summary": "Implement Jira execution",
        "description": "Wire Jira to the runner.",
        "repository": "agent-system",
        "workflow_status": "completed",
    }
    values.update(updates)
    return TicketState(**values)


class _FakeLoader:
    def __init__(self, result: TicketWorkItem | BaseException) -> None:
        self.result = result
        self.calls: list[str] = []

    async def load(self, ticket_key: str) -> TicketWorkItem:
        self.calls.append(ticket_key)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeExecutionService:
    def __init__(self, fail_on: dict[str, BaseException] | None = None) -> None:
        self.fail_on = {} if fail_on is None else fail_on
        self.calls: list[tuple[str, str, object]] = []

    async def mark_claimed(self, ticket_key: str) -> None:
        self.calls.append(("mark_claimed", ticket_key, None))
        self._raise_if_configured("mark_claimed")

    async def mark_failed(self, ticket_key: str, reason: str) -> None:
        self.calls.append(("mark_failed", ticket_key, reason))
        self._raise_if_configured("mark_failed")

    async def mark_in_review(self, ticket_key: str, pull_request_url: str) -> None:
        self.calls.append(("mark_in_review", ticket_key, pull_request_url))
        self._raise_if_configured("mark_in_review")

    async def mark_released(self, ticket_key: str) -> None:
        self.calls.append(("mark_released", ticket_key, None))
        self._raise_if_configured("mark_released")

    def _raise_if_configured(self, method_name: str) -> None:
        exc = self.fail_on.get(method_name)
        if exc is not None:
            raise exc


class _FakeRunner:
    def __init__(self, result: TicketState | BaseException) -> None:
        self.result = result
        self.calls: list[TicketWorkItem] = []

    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        self.calls.append(work_item)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeSlack:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.messages: list[tuple[str | None, str, str, str]] = []

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.messages.append((channel, thread_ts, user_id, text))


class _FakeCleaner:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.states: list[TicketState] = []

    def cleanup(self, state: TicketState) -> None:
        self.states.append(state)
        if self.error is not None:
            raise self.error


class _FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(self, event_name: str, payload: dict[str, object]) -> None:
        self.events.append((event_name, payload))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    @property
    def payloads(self) -> dict[str, dict[str, object]]:
        return {name: payload for name, payload in self.events}

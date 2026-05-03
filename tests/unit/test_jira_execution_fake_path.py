from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    LABEL_AI_READY,
    STATUS_IN_PROGRESS,
    STATUS_IN_REVIEW,
    STATUS_TODO,
)
from ticket_agent.jira.execution_coordinator import JiraExecutionCoordinator
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraExecutionError, JiraTicket
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.orchestrator.runner import TicketClaimFailedError, TicketWorkItem
from ticket_agent.orchestrator.state import TicketState


def test_fake_jira_transition_ticket_uses_status_names():
    client = FakeJiraClient(_ticket())

    asyncio.run(client.transition_ticket("AGENT-123", STATUS_IN_PROGRESS))
    asyncio.run(client.transition_ticket("AGENT-123", STATUS_TODO))

    assert client.ticket("AGENT-123").status == STATUS_TODO
    assert client.calls == [
        ("transition_ticket", "AGENT-123", STATUS_IN_PROGRESS),
        ("transition_ticket", "AGENT-123", STATUS_TODO),
    ]


def test_fake_jira_execution_path_with_pr_marks_in_review_and_comments():
    pull_request_url = "https://github.com/example/agent-system/pull/12"
    client = FakeJiraClient(_ticket())
    coordinator = _coordinator(
        client,
        _FakeRunner(_state(pull_request_url=pull_request_url)),
    )

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    ticket = client.ticket("AGENT-123")
    assert result.pull_request_url == pull_request_url
    assert ticket.status == STATUS_IN_REVIEW
    assert LABEL_AI_CLAIMED not in ticket.labels
    assert LABEL_AI_FAILED not in ticket.labels
    assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments_for("AGENT-123") == [
        f"AI execution opened pull request:\n\n{pull_request_url}"
    ]


def test_fake_jira_execution_path_without_pr_releases_claim():
    client = FakeJiraClient(_ticket())
    coordinator = _coordinator(client, _FakeRunner(_state()))

    result = asyncio.run(coordinator.run_ticket("AGENT-123"))

    ticket = client.ticket("AGENT-123")
    assert result.pull_request_url is None
    assert ticket.status == STATUS_TODO
    assert ticket.labels == [LABEL_AI_READY]
    assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments_for("AGENT-123") == []


def test_fake_jira_execution_path_runner_failure_marks_failed_and_clears_claim():
    error = RuntimeError("runner exploded")
    client = FakeJiraClient(_ticket())
    runner = _FakeRunner(error)
    coordinator = _coordinator(client, runner)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    ticket = client.ticket("AGENT-123")
    assert exc_info.value is error
    assert ticket.status == STATUS_TODO
    assert ticket.labels == [LABEL_AI_READY, LABEL_AI_FAILED]
    assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments_for("AGENT-123") == [
        "AI execution failed:\n\nrunner exploded"
    ]


def test_fake_jira_execution_path_runner_claim_failure_does_not_mark_failed():
    error = TicketClaimFailedError(
        "AGENT-123",
        JiraExecutionError("claim exploded"),
    )
    client = FakeJiraClient(_ticket())
    runner = _FakeRunner(error)
    coordinator = _coordinator(client, runner)

    with pytest.raises(TicketClaimFailedError) as exc_info:
        asyncio.run(coordinator.run_ticket("AGENT-123"))

    ticket = client.ticket("AGENT-123")
    assert exc_info.value is error
    assert runner.calls != []
    assert ticket.status == STATUS_TODO
    assert ticket.labels == [LABEL_AI_READY]
    assert FIELD_AGENT_ASSIGNED_COMPONENT not in ticket.fields
    assert client.comments_for("AGENT-123") == []


def _coordinator(
    client: FakeJiraClient,
    runner: _FakeRunner,
) -> JiraExecutionCoordinator:
    loader = JiraWorkItemLoader(client)
    execution_service = JiraExecutionService(client, component_id="runner-1")
    return JiraExecutionCoordinator(loader, execution_service, runner)


def _ticket(**updates: Any) -> JiraTicket:
    values = {
        "key": "AGENT-123",
        "summary": "Implement Jira execution",
        "description": "Wire execution state to Jira.",
        "status": STATUS_TODO,
        "labels": [LABEL_AI_READY],
        "assignee": None,
        "fields": {
            FIELD_REPOSITORY: "agent-system",
            FIELD_REPO_PATH: "/repos/agent-system",
        },
    }
    values.update(updates)
    return JiraTicket(**values)


def _state(**updates: Any) -> TicketState:
    values = {
        "ticket_key": "AGENT-123",
        "summary": "Implement Jira execution",
        "description": "Wire execution state to Jira.",
        "repository": "agent-system",
        "workflow_status": "completed",
    }
    values.update(updates)
    return TicketState(**values)


class _FakeRunner:
    def __init__(self, result: TicketState | BaseException) -> None:
        self.result = result
        self.calls: list[TicketWorkItem] = []

    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        self.calls.append(work_item)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

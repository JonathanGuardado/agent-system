from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    LABEL_AI_READY,
    STATUS_IN_PROGRESS,
    STATUS_IN_REVIEW,
    STATUS_TODO,
)
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.models import JiraTicket


def test_mark_claimed_updates_jira_execution_state():
    client = _FakeJiraClient(_ticket(labels=[LABEL_AI_READY]))
    service = JiraExecutionService(client, component_id="runner-1")

    asyncio.run(service.mark_claimed("AGENT-123"))

    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        ("transition_ticket", "AGENT-123", STATUS_IN_PROGRESS),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
    ]
    assert client.ticket.status == STATUS_IN_PROGRESS
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == []


def test_mark_failed_releases_claim_marks_failure_and_comments():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        )
    )
    service = JiraExecutionService(client, component_id="runner-1")

    asyncio.run(service.mark_failed("AGENT-123", "tests failed"))

    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_FAILED]),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
        ("add_comment", "AGENT-123", "AI execution failed:\n\ntests failed"),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_FAILED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments == ["AI execution failed:\n\ntests failed"]


def test_mark_in_review_transitions_releases_claim_and_comments_with_pr():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        )
    )
    service = JiraExecutionService(client, component_id="runner-1")
    pull_request_url = "https://github.com/example/agent-system/pull/12"

    asyncio.run(service.mark_in_review("AGENT-123", pull_request_url))

    assert client.calls == [
        ("transition_ticket", "AGENT-123", STATUS_IN_REVIEW),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
        (
            "add_comment",
            "AGENT-123",
            f"AI execution opened pull request:\n\n{pull_request_url}",
        ),
    ]
    assert client.ticket.status == STATUS_IN_REVIEW
    assert client.ticket.labels == [LABEL_AI_READY]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments == [
        f"AI execution opened pull request:\n\n{pull_request_url}"
    ]


def test_mark_released_clears_claim_without_status_change():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        )
    )
    service = JiraExecutionService(client, component_id="runner-1")

    asyncio.run(service.mark_released("AGENT-123"))

    assert client.calls == [
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments == []


def _ticket(
    *,
    labels: list[str],
    fields: dict[str, Any] | None = None,
) -> JiraTicket:
    return JiraTicket(
        key="AGENT-123",
        summary="Implement Jira execution",
        description="Wire execution state to Jira.",
        status=STATUS_TODO,
        labels=labels,
        assignee=None,
        fields={} if fields is None else fields,
    )


class _FakeJiraClient:
    def __init__(self, ticket: JiraTicket) -> None:
        self.ticket = ticket
        self.calls: list[tuple[str, str, object]] = []
        self.comments: list[str] = []

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        self.calls.append(("get_ticket", ticket_key, None))
        return self.ticket

    async def transition_ticket(self, ticket_key: str, status: str) -> None:
        self.calls.append(("transition_ticket", ticket_key, status))
        self.ticket.status = status

    async def add_labels(self, ticket_key: str, labels: list[str]) -> None:
        self.calls.append(("add_labels", ticket_key, list(labels)))
        for label in labels:
            if label not in self.ticket.labels:
                self.ticket.labels.append(label)

    async def remove_labels(self, ticket_key: str, labels: list[str]) -> None:
        self.calls.append(("remove_labels", ticket_key, list(labels)))
        self.ticket.labels = [
            label for label in self.ticket.labels if label not in labels
        ]

    async def update_fields(self, ticket_key: str, fields: dict[str, object]) -> None:
        self.calls.append(("update_fields", ticket_key, dict(fields)))
        self.ticket.fields.update(fields)

    async def add_comment(self, ticket_key: str, body: str) -> None:
        self.calls.append(("add_comment", ticket_key, body))
        self.comments.append(body)

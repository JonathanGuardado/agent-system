from __future__ import annotations

import asyncio
from typing import Any

import pytest

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
from ticket_agent.jira.models import JiraExecutionError, JiraTicket


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


def test_mark_claimed_wraps_transition_failure_and_leaves_partial_state():
    client = _FakeJiraClient(
        _ticket(labels=[LABEL_AI_READY]),
        fail_on={"transition_ticket": RuntimeError("transition exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_claimed("AGENT-123"))

    assert str(exc_info.value) == (
        "mark_claimed failed for AGENT-123: transition exploded"
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        ("transition_ticket", "AGENT-123", STATUS_IN_PROGRESS),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert FIELD_AGENT_ASSIGNED_COMPONENT not in client.ticket.fields
    assert client.comments == []


def test_mark_failed_wraps_remove_label_failure_and_leaves_partial_state():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"remove_labels": RuntimeError("remove exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_failed("AGENT-123", "tests failed"))

    assert str(exc_info.value) == "mark_failed failed for AGENT-123: remove exploded"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_FAILED]),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [
        LABEL_AI_READY,
        LABEL_AI_CLAIMED,
        LABEL_AI_FAILED,
    ]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == []


def test_mark_in_review_wraps_remove_label_failure_and_leaves_partial_state():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"remove_labels": RuntimeError("remove exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")
    pull_request_url = "https://github.com/example/agent-system/pull/12"

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_in_review("AGENT-123", pull_request_url))

    assert str(exc_info.value) == (
        "mark_in_review failed for AGENT-123: remove exploded"
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("transition_ticket", "AGENT-123", STATUS_IN_REVIEW),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
    ]
    assert client.ticket.status == STATUS_IN_REVIEW
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == []


def test_mark_released_wraps_remove_label_failure_and_leaves_partial_state():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"remove_labels": RuntimeError("remove exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_released("AGENT-123"))

    assert str(exc_info.value) == (
        "mark_released failed for AGENT-123: remove exploded"
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == []


def test_mark_released_wraps_update_fields_failure_and_leaves_partial_state():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"update_fields": RuntimeError("update exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_released("AGENT-123"))

    assert str(exc_info.value) == (
        "mark_released failed for AGENT-123: update exploded"
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
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
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == []


def test_client_cancellation_is_not_wrapped():
    client = _FakeJiraClient(
        _ticket(labels=[LABEL_AI_READY]),
        fail_on={"add_labels": asyncio.CancelledError()},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.mark_claimed("AGENT-123"))

    assert client.calls == [("add_labels", "AGENT-123", [LABEL_AI_CLAIMED])]


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
    def __init__(
        self,
        ticket: JiraTicket,
        *,
        fail_on: dict[str, BaseException] | None = None,
    ) -> None:
        self.ticket = ticket
        self.fail_on = {} if fail_on is None else fail_on
        self.calls: list[tuple[str, str, object]] = []
        self.comments: list[str] = []

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        self.calls.append(("get_ticket", ticket_key, None))
        return self.ticket

    async def transition_ticket(self, ticket_key: str, status: str) -> None:
        self.calls.append(("transition_ticket", ticket_key, status))
        self._raise_if_configured("transition_ticket")
        self.ticket.status = status

    async def add_labels(self, ticket_key: str, labels: list[str]) -> None:
        self.calls.append(("add_labels", ticket_key, list(labels)))
        self._raise_if_configured("add_labels")
        for label in labels:
            if label not in self.ticket.labels:
                self.ticket.labels.append(label)

    async def remove_labels(self, ticket_key: str, labels: list[str]) -> None:
        self.calls.append(("remove_labels", ticket_key, list(labels)))
        self._raise_if_configured("remove_labels")
        self.ticket.labels = [
            label for label in self.ticket.labels if label not in labels
        ]

    async def update_fields(self, ticket_key: str, fields: dict[str, object]) -> None:
        self.calls.append(("update_fields", ticket_key, dict(fields)))
        self._raise_if_configured("update_fields")
        self.ticket.fields.update(fields)

    async def add_comment(self, ticket_key: str, body: str) -> None:
        self.calls.append(("add_comment", ticket_key, body))
        self._raise_if_configured("add_comment")
        self.comments.append(body)

    def _raise_if_configured(self, method_name: str) -> None:
        exc = self.fail_on.get(method_name)
        if exc is not None:
            raise exc

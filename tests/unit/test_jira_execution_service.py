from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.jira.constants import (
    EVENT_JIRA_COMPENSATION_FAILED,
    EVENT_JIRA_COMPENSATION_STARTED,
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


def test_mark_claimed_transition_failure_compensates_by_removing_claim():
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


def test_mark_claimed_update_fields_failure_compensates_claim_and_component():
    client = _FakeJiraClient(
        _ticket(labels=[LABEL_AI_READY]),
        fail_on={"update_fields": [RuntimeError("update exploded")]},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_claimed("AGENT-123"))

    assert str(exc_info.value) == "mark_claimed failed for AGENT-123: update exploded"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        ("transition_ticket", "AGENT-123", STATUS_IN_PROGRESS),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
    ]
    assert client.ticket.status == STATUS_IN_PROGRESS
    assert client.ticket.labels == [LABEL_AI_READY]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments == []


def test_mark_claimed_compensation_failure_emits_and_preserves_original_error():
    client = _FakeJiraClient(
        _ticket(labels=[LABEL_AI_READY]),
        fail_on={
            "transition_ticket": RuntimeError("transition exploded"),
            "remove_labels": RuntimeError("remove exploded"),
        },
    )
    events = _EventRecorder()
    service = JiraExecutionService(client, component_id="runner-1", emit=events)

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_claimed("AGENT-123"))

    assert str(exc_info.value) == (
        "mark_claimed failed for AGENT-123: transition exploded"
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        ("transition_ticket", "AGENT-123", STATUS_IN_PROGRESS),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
    ]
    assert events.names == [
        EVENT_JIRA_COMPENSATION_STARTED,
        EVENT_JIRA_COMPENSATION_FAILED,
    ]
    assert events.events[1] == (
        EVENT_JIRA_COMPENSATION_FAILED,
        {
            "ticket_key": "AGENT-123",
            "operation": "mark_claimed",
            "failed_step": "transition_ticket",
            "compensation_action": "remove_claimed_label",
            "error": "remove exploded",
        },
    )
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None


def test_mark_failed_remove_labels_failure_keeps_failed_label_and_raises():
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
        (
            "add_comment",
            "AGENT-123",
            _partial_failure_comment(
                "mark_failed",
                "remove_labels",
                "mark_failed failed for AGENT-123: remove exploded",
            ),
        ),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [
        LABEL_AI_READY,
        LABEL_AI_CLAIMED,
        LABEL_AI_FAILED,
    ]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == [
        _partial_failure_comment(
            "mark_failed",
            "remove_labels",
            "mark_failed failed for AGENT-123: remove exploded",
        )
    ]


def test_mark_failed_cleanup_failure_attempts_partial_failure_comment():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"update_fields": [RuntimeError("update exploded")]},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_failed("AGENT-123", "tests failed"))

    assert str(exc_info.value) == "mark_failed failed for AGENT-123: update exploded"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.calls == [
        ("add_labels", "AGENT-123", [LABEL_AI_FAILED]),
        ("remove_labels", "AGENT-123", [LABEL_AI_CLAIMED]),
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
        (
            "add_comment",
            "AGENT-123",
            _partial_failure_comment(
                "mark_failed",
                "update_fields",
                "mark_failed failed for AGENT-123: update exploded",
            ),
        ),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_FAILED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == [
        _partial_failure_comment(
            "mark_failed",
            "update_fields",
            "mark_failed failed for AGENT-123: update exploded",
        )
    ]


def test_mark_failed_final_comment_failure_raises_but_keeps_failed_state():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"add_comment": RuntimeError("comment exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_failed("AGENT-123", "tests failed"))

    assert str(exc_info.value) == "mark_failed failed for AGENT-123: comment exploded"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
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
    assert client.comments == []


def test_mark_in_review_cleanup_failure_keeps_in_review_and_comments():
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
        (
            "add_comment",
            "AGENT-123",
            _partial_failure_comment(
                "mark_in_review",
                "remove_labels",
                "mark_in_review failed for AGENT-123: remove exploded",
            ),
        ),
    ]
    assert client.ticket.status == STATUS_IN_REVIEW
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == [
        _partial_failure_comment(
            "mark_in_review",
            "remove_labels",
            "mark_in_review failed for AGENT-123: remove exploded",
        )
    ]


def test_mark_in_review_final_pr_comment_failure_keeps_status_and_cleanup():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"add_comment": RuntimeError("comment exploded")},
    )
    service = JiraExecutionService(client, component_id="runner-1")
    pull_request_url = "https://github.com/example/agent-system/pull/12"

    with pytest.raises(JiraExecutionError) as exc_info:
        asyncio.run(service.mark_in_review("AGENT-123", pull_request_url))

    assert str(exc_info.value) == (
        "mark_in_review failed for AGENT-123: comment exploded"
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
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
    assert client.comments == []


def test_mark_released_update_fields_failure_attempts_partial_failure_comment():
    client = _FakeJiraClient(
        _ticket(
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        ),
        fail_on={"update_fields": [RuntimeError("update exploded")]},
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
        (
            "add_comment",
            "AGENT-123",
            _partial_failure_comment(
                "mark_released",
                "update_fields",
                "mark_released failed for AGENT-123: update exploded",
            ),
        ),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] == "runner-1"
    assert client.comments == [
        _partial_failure_comment(
            "mark_released",
            "update_fields",
            "mark_released failed for AGENT-123: update exploded",
        )
    ]


def test_mark_released_remove_labels_failure_clears_component_and_comments():
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
        (
            "update_fields",
            "AGENT-123",
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        ),
        (
            "add_comment",
            "AGENT-123",
            _partial_failure_comment(
                "mark_released",
                "remove_labels",
                "mark_released failed for AGENT-123: remove exploded",
            ),
        ),
    ]
    assert client.ticket.status == STATUS_TODO
    assert client.ticket.labels == [LABEL_AI_READY, LABEL_AI_CLAIMED]
    assert client.ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
    assert client.comments == [
        _partial_failure_comment(
            "mark_released",
            "remove_labels",
            "mark_released failed for AGENT-123: remove exploded",
        )
    ]


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


def _partial_failure_comment(operation: str, failed_step: str, error: str) -> str:
    return (
        f"AI execution Jira update partially failed during {operation}.\n\n"
        f"Failed step: {failed_step}\n"
        f"Error: {error}"
    )


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(self, event_name: str, payload: dict[str, object]) -> None:
        self.events.append((event_name, payload))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class _FakeJiraClient:
    def __init__(
        self,
        ticket: JiraTicket,
        *,
        fail_on: dict[str, BaseException | list[BaseException]] | None = None,
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
        configured = self.fail_on.get(method_name)
        if isinstance(configured, list):
            if configured:
                raise configured.pop(0)
            return
        if configured is not None:
            raise configured

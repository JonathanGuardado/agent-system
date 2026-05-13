from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.jira.constants import (
    FIELD_MAX_ATTEMPTS,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    FIELD_SLACK_CHANNEL,
    FIELD_SLACK_THREAD_TS,
)
from ticket_agent.jira.models import JiraTicket, JiraWorkItemLoadError
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.orchestrator.runner import TicketWorkItem


def test_load_converts_jira_ticket_to_work_item():
    ticket = _ticket(
        fields={
            FIELD_REPOSITORY: "agent-system",
            FIELD_REPO_PATH: "/repos/agent-system",
            FIELD_MAX_ATTEMPTS: 5,
            FIELD_SLACK_CHANNEL: "C-EXEC",
            FIELD_SLACK_THREAD_TS: "thread-1",
        }
    )
    client = _FakeJiraClient(ticket)
    loader = JiraWorkItemLoader(client)

    work_item = asyncio.run(loader.load("AGENT-123"))

    assert client.calls == [("get_ticket", "AGENT-123")]
    assert work_item == TicketWorkItem(
        ticket_key="AGENT-123",
        summary="Implement Jira execution",
        description="Wire execution state to Jira.",
        repository="agent-system",
        repo_path="/repos/agent-system",
        slack_channel="C-EXEC",
        slack_thread_ts="thread-1",
        max_attempts=5,
    )


def test_load_uses_work_item_max_attempts_default_when_field_is_absent():
    ticket = _ticket(
        fields={
            FIELD_REPOSITORY: "agent-system",
            FIELD_REPO_PATH: "/repos/agent-system",
        }
    )
    client = _FakeJiraClient(ticket)
    loader = JiraWorkItemLoader(client)

    work_item = asyncio.run(loader.load("AGENT-123"))

    assert work_item.max_attempts == 3


def test_load_recovers_repo_context_from_description():
    ticket = _ticket(
        description=(
            "Execution context:\n"
            "- Jira project: SCRUM\n"
            "- Repository: agent-system\n"
            "- Repository path: /repos/agent-system\n\n"
            "Scope:\nBuild the validation app."
        ),
        fields={},
    )
    loader = JiraWorkItemLoader(_FakeJiraClient(ticket))

    work_item = asyncio.run(loader.load("AGENT-123"))

    assert work_item.repository == "agent-system"
    assert work_item.repo_path == "/repos/agent-system"


def test_load_recovers_repository_from_scoped_summary_and_path_from_default():
    ticket = _ticket(
        summary="[agent-system] Implement validation app",
        fields={},
    )
    loader = JiraWorkItemLoader(
        _FakeJiraClient(ticket),
        repo_defaults={
            FIELD_REPOSITORY: "agent-system",
            FIELD_REPO_PATH: "/repos/agent-system",
        },
    )

    work_item = asyncio.run(loader.load("AGENT-123"))

    assert work_item.repository == "agent-system"
    assert work_item.repo_path == "/repos/agent-system"


def test_load_uses_single_repo_default_when_ticket_has_no_repo_metadata():
    ticket = _ticket(fields={})
    loader = JiraWorkItemLoader(
        _FakeJiraClient(ticket),
        repo_defaults={
            FIELD_REPOSITORY: "agent-system",
            FIELD_REPO_PATH: "/repos/agent-system",
        },
    )

    work_item = asyncio.run(loader.load("AGENT-123"))

    assert work_item.repository == "agent-system"
    assert work_item.repo_path == "/repos/agent-system"


@pytest.mark.parametrize("missing_field", [FIELD_REPOSITORY, FIELD_REPO_PATH])
def test_load_raises_clear_error_when_required_field_is_missing(missing_field: str):
    fields = {
        FIELD_REPOSITORY: "agent-system",
        FIELD_REPO_PATH: "/repos/agent-system",
    }
    del fields[missing_field]
    loader = JiraWorkItemLoader(_FakeJiraClient(_ticket(fields=fields)))

    with pytest.raises(JiraWorkItemLoadError) as exc_info:
        asyncio.run(loader.load("AGENT-123"))

    assert str(exc_info.value) == (
        f"Jira ticket AGENT-123 is missing required field: {missing_field}"
    )


@pytest.mark.parametrize("blank_field", [FIELD_REPOSITORY, FIELD_REPO_PATH])
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_load_raises_clear_error_when_required_field_is_blank(
    blank_field: str,
    blank_value: str,
):
    fields = {
        FIELD_REPOSITORY: "agent-system",
        FIELD_REPO_PATH: "/repos/agent-system",
    }
    fields[blank_field] = blank_value
    loader = JiraWorkItemLoader(_FakeJiraClient(_ticket(fields=fields)))

    with pytest.raises(JiraWorkItemLoadError) as exc_info:
        asyncio.run(loader.load("AGENT-123"))

    assert str(exc_info.value) == (
        f"Jira ticket AGENT-123 is missing required field: {blank_field}"
    )


@pytest.mark.parametrize(
    "bad_value",
    ["", "0", "5", 1.5, True, 0, -1, "not-a-number"],
)
def test_load_raises_clear_error_when_max_attempts_is_invalid(bad_value: object):
    loader = JiraWorkItemLoader(
        _FakeJiraClient(
            _ticket(
                fields={
                    FIELD_REPOSITORY: "agent-system",
                    FIELD_REPO_PATH: "/repos/agent-system",
                    FIELD_MAX_ATTEMPTS: bad_value,
                }
            )
        )
    )

    with pytest.raises(JiraWorkItemLoadError) as exc_info:
        asyncio.run(loader.load("AGENT-123"))

    assert str(exc_info.value) == (
        f"Jira ticket AGENT-123 has invalid field: {FIELD_MAX_ATTEMPTS}"
    )


def _ticket(fields: dict[str, Any], **updates: Any) -> JiraTicket:
    values = dict(
        key="AGENT-123",
        summary="Implement Jira execution",
        description="Wire execution state to Jira.",
        status="To Do",
        labels=["ai-ready"],
        assignee=None,
        fields=fields,
    )
    values.update(updates)
    return JiraTicket(**values)


class _FakeJiraClient:
    def __init__(self, ticket: JiraTicket) -> None:
        self.ticket = ticket
        self.calls: list[tuple[str, str]] = []

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        self.calls.append(("get_ticket", ticket_key))
        return self.ticket

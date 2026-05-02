from __future__ import annotations

import asyncio
import logging

import pytest

from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraExecutionError, JiraTicket
from ticket_agent.orchestrator.jira_services import (
    JiraEscalationService,
    JiraLabelApprovalService,
)
from ticket_agent.orchestrator.state import TicketState


def test_escalate_calls_mark_failed_with_ticket_key_and_reason():
    execution_service = _FakeExecutionService()
    service = JiraEscalationService(execution_service)
    state = _state("AGENT-99")

    asyncio.run(service.escalate(state, "tests failed after 3 attempts"))

    assert execution_service.mark_failed_calls == [
        ("AGENT-99", "tests failed after 3 attempts")
    ]


def test_escalate_jira_error_is_logged_and_swallowed(caplog):
    execution_service = _FakeExecutionService(
        mark_failed_error=JiraExecutionError("jira unavailable")
    )
    service = JiraEscalationService(execution_service)
    state = _state("AGENT-42")

    with caplog.at_level(logging.WARNING):
        asyncio.run(service.escalate(state, "review rejected"))

    assert execution_service.mark_failed_calls == [("AGENT-42", "review rejected")]
    assert any("jira_escalation_update_failed" in r.message for r in caplog.records)
    assert any("AGENT-42" in r.message for r in caplog.records)


def test_escalate_unexpected_error_bubbles_up():
    execution_service = _FakeExecutionService(
        mark_failed_error=RuntimeError("unexpected crash")
    )
    service = JiraEscalationService(execution_service)
    state = _state("AGENT-7")

    with pytest.raises(RuntimeError, match="unexpected crash"):
        asyncio.run(service.escalate(state, "reason"))


def test_escalate_uses_injected_logger(caplog):
    logger = logging.getLogger("test.jira_escalation")
    execution_service = _FakeExecutionService(
        mark_failed_error=JiraExecutionError("timeout")
    )
    service = JiraEscalationService(execution_service, logger=logger)
    state = _state("AGENT-5")

    with caplog.at_level(logging.WARNING, logger="test.jira_escalation"):
        asyncio.run(service.escalate(state, "reason"))

    assert any(r.name == "test.jira_escalation" for r in caplog.records)


# ---------------------------------------------------------------------------
# JiraLabelApprovalService
# ---------------------------------------------------------------------------


def test_approval_returns_true_when_label_present():
    ticket = JiraTicket(
        key="AGENT-10",
        summary="Add feature",
        labels=["ai-ready", "ai-execution-approved"],
    )
    client = FakeJiraClient(ticket)
    service = JiraLabelApprovalService(client)

    result = asyncio.run(service.request_approval(_state("AGENT-10")))

    assert result is True
    assert ("get_ticket", "AGENT-10", None) in client.calls


def test_approval_returns_false_when_label_absent():
    ticket = JiraTicket(key="AGENT-11", summary="Add feature", labels=["ai-ready"])
    client = FakeJiraClient(ticket)
    service = JiraLabelApprovalService(client)

    result = asyncio.run(service.request_approval(_state("AGENT-11")))

    assert result is False


def test_approval_returns_false_when_no_labels():
    ticket = JiraTicket(key="AGENT-12", summary="Add feature")
    client = FakeJiraClient(ticket)
    service = JiraLabelApprovalService(client)

    result = asyncio.run(service.request_approval(_state("AGENT-12")))

    assert result is False


def test_approval_jira_error_bubbles_up():
    ticket = JiraTicket(key="AGENT-13", summary="Add feature")
    client = FakeJiraClient(ticket, fail_on={"get_ticket": RuntimeError("timeout")})
    service = JiraLabelApprovalService(client)

    with pytest.raises(RuntimeError, match="timeout"):
        asyncio.run(service.request_approval(_state("AGENT-13")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(ticket_key: str) -> TicketState:
    return TicketState(ticket_key=ticket_key, summary="Test ticket")


class _FakeExecutionService:
    def __init__(self, *, mark_failed_error: Exception | None = None) -> None:
        self._error = mark_failed_error
        self.mark_failed_calls: list[tuple[str, str]] = []

    async def mark_failed(self, ticket_key: str, reason: str) -> None:
        self.mark_failed_calls.append((ticket_key, reason))
        if self._error is not None:
            raise self._error

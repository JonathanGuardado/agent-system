"""Tests for execution approval services.

Covers:
- AutoApprovalService: always approves; default MVP path.
- JiraLabelApprovalService: optional manual gate; not wired by default.
"""

from __future__ import annotations

import asyncio

from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket
from ticket_agent.orchestrator.jira_services import JiraLabelApprovalService
from ticket_agent.orchestrator.local_services import AutoApprovalService
from ticket_agent.orchestrator.state import TicketState


# ---------------------------------------------------------------------------
# AutoApprovalService
# ---------------------------------------------------------------------------


def test_auto_approval_returns_true():
    service = AutoApprovalService()
    state = _state("AGENT-1")

    result = asyncio.run(service.request_approval(state))

    assert result is True


def test_auto_approval_returns_true_regardless_of_ticket_key():
    service = AutoApprovalService()

    for key in ("AGENT-1", "PROJ-999", "MVP-0"):
        result = asyncio.run(service.request_approval(_state(key)))
        assert result is True, f"expected True for {key}"


def test_auto_approval_does_not_require_jira_client():
    # AutoApprovalService takes no constructor arguments — no Jira client needed.
    service = AutoApprovalService()
    assert asyncio.run(service.request_approval(_state("AGENT-42"))) is True


# ---------------------------------------------------------------------------
# JiraLabelApprovalService — still works correctly as an optional gate
# ---------------------------------------------------------------------------


def test_jira_label_approval_returns_true_when_label_present():
    ticket = JiraTicket(
        key="AGENT-10",
        summary="Add feature",
        labels=["ai-ready", "ai-execution-approved"],
    )
    client = FakeJiraClient(ticket)
    service = JiraLabelApprovalService(client)

    result = asyncio.run(service.request_approval(_state("AGENT-10")))

    assert result is True


def test_jira_label_approval_returns_false_when_label_missing():
    ticket = JiraTicket(key="AGENT-11", summary="Add feature", labels=["ai-ready"])
    client = FakeJiraClient(ticket)
    service = JiraLabelApprovalService(client)

    result = asyncio.run(service.request_approval(_state("AGENT-11")))

    assert result is False


def test_jira_label_approval_missing_label_does_not_block_auto_approval():
    """Confirm that missing ai-execution-approved does not matter when
    AutoApprovalService is used instead of JiraLabelApprovalService."""
    # Ticket has no ai-execution-approved label.
    ticket = JiraTicket(key="AGENT-20", summary="Normal ticket", labels=["ai-ready"])
    client = FakeJiraClient(ticket)

    # JiraLabelApprovalService would reject this ticket.
    jira_service = JiraLabelApprovalService(client)
    assert asyncio.run(jira_service.request_approval(_state("AGENT-20"))) is False

    # AutoApprovalService approves it regardless — label is irrelevant.
    auto_service = AutoApprovalService()
    assert asyncio.run(auto_service.request_approval(_state("AGENT-20"))) is True


# ---------------------------------------------------------------------------
# JiraLabelApprovalService is optional — not used by default wiring
# ---------------------------------------------------------------------------


def test_jira_label_approval_is_not_used_by_default():
    """AutoApprovalService is the default; JiraLabelApprovalService requires
    explicit configuration and a JiraClient."""
    # AutoApprovalService needs no external dependencies.
    auto = AutoApprovalService()
    assert asyncio.run(auto.request_approval(_state("AGENT-99"))) is True

    # JiraLabelApprovalService cannot be constructed without a client,
    # proving it is not the zero-dependency default path.
    import inspect

    sig = inspect.signature(JiraLabelApprovalService.__init__)
    params = [p for p in sig.parameters if p != "self"]
    assert "client" in params, "JiraLabelApprovalService requires a JiraClient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(ticket_key: str) -> TicketState:
    return TicketState(ticket_key=ticket_key, summary="Test ticket")

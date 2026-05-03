from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from ticket_agent.domain.intake import (
    IntakeMode,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.jira_writer import JiraWriter
from ticket_agent.intake.proposal_store import PROPOSAL_TTL_SECONDS
from ticket_agent.jira.constants import (
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    LABEL_AI_READY,
)
from ticket_agent.jira.fake_client import FakeJiraClient


def _proposal(
    *,
    mode: IntakeMode = IntakeMode.NEW_FEATURE,
    project_key: str | None = "AGENT",
    tickets: list[TicketSpec] | None = None,
) -> Proposal:
    created = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    return Proposal(
        proposal_id="prop-1",
        slack_user_id="U1",
        slack_thread_ts="t1",
        mode=mode,
        project_key=project_key,
        epic_key=None,
        title="Add OAuth login",
        summary="Add OAuth login to AGENT",
        tickets=tickets
        if tickets is not None
        else [
            TicketSpec(
                summary="Add OAuth login",
                description="Implement the OAuth login flow.",
                labels=[LABEL_AI_READY],
                capabilities_needed=["code.implement"],
                repository="agent-system",
                repo_path="/home/agent",
            )
        ],
        revision_count=0,
        status=ProposalStatus.AWAITING_CONFIRMATION,
        created_at=created,
        expires_at=created + timedelta(seconds=PROPOSAL_TTL_SECONDS),
    )


def test_write_creates_ai_ready_ticket_with_required_fields():
    client = FakeJiraClient([])
    writer = JiraWriter(client)

    result = asyncio.run(writer.write(_proposal()))

    assert result.success is True
    assert result.partial is False
    assert result.created_ticket_keys == ("AGENT-1",)
    ticket = client.tickets["AGENT-1"]
    assert LABEL_AI_READY in ticket.labels
    assert ticket.fields[FIELD_AGENT_RETRY_COUNT] == 0
    assert ticket.fields[FIELD_AGENT_CAPABILITIES_NEEDED] == ["code.implement"]
    assert ticket.fields[FIELD_REPOSITORY] == "agent-system"
    assert ticket.fields[FIELD_REPO_PATH] == "/home/agent"


def test_write_creates_multiple_tickets_for_new_tickets_mode():
    proposal = _proposal(
        mode=IntakeMode.NEW_TICKETS,
        tickets=[
            TicketSpec(
                summary=f"Ticket {n}",
                labels=[LABEL_AI_READY],
                capabilities_needed=["ticket.decompose"],
                repository="repo",
                repo_path="/home/repo",
            )
            for n in (1, 2, 3)
        ],
    )
    client = FakeJiraClient([])
    writer = JiraWriter(client)

    result = asyncio.run(writer.write(proposal))

    assert result.created_ticket_keys == ("AGENT-1", "AGENT-2", "AGENT-3")
    assert result.partial is False


def test_partial_failure_returns_partial_result():
    proposal = _proposal(
        tickets=[
            TicketSpec(
                summary="Ticket A",
                labels=[LABEL_AI_READY],
                capabilities_needed=["code.implement"],
                repository="r",
                repo_path="/r",
            ),
            TicketSpec(
                summary="Ticket B",
                labels=[LABEL_AI_READY],
                capabilities_needed=["code.implement"],
                repository="r",
                repo_path="/r",
            ),
        ]
    )
    client = FakeJiraClient([], fail_on={"create_issue": [None, RuntimeError("boom")]})
    writer = JiraWriter(client)

    result = asyncio.run(writer.write(proposal))

    assert result.partial is True
    assert result.created_ticket_keys == ("AGENT-1",)
    assert len(result.failed_items) == 1
    assert result.failed_items[0].spec.summary == "Ticket B"
    assert "boom" in result.failed_items[0].reason


def test_new_project_returns_unsupported_result():
    proposal = _proposal(mode=IntakeMode.NEW_PROJECT)
    client = FakeJiraClient([])
    writer = JiraWriter(client)

    result = asyncio.run(writer.write(proposal))

    assert result.created_ticket_keys == ()
    assert result.unsupported_reason is not None
    assert result.partial is True
    # No Jira create calls were made.
    assert all(call[0] != "create_issue" for call in client.calls)


def test_write_adds_ai_ready_label_when_spec_lacks_it():
    proposal = _proposal(
        tickets=[
            TicketSpec(
                summary="Skinny ticket",
                labels=[],
                capabilities_needed=["code.implement"],
                repository="r",
                repo_path="/r",
            )
        ]
    )
    client = FakeJiraClient([])
    writer = JiraWriter(client)

    asyncio.run(writer.write(proposal))

    ticket = client.tickets["AGENT-1"]
    assert LABEL_AI_READY in ticket.labels

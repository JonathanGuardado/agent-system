"""End-to-end fake Slack -> Jira intake test.

Drives the listener with a Slack message and reply through to Jira ticket
creation, asserting that confirmed Jira tickets receive the ai-ready label
without manual marking — i.e. the existing detection layer can pick them up
directly.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ticket_agent.detection.detector import DetectionComponent
from ticket_agent.detection.jira_search import JiraDetectionSearchClient
from ticket_agent.detection.ownership import OwnershipChecker
from ticket_agent.intake.approval_flow import ApprovalFlow, ApprovalOutcome
from ticket_agent.intake.intent_resolver import IntakeIntentResolver
from ticket_agent.intake.jira_writer import JiraWriter
from ticket_agent.intake.proposal_generator import DeterministicProposalGenerator
from ticket_agent.intake.proposal_store import ProposalStore
from ticket_agent.intake.slack_listener import SlackEvent, SlackIntakeListener
from ticket_agent.jira.constants import LABEL_AI_READY
from ticket_agent.jira.fake_client import FakeJiraClient


class _FakeSlack:
    def __init__(self) -> None:
        self.messages: list[tuple[str | None, str, str, str]] = []

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        self.messages.append((channel, thread_ts, user_id, text))


def test_slack_message_to_ai_ready_jira_ticket(tmp_path):
    store = ProposalStore(tmp_path / "proposals.db")
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    flow = ApprovalFlow(
        resolver=IntakeIntentResolver(),
        generator=DeterministicProposalGenerator(
            clock=lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
            proposal_id_factory=lambda: "prop-int-1",
        ),
        store=store,
        jira_writer=JiraWriter(jira_client),
        slack=slack,
        repo_defaults={
            "AGENT": {
                "repository": "agent-system",
                "repo_path": "/home/jguardado/repos/agent-system",
            }
        },
    )
    listener = SlackIntakeListener(
        approval_flow=flow,
        store=store,
        intake_channel="C-INTAKE",
    )

    new_request = asyncio.run(
        listener.handle_event(
            SlackEvent(
                user_id="U-jonathan",
                text="add OAuth to AGENT project",
                channel="C-INTAKE",
                thread_ts="thr-1",
            )
        )
    )

    assert new_request is not None
    assert new_request.outcome == ApprovalOutcome.PROPOSAL_POSTED
    proposal = new_request.proposal
    assert proposal is not None
    assert proposal.project_key == "AGENT"
    assert proposal.tickets[0].labels == [LABEL_AI_READY]
    assert proposal.tickets[0].capabilities_needed == ["code.implement"]

    approved = asyncio.run(
        listener.handle_event(
            SlackEvent(
                user_id="U-jonathan",
                text="approve",
                channel="C-INTAKE",
                thread_ts="thr-1",
            )
        )
    )

    assert approved is not None
    assert approved.outcome == ApprovalOutcome.PROPOSAL_CONFIRMED
    assert approved.write_result is not None
    created_keys = approved.write_result.created_ticket_keys
    assert len(created_keys) == 1
    created_key = created_keys[0]
    created_ticket = jira_client.tickets[created_key]
    assert LABEL_AI_READY in created_ticket.labels
    assert created_ticket.fields["agent_retry_count"] == 0
    assert created_ticket.fields["repository"] == "agent-system"

    # No Slack message asks the human to manually mark anything; confirmation
    # message references the created keys for visibility.
    assert any(created_key in msg[3] for msg in slack.messages)

    # Detection should pick up the ai-ready ticket without further input.
    detection_client = JiraDetectionSearchClient(jira_client)
    detected = asyncio.run(detection_client.search_ai_ready_tickets())
    detected_keys = {ticket.key for ticket in detected}
    assert created_key in detected_keys

    # And the OwnershipChecker should classify it eligible (rule R3 + R4).
    checker = OwnershipChecker(
        component_id="agent-system",
        lock_lookup=lambda key: None,
    )
    decision = checker.check(created_ticket)
    assert decision.eligible, decision.reason


def test_listener_emits_no_jira_writes_for_clarification_path(tmp_path):
    """If the resolver asks for clarification, no Jira writes happen."""

    store = ProposalStore(tmp_path / "proposals.db")
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    flow = ApprovalFlow(
        resolver=IntakeIntentResolver(),
        generator=DeterministicProposalGenerator(
            clock=lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
        ),
        store=store,
        jira_writer=JiraWriter(jira_client),
        slack=slack,
    )
    listener = SlackIntakeListener(approval_flow=flow, store=store)

    result = asyncio.run(
        listener.handle_event(
            SlackEvent(
                user_id="U1",
                text="implement OAuth login",
                channel="C1",
                thread_ts="t1",
            )
        )
    )

    assert result is not None
    assert result.outcome == ApprovalOutcome.CLARIFICATION_REQUESTED
    assert all(call[0] != "create_issue" for call in jira_client.calls)

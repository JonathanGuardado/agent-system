from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ticket_agent.domain.intake import (
    IntakeMode,
    IntakeResolution,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.approval_flow import ApprovalFlow, ApprovalOutcome
from ticket_agent.intake.jira_writer import JiraWriter
from ticket_agent.intake.proposal_generator import (
    DeterministicProposalGenerator,
    ProposalDraft,
    ProposalRequest,
)
from ticket_agent.intake.proposal_store import ProposalStore
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


class _StubResolver:
    """Resolver stub that returns canned resolutions in order."""

    def __init__(self, resolutions: list[IntakeResolution]) -> None:
        self._resolutions = list(resolutions)
        self.calls: list[str] = []

    def resolve(self, text: str) -> IntakeResolution:
        self.calls.append(text)
        if not self._resolutions:
            raise AssertionError("resolver called more times than configured")
        return self._resolutions.pop(0)


def _resolution_new_feature() -> IntakeResolution:
    return IntakeResolution(
        mode=IntakeMode.NEW_FEATURE,
        capability="code.implement",
        model_primary="deepseek-v4-pro",
        model_fallbacks=("gemini-2.5-flash",),
    )


def _resolution_direct() -> IntakeResolution:
    return IntakeResolution(
        mode=IntakeMode.DIRECT_TICKET,
        capability="code.verify",
        model_primary="deepseek-v4-pro",
    )


def _build_flow(
    tmp_path,
    *,
    resolutions: list[IntakeResolution] | None = None,
    repo_defaults: dict[str, dict[str, str]] | None = None,
):
    store = ProposalStore(tmp_path / "proposals.db")
    jira_client = FakeJiraClient([])
    writer = JiraWriter(jira_client)
    slack = _FakeSlack()
    resolver = _StubResolver(resolutions or [_resolution_new_feature()])
    fixed_now = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    counter = {"n": 0}

    def _proposal_id() -> str:
        counter["n"] += 1
        return f"prop-{counter['n']}"

    generator = DeterministicProposalGenerator(
        clock=lambda: fixed_now,
        proposal_id_factory=_proposal_id,
    )
    flow = ApprovalFlow(
        resolver=resolver,
        generator=generator,
        store=store,
        jira_writer=writer,
        slack=slack,
        repo_defaults=repo_defaults or {"AGENT": {"repository": "agent-system", "repo_path": "/home/agent"}},
    )
    return flow, store, slack, jira_client, resolver


def test_handle_new_request_posts_proposal(tmp_path):
    flow, store, slack, _, _ = _build_flow(tmp_path)

    result = asyncio.run(
        flow.handle_new_request(
            user_id="U1",
            thread_ts="t1",
            text="Add OAuth login to AGENT",
        )
    )

    assert result.outcome == ApprovalOutcome.PROPOSAL_POSTED
    assert result.proposal is not None
    assert result.proposal.project_key == "AGENT"
    assert result.proposal.tickets[0].labels == [LABEL_AI_READY]
    assert store.get_active_for_thread("U1", "t1") is not None
    assert len(slack.messages) == 1


def test_handle_new_request_posts_clarification_when_needed(tmp_path):
    resolution = IntakeResolution(
        mode=IntakeMode.NEW_FEATURE,
        capability="code.implement",
        model_primary="deepseek-v4-pro",
        requires_clarification=True,
        clarification_question="Which Jira project?",
    )
    flow, store, slack, _, _ = _build_flow(tmp_path, resolutions=[resolution])

    result = asyncio.run(
        flow.handle_new_request(
            user_id="U1",
            thread_ts="t1",
            text="Implement OAuth login",
        )
    )

    assert result.outcome == ApprovalOutcome.CLARIFICATION_REQUESTED
    assert result.posted_message == "Which Jira project?"
    assert store.get_active_for_thread("U1", "t1") is None
    assert len(slack.messages) == 1


def test_edit_reply_revises_proposal(tmp_path):
    flow, store, slack, _, _ = _build_flow(
        tmp_path,
        resolutions=[_resolution_new_feature(), _resolution_new_feature()],
    )

    asyncio.run(
        flow.handle_new_request(
            user_id="U1",
            thread_ts="t1",
            text="Add OAuth login to AGENT",
        )
    )
    slack.messages.clear()

    result = asyncio.run(
        flow.handle_reply(
            user_id="U1",
            thread_ts="t1",
            text="Use SAML instead of OAuth and call it Single Sign-On",
        )
    )

    assert result.outcome == ApprovalOutcome.PROPOSAL_REVISED
    assert result.proposal is not None
    assert result.proposal.revision_count == 1
    persisted = store.get_active_for_thread("U1", "t1")
    assert persisted is not None
    assert persisted.revision_count == 1
    assert len(slack.messages) == 1


def test_cancel_marks_cancelled_and_does_not_call_jira_writer(tmp_path):
    flow, store, slack, jira_client, _ = _build_flow(tmp_path)

    asyncio.run(
        flow.handle_new_request(
            user_id="U1",
            thread_ts="t1",
            text="Add OAuth login to AGENT",
        )
    )
    slack.messages.clear()

    result = asyncio.run(
        flow.handle_reply(
            user_id="U1",
            thread_ts="t1",
            text="cancel",
        )
    )

    assert result.outcome == ApprovalOutcome.PROPOSAL_CANCELLED
    assert store.get_active_for_thread("U1", "t1") is None
    assert all(call[0] != "create_issue" for call in jira_client.calls)


def test_approve_calls_jira_writer_and_marks_confirmed(tmp_path):
    flow, store, slack, jira_client, _ = _build_flow(tmp_path)

    asyncio.run(
        flow.handle_new_request(
            user_id="U1",
            thread_ts="t1",
            text="Add OAuth login to AGENT",
        )
    )
    slack.messages.clear()

    result = asyncio.run(
        flow.handle_reply(
            user_id="U1",
            thread_ts="t1",
            text="approve",
        )
    )

    assert result.outcome == ApprovalOutcome.PROPOSAL_CONFIRMED
    assert result.write_result is not None
    assert result.write_result.created_ticket_keys
    assert store.get_active_for_thread("U1", "t1") is None
    confirmed = store.get(result.proposal.proposal_id)
    assert confirmed is not None
    assert confirmed.status == ProposalStatus.CONFIRMED
    create_calls = [call for call in jira_client.calls if call[0] == "create_issue"]
    assert len(create_calls) == 1
    assert len(slack.messages) == 1
    assert "AGENT-1" in slack.messages[0][3]


def test_approve_with_partial_jira_failure_posts_partial_result(tmp_path):
    flow, store, slack, jira_client, _ = _build_flow(
        tmp_path,
        resolutions=[
            IntakeResolution(
                mode=IntakeMode.NEW_TICKETS,
                capability="ticket.decompose",
                model_primary="deepseek-v4-pro",
            ),
        ],
    )
    jira_client.configure_failure(
        "create_issue", [None, RuntimeError("conflict")]
    )

    text = (
        "For AGENT epic:\n"
        "- Add login screen\n"
        "- Add signup screen"
    )
    new_request = asyncio.run(
        flow.handle_new_request(user_id="U1", thread_ts="t1", text=text)
    )
    assert new_request.outcome == ApprovalOutcome.PROPOSAL_POSTED
    assert new_request.proposal is not None
    assert len(new_request.proposal.tickets) >= 2
    slack.messages.clear()

    result = asyncio.run(
        flow.handle_reply(user_id="U1", thread_ts="t1", text="approve")
    )

    assert result.outcome == ApprovalOutcome.PROPOSAL_CONFIRMED
    assert result.write_result is not None
    assert result.write_result.partial is True
    assert len(result.write_result.created_ticket_keys) >= 1
    assert len(result.write_result.failed_items) == 1
    posted = slack.messages[0][3]
    assert "Partial" in posted or "Failures" in posted


def test_reply_with_no_active_proposal_returns_no_active(tmp_path):
    flow, _, _, _, _ = _build_flow(tmp_path, resolutions=[])

    result = asyncio.run(
        flow.handle_reply(
            user_id="U1",
            thread_ts="t1",
            text="approve",
        )
    )

    assert result.outcome == ApprovalOutcome.NO_ACTIVE_PROPOSAL

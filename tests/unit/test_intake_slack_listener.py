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
from ticket_agent.intake.proposal_generator import DeterministicProposalGenerator
from ticket_agent.intake.proposal_store import PROPOSAL_TTL_SECONDS, ProposalStore
from ticket_agent.intake.slack_listener import (
    SlackEvent,
    SlackIntakeListener,
    event_from_slack_payload,
)
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
    def __init__(self, resolutions: list[IntakeResolution]) -> None:
        self._resolutions = list(resolutions)

    def resolve(self, text: str) -> IntakeResolution:
        if not self._resolutions:
            raise AssertionError("resolver called more times than configured")
        return self._resolutions.pop(0)


def _new_feature_resolution() -> IntakeResolution:
    return IntakeResolution(
        mode=IntakeMode.NEW_FEATURE,
        capability="code.implement",
        model_primary="deepseek-v4-pro",
    )


def _build_listener(tmp_path, *, intake_channel: str | None = None):
    store = ProposalStore(tmp_path / "proposals.db")
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    fixed_now = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    counter = {"n": 0}

    def _proposal_id() -> str:
        counter["n"] += 1
        return f"prop-{counter['n']}"

    flow = ApprovalFlow(
        resolver=_StubResolver([_new_feature_resolution()]),
        generator=DeterministicProposalGenerator(
            clock=lambda: fixed_now, proposal_id_factory=_proposal_id
        ),
        store=store,
        jira_writer=JiraWriter(jira_client),
        slack=slack,
        repo_defaults={"AGENT": {"repository": "agent-system", "repo_path": "/home/agent"}},
    )
    listener = SlackIntakeListener(
        approval_flow=flow,
        store=store,
        intake_channel=intake_channel,
    )
    return listener, store, slack


def test_listener_ignores_bot_messages(tmp_path):
    listener, _, slack = _build_listener(tmp_path)

    result = asyncio.run(
        listener.handle_event(
            SlackEvent(user_id="B1", text="hi", channel="C1", thread_ts="t1", is_bot=True)
        )
    )

    assert result is None
    assert slack.messages == []


def test_listener_ignores_messages_in_other_channels(tmp_path):
    listener, _, slack = _build_listener(tmp_path, intake_channel="C-INTAKE")

    result = asyncio.run(
        listener.handle_event(
            SlackEvent(user_id="U1", text="hi", channel="C-OTHER", thread_ts="t1")
        )
    )

    assert result is None
    assert slack.messages == []


def test_listener_routes_new_request_to_handle_new_request(tmp_path):
    listener, store, slack = _build_listener(tmp_path)

    result = asyncio.run(
        listener.handle_event(
            SlackEvent(
                user_id="U1",
                text="Add OAuth login to AGENT",
                channel="C-INTAKE",
                thread_ts="t1",
            )
        )
    )

    assert result is not None
    assert result.outcome == ApprovalOutcome.PROPOSAL_POSTED
    assert store.get_active_for_thread("U1", "t1") is not None
    assert len(slack.messages) == 1


def test_listener_routes_active_thread_to_handle_reply(tmp_path):
    listener, store, slack = _build_listener(tmp_path)

    proposal = Proposal(
        proposal_id="prop-stash",
        slack_user_id="U1",
        slack_thread_ts="t1",
        mode=IntakeMode.NEW_FEATURE,
        project_key="AGENT",
        epic_key=None,
        title="t",
        summary="s",
        tickets=[
            TicketSpec(
                summary="ticket",
                labels=["ai-ready"],
                capabilities_needed=["code.implement"],
                repository="r",
                repo_path="/r",
            )
        ],
        revision_count=0,
        status=ProposalStatus.AWAITING_CONFIRMATION,
        created_at=datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
        expires_at=datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc),
    )
    store.save(proposal)

    result = asyncio.run(
        listener.handle_event(
            SlackEvent(
                user_id="U1",
                text="cancel",
                channel="C-INTAKE",
                thread_ts="t1",
            )
        )
    )

    assert result is not None
    assert result.outcome == ApprovalOutcome.PROPOSAL_CANCELLED


def test_event_from_slack_payload_marks_bot_messages():
    payload = {
        "user": "B1",
        "text": "hi",
        "channel": "C1",
        "ts": "1700000000.0001",
        "bot_id": "B12345",
    }
    event = event_from_slack_payload(payload)

    assert event.is_bot is True
    assert event.thread_ts == "1700000000.0001"


def test_event_from_slack_payload_uses_thread_ts_when_present():
    payload = {
        "user": "U1",
        "text": "edit text",
        "channel": "C1",
        "ts": "1700000005.0002",
        "thread_ts": "1700000000.0001",
    }
    event = event_from_slack_payload(payload)

    assert event.thread_ts == "1700000000.0001"
    assert event.is_bot is False

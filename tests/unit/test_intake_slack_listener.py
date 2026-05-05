from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

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
from ticket_agent.locking.checkpointer import SQLiteCheckpointer
from ticket_agent.orchestrator.execution_approval import (
    ExecutionApprovalCommandHandler,
    SQLiteExecutionApprovalStore,
    SlackExecutionApprovalService,
)
from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.state import TicketState


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


def test_listener_routes_intake_and_execution_approvals_without_collision(tmp_path):
    scenario = _ListenerExecutionScenario(tmp_path)

    try:
        asyncio.run(scenario.start_execution_approval())
        posted = asyncio.run(
            scenario.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="Add OAuth login to AGENT",
                    channel="C-INTAKE",
                    thread_ts="proposal-thread",
                )
            )
        )
        assert posted is not None
        assert posted.outcome == ApprovalOutcome.PROPOSAL_POSTED

        routed = asyncio.run(
            scenario.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve AGENT-123",
                    channel="C-INTAKE",
                    thread_ts="proposal-thread",
                )
            )
        )

        assert routed is None
        active = scenario.proposal_store.get_active_for_thread(
            "U1",
            "proposal-thread",
        )
        assert active is not None
        assert active.status == ProposalStatus.AWAITING_CONFIRMATION
        assert active.revision_count == 0
        assert scenario.jira_client.created_keys == []
        assert scenario.implementation.calls == ["AGENT-123"]
        execution_approval = scenario.execution_store.get("AGENT-123")
        assert execution_approval is not None
        assert execution_approval.status == "approved"

        approved = asyncio.run(
            scenario.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve",
                    channel="C-INTAKE",
                    thread_ts="proposal-thread",
                )
            )
        )

        assert approved is not None
        assert approved.outcome == ApprovalOutcome.PROPOSAL_CONFIRMED
        assert approved.write_result is not None
        assert approved.write_result.created_ticket_keys == ("AGENT-1",)
        assert scenario.jira_client.ticket("AGENT-1").labels == ["ai-ready"]
    finally:
        scenario.close()


def test_listener_routes_execution_rejection_without_touching_active_proposal(tmp_path):
    scenario = _ListenerExecutionScenario(tmp_path)

    try:
        asyncio.run(scenario.start_execution_approval())
        posted = asyncio.run(
            scenario.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="Add OAuth login to AGENT",
                    channel="C-INTAKE",
                    thread_ts="proposal-thread",
                )
            )
        )
        assert posted is not None
        assert posted.outcome == ApprovalOutcome.PROPOSAL_POSTED

        routed = asyncio.run(
            scenario.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="reject AGENT-123",
                    channel="C-INTAKE",
                    thread_ts="proposal-thread",
                )
            )
        )

        assert routed is None
        active = scenario.proposal_store.get_active_for_thread(
            "U1",
            "proposal-thread",
        )
        assert active is not None
        assert active.status == ProposalStatus.AWAITING_CONFIRMATION
        assert active.revision_count == 0
        assert scenario.jira_client.created_keys == []
        assert scenario.implementation.calls == []
        assert scenario.escalation.calls == [
            ("AGENT-123", "execution approval rejected")
        ]
        execution_approval = scenario.execution_store.get("AGENT-123")
        assert execution_approval is not None
        assert execution_approval.status == "rejected"
    finally:
        scenario.close()


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


class _ListenerExecutionScenario:
    def __init__(self, tmp_path) -> None:
        self.proposal_store = ProposalStore(tmp_path / "proposals.db")
        self.execution_store = SQLiteExecutionApprovalStore(
            tmp_path / "execution-approvals.db"
        )
        self.checkpointer = SQLiteCheckpointer(tmp_path / "checkpoints.db")
        self.slack = _FakeSlack()
        self.jira_client = FakeJiraClient([])
        self.implementation = _ExecutionImplementation()
        self.escalation = _ExecutionEscalation()

        flow = ApprovalFlow(
            resolver=_StubResolver([_new_feature_resolution()]),
            generator=DeterministicProposalGenerator(
                clock=lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
                proposal_id_factory=lambda: "prop-with-execution",
            ),
            store=self.proposal_store,
            jira_writer=JiraWriter(self.jira_client),
            slack=self.slack,
            repo_defaults={
                "AGENT": {
                    "repository": "agent-system",
                    "repo_path": "/home/agent",
                }
            },
        )
        approval = SlackExecutionApprovalService(
            store=self.execution_store,
            slack=self.slack,
            default_channel="C-INTAKE",
        )
        runner = TicketNodeRunner(
            planner=_ExecutionPlanner(),
            approval=approval,
            implementation=self.implementation,
            tests=_ExecutionTests(),
            review=_ExecutionReview(),
            pull_request=_ExecutionPullRequest(),
            escalation=self.escalation,
        )
        self.graph = build_ticket_graph(runner, checkpointer=self.checkpointer)
        handler = ExecutionApprovalCommandHandler(
            store=self.execution_store,
            graph=self.graph,
            slack=self.slack,
        )
        self.listener = SlackIntakeListener(
            approval_flow=flow,
            store=self.proposal_store,
            intake_channel="C-INTAKE",
            execution_approval_handler=handler,
        )

    async def start_execution_approval(self) -> dict[str, Any]:
        return await self.graph.ainvoke(
            TicketState(
                ticket_key="AGENT-123",
                summary="Add execution approval command routing",
                slack_channel="C-INTAKE",
                slack_thread_ts="execution-thread",
            ),
            config={"configurable": {"thread_id": "AGENT-123"}},
        )

    def close(self) -> None:
        self.checkpointer.close()
        self.execution_store.close()
        self.proposal_store.close()


class _ExecutionPlanner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {"summary": "Review the execution plan before implementation."}


class _ExecutionImplementation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def implement(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        return {"implementation_result": {"status": "implemented"}}


class _ExecutionTests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed", "tests_passed": True}


class _ExecutionReview:
    async def review(self, state: TicketState) -> dict[str, Any]:
        return {"status": "approved", "review_passed": True}


class _ExecutionPullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/acme/repo/pull/123"


class _ExecutionEscalation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def escalate(self, state: TicketState, reason: str) -> None:
        self.calls.append((state.ticket_key, reason))

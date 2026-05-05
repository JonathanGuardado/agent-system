from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from ticket_agent.locking.checkpointer import SQLiteCheckpointer
from ticket_agent.orchestrator.execution_approval import (
    ExecutionApprovalCommandHandler,
    SQLiteExecutionApprovalStore,
    SlackExecutionApprovalService,
)
from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.state import TicketState


def test_execution_approval_posts_to_slack_and_pauses_before_implement(tmp_path):
    scenario = _Scenario(tmp_path)

    try:
        result = asyncio.run(scenario.start())

        assert "__interrupt__" in result
        assert scenario.store.get("AGENT-123") is not None
        approval = scenario.store.get("AGENT-123")
        assert approval is not None
        assert approval.status == "pending"
        assert approval.slack_channel == "C-EXEC"
        assert approval.slack_thread_ts == "thr-1"
        assert "Approve the generated plan" in approval.plan_summary
        assert scenario.slack.messages
        message = scenario.slack.messages[0][3]
        assert "approve AGENT-123" in message
        assert "src/approval.py" in message
        assert "Implementation must wait for Slack approval" in message
        assert scenario.implementation.calls == []
    finally:
        scenario.close()


def test_approve_command_resumes_graph_and_calls_implement(tmp_path):
    scenario = _Scenario(tmp_path)

    try:
        asyncio.run(scenario.start())
        result = asyncio.run(
            scenario.handler.handle_message(
                text="approve AGENT-123",
                channel="C-EXEC",
                thread_ts="thr-1",
                user_id="U1",
            )
        )

        assert result is not None
        final_state = TicketState.model_validate(result.graph_result)
        assert final_state.workflow_status == "completed"
        assert final_state.execution_approved is True
        assert final_state.execution_approval_status == "approved"
        assert scenario.implementation.calls == ["AGENT-123"]
        approval = scenario.store.get("AGENT-123")
        assert approval is not None
        assert approval.status == "approved"
    finally:
        scenario.close()


def test_reject_command_resumes_graph_and_escalates_without_implement(tmp_path):
    scenario = _Scenario(tmp_path)

    try:
        asyncio.run(scenario.start())
        result = asyncio.run(
            scenario.handler.handle_message(
                text="reject AGENT-123",
                channel="C-EXEC",
                thread_ts="thr-1",
                user_id="U1",
            )
        )

        assert result is not None
        final_state = TicketState.model_validate(result.graph_result)
        assert final_state.workflow_status == "escalated"
        assert final_state.execution_approved is False
        assert final_state.execution_approval_status == "rejected"
        assert final_state.escalation_reason == "execution approval rejected"
        assert scenario.implementation.calls == []
        assert scenario.escalation.calls == [
            ("AGENT-123", "execution approval rejected")
        ]
    finally:
        scenario.close()


def test_expired_approval_resumes_graph_and_escalates(tmp_path):
    clock = _MutableClock()
    scenario = _Scenario(tmp_path, clock=clock, timeout=timedelta(seconds=5))

    try:
        asyncio.run(scenario.start())
        clock.now += timedelta(seconds=6)
        result = asyncio.run(scenario.handler.expire_pending("AGENT-123"))

        assert result is not None
        final_state = TicketState.model_validate(result.graph_result)
        assert final_state.workflow_status == "escalated"
        assert final_state.execution_approved is False
        assert final_state.execution_approval_status == "expired"
        assert final_state.escalation_reason == "execution approval expired"
        assert scenario.implementation.calls == []
        assert scenario.escalation.calls == [
            ("AGENT-123", "execution approval expired")
        ]
    finally:
        scenario.close()


def test_late_approve_does_not_resume_expired_approval(tmp_path):
    clock = _MutableClock()
    scenario = _Scenario(tmp_path, clock=clock, timeout=timedelta(seconds=5))

    try:
        asyncio.run(scenario.start())
        clock.now += timedelta(seconds=6)
        result = asyncio.run(
            scenario.handler.handle_message(
                text="approve AGENT-123",
                channel="C-EXEC",
                thread_ts="thr-1",
                user_id="U1",
            )
        )

        assert result is None
        approval = scenario.store.get("AGENT-123")
        assert approval is not None
        assert approval.status == "expired"
        assert scenario.implementation.calls == []
    finally:
        scenario.close()


def test_execution_approval_store_public_repository_methods(tmp_path):
    clock = _MutableClock()
    store = SQLiteExecutionApprovalStore(tmp_path / "approvals.sqlite3", clock=clock)

    try:
        pending = store.create_pending(
            ticket_key="AGENT-123",
            slack_channel="C-EXEC",
            slack_thread_ts="thr-1",
            plan_summary="Review the implementation plan.",
            timeout=timedelta(seconds=5),
        )
        assert pending.status == "pending"
        assert store.is_approved("AGENT-123") is False

        approved = store.approve("AGENT-123")
        assert approved is not None
        assert approved.status == "approved"
        assert store.is_approved("AGENT-123") is True

        store.create_pending(
            ticket_key="AGENT-456",
            slack_channel="C-EXEC",
            slack_thread_ts="thr-2",
            plan_summary="Reject this plan.",
        )
        rejected = store.reject("AGENT-456")
        assert rejected is not None
        assert rejected.status == "rejected"

        store.create_pending(
            ticket_key="AGENT-789",
            slack_channel="C-EXEC",
            slack_thread_ts="thr-3",
            plan_summary="This plan expires.",
            timeout=timedelta(seconds=5),
        )
        clock.now += timedelta(seconds=6)
        assert store.expire_pending(clock.now) == 1
        expired = store.get("AGENT-789")
        assert expired is not None
        assert expired.status == "expired"
    finally:
        store.close()


def test_execution_approval_store_does_not_reuse_terminal_approval(tmp_path):
    clock = _MutableClock()
    store = SQLiteExecutionApprovalStore(tmp_path / "approvals.sqlite3", clock=clock)

    try:
        store.create_pending(
            ticket_key="AGENT-123",
            slack_channel="C-OLD",
            slack_thread_ts="old-thread",
            plan_summary="Old plan.",
        )
        assert store.approve("AGENT-123") is not None

        clock.now += timedelta(minutes=1)
        pending = store.ensure_pending(
            ticket_key="AGENT-123",
            slack_channel="C-NEW",
            slack_thread_ts="new-thread",
            plan_summary="New plan.",
        )

        assert pending.created is True
        assert pending.approval.status == "pending"
        assert pending.approval.slack_channel == "C-NEW"
        assert pending.approval.slack_thread_ts == "new-thread"
        assert pending.approval.plan_summary == "New plan."
        assert store.is_approved("AGENT-123") is False
    finally:
        store.close()


class _Scenario:
    def __init__(
        self,
        tmp_path,
        *,
        clock: _MutableClock | None = None,
        timeout: timedelta = timedelta(hours=1),
    ) -> None:
        self.checkpointer = SQLiteCheckpointer(tmp_path / "checkpoints.sqlite3")
        self.store = SQLiteExecutionApprovalStore(
            tmp_path / "approvals.sqlite3",
            clock=clock,
        )
        self.slack = _FakeSlack()
        self.implementation = _Implementation()
        self.escalation = _Escalation()
        approval = SlackExecutionApprovalService(
            store=self.store,
            slack=self.slack,
            default_channel="C-EXEC",
            timeout=timeout,
        )
        runner = TicketNodeRunner(
            planner=_Planner(),
            approval=approval,
            implementation=self.implementation,
            tests=_Tests(),
            review=_Review(),
            pull_request=_PullRequest(),
            escalation=self.escalation,
        )
        self.graph = build_ticket_graph(runner, checkpointer=self.checkpointer)
        self.handler = ExecutionApprovalCommandHandler(
            store=self.store,
            graph=self.graph,
            slack=self.slack,
        )

    async def start(self) -> dict[str, Any]:
        return await self.graph.ainvoke(
            TicketState(
                ticket_key="AGENT-123",
                summary="Add execution approval",
                slack_channel="C-EXEC",
                slack_thread_ts="thr-1",
            ),
            config={"configurable": {"thread_id": "AGENT-123"}},
        )

    def close(self) -> None:
        self.checkpointer.close()
        self.store.close()


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


class _Planner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {
            "summary": "Approve the generated plan before writing code.",
            "files_to_modify": ["src/approval.py"],
            "risks": ["Implementation must wait for Slack approval"],
        }


class _Implementation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def implement(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        return {"implementation_result": {"status": "implemented"}}


class _Tests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed", "tests_passed": True}


class _Review:
    async def review(self, state: TicketState) -> dict[str, Any]:
        return {"status": "approved", "review_passed": True}


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/acme/repo/pull/1"


class _Escalation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def escalate(self, state: TicketState, reason: str) -> None:
        self.calls.append((state.ticket_key, reason))


class _MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

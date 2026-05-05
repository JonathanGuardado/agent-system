from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.app import RuntimeConfig, build_runtime
from ticket_agent.intake.slack_listener import SlackEvent
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.orchestrator.state import TicketState


def test_build_runtime_wires_execution_approval_commands_into_listener(tmp_path):
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    implementation = _Implementation()
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
        ),
        planner=_Planner(),
        implementation=implementation,
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )

    try:
        result = asyncio.run(
            runtime.graph.ainvoke(
                TicketState(
                    ticket_key="AGENT-123",
                    summary="Exercise runtime composition",
                    slack_channel="C-INTAKE",
                    slack_thread_ts="execution-thread",
                ),
                config={"configurable": {"thread_id": "AGENT-123"}},
            )
        )
        assert "__interrupt__" in result

        routed = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve AGENT-123",
                    channel="C-INTAKE",
                    thread_ts="execution-thread",
                )
            )
        )

        assert routed is None
        assert implementation.calls == ["AGENT-123"]
        approval = runtime.approval_store.get("AGENT-123")
        assert approval is not None
        assert approval.status == "approved"
    finally:
        runtime.close()


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
        return {"summary": "Approve the runtime plan before implementation."}


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
    async def escalate(self, state: TicketState, reason: str) -> None:
        del state, reason

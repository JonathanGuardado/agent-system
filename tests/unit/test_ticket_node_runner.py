from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.domain.errors import NoChangesToCommitError
from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.local_services import AutoApprovalService
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.state import TicketState


def test_plan_service_output_sets_decomposition():
    state = _apply(
        asyncio.run(
            _runner(planner=_Planner({"steps": ["edit"]})).plan(_initial_state())
        )
    )

    assert state.current_node == "plan"
    assert state.workflow_status == "planned"
    assert state.decomposition == {"steps": ["edit"]}
    assert state.visited_nodes == ["plan"]


def test_auto_approval_service_proceeds_to_implementation():
    """AutoApprovalService is the default MVP gate: execution proceeds without
    any manual per-ticket approval step."""
    implementation = _Implementation({"worktree_path": "/tmp/worktree"})
    runner = _runner(approval=AutoApprovalService(), implementation=implementation)
    graph = build_ticket_graph(runner)

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert "request_execution_approval" in state.visited_nodes
    assert "implement" in state.visited_nodes
    assert state.execution_approved is True
    assert state.workflow_status == "completed"


def test_approval_service_false_routes_to_escalation_in_graph():
    escalation = _Escalation()
    runner = _runner(approval=_Approval(False), escalation=escalation)
    graph = build_ticket_graph(runner)

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "escalate",
        "report",
    ]
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "execution approval rejected"
    assert escalation.calls == [("AGENT-123", "execution approval rejected")]


def test_implementation_service_increments_implementation_attempts():
    state = _apply(
        asyncio.run(
            _runner(
                implementation=_Implementation({"worktree_path": "/tmp/worktree"}),
            ).implement(_initial_state())
        )
    )

    assert state.current_node == "implement"
    assert state.workflow_status == "implementing"
    assert state.implementation_attempts == 1
    assert state.worktree_path == "/tmp/worktree"
    assert state.visited_nodes == ["implement"]


def test_test_service_failed_result_sets_tests_passed_false_and_test_result():
    state = _apply(
        asyncio.run(
            _runner(tests=_Tests({"status": "failed", "command": "pytest"})).run_tests(
                _initial_state(implementation_attempts=1)
            )
        )
    )

    assert state.current_node == "run_tests"
    assert state.workflow_status == "testing"
    assert state.tests_passed is False
    assert state.test_result == {"status": "failed", "command": "pytest"}
    assert state.visited_nodes == ["run_tests"]


def test_review_service_rejected_result_sets_review_passed_false_and_verification_result():
    state = _apply(
        asyncio.run(
            _runner(
                review=_Review({"status": "rejected", "notes": ["missing test"]}),
            ).review(
                _initial_state(),
            )
        )
    )

    assert state.current_node == "review"
    assert state.workflow_status == "reviewing"
    assert state.review_passed is False
    assert state.verification_result == {
        "status": "rejected",
        "notes": ["missing test"],
    }
    assert state.visited_nodes == ["review"]


def test_pull_request_service_sets_pull_request_url():
    state = _apply(
        asyncio.run(
            _runner(
                pull_request=_PullRequest("https://github.test/acme/repo/pull/1"),
            ).open_pull_request(_initial_state())
        )
    )

    assert state.current_node == "open_pull_request"
    assert state.workflow_status == "opening_pull_request"
    assert state.pull_request_url == "https://github.test/acme/repo/pull/1"
    assert state.visited_nodes == ["open_pull_request"]


def test_pull_request_service_exception_sets_escalation_reason():
    state = _apply(
        asyncio.run(
            _runner(
                pull_request=_FailingPullRequest(
                    NoChangesToCommitError("no changes to commit")
                ),
            ).open_pull_request(_initial_state())
        )
    )

    assert state.current_node == "open_pull_request"
    assert state.pull_request_url is None
    assert state.escalation_reason == "no changes to commit"
    assert state.error == "no changes to commit"
    assert state.visited_nodes == ["open_pull_request"]


def test_escalation_service_is_called_with_expected_reason():
    escalation = _Escalation()
    state = _apply(
        asyncio.run(
            _runner(escalation=escalation).escalate(
                _initial_state(tests_passed=False)
            )
        )
    )

    assert state.current_node == "escalate"
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "tests failed"
    assert state.visited_nodes == ["escalate"]
    assert escalation.calls == [("AGENT-123", "tests failed")]


def _runner(**overrides: Any) -> TicketNodeRunner:
    defaults = {
        "planner": _Planner({"steps": []}),
        "approval": _Approval(True),
        "implementation": _Implementation({}),
        "tests": _Tests({"status": "passed"}),
        "review": _Review({"status": "accepted"}),
        "pull_request": _PullRequest("https://github.test/acme/repo/pull/1"),
        "escalation": _Escalation(),
    }
    defaults.update(overrides)
    return TicketNodeRunner(**defaults)


def _initial_state(**updates: Any) -> TicketState:
    return TicketState(
        ticket_key="AGENT-123",
        summary="Add injected node services",
        **updates,
    )


def _apply(
    update: dict[str, Any],
    state: TicketState | None = None,
) -> TicketState:
    return (state or _initial_state()).model_copy(update=update)


class _Planner:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def plan(self, state: TicketState) -> dict[str, Any]:
        return self.result


class _Approval:
    def __init__(self, approved: bool) -> None:
        self.approved = approved

    async def request_approval(self, state: TicketState) -> bool:
        return self.approved


class _Implementation:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def implement(self, state: TicketState) -> dict[str, Any]:
        return self.result


class _Tests:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return self.result


class _Review:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def review(self, state: TicketState) -> dict[str, Any]:
        return self.result


class _PullRequest:
    def __init__(self, result: str) -> None:
        self.result = result

    async def open_pull_request(self, state: TicketState) -> str:
        return self.result


class _FailingPullRequest:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def open_pull_request(self, state: TicketState) -> str:
        raise self._error


class _Escalation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def escalate(self, state: TicketState, reason: str) -> None:
        self.calls.append((state.ticket_key, reason))

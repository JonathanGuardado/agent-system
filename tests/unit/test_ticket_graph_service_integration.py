from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.domain.errors import (
    NoChangesToCommitError,
    PullRequestCreationError,
    PushError,
)
from ticket_agent.orchestrator.graph import TicketWorkflowNodes, build_ticket_graph
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.nodes import service_backed_ticket_nodes
from ticket_agent.orchestrator.state import TicketState


def test_service_backed_node_bindings_delegate_to_ticket_node_runner_methods():
    services = _Services()
    runner = services.runner()
    nodes = service_backed_ticket_nodes(runner)

    state = _apply(asyncio.run(nodes["plan"](_initial_state())))

    assert services.planner.calls == ["AGENT-123"]
    assert state.current_node == "plan"
    assert state.decomposition == {"steps": ["edit"]}
    assert state.visited_nodes == ["plan"]


def test_service_backed_graph_calls_happy_path_services_and_records_visits():
    services = _Services()
    graph = build_ticket_graph(
        TicketWorkflowNodes(**service_backed_ticket_nodes(services.runner()))
    )

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert services.planner.calls == ["AGENT-123"]
    assert services.approval.calls == ["AGENT-123"]
    assert services.implementation.calls == ["AGENT-123"]
    assert services.tests.calls == ["AGENT-123"]
    assert services.review.calls == ["AGENT-123"]
    assert services.pull_request.calls == ["AGENT-123"]
    assert services.escalation.calls == []
    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "report",
    ]
    assert state.decomposition == {"steps": ["edit"]}
    assert state.execution_approved is True
    assert state.implementation_attempts == 1
    assert state.tests_passed is True
    assert state.review_passed is True
    assert state.pull_request_url == "https://github.test/acme/repo/pull/1"
    assert state.workflow_status == "completed"


def test_service_backed_graph_calls_escalation_service_on_rejection():
    services = _Services(approval=_Approval(False))
    graph = build_ticket_graph(services.runner())

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert services.planner.calls == ["AGENT-123"]
    assert services.approval.calls == ["AGENT-123"]
    assert services.implementation.calls == []
    assert services.tests.calls == []
    assert services.review.calls == []
    assert services.pull_request.calls == []
    assert services.escalation.calls == [
        ("AGENT-123", "execution approval rejected")
    ]
    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "escalate",
        "report",
    ]
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "execution approval rejected"


def test_service_backed_graph_retries_implementation_when_tests_fail_with_attempts_left():
    services = _Services(tests=_Tests([False, True]))
    graph = build_ticket_graph(services.runner())

    result = asyncio.run(graph.ainvoke(_initial_state(max_attempts=3)))
    state = TicketState.model_validate(result)

    assert services.implementation.calls == ["AGENT-123", "AGENT-123"]
    assert services.tests.calls == ["AGENT-123", "AGENT-123"]
    assert services.escalation.calls == []
    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "report",
    ]
    assert state.implementation_attempts == 2
    assert state.tests_passed is True
    assert state.workflow_status == "completed"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (NoChangesToCommitError("no changes to commit"), "no changes to commit"),
        (PushError("remote rejected branch"), "remote rejected branch"),
        (PullRequestCreationError("gh pr create failed"), "gh pr create failed"),
    ],
)
def test_service_backed_graph_escalates_pull_request_failures(
    error: Exception,
    reason: str,
):
    services = _Services(pull_request=_FailingPullRequest(error))
    graph = build_ticket_graph(services.runner())

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert services.pull_request.calls == ["AGENT-123"]
    assert services.escalation.calls == [("AGENT-123", reason)]
    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "escalate",
        "report",
    ]
    assert state.workflow_status == "escalated"
    assert state.pull_request_url is None
    assert state.escalation_reason == reason
    assert state.error == reason


def _initial_state(**updates: Any) -> TicketState:
    return TicketState(
        ticket_key="AGENT-123",
        summary="Call service-backed graph nodes",
        **updates,
    )


def _apply(
    update: dict[str, Any],
    state: TicketState | None = None,
) -> TicketState:
    return (state or _initial_state()).model_copy(update=update)


class _Services:
    def __init__(
        self,
        *,
        planner: _Planner | None = None,
        approval: _Approval | None = None,
        implementation: _Implementation | None = None,
        tests: _Tests | None = None,
        review: _Review | None = None,
        pull_request: _PullRequest | None = None,
        escalation: _Escalation | None = None,
    ) -> None:
        self.planner = planner or _Planner()
        self.approval = approval or _Approval(True)
        self.implementation = implementation or _Implementation()
        self.tests = tests or _Tests([True])
        self.review = review or _Review({"status": "accepted"})
        self.pull_request = pull_request or _PullRequest(
            "https://github.test/acme/repo/pull/1"
        )
        self.escalation = escalation or _Escalation()

    def runner(self) -> TicketNodeRunner:
        return TicketNodeRunner(
            planner=self.planner,
            approval=self.approval,
            implementation=self.implementation,
            tests=self.tests,
            review=self.review,
            pull_request=self.pull_request,
            escalation=self.escalation,
        )


class _Planner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def plan(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        return {"steps": ["edit"]}


class _Approval:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.calls: list[str] = []

    async def request_approval(self, state: TicketState) -> bool:
        self.calls.append(state.ticket_key)
        return self.approved


class _Implementation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def implement(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        return {
            "implementation_result": {
                "status": "implemented",
                "attempt": state.implementation_attempts + 1,
            }
        }


class _Tests:
    def __init__(self, results: list[bool]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        passed = self._results.pop(0)
        return {
            "status": "passed" if passed else "failed",
            "tests_passed": passed,
        }


class _Review:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[str] = []

    async def review(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        return self.result


class _PullRequest:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[str] = []

    async def open_pull_request(self, state: TicketState) -> str:
        self.calls.append(state.ticket_key)
        return self.result


class _FailingPullRequest:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[str] = []

    async def open_pull_request(self, state: TicketState) -> str:
        self.calls.append(state.ticket_key)
        raise self.error


class _Escalation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def escalate(self, state: TicketState, reason: str) -> None:
        self.calls.append((state.ticket_key, reason))

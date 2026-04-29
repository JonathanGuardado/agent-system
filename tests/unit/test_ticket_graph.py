from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ticket_agent.orchestrator.graph import (
    TicketWorkflowNodes,
    build_ticket_graph,
)
from ticket_agent.orchestrator.nodes import implement_ticket
from ticket_agent.orchestrator.state import TicketState


def test_ticket_graph_happy_path_reaches_completed_report():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan", workflow_status="planned"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=implement_ticket,
            run_tests=_stub("run_tests", tests_passed=True),
            review=_stub("review", review_passed=True),
            open_pull_request=_stub(
                "open_pull_request",
                pull_request_url="https://github.test/acme/repo/pull/1",
            ),
            report=_stub("report", workflow_status="completed"),
        )
    )

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "report",
    ]
    assert state.current_node == "report"
    assert state.workflow_status == "completed"
    assert state.implementation_attempts == 1
    assert state.pull_request_url == "https://github.test/acme/repo/pull/1"


def test_ticket_graph_retries_implementation_while_attempts_remain():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=implement_ticket,
            run_tests=_pass_after_attempt(2),
            review=_stub("review", review_passed=True),
            open_pull_request=_stub("open_pull_request"),
            escalate=_must_not_run("escalate"),
            report=_stub("report", workflow_status="completed"),
        )
    )

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

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


def test_ticket_graph_escalates_failed_tests_when_max_attempts_is_reached():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=implement_ticket,
            run_tests=_stub(
                "run_tests",
                tests_passed=False,
                test_result={"status": "failed"},
            ),
            review=_must_not_run("review"),
            open_pull_request=_must_not_run("open_pull_request"),
            escalate=_stub(
                "escalate",
                workflow_status="escalated",
                escalation_reason="tests failed",
            ),
            report=_stub("report", workflow_status="escalated"),
        )
    )

    result = asyncio.run(graph.ainvoke(_initial_state(max_attempts=2)))
    state = TicketState.model_validate(result)

    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "implement",
        "run_tests",
        "escalate",
        "report",
    ]
    assert state.implementation_attempts == 2
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "tests failed"
    assert state.test_result == {"status": "failed"}


def test_ticket_graph_rejected_approval_escalates():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=False,
            ),
            implement=_must_not_run("implement"),
            run_tests=_must_not_run("run_tests"),
            review=_must_not_run("review"),
            open_pull_request=_must_not_run("open_pull_request"),
            escalate=_stub(
                "escalate",
                workflow_status="escalated",
                escalation_reason="approval missing",
            ),
            report=_stub("report", workflow_status="escalated"),
        )
    )

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "escalate",
        "report",
    ]
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "approval missing"


def test_ticket_graph_failed_review_escalates():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=implement_ticket,
            run_tests=_stub("run_tests", tests_passed=True),
            review=_stub(
                "review",
                review_passed=False,
                verification_result={"status": "rejected"},
            ),
            open_pull_request=_must_not_run("open_pull_request"),
            escalate=_stub(
                "escalate",
                workflow_status="escalated",
                escalation_reason="review rejected",
            ),
            report=_stub("report", workflow_status="escalated"),
        )
    )

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "review",
        "escalate",
        "report",
    ]
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "review rejected"
    assert state.verification_result == {"status": "rejected"}


def _initial_state(**updates: Any) -> TicketState:
    return TicketState(
        ticket_key="AGENT-123",
        summary="Add the workflow skeleton",
        **updates,
    )


def _stub(name: str, **updates: Any):
    async def node(state: TicketState) -> Mapping[str, Any]:
        return {
            "current_node": name,
            "visited_nodes": [*state.visited_nodes, name],
            **updates,
        }

    return node


def _pass_after_attempt(minimum_attempt: int):
    async def node(state: TicketState) -> Mapping[str, Any]:
        tests_passed = state.implementation_attempts >= minimum_attempt
        return {
            "current_node": "run_tests",
            "visited_nodes": [*state.visited_nodes, "run_tests"],
            "tests_passed": tests_passed,
            "test_result": {"status": "passed" if tests_passed else "failed"},
        }

    return node


def _must_not_run(name: str):
    async def node(state: TicketState) -> Mapping[str, Any]:
        raise AssertionError(f"{name} should not run for this route")

    return node

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ticket_agent.orchestrator.graph import (
    TicketWorkflowNodes,
    build_ticket_graph,
)
from ticket_agent.orchestrator.state import TicketState


def test_ticket_graph_happy_path_reaches_report_after_pr():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan", workflow_status="planned"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=_stub("implement"),
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
    assert state.pull_request_url == "https://github.test/acme/repo/pull/1"


def test_ticket_graph_routes_unapproved_execution_to_escalation():
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


def test_ticket_graph_routes_failed_tests_to_escalation():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=_stub("implement"),
            run_tests=_stub("run_tests", tests_passed=False),
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

    result = asyncio.run(graph.ainvoke(_initial_state()))
    state = TicketState.model_validate(result)

    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "escalate",
        "report",
    ]
    assert state.workflow_status == "escalated"
    assert state.escalation_reason == "tests failed"


def test_ticket_graph_routes_rejected_review_to_escalation():
    graph = build_ticket_graph(
        TicketWorkflowNodes(
            plan=_stub("plan"),
            request_execution_approval=_stub(
                "request_execution_approval",
                execution_approved=True,
            ),
            implement=_stub("implement"),
            run_tests=_stub("run_tests", tests_passed=True),
            review=_stub("review", review_passed=False),
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


def _initial_state() -> TicketState:
    return TicketState(ticket_key="AGENT-123", summary="Add the workflow skeleton")


def _stub(name: str, **updates: Any):
    async def node(state: TicketState) -> Mapping[str, Any]:
        return {
            "current_node": name,
            "visited_nodes": [*state.visited_nodes, name],
            **updates,
        }

    return node


def _must_not_run(name: str):
    async def node(state: TicketState) -> Mapping[str, Any]:
        raise AssertionError(f"{name} should not run for this route")

    return node

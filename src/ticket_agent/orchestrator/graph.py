"""LangGraph builder for the ticket execution workflow."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ticket_agent.orchestrator.nodes import (
    escalate_ticket,
    implement_ticket,
    open_pull_request,
    plan_ticket,
    report_result,
    request_execution_approval,
    review_changes,
    run_tests,
)
from ticket_agent.orchestrator.state import TicketState

if TYPE_CHECKING:
    from ticket_agent.orchestrator.node_runner import TicketNodeRunner

PLAN = "plan"
REQUEST_EXECUTION_APPROVAL = "request_execution_approval"
IMPLEMENT = "implement"
RUN_TESTS = "run_tests"
REVIEW = "review"
OPEN_PULL_REQUEST = "open_pull_request"
ESCALATE = "escalate"
REPORT = "report"

TicketNode = Callable[[TicketState], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class TicketWorkflowNodes:
    plan: TicketNode = plan_ticket
    request_execution_approval: TicketNode = request_execution_approval
    implement: TicketNode = implement_ticket
    run_tests: TicketNode = run_tests
    review: TicketNode = review_changes
    open_pull_request: TicketNode = open_pull_request
    escalate: TicketNode = escalate_ticket
    report: TicketNode = report_result


def build_ticket_graph(
    nodes: TicketWorkflowNodes | TicketNodeRunner | None = None,
) -> CompiledStateGraph:
    workflow_nodes = _coerce_workflow_nodes(nodes)
    graph = StateGraph(TicketState)

    graph.add_node(PLAN, workflow_nodes.plan)
    graph.add_node(
        REQUEST_EXECUTION_APPROVAL,
        workflow_nodes.request_execution_approval,
    )
    graph.add_node(IMPLEMENT, workflow_nodes.implement)
    graph.add_node(RUN_TESTS, workflow_nodes.run_tests)
    graph.add_node(REVIEW, workflow_nodes.review)
    graph.add_node(OPEN_PULL_REQUEST, workflow_nodes.open_pull_request)
    graph.add_node(ESCALATE, workflow_nodes.escalate)
    graph.add_node(REPORT, workflow_nodes.report)

    graph.add_edge(START, PLAN)
    graph.add_edge(PLAN, REQUEST_EXECUTION_APPROVAL)
    graph.add_conditional_edges(
        REQUEST_EXECUTION_APPROVAL,
        route_after_execution_approval,
        {
            "approved": IMPLEMENT,
            "blocked": ESCALATE,
        },
    )
    graph.add_edge(IMPLEMENT, RUN_TESTS)
    graph.add_conditional_edges(
        RUN_TESTS,
        route_after_tests,
        {
            "passed": REVIEW,
            "retry": IMPLEMENT,
            "failed": ESCALATE,
        },
    )
    graph.add_conditional_edges(
        REVIEW,
        route_after_review,
        {
            "accepted": OPEN_PULL_REQUEST,
            "rejected": ESCALATE,
        },
    )
    graph.add_edge(OPEN_PULL_REQUEST, REPORT)
    graph.add_edge(ESCALATE, REPORT)
    graph.add_edge(REPORT, END)

    return graph.compile()


def _coerce_workflow_nodes(
    nodes: TicketWorkflowNodes | TicketNodeRunner | None,
) -> TicketWorkflowNodes:
    if nodes is None:
        return TicketWorkflowNodes()
    if isinstance(nodes, TicketWorkflowNodes):
        return nodes
    return nodes.as_workflow_nodes()


def route_after_execution_approval(
    state: TicketState,
) -> Literal["approved", "blocked"]:
    return "approved" if state.execution_approved is True else "blocked"


def route_after_tests(state: TicketState) -> Literal["passed", "retry", "failed"]:
    if state.tests_passed is True:
        return "passed"
    if (
        state.tests_passed is False
        and state.implementation_attempts < state.max_attempts
    ):
        return "retry"
    return "failed"


def route_after_review(state: TicketState) -> Literal["accepted", "rejected"]:
    return "accepted" if state.review_passed is True else "rejected"

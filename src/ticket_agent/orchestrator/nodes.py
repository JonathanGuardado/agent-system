"""Async nodes and service-backed node bindings for the ticket workflow graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from ticket_agent.orchestrator.state import TicketState, WorkflowStatus

if TYPE_CHECKING:
    from ticket_agent.orchestrator.node_runner import TicketNodeRunner

TicketStateUpdate = Mapping[str, Any]
TicketNode = Callable[[TicketState], Awaitable[TicketStateUpdate]]


def service_backed_ticket_nodes(
    runner: TicketNodeRunner,
) -> dict[str, TicketNode]:
    """Bind graph node names to the injected service-backed runner methods."""

    return {
        "plan": runner.plan,
        "request_execution_approval": runner.request_execution_approval,
        "implement": runner.implement,
        "run_tests": runner.run_tests,
        "review": runner.review,
        "open_pull_request": runner.open_pull_request,
        "escalate": runner.escalate,
        "report": runner.report,
    }


async def plan_ticket(state: TicketState) -> TicketStateUpdate:
    return _mark_node(state, "plan", workflow_status="planned")


async def request_execution_approval(state: TicketState) -> TicketStateUpdate:
    return _mark_node(
        state,
        "request_execution_approval",
        workflow_status="waiting_for_approval",
    )


async def implement_ticket(state: TicketState) -> TicketStateUpdate:
    return _mark_node(
        state,
        "implement",
        workflow_status="implementing",
        implementation_attempts=state.implementation_attempts + 1,
    )


async def run_tests(state: TicketState) -> TicketStateUpdate:
    return _mark_node(state, "run_tests", workflow_status="testing")


async def review_changes(state: TicketState) -> TicketStateUpdate:
    return _mark_node(state, "review", workflow_status="reviewing")


async def open_pull_request(state: TicketState) -> TicketStateUpdate:
    return _mark_node(
        state,
        "open_pull_request",
        workflow_status="opening_pull_request",
    )


async def escalate_ticket(state: TicketState) -> TicketStateUpdate:
    return _mark_node(
        state,
        "escalate",
        workflow_status="escalated",
    )


async def report_result(state: TicketState) -> TicketStateUpdate:
    status: WorkflowStatus = (
        "escalated" if state.workflow_status == "escalated" else "completed"
    )
    return _mark_node(state, "report", workflow_status=status)


def _mark_node(
    state: TicketState,
    node_name: str,
    **updates: Any,
) -> dict[str, Any]:
    return {
        "current_node": node_name,
        "visited_nodes": [*state.visited_nodes, node_name],
        **updates,
    }

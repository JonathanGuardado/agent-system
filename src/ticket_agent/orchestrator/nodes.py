"""Async node skeletons for the ticket workflow graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ticket_agent.orchestrator.state import TicketState, WorkflowStatus

TicketStateUpdate = Mapping[str, Any]


async def plan_ticket(state: TicketState) -> TicketStateUpdate:
    return _mark_node(state, "plan", workflow_status="planned")


async def request_execution_approval(state: TicketState) -> TicketStateUpdate:
    return _mark_node(
        state,
        "request_execution_approval",
        workflow_status="waiting_for_approval",
    )


async def implement_ticket(state: TicketState) -> TicketStateUpdate:
    return _mark_node(state, "implement", workflow_status="implementing")


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

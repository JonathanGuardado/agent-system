"""Dependency-injected node runner for the ticket workflow graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ticket_agent.orchestrator.services import (
    ApprovalDecision,
    ApprovalService,
    EscalationService,
    ImplementationService,
    PlannerService,
    PullRequestService,
    ReviewService,
    TestService,
)
from ticket_agent.orchestrator.state import TicketState, WorkflowStatus

if TYPE_CHECKING:
    from ticket_agent.orchestrator.graph import TicketWorkflowNodes

TicketStateUpdate = dict[str, Any]


class TicketNodeRunner:
    """Run graph nodes by delegating work to injected services."""

    def __init__(
        self,
        *,
        planner: PlannerService,
        approval: ApprovalService,
        implementation: ImplementationService,
        tests: TestService,
        review: ReviewService,
        pull_request: PullRequestService,
        escalation: EscalationService,
    ) -> None:
        self._planner = planner
        self._approval = approval
        self._implementation = implementation
        self._tests = tests
        self._review = review
        self._pull_request = pull_request
        self._escalation = escalation

    async def plan(self, state: TicketState) -> TicketStateUpdate:
        decomposition = await self._planner.plan(state)
        return _mark_node(
            state,
            "plan",
            workflow_status="planned",
            decomposition=decomposition,
        )

    async def request_execution_approval(
        self,
        state: TicketState,
    ) -> TicketStateUpdate:
        decision = _normalize_approval_decision(
            await self._approval.request_approval(state)
        )
        updates: dict[str, Any] = {
            "execution_approved": decision.approved,
            "execution_approval_status": decision.status,
        }
        if decision.approved is False and decision.reason:
            updates["escalation_reason"] = decision.reason
        return _mark_node(
            state,
            "request_execution_approval",
            workflow_status="waiting_for_approval",
            **updates,
        )

    async def implement(self, state: TicketState) -> TicketStateUpdate:
        implementation_update = await self._implementation.implement(state)
        return _mark_node(
            state,
            "implement",
            service_updates=implementation_update,
            workflow_status="implementing",
            implementation_attempts=state.implementation_attempts + 1,
        )

    async def run_tests(self, state: TicketState) -> TicketStateUpdate:
        test_result = await self._tests.run_tests(state)
        return _mark_node(
            state,
            "run_tests",
            workflow_status="testing",
            tests_passed=_result_passed(test_result),
            test_result=test_result,
        )

    async def review(self, state: TicketState) -> TicketStateUpdate:
        verification_result = await self._review.review(state)
        return _mark_node(
            state,
            "review",
            workflow_status="reviewing",
            review_passed=_result_passed(
                verification_result,
                explicit_key="review_passed",
                positive_statuses={"accepted", "approved", "passed", "success"},
                negative_statuses={"rejected", "failed", "failure"},
            ),
            verification_result=verification_result,
        )

    async def open_pull_request(self, state: TicketState) -> TicketStateUpdate:
        pull_request_url = await self._pull_request.open_pull_request(state)
        return _mark_node(
            state,
            "open_pull_request",
            workflow_status="opening_pull_request",
            pull_request_url=pull_request_url,
        )

    async def escalate(self, state: TicketState) -> TicketStateUpdate:
        reason = _escalation_reason(state)
        await self._escalation.escalate(state, reason)
        return _mark_node(
            state,
            "escalate",
            workflow_status="escalated",
            escalation_reason=reason,
        )

    async def report(self, state: TicketState) -> TicketStateUpdate:
        status: WorkflowStatus = (
            "escalated" if state.workflow_status == "escalated" else "completed"
        )
        return _mark_node(state, "report", workflow_status=status)

    def as_workflow_nodes(self) -> TicketWorkflowNodes:
        from ticket_agent.orchestrator.graph import TicketWorkflowNodes

        return TicketWorkflowNodes(
            plan=self.plan,
            request_execution_approval=self.request_execution_approval,
            implement=self.implement,
            run_tests=self.run_tests,
            review=self.review,
            open_pull_request=self.open_pull_request,
            escalate=self.escalate,
            report=self.report,
        )


def _mark_node(
    state: TicketState,
    node_name: str,
    *,
    service_updates: dict[str, Any] | None = None,
    **updates: Any,
) -> TicketStateUpdate:
    return {
        **(service_updates or {}),
        "current_node": node_name,
        "visited_nodes": [*state.visited_nodes, node_name],
        **updates,
    }


def _result_passed(
    result: dict[str, Any],
    *,
    explicit_key: str = "tests_passed",
    positive_statuses: set[str] | None = None,
    negative_statuses: set[str] | None = None,
) -> bool | None:
    positive_statuses = positive_statuses or {"passed", "success"}
    negative_statuses = negative_statuses or {"failed", "failure", "error"}

    explicit_result = result.get(explicit_key, result.get("passed"))
    if isinstance(explicit_result, bool):
        return explicit_result

    status = result.get("status")
    if not isinstance(status, str):
        return None

    normalized_status = status.lower()
    if normalized_status in positive_statuses:
        return True
    if normalized_status in negative_statuses:
        return False
    return None


def _normalize_approval_decision(
    decision: bool | ApprovalDecision,
) -> ApprovalDecision:
    if isinstance(decision, ApprovalDecision):
        return decision
    return ApprovalDecision(
        approved=decision,
        status="approved" if decision else "rejected",
        reason=None if decision else "execution approval rejected",
    )


def _escalation_reason(state: TicketState) -> str:
    if state.escalation_reason:
        return state.escalation_reason
    if state.execution_approved is False:
        return "execution approval rejected"
    if state.tests_passed is False:
        return "tests failed"
    if state.review_passed is False:
        return "review rejected"
    if state.error:
        return state.error
    return "workflow escalated"

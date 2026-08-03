"""Dependency-injected node runner for the ticket workflow graph."""

from __future__ import annotations

from hashlib import sha256
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

from ticket_agent.observability.telemetry import (
    IterationRecord,
    NullTelemetryRecorder,
    TelemetryRecorder,
)
from ticket_agent.observability.transcripts import (
    NullTranscriptRecorder,
    TranscriptEvent,
    TranscriptRecorder,
    safe_record,
)
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


class ActionGuard(Protocol):
    def check(self, goal_id: str | None, operation: str) -> object: ...


class _AllowAllActionGuard:
    def check(self, goal_id: str | None, operation: str) -> object:
        del goal_id, operation
        return None


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
        transcripts: TranscriptRecorder | None = None,
        telemetry: TelemetryRecorder | None = None,
        autonomy_guard: ActionGuard | None = None,
    ) -> None:
        self._planner = planner
        self._approval = approval
        self._implementation = implementation
        self._tests = tests
        self._review = review
        self._pull_request = pull_request
        self._escalation = escalation
        self._transcripts = transcripts or NullTranscriptRecorder()
        self._telemetry = telemetry or NullTelemetryRecorder()
        self._autonomy_guard = autonomy_guard or _AllowAllActionGuard()

    def _mark_node(
        self,
        state: TicketState,
        node_name: str,
        *,
        service_updates: dict[str, Any] | None = None,
        **updates: Any,
    ) -> TicketStateUpdate:
        """Build the state update and record that the node ran.

        Every node returns through here, so this is the one place node
        transitions are observed. What gets recorded is deliberately thin:
        the update *keys* plus a short allowlist of scalars. Recording whole
        update dicts would put gate output -- potentially megabytes, and
        potentially secret -- into the transcript.
        """

        update = _node_update(
            state, node_name, service_updates=service_updates, **updates
        )
        self._record_node(state, node_name, update)
        self._record_stage(state, node_name, update)
        return update

    def _record_stage(
        self,
        state: TicketState,
        node_name: str,
        update: TicketStateUpdate,
    ) -> None:
        """Record the funnel stage a node reached, when it reached one.

        Not every node maps to a stage, and a node running is not the same as
        its stage being reached: approval only counts when it was granted, and
        a PR only counts when a URL came back. Recording on entry regardless
        would make every conversion rate read 100%.
        """

        stage = _NODE_STAGES.get(node_name)
        if stage is None:
            return
        if node_name == "request_execution_approval" and not update.get(
            "execution_approved"
        ):
            return
        if node_name == "open_pull_request" and not update.get("pull_request_url"):
            return

        if stage == "escalated":
            self._telemetry.record_escalation(
                state.ticket_key,
                str(update.get("escalation_reason") or "unspecified"),
            )
            return
        self._telemetry.record_stage(state.ticket_key, stage, goal_id=state.goal_id)

    def _record_node(
        self,
        state: TicketState,
        node_name: str,
        update: TicketStateUpdate,
    ) -> None:
        payload: dict[str, Any] = {
            "update_keys": sorted(k for k in update if k != "visited_nodes"),
            "attempt": state.implementation_attempts,
        }
        for key in _RECORDED_SCALARS:
            value = update.get(key)
            if value is not None:
                payload[key] = value
        safe_record(
            self._transcripts,
            TranscriptEvent(
                ticket_key=state.ticket_key,
                kind="node",
                name=node_name,
                payload=payload,
                run_id=state.lock_id,
                goal_id=state.goal_id,
                phase=str(update.get("workflow_status") or state.workflow_status),
            ),
        )

    async def plan(self, state: TicketState) -> TicketStateUpdate:
        self._autonomy_guard.check(state.goal_id, "plan")
        decomposition = await self._planner.plan(state)
        return self._mark_node(
            state,
            "plan",
            workflow_status="planned",
            decomposition=decomposition,
        )

    async def request_execution_approval(
        self,
        state: TicketState,
    ) -> TicketStateUpdate:
        self._autonomy_guard.check(state.goal_id, "request_execution_approval")
        decision = _normalize_approval_decision(
            await self._approval.request_approval(state)
        )
        updates: dict[str, Any] = {
            "execution_approved": decision.approved,
            "execution_approval_status": decision.status,
        }
        if decision.approved is False and decision.reason:
            updates["escalation_reason"] = decision.reason
        return self._mark_node(
            state,
            "request_execution_approval",
            workflow_status="waiting_for_approval",
            **updates,
        )

    async def implement(self, state: TicketState) -> TicketStateUpdate:
        started = monotonic()
        try:
            self._autonomy_guard.check(state.goal_id, "implement")
            implementation_update = await self._implementation.implement(state)
        except Exception as exc:  # noqa: BLE001 - any node failure becomes a recorded failed result
            error = _error_message(exc)
            implementation_update = {
                "implementation_result": {
                    "status": "failed",
                    "changed_files": [],
                    "error": error,
                },
                "error": error,
                "errors": [*state.errors, error],
            }
        self._record_implementation_iteration(
            state,
            implementation_update,
            wall_ms=max(0, round((monotonic() - started) * 1000)),
        )
        return self._mark_node(
            state,
            "implement",
            service_updates=implementation_update,
            workflow_status="implementing",
            implementation_attempts=state.implementation_attempts + 1,
            # A fresh attempt invalidates any reason carried from a prior
            # review rejection; escalation reasons must describe this attempt.
            escalation_reason=None,
        )

    def _record_implementation_iteration(
        self,
        state: TicketState,
        update: dict[str, Any],
        *,
        wall_ms: int,
    ) -> None:
        if state.goal_id is None:
            return
        result = update.get("implementation_result")
        result = result if isinstance(result, dict) else {}
        outcome = result.get("status")
        error = result.get("error") or update.get("error")
        fingerprint = None
        if error:
            fingerprint = sha256(str(error).encode("utf-8")).hexdigest()[:16]
        self._telemetry.record_iteration(
            IterationRecord(
                goal_id=state.goal_id,
                loop="implement",
                iteration=state.implementation_attempts + 1,
                outcome=str(outcome) if outcome is not None else None,
                fingerprint=fingerprint,
                tokens=_optional_int(result.get("tokens")),
                cost_usd=_optional_float(result.get("cost_usd")),
                wall_ms=wall_ms,
            )
        )

    async def run_tests(self, state: TicketState) -> TicketStateUpdate:
        try:
            self._autonomy_guard.check(state.goal_id, "run_tests")
            test_result = await self._tests.run_tests(state)
        except Exception as exc:  # noqa: BLE001 - any node failure becomes a recorded failed result
            error = _error_message(exc)
            test_result = {"status": "failed", "tests_passed": False, "error": error}
        return self._mark_node(
            state,
            "run_tests",
            workflow_status="testing",
            tests_passed=_result_passed(test_result),
            test_result=test_result,
        )

    async def review(self, state: TicketState) -> TicketStateUpdate:
        try:
            self._autonomy_guard.check(state.goal_id, "review")
            verification_result = await self._review.review(state)
        except Exception as exc:  # noqa: BLE001 - any node failure becomes a recorded failed result
            error = _error_message(exc)
            verification_result = {
                "status": "failed",
                "review_passed": False,
                "error": error,
            }
        review_passed = _result_passed(
            verification_result,
            explicit_key="review_passed",
            positive_statuses={"accepted", "approved", "passed", "success"},
            negative_statuses={"rejected", "failed", "failure"},
        )
        return self._mark_node(
            state,
            "review",
            workflow_status="reviewing",
            review_passed=review_passed,
            escalation_reason=(
                _review_failure_reason(verification_result)
                if review_passed is False
                else _result_error(verification_result)
            ),
            verification_result=verification_result,
        )

    async def open_pull_request(self, state: TicketState) -> TicketStateUpdate:
        try:
            self._autonomy_guard.check(state.goal_id, "open_pull_request")
            pull_request_url = await self._pull_request.open_pull_request(state)
        except Exception as exc:  # noqa: BLE001 - any node failure becomes a recorded failed result
            error = _error_message(exc)
            return self._mark_node(
                state,
                "open_pull_request",
                workflow_status="opening_pull_request",
                escalation_reason=error,
                error=error,
                errors=[*state.errors, error],
            )
        if not pull_request_url:
            error = "pull request service did not return a PR URL"
            return self._mark_node(
                state,
                "open_pull_request",
                workflow_status="opening_pull_request",
                escalation_reason=error,
                error=error,
                errors=[*state.errors, error],
            )
        return self._mark_node(
            state,
            "open_pull_request",
            workflow_status="opening_pull_request",
            pull_request_url=pull_request_url,
        )

    async def escalate(self, state: TicketState) -> TicketStateUpdate:
        reason = _escalation_reason(state)
        await self._escalation.escalate(state, reason)
        return self._mark_node(
            state,
            "escalate",
            workflow_status="escalated",
            escalation_reason=reason,
        )

    async def report(self, state: TicketState) -> TicketStateUpdate:
        status: WorkflowStatus = (
            "escalated" if state.workflow_status == "escalated" else "completed"
        )
        return self._mark_node(state, "report", workflow_status=status)

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


#: Nodes that mark a funnel stage. Nodes absent here reach no stage:
#: run_tests and review get theirs when the VERIFY topology lands.
_NODE_STAGES: dict[str, str] = {
    "plan": "planned",
    "request_execution_approval": "approved",
    "implement": "implemented",
    "open_pull_request": "pr_opened",
    "escalate": "escalated",
}


#: Update values small and non-sensitive enough to record verbatim. Anything
#: not listed here is represented only by its key.
_RECORDED_SCALARS: tuple[str, ...] = (
    "workflow_status",
    "execution_approved",
    "execution_approval_status",
    "tests_passed",
    "review_passed",
    "candidate_sha",
    "verification_attempts",
)


def _node_update(
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


def _result_error(result: dict[str, Any]) -> str | None:
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error
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
    implementation_reason = _implementation_failure_reason(state)
    if implementation_reason is not None:
        return implementation_reason
    if state.tests_passed is False:
        test_error = _test_failure_reason(state)
        if test_error is not None:
            return test_error
        return "tests failed"
    if state.review_passed is False:
        review_error = _review_failure_reason(state.verification_result)
        if review_error is not None:
            return review_error
        return "review rejected"
    if state.error:
        return state.error
    return "workflow escalated"


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _implementation_failure_reason(state: TicketState) -> str | None:
    result = state.implementation_result
    if not isinstance(result, dict):
        return None

    status = result.get("status")
    if not isinstance(status, str) or status.lower() not in {
        "failed",
        "failure",
        "error",
    }:
        return None

    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error

    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary

    error_code = result.get("error_code")
    if isinstance(error_code, str) and error_code.strip():
        return f"implementation failed: {error_code}"

    return "implementation failed"


def _test_failure_reason(state: TicketState) -> str | None:
    result = state.test_result
    if not isinstance(result, dict):
        return None
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error
    return None


def _review_failure_reason(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None

    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error

    issues = _string_list(result.get("issues")) or _string_list(result.get("notes"))
    if issues:
        return _format_review_rejection(issues)

    for key in ("reason", "reasoning", "summary"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return f"review rejected: {value.strip()}"

    status = result.get("status")
    if isinstance(status, str) and status.strip():
        return f"review {status.strip()}"

    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _format_review_rejection(issues: list[str]) -> str:
    if len(issues) == 1:
        return f"review rejected: {issues[0]}"
    return "review rejected:\n" + "\n".join(f"- {issue}" for issue in issues)

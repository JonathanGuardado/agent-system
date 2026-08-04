"""Slack approval flow that turns proposals into Jira work."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from inspect import isawaitable
from typing import Protocol

from ticket_agent.domain.intake import (
    Proposal,
    ProposalStatus,
    SlackPoster,
    TicketSpec,
)
from ticket_agent.goal.authorizer import GoalAuthorizer
from ticket_agent.goal.contract import AuthorizationOutcome
from ticket_agent.goal.types import AutonomyDecision, AutonomyMode, GoalContract
from ticket_agent.intake.intent_resolver import IntakeIntentResolver
from ticket_agent.intake.jira_writer import JiraWriter, JiraWriteResult
from ticket_agent.intake.proposal_generator import (
    MAX_TICKETS,
    ProposalDraft,
    ProposalGenerator,
    ProposalRequest,
)
from ticket_agent.intake.proposal_store import ProposalStore


class GoalAutonomyResolver(Protocol):
    def decide(
        self,
        contract: GoalContract,
        *,
        sandbox_available: bool,
        per_command_approval: bool = False,
        halted: bool = False,
    ) -> AutonomyDecision: ...


class ApprovalOutcome(StrEnum):
    """High-level result of handling a Slack message."""

    CLARIFICATION_REQUESTED = "clarification_requested"
    PROPOSAL_POSTED = "proposal_posted"
    PROPOSAL_REVISED = "proposal_revised"
    PROPOSAL_CONFIRMED = "proposal_confirmed"
    PROPOSAL_CANCELLED = "proposal_cancelled"
    PROPOSAL_EXPIRED = "proposal_expired"
    NO_ACTIVE_PROPOSAL = "no_active_proposal"
    JIRA_WRITE_FAILED = "jira_write_failed"


@dataclass(frozen=True)
class ApprovalResult:
    """Structured outcome of one approval-flow turn."""

    outcome: ApprovalOutcome
    proposal: Proposal | None = None
    write_result: JiraWriteResult | None = None
    posted_message: str | None = None


_APPROVE_WORDS = {"approve", "approved", "confirm", "confirmed", "yes", "lgtm"}
_CANCEL_WORDS = {"cancel", "cancelled", "abort", "stop", "no"}


class ApprovalFlow:
    """Mediates Slack messages, the proposal store, and the Jira writer."""

    def __init__(
        self,
        *,
        resolver: IntakeIntentResolver,
        generator: ProposalGenerator,
        store: ProposalStore,
        jira_writer: JiraWriter,
        slack: SlackPoster,
        repo_defaults: Mapping[str, Mapping[str, str]] | None = None,
        emit: Callable[[str, dict[str, object]], None] | None = None,
        goal_authorizer: GoalAuthorizer | None = None,
        autonomy_resolver: GoalAutonomyResolver | None = None,
    ) -> None:
        self._resolver = resolver
        self._generator = generator
        self._store = store
        self._jira_writer = jira_writer
        self._slack = slack
        self._repo_defaults: Mapping[str, Mapping[str, str]] = repo_defaults or {}
        self._emit = emit
        self._goal_authorizer = goal_authorizer
        self._autonomy_resolver = autonomy_resolver

    async def handle_new_request(
        self,
        *,
        user_id: str,
        thread_ts: str,
        text: str,
        channel: str | None = None,
    ) -> ApprovalResult:
        """Process a brand-new Slack message in a thread."""

        resolution = self._resolver.resolve(text)
        if resolution.requires_clarification and resolution.clarification_question:
            await self._post(
                channel,
                thread_ts,
                user_id,
                resolution.clarification_question,
            )
            self._emit_event(
                "intake.clarification_requested",
                {"user_id": user_id, "thread_ts": thread_ts},
            )
            return ApprovalResult(
                outcome=ApprovalOutcome.CLARIFICATION_REQUESTED,
                posted_message=resolution.clarification_question,
            )

        request = ProposalRequest(
            slack_user_id=user_id,
            slack_thread_ts=thread_ts,
            text=text,
            resolution=resolution,
            slack_channel=channel,
            repo_defaults=self._repo_defaults,
        )
        draft = await _generate_proposal(self._generator, request)
        if draft.needs_clarification:
            if draft.clarification is None:
                raise ValueError(
                    "generator asked for clarification without providing one"
                )
            # Persist the placeholder so the answer returns through _revise with
            # this as prior, which disables a second clarification round.
            if draft.pending_proposal is not None:
                self._store.save(draft.pending_proposal)
            await self._post(channel, thread_ts, user_id, draft.clarification)
            self._emit_event(
                "intake.clarification_requested",
                {"user_id": user_id, "thread_ts": thread_ts},
            )
            return ApprovalResult(
                outcome=ApprovalOutcome.CLARIFICATION_REQUESTED,
                posted_message=draft.clarification,
            )

        if draft.proposal is None:
            raise ValueError(
                "generator returned neither a proposal nor a clarification"
            )
        proposal = draft.proposal
        self._store.save(proposal)
        message = _format_proposal_message(proposal)
        await self._post(channel, thread_ts, user_id, message)
        self._emit_event(
            "intake.proposal_posted",
            {
                "proposal_id": proposal.proposal_id,
                "user_id": user_id,
                "thread_ts": thread_ts,
            },
        )
        return ApprovalResult(
            outcome=ApprovalOutcome.PROPOSAL_POSTED,
            proposal=proposal,
            posted_message=message,
        )

    async def handle_reply(
        self,
        *,
        user_id: str,
        thread_ts: str,
        text: str,
        channel: str | None = None,
    ) -> ApprovalResult:
        """Process a reply to an existing thread that already has a proposal."""

        proposal = self._store.get_active_for_thread(user_id, thread_ts)
        if proposal is None:
            return ApprovalResult(outcome=ApprovalOutcome.NO_ACTIVE_PROPOSAL)

        decision = _classify_reply(text)
        if decision == "approve":
            return await self._approve(proposal, channel)
        if decision == "cancel":
            return await self._cancel(proposal, channel)
        return await self._revise(proposal, text, channel)

    async def _approve(
        self,
        proposal: Proposal,
        channel: str | None,
    ) -> ApprovalResult:
        if proposal.status != ProposalStatus.AWAITING_CONFIRMATION:
            await self._post(
                channel,
                proposal.slack_thread_ts,
                proposal.slack_user_id,
                "This proposal is not ready to approve yet — please reply with "
                "edits or wait for the proposal to be posted.",
            )
            return ApprovalResult(outcome=ApprovalOutcome.NO_ACTIVE_PROPOSAL)

        authorization = await self._authorize_goal(proposal)
        autonomy = self._resolve_autonomy(proposal, authorization)
        autonomy_mode = (
            autonomy.effective_mode
            if autonomy is not None
            else AutonomyMode.AUTONOMOUS
        )
        result = await self._jira_writer.write(
            proposal,
            publish_ai_ready=(
                authorization is not None
                and authorization.authorized
                and autonomy_mode >= AutonomyMode.IMPLEMENT
            ),
            autonomy_mode=autonomy_mode,
        )
        if not result.created_ticket_keys and not result.created_epic_key:
            self._store.mark_status(
                proposal.proposal_id, ProposalStatus.AWAITING_CONFIRMATION
            )
            message = _format_jira_failure_message(result)
            await self._post(
                channel,
                proposal.slack_thread_ts,
                proposal.slack_user_id,
                message,
            )
            self._emit_event(
                "intake.jira_write_failed",
                {
                    "proposal_id": proposal.proposal_id,
                    "failures": [item.reason for item in result.failed_items],
                },
            )
            return ApprovalResult(
                outcome=ApprovalOutcome.JIRA_WRITE_FAILED,
                proposal=proposal,
                write_result=result,
                posted_message=message,
            )

        self._store.mark_status(proposal.proposal_id, ProposalStatus.CONFIRMED)
        self._emit_goal_contract(proposal, authorization)
        if autonomy is not None:
            self._emit_event(
                "goal.autonomy_decided",
                {
                    "proposal_id": proposal.proposal_id,
                    "goal_id": autonomy.goal_id,
                    "effective_mode": str(autonomy.effective_mode),
                    "binding_sources": list(autonomy.binding_sources),
                    "decision_digest": autonomy.decision_digest,
                },
            )
        message = _format_confirmation_message(result)
        await self._post(
            channel,
            proposal.slack_thread_ts,
            proposal.slack_user_id,
            message,
        )
        self._emit_event(
            "intake.proposal_confirmed",
            {
                "proposal_id": proposal.proposal_id,
                "ticket_keys": list(result.created_ticket_keys),
                "partial": result.partial,
            },
        )
        return ApprovalResult(
            outcome=ApprovalOutcome.PROPOSAL_CONFIRMED,
            proposal=proposal,
            write_result=result,
            posted_message=message,
        )

    async def _authorize_goal(
        self,
        proposal: Proposal,
    ) -> AuthorizationOutcome | None:
        """Persist authority before Jira can publish executable work."""

        if self._goal_authorizer is None:
            self._emit_event(
                "goal.contract_failed",
                {
                    "proposal_id": proposal.proposal_id,
                    "error": "goal authorizer is not configured",
                },
            )
            return None
        try:
            return await self._goal_authorizer.authorize(proposal)
        except Exception as exc:  # noqa: BLE001 - never break intake
            self._emit_event(
                "goal.contract_failed",
                {"proposal_id": proposal.proposal_id, "error": str(exc)},
            )
            return None

    def _emit_goal_contract(
        self,
        proposal: Proposal,
        outcome: AuthorizationOutcome | None,
    ) -> None:
        if outcome is None:
            return
        self._emit_event(
            "goal.contract_recorded",
            {
                "proposal_id": proposal.proposal_id,
                "goal_id": outcome.contract.goal_id,
                "risk_class": outcome.contract.risk_class,
                "authorized": outcome.authorized,
                "signed": outcome.signature is not None,
                "reasons": list(outcome.escalation_reasons()),
            },
        )

    def _resolve_autonomy(
        self,
        proposal: Proposal,
        authorization: AuthorizationOutcome | None,
    ) -> AutonomyDecision | None:
        if authorization is None or self._autonomy_resolver is None:
            return None
        try:
            # Intake remains available without probing the host sandbox. The
            # shared execution preflight recomputes with actual availability.
            return self._autonomy_resolver.decide(
                authorization.contract,
                sandbox_available=True,
            )
        except Exception as exc:  # noqa: BLE001 - resolution fails closed
            self._emit_event(
                "goal.autonomy_failed",
                {"proposal_id": proposal.proposal_id, "error": str(exc)},
            )
            return AutonomyDecision(
                goal_id=proposal.proposal_id,
                contract_version=authorization.contract.version,
                effective_mode=AutonomyMode.OBSERVE,
                ceilings=(),
            )

    async def _cancel(
        self,
        proposal: Proposal,
        channel: str | None,
    ) -> ApprovalResult:
        self._store.mark_status(proposal.proposal_id, ProposalStatus.CANCELLED)
        message = "Proposal cancelled. Reply with a new request when you're ready."
        await self._post(
            channel,
            proposal.slack_thread_ts,
            proposal.slack_user_id,
            message,
        )
        self._emit_event(
            "intake.proposal_cancelled",
            {"proposal_id": proposal.proposal_id},
        )
        return ApprovalResult(
            outcome=ApprovalOutcome.PROPOSAL_CANCELLED,
            proposal=proposal,
            posted_message=message,
        )

    async def _revise(
        self,
        proposal: Proposal,
        text: str,
        channel: str | None,
    ) -> ApprovalResult:
        edit_resolution = self._resolver.resolve(text)
        resolution = _revision_resolution(proposal, edit_resolution)
        request = ProposalRequest(
            slack_user_id=proposal.slack_user_id,
            slack_thread_ts=proposal.slack_thread_ts,
            text=text.strip(),
            resolution=resolution,
            slack_channel=channel or proposal.slack_channel,
            repo_defaults=self._repo_defaults,
        )
        draft = await _generate_proposal(self._generator, request, prior=proposal)
        if draft.needs_clarification:
            if draft.clarification is None:
                raise ValueError(
                    "generator asked for clarification without providing one"
                )
            await self._post(
                channel,
                proposal.slack_thread_ts,
                proposal.slack_user_id,
                draft.clarification,
            )
            return ApprovalResult(
                outcome=ApprovalOutcome.CLARIFICATION_REQUESTED,
                proposal=proposal,
                posted_message=draft.clarification,
            )

        if draft.proposal is None:
            raise ValueError(
                "generator returned neither a revision nor a clarification"
            )
        revised = draft.proposal
        self._store.update(revised)
        message = _format_proposal_message(revised, revised=True)
        await self._post(
            channel,
            proposal.slack_thread_ts,
            proposal.slack_user_id,
            message,
        )
        self._emit_event(
            "intake.proposal_revised",
            {
                "proposal_id": revised.proposal_id,
                "revision_count": revised.revision_count,
            },
        )
        return ApprovalResult(
            outcome=ApprovalOutcome.PROPOSAL_REVISED,
            proposal=revised,
            posted_message=message,
        )

    async def _post(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        await self._slack.post_thread_reply(channel, thread_ts, user_id, text)

    def _emit_event(self, name: str, payload: dict[str, object]) -> None:
        if self._emit is None:
            return
        self._emit(name, payload)


def _classify_reply(text: str) -> str:
    normalized = _normalize_reply_command(text)
    if not normalized:
        return "edit"
    first_word = normalized.split()[0]
    if first_word in _APPROVE_WORDS and len(normalized.split()) <= 3:
        return "approve"
    if first_word in _CANCEL_WORDS and len(normalized.split()) <= 3:
        return "cancel"
    return "edit"


def _normalize_reply_command(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    normalized = normalized.strip("`*_~")
    if normalized.startswith("<") and ">" in normalized:
        normalized = normalized.split(">", maxsplit=1)[1].strip()
    return normalized.strip("`*_~")


async def _generate_proposal(
    generator: ProposalGenerator,
    request: ProposalRequest,
    *,
    prior: Proposal | None = None,
) -> ProposalDraft:
    draft = generator.generate(request, prior=prior)
    if isawaitable(draft):
        draft = await draft
    return draft


def _revision_resolution(
    proposal: Proposal,
    edit_resolution,
):
    return edit_resolution.model_copy(
        update={
            "mode": proposal.mode,
            "capability": _proposal_capability(proposal)
            or edit_resolution.capability,
            "requires_clarification": False,
            "clarification_question": None,
        }
    )


def _proposal_capability(proposal: Proposal) -> str | None:
    for ticket in proposal.tickets:
        if ticket.capabilities_needed:
            return ticket.capabilities_needed[0]
    return None


def _format_proposal_message(proposal: Proposal, *, revised: bool = False) -> str:
    header = (
        f"Updated proposal (revision {proposal.revision_count})"
        if revised
        else "Proposal ready for review"
    )
    project_line = (
        f"Project: {proposal.project_key}" if proposal.project_key else "Project: _unset_"
    )
    epic_line = f"Epic: {proposal.epic_key}" if proposal.epic_key else None
    epic_create_line = (
        f"Epic to create: {proposal.epic_summary}"
        if proposal.epic_summary and not proposal.epic_key and len(proposal.tickets) > 1
        else None
    )
    ticket_lines = [
        _format_ticket_line(index, ticket)
        for index, ticket in enumerate(proposal.tickets, start=1)
    ]

    body_lines: list[str] = [
        header,
        f"Mode: {proposal.mode.value}",
        project_line,
    ]
    if epic_line:
        body_lines.append(epic_line)
    if epic_create_line:
        body_lines.append(epic_create_line)
    body_lines.append(f"Title: {proposal.title}")
    if proposal.effort_estimate:
        body_lines.append(f"Effort: {proposal.effort_estimate}")
    if proposal.assumptions:
        body_lines.append("Assumptions:")
        body_lines.extend(f"  - {item}" for item in proposal.assumptions)
    body_lines.append(f"Tickets ({len(proposal.tickets)}):")
    body_lines.extend(ticket_lines)
    if proposal.truncated_ticket_count > 0:
        body_lines.append("")
        body_lines.append(
            "Note: model output exceeded the MVP max ticket limit; "
            f"only the first {MAX_TICKETS} tickets are included "
            f"({proposal.truncated_ticket_count} additional ticket(s) omitted)."
        )
    body_lines.append("")
    body_lines.append("Reply `approve` to write to Jira, `cancel` to discard, or describe edits.")
    return "\n".join(body_lines)


def _format_ticket_line(index: int, ticket: TicketSpec) -> str:
    repo = ""
    if ticket.repository and ticket.repository.lower() not in ticket.summary.lower():
        repo = f" [{ticket.repository}]"
    return f"  {index}. {ticket.summary}{repo}"


def _format_confirmation_message(result: JiraWriteResult) -> str:
    keys = ", ".join(result.created_ticket_keys) or "(none)"
    epic_prefix = (
        f"Created Jira epic: {result.created_epic_key}. "
        if result.created_epic_key
        else ""
    )
    if result.partial:
        failures = "; ".join(
            f"{item.spec.summary}: {item.reason}" for item in result.failed_items
        )
        return (
            f"{epic_prefix}Partially created: {keys}. Failures: {failures}. "
            "Reply with edits to retry the failed items."
        )
    readiness_note = ""
    if result.execution_ready_ticket_keys and (
        len(result.execution_ready_ticket_keys) < len(result.created_ticket_keys)
    ):
        ready_keys = ", ".join(result.execution_ready_ticket_keys)
        readiness_note = (
            f" Execution starts with: {ready_keys}; remaining tickets are created "
            "without ai-ready until the preceding pull request is merged or "
            "manually released."
        )
    return (
        f"{epic_prefix}Created Jira tickets: {keys}. "
        "The detection pipeline will pick them up automatically."
        f"{readiness_note}"
    )


def _format_jira_failure_message(result: JiraWriteResult) -> str:
    if result.unsupported_reason:
        return (
            "Could not write to Jira: "
            f"{result.unsupported_reason}. Reply with edits or `cancel`."
        )
    failures = _format_failures(result)
    return (
        "Could not create any Jira tickets. "
        f"Failures: {failures}. Reply with edits or `cancel`."
    )


def _format_failures(result: JiraWriteResult) -> str:
    if not result.failed_items:
        return "unknown Jira write failure"
    reasons = {item.reason for item in result.failed_items}
    if len(reasons) == 1:
        return next(iter(reasons))
    return "; ".join(
        f"{item.spec.summary}: {item.reason}" for item in result.failed_items
    )


__all__ = [
    "ApprovalFlow",
    "ApprovalOutcome",
    "ApprovalResult",
    "SlackPoster",
]

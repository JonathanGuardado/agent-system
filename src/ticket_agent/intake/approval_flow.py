"""Slack approval flow that turns proposals into Jira work."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ticket_agent.domain.intake import (
    IntakeMode,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.intent_resolver import IntakeIntentResolver
from ticket_agent.intake.jira_writer import JiraWriteResult, JiraWriter
from ticket_agent.intake.proposal_generator import (
    ProposalGenerator,
    ProposalRequest,
)
from ticket_agent.intake.proposal_store import ProposalStore


class SlackPoster(Protocol):
    """Boundary for posting messages back to Slack threads."""

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None: ...


class ApprovalOutcome(str, Enum):
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
    ) -> None:
        self._resolver = resolver
        self._generator = generator
        self._store = store
        self._jira_writer = jira_writer
        self._slack = slack
        self._repo_defaults: Mapping[str, Mapping[str, str]] = repo_defaults or {}
        self._emit = emit

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
            repo_defaults=self._repo_defaults,
        )
        draft = self._generator.generate(request)
        if draft.needs_clarification:
            assert draft.clarification is not None
            await self._post(channel, thread_ts, user_id, draft.clarification)
            self._emit_event(
                "intake.clarification_requested",
                {"user_id": user_id, "thread_ts": thread_ts},
            )
            return ApprovalResult(
                outcome=ApprovalOutcome.CLARIFICATION_REQUESTED,
                posted_message=draft.clarification,
            )

        assert draft.proposal is not None
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

        result = await self._jira_writer.write(proposal)
        if not result.created_ticket_keys:
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
        merged_text = _merge_revision_text(proposal, text)
        resolution = self._resolver.resolve(merged_text)
        request = ProposalRequest(
            slack_user_id=proposal.slack_user_id,
            slack_thread_ts=proposal.slack_thread_ts,
            text=merged_text,
            resolution=resolution,
            repo_defaults=self._repo_defaults,
        )
        draft = self._generator.generate(request, prior=proposal)
        if draft.needs_clarification:
            assert draft.clarification is not None
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

        assert draft.proposal is not None
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
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return "edit"
    first_word = normalized.split()[0]
    if first_word in _APPROVE_WORDS and len(normalized.split()) <= 3:
        return "approve"
    if first_word in _CANCEL_WORDS and len(normalized.split()) <= 3:
        return "cancel"
    return "edit"


def _merge_revision_text(proposal: Proposal, edit_text: str) -> str:
    pieces: list[str] = []
    if proposal.summary:
        pieces.append(proposal.summary)
    pieces.append(edit_text.strip())
    if proposal.project_key and proposal.project_key not in edit_text:
        pieces.append(f"(project {proposal.project_key})")
    return "\n".join(piece for piece in pieces if piece)


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
    ticket_lines = [_format_ticket_line(index, ticket) for index, ticket in enumerate(proposal.tickets, start=1)]

    body_lines: list[str] = [
        header,
        f"Mode: {proposal.mode.value}",
        project_line,
    ]
    if epic_line:
        body_lines.append(epic_line)
    body_lines.append(f"Title: {proposal.title}")
    body_lines.append(f"Tickets ({len(proposal.tickets)}):")
    body_lines.extend(ticket_lines)
    body_lines.append("")
    body_lines.append("Reply `approve` to write to Jira, `cancel` to discard, or describe edits.")
    return "\n".join(body_lines)


def _format_ticket_line(index: int, ticket: TicketSpec) -> str:
    repo = f" [{ticket.repository}]" if ticket.repository else ""
    return f"  {index}. {ticket.summary}{repo}"


def _format_confirmation_message(result: JiraWriteResult) -> str:
    keys = ", ".join(result.created_ticket_keys) or "(none)"
    if result.partial:
        failures = "; ".join(
            f"{item.spec.summary}: {item.reason}" for item in result.failed_items
        )
        return (
            f"Partially created: {keys}. Failures: {failures}. "
            "Reply with edits to retry the failed items."
        )
    return (
        f"Created Jira tickets: {keys}. "
        "The detection pipeline will pick them up automatically."
    )


def _format_jira_failure_message(result: JiraWriteResult) -> str:
    if result.unsupported_reason:
        return (
            "Could not write to Jira: "
            f"{result.unsupported_reason}. Reply with edits or `cancel`."
        )
    failures = "; ".join(
        f"{item.spec.summary}: {item.reason}" for item in result.failed_items
    )
    return (
        "Could not create any Jira tickets. "
        f"Failures: {failures}. Reply with edits or `cancel`."
    )


__all__ = [
    "ApprovalFlow",
    "ApprovalOutcome",
    "ApprovalResult",
    "SlackPoster",
]

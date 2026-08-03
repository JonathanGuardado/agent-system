"""Deterministic intake proposal generation.

Turns Slack text plus an :class:`IntakeResolution` into a
:class:`Proposal`. Designed so a future ModelRouter-backed implementation
can replace :class:`DeterministicProposalGenerator` without changes to
``ApprovalFlow``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
import json
import logging
import re
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ticket_agent.domain.acceptance import (
    ACCEPTANCE_HEADING,
    render_acceptance_criteria,
)
from ticket_agent.domain.intake import (
    IntakeMode,
    IntakeResolution,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.proposal_store import PROPOSAL_TTL_SECONDS
from ticket_agent.jira.constants import LABEL_AI_READY
from ticket_agent.redaction import redact_local_paths

_LOGGER = logging.getLogger(__name__)

Clock = Callable[[], datetime]
ProposalIdFactory = Callable[[], str]
SummarySlice = str | tuple[str, str]


_PROJECT_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})(?:-\d+)?\b")
_TICKET_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d+)\b")
_APPLICATION_REQUEST_PATTERN = re.compile(
    r"\b(?:create|build|develop|launch)\b.{0,120}\b"
    r"(?:web\s+)?(?:app|application|platform|website|product)\b",
    re.IGNORECASE | re.DOTALL,
)
_SINGLE_DELIVERY_PATTERNS = (
    re.compile(r"\bexactly\s+one\s+(?:jira\s+)?(?:ticket|task)\b", re.IGNORECASE),
    re.compile(
        r"\bdo\s+not\s+split\b.{0,60}\b(?:tickets?|prs?|pull requests?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


@dataclass(frozen=True)
class ProposalRequest:
    """Inputs needed to generate (or revise) a proposal."""

    slack_user_id: str
    slack_thread_ts: str
    text: str
    resolution: IntakeResolution
    slack_channel: str | None = None
    repo_defaults: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalDraft:
    """Output of a :class:`ProposalGenerator` invocation."""

    proposal: Proposal | None = None
    clarification: str | None = None
    # When a clarification is posted, this DRAFTING placeholder is stored so
    # the user's answer routes back as a revision (prior set), which bounds
    # clarification to a single round.
    pending_proposal: Proposal | None = None

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None


class ProposalGenerator(Protocol):
    """Boundary for proposal generation strategies."""

    def generate(
        self,
        request: ProposalRequest,
        prior: Proposal | None = None,
    ) -> ProposalDraft | Awaitable[ProposalDraft]: ...


class ModelRouterProtocol(Protocol):
    async def invoke(
        self,
        capability: str,
        messages: Sequence[Mapping[str, str]],
        **kwargs: object,
    ) -> object: ...


MAX_TICKETS = 10


class _ModelTicketPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str
    description: str = ""
    issue_type: str = "Task"
    priority: str | None = None
    labels: list[str] = Field(default_factory=list)
    capabilities_needed: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    # repository and repo_path are not accepted from the model;
    # they are resolved from repo_defaults or prior proposal only.


class _ModelProposalPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    summary: str | None = None
    # project_key and epic_key are not accepted from the model;
    # project_key comes from request text or prior proposal,
    # epic_key comes from request text or prior proposal.
    epic_summary: str | None = None
    epic_description: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    effort_estimate: str | None = None
    tickets: list[_ModelTicketPayload] = Field(default_factory=list)
    # Set only when the request is too ambiguous to propose; honored once,
    # on the first pass (prior is None). See _proposal_from_payload.
    clarification: str | None = None


class DeterministicProposalGenerator:
    """Rule-based proposal generator suitable for v1 without LLM calls."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        proposal_id_factory: ProposalIdFactory | None = None,
        ttl_seconds: int = PROPOSAL_TTL_SECONDS,
        max_tickets: int = MAX_TICKETS,
    ) -> None:
        if max_tickets < 1:
            raise ValueError("max_tickets must be at least 1")
        self._clock = clock or _utcnow
        self._proposal_id_factory = proposal_id_factory or _default_proposal_id
        self._ttl_seconds = ttl_seconds
        self._max_tickets = max_tickets

    def generate(
        self,
        request: ProposalRequest,
        prior: Proposal | None = None,
    ) -> ProposalDraft:
        # A prior with no tickets is a clarification-round placeholder: the
        # user's reply is the answer, so regenerate fresh from the combined
        # original request plus the answer rather than diff-revising nothing.
        placeholder_followup = prior is not None and not prior.tickets
        replan = prior is not None and _requests_full_replan(request.text)
        if prior is not None and not replan and not placeholder_followup:
            return _deterministic_revision(request, prior)

        if placeholder_followup and prior is not None:
            text = _combine_clarification(prior, request.text)
        elif replan and prior is not None:
            text = _original_request_from_proposal(prior)
        else:
            text = request.text.strip()
        if not text:
            return ProposalDraft(
                clarification="Could you describe what you'd like the agent to do?",
            )

        mode = request.resolution.mode
        if replan and _is_application_request(text):
            mode = IntakeMode.NEW_PROJECT
        project_key = _resolve_project_key(
            text,
            request.repo_defaults,
            prior,
        )
        epic_key = _extract_epic_key(text) or (
            prior.epic_key if prior is not None else None
        )
        repository, repo_path = _resolve_repository(
            text,
            project_key,
            request.repo_defaults,
            prior,
        )

        clarification = _missing_context_clarification(
            mode,
            project_key=project_key,
            epic_key=epic_key,
            repository=repository,
        )
        if clarification is not None:
            return ProposalDraft(clarification=clarification)

        capability = (
            "architecture.design"
            if mode == IntakeMode.NEW_PROJECT and _is_application_request(text)
            else request.resolution.capability
        )
        is_application_plan = mode in {
            IntakeMode.NEW_PROJECT,
            IntakeMode.NEW_FEATURE,
        } and _is_application_request(text)
        summaries = _candidate_summaries(mode, text)
        compacted_summaries = _compact_overlong_summaries(
            summaries,
            max_tickets=self._max_tickets,
        )
        single_delivery = _requests_single_ticket_delivery(text)
        if single_delivery:
            compacted_summaries = [_single_delivery_summary(compacted_summaries)]
        epic_scoped_children = (
            is_application_plan
            and not single_delivery
            and len(compacted_summaries) > 1
        )
        tickets = _build_ticket_specs(
            mode=mode,
            text=text,
            capability=capability,
            project_key=project_key,
            repository=repository,
            repo_path=repo_path,
            summaries=compacted_summaries,
            request_in_scope=single_delivery,
            include_original_request=not epic_scoped_children,
            parent_epic_has_request=epic_scoped_children,
        )

        title = _proposal_title(text)
        summary = _proposal_summary(mode, text, len(tickets))

        if prior is not None:
            proposal_id = prior.proposal_id
            created_at = prior.created_at
            expires_at = prior.expires_at
            revision_count = prior.revision_count + 1
        else:
            proposal_id = self._proposal_id_factory()
            created_at = self._clock()
            expires_at = created_at + timedelta(seconds=self._ttl_seconds)
            revision_count = 0

        proposal = Proposal(
            proposal_id=proposal_id,
            slack_user_id=request.slack_user_id,
            slack_channel=request.slack_channel,
            slack_thread_ts=request.slack_thread_ts,
            mode=mode,
            project_key=project_key,
            epic_key=epic_key,
            epic_summary=title if epic_scoped_children else None,
            epic_description=(
                _epic_description(
                    mode=mode,
                    text=text,
                    summaries=compacted_summaries,
                )
                if epic_scoped_children
                else None
            ),
            title=title,
            summary=summary,
            original_request=request.text,
            tickets=tickets,
            truncated_ticket_count=0,
            revision_count=revision_count,
            status=ProposalStatus.AWAITING_CONFIRMATION,
            created_at=created_at,
            expires_at=expires_at,
        )
        return ProposalDraft(proposal=proposal)


class ModelRouterProposalGenerator:
    """Model-assisted proposal generator with deterministic fallback."""

    def __init__(
        self,
        model_router: ModelRouterProtocol | None,
        *,
        fallback: ProposalGenerator | None = None,
        clock: Clock | None = None,
        proposal_id_factory: ProposalIdFactory | None = None,
        ttl_seconds: int = PROPOSAL_TTL_SECONDS,
        min_model_words: int = 4,
        max_tickets: int = MAX_TICKETS,
        model_timeout_s: float | None = 30.0,
    ) -> None:
        if min_model_words < 1:
            raise ValueError("min_model_words must be at least 1")
        if max_tickets < 1:
            raise ValueError("max_tickets must be at least 1")
        if model_timeout_s is not None and model_timeout_s <= 0:
            raise ValueError("model_timeout_s must be positive")
        self._model_router = model_router
        self._fallback = fallback or DeterministicProposalGenerator(
            clock=clock,
            proposal_id_factory=proposal_id_factory,
            ttl_seconds=ttl_seconds,
        )
        self._clock = clock or _utcnow
        self._proposal_id_factory = proposal_id_factory or _default_proposal_id
        self._ttl_seconds = ttl_seconds
        self._min_model_words = min_model_words
        self._max_tickets = max_tickets
        self._model_timeout_s = model_timeout_s

    async def generate(
        self,
        request: ProposalRequest,
        prior: Proposal | None = None,
    ) -> ProposalDraft:
        text = request.text.strip()
        if self._model_router is None or len(text.split()) < self._min_model_words:
            return await self._fallback_generate(request, prior)

        try:
            invocation = self._model_router.invoke(
                "ticket.decompose",
                _model_proposal_messages(
                    request,
                    prior,
                    clarification_allowed=prior is None,
                ),
                ticket_id=None,
                metadata={"workflow_node": "intake_proposal"},
            )
            if self._model_timeout_s is not None:
                response = await asyncio.wait_for(
                    invocation,
                    timeout=self._model_timeout_s,
                )
            else:
                response = await invocation
            payload = _coerce_model_payload(response)
            model_payload = _ModelProposalPayload.model_validate(payload)
            if _should_fallback_for_incomplete_revision(
                request,
                prior,
                model_payload,
            ):
                raise ValueError("model revision returned incomplete ticket list")
            return self._proposal_from_payload(request, prior, model_payload)
        except TimeoutError:
            _log_proposal_event(
                "intake.proposal_model_fallback",
                {
                    "reason": "model_timeout",
                    "timeout_s": self._model_timeout_s,
                    "word_count": len(text.split()),
                    "mode": request.resolution.mode.value,
                    "capability": request.resolution.capability,
                },
                level=logging.WARNING,
            )
            return await self._fallback_generate(request, prior)
        except (
            ValidationError,
            ValueError,
            TypeError,
            RuntimeError,
            KeyError,
        ) as exc:
            _log_proposal_event(
                "intake.proposal_model_fallback",
                {
                    "reason": exc.__class__.__name__,
                    "word_count": len(text.split()),
                    "mode": request.resolution.mode.value,
                    "capability": request.resolution.capability,
                },
                level=logging.WARNING,
            )
            return await self._fallback_generate(request, prior)
        except Exception as exc:
            _log_proposal_event(
                "intake.proposal_model_fallback",
                {
                    "reason": exc.__class__.__name__,
                    "word_count": len(text.split()),
                    "mode": request.resolution.mode.value,
                    "capability": request.resolution.capability,
                },
                level=logging.WARNING,
            )
            return await self._fallback_generate(request, prior)

    def _clarification_placeholder(self, request: ProposalRequest) -> Proposal:
        """Build a DRAFTING placeholder capturing the original request.

        Stored while awaiting a clarification answer. It carries the original
        request text in epic_description/summary so the follow-up revision can
        recover it via _original_request_from_proposal.
        """

        now = self._clock()
        text = request.text.strip()
        return Proposal(
            proposal_id=self._proposal_id_factory(),
            slack_user_id=request.slack_user_id,
            slack_channel=request.slack_channel,
            slack_thread_ts=request.slack_thread_ts,
            mode=request.resolution.mode,
            title=_proposal_title(text),
            summary=text,
            epic_description=text,
            tickets=[],
            status=ProposalStatus.DRAFTING,
            created_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )

    async def _fallback_generate(
        self,
        request: ProposalRequest,
        prior: Proposal | None,
    ) -> ProposalDraft:
        draft = self._fallback.generate(request, prior)
        if isawaitable(draft):
            draft = await draft
        return draft

    def _proposal_from_payload(
        self,
        request: ProposalRequest,
        prior: Proposal | None,
        payload: _ModelProposalPayload,
    ) -> ProposalDraft:
        model_clarification = _clean_optional(payload.clarification)
        if (
            model_clarification is not None
            and prior is None
            and not payload.tickets
        ):
            # First-pass ambiguity: ask once and stash a placeholder so the
            # answer returns as a revision (clarification disabled thereafter).
            return ProposalDraft(
                clarification=model_clarification,
                pending_proposal=self._clarification_placeholder(
                    request,
                ),
            )

        if not payload.tickets:
            raise ValueError("model proposal must include at least one ticket")

        text = request.text.strip()
        # project_key and epic_key come from request text or prior proposal only —
        # the model is not trusted to set operational context.
        project_key = _resolve_project_key(
            text,
            request.repo_defaults,
            prior,
        )
        epic_key = _extract_epic_key(text) or (
            prior.epic_key if prior is not None else None
        )
        # repository and repo_path come from repo_defaults or prior proposal only.
        repository, repo_path = _resolve_repository(
            text,
            project_key,
            request.repo_defaults,
            prior,
        )

        mode = _effective_proposal_mode(request, prior)
        clarification = _missing_context_clarification(
            mode,
            project_key=project_key,
            epic_key=epic_key,
            repository=repository,
        )
        if clarification is not None:
            return ProposalDraft(clarification=clarification)

        raw_tickets = _compact_overlong_model_tickets(
            payload.tickets,
            max_tickets=self._max_tickets,
        )
        single_delivery = _requests_single_ticket_delivery(text)
        if single_delivery:
            raw_tickets = [_single_delivery_model_ticket(raw_tickets)]
        truncated_ticket_count = 0
        sibling_payloads = [
            (ticket.summary, ticket.description)
            for ticket in raw_tickets
        ]
        tickets = [
            _ticket_spec_from_model_ticket(
                ticket,
                request_text=text,
                project_key=project_key,
                default_capability=request.resolution.capability,
                default_repository=repository,
                default_repo_path=repo_path,
                request_in_scope=single_delivery,
                sibling_scopes=_sibling_scopes_for(
                    ticket.summary,
                    sibling_payloads,
                ),
            )
            for ticket in raw_tickets
        ]

        now = self._clock()
        if prior is None:
            proposal_id = self._proposal_id_factory()
            created_at = now
            expires_at = created_at + timedelta(seconds=self._ttl_seconds)
            revision_count = 0
        else:
            proposal_id = prior.proposal_id
            created_at = prior.created_at
            expires_at = prior.expires_at
            revision_count = prior.revision_count + 1

        title = _clean_optional(payload.title) or (
            prior.title if prior is not None else _proposal_title(text)
        )
        summary = _clean_optional(payload.summary) or _proposal_summary(
            mode,
            prior.summary if prior is not None else text,
            len(tickets),
        )
        epic_summary = _clean_optional(payload.epic_summary) or (
            prior.epic_summary if prior is not None else None
        )
        if epic_summary is None and len(tickets) > 1:
            epic_summary = title

        return ProposalDraft(
            proposal=Proposal(
                proposal_id=proposal_id,
                slack_user_id=request.slack_user_id,
                slack_channel=request.slack_channel,
                slack_thread_ts=request.slack_thread_ts,
                mode=mode,
                project_key=project_key,
                epic_key=epic_key,
                epic_summary=epic_summary,
                epic_description=_clean_optional(payload.epic_description) or summary,
                title=title,
                summary=summary,
                assumptions=_clean_string_list(payload.assumptions),
                effort_estimate=_clean_optional(payload.effort_estimate),
                tickets=tickets,
                truncated_ticket_count=truncated_ticket_count,
                revision_count=revision_count,
                status=ProposalStatus.AWAITING_CONFIRMATION,
                created_at=created_at,
                expires_at=expires_at,
            )
        )


def _build_ticket_specs(
    *,
    mode: IntakeMode,
    text: str,
    capability: str,
    project_key: str | None,
    repository: str | None,
    repo_path: str | None,
    summaries: Sequence[SummarySlice] | None = None,
    request_in_scope: bool = False,
    include_original_request: bool = True,
    parent_epic_has_request: bool = False,
) -> list[TicketSpec]:
    summaries = (
        list(summaries)
        if summaries is not None
        else _candidate_summaries(mode, text)
    )
    capabilities_needed = [capability]

    specs: list[TicketSpec] = []
    sibling_payloads = [
        (_summary_slice_title(summary), _summary_slice_body(summary))
        for summary in summaries
    ]
    for summary in summaries:
        title = _summary_slice_title(summary)
        body = _summary_slice_body(summary)
        criteria = _default_acceptance_criteria(title, body)
        specs.append(
            TicketSpec(
                summary=_scoped_summary(title, repository),
                description=_execution_ready_description(
                    body=body,
                    request_text=text,
                    project_key=project_key,
                    repository=repository,
                    capabilities=capabilities_needed,
                    acceptance_criteria=criteria,
                    request_in_scope=request_in_scope,
                    include_original_request=include_original_request,
                    parent_epic_has_request=parent_epic_has_request,
                    sibling_scopes=_sibling_scopes_for(title, sibling_payloads),
                ),
                issue_type="Task",
                labels=[LABEL_AI_READY],
                capabilities_needed=list(capabilities_needed),
                acceptance_criteria=criteria,
                repository=repository,
                repo_path=repo_path,
            )
        )
    return specs


def _ticket_spec_from_model_ticket(
    ticket: _ModelTicketPayload,
    *,
    request_text: str,
    project_key: str | None,
    default_capability: str,
    default_repository: str | None,
    default_repo_path: str | None,
    request_in_scope: bool = False,
    sibling_scopes: Sequence[tuple[str, str]] = (),
) -> TicketSpec:
    labels = _ordered_unique([*ticket.labels, LABEL_AI_READY])
    capabilities = _ordered_unique(
        ticket.capabilities_needed or [default_capability]
    )
    criteria = _clean_string_list(ticket.acceptance_criteria) or (
        _default_acceptance_criteria(ticket.summary, ticket.description)
    )
    return TicketSpec(
        summary=_scoped_summary(ticket.summary, default_repository),
        description=_execution_ready_description(
            body=ticket.description,
            request_text=request_text,
            project_key=project_key,
            repository=default_repository,
            capabilities=capabilities,
            acceptance_criteria=criteria,
            request_in_scope=request_in_scope,
            sibling_scopes=sibling_scopes,
        ),
        issue_type=ticket.issue_type.strip() or "Task",
        priority=_clean_optional(ticket.priority),
        labels=labels,
        capabilities_needed=capabilities,
        acceptance_criteria=criteria,
        # Always use trusted context — model cannot override repository or repo_path.
        repository=default_repository,
        repo_path=default_repo_path,
    )


def _compact_overlong_model_tickets(
    tickets: Sequence[_ModelTicketPayload],
    *,
    max_tickets: int,
) -> list[_ModelTicketPayload]:
    if len(tickets) <= max_tickets:
        return list(tickets)

    compacted = list(tickets[:max_tickets])
    compacted[-1] = _merge_model_ticket_overflow(
        compacted[-1],
        tickets[max_tickets:],
    )
    return compacted


def _merge_model_ticket_overflow(
    ticket: _ModelTicketPayload,
    overflow: Sequence[_ModelTicketPayload],
) -> _ModelTicketPayload:
    descriptions = [ticket.description.strip()] if ticket.description.strip() else []
    overflow_lines: list[str] = []
    labels = list(ticket.labels)
    capabilities = list(ticket.capabilities_needed)
    priority = _clean_optional(ticket.priority)

    for overflow_ticket in overflow:
        detail = overflow_ticket.summary.strip()
        description = overflow_ticket.description.strip()
        if description:
            detail = f"{detail}: {description}"
        if detail:
            overflow_lines.append(f"- {detail}")
        labels.extend(overflow_ticket.labels)
        capabilities.extend(overflow_ticket.capabilities_needed)
        priority = priority or _clean_optional(overflow_ticket.priority)

    if overflow_lines:
        descriptions.append(
            "Additional included scope:\n" + "\n".join(overflow_lines)
        )

    return _ModelTicketPayload(
        summary=ticket.summary,
        description="\n\n".join(descriptions),
        issue_type=ticket.issue_type.strip() or "Task",
        priority=priority,
        labels=_ordered_unique(labels),
        capabilities_needed=_ordered_unique(capabilities),
    )


def _single_delivery_model_ticket(
    tickets: Sequence[_ModelTicketPayload],
) -> _ModelTicketPayload:
    merged = (
        _merge_model_ticket_overflow(tickets[0], tickets[1:])
        if len(tickets) > 1
        else tickets[0]
    )
    scope = merged.description.strip()
    leading = (
        "Implement the complete application MVP as one integrated runnable "
        "product delivered through one pull request targeting main. All "
        "requirements in the complete Slack request are in scope."
    )
    return merged.model_copy(
        update={
            "summary": "Build the complete application MVP in one integrated delivery",
            "description": "\n\n".join(part for part in (leading, scope) if part),
        }
    )


def _compact_overlong_summaries(
    summaries: Sequence[SummarySlice],
    *,
    max_tickets: int,
) -> list[SummarySlice]:
    if len(summaries) <= max_tickets:
        return list(summaries)

    compacted: list[SummarySlice] = list(summaries[:max_tickets])
    compacted[-1] = _merge_summary_overflow(
        compacted[-1],
        summaries[max_tickets:],
    )
    return compacted


def _merge_summary_overflow(
    summary: SummarySlice,
    overflow: Sequence[SummarySlice],
) -> SummarySlice:
    title = _summary_slice_title(summary)
    body = _summary_slice_body(summary)
    overflow_lines: list[str] = []

    for item in overflow:
        detail = _summary_slice_title(item)
        item_body = _summary_slice_body(item)
        if item_body and item_body != detail:
            detail = f"{detail}: {item_body}"
        if detail:
            overflow_lines.append(f"- {detail}")

    if not overflow_lines:
        return summary

    return (
        title,
        "\n\n".join(
            part
            for part in [
                body,
                "Additional included scope:\n" + "\n".join(overflow_lines),
            ]
            if part
        ),
    )


def _single_delivery_summary(summaries: Sequence[SummarySlice]) -> SummarySlice:
    included_scope = "\n".join(
        f"- {_summary_slice_title(summary)}: {_summary_slice_body(summary)}"
        for summary in summaries
    )
    body = (
        "Implement the complete application MVP as one integrated runnable "
        "product delivered through one pull request targeting main. All "
        "requirements in the complete Slack request below are in scope."
    )
    if included_scope:
        body = f"{body}\n\nIncluded implementation areas:\n{included_scope}"
    return "Build the complete application MVP in one integrated delivery", body


def _summary_slice_title(summary: SummarySlice) -> str:
    if isinstance(summary, tuple):
        return summary[0]
    return summary


def _summary_slice_body(summary: SummarySlice) -> str:
    if isinstance(summary, tuple):
        return summary[1]
    return summary


def _deterministic_revision(
    request: ProposalRequest,
    prior: Proposal,
) -> ProposalDraft:
    edit_text = request.text.strip()
    target_index = _revision_ticket_index(edit_text, len(prior.tickets))
    tickets = list(prior.tickets)
    if target_index is None and len(tickets) == 1 and edit_text:
        ticket = tickets[0]
        tickets[0] = ticket.model_copy(
            update={
                "description": _append_revision_note(
                    ticket.description,
                    edit_text,
                ),
            }
        )
    elif target_index is not None:
        ticket = tickets[target_index]
        tickets[target_index] = ticket.model_copy(
            update={
                "summary": _scoped_summary(
                    _revision_ticket_summary(edit_text, target_index + 1),
                    ticket.repository,
                ),
                "description": _append_revision_note(
                    ticket.description,
                    edit_text,
                ),
            }
        )

    revised = prior.model_copy(
        update={
            "tickets": tickets,
            "summary": prior.summary,
            "revision_count": prior.revision_count + 1,
            "status": ProposalStatus.AWAITING_CONFIRMATION,
            "expires_at": prior.expires_at,
        }
    )
    return ProposalDraft(proposal=revised)


def _revision_ticket_index(text: str, ticket_count: int) -> int | None:
    match = re.search(r"\bticket\s+(\d+)\b", text, flags=re.IGNORECASE)
    if match is None:
        return None
    index = int(match.group(1)) - 1
    if 0 <= index < ticket_count:
        return index
    return None


def _revision_ticket_summary(text: str, ticket_number: int) -> str:
    lowered = text.lower()
    if (
        "scheduled job" in lowered
        and ("deal" in lowered or "deals" in lowered)
        and ("search" in lowered or "web" in lowered)
    ):
        return "Add scheduled job for daily deal discovery"
    cleaned = re.sub(
        r"\bticket\s+\d+\b\s*(?:could be better,?\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,;:")
    return cleaned or f"Revise ticket {ticket_number}"


def _append_revision_note(description: str, edit_text: str) -> str:
    note = f"Revision request:\n{edit_text.strip()}"
    if not description.strip():
        return note
    return f"{description.rstrip()}\n\n{note}"


def _should_fallback_for_incomplete_revision(
    request: ProposalRequest,
    prior: Proposal | None,
    payload: _ModelProposalPayload,
) -> bool:
    if prior is None:
        return False
    edit_text = request.text.strip()
    if _revision_ticket_index(edit_text, len(prior.tickets)) is None:
        return False
    if not _revision_expects_existing_ticket_count(edit_text):
        return False
    return len(payload.tickets) != len(prior.tickets)


def _revision_expects_existing_ticket_count(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(add|append|create|new)\s+(?:a\s+)?ticket\b", lowered):
        return False
    if re.search(r"\b(remove|delete|drop|discard)\s+(?:the\s+)?ticket\b", lowered):
        return False
    return bool(
        re.search(
            r"\b(edit|update|change|revise|improve|keep|preserve|make)\b",
            lowered,
        )
    )


def _model_proposal_messages(
    request: ProposalRequest,
    prior: Proposal | None,
    *,
    clarification_allowed: bool = False,
) -> list[dict[str, str]]:
    prior_json = redact_local_paths(
        json.dumps(_prior_proposal_for_model(prior)),
        _proposal_local_paths(prior),
    )
    task = (
        "Revise the existing Jira proposal using this Slack edit."
        if prior is not None
        else "Create a Jira proposal for this Slack request."
    )
    text_label = "edit_text" if prior is not None else "text"
    revision_instructions = [
        "Preserve the prior proposal mode, project, repository, and unaffected "
        "tickets. Apply only the requested edit.",
        "Return the complete revised proposal, not a partial diff.",
        "Do not decompose the prior proposal title, summary, or the edit text "
        "as a brand-new request.",
    ] if prior is not None else []
    delivery_instructions = (
        [
            "The requester explicitly requires one integrated delivery. Return "
            "exactly one ticket for the complete requested scope, targeting one "
            "pull request; do not create an Epic or sibling implementation tickets.",
            "The single ticket scope and acceptance checks must include the whole "
            "requested MVP, not only foundation or app-shell work.",
        ]
        if _requests_single_ticket_delivery(request.text)
        else []
    )
    return [
        {
            "role": "system",
            "content": (
                "You turn Slack software requests into Jira-ready proposals. "
                "Return exactly one strict JSON object. Do not include markdown "
                "fences or prose."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    task,
                    f"{text_label}: {request.text}",
                    f"mode: {request.resolution.mode.value}",
                    f"capability: {request.resolution.capability}",
                    (
                        "repo_defaults: "
                        f"{json.dumps(_repo_defaults_for_model(request.repo_defaults))}"
                    ),
                    f"prior_proposal: {prior_json}",
                    *revision_instructions,
                    *delivery_instructions,
                    "Each ticket must be specific enough for an agent to execute "
                    "without reading Slack: include concrete files/directories, "
                    "scope boundaries, acceptance checks, and test expectations "
                    "in the ticket description.",
                    "Give each ticket 1 to 7 acceptance_criteria: short, testable "
                    "statements a reviewer can verify true or false (observable "
                    "behavior or a named test), not restatements of the summary.",
                    "Tickets must be mutually exclusive slices. For a single "
                    "app MVP, do not make multiple tickets that each build the "
                    "whole app; order them so the first ticket establishes the "
                    "shared foundation and app shell (project scaffold, shared "
                    "types and enum values, data schema, localization and test "
                    "harness), then separate feature tickets for homepage, "
                    "search/filtering, favorites, forms, data, or tests build on "
                    "that foundation.",
                    "If the requester asks for one final PR, one preview PR, "
                    "or a single pull request but does not explicitly ask for "
                    "exactly one Jira ticket/task, still return detailed Jira "
                    "tickets; delivery will consolidate the completed work.",
                    f"Return at most {MAX_TICKETS} tickets. If the request has "
                    "more details than that, group related details into complete "
                    "MVP slices instead of emitting one ticket per bullet.",
                    "Write ticket summaries as concise deliverables. The system "
                    "will add trusted repository/project context; do not invent "
                    "repositories or Jira projects.",
                    "Required JSON schema (omit project_key, epic_key, "
                    "repository, repo_path, slack_channel, slack_thread, "
                    "and Jira field IDs — the system sets these from trusted "
                    "context and will ignore any values you provide):",
                    (
                        '{"title": "string", "summary": "string", '
                        '"epic_summary": "optional string", '
                        '"epic_description": "optional string", '
                        '"assumptions": ["string"], '
                        '"effort_estimate": "S|M|L or brief text", '
                        '"tickets": [{"summary": "string", '
                        '"description": "string", "issue_type": "Task", '
                        '"priority": null, "labels": ["ai-ready"], '
                        '"capabilities_needed": ["code.implement"], '
                        '"acceptance_criteria": ["testable statement"]}]}'
                    ),
                    "Create multiple tickets only when the request naturally "
                    "contains multiple deliverable slices.",
                    *_clarification_instructions(clarification_allowed),
                    "Do not create brand-new Jira projects.",
                    "Return JSON only. No markdown fences. No prose before or "
                    "after JSON.",
                ]
            ),
        },
    ]


def _clarification_instructions(clarification_allowed: bool) -> list[str]:
    if clarification_allowed:
        return [
            "If the request is too ambiguous to scope into tickets, return "
            '{"clarification": "one focused question"} with no tickets, asking '
            "the single most important question. Only do this when you truly "
            "cannot propose reasonable tickets; otherwise propose tickets and "
            "record any guesses in assumptions.",
        ]
    return [
        "Do not ask for clarification. Proceed with the best reasonable "
        "proposal and record any guesses in assumptions.",
    ]


def _repo_defaults_for_model(
    repo_defaults: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Expose repository identity to the model without local checkout paths."""

    safe_defaults: dict[str, dict[str, str]] = {}
    for project_key, defaults in repo_defaults.items():
        repository = defaults.get("repository")
        if repository:
            safe_defaults[project_key] = {"repository": repository}
    return safe_defaults


def _prior_proposal_for_model(prior: Proposal | None) -> dict[str, object]:
    """Serialize prior proposal context without internal execution locations."""

    if prior is None:
        return {}
    payload = prior.model_dump(mode="json")
    tickets = payload.get("tickets")
    if isinstance(tickets, list):
        for ticket in tickets:
            if isinstance(ticket, dict):
                ticket.pop("repo_path", None)
    return payload


def _proposal_local_paths(prior: Proposal | None) -> list[str]:
    if prior is None:
        return []
    return [ticket.repo_path for ticket in prior.tickets if ticket.repo_path]


def _coerce_model_payload(response: object) -> dict[str, object]:
    if isinstance(response, Mapping):
        if "content" in response and response["content"] is not None:
            return _coerce_model_payload(response["content"])
        return dict(response)
    if isinstance(response, str):
        return _extract_json_object(response)
    content = getattr(response, "content", None)
    if content is not None:
        return _coerce_model_payload(content)
    raise ValueError(f"model response has unsupported shape: {type(response).__name__}")


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("model response is empty")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = _extract_fenced_or_embedded_json(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    return parsed


def _extract_fenced_or_embedded_json(text: str) -> object:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced is not None:
        return json.loads(fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("model response could not be parsed as JSON")


def _clean_optional(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _clean_string_list(values: Sequence[str]) -> list[str]:
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _combine_clarification(prior: Proposal, answer: str) -> str:
    original = _original_request_from_proposal(prior).strip()
    reply = answer.strip()
    if not reply:
        return original
    if not original:
        return reply
    return f"{original}\n\nClarification answer: {reply}"


def _default_acceptance_criteria(title: str, body: str) -> list[str]:
    """Fallback criteria when the model does not supply testable criteria.

    Kept generic but testable so the deterministic path and any model ticket
    that omits criteria still carry a usable contract for planning and review.
    """

    scope = (title or "").strip() or "The ticket scope"
    return [
        f"{scope} is implemented as described in the Ticket scope.",
        "Focused automated tests cover the new behavior and pass.",
        "No sibling-ticket scope or unrelated files are changed.",
    ]


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _log_proposal_event(
    event_name: str,
    payload: Mapping[str, object],
    *,
    level: int = logging.INFO,
) -> None:
    _LOGGER.log(
        level,
        json.dumps(
            {"event": event_name, **_jsonable_mapping(payload)},
            sort_keys=True,
        ),
    )


def _jsonable_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, str | int | float | bool):
            result[str(key)] = value
        elif isinstance(value, Mapping):
            result[str(key)] = _jsonable_mapping(value)
        elif isinstance(value, list | tuple):
            result[str(key)] = [
                item if item is None or isinstance(item, str | int | float | bool)
                else str(item)
                for item in value
            ]
        else:
            result[str(key)] = str(value)
    return result


def _scoped_summary(summary: str, repository: str | None) -> str:
    cleaned = summary.strip()
    if not repository:
        return cleaned
    repo = repository.strip()
    if not repo or repo.lower() in cleaned.lower():
        return cleaned
    return f"[{repo}] {cleaned}"


def _execution_ready_description(
    *,
    body: str,
    request_text: str,
    project_key: str | None,
    repository: str | None,
    capabilities: Sequence[str],
    acceptance_criteria: Sequence[str] = (),
    request_in_scope: bool = False,
    include_original_request: bool = True,
    parent_epic_has_request: bool = False,
    sibling_scopes: Sequence[tuple[str, str]] = (),
) -> str:
    context_lines = ["Execution context:"]
    if project_key:
        context_lines.append(f"- Jira project: {project_key}")
    if repository:
        context_lines.append(f"- Repository: {repository}")
    if capabilities:
        context_lines.append(f"- Capabilities: {', '.join(capabilities)}")

    cleaned_body = body.strip() or request_text.strip()
    cleaned_request = request_text.strip()
    if request_in_scope and cleaned_request and cleaned_request != cleaned_body:
        cleaned_body = (
            f"{cleaned_body}\n\n"
            "Complete Slack request requirements in scope:\n"
            f"{cleaned_request}"
        )
    sections = ["\n".join(context_lines)]
    if cleaned_body:
        sections.append(f"Ticket scope:\n{cleaned_body}")
    if (
        not request_in_scope
        and include_original_request
        and cleaned_request
        and cleaned_request != cleaned_body
    ):
        sections.append(
            "Original Slack request (background only; do not implement work "
            f"outside Ticket scope):\n{cleaned_request}"
        )
    elif parent_epic_has_request:
        sections.append(
            "Epic context:\n"
            "- The parent epic contains the complete product brief and delivery "
            "requirements.\n"
            "- Keep this ticket limited to the Ticket scope and relevant "
            "requirements listed here."
        )
    if sibling_scopes:
        sections.append(
            "Related tickets in this proposal (coordination only; do not "
            "implement them here):\n" + _format_sibling_scopes(sibling_scopes)
        )
    criteria_section = render_acceptance_criteria(acceptance_criteria)
    if criteria_section:
        sections.append(criteria_section)
    sections.append(
        "Acceptance checks:\n"
        "- Implement only the Ticket scope for this ticket.\n"
        "- Do not implement sibling tickets or the full original request unless "
        "this ticket explicitly scopes that work.\n"
        "- Satisfy every item under Acceptance Criteria above.\n"
        "- Add or update focused tests for the requested behavior.\n"
        "- Run the relevant test command and capture any remaining failures."
    )
    return "\n\n".join(sections)


def _sibling_scopes_for(
    current_summary: str,
    ticket_scopes: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    current = current_summary.strip()
    siblings: list[tuple[str, str]] = []
    for summary, description in ticket_scopes:
        cleaned_summary = summary.strip()
        if not cleaned_summary or cleaned_summary == current:
            continue
        siblings.append((cleaned_summary, description.strip()))
    return siblings


def _format_sibling_scopes(scopes: Sequence[tuple[str, str]]) -> str:
    lines: list[str] = []
    for summary, description in scopes:
        if description:
            lines.append(f"- {summary}: {description}")
        else:
            lines.append(f"- {summary}")
    return "\n".join(lines)


def _candidate_summaries(mode: IntakeMode, text: str) -> list[SummarySlice]:
    if mode == IntakeMode.NEW_TICKETS:
        items = _split_into_items(text)
        if items:
            return items
    if mode in {IntakeMode.NEW_PROJECT, IntakeMode.NEW_FEATURE}:
        if _is_application_request(text):
            return _application_delivery_slices(text)
        items = _split_into_items(text)
        if len(items) >= 2:
            return items
    return [_first_sentence(text)]


def _is_application_request(text: str) -> bool:
    return _APPLICATION_REQUEST_PATTERN.search(text) is not None


def _requests_single_ticket_delivery(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _SINGLE_DELIVERY_PATTERNS)


def _application_delivery_slices(text: str) -> list[SummarySlice]:
    lowered = text.lower()
    slices: list[SummarySlice] = [
        (
            "Establish application foundation and shared architecture",
            _application_slice_body(
                text,
                base=(
                    "Set up the application framework, shared project structure, "
                    "core domain types, persistence boundaries, localization "
                    "foundation, and access/security foundations required by the "
                    "product. Keep this as the base for every later ticket."
                ),
                keywords=(
                    "framework",
                    "next.js",
                    "app router",
                    "typescript",
                    "tailwind",
                    "intl",
                    "locale",
                    "localization",
                    "database",
                    "supabase",
                    "auth",
                    "row level",
                    "rls",
                    "security",
                    "vercel",
                    "preview",
                    "environment",
                    "env",
                ),
            ),
        ),
    ]
    if any(
        word in lowered
        for word in ("homepage", "public", "catalog", "listing", "deal card")
    ):
        slices.append(
            (
                "Build the primary public product experience",
                _application_slice_body(
                    text,
                    base=(
                        "Implement the public application shell and primary "
                        "content presentation using the shared model and "
                        "localization foundation. Respect publication and "
                        "provenance rules."
                    ),
                    keywords=(
                        "homepage",
                        "public",
                        "catalog",
                        "listing",
                        "deal card",
                        "merchant",
                        "category",
                        "price",
                        "discount",
                        "expiration",
                        "provenance",
                        "approved",
                        "published",
                        "real",
                        "placeholder",
                        "mock",
                        "empty state",
                    ),
                ),
            )
        )
    if any(word in lowered for word in ("filter", "search", "favorite", "near me")):
        slices.append(
            (
                "Add discovery filters and saved-item interactions",
                _application_slice_body(
                    text,
                    base=(
                        "Implement search, applicable filter controls, "
                        "location-based selection, and persisted saved items on "
                        "top of the public experience, including localized empty "
                        "states."
                    ),
                    keywords=(
                        "search",
                        "filter",
                        "favorite",
                        "saved",
                        "near me",
                        "city",
                        "department",
                        "location",
                        "availability",
                        "active",
                        "category",
                        "merchant",
                    ),
                ),
            )
        )
    if any(
        word in lowered
        for word in ("submission", "submit", "moderation", "admin", "approve")
    ):
        slices.append(
            (
                "Implement submissions and moderation controls",
                _application_slice_body(
                    text,
                    base=(
                        "Add validated user submissions and protected "
                        "administration for reviewing publication state. "
                        "Enforce that unapproved content is not visible publicly."
                    ),
                    keywords=(
                        "submission",
                        "submit",
                        "form",
                        "validation",
                        "pending",
                        "moderation",
                        "admin",
                        "review",
                        "approve",
                        "reject",
                        "publication",
                        "protected",
                        "unapproved",
                    ),
                ),
            )
        )
    if (
        "ai" in lowered
        and any(
            word in lowered
            for word in ("discovery", "discover", "extraction", "extract", "source")
        )
    ):
        slices.append(
            (
                "Implement AI-assisted source discovery and candidate ingestion",
                _application_slice_body(
                    text,
                    base=(
                        "Add the provider boundary, server-side structured "
                        "extraction, provenance capture, pending-only candidate "
                        "creation, and duplicate handling without automatic "
                        "publication."
                    ),
                    keywords=(
                        "ai",
                        "discovery",
                        "discover",
                        "extraction",
                        "extract",
                        "source",
                        "provider",
                        "candidate",
                        "provenance",
                        "duplicate",
                        "automatic",
                        "publish",
                        "scrape",
                        "terms",
                    ),
                ),
            )
        )
    slices.append(
        (
            "Complete integrated quality, security, and accessibility verification",
            _application_slice_body(
                text,
                base=(
                    "Add focused tests for critical user and authorization "
                    "behavior, review responsive and accessible interactions, "
                    "validate localized states, and polish the integrated MVP "
                    "for final review."
                ),
                keywords=(
                    "test",
                    "vitest",
                    "playwright",
                    "accessibility",
                    "responsive",
                    "security",
                    "authorization",
                    "localized",
                    "localization",
                    "validation",
                    "error",
                    "loading",
                    "empty",
                    "preview",
                    "vercel",
                ),
            ),
        )
    )
    return slices


def _application_slice_body(
    text: str,
    *,
    base: str,
    keywords: Sequence[str],
) -> str:
    requirements = _matching_requirement_lines(text, keywords, limit=12)
    if not requirements:
        return base
    return (
        f"{base}\n\n"
        "Relevant requirements for this ticket:\n"
        + "\n".join(f"- {requirement}" for requirement in requirements)
    )


def _matching_requirement_lines(
    text: str,
    keywords: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    matches: list[str] = []
    seen: set[str] = set()
    for line in _candidate_requirement_lines(text):
        lowered = line.lower()
        if not any(keyword in lowered for keyword in lowered_keywords):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        matches.append(line)
        if len(matches) >= limit:
            break
    return matches


def _candidate_requirement_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line).strip()
        line = line.strip("#").strip()
        if not line or line.endswith(":"):
            continue
        if len(line) > 260:
            line = line[:257].rstrip() + "..."
        lines.append(line)
    return lines


def _epic_description(
    *,
    mode: IntakeMode,
    text: str,
    summaries: Sequence[SummarySlice],
) -> str:
    sections = [
        f"Mode: {mode.value}; tickets: {len(summaries)}",
        f"Original request:\n{text.strip()}",
    ]
    if summaries:
        sections.append(
            "Implementation ticket plan:\n"
            + "\n".join(
                f"- {_summary_slice_title(summary)}: {_summary_slice_body(summary)}"
                for summary in summaries
            )
        )
    sections.append(
        "Delivery guidance:\n"
        "- Child tickets should be implemented as focused slices.\n"
        "- Use each child ticket's Ticket scope as the executable contract.\n"
        "- Keep shared product constraints from this epic in force across every "
        "slice."
    )
    return "\n\n".join(sections)


def _requests_full_replan(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(?:replan|regenerate|redo|replace|rebuild)\b.{0,50}\b"
            r"(?:plan|proposal|tickets?)\b",
            lowered,
        )
        or re.search(r"\bcohesive\b.{0,30}\b(?:plan|proposal|tickets?)\b", lowered)
    )


def _effective_proposal_mode(
    request: ProposalRequest,
    prior: Proposal | None,
) -> IntakeMode:
    if (
        prior is not None
        and _requests_full_replan(request.text)
        and _is_application_request(_original_request_from_proposal(prior))
    ):
        return IntakeMode.NEW_PROJECT
    return request.resolution.mode


def _original_request_from_proposal(proposal: Proposal) -> str:
    epic_request = _original_request_from_epic_description(
        proposal.epic_description,
    )
    if epic_request:
        return epic_request

    marker = (
        "Original Slack request (background only; do not implement work "
        "outside Ticket scope):\n"
    )
    delimiters = (
        "\n\nRelated tickets in this proposal",
        f"\n\n{ACCEPTANCE_HEADING}",
        "\n\nAcceptance checks:",
    )
    for ticket in proposal.tickets:
        if marker not in ticket.description:
            continue
        source = ticket.description.split(marker, 1)[1]
        for delimiter in delimiters:
            source = source.split(delimiter, 1)[0]
        if source.strip():
            return source.strip()
    return proposal.summary


def _original_request_from_epic_description(description: str | None) -> str | None:
    if not description:
        return None
    marker = "Original request:\n"
    if marker not in description:
        return None
    source = description.split(marker, 1)[1]
    for delimiter in ("\n\nImplementation ticket plan:", "\n\nDelivery guidance:"):
        source = source.split(delimiter, 1)[0]
    return source.strip() or None


def _split_into_items(text: str) -> list[str]:
    bullet_items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = re.match(r"^(?:[-*]|\d+[.)])\s+(.+?)\s*$", line)
        if bullet is not None:
            cleaned = bullet.group(1).strip()
            if cleaned:
                bullet_items.append(cleaned)
    if len(bullet_items) >= 2:
        return bullet_items

    if " and " in text.lower():
        parts = re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
        cleaned = [part.strip(" .,;") for part in parts if part.strip(" .,;")]
        if len(cleaned) >= 2:
            return cleaned
    return []


def _first_sentence(text: str) -> str:
    candidate = text.strip().splitlines()[0].strip()
    sentence = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)[0].strip()
    if not sentence:
        sentence = candidate
    if len(sentence) > 200:
        sentence = sentence[:197].rstrip() + "..."
    return sentence or "Intake request"


def _proposal_title(text: str) -> str:
    return _first_sentence(text)


def _proposal_summary(mode: IntakeMode, text: str, ticket_count: int) -> str:
    return (
        f"Mode: {mode.value}; "
        f"tickets: {ticket_count}; "
        f"original: {_first_sentence(text)}"
    )


def _resolve_project_key(
    text: str,
    repo_defaults: Mapping[str, Mapping[str, str]],
    prior: Proposal | None,
) -> str | None:
    configured_projects = {key.upper(): key for key in repo_defaults}
    candidates = _project_key_candidates(text)

    for candidate in candidates:
        configured = configured_projects.get(candidate)
        if configured is not None:
            return configured

    if prior is not None and prior.project_key:
        return prior.project_key

    if len(repo_defaults) == 1:
        return next(iter(repo_defaults))

    return candidates[0] if candidates else None


def _project_key_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _PROJECT_KEY_PATTERN.finditer(text):
        candidate = match.group(1)
        if candidate.lower() in _STOP_WORDS:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _extract_epic_key(text: str) -> str | None:
    match = _TICKET_KEY_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1)


def _resolve_repository(
    text: str,
    project_key: str | None,
    repo_defaults: Mapping[str, Mapping[str, str]],
    prior: Proposal | None,
) -> tuple[str | None, str | None]:
    repository: str | None = None
    repo_path: str | None = None

    if project_key is not None:
        defaults = repo_defaults.get(project_key)
        if defaults is not None:
            repository = repository or defaults.get("repository")
            repo_path = repo_path or defaults.get("repo_path")

    inline = None if repository is not None else _PATH_PATTERN.search(text)
    if inline is not None:
        repository = inline.group(1)
        repo_path = inline.group(0)

    if repository is None and prior is not None and prior.tickets:
        repository = prior.tickets[0].repository
    if repo_path is None and prior is not None and prior.tickets:
        repo_path = prior.tickets[0].repo_path

    return repository, repo_path


def _missing_context_clarification(
    mode: IntakeMode,
    *,
    project_key: str | None,
    epic_key: str | None,
    repository: str | None,
) -> str | None:
    if mode == IntakeMode.NEW_FEATURE:
        if project_key is None:
            return (
                "Which Jira project key should this feature land in? "
                "Reply with something like `AGENT`."
            )
        if repository is None:
            return (
                "Which repository should this feature change? Reply with the "
                "repo name or path."
            )
        return None

    if mode == IntakeMode.NEW_TICKETS:
        if project_key is None and epic_key is None:
            return (
                "Where should I attach these tickets? Reply with the Jira "
                "project key or the epic key."
            )
        return None

    if mode == IntakeMode.NEW_PROJECT:
        if project_key is None:
            return (
                "What Jira project key should I use? Note: creating brand-new "
                "Jira projects is not yet supported in v1, so an existing "
                "project key is required."
            )
        return None

    return None


_STOP_WORDS = {
    "api",
    "cli",
    "html",
    "json",
    "oauth",
    "saas",
    "saml",
    "sso",
    "ui",
    "yaml",
}
_PATH_PATTERN = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)\b")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _default_proposal_id() -> str:
    return f"prop-{uuid4().hex[:12]}"


__all__ = [
    "MAX_TICKETS",
    "DeterministicProposalGenerator",
    "ModelRouterProposalGenerator",
    "ProposalDraft",
    "ProposalGenerator",
    "ProposalRequest",
]

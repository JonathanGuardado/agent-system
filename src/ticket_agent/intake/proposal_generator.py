"""Deterministic intake proposal generation.

Turns Slack text plus an :class:`IntakeResolution` into a
:class:`Proposal`. Designed so a future ModelRouter-backed implementation
can replace :class:`DeterministicProposalGenerator` without changes to
``ApprovalFlow``.
"""

from __future__ import annotations

import asyncio
import re
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from inspect import isawaitable
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ticket_agent.domain.intake import (
    IntakeMode,
    IntakeResolution,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.proposal_store import PROPOSAL_TTL_SECONDS
from ticket_agent.jira.constants import LABEL_AI_READY


_LOGGER = logging.getLogger(__name__)

Clock = Callable[[], datetime]
ProposalIdFactory = Callable[[], str]
SummarySlice = str | tuple[str, str]


_PROJECT_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})(?:-\d+)?\b")
_TICKET_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d+)\b")


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
        if prior is not None:
            return _deterministic_revision(request, prior)

        text = request.text.strip()
        if not text:
            return ProposalDraft(
                clarification="Could you describe what you'd like the agent to do?",
            )

        mode = request.resolution.mode
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

        capability = request.resolution.capability
        summaries = _candidate_summaries(mode, text)
        compacted_summaries = _compact_overlong_summaries(
            summaries,
            max_tickets=self._max_tickets,
        )
        tickets = _build_ticket_specs(
            mode=mode,
            text=text,
            capability=capability,
            project_key=project_key,
            repository=repository,
            repo_path=repo_path,
            summaries=compacted_summaries,
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
            title=title,
            summary=summary,
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
        model_timeout_s: float | None = 10.0,
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
                _model_proposal_messages(request, prior),
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

        clarification = _missing_context_clarification(
            request.resolution.mode,
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
            request.resolution.mode,
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
                mode=request.resolution.mode,
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
        specs.append(
            TicketSpec(
                summary=_scoped_summary(title, repository),
                description=_execution_ready_description(
                    body=body,
                    request_text=text,
                    project_key=project_key,
                    repository=repository,
                    repo_path=repo_path,
                    capabilities=capabilities_needed,
                    sibling_scopes=_sibling_scopes_for(title, sibling_payloads),
                ),
                issue_type="Task",
                labels=[LABEL_AI_READY],
                capabilities_needed=list(capabilities_needed),
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
    sibling_scopes: Sequence[tuple[str, str]] = (),
) -> TicketSpec:
    labels = _ordered_unique([*ticket.labels, LABEL_AI_READY])
    capabilities = _ordered_unique(
        ticket.capabilities_needed or [default_capability]
    )
    return TicketSpec(
        summary=_scoped_summary(ticket.summary, default_repository),
        description=_execution_ready_description(
            body=ticket.description,
            request_text=request_text,
            project_key=project_key,
            repository=default_repository,
            repo_path=default_repo_path,
            capabilities=capabilities,
            sibling_scopes=sibling_scopes,
        ),
        issue_type=ticket.issue_type.strip() or "Task",
        priority=_clean_optional(ticket.priority),
        labels=labels,
        capabilities_needed=capabilities,
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


def _compact_overlong_summaries(
    summaries: Sequence[str],
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
    if target_index is not None:
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
) -> list[dict[str, str]]:
    prior_json = "{}" if prior is None else json.dumps(prior.model_dump(mode="json"))
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
                    f"repo_defaults: {json.dumps(request.repo_defaults)}",
                    f"prior_proposal: {prior_json}",
                    *revision_instructions,
                    "Each ticket must be specific enough for an agent to execute "
                    "without reading Slack: include concrete files/directories, "
                    "scope boundaries, acceptance checks, and test expectations "
                    "in the ticket description.",
                    "Tickets must be mutually exclusive slices. For a single "
                    "app MVP, do not make multiple tickets that each build the "
                    "whole app; create one foundation/app-shell ticket and "
                    "separate feature tickets for homepage, search/filtering, "
                    "favorites, forms, data, or tests as needed.",
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
                        '"capabilities_needed": ["code.implement"]}]}'
                    ),
                    "Create multiple tickets only when the request naturally "
                    "contains multiple deliverable slices.",
                    "Do not create brand-new Jira projects.",
                    "Return JSON only. No markdown fences. No prose before or "
                    "after JSON.",
                ]
            ),
        },
    ]


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
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
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
    repo_path: str | None,
    capabilities: Sequence[str],
    sibling_scopes: Sequence[tuple[str, str]] = (),
) -> str:
    context_lines = ["Execution context:"]
    if project_key:
        context_lines.append(f"- Jira project: {project_key}")
    if repository:
        context_lines.append(f"- Repository: {repository}")
    if repo_path:
        context_lines.append(f"- Repository path: {repo_path}")
    if capabilities:
        context_lines.append(f"- Capabilities: {', '.join(capabilities)}")

    cleaned_body = body.strip() or request_text.strip()
    sections = ["\n".join(context_lines)]
    if cleaned_body:
        sections.append(f"Ticket scope:\n{cleaned_body}")
    if request_text.strip() and request_text.strip() != cleaned_body:
        sections.append(
            "Original Slack request (background only; do not implement work "
            f"outside Ticket scope):\n{request_text.strip()}"
        )
    if sibling_scopes:
        sections.append(
            "Related tickets in this proposal (coordination only; do not "
            "implement them here):\n" + _format_sibling_scopes(sibling_scopes)
        )
    sections.append(
        "Acceptance checks:\n"
        "- Implement only the Ticket scope for this ticket.\n"
        "- Do not implement sibling tickets or the full original request unless "
        "this ticket explicitly scopes that work.\n"
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


def _candidate_summaries(mode: IntakeMode, text: str) -> list[str]:
    if mode == IntakeMode.NEW_TICKETS:
        items = _split_into_items(text)
        if items:
            return items
    if mode in {IntakeMode.NEW_PROJECT, IntakeMode.NEW_FEATURE}:
        items = _split_into_items(text)
        if len(items) >= 2:
            return items
    return [_first_sentence(text)]


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
    return datetime.now(timezone.utc)


def _default_proposal_id() -> str:
    return f"prop-{uuid4().hex[:12]}"


__all__ = [
    "DeterministicProposalGenerator",
    "MAX_TICKETS",
    "ModelRouterProposalGenerator",
    "ProposalDraft",
    "ProposalGenerator",
    "ProposalRequest",
]

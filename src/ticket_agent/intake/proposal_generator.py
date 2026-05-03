"""Deterministic intake proposal generation.

Turns Slack text plus an :class:`IntakeResolution` into a
:class:`Proposal`. Designed so a future ModelRouter-backed implementation
can replace :class:`DeterministicProposalGenerator` without changes to
``ApprovalFlow``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import uuid4

from ticket_agent.domain.intake import (
    IntakeMode,
    IntakeResolution,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.proposal_store import PROPOSAL_TTL_SECONDS
from ticket_agent.jira.constants import LABEL_AI_READY


Clock = Callable[[], datetime]
ProposalIdFactory = Callable[[], str]


_PROJECT_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})(?:-\d+)?\b")
_TICKET_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d+)\b")


@dataclass(frozen=True)
class ProposalRequest:
    """Inputs needed to generate (or revise) a proposal."""

    slack_user_id: str
    slack_thread_ts: str
    text: str
    resolution: IntakeResolution
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
    ) -> ProposalDraft: ...


class DeterministicProposalGenerator:
    """Rule-based proposal generator suitable for v1 without LLM calls."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        proposal_id_factory: ProposalIdFactory | None = None,
        ttl_seconds: int = PROPOSAL_TTL_SECONDS,
    ) -> None:
        self._clock = clock or _utcnow
        self._proposal_id_factory = proposal_id_factory or _default_proposal_id
        self._ttl_seconds = ttl_seconds

    def generate(
        self,
        request: ProposalRequest,
        prior: Proposal | None = None,
    ) -> ProposalDraft:
        text = request.text.strip()
        if not text:
            return ProposalDraft(
                clarification="Could you describe what you'd like the agent to do?",
            )

        mode = request.resolution.mode
        project_key = _extract_project_key(text) or (
            prior.project_key if prior is not None else None
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
        tickets = _build_ticket_specs(
            mode=mode,
            text=text,
            capability=capability,
            repository=repository,
            repo_path=repo_path,
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
            slack_thread_ts=request.slack_thread_ts,
            mode=mode,
            project_key=project_key,
            epic_key=epic_key,
            title=title,
            summary=summary,
            tickets=tickets,
            revision_count=revision_count,
            status=ProposalStatus.AWAITING_CONFIRMATION,
            created_at=created_at,
            expires_at=expires_at,
        )
        return ProposalDraft(proposal=proposal)


def _build_ticket_specs(
    *,
    mode: IntakeMode,
    text: str,
    capability: str,
    repository: str | None,
    repo_path: str | None,
) -> list[TicketSpec]:
    summaries = _candidate_summaries(mode, text)
    capabilities_needed = [capability]

    specs: list[TicketSpec] = []
    for summary in summaries:
        specs.append(
            TicketSpec(
                summary=summary,
                description=text,
                issue_type="Task",
                labels=[LABEL_AI_READY],
                capabilities_needed=list(capabilities_needed),
                repository=repository,
                repo_path=repo_path,
            )
        )
    return specs


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
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if cleaned:
            items.append(cleaned)
    if len(items) >= 2:
        return items

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


def _extract_project_key(text: str) -> str | None:
    for match in _PROJECT_KEY_PATTERN.finditer(text):
        candidate = match.group(1)
        if candidate.lower() in _STOP_WORDS:
            continue
        return candidate
    return None


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

    inline = _PATH_PATTERN.search(text)
    if inline is not None:
        repository = inline.group(1)
        repo_path = inline.group(0)

    if project_key is not None:
        defaults = repo_defaults.get(project_key)
        if defaults is not None:
            repository = repository or defaults.get("repository")
            repo_path = repo_path or defaults.get("repo_path")

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


_STOP_WORDS = {"oauth", "api", "ui", "cli", "saas", "json", "yaml"}
_PATH_PATTERN = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)\b")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_proposal_id() -> str:
    return f"prop-{uuid4().hex[:12]}"


__all__ = [
    "DeterministicProposalGenerator",
    "ProposalDraft",
    "ProposalGenerator",
    "ProposalRequest",
]

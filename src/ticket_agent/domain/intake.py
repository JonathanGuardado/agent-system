"""Intake domain models for the Slack-driven proposal lifecycle."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class IntakeMode(StrEnum):
    """High-level intake intent that drives proposal shape and Jira writes."""

    NEW_PROJECT = "new_project"
    NEW_FEATURE = "new_feature"
    NEW_TICKETS = "new_tickets"
    BACKLOG_UPDATE = "backlog_update"
    DIRECT_TICKET = "direct_ticket"


class ProposalStatus(StrEnum):
    """Lifecycle states tracked by the proposal store."""

    DRAFTING = "drafting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class IntakeResolution(BaseModel):
    """Result of deterministic intent resolution for an intake message."""

    model_config = ConfigDict(frozen=True)

    mode: IntakeMode
    capability: str
    model_primary: str
    model_fallbacks: tuple[str, ...] = ()
    requires_clarification: bool = False
    clarification_question: str | None = None


class TicketSpec(BaseModel):
    """A single Jira ticket the intake layer wants the writer to create."""

    summary: str
    description: str = ""
    issue_type: str = "Task"
    priority: str | None = None
    labels: list[str] = Field(default_factory=list)
    capabilities_needed: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    repository: str | None = None
    repo_path: str | None = None


class Proposal(BaseModel):
    """A Slack-originated proposal that, once approved, becomes Jira work."""

    proposal_id: str
    slack_user_id: str
    slack_channel: str | None = None
    slack_thread_ts: str
    mode: IntakeMode
    project_key: str | None = None
    epic_key: str | None = None
    epic_summary: str | None = None
    epic_description: str | None = None
    created_epic_key: str | None = None
    title: str
    summary: str
    #: The requester's verbatim words, captured at intake.
    #:
    #: Kept separately from `summary` because `summary` is model-written. The
    #: semantic check in goal/semantic_check.py compares the compiled contract
    #: against *this*; comparing it against a model's summary would be checking
    #: a summary with a summary, which catches nothing.
    original_request: str = ""
    assumptions: list[str] = Field(default_factory=list)
    effort_estimate: str | None = None
    tickets: list[TicketSpec] = Field(default_factory=list)
    truncated_ticket_count: int = 0
    revision_count: int = 0
    status: ProposalStatus = ProposalStatus.DRAFTING
    created_at: datetime
    expires_at: datetime


class SlackPoster(Protocol):
    """Boundary for posting messages back to Slack threads.

    One definition, in the layer both the intake and orchestrator packages
    already depend on. There were two protocols of this name -- intake's
    required `thread_ts: str` and execution approval's allowed `str | None` --
    so the same object could not be passed to both, though the only
    implementation (`SlackSDKPoster`) accepts None and branches on it. The
    wider signature is the one that matches reality.
    """

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str | None,
        user_id: str,
        text: str,
    ) -> None: ...


__all__ = [
    "IntakeMode",
    "IntakeResolution",
    "Proposal",
    "ProposalStatus",
    "SlackPoster",
    "TicketSpec",
]

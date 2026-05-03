"""Intake domain models for the Slack-driven proposal lifecycle."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class IntakeMode(str, Enum):
    """High-level intake intent that drives proposal shape and Jira writes."""

    NEW_PROJECT = "new_project"
    NEW_FEATURE = "new_feature"
    NEW_TICKETS = "new_tickets"
    BACKLOG_UPDATE = "backlog_update"
    DIRECT_TICKET = "direct_ticket"


class ProposalStatus(str, Enum):
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
    repository: str | None = None
    repo_path: str | None = None


class Proposal(BaseModel):
    """A Slack-originated proposal that, once approved, becomes Jira work."""

    proposal_id: str
    slack_user_id: str
    slack_thread_ts: str
    mode: IntakeMode
    project_key: str | None = None
    epic_key: str | None = None
    title: str
    summary: str
    tickets: list[TicketSpec] = Field(default_factory=list)
    revision_count: int = 0
    status: ProposalStatus = ProposalStatus.DRAFTING
    created_at: datetime
    expires_at: datetime


__all__ = [
    "IntakeMode",
    "IntakeResolution",
    "Proposal",
    "ProposalStatus",
    "TicketSpec",
]

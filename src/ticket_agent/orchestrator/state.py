"""State carried through the ticket execution graph."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


WorkflowStatus = Literal[
    "new",
    "planned",
    "waiting_for_approval",
    "implementing",
    "testing",
    "reviewing",
    "opening_pull_request",
    "reporting",
    "completed",
    "escalated",
]


class TicketState(BaseModel):
    ticket_key: str
    summary: str
    description: str = ""
    repository: str | None = None
    current_node: str | None = None
    workflow_status: WorkflowStatus = "new"
    execution_approved: bool | None = None
    tests_passed: bool | None = None
    review_passed: bool | None = None
    pull_request_url: str | None = None
    escalation_reason: str | None = None
    visited_nodes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

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
    repo_path: str | None = None
    worktree_path: str | None = None
    slack_channel: str | None = None
    slack_thread_ts: str | None = None
    decomposition: dict | None = None
    current_node: str | None = None
    workflow_status: WorkflowStatus = "new"
    execution_approved: bool | None = None
    execution_approval_status: str | None = None
    implementation_attempts: int = 0
    max_attempts: int = 3
    implementation_result: dict | None = None
    branch_name: str | None = None
    lock_id: str | None = None
    tests_passed: bool | None = None
    test_result: dict | None = None
    review_passed: bool | None = None
    verification_result: dict | None = None
    pull_request_url: str | None = None
    escalation_reason: str | None = None
    error: str | None = None
    visited_nodes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

"""State carried through the ticket execution graph."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WorkflowStatus = Literal[
    "new",
    "planned",
    "waiting_for_approval",
    "implementing",
    "committing",
    "testing",
    "verifying",
    "reviewing",
    "opening_pull_request",
    "reporting",
    "completed",
    "escalated",
]


class TicketState(BaseModel):
    """Graph state.

    ``extra="forbid"`` is load-bearing, not tidiness. Pydantic's default of
    ``extra="ignore"`` means a node returning an undeclared field has its
    update *silently dropped*: the graph advances, the field is absent
    downstream, and the ticket escalates with no explanation. Forbidding
    extras turns that into a loud ``ValidationError`` at the point of the
    mistake. Adding a node field therefore requires declaring it here, and
    adding a workflow status requires extending ``WorkflowStatus`` above.
    """

    model_config = ConfigDict(extra="forbid")

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
    pull_request_base_branch: str | None = None
    lock_id: str | None = None
    tests_passed: bool | None = None
    test_result: dict | None = None
    review_passed: bool | None = None
    verification_result: dict | None = None
    #: Goal this ticket serves. The durable spine is keyed on it; the ticket
    #: graph only carries it so evidence and transcripts can be correlated.
    goal_id: str | None = None
    autonomy_mode: str | None = None
    autonomy_decision_digest: str | None = None
    #: The commit actually verified. Set by the COMMIT node, and the thing an
    #: attestation binds -- never re-derived from the worktree afterwards.
    candidate_sha: str | None = None
    verification_record: dict[str, Any] | None = None
    verification_attempts: int = 0
    pull_request_url: str | None = None
    escalation_reason: str | None = None
    error: str | None = None
    visited_nodes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

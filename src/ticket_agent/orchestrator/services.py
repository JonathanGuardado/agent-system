"""Service protocols for ticket workflow nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ticket_agent.orchestrator.state import TicketState


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    approved: bool
    status: str | None = None
    reason: str | None = None


class PlannerService(Protocol):
    async def plan(self, state: TicketState) -> dict[str, Any]: ...


class ApprovalService(Protocol):
    async def request_approval(self, state: TicketState) -> bool | ApprovalDecision: ...


class ImplementationService(Protocol):
    async def implement(self, state: TicketState) -> dict[str, Any]: ...


class TestService(Protocol):
    async def run_tests(self, state: TicketState) -> dict[str, Any]: ...


class ReviewService(Protocol):
    async def review(self, state: TicketState) -> dict[str, Any]: ...


class PullRequestService(Protocol):
    async def open_pull_request(self, state: TicketState) -> str: ...


class EscalationService(Protocol):
    async def escalate(self, state: TicketState, reason: str) -> None: ...

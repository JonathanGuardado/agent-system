"""Bridge from an approved Slack proposal to a stored goal contract.

Kept separate from `ApprovalFlow` so intake does not need to know about risk
policy, signing, or the semantic checker, and separate from
`GoalContractCompiler` so the compiler does not need to know what a Jira
proposal looks like.
"""

from __future__ import annotations

from typing import Any, Protocol

from ticket_agent.domain.intake import Proposal
from ticket_agent.goal.contract import (
    AuthorizationOutcome,
    GoalContractCompiler,
    SQLiteGoalContractStore,
)
from ticket_agent.goal.identity import normalize_goal_id
from ticket_agent.goal.types import Budgets


class GoalAuthorizer(Protocol):
    async def authorize(
        self, proposal: Proposal, write_result: Any = None
    ) -> AuthorizationOutcome | None: ...


class NullGoalAuthorizer:
    """The default. Records nothing."""

    __slots__ = ()

    async def authorize(
        self, proposal: Proposal, write_result: Any = None
    ) -> AuthorizationOutcome | None:
        return None


class ProposalGoalAuthorizer:
    """Compile, judge, sign, and store the contract for an approved proposal."""

    def __init__(
        self,
        compiler: GoalContractCompiler,
        store: SQLiteGoalContractStore | None = None,
        *,
        default_budgets: Budgets | None = None,
    ) -> None:
        self._compiler = compiler
        self._store = store
        self._default_budgets = default_budgets or Budgets()

    async def authorize(
        self, proposal: Proposal, write_result: Any = None
    ) -> AuthorizationOutcome | None:
        goal_id = _goal_id(proposal, write_result)

        outcome = await self._compiler.compile(
            goal_id=goal_id,
            original_request=_verbatim_request(proposal),
            objective=proposal.summary or proposal.title,
            acceptance_criteria=_criteria(proposal),
            user_id=proposal.slack_user_id,
            channel=proposal.slack_channel,
            thread_ts=proposal.slack_thread_ts,
            repositories=_repositories(proposal),
            non_goals=(),
            budgets=self._default_budgets,
        )

        if self._store is not None:
            self._store.save_outcome(outcome)

        return outcome


def _verbatim_request(proposal: Proposal) -> str:
    """The requester's own words.

    Falls back to the proposal summary only when the verbatim text is absent
    -- for proposals created before it was captured. That fallback is a
    *model-written* summary, so the semantic check comparing against it is
    much weaker; the marker makes that visible rather than silent.
    """

    if proposal.original_request.strip():
        return proposal.original_request
    return f"[verbatim request unavailable; summary only] {proposal.summary}"


def _criteria(proposal: Proposal) -> list[str]:
    """Every acceptance criterion across the proposal's tickets, de-duplicated."""

    seen: set[str] = set()
    ordered: list[str] = []
    for ticket in proposal.tickets:
        for criterion in ticket.acceptance_criteria:
            text = criterion.strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
    if not ordered:
        # A contract requires at least one criterion. Rather than invent one,
        # state the absence so it reads as a gap in review.
        ordered.append(f"Deliver: {proposal.title}")
    return ordered


def _repositories(proposal: Proposal) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ticket in proposal.tickets:
        name = (ticket.repository or "").strip()
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _goal_id(proposal: Proposal, write_result: Any) -> str:
    """The proposal id is identity; Jira keys are display metadata only."""

    del write_result
    return normalize_goal_id(proposal.proposal_id)


__all__ = [
    "GoalAuthorizer",
    "NullGoalAuthorizer",
    "ProposalGoalAuthorizer",
]

"""P7: acceptance-criteria pipeline (intake -> Jira -> plan -> review).

Covers the shared render/parse format, criteria attached to proposal tickets,
the single-round model clarification, and the planner/review changes that make
criteria drive planning coverage and review routing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ticket_agent.domain.acceptance import (
    ACCEPTANCE_HEADING,
    parse_acceptance_criteria,
    render_acceptance_criteria,
)
from ticket_agent.domain.intake import (
    IntakeMode,
    IntakeResolution,
    Proposal,
    ProposalStatus,
)
from ticket_agent.intake.proposal_generator import (
    DeterministicProposalGenerator,
    ModelRouterProposalGenerator,
    ProposalRequest,
    _model_proposal_messages,
    _original_request_from_proposal,
)
from ticket_agent.orchestrator.model_services import (
    ModelRouterPlannerService,
    ModelRouterReviewService,
    _planning_messages,
    _review_messages,
)
from ticket_agent.orchestrator.state import TicketState

# ---------------------------------------------------------------------------
# Shared acceptance-criteria format
# ---------------------------------------------------------------------------


def test_render_then_parse_round_trips():
    criteria = [
        "Homepage lists only approved deals",
        "Empty state is shown when no deals exist",
    ]
    rendered = render_acceptance_criteria(criteria)
    assert rendered.startswith(ACCEPTANCE_HEADING)
    assert parse_acceptance_criteria(rendered) == criteria


def test_parse_accepts_dash_star_and_numbered_bullets():
    text = (
        "Acceptance Criteria:\n"
        "- dash item\n"
        "* star item\n"
        "1. numbered item\n"
    )
    assert parse_acceptance_criteria(text) == [
        "dash item",
        "star item",
        "numbered item",
    ]


def test_parse_stops_at_blank_or_nonbullet_line():
    text = (
        "Acceptance Criteria:\n"
        "- first\n"
        "- second\n"
        "\n"
        "Acceptance checks:\n"
        "- not a criterion\n"
    )
    assert parse_acceptance_criteria(text) == ["first", "second"]


def test_parse_returns_empty_without_heading():
    assert parse_acceptance_criteria("Ticket scope:\n- do a thing") == []


def test_render_empty_when_no_criteria():
    assert render_acceptance_criteria([]) == ""
    assert render_acceptance_criteria(["   ", ""]) == ""


# ---------------------------------------------------------------------------
# Generator: criteria attached to tickets
# ---------------------------------------------------------------------------


def test_deterministic_generator_attaches_default_criteria():
    draft = DeterministicProposalGenerator(clock=_clock).generate(
        _request("Add pagination to the users endpoint")
    )
    assert draft.proposal is not None
    for ticket in draft.proposal.tickets:
        assert ticket.acceptance_criteria
        assert ACCEPTANCE_HEADING in ticket.description
        assert parse_acceptance_criteria(ticket.description) == (
            ticket.acceptance_criteria
        )


def test_model_generator_uses_model_supplied_criteria():
    router = _Router(
        {
            "title": "Users pagination",
            "summary": "Add pagination",
            "tickets": [
                {
                    "summary": "Paginate users endpoint",
                    "description": "Add page and page_size query params.",
                    "acceptance_criteria": [
                        "GET /users accepts page and page_size",
                        "A test covers the second page",
                    ],
                }
            ],
        }
    )
    draft = asyncio.run(
        ModelRouterProposalGenerator(
            router, clock=_clock, min_model_words=1
        ).generate(_request("Add pagination to the users endpoint"))
    )
    assert draft.proposal is not None
    ticket = draft.proposal.tickets[0]
    assert ticket.acceptance_criteria == [
        "GET /users accepts page and page_size",
        "A test covers the second page",
    ]
    assert "GET /users accepts page and page_size" in ticket.description


def test_model_generator_falls_back_to_default_criteria_when_omitted():
    router = _Router(
        {
            "title": "Users pagination",
            "summary": "Add pagination",
            "tickets": [
                {
                    "summary": "Paginate users endpoint",
                    "description": "Add page and page_size query params.",
                }
            ],
        }
    )
    draft = asyncio.run(
        ModelRouterProposalGenerator(
            router, clock=_clock, min_model_words=1
        ).generate(_request("Add pagination to the users endpoint"))
    )
    assert draft.proposal is not None
    assert draft.proposal.tickets[0].acceptance_criteria  # non-empty defaults


# ---------------------------------------------------------------------------
# Single-round clarification
# ---------------------------------------------------------------------------


def test_model_clarification_first_pass_returns_pending_placeholder():
    router = _Router({"clarification": "Which deal categories should launch first?"})
    draft = asyncio.run(
        ModelRouterProposalGenerator(
            router, clock=_clock, min_model_words=1
        ).generate(_request("Build a deals app"))
    )
    assert draft.needs_clarification
    assert draft.clarification == "Which deal categories should launch first?"
    assert draft.pending_proposal is not None
    assert draft.pending_proposal.tickets == []
    assert draft.pending_proposal.status is ProposalStatus.DRAFTING
    # Original request is recoverable for the follow-up revision.
    assert "Build a deals app" in (draft.pending_proposal.epic_description or "")


def test_clarification_offered_only_on_first_pass_prompt():
    first_pass = _model_proposal_messages(
        _request("Build a deals app"), None, clarification_allowed=True
    )[1]["content"]
    assert "too ambiguous" in first_pass

    revision = _model_proposal_messages(
        _request("use category and location"),
        _placeholder_prior(),
        clarification_allowed=False,
    )[1]["content"]
    assert "Do not ask for clarification" in revision


def test_model_revision_of_placeholder_reuses_identity_and_proposes():
    router = _Router(
        {
            "title": "Deals app",
            "summary": "Deals MVP",
            "tickets": [
                {
                    "summary": "Homepage of approved deals",
                    "description": "List approved deals.",
                    "acceptance_criteria": ["Homepage lists approved deals"],
                }
            ],
        }
    )
    prior = _placeholder_prior()
    draft = asyncio.run(
        ModelRouterProposalGenerator(
            router, clock=_clock, min_model_words=1
        ).generate(_request("use category and location"), prior=prior)
    )
    assert draft.proposal is not None
    assert draft.proposal.tickets  # real tickets now
    assert draft.proposal.proposal_id == prior.proposal_id
    assert draft.proposal.revision_count == prior.revision_count + 1


def test_original_request_recovery_excludes_criteria_section():
    draft = DeterministicProposalGenerator(clock=_clock).generate(
        _request("Add a settings page to the users dashboard")
    )
    assert draft.proposal is not None
    recovered = _original_request_from_proposal(draft.proposal)
    assert "settings page" in recovered.lower()
    assert ACCEPTANCE_HEADING not in recovered
    assert "Acceptance checks:" not in recovered


def test_deterministic_placeholder_followup_generates_real_tickets():
    prior = _placeholder_prior()
    draft = DeterministicProposalGenerator(clock=_clock).generate(
        _request("focus on supermarket and restaurant deals"),
        prior=prior,
    )
    assert draft.proposal is not None
    assert draft.proposal.tickets
    assert draft.proposal.proposal_id == prior.proposal_id


# ---------------------------------------------------------------------------
# Planner: criteria coverage
# ---------------------------------------------------------------------------


def test_planning_messages_include_criteria_and_coverage_schema():
    state = _state_with_criteria(["Homepage lists approved deals"])
    content = _planning_messages(state)[1]["content"]
    assert "acceptance_criteria" in content
    assert "criteria_coverage" in content
    assert "Homepage lists approved deals" in content


def test_plan_captures_criteria_coverage():
    criteria = ["Homepage lists approved deals", "Empty state when none"]
    state = _state_with_criteria(criteria)
    router = _DictRouter(
        {
            "plan": "Build the homepage",
            "files_to_modify": ["app/page.tsx"],
            "criteria_coverage": {
                criteria[0]: "render deal cards",
                criteria[1]: "render empty state",
            },
        }
    )
    result = asyncio.run(ModelRouterPlannerService(router).plan(state))
    assert result["acceptance_criteria"] == criteria
    assert result["criteria_coverage"][criteria[0]] == "render deal cards"


# ---------------------------------------------------------------------------
# Review: per-criterion verdicts drive routing
# ---------------------------------------------------------------------------


def test_review_messages_include_criteria_and_verdict_schema():
    state = _state_with_criteria(["Homepage lists approved deals"])
    content = _review_messages(state)[1]["content"]
    assert "criteria_verdicts" in content
    assert "Homepage lists approved deals" in content


def test_review_rejects_when_a_criterion_is_unmet():
    criteria = ["Homepage lists approved deals", "Empty state when none"]
    state = _state_with_criteria(criteria)
    router = _DictRouter(
        {
            "passed": True,
            "reasoning": "Looks fine",
            "criteria_verdicts": [
                {"criterion": criteria[0], "met": True, "evidence": "cards render"},
                {"criterion": criteria[1], "met": False, "evidence": "no empty state"},
            ],
        }
    )
    result = asyncio.run(ModelRouterReviewService(router).review(state))
    assert result["passed"] is False
    assert result["status"] == "rejected"
    assert any("Unmet acceptance criterion" in issue for issue in result["issues"])


def test_review_passes_when_all_criteria_met():
    criteria = ["Homepage lists approved deals"]
    state = _state_with_criteria(criteria)
    router = _DictRouter(
        {
            "passed": True,
            "reasoning": "All good",
            "criteria_verdicts": [
                {"criterion": criteria[0], "met": True, "evidence": "cards render"},
            ],
        }
    )
    result = asyncio.run(ModelRouterReviewService(router).review(state))
    assert result["passed"] is True
    assert result["status"] == "approved"
    assert result["criteria_verdicts"][0]["met"] is True


def test_review_missing_verdict_met_flag_counts_as_unmet():
    criteria = ["Homepage lists approved deals"]
    state = _state_with_criteria(criteria)
    router = _DictRouter(
        {
            "passed": True,
            "criteria_verdicts": [
                {"criterion": criteria[0], "evidence": "unclear"},
            ],
        }
    )
    result = asyncio.run(ModelRouterReviewService(router).review(state))
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clock() -> datetime:
    return datetime(2026, 5, 3, 12, 0, tzinfo=UTC)


def _request(
    text: str,
    *,
    mode: IntakeMode = IntakeMode.NEW_FEATURE,
    capability: str = "code.implement",
) -> ProposalRequest:
    return ProposalRequest(
        slack_user_id="U1",
        slack_thread_ts="t1",
        text=text,
        resolution=IntakeResolution(
            mode=mode,
            capability=capability,
            model_primary="deepseek-v4-pro",
        ),
        repo_defaults={
            "AGENT": {"repository": "agent-system", "repo_path": "/home/agent"},
        },
    )


def _placeholder_prior() -> Proposal:
    now = _clock()
    return Proposal(
        proposal_id="prop-0000000000af",
        slack_user_id="U1",
        slack_thread_ts="t1",
        mode=IntakeMode.NEW_FEATURE,
        project_key="AGENT",
        title="Build a deals app",
        summary="Build a deals app",
        epic_description="Build a deals app",
        tickets=[],
        status=ProposalStatus.DRAFTING,
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )


def _state_with_criteria(criteria: list[str]) -> TicketState:
    description = "\n\n".join(
        [
            "Ticket scope:\nBuild the homepage of approved deals.",
            render_acceptance_criteria(criteria),
        ]
    )
    return TicketState(
        ticket_key="LAB-1",
        summary="Homepage",
        description=description,
        repository="ofertas-sv",
    )


@dataclass
class _Router:
    """Proposal-generator router double returning one fixed payload."""

    response: Any
    calls: list[str] = field(default_factory=list)

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        del messages, kwargs
        self.calls.append(capability)
        return self.response


@dataclass
class _DictRouter:
    """Planner/review router double returning one fixed dict payload."""

    payload: dict[str, Any]

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        del capability, messages, kwargs
        return self.payload

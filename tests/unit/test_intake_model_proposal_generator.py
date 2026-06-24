from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ticket_agent.domain.intake import IntakeMode, IntakeResolution
from ticket_agent.intake.proposal_generator import (
    DeterministicProposalGenerator,
    ModelRouterProposalGenerator,
    ProposalRequest,
)
from ticket_agent.jira.constants import LABEL_AI_READY


def test_model_proposal_generator_builds_multi_ticket_proposal():
    router = _Router(
        {
            "title": "Ship login MVP",
            "summary": "Implement the login MVP in two slices.",
            "project_key": "AGENT",
            "epic_summary": "Login MVP",
            "epic_description": "Track the login MVP work.",
            "assumptions": ["Existing auth provider is available"],
            "effort_estimate": "M",
            "tickets": [
                {
                    "summary": "Add login API",
                    "description": "Create the backend login API.",
                },
                {
                    "summary": "Add login UI",
                    "description": "Create the Slack-requested login UI.",
                    "labels": ["frontend"],
                },
            ],
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(
            router,
            clock=_clock,
            proposal_id_factory=lambda: "prop-model-1",
        ).generate(
            _request(
                "For AGENT project, ship login MVP with API and UI",
                mode=IntakeMode.NEW_TICKETS,
                capability="ticket.decompose",
            )
        )
    ).proposal

    assert proposal is not None
    assert proposal.proposal_id == "prop-model-1"
    assert proposal.title == "Ship login MVP"
    assert proposal.project_key == "AGENT"
    assert proposal.epic_summary == "Login MVP"
    assert proposal.assumptions == ["Existing auth provider is available"]
    assert proposal.effort_estimate == "M"
    assert [ticket.summary for ticket in proposal.tickets] == [
        "[agent-system] Add login API",
        "[agent-system] Add login UI",
    ]
    assert proposal.tickets[0].labels == [LABEL_AI_READY]
    assert proposal.tickets[1].labels == ["frontend", LABEL_AI_READY]
    assert proposal.tickets[0].repository == "agent-system"
    assert "Execution context:" in proposal.tickets[0].description
    assert "- Jira project: AGENT" in proposal.tickets[0].description
    assert "- Repository: agent-system" in proposal.tickets[0].description
    assert "Repository path:" not in proposal.tickets[0].description
    assert "/home/agent" not in proposal.tickets[0].description
    assert "Related tickets in this proposal" in proposal.tickets[0].description
    assert "Add login UI: Create the Slack-requested login UI" in (
        proposal.tickets[0].description
    )
    assert "Acceptance checks:" in proposal.tickets[0].description
    assert router.calls == ["ticket.decompose"]
    prompt = router.call_messages[0][1]["content"]
    assert "Tickets must be mutually exclusive slices" in prompt
    assert "do not make multiple tickets that each build the whole app" in prompt
    assert "Return at most 10 tickets" in prompt
    assert "/home/agent" not in prompt


def test_model_proposal_generator_falls_back_on_invalid_model_response():
    router = _Router("not json")

    proposal = asyncio.run(
        ModelRouterProposalGenerator(
            router,
            fallback=DeterministicProposalGenerator(
                clock=_clock,
                proposal_id_factory=lambda: "prop-fallback-1",
            ),
        ).generate(_request("Add OAuth login to AGENT"))
    ).proposal

    assert proposal is not None
    assert proposal.proposal_id == "prop-fallback-1"
    assert proposal.title == "Add OAuth login to AGENT"
    assert len(proposal.tickets) == 1
    assert proposal.tickets[0].labels == [LABEL_AI_READY]


def test_model_proposal_generator_falls_back_on_model_timeout():
    router = _HangingRouter()

    proposal = asyncio.run(
        ModelRouterProposalGenerator(
            router,
            fallback=DeterministicProposalGenerator(
                clock=_clock,
                proposal_id_factory=lambda: "prop-timeout-fallback",
            ),
            model_timeout_s=0.01,
        ).generate(_request("Add OAuth login to AGENT"))
    ).proposal

    assert proposal is not None
    assert proposal.proposal_id == "prop-timeout-fallback"
    assert proposal.project_key == "AGENT"
    assert proposal.tickets[0].summary == "[agent-system] Add OAuth login to AGENT"
    assert router.calls == ["ticket.decompose"]


def test_deterministic_proposal_generator_compacts_overlong_ticket_list():
    text = "\n".join(
        ["Break this AGENT work into tickets:"]
        + [f"- Ticket {index}" for index in range(1, 13)]
    )

    proposal = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-deterministic-truncated",
    ).generate(
        _request(
            text,
            mode=IntakeMode.NEW_TICKETS,
            capability="ticket.decompose",
        )
    ).proposal

    assert proposal is not None
    assert len(proposal.tickets) == 10
    assert proposal.truncated_ticket_count == 0
    description_text = "\n".join(ticket.description for ticket in proposal.tickets)
    assert "Ticket 11" in description_text
    assert "Ticket 12" in description_text


def test_deterministic_generator_uses_bullets_not_headings_as_ticket_slices():
    text = "\n".join(
        [
            "Create Ofertas SV for LAB.",
            "",
            "Core goal:",
            "Help users discover local deals.",
            "",
            "Main features:",
            "- Homepage with featured deals",
            "- Search and filters",
            "- Favorites page",
            "- Submit-a-deal form",
            "",
            "Design:",
            "Use Spanish UI copy and USD prices.",
        ]
    )

    proposal = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-bullet-slices",
    ).generate(
        _request(
            text,
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal

    assert proposal is not None
    assert [ticket.summary for ticket in proposal.tickets] == [
        "[agent-system] Homepage with featured deals",
        "[agent-system] Search and filters",
        "[agent-system] Favorites page",
        "[agent-system] Submit-a-deal form",
    ]
    assert all("Core goal:" not in ticket.summary for ticket in proposal.tickets)
    first_description = proposal.tickets[0].description
    assert "Ticket scope:\nHomepage with featured deals" in first_description
    assert "Original Slack request (background only" in first_description
    assert "Related tickets in this proposal" in first_description
    assert "- Search and filters" in first_description
    assert "Do not implement sibling tickets" in first_description


def test_deterministic_generator_groups_app_requirements_into_delivery_stages():
    text = """
Create a web application called Ofertas SV for LAB.

Core product principles:
- Public users should see only approved, current, real deals.
- Do not present invented or mock promotions as real offers.

Main features:
- Public homepage with real deal cards.
- Search, filters, favorites, and near me behavior.
- Submit-a-deal form and admin moderation.
- AI-assisted discovery and extraction from permitted sources.

Quality requirements:
- Include localization, security, accessibility, and tests.
"""

    proposal = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-app-stages",
    ).generate(
        _request(
            text,
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal

    assert proposal is not None
    summaries = [ticket.summary for ticket in proposal.tickets]
    assert summaries == [
        "[agent-system] Establish application foundation and shared architecture",
        "[agent-system] Build the primary public product experience",
        "[agent-system] Add discovery filters and saved-item interactions",
        "[agent-system] Implement submissions and moderation controls",
        "[agent-system] Implement AI-assisted source discovery and candidate ingestion",
        (
            "[agent-system] Complete integrated quality, security, "
            "and accessibility verification"
        ),
    ]
    assert all("Public users should see only" not in summary for summary in summaries)
    assert all("Framework:" not in summary for summary in summaries)
    assert proposal.epic_summary == (
        "Create a web application called Ofertas SV for LAB."
    )
    assert proposal.epic_description is not None
    assert "Original request:" in proposal.epic_description
    assert "Implementation ticket plan:" in proposal.epic_description
    assert "Public users should see only approved, current, real deals." in (
        proposal.epic_description
    )

    descriptions = [ticket.description for ticket in proposal.tickets]
    assert all(
        "Original Slack request (background only" not in description
        for description in descriptions
    )
    assert all("Epic context:" in description for description in descriptions)
    assert "Relevant requirements for this ticket:" in descriptions[1]
    assert "Public homepage with real deal cards." in descriptions[1]
    assert "Search, filters, favorites, and near me behavior." in descriptions[2]
    assert "Submit-a-deal form and admin moderation." in descriptions[3]
    assert "AI-assisted discovery and extraction from permitted sources." in (
        descriptions[4]
    )


def test_deterministic_generator_keeps_explicit_single_pr_app_as_one_ticket():
    text = """
Create a web application called Ofertas SV for LAB in exactly one Jira task
and one pull request to main. Do not split this delivery into separate tickets.

Main features:
- Public homepage for approved real deals.
- Search, filters, favorites, and near me behavior.
- Submit-a-deal form and admin moderation.
- AI-assisted discovery from permitted sources.
"""

    proposal = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-single-pr",
    ).generate(
        _request(
            text,
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal

    assert proposal is not None
    assert len(proposal.tickets) == 1
    assert proposal.tickets[0].summary == (
        "[agent-system] Build the complete application MVP in one integrated delivery"
    )
    description = proposal.tickets[0].description
    assert "Build the primary public product experience" in description
    assert "Implement AI-assisted source discovery" in description
    assert "Complete Slack request requirements in scope:" in description
    assert "Original Slack request (background only" not in description


def test_deterministic_generator_keeps_final_pr_request_as_detailed_plan():
    text = """
Create a web application called Ofertas SV for LAB.
I want detailed Jira tickets for the plan, but when everything is finished
open one single PR that I can preview.

Main features:
- Public homepage for approved real deals.
- Search, filters, favorites, and near me behavior.
- Submit-a-deal form and admin moderation.
"""

    proposal = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-final-pr-plan",
    ).generate(
        _request(
            text,
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal

    assert proposal is not None
    assert [ticket.summary for ticket in proposal.tickets] == [
        "[agent-system] Establish application foundation and shared architecture",
        "[agent-system] Build the primary public product experience",
        "[agent-system] Add discovery filters and saved-item interactions",
        "[agent-system] Implement submissions and moderation controls",
        (
            "[agent-system] Complete integrated quality, security, "
            "and accessibility verification"
        ),
    ]


def test_model_generator_folds_explicit_single_pr_delivery_into_one_ticket():
    text = (
        "Build the Ofertas SV app for LAB as a single pull request to main. "
        "Create exactly one Jira task; include public deals and admin moderation."
    )
    router = _Router(
        {
            "title": "Ofertas SV MVP",
            "tickets": [
                {
                    "summary": "Build foundation",
                    "description": "Create the application shell.",
                },
                {
                    "summary": "Add moderation",
                    "description": "Protect approval and publication controls.",
                },
            ],
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock, min_model_words=1).generate(
            _request(
                text,
                mode=IntakeMode.NEW_PROJECT,
                capability="architecture.design",
            )
        )
    ).proposal

    assert proposal is not None
    assert len(proposal.tickets) == 1
    assert proposal.tickets[0].summary == (
        "[agent-system] Build the complete application MVP in one integrated delivery"
    )
    assert "Add moderation" in proposal.tickets[0].description
    assert "Complete Slack request requirements in scope:" in (
        proposal.tickets[0].description
    )
    assert "Original Slack request (background only" not in (
        proposal.tickets[0].description
    )
    prompt = router.call_messages[0][1]["content"]
    assert "explicitly requires one integrated delivery" in prompt
    assert "exactly one ticket for the complete requested scope" in prompt


def test_model_generator_keeps_single_pr_request_as_multi_ticket_plan():
    text = (
        "Build the Ofertas SV app for LAB as a single pull request to main. "
        "Include public deals, search, favorites, and admin moderation."
    )
    router = _Router(
        {
            "title": "Ofertas SV MVP",
            "tickets": [
                {
                    "summary": "Build public deals",
                    "description": "Create public listing and detail views.",
                },
                {
                    "summary": "Add moderation",
                    "description": "Protect approval and publication controls.",
                },
            ],
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock, min_model_words=1).generate(
            _request(
                text,
                mode=IntakeMode.NEW_PROJECT,
                capability="architecture.design",
            )
        )
    ).proposal

    assert proposal is not None
    assert [ticket.summary for ticket in proposal.tickets] == [
        "[agent-system] Build public deals",
        "[agent-system] Add moderation",
    ]
    assert all(
        "Complete Slack request requirements in scope:" not in ticket.description
        for ticket in proposal.tickets
    )
    prompt = router.call_messages[0][1]["content"]
    assert "exactly one ticket for the complete requested scope" not in prompt
    assert "one final PR" in prompt
    assert "still return detailed Jira tickets" in prompt


def test_single_ticket_revision_fallback_appends_unnumbered_scope_edit():
    generator = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-single-revision",
    )
    prior = generator.generate(
        _request(
            (
                "Build the Ofertas SV app for LAB as exactly one Jira task and "
                "one pull request targeting main."
            ),
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal
    assert prior is not None

    revised = generator.generate(
        _request(
            "Add Data And Security Requirements: enforce Supabase Row Level Security.",
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        ),
        prior=prior,
    ).proposal

    assert revised is not None
    assert revised.tickets[0].summary == prior.tickets[0].summary
    assert "Revision request:" in revised.tickets[0].description
    assert "Data And Security Requirements" in revised.tickets[0].description
    assert "Supabase Row Level Security" in revised.tickets[0].description


def test_model_timeout_single_ticket_revision_retains_requested_scope_edit():
    generator = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-timeout-single-revision",
    )
    prior = generator.generate(
        _request(
            (
                "Build the Ofertas SV app for LAB as exactly one Jira task and "
                "one pull request targeting main."
            ),
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal
    assert prior is not None

    revised = asyncio.run(
        ModelRouterProposalGenerator(
            _HangingRouter(),
            fallback=generator,
            model_timeout_s=0.01,
        ).generate(
            _request(
                "Add Discovery Requirements: AI candidates must never auto-publish.",
                mode=IntakeMode.NEW_PROJECT,
                capability="architecture.design",
            ),
            prior=prior,
        )
    ).proposal

    assert revised is not None
    assert revised.tickets[0].summary == prior.tickets[0].summary
    assert "Discovery Requirements" in revised.tickets[0].description
    assert "never auto-publish" in revised.tickets[0].description


def test_replan_revision_rebuilds_bad_app_bullets_as_cohesive_stages():
    original = """
Create a web application called Ofertas SV for LAB.
- Public homepage and filters for approved real deals.
- Submit deals with admin moderation.
- AI discovery from permitted sources.
"""
    generator = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-replan",
    )
    prior = generator.generate(
        _request(
            original,
            mode=IntakeMode.NEW_FEATURE,
            capability="code.implement",
        )
    ).proposal
    assert prior is not None

    revised = generator.generate(
        _request(
            "Regenerate this proposal as cohesive implementation tickets.",
            mode=IntakeMode.NEW_FEATURE,
            capability="code.implement",
        ),
        prior=prior,
    ).proposal

    assert revised is not None
    assert revised.mode == IntakeMode.NEW_PROJECT
    assert revised.revision_count == 1
    assert revised.tickets[0].summary == (
        "[agent-system] Establish application foundation and shared architecture"
    )
    assert any(
        "AI-assisted source discovery" in ticket.summary
        for ticket in revised.tickets
    )


def test_model_replan_revision_promotes_app_request_to_new_project_mode():
    original = """
Create a web application called Ofertas SV for LAB.
- Public homepage for approved real deals.
- AI discovery from permitted sources.
"""
    prior = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-model-replan",
    ).generate(
        _request(
            original,
            mode=IntakeMode.NEW_FEATURE,
            capability="code.implement",
        )
    ).proposal
    assert prior is not None
    router = _Router(
        {
            "title": "Ofertas SV initiative",
            "tickets": [
                {
                    "summary": "Build foundation",
                    "description": "Set up the cohesive application.",
                },
                {
                    "summary": "Build discovery",
                    "description": "Add AI-assisted candidate discovery.",
                },
            ],
        }
    )

    revised = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock, min_model_words=1).generate(
            _request(
                "Regenerate this proposal as cohesive implementation tickets.",
                mode=IntakeMode.NEW_FEATURE,
                capability="code.implement",
            ),
            prior=prior,
        )
    ).proposal

    assert revised is not None
    assert revised.mode == IntakeMode.NEW_PROJECT


def test_model_proposal_generator_revision_preserves_prior_context():
    prior = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-prior",
    ).generate(_request("Add OAuth login to AGENT")).proposal
    assert prior is not None

    router = _Router(
        {
            "title": "Add SAML login",
            "summary": "Revise the login ticket to use SAML.",
            "tickets": [
                {
                    "summary": "Add SAML login",
                    "description": "Replace OAuth with SAML.",
                }
            ],
        }
    )

    revised = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock, min_model_words=1).generate(
            _request("Use SAML instead"),
            prior=prior,
        )
    ).proposal

    assert revised is not None
    assert revised.proposal_id == prior.proposal_id
    assert revised.revision_count == 1
    assert revised.project_key == "AGENT"
    assert revised.tickets[0].repository == "agent-system"
    assert revised.tickets[0].repo_path == "/home/agent"
    prompt = router.call_messages[0][1]["content"]
    assert "Revise the existing Jira proposal" in prompt
    assert "edit_text: Use SAML instead" in prompt
    assert "Return the complete revised proposal, not a partial diff." in prompt
    assert "/home/agent" not in prompt


def test_model_revision_falls_back_when_targeted_edit_returns_partial_proposal():
    prior_text = "\n".join(
        ["Create an AGENT project"]
        + [f"- Existing ticket {index}" for index in range(1, 11)]
    )
    prior = DeterministicProposalGenerator(
        clock=_clock,
        proposal_id_factory=lambda: "prop-prior",
    ).generate(
        _request(
            prior_text,
            mode=IntakeMode.NEW_PROJECT,
            capability="architecture.design",
        )
    ).proposal
    assert prior is not None
    assert len(prior.tickets) == 10

    router = _Router(
        {
            "title": "Mode: new_project; tickets: 10",
            "tickets": [
                {
                    "summary": "Mode: new_project; tickets: 10",
                    "description": "Create an AGENT project",
                },
                {
                    "summary": "promotions",
                    "description": "keep the first 9 tickets",
                },
                {
                    "summary": "the offers found will be the main source of data",
                    "description": "",
                },
            ],
        }
    )

    revised = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock, min_model_words=1).generate(
            _request(
                "keep the first 9 tickets, but edit the ticket 10, make it "
                "better: use a scheduled job for searching in the web for "
                "deals everyday and the offers found will be the main source "
                "of data",
                mode=IntakeMode.NEW_PROJECT,
                capability="architecture.design",
            ),
            prior=prior,
        )
    ).proposal

    assert revised is not None
    assert revised.proposal_id == prior.proposal_id
    assert revised.revision_count == 1
    assert revised.mode == IntakeMode.NEW_PROJECT
    assert len(revised.tickets) == 10
    assert revised.tickets[:9] == prior.tickets[:9]
    assert revised.tickets[9].summary == (
        "[agent-system] Add scheduled job for daily deal discovery"
    )
    assert "Revision request:" in revised.tickets[9].description
    assert "Mode: new_project; tickets: 10" not in revised.tickets[0].summary


def test_model_provided_project_key_cannot_override_trusted_context():
    """Model-returned project_key must be ignored; value must come from request text."""
    router = _Router(
        {
            "title": "Add feature",
            "summary": "Implement the feature.",
            "project_key": "ATTACKER",  # model tries to override
            "tickets": [{"summary": "Implement feature", "description": ""}],
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(
            router,
            clock=_clock,
        ).generate(_request("Add feature to AGENT project"))
    ).proposal

    assert proposal is not None
    assert proposal.project_key == "AGENT"


def test_model_provided_repository_cannot_override_trusted_context():
    """Model-returned repository and repo_path in tickets must be ignored."""
    router = _Router(
        {
            "title": "Add feature",
            "tickets": [
                {
                    "summary": "Implement feature",
                    "description": "",
                    "repository": "attacker-repo",
                    "repo_path": "/evil/path",
                }
            ],
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(
            router,
            clock=_clock,
        ).generate(_request("Add feature to AGENT project"))
    ).proposal

    assert proposal is not None
    assert proposal.tickets[0].repository == "agent-system"
    assert proposal.tickets[0].repo_path == "/home/agent"


def test_single_configured_project_wins_over_source_paths_and_html_word():
    router = _Router(
        {
            "title": "Build Validation App",
            "summary": "Create a tiny validation app.",
            "tickets": [
                {
                    "summary": "Create validation app",
                    "description": "Add src/validation_app with an HTML renderer.",
                }
            ],
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock).generate(
            _request(
                "Create a tiny Python package under src/validation_app/ with "
                "an HTML renderer",
                repo_defaults={
                    "SCRUM": {
                        "repository": "agent-system",
                        "repo_path": "/home/agent-system",
                    }
                },
            )
        )
    ).proposal

    assert proposal is not None
    assert proposal.project_key == "SCRUM"
    assert proposal.tickets[0].repository == "agent-system"
    assert proposal.tickets[0].repo_path == "/home/agent-system"


def test_model_tickets_compacted_to_max_tickets():
    """Model output exceeding max_tickets is compacted without omitting work."""
    seven_tickets = [
        {"summary": f"Ticket {i}", "description": ""} for i in range(7)
    ]
    router = _Router(
        {
            "title": "Big project",
            "tickets": seven_tickets,
        }
    )

    proposal = asyncio.run(
        ModelRouterProposalGenerator(
            router,
            clock=_clock,
            max_tickets=5,
        ).generate(_request("For AGENT project, do many things"))
    ).proposal

    assert proposal is not None
    assert len(proposal.tickets) == 5
    assert proposal.truncated_ticket_count == 0
    description_text = "\n".join(ticket.description for ticket in proposal.tickets)
    assert "Ticket 5" in description_text
    assert "Ticket 6" in description_text


def test_model_tickets_default_max_is_ten():
    """Default max_tickets is MAX_TICKETS (10)."""
    from ticket_agent.intake.proposal_generator import MAX_TICKETS

    assert MAX_TICKETS == 10

    twelve_tickets = [
        {"summary": f"Ticket {i}", "description": ""} for i in range(12)
    ]
    router = _Router({"title": "Project", "tickets": twelve_tickets})

    proposal = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock).generate(
            _request("For AGENT project, do things")
        )
    ).proposal

    assert proposal is not None
    assert len(proposal.tickets) == MAX_TICKETS
    assert proposal.truncated_ticket_count == 0
    description_text = "\n".join(ticket.description for ticket in proposal.tickets)
    assert "Ticket 10" in description_text
    assert "Ticket 11" in description_text


def _request(
    text: str,
    *,
    mode: IntakeMode = IntakeMode.NEW_FEATURE,
    capability: str = "code.implement",
    repo_defaults: dict[str, dict[str, str]] | None = None,
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
        repo_defaults=repo_defaults or {
            "AGENT": {
                "repository": "agent-system",
                "repo_path": "/home/agent",
            }
        },
    )


def _clock() -> datetime:
    return datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


@dataclass
class _Router:
    response: Any

    def __post_init__(self) -> None:
        self.calls: list[str] = []
        self.call_messages: list[list[dict[str, str]]] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        del kwargs
        self.calls.append(capability)
        self.call_messages.append(messages)
        return self.response


@dataclass
class _HangingRouter:
    def __post_init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        del messages, kwargs
        self.calls.append(capability)
        await asyncio.sleep(60)
        return {"title": "Too late", "tickets": [{"summary": "Too late"}]}

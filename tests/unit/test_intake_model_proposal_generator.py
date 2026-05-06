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
        "Add login API",
        "Add login UI",
    ]
    assert proposal.tickets[0].labels == [LABEL_AI_READY]
    assert proposal.tickets[1].labels == ["frontend", LABEL_AI_READY]
    assert proposal.tickets[0].repository == "agent-system"
    assert router.calls == ["ticket.decompose"]


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
        ModelRouterProposalGenerator(router, clock=_clock).generate(
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


def test_model_tickets_truncated_to_max_tickets():
    """Model output exceeding max_tickets is truncated deterministically."""
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
    assert proposal.truncated_ticket_count == 2
    assert proposal.tickets[0].summary == "Ticket 0"
    assert proposal.tickets[4].summary == "Ticket 4"


def test_model_tickets_default_max_is_five():
    """Default max_tickets is MAX_TICKETS (5)."""
    from ticket_agent.intake.proposal_generator import MAX_TICKETS

    assert MAX_TICKETS == 5

    six_tickets = [{"summary": f"Ticket {i}", "description": ""} for i in range(6)]
    router = _Router({"title": "Project", "tickets": six_tickets})

    proposal = asyncio.run(
        ModelRouterProposalGenerator(router, clock=_clock).generate(
            _request("For AGENT project, do things")
        )
    ).proposal

    assert proposal is not None
    assert len(proposal.tickets) == MAX_TICKETS
    assert proposal.truncated_ticket_count == 1


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

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        del messages, kwargs
        self.calls.append(capability)
        return self.response

"""Intake pipeline: Slack message -> proposal -> Jira tickets."""

from ticket_agent.intake.approval_flow import ApprovalFlow, ApprovalOutcome
from ticket_agent.intake.intent_resolver import (
    CAPABILITY_TO_MODE,
    IntakeIntentResolver,
)
from ticket_agent.intake.jira_writer import JiraWriteResult, JiraWriter
from ticket_agent.intake.proposal_generator import (
    DeterministicProposalGenerator,
    ProposalDraft,
    ProposalGenerator,
    ProposalRequest,
)
from ticket_agent.intake.proposal_store import ProposalStore
from ticket_agent.intake.slack_listener import (
    SlackClient,
    SlackEvent,
    SlackIntakeListener,
)

__all__ = [
    "ApprovalFlow",
    "ApprovalOutcome",
    "CAPABILITY_TO_MODE",
    "DeterministicProposalGenerator",
    "IntakeIntentResolver",
    "JiraWriteResult",
    "JiraWriter",
    "ProposalDraft",
    "ProposalGenerator",
    "ProposalRequest",
    "ProposalStore",
    "SlackClient",
    "SlackEvent",
    "SlackIntakeListener",
]

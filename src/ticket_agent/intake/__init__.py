"""Intake pipeline: Slack message -> proposal -> Jira tickets."""

from ticket_agent.intake.approval_flow import ApprovalFlow, ApprovalOutcome
from ticket_agent.intake.intent_resolver import (
    CAPABILITY_TO_MODE,
    IntakeIntentResolver,
)
from ticket_agent.intake.jira_writer import JiraWriter, JiraWriteResult
from ticket_agent.intake.proposal_generator import (
    DeterministicProposalGenerator,
    ModelRouterProposalGenerator,
    ProposalDraft,
    ProposalGenerator,
    ProposalRequest,
)
from ticket_agent.intake.proposal_store import ProposalStore
from ticket_agent.intake.question_answerer import (
    JiraQuestionAnswerHandler,
    QuestionAnswerResult,
    is_question_text,
)
from ticket_agent.intake.slack_listener import (
    SlackClient,
    SlackEvent,
    SlackIntakeListener,
)

__all__ = [
    "CAPABILITY_TO_MODE",
    "ApprovalFlow",
    "ApprovalOutcome",
    "DeterministicProposalGenerator",
    "IntakeIntentResolver",
    "JiraQuestionAnswerHandler",
    "JiraWriteResult",
    "JiraWriter",
    "ModelRouterProposalGenerator",
    "ProposalDraft",
    "ProposalGenerator",
    "ProposalRequest",
    "ProposalStore",
    "QuestionAnswerResult",
    "SlackClient",
    "SlackEvent",
    "SlackIntakeListener",
    "is_question_text",
]

"""Jira-backed orchestrator service implementations."""

from __future__ import annotations

import logging
from logging import Logger

from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import LABEL_AI_EXECUTION_APPROVED
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.models import JiraExecutionError
from ticket_agent.orchestrator.state import TicketState

_LOGGER = logging.getLogger(__name__)


class JiraEscalationService:
    """Update Jira ticket state when the workflow escalates.

    Jira update failures are logged and swallowed — the graph is already
    in escalation and a secondary Jira error must not mask the real cause.
    """

    def __init__(
        self,
        execution_service: JiraExecutionService,
        *,
        logger: Logger | None = None,
    ) -> None:
        self._execution_service = execution_service
        self._logger = logger or _LOGGER

    async def escalate(self, state: TicketState, reason: str) -> None:
        """Mark the ticket as failed in Jira with the escalation reason."""
        ticket_key = state.ticket_key
        try:
            await self._execution_service.mark_failed(ticket_key, reason)
        except JiraExecutionError as exc:
            self._logger.warning(
                "jira_escalation_update_failed for %s: %s",
                ticket_key,
                exc,
                exc_info=True,
            )


class JiraLabelApprovalService:
    """Approve execution when the ticket carries the ai-execution-approved label.

    A human (or Slack bot) adds the label to signal consent.  The graph checks
    it once at the approval node; if absent it returns False and the graph
    escalates.  Errors from Jira bubble up so the OrchestratorRunner can
    escalate the ticket correctly.
    """

    def __init__(self, client: JiraClient) -> None:
        self._client = client

    async def request_approval(self, state: TicketState) -> bool:
        ticket = await self._client.get_ticket(state.ticket_key)
        return LABEL_AI_EXECUTION_APPROVED in ticket.labels


__all__ = ["JiraEscalationService", "JiraLabelApprovalService"]

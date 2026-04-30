"""Jira execution-state updates for claimed work."""

from __future__ import annotations

from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_IN_REVIEW,
)


class JiraExecutionService:
    """Update Jira with execution lifecycle state."""

    def __init__(self, client: JiraClient, component_id: str) -> None:
        self._client = client
        self._component_id = component_id

    async def mark_claimed(self, ticket_key: str) -> None:
        """Mark a Jira ticket as claimed by this component."""

        await self._client.add_labels(ticket_key, [LABEL_AI_CLAIMED])
        await self._client.transition_ticket(ticket_key, STATUS_IN_PROGRESS)
        await self._client.update_fields(
            ticket_key,
            {FIELD_AGENT_ASSIGNED_COMPONENT: self._component_id},
        )

    async def mark_failed(self, ticket_key: str, reason: str) -> None:
        """Mark a Jira ticket as failed and leave an execution comment."""

        await self._client.add_labels(ticket_key, [LABEL_AI_FAILED])
        await self._client.remove_labels(ticket_key, [LABEL_AI_CLAIMED])
        await self._client.update_fields(
            ticket_key,
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        )
        await self._client.add_comment(
            ticket_key,
            f"AI execution failed:\n\n{reason}",
        )

    async def mark_in_review(self, ticket_key: str, pull_request_url: str) -> None:
        """Move a Jira ticket to review after opening a pull request."""

        await self._client.transition_ticket(ticket_key, STATUS_IN_REVIEW)
        await self._client.remove_labels(ticket_key, [LABEL_AI_CLAIMED])
        await self._client.update_fields(
            ticket_key,
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        )
        await self._client.add_comment(
            ticket_key,
            f"AI execution opened pull request:\n\n{pull_request_url}",
        )

    async def mark_released(self, ticket_key: str) -> None:
        """Release this component's execution claim without changing status."""

        await self._client.remove_labels(ticket_key, [LABEL_AI_CLAIMED])
        await self._client.update_fields(
            ticket_key,
            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
        )


__all__ = ["JiraExecutionService"]

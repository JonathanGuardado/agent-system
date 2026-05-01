"""Jira execution-state updates for claimed work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_IN_REVIEW,
)
from ticket_agent.jira.models import JiraExecutionError

_CLAIMED_LABELS = (LABEL_AI_CLAIMED,)
_FAILED_LABELS = (LABEL_AI_FAILED,)
_MARK_CLAIMED_OPERATION = "mark_claimed"
_MARK_FAILED_OPERATION = "mark_failed"
_MARK_IN_REVIEW_OPERATION = "mark_in_review"
_MARK_RELEASED_OPERATION = "mark_released"


class JiraExecutionService:
    """Update Jira with execution lifecycle state."""

    def __init__(self, client: JiraClient, component_id: str) -> None:
        self._client = client
        self._component_id = component_id

    async def mark_claimed(self, ticket_key: str) -> None:
        """Mark a Jira ticket as claimed by this component."""

        await self._call_jira(
            _MARK_CLAIMED_OPERATION,
            ticket_key,
            lambda: self._client.add_labels(ticket_key, list(_CLAIMED_LABELS)),
        )
        await self._call_jira(
            _MARK_CLAIMED_OPERATION,
            ticket_key,
            lambda: self._client.transition_ticket(ticket_key, STATUS_IN_PROGRESS),
        )
        await self._call_jira(
            _MARK_CLAIMED_OPERATION,
            ticket_key,
            lambda: self._client.update_fields(
                ticket_key,
                {FIELD_AGENT_ASSIGNED_COMPONENT: self._component_id},
            ),
        )

    async def mark_failed(self, ticket_key: str, reason: str) -> None:
        """Mark a Jira ticket as failed and leave an execution comment."""

        await self._call_jira(
            _MARK_FAILED_OPERATION,
            ticket_key,
            lambda: self._client.add_labels(ticket_key, list(_FAILED_LABELS)),
        )
        await self._call_jira(
            _MARK_FAILED_OPERATION,
            ticket_key,
            lambda: self._client.remove_labels(ticket_key, list(_CLAIMED_LABELS)),
        )
        await self._call_jira(
            _MARK_FAILED_OPERATION,
            ticket_key,
            lambda: self._client.update_fields(
                ticket_key,
                {FIELD_AGENT_ASSIGNED_COMPONENT: None},
            ),
        )
        await self._call_jira(
            _MARK_FAILED_OPERATION,
            ticket_key,
            lambda: self._client.add_comment(
                ticket_key,
                f"AI execution failed:\n\n{reason}",
            ),
        )

    async def mark_in_review(self, ticket_key: str, pull_request_url: str) -> None:
        """Move a Jira ticket to review after opening a pull request."""

        await self._call_jira(
            _MARK_IN_REVIEW_OPERATION,
            ticket_key,
            lambda: self._client.transition_ticket(ticket_key, STATUS_IN_REVIEW),
        )
        await self._call_jira(
            _MARK_IN_REVIEW_OPERATION,
            ticket_key,
            lambda: self._client.remove_labels(ticket_key, list(_CLAIMED_LABELS)),
        )
        await self._call_jira(
            _MARK_IN_REVIEW_OPERATION,
            ticket_key,
            lambda: self._client.update_fields(
                ticket_key,
                {FIELD_AGENT_ASSIGNED_COMPONENT: None},
            ),
        )
        await self._call_jira(
            _MARK_IN_REVIEW_OPERATION,
            ticket_key,
            lambda: self._client.add_comment(
                ticket_key,
                f"AI execution opened pull request:\n\n{pull_request_url}",
            ),
        )

    async def mark_released(self, ticket_key: str) -> None:
        """Release this component's execution claim without changing status."""

        await self._call_jira(
            _MARK_RELEASED_OPERATION,
            ticket_key,
            lambda: self._client.remove_labels(ticket_key, list(_CLAIMED_LABELS)),
        )
        await self._call_jira(
            _MARK_RELEASED_OPERATION,
            ticket_key,
            lambda: self._client.update_fields(
                ticket_key,
                {FIELD_AGENT_ASSIGNED_COMPONENT: None},
            ),
        )

    async def _call_jira(
        self,
        method_name: str,
        ticket_key: str,
        call: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            await call()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            raise JiraExecutionError(
                f"{method_name} failed for {ticket_key}: {error}"
            ) from exc


__all__ = ["JiraExecutionService"]

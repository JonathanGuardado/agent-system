"""Jira client protocol used by execution services."""

from __future__ import annotations

from typing import Protocol

from ticket_agent.jira.models import JiraTicket


class JiraClient(Protocol):
    """Async boundary for Jira issue reads and execution-state updates."""

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        """Return one Jira ticket by key."""

    async def transition_ticket(self, ticket_key: str, status: str) -> None:
        """Move a Jira ticket to a workflow status."""

    async def add_labels(self, ticket_key: str, labels: list[str]) -> None:
        """Add labels to a Jira ticket."""

    async def remove_labels(self, ticket_key: str, labels: list[str]) -> None:
        """Remove labels from a Jira ticket."""

    async def update_fields(self, ticket_key: str, fields: dict[str, object]) -> None:
        """Update logical Jira fields for a ticket."""

    async def add_comment(self, ticket_key: str, body: str) -> None:
        """Add a comment to a Jira ticket."""


__all__ = ["JiraClient"]

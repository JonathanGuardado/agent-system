"""Convert Jira tickets into orchestrator work items."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_MAX_ATTEMPTS,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
)
from ticket_agent.jira.models import JiraTicket, JiraWorkItemLoadError
from ticket_agent.orchestrator.runner import TicketWorkItem


class JiraWorkItemLoader:
    """Load an execution-ready work item from Jira."""

    def __init__(self, client: JiraClient) -> None:
        self._client = client

    async def load(self, ticket_key: str) -> TicketWorkItem:
        """Fetch a Jira ticket and convert it into a TicketWorkItem."""

        ticket = await self._client.get_ticket(ticket_key)
        repository = _required_string(ticket, FIELD_REPOSITORY)
        repo_path = _required_string(ticket, FIELD_REPO_PATH)

        return TicketWorkItem(
            ticket_key=ticket.key,
            summary=ticket.summary,
            description=ticket.description,
            repository=repository,
            repo_path=repo_path,
            max_attempts=_max_attempts(ticket),
        )


def _required_string(ticket: JiraTicket, field_name: str) -> str:
    value = ticket.fields.get(field_name)
    if not isinstance(value, str) or value.strip() == "":
        raise JiraWorkItemLoadError(
            f"Jira ticket {ticket.key} is missing required field: {field_name}"
        )
    return value


def _max_attempts(ticket: JiraTicket) -> int:
    value = ticket.fields.get(
        FIELD_MAX_ATTEMPTS,
        _ticket_work_item_default("max_attempts"),
    )
    try:
        attempts = int(value)
    except (TypeError, ValueError) as exc:
        raise JiraWorkItemLoadError(
            f"Jira ticket {ticket.key} has invalid field: {FIELD_MAX_ATTEMPTS}"
        ) from exc
    if attempts < 1:
        raise JiraWorkItemLoadError(
            f"Jira ticket {ticket.key} has invalid field: {FIELD_MAX_ATTEMPTS}"
        )
    return attempts


def _ticket_work_item_default(field_name: str) -> Any:
    for item_field in fields(TicketWorkItem):
        if item_field.name == field_name:
            return item_field.default
    raise JiraWorkItemLoadError(
        f"TicketWorkItem does not define expected field: {field_name}"
    )


__all__ = ["JiraWorkItemLoader"]

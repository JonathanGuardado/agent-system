"""Convert Jira tickets into orchestrator work items."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import re
from typing import Any

from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_MAX_ATTEMPTS,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    FIELD_SLACK_CHANNEL,
    FIELD_SLACK_THREAD_TS,
)
from ticket_agent.jira.models import JiraTicket, JiraWorkItemLoadError
from ticket_agent.orchestrator.runner import TicketWorkItem


class JiraWorkItemLoader:
    """Load an execution-ready work item from Jira."""

    def __init__(
        self,
        client: JiraClient,
        repo_defaults: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._repo_defaults = repo_defaults or {}

    async def load(self, ticket_key: str) -> TicketWorkItem:
        """Fetch a Jira ticket and convert it into a TicketWorkItem."""

        ticket = await self._client.get_ticket(ticket_key)
        repository = _repo_value(ticket, FIELD_REPOSITORY, self._repo_defaults)
        repo_path = _repo_value(ticket, FIELD_REPO_PATH, self._repo_defaults)

        return TicketWorkItem(
            ticket_key=ticket.key,
            summary=ticket.summary,
            description=ticket.description,
            repository=repository,
            repo_path=repo_path,
            slack_channel=_optional_string(ticket, FIELD_SLACK_CHANNEL),
            slack_thread_ts=_optional_string(ticket, FIELD_SLACK_THREAD_TS),
            max_attempts=_max_attempts(ticket),
        )


def _repo_value(
    ticket: JiraTicket,
    field_name: str,
    defaults: dict[str, str],
) -> str:
    value = _optional_string(ticket, field_name)
    if value is not None:
        return value

    value = _description_context_value(ticket.description, field_name)
    if value is not None:
        return value

    if field_name == FIELD_REPOSITORY:
        value = _summary_repository(ticket.summary)
        if value is not None:
            return value

    value = defaults.get(field_name)
    if value is not None and value.strip():
        return value.strip()

    raise JiraWorkItemLoadError(
        f"Jira ticket {ticket.key} is missing required field: {field_name}"
    )


def _description_context_value(description: str, field_name: str) -> str | None:
    labels = {
        FIELD_REPOSITORY: "Repository",
        FIELD_REPO_PATH: "Repository path",
    }
    label = labels.get(field_name)
    if label is None:
        return None
    pattern = re.compile(rf"^\s*-\s*{re.escape(label)}:\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(description)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _summary_repository(summary: str) -> str | None:
    match = re.match(r"^\s*\[([^\]]+)\]", summary)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _optional_string(ticket: JiraTicket, field_name: str) -> str | None:
    value = ticket.fields.get(field_name)
    if not isinstance(value, str) or value.strip() == "":
        return None
    return value


def _max_attempts(ticket: JiraTicket) -> int:
    if FIELD_MAX_ATTEMPTS not in ticket.fields:
        return _ticket_work_item_default("max_attempts")

    value = ticket.fields[FIELD_MAX_ATTEMPTS]
    if type(value) is not int:
        raise JiraWorkItemLoadError(
            f"Jira ticket {ticket.key} has invalid field: {FIELD_MAX_ATTEMPTS}"
        )
    if value < 1:
        raise JiraWorkItemLoadError(
            f"Jira ticket {ticket.key} has invalid field: {FIELD_MAX_ATTEMPTS}"
        )
    return value


def _ticket_work_item_default(field_name: str) -> Any:
    for item_field in fields(TicketWorkItem):
        if item_field.name == field_name:
            return item_field.default
    raise JiraWorkItemLoadError(
        f"TicketWorkItem does not define expected field: {field_name}"
    )


def repo_defaults_from_mapping(
    repo_defaults: dict[str, dict[str, str]] | None,
) -> dict[str, str]:
    if not repo_defaults or len(repo_defaults) != 1:
        return {}
    defaults = next(iter(repo_defaults.values()))
    result: dict[str, str] = {}
    repository = defaults.get("repository")
    repo_path = defaults.get("repo_path")
    if repository:
        result[FIELD_REPOSITORY] = repository
    if repo_path:
        result[FIELD_REPO_PATH] = str(Path(repo_path).expanduser())
    return result


__all__ = ["JiraWorkItemLoader", "repo_defaults_from_mapping"]

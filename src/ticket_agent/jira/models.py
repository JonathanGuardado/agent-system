"""Jira boundary models for execution-oriented ticket loading."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class JiraTicket:
    """Jira issue data needed by the local execution boundary."""

    key: str
    summary: str
    description: str = ""
    status: str = ""
    labels: list[str] = field(default_factory=list)
    assignee: str | None = None
    fields: dict[str, object] = field(default_factory=dict)


class JiraWorkItemLoadError(ValueError):
    """Raised when a Jira ticket cannot be converted into a work item."""


class JiraExecutionError(RuntimeError):
    """Raised when Jira execution state cannot be updated."""


__all__ = [
    "JiraExecutionError",
    "JiraTicket",
    "JiraWorkItemLoadError",
]

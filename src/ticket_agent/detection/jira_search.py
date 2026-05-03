"""Jira-backed detection search client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ticket_agent.jira.constants import (
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    LABEL_AI_READY,
    LABEL_DO_NOT_AUTOMATE,
    STATUS_TODO,
)
from ticket_agent.jira.models import JiraTicket


DETECTION_JQL = (
    'project in agentProjects() '
    f'AND status = "{STATUS_TODO}" '
    f'AND labels = "{LABEL_AI_READY}" '
    f'AND labels != "{LABEL_AI_CLAIMED}" '
    f'AND labels != "{LABEL_AI_FAILED}" '
    f'AND labels != "{LABEL_DO_NOT_AUTOMATE}" '
    "ORDER BY priority DESC, created ASC"
)


DEFAULT_DETECTION_FIELDS: tuple[str, ...] = (
    "summary",
    "description",
    "status",
    "labels",
    "assignee",
    "issuelinks",
    "priority",
    "created",
    "*all",
)


class JiraIssueSearchClient(Protocol):
    """Jira boundary capable of running a JQL issue search."""

    async def search_issues(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> Sequence[JiraTicket | Mapping[str, Any]] | Mapping[str, Any]:
        """Return Jira issues for ``jql``."""


class JiraDetectionSearchClient:
    """Concrete DetectionSearchClient backed by Jira JQL search."""

    def __init__(
        self,
        client: JiraIssueSearchClient,
        *,
        jql: str = DETECTION_JQL,
        fields: Sequence[str] = DEFAULT_DETECTION_FIELDS,
    ) -> None:
        self._client = client
        self._jql = jql
        self._fields = tuple(fields)

    async def search_ai_ready_tickets(self) -> Sequence[JiraTicket]:
        """Search Jira and normalize candidate ai-ready tickets."""

        result = await self._client.search_issues(
            self._jql,
            fields=self._fields,
        )
        return [
            _normalize_issue(issue)
            for issue in _coerce_issue_sequence(result)
        ]


def _coerce_issue_sequence(
    result: Sequence[JiraTicket | Mapping[str, Any]] | Mapping[str, Any],
) -> Sequence[JiraTicket | Mapping[str, Any]]:
    if isinstance(result, Mapping):
        issues = result.get("issues")
        if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes)):
            return issues
        return [result]
    return result


def _normalize_issue(issue: JiraTicket | Mapping[str, Any]) -> JiraTicket:
    if isinstance(issue, JiraTicket):
        return issue

    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        fields = {}

    key = _string(issue.get("key"))
    summary = _string(fields.get("summary"))
    status = _status_name(fields.get("status"))
    labels = _string_list(fields.get("labels"))
    assignee, assignee_fields = _assignee(fields.get("assignee"))

    logical_fields = _logical_fields(fields)
    logical_fields.update(assignee_fields)

    blockers = _blocking_issue_links(fields.get("issuelinks"))
    if blockers:
        logical_fields["blocked_by"] = blockers
        logical_fields["blocking_issues"] = blockers

    return JiraTicket(
        key=key,
        summary=summary,
        description=_description_text(fields.get("description")),
        status=status,
        labels=labels,
        assignee=assignee,
        fields=logical_fields,
    )


def _logical_fields(fields: Mapping[str, Any]) -> dict[str, object]:
    excluded = {
        "summary",
        "description",
        "status",
        "labels",
        "assignee",
        "issuelinks",
    }
    return {
        str(key): value
        for key, value in fields.items()
        if key not in excluded
    }


def _assignee(value: object) -> tuple[str | None, dict[str, object]]:
    if not isinstance(value, Mapping):
        text = _string(value)
        return (text or None), {}

    account_id = _string(value.get("accountId"))
    email = _string(value.get("emailAddress"))
    display_name = _string(value.get("displayName"))
    name = _string(value.get("name"))
    assignee = account_id or email or display_name or name or None

    fields: dict[str, object] = {}
    if account_id:
        fields["assignee_account_id"] = account_id
    if email:
        fields["assignee_email"] = email
    if display_name:
        fields["assignee_display_name"] = display_name
    if name:
        fields["assignee_name"] = name
    return assignee, fields


def _blocking_issue_links(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        candidate = _blocking_issue_from_link(item)
        if candidate is None:
            continue
        key = candidate["key"]
        if isinstance(key, str) and key not in seen:
            seen.add(key)
            blockers.append(candidate)
    return blockers


def _blocking_issue_from_link(link: Mapping[str, Any]) -> dict[str, object] | None:
    link_type = link.get("type")
    type_fields = link_type if isinstance(link_type, Mapping) else {}

    for direction, issue_field in (
        ("inward", "inwardIssue"),
        ("outward", "outwardIssue"),
    ):
        phrase = _string(type_fields.get(direction)).lower()
        if not _phrase_indicates_current_issue_is_blocked(phrase):
            continue
        linked_issue = link.get(issue_field)
        if not isinstance(linked_issue, Mapping):
            continue
        key = _string(linked_issue.get("key"))
        if not key or _issue_is_resolved(linked_issue):
            continue
        status = _linked_issue_status(linked_issue)
        return {"key": key, "status": status, "resolved": False}
    return None


def _phrase_indicates_current_issue_is_blocked(phrase: str) -> bool:
    normalized = " ".join(phrase.split())
    return (
        "blocked by" in normalized
        or "is blocked by" in normalized
        or "depends on" in normalized
    )


def _issue_is_resolved(issue: Mapping[str, Any]) -> bool:
    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        return False

    status = fields.get("status")
    if not isinstance(status, Mapping):
        return False

    status_name = _string(status.get("name")).lower()
    if status_name in {"done", "closed", "resolved"}:
        return True

    category = status.get("statusCategory")
    if isinstance(category, Mapping):
        category_name = _string(category.get("name")).lower()
        category_key = _string(category.get("key")).lower()
        return category_name == "done" or category_key == "done"
    return False


def _linked_issue_status(issue: Mapping[str, Any]) -> str:
    fields = issue.get("fields")
    if isinstance(fields, Mapping):
        return _status_name(fields.get("status"))
    return ""


def _status_name(value: object) -> str:
    if isinstance(value, Mapping):
        return _string(value.get("name"))
    return _string(value)


def _description_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(_text_nodes(value)).strip()
    return _string(value)


def _text_nodes(value: object) -> list[str]:
    if isinstance(value, Mapping):
        parts: list[str] = []
        text = value.get("text")
        if isinstance(text, str):
            parts.append(text)
        content = value.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for item in content:
                parts.extend(_text_nodes(item))
        return parts
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = []
        for item in value:
            parts.extend(_text_nodes(item))
        return parts
    return []


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [_string(item) for item in value if _string(item)]


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


__all__ = [
    "DEFAULT_DETECTION_FIELDS",
    "DETECTION_JQL",
    "JiraDetectionSearchClient",
    "JiraIssueSearchClient",
]

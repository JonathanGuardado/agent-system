"""Jira client protocol and REST implementation used by execution services."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx

from ticket_agent.jira.models import JiraTicket


class JiraClient(Protocol):
    """Async boundary for Jira issue reads and execution-state updates."""

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        """Return one Jira ticket by key."""

    async def search_issues(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> Sequence[JiraTicket | Mapping[str, Any]] | Mapping[str, Any]:
        """Return Jira issues matching a JQL query."""

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

    async def create_issue(
        self,
        project_key: str,
        *,
        summary: str,
        description: str = "",
        issue_type: str = "Task",
        priority: str | None = None,
        labels: list[str] | None = None,
        fields: dict[str, object] | None = None,
        parent_key: str | None = None,
    ) -> JiraTicket:
        """Create a new Jira issue and return the created ticket."""


class JiraClientError(RuntimeError):
    """Raised when the Jira REST API cannot complete an operation."""


@dataclass(frozen=True, slots=True)
class JiraRestClient:
    """Async Jira Cloud REST client for the production runtime."""

    base_url: str
    user_email: str
    api_key: str
    timeout_s: float = 30.0
    field_map: Mapping[str, str] = field(default_factory=dict)

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        result = await self._request(
            "GET",
            f"/rest/api/3/issue/{ticket_key}",
            params={"fields": "*all"},
        )
        if not isinstance(result, Mapping):
            raise JiraClientError(f"Jira issue {ticket_key} response was not an object")
        return _normalize_issue(result, field_map=self.field_map)

    async def search_issues(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        jira_fields = None
        if fields is not None:
            jira_fields = [self._jira_field_name(field_name) for field_name in fields]
        payload: dict[str, object] = {"jql": jql}
        if jira_fields is not None:
            payload["fields"] = jira_fields
        result = await self._request(
            "POST",
            "/rest/api/3/search",
            json=payload,
        )
        if not isinstance(result, Mapping):
            raise JiraClientError("Jira search response was not an object")
        return result

    async def transition_ticket(self, ticket_key: str, status: str) -> None:
        result = await self._request(
            "GET",
            f"/rest/api/3/issue/{ticket_key}/transitions",
        )
        transitions = result.get("transitions") if isinstance(result, Mapping) else None
        if not isinstance(transitions, Sequence):
            raise JiraClientError(
                f"Jira transitions response for {ticket_key} was not a list"
            )

        target = _find_transition(transitions, status)
        if target is None:
            ticket = await self.get_ticket(ticket_key)
            if ticket.status == status:
                return
            raise JiraClientError(
                f"Jira ticket {ticket_key} has no transition to status {status!r}"
            )

        await self._request(
            "POST",
            f"/rest/api/3/issue/{ticket_key}/transitions",
            json={"transition": {"id": str(target)}},
            expect_json=False,
        )

    async def add_labels(self, ticket_key: str, labels: list[str]) -> None:
        await self._update_issue(
            ticket_key,
            {"update": {"labels": [{"add": label} for label in labels]}},
        )

    async def remove_labels(self, ticket_key: str, labels: list[str]) -> None:
        await self._update_issue(
            ticket_key,
            {"update": {"labels": [{"remove": label} for label in labels]}},
        )

    async def update_fields(self, ticket_key: str, fields: dict[str, object]) -> None:
        await self._update_issue(
            ticket_key,
            {
                "fields": {
                    self._jira_field_name(field_name): value
                    for field_name, value in fields.items()
                }
            },
        )

    async def add_comment(self, ticket_key: str, body: str) -> None:
        await self._request(
            "POST",
            f"/rest/api/3/issue/{ticket_key}/comment",
            json={"body": _adf_doc(body)},
        )

    async def create_issue(
        self,
        project_key: str,
        *,
        summary: str,
        description: str = "",
        issue_type: str = "Task",
        priority: str | None = None,
        labels: list[str] | None = None,
        fields: dict[str, object] | None = None,
        parent_key: str | None = None,
    ) -> JiraTicket:
        issue_fields: dict[str, object] = {
            "project": {"key": project_key},
            "summary": summary,
            "description": _adf_doc(description),
            "issuetype": {"name": issue_type},
            "labels": list(labels or []),
        }
        if priority:
            issue_fields["priority"] = {"name": priority}
        if parent_key:
            issue_fields["parent"] = {"key": parent_key}
        for field_name, value in (fields or {}).items():
            issue_fields[self._jira_field_name(field_name)] = value

        result = await self._request(
            "POST",
            "/rest/api/3/issue",
            json={"fields": issue_fields},
        )
        if not isinstance(result, Mapping):
            raise JiraClientError("Jira create issue response was not an object")

        ticket_key = _string(result.get("key"))
        if not ticket_key:
            raise JiraClientError("Jira create issue response did not include a key")
        return await self.get_ticket(ticket_key)

    async def _update_issue(self, ticket_key: str, payload: Mapping[str, Any]) -> None:
        await self._request(
            "PUT",
            f"/rest/api/3/issue/{ticket_key}",
            json=dict(payload),
            expect_json=False,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        json: Mapping[str, object] | None = None,
        expect_json: bool = True,
    ) -> Any:
        if self.timeout_s <= 0:
            raise JiraClientError("Jira timeout must be positive")

        url = _join_url(self.base_url, path)
        try:
            async with httpx.AsyncClient(
                auth=(self.user_email, self.api_key),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_s,
            ) as client:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise JiraClientError(f"Jira request timed out: {method} {path}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            text = _safe_response_text(exc.response)
            raise JiraClientError(
                f"Jira request failed: {method} {path} returned {status}: {text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise JiraClientError(
                f"Jira request failed: {method} {path}: {exc.__class__.__name__}"
            ) from exc

        if not expect_json or response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise JiraClientError(
                f"Jira response was not valid JSON: {method} {path}"
            ) from exc

    def _jira_field_name(self, field_name: str) -> str:
        return str(self.field_map.get(field_name, field_name))


def _normalize_issue(
    issue: Mapping[str, Any],
    *,
    field_map: Mapping[str, str],
) -> JiraTicket:
    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        fields = {}

    inverse_field_map = {jira_name: logical for logical, jira_name in field_map.items()}
    logical_fields = {
        inverse_field_map.get(str(key), str(key)): value
        for key, value in fields.items()
        if key
        not in {
            "summary",
            "description",
            "status",
            "labels",
            "assignee",
        }
    }
    assignee, assignee_fields = _assignee(fields.get("assignee"))
    logical_fields.update(assignee_fields)

    return JiraTicket(
        key=_string(issue.get("key")),
        summary=_string(fields.get("summary")),
        description=_description_text(fields.get("description")),
        status=_status_name(fields.get("status")),
        labels=_string_list(fields.get("labels")),
        assignee=assignee,
        fields=logical_fields,
    )


def _find_transition(transitions: Sequence[object], status: str) -> str | None:
    for transition in transitions:
        if not isinstance(transition, Mapping):
            continue
        transition_id = _string(transition.get("id"))
        transition_name = _string(transition.get("name"))
        to_status = transition.get("to")
        to_name = _status_name(to_status) if isinstance(to_status, Mapping) else ""
        if transition_id and status in {transition_name, to_name}:
            return transition_id
    return None


def _adf_doc(text: str) -> dict[str, object]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def _description_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return _string(value)
    parts: list[str] = []
    _collect_adf_text(value, parts)
    return " ".join(part for part in parts if part).strip()


def _collect_adf_text(value: object, parts: list[str]) -> None:
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            parts.append(text)
        content = value.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for child in content:
                _collect_adf_text(child, parts)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _collect_adf_text(child, parts)


def _status_name(value: object) -> str:
    if isinstance(value, Mapping):
        return _string(value.get("name"))
    return _string(value)


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


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [_string(item) for item in value if _string(item)]


def _string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _safe_response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    if not text:
        return response.reason_phrase or "HTTP error"
    return text[:500]


__all__ = ["JiraClient", "JiraClientError", "JiraRestClient"]

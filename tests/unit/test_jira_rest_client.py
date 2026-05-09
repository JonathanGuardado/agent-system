from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.jira import client as jira_client_module
from ticket_agent.jira.client import JiraRestClient
from ticket_agent.jira.constants import FIELD_EPIC_LINK


def test_jira_rest_client_uses_configured_epic_link_field(monkeypatch):
    fake_client = _FakeAsyncClient()
    monkeypatch.setattr(
        jira_client_module.httpx,
        "AsyncClient",
        lambda **kwargs: fake_client,
    )

    result = asyncio.run(
        JiraRestClient(
            base_url="https://jira.example.test",
            user_email="agent@example.test",
            api_key="secret",
            field_map={FIELD_EPIC_LINK: "customfield_10014"},
        ).create_issue(
            "AGENT",
            summary="Child task",
            description="Do the child work",
            parent_key="AGENT-1",
        )
    )

    assert result.key == "AGENT-2"
    create_payload = fake_client.requests[0]["json"]["fields"]
    assert create_payload["customfield_10014"] == "AGENT-1"
    assert "parent" not in create_payload


def test_jira_rest_client_search_uses_current_jql_endpoint(monkeypatch):
    fake_client = _FakeAsyncClient()
    monkeypatch.setattr(
        jira_client_module.httpx,
        "AsyncClient",
        lambda **kwargs: fake_client,
    )

    result = asyncio.run(
        JiraRestClient(
            base_url="https://jira.example.test",
            user_email="agent@example.test",
            api_key="secret",
            field_map={"agent_assigned_component": "customfield_10020"},
        ).search_issues(
            'project = AGENT AND labels = "ai-ready"',
            fields=["summary", "agent_assigned_component"],
        )
    )

    assert result == {"issues": []}
    search_request = fake_client.requests[0]
    assert search_request["method"] == "POST"
    assert search_request["url"] == "https://jira.example.test/rest/api/3/search/jql"
    assert search_request["json"] == {
        "jql": 'project = AGENT AND labels = "ai-ready"',
        "fields": ["summary", "customfield_10020"],
    }


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def request(self, method: str, url: str, **kwargs: Any) -> "_Response":
        self.requests.append({"method": method, "url": url, **kwargs})
        if url.endswith("/rest/api/3/search/jql"):
            return _Response({"issues": []})
        if method == "POST":
            return _Response({"key": "AGENT-2"})
        return _Response(
            {
                "key": "AGENT-2",
                "fields": {
                    "summary": "Child task",
                    "description": "Do the child work",
                    "status": {"name": "To Do"},
                    "labels": [],
                    "assignee": None,
                },
            }
        )


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload

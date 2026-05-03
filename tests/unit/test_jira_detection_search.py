from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from ticket_agent.detection.detector import DetectionComponent
from ticket_agent.detection.jira_search import (
    DEFAULT_DETECTION_FIELDS,
    DETECTION_JQL,
    JiraDetectionSearchClient,
)
from ticket_agent.detection.ownership import OwnershipChecker
from ticket_agent.jira.constants import (
    FIELD_REPOSITORY,
    LABEL_AI_READY,
    STATUS_TODO,
)
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket


def test_jira_detection_search_client_returns_normalized_ai_ready_ticket():
    raw_issue = {
        "key": "AGENT-123",
        "fields": {
            "summary": "Implement detection",
            "description": {
                "content": [
                    {
                        "content": [
                            {"text": "Wire Jira polling into the agent."},
                        ],
                    }
                ],
            },
            "status": {"name": STATUS_TODO},
            "labels": [LABEL_AI_READY],
            "assignee": {
                "accountId": "agent-account-1",
                "displayName": "Agent Runner",
            },
            FIELD_REPOSITORY: "agent-system",
            "issuelinks": [
                {
                    "type": {"inward": "is blocked by", "outward": "blocks"},
                    "inwardIssue": {
                        "key": "AGENT-100",
                        "fields": {
                            "status": {
                                "name": "In Progress",
                                "statusCategory": {"key": "indeterminate"},
                            }
                        },
                    },
                },
                {
                    "type": {"inward": "is blocked by", "outward": "blocks"},
                    "inwardIssue": {
                        "key": "AGENT-99",
                        "fields": {
                            "status": {
                                "name": "Done",
                                "statusCategory": {"key": "done"},
                            }
                        },
                    },
                },
            ],
        },
    }
    client = _RawSearchClient({"issues": [raw_issue]})
    detection_client = JiraDetectionSearchClient(client)

    tickets = asyncio.run(detection_client.search_ai_ready_tickets())

    assert client.calls == [(DETECTION_JQL, DEFAULT_DETECTION_FIELDS)]
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.key == "AGENT-123"
    assert ticket.summary == "Implement detection"
    assert ticket.description == "Wire Jira polling into the agent."
    assert ticket.status == STATUS_TODO
    assert ticket.labels == [LABEL_AI_READY]
    assert ticket.assignee == "agent-account-1"
    assert ticket.fields[FIELD_REPOSITORY] == "agent-system"
    assert ticket.fields["assignee_account_id"] == "agent-account-1"
    assert ticket.fields["blocked_by"] == [
        {"key": "AGENT-100", "status": "In Progress", "resolved": False}
    ]


def test_concrete_jira_detection_client_feeds_detection_queue():
    fake_jira = FakeJiraClient(
        JiraTicket(
            key="AGENT-200",
            summary="Queue me",
            description="",
            status=STATUS_TODO,
            labels=[LABEL_AI_READY],
            assignee=None,
            fields={FIELD_REPOSITORY: "agent-system"},
        )
    )
    detection_client = JiraDetectionSearchClient(fake_jira)
    queue: asyncio.Queue[str] = asyncio.Queue()
    detector = DetectionComponent(
        client=detection_client,
        queue=queue,
        ownership_checker=OwnershipChecker(
            component_id="runner-1",
            lock_lookup=lambda key: None,
        ),
        poll_interval_seconds=0.001,
    )

    enqueued = asyncio.run(detector.poll_once())

    assert enqueued == 1
    assert queue.get_nowait() == "AGENT-200"


class _RawSearchClient:
    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[str, ...] | None]] = []

    async def search_issues(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append((jql, None if fields is None else tuple(fields)))
        return self.result

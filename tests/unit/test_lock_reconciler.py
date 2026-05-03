from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_RETRY_COUNT,
    LABEL_AI_CLAIMED,
    LABEL_AI_READY,
    STATUS_IN_PROGRESS,
    STATUS_TODO,
)
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket
from ticket_agent.locking.reconciler import (
    EVENT_LOCK_RECONCILE_FAILED,
    EVENT_LOCK_RECONCILED,
    reconcile_expired_locks,
)
from ticket_agent.locking.sqlite_store import SQLiteLockManager


def test_reconcile_expired_locks_restores_jira_and_deletes_lock(tmp_path):
    clock = _MutableClock()
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
        clock=clock,
    )
    client = FakeJiraClient(
        _ticket(
            status=STATUS_IN_PROGRESS,
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={
                FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1",
                FIELD_AGENT_RETRY_COUNT: 2,
            },
        )
    )
    events = _EventRecorder()

    try:
        assert manager.acquire("AGENT-123", ttl_s=1) is not None
        clock.now = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        reconciled = asyncio.run(
            reconcile_expired_locks(manager, client, emit=events)
        )

        ticket = client.ticket("AGENT-123")
        assert reconciled == 1
        assert manager.expired_locks() == []
        assert ticket.status == STATUS_TODO
        assert ticket.labels == [LABEL_AI_READY]
        assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
        assert ticket.fields[FIELD_AGENT_RETRY_COUNT] == 3
        assert len(client.comments_for("AGENT-123")) == 1
        assert events.names == [EVENT_LOCK_RECONCILED]
        assert events.payloads[EVENT_LOCK_RECONCILED]["jira_cleaned"] is True
    finally:
        manager.close()


def test_reconcile_expired_locks_keeps_row_when_jira_unreachable(tmp_path):
    clock = _MutableClock()
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
        clock=clock,
    )
    client = FakeJiraClient(
        _ticket(),
        fail_on={"get_ticket": RuntimeError("jira unavailable")},
    )
    events = _EventRecorder()

    try:
        assert manager.acquire("AGENT-123", ttl_s=1) is not None
        clock.now = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        reconciled = asyncio.run(
            reconcile_expired_locks(manager, client, emit=events)
        )

        assert reconciled == 0
        assert len(manager.expired_locks()) == 1
        assert events.names == [EVENT_LOCK_RECONCILE_FAILED]
        assert events.payloads[EVENT_LOCK_RECONCILE_FAILED] == {
            "ticket_key": "AGENT-123",
            "component_id": "runner-1",
            "lock_id": "lock-1",
            "error_type": "RuntimeError",
            "error": "jira unavailable",
        }
    finally:
        manager.close()


def test_reconcile_expired_locks_continues_after_one_ticket_fails(tmp_path):
    clock = _MutableClock()
    lock_ids = iter(["lock-1", "lock-2"])
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: next(lock_ids),
        clock=clock,
    )
    client = FakeJiraClient(
        [
            _ticket(key="AGENT-123"),
            _ticket(key="AGENT-456"),
        ],
        fail_on={"remove_labels": [RuntimeError("remove exploded"), None]},
    )
    events = _EventRecorder()

    try:
        assert manager.acquire("AGENT-123", ttl_s=1) is not None
        assert manager.acquire("AGENT-456", ttl_s=1) is not None
        clock.now = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        reconciled = asyncio.run(
            reconcile_expired_locks(manager, client, emit=events)
        )

        remaining = manager.expired_locks()
        assert reconciled == 1
        assert [lock.ticket_key for lock in remaining] == ["AGENT-123"]
        assert events.names == [EVENT_LOCK_RECONCILE_FAILED, EVENT_LOCK_RECONCILED]
        assert events.events[0][1]["ticket_key"] == "AGENT-123"
        assert events.events[1][1]["ticket_key"] == "AGENT-456"
    finally:
        manager.close()


def test_reconcile_expired_locks_is_idempotent_after_success(tmp_path):
    clock = _MutableClock()
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
        clock=clock,
    )
    client = FakeJiraClient(
        _ticket(
            status=STATUS_IN_PROGRESS,
            labels=[LABEL_AI_READY, LABEL_AI_CLAIMED],
            fields={FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
        )
    )

    try:
        assert manager.acquire("AGENT-123", ttl_s=1) is not None
        clock.now = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        first = asyncio.run(reconcile_expired_locks(manager, client))
        second = asyncio.run(reconcile_expired_locks(manager, client))

        ticket = client.ticket("AGENT-123")
        assert first == 1
        assert second == 0
        assert ticket.fields[FIELD_AGENT_RETRY_COUNT] == 1
        assert len(client.comments_for("AGENT-123")) == 1
    finally:
        manager.close()


class _MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def _ticket(**updates: Any) -> JiraTicket:
    values = {
        "key": "AGENT-123",
        "summary": "Implement locks",
        "description": "",
        "status": STATUS_IN_PROGRESS,
        "labels": [LABEL_AI_READY, LABEL_AI_CLAIMED],
        "assignee": None,
        "fields": {FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1"},
    }
    values.update(updates)
    return JiraTicket(**values)


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, payload))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    @property
    def payloads(self) -> dict[str, dict[str, Any]]:
        return dict(self.events)

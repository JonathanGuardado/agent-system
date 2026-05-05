from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

import pytest

from ticket_agent.detection.detector import DetectionComponent
from ticket_agent.detection.jira_search import JiraDetectionSearchClient
from ticket_agent.detection.ownership import OwnershipChecker
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    LABEL_AI_CLAIMED,
    LABEL_AI_READY,
    STATUS_IN_PROGRESS,
    STATUS_IN_REVIEW,
    STATUS_TODO,
)
from ticket_agent.jira.execution_coordinator import JiraExecutionCoordinator
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.locking.reconciler import (
    EVENT_LOCK_RECONCILE_FAILED,
    reconcile_expired_locks,
)
from ticket_agent.locking.sqlite_store import SQLiteLockManager
from ticket_agent.orchestrator.execution_worker import ExecutionWorker
from ticket_agent.orchestrator.runner import (
    EVENT_RUNNER_CLAIM_FAILED,
    OrchestratorRunner,
    TicketAlreadyLockedError,
    TicketClaimFailedError,
    TicketWorkItem,
)
from ticket_agent.orchestrator.state import TicketState


@pytest.mark.asyncio
async def test_detection_to_runner_success(tmp_path):
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
    )
    client = FakeJiraClient(_ticket())
    events = _EventRecorder()
    claim_snapshots: list[tuple[str, list[str], object]] = []
    graph = _RecordingGraph(
        {
            "workflow_status": "completed",
            "pull_request_url": "https://github.com/example/agent-system/pull/42",
        },
        snapshot=lambda: (
            client.ticket("AGENT-123").status,
            list(client.ticket("AGENT-123").labels),
            client.ticket("AGENT-123").fields.get(FIELD_AGENT_ASSIGNED_COMPONENT),
        ),
        snapshots=claim_snapshots,
    )
    queue: asyncio.Queue[str] = asyncio.Queue()
    execution_service = JiraExecutionService(client, "runner-1")
    runner = OrchestratorRunner(
        graph=graph,
        lock_manager=manager,
        component_id="runner-1",
        claim_ticket=execution_service.mark_claimed,
        event_emitter=events,
    )
    coordinator = JiraExecutionCoordinator(
        JiraWorkItemLoader(client),
        execution_service,
        runner,
        emit=events,
    )
    worker = ExecutionWorker(queue, coordinator, emit=events, stop_on_error=True)
    detector = _detector(client, manager, queue)

    try:
        assert await detector.poll_once() == 1
        assert await worker.run_once() is True

        ticket = client.ticket("AGENT-123")
        assert graph.invocations == 1
        assert graph.last_state is not None
        assert graph.last_state.lock_id == "lock-1"
        assert claim_snapshots == [
            (STATUS_IN_PROGRESS, [LABEL_AI_READY, LABEL_AI_CLAIMED], "runner-1")
        ]
        assert manager.current_lock("AGENT-123") is None
        assert ticket.status == STATUS_IN_REVIEW
        assert ticket.labels == [LABEL_AI_READY]
        assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
        assert client.comments_for("AGENT-123") == [
            "AI execution opened pull request:\n\n"
            "https://github.com/example/agent-system/pull/42"
        ]
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_claim_failure_releases_lock(tmp_path):
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
    )
    client = FakeJiraClient(
        _ticket(),
        fail_on={"update_fields": [RuntimeError("claim exploded"), None]},
    )
    events = _EventRecorder()
    graph = _RecordingGraph({"workflow_status": "completed"})
    queue: asyncio.Queue[str] = asyncio.Queue()
    execution_service = JiraExecutionService(client, "runner-1")
    runner = OrchestratorRunner(
        graph=graph,
        lock_manager=manager,
        component_id="runner-1",
        claim_ticket=execution_service.mark_claimed,
        event_emitter=events,
    )
    coordinator = JiraExecutionCoordinator(
        JiraWorkItemLoader(client),
        execution_service,
        runner,
    )
    worker = ExecutionWorker(queue, coordinator, stop_on_error=True)
    detector = _detector(client, manager, queue)

    try:
        assert await detector.poll_once() == 1
        with pytest.raises(TicketClaimFailedError):
            await worker.run_once()

        assert graph.invocations == 0
        assert manager.current_lock("AGENT-123") is None
        assert manager.expired_locks() == []
        assert EVENT_RUNNER_CLAIM_FAILED in events.names
        assert events.payloads[EVENT_RUNNER_CLAIM_FAILED]["error"] == (
            "mark_claimed failed for AGENT-123: claim exploded"
        )
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_second_runner_cannot_claim_same_ticket(tmp_path):
    db_path = tmp_path / "locks.sqlite3"
    manager_1 = SQLiteLockManager(
        db_path,
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
    )
    manager_2 = SQLiteLockManager(
        db_path,
        component_id="runner-2",
        lock_id_factory=lambda: "lock-2",
    )
    client = FakeJiraClient(_ticket())
    graph_1 = _BlockingGraph({"workflow_status": "completed"})
    graph_2 = _RecordingGraph({"workflow_status": "completed"})
    runner_1 = OrchestratorRunner(
        graph=graph_1,
        lock_manager=manager_1,
        component_id="runner-1",
        claim_ticket=JiraExecutionService(client, "runner-1").mark_claimed,
    )
    runner_2 = OrchestratorRunner(
        graph=graph_2,
        lock_manager=manager_2,
        component_id="runner-2",
        claim_ticket=JiraExecutionService(client, "runner-2").mark_claimed,
    )

    try:
        first_run = asyncio.create_task(runner_1.run_ticket(_work_item()))
        await graph_1.started.wait()

        with pytest.raises(TicketAlreadyLockedError):
            await runner_2.run_ticket(_work_item())

        graph_1.release_graph.set()
        final_state = await first_run

        assert final_state.workflow_status == "completed"
        assert graph_1.invocations == 1
        assert graph_2.invocations == 0
        assert manager_1.current_lock("AGENT-123") is None
        assert not any(
            call
            == (
                "update_fields",
                "AGENT-123",
                {FIELD_AGENT_ASSIGNED_COMPONENT: "runner-2"},
            )
            for call in client.calls
        )
    finally:
        manager_1.close()
        manager_2.close()


@pytest.mark.asyncio
async def test_expired_lock_reconciler_restores_jira(tmp_path):
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
                FIELD_REPOSITORY: "agent-system",
                FIELD_REPO_PATH: "/repos/agent-system",
                FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1",
                FIELD_AGENT_RETRY_COUNT: 0,
            },
        )
    )
    queue: asyncio.Queue[str] = asyncio.Queue()
    detector = _detector(client, manager, queue)

    try:
        assert manager.acquire("AGENT-123", ttl_s=1) is not None
        clock.now = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        assert manager.acquire("AGENT-123", ttl_s=60) is None
        assert await detector.poll_once() == 0

        assert await reconcile_expired_locks(manager, client) == 1

        ticket = client.ticket("AGENT-123")
        assert ticket.status == STATUS_TODO
        assert ticket.labels == [LABEL_AI_READY]
        assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
        assert ticket.fields[FIELD_AGENT_RETRY_COUNT] == 1
        assert manager.expired_locks() == []
        assert await detector.poll_once() == 1
        assert queue.get_nowait() == "AGENT-123"
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_reconciler_failure_keeps_lock(tmp_path):
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
                FIELD_REPOSITORY: "agent-system",
                FIELD_REPO_PATH: "/repos/agent-system",
                FIELD_AGENT_ASSIGNED_COMPONENT: "runner-1",
            },
        ),
        fail_on={"remove_labels": RuntimeError("remove exploded")},
    )
    events = _EventRecorder()

    try:
        assert manager.acquire("AGENT-123", ttl_s=1) is not None
        clock.now = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        assert await reconcile_expired_locks(manager, client, emit=events) == 0

        assert len(manager.expired_locks()) == 1
        assert events.names == [EVENT_LOCK_RECONCILE_FAILED]
        assert events.payloads[EVENT_LOCK_RECONCILE_FAILED] == {
            "ticket_key": "AGENT-123",
            "component_id": "runner-1",
            "lock_id": "lock-1",
            "error_type": "RuntimeError",
            "error": "remove exploded",
        }
    finally:
        manager.close()


def _detector(
    client: FakeJiraClient,
    manager: SQLiteLockManager,
    queue: asyncio.Queue[str],
) -> DetectionComponent:
    return DetectionComponent(
        client=JiraDetectionSearchClient(client),
        queue=queue,
        ownership_checker=OwnershipChecker(
            component_id="runner-1",
            lock_lookup=manager.current_lock,
        ),
        poll_interval_seconds=0.001,
    )


def _ticket(**updates: Any) -> JiraTicket:
    values = {
        "key": "AGENT-123",
        "summary": "Implement pickup",
        "description": "Flow one ticket through the execution pickup path.",
        "status": STATUS_TODO,
        "labels": [LABEL_AI_READY],
        "assignee": None,
        "fields": {
            FIELD_REPOSITORY: "agent-system",
            FIELD_REPO_PATH: "/repos/agent-system",
        },
    }
    values.update(updates)
    return JiraTicket(**values)


def _work_item() -> TicketWorkItem:
    return TicketWorkItem(
        ticket_key="AGENT-123",
        summary="Implement pickup",
        description="Flow one ticket through the execution pickup path.",
        repository="agent-system",
        repo_path="/repos/agent-system",
    )


class _RecordingGraph:
    def __init__(
        self,
        result: dict[str, Any],
        *,
        snapshot: Callable[[], Any] | None = None,
        snapshots: list[Any] | None = None,
    ) -> None:
        self.result = result
        self.snapshot = snapshot
        self.snapshots = snapshots if snapshots is not None else []
        self.invocations = 0
        self.last_state: TicketState | None = None

    async def ainvoke(
        self,
        state: TicketState,
        config: dict[str, Any] | None = None,
    ) -> TicketState:
        del config
        self.invocations += 1
        self.last_state = state
        if self.snapshot is not None:
            self.snapshots.append(self.snapshot())
        return state.model_copy(update=self.result)


class _BlockingGraph(_RecordingGraph):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(result)
        self.started = asyncio.Event()
        self.release_graph = asyncio.Event()

    async def ainvoke(
        self,
        state: TicketState,
        config: dict[str, Any] | None = None,
    ) -> TicketState:
        del config
        self.invocations += 1
        self.last_state = state
        self.started.set()
        await self.release_graph.wait()
        return state.model_copy(update=self.result)


class _MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


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

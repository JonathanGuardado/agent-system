from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from ticket_agent.locking.sqlite_store import SQLiteLockManager
from ticket_agent.orchestrator.runner import (
    EVENT_LOCK_ACQUIRED,
    EVENT_LOCK_HEARTBEAT_FAILED,
    EVENT_LOCK_RELEASED,
    EVENT_LOCK_RELEASE_FAILED,
    EVENT_RUNNER_CLAIM_FAILED,
    EVENT_TICKET_COMPLETED,
    EVENT_TICKET_FAILED,
    EVENT_TICKET_SKIPPED,
    EVENT_TICKET_STARTED,
    OrchestratorRunner,
    TicketAlreadyLockedError,
    TicketClaimFailedError,
    TicketWorkItem,
)
from ticket_agent.orchestrator.state import TicketState


def test_successful_run_acquires_lock_invokes_graph_and_releases_lock():
    graph = _Graph({"workflow_status": "completed"})
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
    events = _EventRecorder()
    runner = _runner(graph, lock_manager, event_emitter=events)

    state = asyncio.run(runner.run_ticket(_work_item()))

    assert lock_manager.acquires == ["AGENT-123"]
    assert graph.invocations == 1
    assert lock_manager.releases == [lock_manager.lock]
    assert state.workflow_status == "completed"
    assert events.names == [
        EVENT_LOCK_ACQUIRED,
        EVENT_TICKET_STARTED,
        EVENT_TICKET_COMPLETED,
        EVENT_LOCK_RELEASED,
    ]
    assert events.payloads[EVENT_LOCK_ACQUIRED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
    }
    assert events.payloads[EVENT_TICKET_STARTED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
    }
    assert events.payloads[EVENT_TICKET_COMPLETED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "workflow_status": "completed",
        "lock_id": "lock-123",
    }
    assert events.payloads[EVENT_LOCK_RELEASED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
    }


def test_lock_unavailable_skips_graph_execution():
    graph = _Graph({"workflow_status": "completed"})
    lock_manager = _LockManager(lock=None)
    events = _EventRecorder()
    runner = _runner(graph, lock_manager, event_emitter=events)

    with pytest.raises(TicketAlreadyLockedError, match="ticket is already locked"):
        asyncio.run(runner.run_ticket(_work_item()))

    assert lock_manager.acquires == ["AGENT-123"]
    assert graph.invocations == 0
    assert lock_manager.releases == []
    assert events.names == [EVENT_TICKET_SKIPPED]
    assert events.payloads[EVENT_TICKET_SKIPPED] == {
        "ticket_key": "AGENT-123",
        "reason": "already_locked",
        "component_id": "orchestrator-test",
    }


def test_graph_failure_still_releases_lock():
    graph = _Graph(RuntimeError("graph exploded"))
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
    events = _EventRecorder()
    runner = _runner(graph, lock_manager, event_emitter=events)

    state = asyncio.run(runner.run_ticket(_work_item()))

    assert graph.invocations == 1
    assert lock_manager.releases == [lock_manager.lock]
    assert state.workflow_status == "escalated"
    assert state.error == "graph exploded"
    assert state.errors == ["graph exploded"]
    assert events.names == [
        EVENT_LOCK_ACQUIRED,
        EVENT_TICKET_STARTED,
        EVENT_TICKET_FAILED,
        EVENT_LOCK_RELEASED,
    ]
    assert events.payloads[EVENT_TICKET_FAILED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "error": "graph exploded",
        "lock_id": "lock-123",
    }


def test_release_failure_after_graph_success_is_emitted_and_raised():
    graph = _Graph({"workflow_status": "completed"})
    release_error = RuntimeError("release exploded")
    lock_manager = _LockManager(
        lock=_Lock("AGENT-123", lock_id="lock-123"),
        release_error=release_error,
    )
    events = _EventRecorder()
    runner = _runner(graph, lock_manager, event_emitter=events)

    with pytest.raises(RuntimeError, match="release exploded") as exc_info:
        asyncio.run(runner.run_ticket(_work_item()))

    assert exc_info.value is release_error
    assert graph.invocations == 1
    assert lock_manager.releases == [lock_manager.lock]
    assert events.names == [
        EVENT_LOCK_ACQUIRED,
        EVENT_TICKET_STARTED,
        EVENT_TICKET_COMPLETED,
        EVENT_LOCK_RELEASE_FAILED,
    ]
    assert events.payloads[EVENT_LOCK_RELEASE_FAILED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
        "error": "release exploded",
    }


def test_release_failure_after_graph_failure_preserves_graph_exception():
    graph_error = RuntimeError("graph exploded")
    graph = _Graph(graph_error)
    lock_manager = _LockManager(
        lock=_Lock("AGENT-123", lock_id="lock-123"),
        release_error=RuntimeError("release exploded"),
    )
    events = _EventRecorder()
    runner = _runner(graph, lock_manager, event_emitter=events)

    with pytest.raises(RuntimeError, match="graph exploded") as exc_info:
        asyncio.run(runner.run_ticket(_work_item()))

    assert exc_info.value is graph_error
    assert graph.invocations == 1
    assert lock_manager.releases == [lock_manager.lock]
    assert events.names == [
        EVENT_LOCK_ACQUIRED,
        EVENT_TICKET_STARTED,
        EVENT_TICKET_FAILED,
        EVENT_LOCK_RELEASE_FAILED,
    ]
    assert events.payloads[EVENT_LOCK_RELEASE_FAILED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
        "error": "release exploded",
    }


def test_initial_ticket_state_is_built_from_work_item():
    graph = _Graph({})
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
    runner = _runner(graph, lock_manager)
    work_item = _work_item(
        description="Add the runtime shell",
        repository="agent-system",
        repo_path="/repos/agent-system",
        worktree_path="/worktrees/AGENT-123",
    )

    asyncio.run(runner.run_ticket(work_item))

    assert graph.last_state == TicketState(
        ticket_key="AGENT-123",
        summary="Implement runner",
        description="Add the runtime shell",
        repository="agent-system",
        repo_path="/repos/agent-system",
        worktree_path="/worktrees/AGENT-123",
        lock_id="lock-123",
        branch_name="agent/AGENT-123/lock-123",
    )


def test_branch_name_uses_lock_id_when_available():
    graph = _Graph({})
    lock_manager = _LockManager(lock=_Lock("PROJ-42", lock_id="abc123"))
    runner = _runner(graph, lock_manager)

    asyncio.run(runner.run_ticket(_work_item()))

    assert graph.last_state is not None
    assert graph.last_state.branch_name == "agent/AGENT-123/abc123"


def test_branch_name_falls_back_to_owner_when_no_lock_id():
    graph = _Graph({})
    lock_manager = _LockManager(lock=_LockWithOwner("AGENT-123", owner="orchestrator-7"))
    runner = _runner(graph, lock_manager)

    asyncio.run(runner.run_ticket(_work_item()))

    assert graph.last_state is not None
    assert graph.last_state.branch_name == "agent/AGENT-123/orchestrator-7"


def test_lock_id_is_copied_into_state_when_available():
    graph = _Graph({})
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="run-456"))
    runner = _runner(graph, lock_manager)

    state = asyncio.run(runner.run_ticket(_work_item()))

    assert graph.last_state is not None
    assert graph.last_state.lock_id == "run-456"
    assert state.lock_id == "run-456"


def test_claim_failure_releases_lock_without_running_graph():
    graph = _Graph({"workflow_status": "completed"})
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
    events = _EventRecorder()
    claim_error = RuntimeError("claim exploded")
    runner = _runner(
        graph,
        lock_manager,
        event_emitter=events,
        claim_ticket=lambda ticket_key: (_raise(claim_error)),
    )

    with pytest.raises(TicketClaimFailedError, match="claim exploded"):
        asyncio.run(runner.run_ticket(_work_item()))

    assert graph.invocations == 0
    assert lock_manager.releases == [lock_manager.lock]
    assert events.names == [
        EVENT_LOCK_ACQUIRED,
        EVENT_RUNNER_CLAIM_FAILED,
        EVENT_LOCK_RELEASED,
    ]
    assert events.payloads[EVENT_RUNNER_CLAIM_FAILED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
        "error": "claim exploded",
    }


def test_claim_failure_releases_sqlite_lock(tmp_path):
    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="orchestrator-test",
        lock_id_factory=lambda: "lock-123",
    )
    graph = _Graph({"workflow_status": "completed"})
    claim_error = RuntimeError("claim exploded")
    runner = OrchestratorRunner(
        graph=graph,
        lock_manager=manager,
        component_id="orchestrator-test",
        claim_ticket=lambda ticket_key: (_raise(claim_error)),
    )

    try:
        with pytest.raises(TicketClaimFailedError):
            asyncio.run(runner.run_ticket(_work_item()))

        assert graph.invocations == 0
        assert manager.has_active_lock("AGENT-123") is False
        assert manager.acquire("AGENT-123", ttl_s=60) is not None
    finally:
        manager.close()


def test_heartbeat_runs_while_graph_is_active_and_stops_after_success():
    async def scenario() -> tuple[TicketState, _BlockingGraph, _LockManager]:
        graph = _BlockingGraph()
        lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
        runner = _runner(
            graph,
            lock_manager,
            heartbeat_interval_s=0.001,
        )

        task = asyncio.create_task(runner.run_ticket(_work_item()))
        await graph.started.wait()
        while not lock_manager.heartbeats:
            await asyncio.sleep(0.001)
        graph.release_graph.set()
        state = await task
        return state, graph, lock_manager

    state, graph, lock_manager = asyncio.run(scenario())

    assert state.workflow_status == "completed"
    assert graph.invocations == 1
    assert lock_manager.heartbeats == [lock_manager.lock]
    assert lock_manager.releases == [lock_manager.lock]


def test_heartbeat_false_cancels_graph_and_returns_escalated_state():
    async def scenario() -> tuple[TicketState, _BlockingGraph, _LockManager, _EventRecorder]:
        graph = _BlockingGraph()
        lock_manager = _LockManager(
            lock=_Lock("AGENT-123", lock_id="lock-123"),
            heartbeat_results=[False],
        )
        events = _EventRecorder()
        runner = _runner(
            graph,
            lock_manager,
            event_emitter=events,
            heartbeat_interval_s=0.001,
        )

        task = asyncio.create_task(runner.run_ticket(_work_item()))
        await graph.started.wait()
        state = await task
        return state, graph, lock_manager, events

    state, graph, lock_manager, events = asyncio.run(scenario())

    assert graph.invocations == 1
    assert graph.cancelled is True
    assert lock_manager.heartbeats == [lock_manager.lock]
    assert lock_manager.releases == [lock_manager.lock]
    assert state.workflow_status == "escalated"
    assert state.error == "ticket lock heartbeat failed for AGENT-123"
    assert events.names == [
        EVENT_LOCK_ACQUIRED,
        EVENT_TICKET_STARTED,
        EVENT_LOCK_HEARTBEAT_FAILED,
        EVENT_TICKET_FAILED,
        EVENT_LOCK_RELEASED,
    ]
    assert events.payloads[EVENT_LOCK_HEARTBEAT_FAILED] == {
        "ticket_key": "AGENT-123",
        "component_id": "orchestrator-test",
        "lock_id": "lock-123",
        "error": "ticket lock heartbeat failed for AGENT-123",
    }


def test_heartbeat_interval_must_be_positive():
    with pytest.raises(ValueError, match="heartbeat_interval_s must be positive"):
        _runner(
            _Graph({}),
            _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123")),
            heartbeat_interval_s=0,
        )


def _runner(
    graph: _Graph | _BlockingGraph,
    lock_manager: _LockManager,
    *,
    event_emitter: _EventRecorder | None = None,
    claim_ticket=None,
    heartbeat_interval_s: float = 600.0,
) -> OrchestratorRunner:
    return OrchestratorRunner(
        graph=graph,
        lock_manager=lock_manager,
        component_id="orchestrator-test",
        event_emitter=event_emitter,
        claim_ticket=claim_ticket,
        heartbeat_interval_s=heartbeat_interval_s,
    )


def _work_item(**updates: Any) -> TicketWorkItem:
    values = {
        "ticket_key": "AGENT-123",
        "summary": "Implement runner",
        "description": "",
        "repository": "agent-system",
    }
    values.update(updates)
    return TicketWorkItem(**values)


@dataclass(frozen=True)
class _Lock:
    ticket_key: str
    lock_id: str


@dataclass(frozen=True)
class _LockWithOwner:
    """Mimics real TicketLock which has owner but no lock_id."""

    ticket_key: str
    owner: str


class _LockManager:
    def __init__(
        self,
        *,
        lock: _Lock | None,
        release_error: Exception | None = None,
        heartbeat_results: list[bool] | None = None,
    ) -> None:
        self.lock = lock
        self.release_error = release_error
        self.heartbeat_results = [] if heartbeat_results is None else heartbeat_results
        self.acquires: list[str] = []
        self.heartbeats: list[_Lock] = []
        self.releases: list[_Lock] = []

    def acquire(self, ticket_key: str) -> _Lock | None:
        self.acquires.append(ticket_key)
        return self.lock

    def heartbeat(self, lock: _Lock) -> bool:
        self.heartbeats.append(lock)
        if self.heartbeat_results:
            return self.heartbeat_results.pop(0)
        return True

    def release(self, lock: _Lock) -> None:
        self.releases.append(lock)
        if self.release_error is not None:
            raise self.release_error


class _Graph:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.invocations = 0
        self.last_state: TicketState | None = None

    async def ainvoke(self, state: TicketState) -> TicketState:
        self.invocations += 1
        self.last_state = state
        if isinstance(self.result, Exception):
            raise self.result
        return state.model_copy(update=self.result)


class _BlockingGraph:
    def __init__(self) -> None:
        self.invocations = 0
        self.last_state: TicketState | None = None
        self.started = asyncio.Event()
        self.release_graph = asyncio.Event()
        self.cancelled = False

    async def ainvoke(self, state: TicketState) -> TicketState:
        self.invocations += 1
        self.last_state = state
        self.started.set()
        try:
            await self.release_graph.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return state.model_copy(update={"workflow_status": "completed"})


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


def _raise(error: BaseException) -> None:
    raise error

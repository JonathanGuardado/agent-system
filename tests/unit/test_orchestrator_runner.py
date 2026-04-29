from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from ticket_agent.orchestrator.runner import (
    OrchestratorRunner,
    TicketAlreadyLockedError,
    TicketWorkItem,
)
from ticket_agent.orchestrator.state import TicketState


def test_successful_run_acquires_lock_invokes_graph_and_releases_lock():
    graph = _Graph({"workflow_status": "completed"})
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
    runner = _runner(graph, lock_manager)

    state = asyncio.run(runner.run_ticket(_work_item()))

    assert lock_manager.acquires == ["AGENT-123"]
    assert graph.invocations == 1
    assert lock_manager.releases == [lock_manager.lock]
    assert state.workflow_status == "completed"


def test_lock_unavailable_skips_graph_execution():
    graph = _Graph({"workflow_status": "completed"})
    lock_manager = _LockManager(lock=None)
    runner = _runner(graph, lock_manager)

    with pytest.raises(TicketAlreadyLockedError, match="ticket is already locked"):
        asyncio.run(runner.run_ticket(_work_item()))

    assert lock_manager.acquires == ["AGENT-123"]
    assert graph.invocations == 0
    assert lock_manager.releases == []


def test_graph_failure_still_releases_lock():
    graph = _Graph(RuntimeError("graph exploded"))
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="lock-123"))
    runner = _runner(graph, lock_manager)

    state = asyncio.run(runner.run_ticket(_work_item()))

    assert graph.invocations == 1
    assert lock_manager.releases == [lock_manager.lock]
    assert state.workflow_status == "escalated"
    assert state.error == "graph exploded"
    assert state.errors == ["graph exploded"]


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
    )


def test_lock_id_is_copied_into_state_when_available():
    graph = _Graph({})
    lock_manager = _LockManager(lock=_Lock("AGENT-123", lock_id="run-456"))
    runner = _runner(graph, lock_manager)

    state = asyncio.run(runner.run_ticket(_work_item()))

    assert graph.last_state is not None
    assert graph.last_state.lock_id == "run-456"
    assert state.lock_id == "run-456"


def _runner(graph: _Graph, lock_manager: _LockManager) -> OrchestratorRunner:
    return OrchestratorRunner(
        graph=graph,
        lock_manager=lock_manager,
        component_id="orchestrator-test",
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


class _LockManager:
    def __init__(self, *, lock: _Lock | None) -> None:
        self.lock = lock
        self.acquires: list[str] = []
        self.heartbeats: list[_Lock] = []
        self.releases: list[_Lock] = []

    def acquire(self, ticket_key: str) -> _Lock | None:
        self.acquires.append(ticket_key)
        return self.lock

    def heartbeat(self, lock: _Lock) -> bool:
        self.heartbeats.append(lock)
        return True

    def release(self, lock: _Lock) -> None:
        self.releases.append(lock)


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

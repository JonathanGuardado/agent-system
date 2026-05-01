from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.orchestrator.execution_worker import (
    EVENT_EXECUTION_WORKER_COMPLETED,
    EVENT_EXECUTION_WORKER_FAILED,
    EVENT_EXECUTION_WORKER_STARTED,
    ExecutionWorker,
)


def test_run_once_returns_false_when_queue_is_empty():
    queue = _FakeQueue()
    coordinator = _FakeCoordinator()
    worker = ExecutionWorker(queue, coordinator)

    processed = asyncio.run(worker.run_once())

    assert processed is False
    assert coordinator.calls == []
    assert queue.task_done_calls == 0


def test_run_once_processes_one_ticket_and_calls_task_done():
    queue = _FakeQueue(["AGENT-123", "AGENT-456"])
    coordinator = _FakeCoordinator()
    worker = ExecutionWorker(queue, coordinator)

    processed = asyncio.run(worker.run_once())

    assert processed is True
    assert coordinator.calls == ["AGENT-123"]
    assert queue.items == ["AGENT-456"]
    assert queue.task_done_calls == 1


def test_success_emits_started_and_completed():
    queue = _FakeQueue(["AGENT-123"])
    coordinator = _FakeCoordinator()
    events = _EventRecorder()
    worker = ExecutionWorker(queue, coordinator, emit=events)

    asyncio.run(worker.run_once())

    assert events.names == [
        EVENT_EXECUTION_WORKER_STARTED,
        EVENT_EXECUTION_WORKER_COMPLETED,
    ]
    assert events.payloads[EVENT_EXECUTION_WORKER_STARTED] == {
        "ticket_key": "AGENT-123"
    }
    assert events.payloads[EVENT_EXECUTION_WORKER_COMPLETED] == {
        "ticket_key": "AGENT-123"
    }


def test_failure_emits_failed_and_does_not_raise_when_stop_on_error_is_false():
    error = RuntimeError("runner exploded")
    queue = _FakeQueue(["AGENT-123"])
    coordinator = _FakeCoordinator(error)
    events = _EventRecorder()
    worker = ExecutionWorker(queue, coordinator, emit=events)

    processed = asyncio.run(worker.run_once())

    assert processed is True
    assert coordinator.calls == ["AGENT-123"]
    assert events.names == [
        EVENT_EXECUTION_WORKER_STARTED,
        EVENT_EXECUTION_WORKER_FAILED,
    ]
    assert events.payloads[EVENT_EXECUTION_WORKER_FAILED] == {
        "ticket_key": "AGENT-123",
        "error": "runner exploded",
    }


def test_failure_raises_when_stop_on_error_is_true():
    error = RuntimeError("runner exploded")
    queue = _FakeQueue(["AGENT-123"])
    coordinator = _FakeCoordinator(error)
    events = _EventRecorder()
    worker = ExecutionWorker(
        queue,
        coordinator,
        emit=events,
        stop_on_error=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(worker.run_once())

    assert exc_info.value is error
    assert events.names == [
        EVENT_EXECUTION_WORKER_STARTED,
        EVENT_EXECUTION_WORKER_FAILED,
    ]


def test_task_done_is_called_even_on_failure():
    queue = _FakeQueue(["AGENT-123"])
    coordinator = _FakeCoordinator(RuntimeError("runner exploded"))
    worker = ExecutionWorker(queue, coordinator, stop_on_error=True)

    with pytest.raises(RuntimeError):
        asyncio.run(worker.run_once())

    assert queue.task_done_calls == 1


def test_run_forever_propagates_cancelled_error():
    queue = _FakeQueue()
    coordinator = _FakeCoordinator()
    worker = ExecutionWorker(queue, coordinator)

    async def cancel_during_sleep() -> None:
        task = asyncio.create_task(worker.run_forever(poll_interval_s=60.0))
        await asyncio.sleep(0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_during_sleep())


class _FakeQueue:
    def __init__(self, items: list[str] | None = None) -> None:
        self.items = [] if items is None else list(items)
        self.task_done_calls = 0

    def get_nowait(self) -> str:
        if not self.items:
            raise asyncio.QueueEmpty
        return self.items.pop(0)

    def task_done(self) -> None:
        self.task_done_calls += 1


class _FakeCoordinator:
    def __init__(self, result: BaseException | None = None) -> None:
        self.result = result
        self.calls: list[str] = []

    async def run_ticket(self, ticket_key: str) -> object:
        self.calls.append(ticket_key)
        if self.result is not None:
            raise self.result
        return object()


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

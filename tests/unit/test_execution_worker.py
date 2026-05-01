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


@pytest.mark.asyncio
async def test_run_once_returns_false_when_queue_is_empty():
    queue = _RecordingQueue()
    coordinator = _FakeCoordinator()
    worker = ExecutionWorker(queue, coordinator)

    processed = await worker.run_once()

    assert processed is False
    assert coordinator.calls == []
    assert queue.task_done_calls == 0


@pytest.mark.asyncio
async def test_run_once_processes_one_ticket_and_calls_coordinator():
    queue = _queue_with("AGENT-123", "AGENT-456")
    coordinator = _FakeCoordinator()
    worker = ExecutionWorker(queue, coordinator)

    processed = await worker.run_once()

    assert processed is True
    assert coordinator.calls == ["AGENT-123"]
    assert queue.get_nowait() == "AGENT-456"
    assert queue.task_done_calls == 1


@pytest.mark.asyncio
async def test_successful_run_emits_started_and_completed():
    queue = _queue_with("AGENT-123")
    coordinator = _FakeCoordinator()
    events = _EventRecorder()
    worker = ExecutionWorker(queue, coordinator, emit=events)

    await worker.run_once()

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


@pytest.mark.asyncio
async def test_failure_emits_failed_and_does_not_raise_when_stop_on_error_is_false():
    error = RuntimeError("runner exploded")
    queue = _queue_with("AGENT-123")
    coordinator = _FakeCoordinator(error)
    events = _EventRecorder()
    worker = ExecutionWorker(queue, coordinator, emit=events)

    processed = await worker.run_once()

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


@pytest.mark.asyncio
async def test_failure_raises_when_stop_on_error_is_true():
    error = RuntimeError("runner exploded")
    queue = _queue_with("AGENT-123")
    coordinator = _FakeCoordinator(error)
    events = _EventRecorder()
    worker = ExecutionWorker(
        queue,
        coordinator,
        emit=events,
        stop_on_error=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await worker.run_once()

    assert exc_info.value is error
    assert events.names == [
        EVENT_EXECUTION_WORKER_STARTED,
        EVENT_EXECUTION_WORKER_FAILED,
    ]


@pytest.mark.asyncio
async def test_task_done_is_called_even_when_coordinator_run_ticket_fails():
    queue = _queue_with("AGENT-123")
    coordinator = _FakeCoordinator(RuntimeError("runner exploded"))
    worker = ExecutionWorker(queue, coordinator, stop_on_error=True)

    with pytest.raises(RuntimeError):
        await worker.run_once()

    assert queue.task_done_calls == 1


@pytest.mark.asyncio
async def test_run_forever_propagates_cancelled_error():
    queue = _RecordingQueue()
    coordinator = _FakeCoordinator()
    worker = ExecutionWorker(queue, coordinator)

    task = asyncio.create_task(worker.run_forever(poll_interval_s=60.0))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


class _RecordingQueue(asyncio.Queue[str]):
    def __init__(self) -> None:
        super().__init__()
        self.task_done_calls = 0

    def task_done(self) -> None:
        self.task_done_calls += 1
        super().task_done()


def _queue_with(*ticket_keys: str) -> _RecordingQueue:
    queue = _RecordingQueue()
    for ticket_key in ticket_keys:
        queue.put_nowait(ticket_key)
    return queue


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

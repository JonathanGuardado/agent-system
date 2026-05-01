"""Queue worker for running ticket execution coordinators."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from inspect import isawaitable
from typing import Protocol

EventEmitter = Callable[[str, dict[str, object]], object]

EVENT_EXECUTION_WORKER_STARTED = "execution_worker_started"
EVENT_EXECUTION_WORKER_COMPLETED = "execution_worker_completed"
EVENT_EXECUTION_WORKER_FAILED = "execution_worker_failed"


class TicketExecutionCoordinator(Protocol):
    """Minimal execution boundary used by the queue worker."""

    async def run_ticket(self, ticket_key: str) -> object:
        """Run one ticket by key."""


class ExecutionWorker:
    """Consume ticket keys from a queue and run them through a coordinator."""

    def __init__(
        self,
        queue: asyncio.Queue[str],
        coordinator: TicketExecutionCoordinator,
        emit: EventEmitter | None = None,
        stop_on_error: bool = False,
    ) -> None:
        self._queue = queue
        self._coordinator = coordinator
        self._event_emitter = emit
        self._stop_on_error = stop_on_error

    async def run_once(self) -> bool:
        """Process one queued ticket key if one is immediately available."""

        try:
            ticket_key = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False

        try:
            await self._emit(EVENT_EXECUTION_WORKER_STARTED, ticket_key=ticket_key)
            try:
                await self._coordinator.run_ticket(ticket_key)
            except Exception as exc:
                await self._emit_failed(ticket_key, exc)
                if self._stop_on_error:
                    raise
            else:
                await self._emit(
                    EVENT_EXECUTION_WORKER_COMPLETED,
                    ticket_key=ticket_key,
                )
            return True
        finally:
            self._queue.task_done()

    async def run_forever(self, poll_interval_s: float = 1.0) -> None:
        """Poll for queued ticket keys until the worker task is cancelled."""

        while True:
            processed = await self.run_once()
            if not processed:
                await asyncio.sleep(poll_interval_s)

    async def _emit(self, event_name: str, **payload: object) -> None:
        if self._event_emitter is None:
            return
        result = self._event_emitter(event_name, payload)
        if isawaitable(result):
            await result

    async def _emit_failed(self, ticket_key: str, error: BaseException) -> None:
        await self._emit(
            EVENT_EXECUTION_WORKER_FAILED,
            ticket_key=ticket_key,
            error=_error_message(error),
        )


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


__all__ = [
    "EVENT_EXECUTION_WORKER_COMPLETED",
    "EVENT_EXECUTION_WORKER_FAILED",
    "EVENT_EXECUTION_WORKER_STARTED",
    "EventEmitter",
    "ExecutionWorker",
    "TicketExecutionCoordinator",
]

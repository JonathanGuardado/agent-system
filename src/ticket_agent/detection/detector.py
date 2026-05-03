"""Polling-based detector for ai-ready Jira tickets."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from inspect import isawaitable
from logging import Logger
from typing import Any, Protocol

from ticket_agent.detection.ownership import OwnershipChecker, OwnershipDecision
from ticket_agent.jira.constants import LABEL_AI_READY, STATUS_TODO
from ticket_agent.jira.models import JiraTicket


EventEmitter = Callable[[str, Mapping[str, Any]], Any]
TicketQueue = asyncio.Queue


EVENT_DETECTION_POLL_STARTED = "detection.poll_started"
EVENT_DETECTION_ENQUEUED = "detection.enqueued"
EVENT_DETECTION_SKIPPED = "detection.skipped"
EVENT_DETECTION_POLL_COMPLETED = "detection.poll_completed"
EVENT_DETECTION_POLL_FAILED = "detection.poll_failed"


class DetectionSearchClient(Protocol):
    """Async boundary for querying candidate ai-ready Jira tickets."""

    async def search_ai_ready_tickets(self) -> Sequence[JiraTicket]:
        """Return a list of candidate ai-ready Jira tickets."""


class DetectionComponent:
    """Poll a Jira search boundary and enqueue eligible ticket keys."""

    def __init__(
        self,
        *,
        client: DetectionSearchClient,
        queue: TicketQueue,
        ownership_checker: OwnershipChecker,
        poll_interval_seconds: float = 30.0,
        max_backoff_seconds: float = 300.0,
        emit: EventEmitter | None = None,
        logger: Logger | None = None,
        clock: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if max_backoff_seconds < poll_interval_seconds:
            raise ValueError(
                "max_backoff_seconds must be >= poll_interval_seconds"
            )

        self._client = client
        self._queue = queue
        self._checker = ownership_checker
        self._poll_interval = float(poll_interval_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._emit_fn = emit
        self._logger = logger
        self._clock = clock or asyncio.sleep

        self._in_flight: set[str] = set()

    @property
    def in_flight(self) -> frozenset[str]:
        """Snapshot of ticket keys we have enqueued and not yet released."""

        return frozenset(self._in_flight)

    def mark_done(self, ticket_key: str) -> None:
        """Allow the same ticket to be re-enqueued by future polls."""

        self._in_flight.discard(ticket_key)

    async def poll_once(self) -> int:
        """Run a single detection poll. Return the number of enqueued keys."""

        await self._emit(EVENT_DETECTION_POLL_STARTED, query=_query_descriptor())

        try:
            tickets = await self._client.search_ai_ready_tickets()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                EVENT_DETECTION_POLL_FAILED,
                error=str(exc) or exc.__class__.__name__,
            )
            if self._logger is not None:
                self._logger.exception("detection_poll_failed")
            raise

        seen_in_batch: set[str] = set()
        enqueued = 0
        skipped = 0

        for ticket in tickets:
            key = getattr(ticket, "key", None)
            if not isinstance(key, str) or not key.strip():
                continue
            ticket_key = key.strip()
            if ticket_key in seen_in_batch:
                continue
            seen_in_batch.add(ticket_key)

            if ticket_key in self._in_flight:
                await self._emit(
                    EVENT_DETECTION_SKIPPED,
                    ticket_key=ticket_key,
                    reason="already_in_flight",
                )
                skipped += 1
                continue

            decision = self._checker.check(ticket)
            if not decision.eligible:
                if decision.reason:
                    await self._emit(
                        EVENT_DETECTION_SKIPPED,
                        ticket_key=ticket_key,
                        reason=decision.reason,
                    )
                skipped += 1
                continue

            await self._queue.put(ticket_key)
            self._in_flight.add(ticket_key)
            enqueued += 1
            await self._emit(
                EVENT_DETECTION_ENQUEUED,
                ticket_key=ticket_key,
            )

        await self._emit(
            EVENT_DETECTION_POLL_COMPLETED,
            considered=len(seen_in_batch),
            enqueued=enqueued,
            skipped=skipped,
        )
        return enqueued

    async def run_forever(self) -> None:
        """Loop poll_once forever with backoff-on-error semantics."""

        backoff = self._poll_interval
        while True:
            try:
                await self.poll_once()
                backoff = self._poll_interval
            except asyncio.CancelledError:
                raise
            except Exception:
                backoff = min(backoff * 2, self._max_backoff)
                if backoff <= 0:
                    backoff = self._poll_interval
            try:
                await self._clock(backoff if backoff > 0 else self._poll_interval)
            except asyncio.CancelledError:
                raise

    async def _emit(self, event_name: str, **payload: Any) -> None:
        if self._emit_fn is None:
            return
        result = self._emit_fn(event_name, payload)
        if isawaitable(result):
            await result


def _query_descriptor() -> dict[str, Any]:
    return {
        "label": LABEL_AI_READY,
        "status": STATUS_TODO,
    }


__all__ = [
    "EVENT_DETECTION_ENQUEUED",
    "EVENT_DETECTION_POLL_COMPLETED",
    "EVENT_DETECTION_POLL_FAILED",
    "EVENT_DETECTION_POLL_STARTED",
    "EVENT_DETECTION_SKIPPED",
    "DetectionComponent",
    "DetectionSearchClient",
]

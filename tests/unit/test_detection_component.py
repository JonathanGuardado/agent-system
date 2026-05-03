from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from ticket_agent.detection.detector import (
    DetectionComponent,
    EVENT_DETECTION_ENQUEUED,
    EVENT_DETECTION_POLL_COMPLETED,
    EVENT_DETECTION_POLL_FAILED,
    EVENT_DETECTION_POLL_STARTED,
    EVENT_DETECTION_SKIPPED,
)
from ticket_agent.detection.ownership import OwnershipChecker
from ticket_agent.jira.constants import (
    FIELD_AGENT_RETRY_COUNT,
    LABEL_AI_READY,
    STATUS_IN_PROGRESS,
    STATUS_TODO,
)
from ticket_agent.jira.models import JiraTicket


COMPONENT_ID = "agent-system"


class _FakeSearchClient:
    def __init__(self, batches: Sequence[Sequence[JiraTicket] | Exception]) -> None:
        self._batches = list(batches)
        self.calls = 0

    async def search_ai_ready_tickets(self) -> Sequence[JiraTicket]:
        self.calls += 1
        if not self._batches:
            return []
        item = self._batches.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, dict(payload)))


def _eligible_ticket(key: str = "AGENT-1") -> JiraTicket:
    return JiraTicket(
        key=key,
        summary="Add feature",
        description="",
        status=STATUS_TODO,
        labels=[LABEL_AI_READY],
        assignee=None,
        fields={},
    )


def _ineligible_ticket(key: str = "AGENT-2") -> JiraTicket:
    return JiraTicket(
        key=key,
        summary="In progress",
        description="",
        status=STATUS_IN_PROGRESS,
        labels=[LABEL_AI_READY],
        assignee=None,
        fields={},
    )


def _build_component(
    *,
    client: _FakeSearchClient,
    queue: asyncio.Queue,
    emit: _Recorder | None = None,
    max_retries: int = 3,
    lock_lookup=None,
    poll_interval_seconds: float = 0.001,
) -> DetectionComponent:
    checker = OwnershipChecker(
        component_id=COMPONENT_ID,
        lock_lookup=lock_lookup or (lambda key: None),
        max_retries=max_retries,
    )
    return DetectionComponent(
        client=client,
        queue=queue,
        ownership_checker=checker,
        poll_interval_seconds=poll_interval_seconds,
        max_backoff_seconds=1.0,
        emit=emit,
    )


def test_poll_once_enqueues_eligible_ticket():
    client = _FakeSearchClient([[_eligible_ticket("AGENT-1")]])
    queue: asyncio.Queue = asyncio.Queue()
    recorder = _Recorder()
    detector = _build_component(client=client, queue=queue, emit=recorder)

    enqueued = asyncio.run(detector.poll_once())

    assert enqueued == 1
    assert queue.get_nowait() == "AGENT-1"
    event_names = [name for name, _ in recorder.events]
    assert EVENT_DETECTION_POLL_STARTED in event_names
    assert EVENT_DETECTION_ENQUEUED in event_names
    assert EVENT_DETECTION_POLL_COMPLETED in event_names


def test_poll_once_skips_ineligible_ticket_and_emits_reason():
    client = _FakeSearchClient([[_ineligible_ticket("AGENT-2")]])
    queue: asyncio.Queue = asyncio.Queue()
    recorder = _Recorder()
    detector = _build_component(client=client, queue=queue, emit=recorder)

    enqueued = asyncio.run(detector.poll_once())

    assert enqueued == 0
    assert queue.empty()
    skipped_events = [
        payload
        for name, payload in recorder.events
        if name == EVENT_DETECTION_SKIPPED
    ]
    assert len(skipped_events) == 1
    assert skipped_events[0]["ticket_key"] == "AGENT-2"
    assert skipped_events[0]["reason"].startswith("wrong_status:")


def test_poll_once_handles_empty_search_results():
    client = _FakeSearchClient([[]])
    queue: asyncio.Queue = asyncio.Queue()
    recorder = _Recorder()
    detector = _build_component(client=client, queue=queue, emit=recorder)

    enqueued = asyncio.run(detector.poll_once())

    assert enqueued == 0
    assert queue.empty()
    completed = [
        payload
        for name, payload in recorder.events
        if name == EVENT_DETECTION_POLL_COMPLETED
    ]
    assert completed == [{"considered": 0, "enqueued": 0, "skipped": 0}]


def test_poll_once_does_not_enqueue_duplicates_within_one_batch():
    duplicate = _eligible_ticket("AGENT-3")
    client = _FakeSearchClient([[duplicate, duplicate]])
    queue: asyncio.Queue = asyncio.Queue()
    detector = _build_component(client=client, queue=queue)

    enqueued = asyncio.run(detector.poll_once())

    assert enqueued == 1
    assert queue.qsize() == 1


def test_poll_once_does_not_re_enqueue_in_flight_ticket():
    client = _FakeSearchClient(
        [
            [_eligible_ticket("AGENT-1")],
            [_eligible_ticket("AGENT-1")],
        ]
    )
    queue: asyncio.Queue = asyncio.Queue()
    recorder = _Recorder()
    detector = _build_component(client=client, queue=queue, emit=recorder)

    asyncio.run(detector.poll_once())
    enqueued_second = asyncio.run(detector.poll_once())

    assert enqueued_second == 0
    assert queue.qsize() == 1
    skipped_reasons = [
        payload["reason"]
        for name, payload in recorder.events
        if name == EVENT_DETECTION_SKIPPED
    ]
    assert "already_in_flight" in skipped_reasons


def test_mark_done_allows_re_enqueue():
    client = _FakeSearchClient(
        [
            [_eligible_ticket("AGENT-1")],
            [_eligible_ticket("AGENT-1")],
        ]
    )
    queue: asyncio.Queue = asyncio.Queue()
    detector = _build_component(client=client, queue=queue)

    asyncio.run(detector.poll_once())
    queue.get_nowait()
    detector.mark_done("AGENT-1")
    enqueued_second = asyncio.run(detector.poll_once())

    assert enqueued_second == 1


def test_poll_once_emits_failed_event_and_reraises_on_search_error():
    client = _FakeSearchClient([RuntimeError("boom")])
    queue: asyncio.Queue = asyncio.Queue()
    recorder = _Recorder()
    detector = _build_component(client=client, queue=queue, emit=recorder)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(detector.poll_once())

    failed_events = [
        payload
        for name, payload in recorder.events
        if name == EVENT_DETECTION_POLL_FAILED
    ]
    assert len(failed_events) == 1
    assert failed_events[0]["error"] == "boom"


def test_run_forever_propagates_cancelled_error():
    client = _FakeSearchClient([[_eligible_ticket("AGENT-1")]])
    queue: asyncio.Queue = asyncio.Queue()
    detector = _build_component(
        client=client,
        queue=queue,
        poll_interval_seconds=0.001,
    )

    async def main() -> None:
        task = asyncio.create_task(detector.run_forever())
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())


def test_run_forever_uses_backoff_then_recovers():
    sleeps: list[float] = []
    error = RuntimeError("transient")
    client = _FakeSearchClient(
        [
            error,
            error,
            [_eligible_ticket("AGENT-1")],
        ]
    )
    queue: asyncio.Queue = asyncio.Queue()

    async def fake_clock(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError()

    detector = DetectionComponent(
        client=client,
        queue=queue,
        ownership_checker=OwnershipChecker(
            component_id=COMPONENT_ID,
            lock_lookup=lambda key: None,
        ),
        poll_interval_seconds=1.0,
        max_backoff_seconds=10.0,
        clock=fake_clock,
    )

    async def main() -> None:
        with pytest.raises(asyncio.CancelledError):
            await detector.run_forever()

    asyncio.run(main())

    assert sleeps[0] == 2.0
    assert sleeps[1] == 4.0
    assert sleeps[2] == 1.0
    assert queue.qsize() == 1

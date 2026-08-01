"""Tests that the transcript hooks fire, and stay silent by default.

The wiring here is the part most likely to rot: every hook takes a recorder
that defaults to the no-op, so a hook that is never called looks exactly like
a hook that works.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.domain.model import ModelResponse
from ticket_agent.observability.telemetry import SQLiteTelemetryStore
from ticket_agent.observability.transcripts import TranscriptEvent
from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.model_services import IterativeImplementationService
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.router.model_router import ModelRouter


class _Collecting:
    """A recorder that keeps events in memory."""

    def __init__(self) -> None:
        self.events: list[TranscriptEvent] = []

    def record(self, event: TranscriptEvent) -> None:
        self.events.append(event)

    def names(self, kind: str) -> list[str]:
        return [e.name for e in self.events if e.kind == kind]


class _Exploding:
    def record(self, event: TranscriptEvent) -> None:
        raise RuntimeError("recorder is broken")


# -- node runner -----------------------------------------------------------


class _Service:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def plan(self, state: TicketState) -> Any:
        return self._result

    async def request_approval(self, state: TicketState) -> Any:
        return self._result

    async def implement(self, state: TicketState) -> Any:
        return self._result

    async def run_tests(self, state: TicketState) -> Any:
        return self._result

    async def review(self, state: TicketState) -> Any:
        return self._result

    async def open_pull_request(self, state: TicketState) -> Any:
        return self._result

    async def escalate(self, state: TicketState, reason: str) -> Any:
        return None


def _runner(transcripts: Any = None) -> TicketNodeRunner:
    return TicketNodeRunner(
        planner=_Service({"steps": []}),
        approval=_Service(True),
        implementation=_Service(
            {
                "implementation_result": {
                    "status": "success",
                    "changed_files": ["a.py"],
                }
            }
        ),
        tests=_Service({"status": "passed", "tests_passed": True}),
        review=_Service({"status": "accepted", "passed": True}),
        pull_request=_Service("https://github.test/acme/repo/pull/1"),
        escalation=_Service(None),
        transcripts=transcripts,
    )


def _state(**updates: Any) -> TicketState:
    return TicketState(ticket_key="LAB-30", summary="Ship it", **updates)


def test_node_runner_records_every_node_it_visits():
    recorder = _Collecting()
    graph = build_ticket_graph(_runner(recorder))

    asyncio.run(graph.ainvoke(_state(lock_id="abcd1234ef")))

    assert recorder.names("node") == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "report",
    ]


def test_node_runner_records_keys_and_scalars_but_not_whole_updates():
    recorder = _Collecting()

    asyncio.run(_runner(recorder).implement(_state(lock_id="abcd1234ef")))

    payload = recorder.events[0].payload
    assert payload["workflow_status"] == "implementing"
    assert "implementation_result" in payload["update_keys"]
    # The result itself must not be inlined -- it can carry file content.
    assert "implementation_result" not in payload


def test_node_runner_propagates_goal_and_run_identifiers():
    recorder = _Collecting()

    asyncio.run(_runner(recorder).plan(_state(lock_id="abcd1234ef", goal_id="g1")))

    event = recorder.events[0]
    assert event.ticket_key == "LAB-30"
    assert event.goal_id == "g1"
    assert event.run_id == "abcd1234ef"


def test_node_runner_defaults_to_the_no_op_recorder():
    update = asyncio.run(_runner().plan(_state()))

    assert update["current_node"] == "plan"


# -- router ----------------------------------------------------------------


class _Selected:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.model_name = "fake-model"
        self.deployment_name = "fake-model"


class _Decision:
    capability = "code.implement"

    def __init__(self) -> None:
        self.primary = _Selected("fake")
        self.fallbacks = ()


class _Selector:
    """Matches the router's simplest duck-typed branch: select(capability)."""

    def select(self, _capability: str) -> _Decision:
        return _Decision()


class _Provider:
    class _Response:
        content = '{"action": "finish", "args": {"summary": "done"}}'
        input_tokens = 10
        output_tokens = 5
        estimated_cost_usd = 0.001

    async def chat(self, model: str, messages: list[dict], timeout_s: int) -> Any:
        return self._Response()


def test_router_fans_model_events_out_to_the_transcript():
    recorder = _Collecting()
    router = ModelRouter(
        selector=_Selector(),
        providers={"fake": _Provider()},
        transcripts=recorder,
    )

    asyncio.run(
        router.invoke(
            capability="code.implement",
            messages=[{"role": "user", "content": "hi"}],
            ticket_id="LAB-30",
            metadata={"workflow_node": "implement", "implementation_turn": 1},
        )
    )

    assert recorder.names("model") == [
        "model.invoke_attempt_started",
        "model.invoke_attempt_completed",
    ]
    payload = recorder.events[-1].payload
    assert payload["provider"] == "fake"
    assert payload["workflow_node"] == "implement"
    # ticket_id is promoted to the event, not duplicated in the payload.
    assert "ticket_id" not in payload


def test_router_skips_recording_when_no_ticket_is_supplied():
    recorder = _Collecting()
    router = ModelRouter(
        selector=_Selector(), providers={"fake": _Provider()}, transcripts=recorder
    )

    asyncio.run(
        router.invoke(
            capability="code.implement",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert recorder.events == []


# -- implementation loop ---------------------------------------------------


class _LoopRouter:
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)

    async def invoke(self, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content=self._contents.pop(0),
            model="m",
            provider="p",
            capability="code.implement",
        )


def test_implementation_loop_records_start_turns_and_end():
    recorder = _Collecting()
    service = IterativeImplementationService(
        _LoopRouter(['{"action": "finish", "args": {"summary": "done"}}']),
        transcripts=recorder,
    )

    asyncio.run(
        service._run_loop(_state(lock_id="abcd1234ef", worktree_path="/tmp/wt"), None)
    )

    names = recorder.names("loop")
    assert names[0] == "loop.start"
    assert "turn.1.call" in names
    assert names[-1] == "loop.end"
    assert recorder.events[-1].payload["turns_used"] == 1


class _Files:
    """Minimal FileAdapter stand-in for actions that touch the worktree."""

    def list_files(self, path: str) -> list[str]:
        return ["a.py"]


def test_implementation_loop_records_end_when_max_turns_exhausted():
    recorder = _Collecting()
    service = IterativeImplementationService(
        _LoopRouter(['{"action": "list_dir", "args": {"path": "."}}'] * 2),
        file_adapter_factory=lambda _path: _Files(),
        max_turns=2,
        transcripts=recorder,
    )

    asyncio.run(
        service._run_loop(
            _state(lock_id="abcd1234ef", worktree_path="/tmp/wt"), _Files()
        )
    )

    end = recorder.events[-1]
    assert end.name == "loop.end"
    assert end.payload["turns_used"] == 2


def test_a_broken_recorder_never_breaks_the_run():
    """Observability failures must not become pipeline failures."""

    update = asyncio.run(_runner(_Exploding()).plan(_state()))

    assert update["current_node"] == "plan"


# -- funnel stages ---------------------------------------------------------


def _telemetry_runner(store: Any, **overrides: Any) -> TicketNodeRunner:
    runner = _runner()
    return TicketNodeRunner(
        planner=runner._planner,
        approval=overrides.get("approval", runner._approval),
        implementation=runner._implementation,
        tests=runner._tests,
        review=runner._review,
        pull_request=overrides.get("pull_request", runner._pull_request),
        escalation=runner._escalation,
        telemetry=store,
    )


def test_graph_run_fills_the_funnel(tmp_path):
    """The funnel must actually fill from a run, not just be writable."""

    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    graph = build_ticket_graph(_telemetry_runner(store))

    asyncio.run(graph.ainvoke(_state(lock_id="abcd1234ef", goal_id="g1")))
    reached = {k for k, v in store.funnel_counts().items() if v}
    store.close()

    assert reached == {"planned", "approved", "implemented", "pr_opened"}


def test_implementation_node_produces_loop_iteration_telemetry(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    runner = _telemetry_runner(store)

    asyncio.run(
        runner.implement(
            _state(goal_id="prop-0123456789ab", implementation_attempts=2)
        )
    )
    totals = store.iteration_totals()
    store.close()

    assert totals == [
        {
            "goal_id": "prop-0123456789ab",
            "iterations": 1,
            "tokens": 0,
            "cost_usd": 0,
        }
    ]


def test_unapproved_execution_does_not_count_as_approved(tmp_path):
    """A node running is not the same as its stage being reached."""

    class _Denied:
        async def request_approval(self, state: TicketState) -> Any:
            return False

    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    runner = _telemetry_runner(store, approval=_Denied())

    asyncio.run(runner.request_execution_approval(_state()))
    reached = {k for k, v in store.funnel_counts().items() if v}
    store.close()

    assert "approved" not in reached


def test_pull_request_without_a_url_does_not_count_as_opened(tmp_path):
    class _NoUrl:
        async def open_pull_request(self, state: TicketState) -> Any:
            return ""

    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    runner = _telemetry_runner(store, pull_request=_NoUrl())

    asyncio.run(runner.open_pull_request(_state()))
    reached = {k for k, v in store.funnel_counts().items() if v}
    store.close()

    assert "pr_opened" not in reached


def test_escalation_records_its_reason(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    runner = _telemetry_runner(store)

    asyncio.run(runner.escalate(_state(escalation_reason="tests failed")))
    reasons = store.escalation_reasons()
    store.close()

    assert reasons[0]["reason"] == "tests failed"

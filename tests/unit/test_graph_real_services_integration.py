"""Integration tests that wire real service implementations through the graph.

Each test uses a scripted fake router (no network) and real file adapter
(real disk writes in tmp_path). This exercises the full plan → approve →
implement → test → review → PR → report path with actual service logic,
not just protocol stubs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.local_services import AutoApprovalService
from ticket_agent.orchestrator.model_services import (
    IterativeImplementationService,
    ModelRouterPlannerService,
    ModelRouterReviewService,
)
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.state import TicketState


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_full_graph_writes_file_through_iterative_implementation(tmp_path):
    """Real planner + real iterative impl + real file adapter → file on disk."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "hello.py").write_text("# original\n")

    router = _ScriptedRouter(
        {
            "ticket.decompose": [
                {
                    "plan": "Add a hello() function to src/hello.py",
                    "files_to_modify": ["src/hello.py"],
                    "risks": [],
                    "complexity": "low",
                    "requires_human_review": False,
                }
            ],
            "code.implement": [
                {"action": "read_file", "args": {"path": "src/hello.py"}},
                {
                    "action": "write_file",
                    "args": {
                        "path": "src/hello.py",
                        "content": "# modified\ndef hello():\n    return 'hello'\n",
                    },
                },
                {
                    "action": "finish",
                    "args": {"summary": "Added hello() to src/hello.py"},
                },
            ],
            "code.verify": [
                {
                    "passed": True,
                    "reasoning": "Implementation looks correct.",
                    "issues": [],
                }
            ],
        }
    )

    runner = TicketNodeRunner(
        planner=ModelRouterPlannerService(router),
        approval=AutoApprovalService(),
        implementation=IterativeImplementationService(
            router,
            file_adapter_factory=LocalFileAdapter,
        ),
        tests=_AlwaysPassTestService(),
        review=ModelRouterReviewService(router),
        pull_request=_FakePullRequestService("https://github.test/acme/repo/pull/7"),
        escalation=_RecordingEscalationService(),
    )

    graph = build_ticket_graph(runner)
    result = asyncio.run(
        graph.ainvoke(
            TicketState(
                ticket_key="AGENT-123",
                summary="Add greeting function",
                description="Add a hello() function to src/hello.py",
                worktree_path=str(tmp_path),
                branch_name="agent/AGENT-123/abcdef12",
                repository="test-repo",
            )
        )
    )

    state = TicketState.model_validate(result)

    assert state.workflow_status == "completed"
    assert state.pull_request_url == "https://github.test/acme/repo/pull/7"
    assert state.implementation_result is not None
    assert state.implementation_result["status"] == "success"
    assert "src/hello.py" in state.implementation_result["changed_files"]
    assert state.tests_passed is True
    assert state.review_passed is True
    assert state.escalation_reason is None
    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "report",
    ]

    written = (tmp_path / "src" / "hello.py").read_text()
    assert "def hello" in written


# ---------------------------------------------------------------------------
# Policy-violation escalation
# ---------------------------------------------------------------------------


def test_full_graph_escalates_immediately_on_policy_violation_write(tmp_path):
    """A write to a protected path (`.env`) triggers immediate escalation."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("# app\n")

    router = _ScriptedRouter(
        {
            "ticket.decompose": [
                {
                    "plan": "Update app",
                    "files_to_modify": ["src/app.py"],
                    "risks": [],
                    "complexity": "low",
                    "requires_human_review": False,
                }
            ],
            "code.implement": [
                # Attempt to write a protected file → tool result will be ok=False
                {
                    "action": "write_file",
                    "args": {"path": ".env", "content": "SECRET=leaked"},
                },
                # Should never be reached — loop exits on failed tool result
            ],
        }
    )
    escalation = _RecordingEscalationService()

    runner = TicketNodeRunner(
        planner=ModelRouterPlannerService(router),
        approval=AutoApprovalService(),
        implementation=IterativeImplementationService(
            router,
            file_adapter_factory=LocalFileAdapter,
            max_turns=5,
        ),
        tests=_AlwaysPassTestService(),
        review=_FakeReviewService({"passed": True, "status": "accepted"}),
        pull_request=_FakePullRequestService("https://github.test/acme/repo/pull/8"),
        escalation=escalation,
    )

    graph = build_ticket_graph(runner)
    result = asyncio.run(
        graph.ainvoke(
            TicketState(
                ticket_key="AGENT-456",
                summary="Update app config",
                worktree_path=str(tmp_path),
                branch_name="agent/AGENT-456/deadbeef",
                repository="test-repo",
            )
        )
    )

    state = TicketState.model_validate(result)

    assert state.workflow_status == "escalated"
    assert state.pull_request_url is None
    assert len(escalation.calls) == 1
    ticket_key, reason = escalation.calls[0]
    assert ticket_key == "AGENT-456"
    assert "policy_violation" in state.implementation_result.get("error_code", "") or (
        "implementation" in reason.lower()
    )

    assert not (tmp_path / ".env").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _RouterCall:
    capability: str
    messages: list[dict]


class _ScriptedRouter:
    """Returns pre-scripted responses per capability in order, no network."""

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses = {cap: list(resps) for cap, resps in responses.items()}
        self.calls: list[_RouterCall] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> Any:
        queue = self._responses.get(capability)
        if not queue:
            raise RuntimeError(
                f"_ScriptedRouter: no more scripted responses for capability={capability!r}"
            )
        response = queue.pop(0)
        self.calls.append(_RouterCall(capability=capability, messages=list(messages)))
        return response


class _AlwaysPassTestService:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed", "tests_passed": True, "output": ""}


class _FakeReviewService:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    async def review(self, state: TicketState) -> dict[str, Any]:
        return self._result


class _FakePullRequestService:
    def __init__(self, url: str) -> None:
        self._url = url

    async def open_pull_request(self, state: TicketState) -> str:
        return self._url


class _RecordingEscalationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def escalate(self, state: TicketState, reason: str) -> None:
        self.calls.append((state.ticket_key, reason))

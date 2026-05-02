from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.model_services import (
    ModelRouterImplementationService,
    ModelRouterPlannerService,
    ModelRouterReviewService,
    ModelServiceError,
)
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.state import TicketState


def test_planner_calls_router_and_parses_dict_response():
    router = _FakeRouter(
        {
            "ticket.decompose": {
                "plan": "Add pagination support.",
                "files_to_modify": ["src/api/users.py"],
                "risks": ["Query behavior may change"],
                "complexity": "medium",
                "requires_human_review": False,
            }
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert router.calls[0].capability == "ticket.decompose"
    assert router.calls[0].messages[0]["role"] == "system"
    assert result == {
        "plan": "Add pagination support.",
        "files_to_modify": ["src/api/users.py"],
        "risks": ["Query behavior may change"],
        "complexity": "medium",
        "requires_human_review": False,
    }


def test_planner_parses_json_string_response():
    router = _FakeRouter(
        {
            "ticket.decompose": (
                '{"plan": "Edit files", "files_to_modify": ["src/app.py"]}'
            )
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert result["plan"] == "Edit files"
    assert result["files_to_modify"] == ["src/app.py"]
    assert result["risks"] == []
    assert result["complexity"] == "medium"
    assert result["requires_human_review"] is False


def test_planner_parses_fenced_json_response_and_fills_defaults():
    router = _FakeRouter(
        {
            "ticket.decompose": (
                "```json\n"
                '{"summary": "Add tests for retry behavior"}\n'
                "```"
            )
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert result == {
        "plan": "Add tests for retry behavior",
        "files_to_modify": [],
        "risks": [],
        "complexity": "medium",
        "requires_human_review": False,
    }


def test_planner_raises_model_service_error_on_invalid_json():
    router = _FakeRouter({"ticket.decompose": "not json"})

    with pytest.raises(ModelServiceError, match="could not be parsed"):
        asyncio.run(ModelRouterPlannerService(router).plan(_state()))


def test_review_calls_router_and_accepts_passed():
    router = _FakeRouter(
        {
            "code.verify": {
                "passed": True,
                "reasoning": "Tests passed and behavior matches the ticket.",
                "issues": [],
                "confidence": 0.82,
            }
        }
    )

    result = asyncio.run(ModelRouterReviewService(router).review(_state()))

    assert router.calls[0].capability == "code.verify"
    assert result == {
        "passed": True,
        "status": "approved",
        "reasoning": "Tests passed and behavior matches the ticket.",
        "issues": [],
        "confidence": 0.82,
    }


def test_review_accepts_approved_alias_and_fills_issues_default():
    router = _FakeRouter(
        {
            "code.verify": {
                "approved": True,
                "reasoning": "Implementation is sufficient.",
            }
        }
    )

    result = asyncio.run(ModelRouterReviewService(router).review(_state()))

    assert result["passed"] is True
    assert result["status"] == "approved"
    assert result["issues"] == []


def test_review_raises_model_service_error_when_response_is_invalid():
    router = _FakeRouter({"code.verify": ["not", "an", "object"]})

    with pytest.raises(ModelServiceError, match="unsupported shape"):
        asyncio.run(ModelRouterReviewService(router).review(_state()))


def test_implementation_calls_router_and_writes_files_through_adapter():
    adapter = _FakeFileAdapter()
    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Implemented pagination.",
                "files": [
                    {
                        "operation": "write_file",
                        "path": "src/api/users.py",
                        "content": "def list_users():\n    return []\n",
                    },
                    {
                        "operation": "write_file",
                        "path": "tests/test_users.py",
                        "content": "def test_users():\n    assert True\n",
                    },
                ],
                "notes": ["Added endpoint tests"],
            }
        }
    )
    factory = _AdapterFactory(adapter)

    result = asyncio.run(
        ModelRouterImplementationService(router, factory).implement(
            _state(worktree_path="/tmp/worktree")
        )
    )

    assert router.calls[0].capability == "code.implement"
    assert factory.calls == ["/tmp/worktree"]
    assert adapter.writes == [
        ("src/api/users.py", "def list_users():\n    return []\n"),
        ("tests/test_users.py", "def test_users():\n    assert True\n"),
    ]
    assert result == {
        "implementation_result": {
            "status": "implemented",
            "changed_files": ["src/api/users.py", "tests/test_users.py"],
            "summary": "Implemented pagination.",
            "notes": ["Added endpoint tests"],
        }
    }


def test_implementation_rejects_absolute_paths():
    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Unsafe",
                "files": [
                    {
                        "operation": "write_file",
                        "path": "/tmp/outside.py",
                        "content": "nope\n",
                    }
                ],
            }
        }
    )

    with pytest.raises(ModelServiceError, match="must be relative"):
        asyncio.run(
            ModelRouterImplementationService(router, _AdapterFactory()).implement(
                _state(worktree_path="/tmp/worktree")
            )
        )


def test_implementation_rejects_traversal_paths():
    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Unsafe",
                "files": [
                    {
                        "operation": "write_file",
                        "path": "src/../outside.py",
                        "content": "nope\n",
                    }
                ],
            }
        }
    )

    with pytest.raises(ModelServiceError, match="must not contain '\\.\\.'"):
        asyncio.run(
            ModelRouterImplementationService(router, _AdapterFactory()).implement(
                _state(worktree_path="/tmp/worktree")
            )
        )


def test_implementation_rejects_unsupported_operations():
    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Unsupported",
                "files": [
                    {
                        "operation": "delete_file",
                        "path": "src/api/users.py",
                    }
                ],
            }
        }
    )

    with pytest.raises(ModelServiceError, match="unsupported file operation"):
        asyncio.run(
            ModelRouterImplementationService(router, _AdapterFactory()).implement(
                _state(worktree_path="/tmp/worktree")
            )
        )


def test_implementation_rejects_missing_content():
    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Missing content",
                "files": [
                    {
                        "operation": "write_file",
                        "path": "src/api/users.py",
                    }
                ],
            }
        }
    )

    with pytest.raises(ModelServiceError, match="content is required"):
        asyncio.run(
            ModelRouterImplementationService(router, _AdapterFactory()).implement(
                _state(worktree_path="/tmp/worktree")
            )
        )


def test_implementation_propagates_file_adapter_errors():
    adapter = _FakeFileAdapter(error=RuntimeError("disk full"))
    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Write file",
                "files": [
                    {
                        "operation": "write_file",
                        "path": "src/api/users.py",
                        "content": "content\n",
                    }
                ],
            }
        }
    )

    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(
            ModelRouterImplementationService(
                router,
                _AdapterFactory(adapter),
            ).implement(_state(worktree_path="/tmp/worktree"))
        )


def test_service_backed_graph_completes_with_model_services():
    adapter = _FakeFileAdapter()
    router = _FakeRouter(
        {
            "ticket.decompose": {
                "plan": "Write the implementation file.",
                "files_to_modify": ["src/feature.py"],
            },
            "code.implement": {
                "summary": "Added feature.",
                "files": [
                    {
                        "operation": "write_file",
                        "path": "src/feature.py",
                        "content": "VALUE = 1\n",
                    }
                ],
            },
            "code.verify": {
                "approved": True,
                "reasoning": "Implementation and tests are acceptable.",
            },
        }
    )
    runner = TicketNodeRunner(
        planner=ModelRouterPlannerService(router),
        approval=_Approval(),
        implementation=ModelRouterImplementationService(
            router,
            _AdapterFactory(adapter),
        ),
        tests=_Tests(),
        review=ModelRouterReviewService(router),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )
    graph = build_ticket_graph(runner)

    result = asyncio.run(graph.ainvoke(_state(worktree_path="/tmp/worktree")))
    state = TicketState.model_validate(result)

    assert state.workflow_status == "completed"
    assert state.pull_request_url == "https://github.test/acme/repo/pull/1"
    assert state.review_passed is True
    assert state.implementation_result == {
        "status": "implemented",
        "changed_files": ["src/feature.py"],
        "summary": "Added feature.",
        "notes": [],
    }
    assert adapter.writes == [("src/feature.py", "VALUE = 1\n")]


def _state(**updates: Any) -> TicketState:
    values = {
        "ticket_key": "AGENT-123",
        "summary": "Add pagination",
        "description": "Users endpoint needs page and page_size support.",
        "repository": "agent-system",
        "repo_path": "/repos/agent-system",
        "max_attempts": 3,
    }
    values.update(updates)
    return TicketState(**values)


@dataclass(frozen=True)
class _RouterCall:
    capability: str
    messages: list[dict[str, str]]
    kwargs: dict[str, Any]


class _FakeRouter:
    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[_RouterCall] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        self.calls.append(_RouterCall(capability, messages, kwargs))
        return self._responses[capability]


class _FakeFileAdapter:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.writes: list[tuple[str, str]] = []

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        del encoding
        if self._error is not None:
            raise self._error
        self.writes.append((path, content))


class _AdapterFactory:
    def __init__(self, adapter: _FakeFileAdapter | None = None) -> None:
        self._adapter = adapter or _FakeFileAdapter()
        self.calls: list[str] = []

    def __call__(self, worktree_path: str) -> _FakeFileAdapter:
        self.calls.append(worktree_path)
        return self._adapter


class _Approval:
    async def request_approval(self, state: TicketState) -> bool:
        return True


class _Tests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed", "tests_passed": True}


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/acme/repo/pull/1"


class _Escalation:
    async def escalate(self, state: TicketState, reason: str) -> None:
        raise AssertionError(f"should not escalate: {reason}")

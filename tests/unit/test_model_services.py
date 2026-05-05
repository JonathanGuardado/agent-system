from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from ticket_agent.orchestrator.graph import build_ticket_graph
from ticket_agent.orchestrator.model_services import (
    IterativeImplementationService,
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


def test_planner_accepts_dict_envelope_with_content_json_string():
    router = _FakeRouter(
        {
            "ticket.decompose": {
                "content": json.dumps(
                    {
                        "plan": "Do it",
                        "files_to_modify": ["src/app.py"],
                        "risks": [],
                        "complexity": "medium",
                        "requires_human_review": False,
                    }
                ),
                "model": "gemini-pro",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert result == {
        "plan": "Do it",
        "files_to_modify": ["src/app.py"],
        "risks": [],
        "complexity": "medium",
        "requires_human_review": False,
    }


def test_planner_accepts_dict_envelope_with_content_dict_payload():
    router = _FakeRouter(
        {
            "ticket.decompose": {
                "content": {
                    "plan": "Use dict content",
                    "files_to_modify": ["src/app.py"],
                    "risks": [],
                    "complexity": "medium",
                    "requires_human_review": False,
                },
                "model": "gemini-pro",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert result["plan"] == "Use dict content"
    assert result["files_to_modify"] == ["src/app.py"]


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


def test_planner_accepts_dict_envelope_with_fenced_json_content():
    router = _FakeRouter(
        {
            "ticket.decompose": {
                "content": (
                    "```json\n"
                    '{"plan": "Do it", "files_to_modify": ["src/app.py"], '
                    '"risks": [], "complexity": "low", '
                    '"requires_human_review": false}\n'
                    "```"
                ),
                "model": "gemini-pro",
            }
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert result["plan"] == "Do it"
    assert result["files_to_modify"] == ["src/app.py"]
    assert result["complexity"] == "low"


def test_planner_raises_model_service_error_on_invalid_json():
    router = _FakeRouter({"ticket.decompose": "not json"})

    with pytest.raises(ModelServiceError, match="could not be parsed"):
        asyncio.run(ModelRouterPlannerService(router).plan(_state()))


def test_planner_raises_model_service_error_on_invalid_envelope_content():
    router = _FakeRouter(
        {
            "ticket.decompose": {
                "content": "not json",
                "model": "gemini-pro",
                "usage": {"input_tokens": 10},
            }
        }
    )

    with pytest.raises(ModelServiceError, match="envelope field 'content'"):
        asyncio.run(ModelRouterPlannerService(router).plan(_state()))


def test_planner_accepts_object_envelope_with_content():
    router = _FakeRouter(
        {
            "ticket.decompose": _ObjectEnvelope(
                content='{"plan": "Object content", "files_to_modify": []}'
            )
        }
    )

    result = asyncio.run(ModelRouterPlannerService(router).plan(_state()))

    assert result["plan"] == "Object content"


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


def test_review_accepts_dict_envelope_with_content_json_string():
    router = _FakeRouter(
        {
            "code.verify": {
                "content": json.dumps(
                    {
                        "passed": True,
                        "reasoning": "Looks good",
                        "issues": [],
                        "confidence": 0.8,
                    }
                ),
                "model": "gemini-pro",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        }
    )

    result = asyncio.run(ModelRouterReviewService(router).review(_state()))

    assert result == {
        "passed": True,
        "status": "approved",
        "reasoning": "Looks good",
        "issues": [],
        "confidence": 0.8,
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


def test_review_accepts_approved_alias_from_dict_envelope():
    router = _FakeRouter(
        {
            "code.verify": {
                "content": json.dumps(
                    {
                        "approved": True,
                        "reasoning": "Implementation is sufficient.",
                    }
                ),
                "model": "gemini-pro",
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


def test_review_raises_model_service_error_on_invalid_envelope_content():
    router = _FakeRouter(
        {
            "code.verify": {
                "content": "not json",
                "model": "gemini-pro",
            }
        }
    )

    with pytest.raises(ModelServiceError, match="envelope field 'content'"):
        asyncio.run(ModelRouterReviewService(router).review(_state()))


def test_implementation_calls_router_and_writes_files_through_adapter(tmp_path):
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
            _state(worktree_path=str(tmp_path))
        )
    )

    assert router.calls[0].capability == "code.implement"
    assert factory.calls == [str(tmp_path)]
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


def test_implementation_accepts_dict_envelope_and_writes_files_through_adapter():
    adapter = _FakeFileAdapter()
    router = _FakeRouter(
        {
            "code.implement": {
                "content": json.dumps(
                    {
                        "summary": "Updated files",
                        "operations": [
                            {
                                "type": "write_file",
                                "path": "src/hello.py",
                                "content": "def hello():\n    return 'hello'\n",
                            }
                        ],
                    }
                ),
                "model": "deepseek-v4-pro",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        }
    )
    factory = _AdapterFactory(adapter)

    result = asyncio.run(
        ModelRouterImplementationService(router, factory).implement(
            _state(worktree_path="/tmp/worktree")
        )
    )

    assert adapter.writes == [
        ("src/hello.py", "def hello():\n    return 'hello'\n")
    ]
    assert result["implementation_result"]["changed_files"] == ["src/hello.py"]
    assert result["implementation_result"]["summary"] == "Updated files"


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


def test_implementation_rejects_absolute_paths_from_envelope_payload():
    router = _FakeRouter(
        {
            "code.implement": {
                "content": json.dumps(
                    {
                        "summary": "Unsafe",
                        "operations": [
                            {
                                "type": "write_file",
                                "path": "/tmp/outside.py",
                                "content": "nope\n",
                            }
                        ],
                    }
                ),
                "model": "deepseek-v4-pro",
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


def test_implementation_rejects_traversal_paths_from_envelope_payload():
    router = _FakeRouter(
        {
            "code.implement": {
                "content": json.dumps(
                    {
                        "summary": "Unsafe",
                        "operations": [
                            {
                                "type": "write_file",
                                "path": "src/../outside.py",
                                "content": "nope\n",
                            }
                        ],
                    }
                ),
                "model": "deepseek-v4-pro",
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


def test_implementation_raises_model_service_error_on_invalid_envelope_content():
    router = _FakeRouter(
        {
            "code.implement": {
                "content": "not json",
                "model": "deepseek-v4-pro",
            }
        }
    )

    with pytest.raises(ModelServiceError, match="envelope field 'content'"):
        asyncio.run(
            ModelRouterImplementationService(router, _AdapterFactory()).implement(
                _state(worktree_path="/tmp/worktree")
            )
        )


def test_implementation_includes_repo_context_in_messages(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "users.py").write_text(
        "def list_users():\n    return []\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Touch README",
                "operations": [
                    {
                        "type": "write_file",
                        "path": "README.md",
                        "content": "# Project\n",
                    }
                ],
            }
        }
    )

    asyncio.run(
        ModelRouterImplementationService(
            router,
            _AdapterFactory(),
        ).implement(
            _state(
                worktree_path=str(tmp_path),
                decomposition={"files_to_modify": ["src/users.py"]},
            )
        )
    )

    user_message = router.calls[0].messages[1]["content"]
    assert "repo_context" in user_message
    assert "src/users.py" in user_message


def test_implementation_includes_failed_test_excerpt_on_retry(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "users.py").write_text("VALUE = 1\n", encoding="utf-8")

    router = _FakeRouter(
        {
            "code.implement": {
                "summary": "Retry",
                "operations": [
                    {
                        "type": "write_file",
                        "path": "src/users.py",
                        "content": "VALUE = 2\n",
                    }
                ],
            }
        }
    )

    asyncio.run(
        ModelRouterImplementationService(
            router,
            _AdapterFactory(),
        ).implement(
            _state(
                worktree_path=str(tmp_path),
                implementation_attempts=1,
                test_result={
                    "status": "failed",
                    "tests_passed": False,
                    "stdout": "FAILED tests/test_users.py::test_pagination",
                    "stderr": "AssertionError: pagination missing",
                },
            )
        )
    )

    user_message = router.calls[0].messages[1]["content"]
    assert "previous_test_failure" in user_message
    assert "AssertionError" in user_message
    assert "implementation_attempts: 1" in user_message


def test_iterative_implementation_reads_writes_and_finishes_through_adapter(tmp_path):
    adapter = _LoopFileAdapter(reads={"src/users.py": "VALUE = 1\n"})
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "read_file",
                    "args": {"path": "src/users.py"},
                },
                {
                    "action": "write_file",
                    "args": {
                        "path": "src/users.py",
                        "content": "VALUE = 2\n",
                    },
                },
                {
                    "action": "finish",
                    "args": {
                        "summary": "Updated user pagination value.",
                        "notes": ["Touched one file"],
                    },
                },
            ]
        }
    )
    factory = _AdapterFactory(adapter)

    result = asyncio.run(
        IterativeImplementationService(router, factory).implement(
            _state(worktree_path=str(tmp_path))
        )
    )

    assert factory.calls == [str(tmp_path)]
    assert adapter.reads == ["src/users.py"]
    assert adapter.writes == [("src/users.py", "VALUE = 2\n")]
    assert result == {
        "implementation_result": {
            "status": "success",
            "summary": "Updated user pagination value.",
            "changed_files": ["src/users.py"],
            "notes": ["Touched one file"],
            "errors": [],
        }
    }
    assert any(
        "tool_result" in message["content"]
        for message in router.calls[1].messages
    )


def test_iterative_implementation_lists_directory_through_adapter(tmp_path):
    adapter = _LoopFileAdapter(files=("src/users.py", "src/orders.py"))
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "list_dir", "args": {"path": "src"}},
                {
                    "action": "finish",
                    "args": {"summary": "Inspected source files."},
                },
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(
            router,
            _AdapterFactory(adapter),
        ).implement(_state(worktree_path=str(tmp_path)))
    )

    assert adapter.lists == ["src"]
    assert result["implementation_result"]["status"] == "success"
    assert any(
        "src/users.py" in message["content"]
        for message in router.calls[1].messages
    )


def test_iterative_implementation_returns_failed_result_for_path_escape(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "write_file",
                    "args": {"path": "../outside.py", "content": "nope\n"},
                }
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(router).implement(
            _state(worktree_path=str(tmp_path))
        )
    )

    implementation_result = result["implementation_result"]
    assert implementation_result["status"] == "failed"
    assert implementation_result["changed_files"] == []
    assert implementation_result["error_code"] == "tool_execution_failed"
    assert "outside" in implementation_result["error"]


def test_iterative_implementation_returns_failed_result_for_protected_write(
    tmp_path,
):
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "write_file",
                    "args": {
                        "path": ".github/workflows/ci.yml",
                        "content": "name: ci\n",
                    },
                }
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(router).implement(
            _state(worktree_path=str(tmp_path))
        )
    )

    implementation_result = result["implementation_result"]
    assert implementation_result["status"] == "failed"
    assert implementation_result["changed_files"] == []
    assert implementation_result["error_code"] == "tool_execution_failed"
    assert "policy violation" in implementation_result["error"]


def test_iterative_implementation_malformed_json_fails_cleanly(tmp_path):
    router = _SequenceRouter({"code.implement": ["not json"]})

    result = asyncio.run(
        IterativeImplementationService(router).implement(
            _state(worktree_path=str(tmp_path))
        )
    )

    implementation_result = result["implementation_result"]
    assert implementation_result["status"] == "failed"
    assert implementation_result["error_code"] == "invalid_tool_call"
    assert "could not be parsed" in implementation_result["error"]


def test_iterative_implementation_unknown_action_fails_cleanly(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "delete_file",
                    "args": {"path": "src/users.py"},
                }
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(router).implement(
            _state(worktree_path=str(tmp_path))
        )
    )

    implementation_result = result["implementation_result"]
    assert implementation_result["status"] == "failed"
    assert implementation_result["error_code"] == "unknown_action"
    assert "delete_file" in implementation_result["error"]


def test_iterative_implementation_max_turns_exhausted_fails_without_hanging(
    tmp_path,
):
    adapter = _LoopFileAdapter(reads={"src/users.py": "VALUE = 1\n"})
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "read_file", "args": {"path": "src/users.py"}},
                {"action": "read_file", "args": {"path": "src/users.py"}},
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(
            router,
            _AdapterFactory(adapter),
            max_turns=2,
        ).implement(_state(worktree_path=str(tmp_path)))
    )

    implementation_result = result["implementation_result"]
    assert len(router.calls) == 2
    assert implementation_result["status"] == "failed"
    assert implementation_result["error_code"] == "max_turns_exhausted"


def test_iterative_implementation_includes_failed_test_excerpt_on_retry(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "finish",
                    "args": {"summary": "No more edits needed."},
                }
            ]
        }
    )

    asyncio.run(
        IterativeImplementationService(
            router,
            _AdapterFactory(_LoopFileAdapter()),
        ).implement(
            _state(
                worktree_path=str(tmp_path),
                implementation_attempts=1,
                test_result={
                    "status": "failed",
                    "tests_passed": False,
                    "stdout": "FAILED tests/test_users.py::test_pagination",
                    "stderr": "AssertionError: page_size ignored",
                },
            )
        )
    )

    user_message = router.calls[0].messages[1]["content"]
    assert "previous_test_failure" in user_message
    assert "AssertionError: page_size ignored" in user_message
    assert "current_implementation_attempt: 2" in user_message


def test_iterative_graph_retry_uses_failed_test_excerpt_then_opens_pr(tmp_path):
    adapter = _LoopFileAdapter()
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "write_file",
                    "args": {"path": "src/feature.py", "content": "VALUE = 1\n"},
                },
                {"action": "finish", "args": {"summary": "First attempt."}},
                {
                    "action": "write_file",
                    "args": {"path": "src/feature.py", "content": "VALUE = 2\n"},
                },
                {"action": "finish", "args": {"summary": "Retry fixed tests."}},
            ],
            "code.verify": {"approved": True, "reasoning": "Looks good."},
        }
    )
    runner = TicketNodeRunner(
        planner=_Planner(),
        approval=_Approval(),
        implementation=IterativeImplementationService(
            router,
            _AdapterFactory(adapter),
        ),
        tests=_SequencedTests([False, True]),
        review=ModelRouterReviewService(router),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )
    graph = build_ticket_graph(runner)

    result = asyncio.run(graph.ainvoke(_state(worktree_path=str(tmp_path))))
    state = TicketState.model_validate(result)

    assert state.workflow_status == "completed"
    assert state.pull_request_url == "https://github.test/acme/repo/pull/1"
    assert state.visited_nodes == [
        "plan",
        "request_execution_approval",
        "implement",
        "run_tests",
        "implement",
        "run_tests",
        "review",
        "open_pull_request",
        "report",
    ]
    assert adapter.writes == [
        ("src/feature.py", "VALUE = 1\n"),
        ("src/feature.py", "VALUE = 2\n"),
    ]
    assert "previous_test_failure" in router.calls[2].messages[1]["content"]
    assert "AssertionError: still broken" in router.calls[2].messages[1]["content"]


def test_iterative_graph_retry_exhaustion_escalates(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "finish", "args": {"summary": "Attempt one."}},
                {"action": "finish", "args": {"summary": "Attempt two."}},
            ]
        }
    )
    escalation = _CapturingEscalation()
    runner = TicketNodeRunner(
        planner=_Planner(),
        approval=_Approval(),
        implementation=IterativeImplementationService(
            router,
            _AdapterFactory(_LoopFileAdapter()),
        ),
        tests=_SequencedTests([False, False]),
        review=ModelRouterReviewService(router),
        pull_request=_PullRequest(),
        escalation=escalation,
    )
    graph = build_ticket_graph(runner)

    result = asyncio.run(
        graph.ainvoke(_state(worktree_path=str(tmp_path), max_attempts=2))
    )
    state = TicketState.model_validate(result)

    assert state.workflow_status == "escalated"
    assert state.implementation_attempts == 2
    assert escalation.calls == [("AGENT-123", "tests failed")]


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
class _ObjectEnvelope:
    content: Any


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


class _SequenceRouter:
    def __init__(self, responses: dict[str, list[Any] | Any]) -> None:
        self._responses = {
            capability: list(response) if isinstance(response, list) else response
            for capability, response in responses.items()
        }
        self.calls: list[_RouterCall] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        self.calls.append(_RouterCall(capability, messages, kwargs))
        response = self._responses[capability]
        if isinstance(response, list):
            if not response:
                raise AssertionError(f"no response left for {capability}")
            return response.pop(0)
        return response


class _FakeFileAdapter:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.writes: list[tuple[str, str]] = []

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        del encoding
        if self._error is not None:
            raise self._error
        self.writes.append((path, content))


class _LoopFileAdapter:
    def __init__(
        self,
        *,
        reads: dict[str, str] | None = None,
        files: tuple[str, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self._read_values = reads or {}
        self._files = files
        self._error = error
        self.reads: list[str] = []
        self.lists: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        del encoding
        if self._error is not None:
            raise self._error
        self.reads.append(path)
        return self._read_values.get(path, "")

    def list_files(self, path: str = ".") -> tuple[str, ...]:
        if self._error is not None:
            raise self._error
        self.lists.append(path)
        return self._files

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        del encoding
        if self._error is not None:
            raise self._error
        self.writes.append((path, content))


class _AdapterFactory:
    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter or _FakeFileAdapter()
        self.calls: list[str] = []

    def __call__(self, worktree_path: str) -> Any:
        self.calls.append(worktree_path)
        return self._adapter


class _Planner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {"plan": "Edit the requested files."}


class _Approval:
    async def request_approval(self, state: TicketState) -> bool:
        return True


class _Tests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed", "tests_passed": True}


class _SequencedTests:
    def __init__(self, results: list[bool]) -> None:
        self._results = results

    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        passed = self._results.pop(0)
        return {
            "status": "passed" if passed else "failed",
            "tests_passed": passed,
            "stdout": "ok" if passed else "FAILED tests/test_feature.py::test_feature",
            "stderr": "" if passed else "AssertionError: still broken",
        }


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/acme/repo/pull/1"


class _Escalation:
    async def escalate(self, state: TicketState, reason: str) -> None:
        raise AssertionError(f"should not escalate: {reason}")


class _CapturingEscalation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def escalate(self, state: TicketState, reason: str) -> None:
        self.calls.append((state.ticket_key, reason))

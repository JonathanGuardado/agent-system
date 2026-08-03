"""P6 implementation-loop upgrade: search, edit_file, paginated read, run_tests.

These tests exercise the provider-agnostic tool-call loop in
``IterativeImplementationService`` through the real ``LocalFileAdapter`` (for
filesystem-touching actions) and a prepared ``ImplementationContext`` (for the
model-callable ``run_tests`` action). They cover the new capabilities and the
truncated-write guard that blocks blind overwrites of unseen files.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.config.repo_contract import (
    CommandSpec,
    ExecutionPolicy,
    LanguageInfo,
    RepoCommands,
    RepoContract,
    RepoInfo,
)
from ticket_agent.orchestrator.local_services import ImplementationContext
from ticket_agent.orchestrator.model_services import IterativeImplementationService
from ticket_agent.orchestrator.state import TicketState

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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


def _repo_contract(repo_root: str) -> RepoContract:
    return RepoContract(
        repo=RepoInfo(name="agent-system", root=repo_root, default_branch="main"),
        language=LanguageInfo(primary="python", package_manager="pip"),
        commands=RepoCommands(
            test=CommandSpec(
                command=("pytest", "-q"),
                timeout_seconds=120,
                working_directory=".",
            ),
            lint=None,
            install=None,
        ),
        policy=ExecutionPolicy(
            dependency_install_allowed=False,
            config_paths_allowed=("pyproject.toml",),
            protected_paths=(".env",),
        ),
        source_dirs=("src/",),
        test_dirs=("tests/",),
    )


class _SequenceRouter:
    """Return a scripted sequence of tool-call payloads per capability."""

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses = {
            capability: list(payloads) for capability, payloads in responses.items()
        }
        self.calls: list[_RouterCall] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        self.calls.append(_RouterCall(capability, messages, kwargs))
        payloads = self._responses[capability]
        if not payloads:
            raise AssertionError(f"no response left for {capability}")
        return payloads.pop(0)


class _RouterCall:
    def __init__(
        self,
        capability: str,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
    ) -> None:
        self.capability = capability
        self.messages = messages
        self.kwargs = kwargs


class _InertFileAdapter:
    """File adapter that is never touched (run_tests-only scenarios)."""

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        raise AssertionError("read_text should not be called")

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        raise AssertionError("write_text should not be called")

    def list_files(self, path: str = ".") -> tuple[str, ...]:
        raise AssertionError("list_files should not be called")


def _run_implement(router: _SequenceRouter, tmp_path, **service_kwargs) -> dict[str, Any]:
    service = IterativeImplementationService(router, **service_kwargs)
    return asyncio.run(service.implement(_state(worktree_path=str(tmp_path))))


def _context(tmp_path, *, test_runner, files: Any | None = None) -> ImplementationContext:
    return ImplementationContext(
        state=_state(),
        contract=_repo_contract(str(tmp_path)),
        repo_path=tmp_path,
        worktree_path=tmp_path,
        branch_name="agent/AGENT-123/lock",
        lock_id="lock",
        files=files or _InertFileAdapter(),
        test_runner=test_runner,
    )


def _last_tool_result(router: _SequenceRouter, call_index: int) -> str:
    return router.calls[call_index].messages[-1]["content"]


# ---------------------------------------------------------------------------
# Paginated read_file
# ---------------------------------------------------------------------------


def test_read_file_paginated_returns_line_window(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "big.py").write_text(
        "".join(f"line{n}\n" for n in range(1, 11))
    )
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "read_file",
                    "args": {"path": "src/big.py", "offset": 3, "limit": 2},
                },
                {"action": "finish", "args": {"summary": "Inspected window."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["status"] == "success"
    tool_result = _last_tool_result(router, 1)
    assert '"start_line": 3' in tool_result
    assert '"end_line": 4' in tool_result
    assert '"total_lines": 10' in tool_result
    assert '"complete": false' in tool_result
    assert "line3" in tool_result and "line4" in tool_result
    assert "line5" not in tool_result


def test_read_file_full_marks_complete_true(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "small.py").write_text("VALUE = 1\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "read_file", "args": {"path": "src/small.py"}},
                {"action": "finish", "args": {"summary": "Read whole file."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["status"] == "success"
    tool_result = _last_tool_result(router, 1)
    assert '"complete": true' in tool_result
    assert '"truncated": false' in tool_result


def test_read_file_truncated_marks_complete_false(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "huge.py").write_text("x = 1  # padding padding\n" * 40)
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "read_file", "args": {"path": "src/huge.py"}},
                {"action": "finish", "args": {"summary": "Partial read."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path, tool_result_max_chars=256)

    assert result["implementation_result"]["status"] == "success"
    tool_result = _last_tool_result(router, 1)
    assert '"complete": false' in tool_result
    assert '"truncated": true' in tool_result


def test_read_file_offset_must_be_positive_integer(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "read_file", "args": {"path": "src/x.py", "offset": 0}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    impl = result["implementation_result"]
    assert impl["status"] == "failed"
    assert impl["error_code"] == "invalid_tool_call"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_finds_matches_and_skips_binary(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("# TODO fix pagination\nvalue = 1\n")
    (tmp_path / "src" / "b.py").write_text("clean = True\n")
    (tmp_path / "src" / "blob.dat").write_bytes(b"TODO\x00binary blob\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "search", "args": {"pattern": "TODO"}},
                {"action": "finish", "args": {"summary": "Searched."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["status"] == "success"
    tool_result = _last_tool_result(router, 1)
    assert "src/a.py:1:# TODO fix pagination" in tool_result
    assert '"match_count": 1' in tool_result
    assert "blob.dat" not in tool_result


def test_search_respects_max_results_cap(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "many.py").write_text("MATCH\n" * 10)
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "search", "args": {"pattern": "MATCH", "max_results": 2}},
                {"action": "finish", "args": {"summary": "Capped."}},
            ]
        }
    )

    _run_implement(router, tmp_path)

    tool_result = _last_tool_result(router, 1)
    assert '"match_count": 2' in tool_result
    assert '"truncated": true' in tool_result


def test_search_invalid_pattern_is_non_fatal(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "search", "args": {"pattern": "[unterminated"}},
                {"action": "finish", "args": {"summary": "Recovered."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["status"] == "success"
    tool_result = _last_tool_result(router, 1)
    assert '"error_code": "invalid_pattern"' in tool_result


def test_search_boundary_escape_is_fatal(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "search", "args": {"pattern": "x", "path": "../"}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    impl = result["implementation_result"]
    assert impl["status"] == "failed"
    assert impl["error_code"] == "path_boundary_violation"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_replaces_unique_occurrence(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "c.py").write_text("value = 1\nother = 2\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "edit_file",
                    "args": {
                        "path": "src/c.py",
                        "old_string": "value = 1",
                        "new_string": "value = 42",
                    },
                },
                {"action": "finish", "args": {"summary": "Edited."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    impl = result["implementation_result"]
    assert impl["status"] == "success"
    assert impl["changed_files"] == ["src/c.py"]
    assert (tmp_path / "src" / "c.py").read_text() == "value = 42\nother = 2\n"


def test_edit_file_missing_target_string_is_non_fatal(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "c.py").write_text("value = 1\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "edit_file",
                    "args": {
                        "path": "src/c.py",
                        "old_string": "nonexistent",
                        "new_string": "x",
                    },
                },
                {"action": "finish", "args": {"summary": "Gave up on edit."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["changed_files"] == []
    tool_result = _last_tool_result(router, 1)
    assert '"error_code": "edit_target_not_found"' in tool_result


def test_edit_file_missing_file_reports_not_found(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "edit_file",
                    "args": {"path": "src/absent.py", "old_string": "a", "new_string": "b"},
                },
                {"action": "finish", "args": {"summary": "No file."}},
            ]
        }
    )

    _run_implement(router, tmp_path)

    tool_result = _last_tool_result(router, 1)
    assert '"error_code": "edit_target_not_found"' in tool_result


def test_edit_file_ambiguous_without_replace_all(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "d.py").write_text("x = 1\nx = 1\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "edit_file",
                    "args": {"path": "src/d.py", "old_string": "x = 1", "new_string": "x = 2"},
                },
                {"action": "finish", "args": {"summary": "Ambiguous."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["changed_files"] == []
    tool_result = _last_tool_result(router, 1)
    assert '"error_code": "edit_target_ambiguous"' in tool_result
    assert (tmp_path / "src" / "d.py").read_text() == "x = 1\nx = 1\n"


def test_edit_file_replace_all_replaces_every_occurrence(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "d.py").write_text("x = 1\nx = 1\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "edit_file",
                    "args": {
                        "path": "src/d.py",
                        "old_string": "x = 1",
                        "new_string": "x = 2",
                        "replace_all": True,
                    },
                },
                {"action": "finish", "args": {"summary": "Replaced all."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["changed_files"] == ["src/d.py"]
    assert (tmp_path / "src" / "d.py").read_text() == "x = 2\nx = 2\n"
    tool_result = _last_tool_result(router, 1)
    assert '"replacements": 2' in tool_result


# ---------------------------------------------------------------------------
# Truncated-write guard
# ---------------------------------------------------------------------------


def test_write_file_rejected_without_complete_view(tmp_path):
    (tmp_path / "src").mkdir()
    original = "line\n" * 100
    (tmp_path / "src" / "e.py").write_text(original)
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "write_file",
                    "args": {"path": "src/e.py", "content": "clobbered\n"},
                },
                {"action": "finish", "args": {"summary": "Tried blind write."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path, tool_result_max_chars=256)

    assert result["implementation_result"]["changed_files"] == []
    assert (tmp_path / "src" / "e.py").read_text() == original
    tool_result = _last_tool_result(router, 1)
    assert '"error_code": "truncated_write_rejected"' in tool_result


def test_write_file_allowed_after_complete_read(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "f.py").write_text("old = 1\n")
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "read_file", "args": {"path": "src/f.py"}},
                {
                    "action": "write_file",
                    "args": {"path": "src/f.py", "content": "new = 2\n"},
                },
                {"action": "finish", "args": {"summary": "Rewrote after read."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["changed_files"] == ["src/f.py"]
    assert (tmp_path / "src" / "f.py").read_text() == "new = 2\n"


def test_write_file_allowed_for_new_file(tmp_path):
    (tmp_path / "src").mkdir()
    router = _SequenceRouter(
        {
            "code.implement": [
                {
                    "action": "write_file",
                    "args": {"path": "src/new.py", "content": "created = 1\n"},
                },
                {"action": "finish", "args": {"summary": "Created file."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["changed_files"] == ["src/new.py"]
    assert (tmp_path / "src" / "new.py").read_text() == "created = 1\n"


# ---------------------------------------------------------------------------
# run_tests action
# ---------------------------------------------------------------------------


def test_run_tests_unavailable_without_runner(tmp_path):
    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "run_tests", "args": {}},
                {"action": "finish", "args": {"summary": "No tests available."}},
            ]
        }
    )

    result = _run_implement(router, tmp_path)

    assert result["implementation_result"]["status"] == "success"
    tool_result = _last_tool_result(router, 1)
    assert '"error_code": "tests_unavailable"' in tool_result


def test_run_tests_via_context_runner_returns_result(tmp_path):
    calls: list[int] = []

    def runner() -> dict[str, Any]:
        calls.append(1)
        return {"tests_passed": True, "status": "passed", "output": "3 passed"}

    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "run_tests", "args": {}},
                {"action": "finish", "args": {"summary": "Tests green."}},
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(router).implement_context(
            _context(tmp_path, test_runner=runner)
        )
    )

    assert result["status"] == "success"
    assert calls == [1]
    tool_result = _last_tool_result(router, 1)
    assert '"tests_passed": true' in tool_result
    assert "3 passed" in tool_result


def test_run_tests_budget_exhausted(tmp_path):
    def runner() -> dict[str, Any]:
        return {"tests_passed": False, "status": "failed", "output": "1 failed"}

    router = _SequenceRouter(
        {
            "code.implement": [
                {"action": "run_tests", "args": {}},
                {"action": "run_tests", "args": {}},
                {"action": "finish", "args": {"summary": "Out of runs."}},
            ]
        }
    )

    result = asyncio.run(
        IterativeImplementationService(router, max_test_runs=1).implement_context(
            _context(tmp_path, test_runner=runner)
        )
    )

    assert result["status"] == "success"
    second_tool_result = _last_tool_result(router, 2)
    assert '"error_code": "test_budget_exhausted"' in second_tool_result


def test_run_tests_offered_in_prompt_only_when_available(tmp_path):
    router = _SequenceRouter(
        {"code.implement": [{"action": "finish", "args": {"summary": "done"}}]}
    )

    asyncio.run(
        IterativeImplementationService(router).implement_context(
            _context(tmp_path, test_runner=lambda: {"tests_passed": True})
        )
    )

    prompt = router.calls[0].messages[1]["content"]
    assert '"action": "run_tests"' in prompt
    assert "Call run_tests to run the repo-contract test command" in prompt


def test_run_tests_not_offered_in_prompt_without_runner(tmp_path):
    router = _SequenceRouter(
        {"code.implement": [{"action": "finish", "args": {"summary": "done"}}]}
    )

    _run_implement(router, tmp_path)

    prompt = router.calls[0].messages[1]["content"]
    assert "You cannot run tests in this run" in prompt
    assert '"action": "run_tests"' not in prompt


# ---------------------------------------------------------------------------
# No shell / git actions are exposed to the model
# ---------------------------------------------------------------------------


def test_shell_and_git_actions_are_unknown_actions(tmp_path):
    for action in ("run_shell", "git_commit", "exec", "bash"):
        router = _SequenceRouter(
            {"code.implement": [{"action": action, "args": {"cmd": "rm -rf /"}}]}
        )

        result = _run_implement(router, tmp_path)

        impl = result["implementation_result"]
        assert impl["status"] == "failed", action
        assert impl["error_code"] == "unknown_action", action

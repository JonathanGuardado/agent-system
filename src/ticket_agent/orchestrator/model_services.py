"""ModelRouter-backed services for ticket workflow orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.domain.acceptance import parse_acceptance_criteria
from ticket_agent.domain.errors import (
    AgentSystemError,
    PathBoundaryError,
    PolicyViolationError,
)
from ticket_agent.observability.transcripts import (
    NullTranscriptRecorder,
    TranscriptEvent,
    TranscriptRecorder,
    safe_record,
    safe_tool_args,
)
from ticket_agent.orchestrator.repo_context import (
    RepoContext,
    RepoContextBuilder,
)
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.redaction import redact_local_paths


class ModelServiceError(RuntimeError):
    """Raised when a model-backed service receives an unusable response."""


class ModelRouterProtocol(Protocol):
    async def invoke(
        self,
        capability: str,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> Any: ...


FileAdapterFactory = Callable[[str], Any]
TestRunner = Callable[[], Mapping[str, Any]]
ToolAction = Literal[
    "read_file",
    "write_file",
    "list_dir",
    "search",
    "edit_file",
    "run_tests",
    "finish",
]

_MODEL_ENVELOPE_FIELDS = ("content", "text", "message", "data")
_MODEL_ENVELOPE_METADATA_FIELDS = frozenset(
    {
        "model",
        "usage",
        "provider",
        "raw",
        "response_id",
        "created_at",
        "id",
        "finish_reason",
    }
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TOOL_ACTIONS = frozenset(
    {
        "read_file",
        "write_file",
        "list_dir",
        "search",
        "edit_file",
        "run_tests",
        "finish",
    }
)
_DEFAULT_MAX_TOOL_RESULT_CHARS = 6000
_DEFAULT_MAX_IMPLEMENTATION_TURNS = 80
_DEFAULT_MAX_TEST_RUNS = 5
_DEFAULT_SEARCH_RESULT_CAP = 50
_MAX_SEARCH_RESULT_CAP = 200
_MAX_SEARCH_LINE_CHARS = 200
_MAX_FAILURE_EXCERPT_CHARS = 6000


class ToolCallValidationError(ValueError):
    """Raised when a model response is not a valid implementation tool call."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolCall(BaseModel):
    """Provider-agnostic JSON tool call used by iterative implementation."""

    model_config = ConfigDict(extra="forbid")

    action: ToolAction
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_args_for_action(self) -> ToolCall:
        if self.action not in _TOOL_ACTIONS:
            raise ValueError(f"unknown action: {self.action!r}")

        if self.action in {"read_file", "write_file", "list_dir", "edit_file"}:
            _tool_arg_string(self.args, "path", self.action)
        if self.action == "read_file":
            _optional_tool_arg_int(self.args, "offset", self.action, minimum=1)
            _optional_tool_arg_int(self.args, "limit", self.action, minimum=1)
        if self.action == "write_file":
            _tool_arg_string(
                self.args,
                "content",
                self.action,
                allow_empty=True,
            )
        if self.action == "search":
            _tool_arg_string(self.args, "pattern", self.action)
            if "path" in self.args:
                _tool_arg_string(self.args, "path", self.action)
            _optional_tool_arg_int(
                self.args, "max_results", self.action, minimum=1
            )
        if self.action == "edit_file":
            _tool_arg_string(self.args, "old_string", self.action)
            _tool_arg_string(
                self.args,
                "new_string",
                self.action,
                allow_empty=True,
            )
            if "replace_all" in self.args:
                _optional_tool_arg_bool(self.args, "replace_all", self.action)
        if self.action == "finish":
            _tool_arg_string(self.args, "summary", self.action)
            notes = self.args.get("notes")
            if notes is not None:
                _tool_arg_string_list(notes, "notes", self.action)
        return self


class ModelRouterPlannerService:
    """Plan ticket execution through the internal model router."""

    def __init__(self, model_router: ModelRouterProtocol) -> None:
        self._model_router = model_router

    async def plan(self, state: TicketState) -> dict[str, Any]:
        response = await self._model_router.invoke(
            capability="ticket.decompose",
            messages=_messages_for_model(_planning_messages(state), state),
            ticket_id=state.ticket_key,
            metadata={
                "workflow_node": "plan",
                "goal_id": state.goal_id,
                "goal_iteration": state.implementation_attempts,
            },
        )
        payload = _coerce_model_payload(response)
        plan = _required_string(
            payload.get("plan", payload.get("summary")),
            "plan or summary",
        )

        decomposition: dict[str, Any] = {
            "plan": plan,
            "files_to_modify": _optional_string_list(
                payload.get("files_to_modify"),
                "files_to_modify",
            ),
            "risks": _optional_string_list(payload.get("risks"), "risks"),
            "complexity": _optional_string(payload.get("complexity"), "medium"),
            "requires_human_review": _optional_bool(
                payload.get("requires_human_review"),
                False,
                "requires_human_review",
            ),
        }
        criteria = parse_acceptance_criteria(state.description)
        if criteria:
            decomposition["acceptance_criteria"] = criteria
            coverage = payload.get("criteria_coverage")
            if isinstance(coverage, Mapping):
                decomposition["criteria_coverage"] = {
                    str(key): _optional_string(value, "")
                    for key, value in coverage.items()
                }
        return decomposition


class ModelRouterImplementationService:
    """Apply explicit model-generated file write plans through FileAdapter."""

    def __init__(
        self,
        model_router: ModelRouterProtocol,
        file_adapter_factory: FileAdapterFactory | None = None,
        repo_context_builder: RepoContextBuilder | None = None,
    ) -> None:
        self._model_router = model_router
        self._file_adapter_factory = file_adapter_factory or LocalFileAdapter
        self._repo_context_builder = repo_context_builder or RepoContextBuilder()

    async def implement(self, state: TicketState) -> dict[str, Any]:
        if not state.worktree_path:
            raise ModelServiceError("worktree_path is required to apply file writes")

        repo_context = self._repo_context_builder.build(state)
        response = await self._model_router.invoke(
            capability="code.implement",
            messages=_messages_for_model(
                _implementation_messages(state, repo_context),
                state,
            ),
            ticket_id=state.ticket_key,
            metadata={
                "workflow_node": "implement",
                "goal_id": state.goal_id,
                "goal_iteration": state.implementation_attempts + 1,
            },
        )
        payload = _coerce_model_payload(response)
        file_operations = _required_file_operations(payload)

        files = self._file_adapter_factory(state.worktree_path)
        changed_files: list[str] = []
        for index, operation in enumerate(file_operations):
            changed_files.append(_apply_file_operation(files, operation, index))

        result: dict[str, Any] = {
            "status": "implemented",
            "changed_files": changed_files,
            "summary": _optional_string(payload.get("summary"), ""),
            "notes": _optional_string_list(payload.get("notes"), "notes"),
        }
        return {"implementation_result": result}


@dataclass
class _LoopContext:
    """Mutable per-run state threaded through implementation tool calls."""

    files: Any
    test_runner: TestRunner | None
    max_test_runs: int
    changed_files: list[str] = field(default_factory=list)
    # Paths for which the model has seen a complete, untruncated view (or that
    # did not exist, so there is nothing to destroy). A write to any other
    # existing path is a blind overwrite and is rejected.
    complete_view_paths: set[str] = field(default_factory=set)
    test_runs: int = 0


class IterativeImplementationService:
    """Run a provider-agnostic JSON tool-call implementation loop."""

    def __init__(
        self,
        model_router: ModelRouterProtocol,
        file_adapter_factory: FileAdapterFactory | None = None,
        repo_context_builder: RepoContextBuilder | None = None,
        *,
        max_turns: int = _DEFAULT_MAX_IMPLEMENTATION_TURNS,
        tool_result_max_chars: int = _DEFAULT_MAX_TOOL_RESULT_CHARS,
        max_test_runs: int = _DEFAULT_MAX_TEST_RUNS,
        transcripts: TranscriptRecorder | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if tool_result_max_chars < 256:
            raise ValueError("tool_result_max_chars must be at least 256")
        if max_test_runs < 1:
            raise ValueError("max_test_runs must be at least 1")

        self._model_router = model_router
        self._file_adapter_factory = file_adapter_factory or LocalFileAdapter
        self._repo_context_builder = repo_context_builder or RepoContextBuilder()
        self._max_turns = max_turns
        self._tool_result_max_chars = tool_result_max_chars
        self._max_test_runs = max_test_runs
        self._transcripts = transcripts or NullTranscriptRecorder()

    def _record_loop(
        self,
        state: TicketState,
        name: str,
        payload: dict[str, Any],
    ) -> None:
        safe_record(
            self._transcripts,
            TranscriptEvent(
                ticket_key=state.ticket_key,
                kind="loop",
                name=name,
                payload=payload,
                run_id=state.lock_id,
                goal_id=state.goal_id,
                phase="implementing",
            ),
        )

    def _record_loop_end(
        self,
        state: TicketState,
        result: dict[str, Any],
        *,
        turns_used: int,
    ) -> dict[str, Any]:
        """Record why the loop stopped, then return the result unchanged.

        Wrapping the return keeps every exit path instrumented without adding
        a branch at each of the six ``return`` statements -- and an exit that
        is added later but not wrapped shows up as a missing ``loop.end``
        rather than as silence.
        """

        self._record_loop(
            state,
            "loop.end",
            {
                "status": result.get("status"),
                "error_code": result.get("error_code"),
                "turns_used": turns_used,
                "changed_file_count": len(result.get("changed_files") or ()),
            },
        )
        return result

    async def implement(self, state: TicketState) -> dict[str, Any]:
        if not state.worktree_path:
            return {
                "implementation_result": _failed_implementation_result(
                    "Implementation could not start.",
                    ["worktree_path is required to apply file writes"],
                    changed_files=[],
                )
            }

        files = self._file_adapter_factory(state.worktree_path)
        result = await self._run_loop(state, files)
        return {"implementation_result": result}

    async def implement_context(self, context: Any) -> dict[str, Any]:
        """Run against a prepared LocalImplementationService context."""

        state = context.state.model_copy(
            update={
                "repo_path": str(context.repo_path),
                "worktree_path": str(context.worktree_path),
            }
        )
        return await self._run_loop(
            state,
            context.files,
            repo_contract=context.contract,
            test_runner=getattr(context, "test_runner", None),
        )

    async def _run_loop(
        self,
        state: TicketState,
        files: Any,
        *,
        repo_contract: Any | None = None,
        test_runner: TestRunner | None = None,
    ) -> dict[str, Any]:
        context_builder = (
            self._repo_context_builder
            if repo_contract is None
            else RepoContextBuilder(repo_contract=repo_contract)
        )
        repo_context = context_builder.build(state)
        messages = _implementation_loop_messages(
            state,
            repo_context,
            tests_available=test_runner is not None,
        )
        loop = _LoopContext(
            files=files,
            test_runner=test_runner,
            max_test_runs=self._max_test_runs,
        )
        last_failed_tool_result: dict[str, Any] | None = None

        self._record_loop(
            state,
            "loop.start",
            {
                "max_turns": self._max_turns,
                "max_test_runs": self._max_test_runs,
                "tests_available": test_runner is not None,
                "message_count": len(messages),
            },
        )
        turns_used = 0

        for turn_index in range(self._max_turns):
            turns_used = turn_index + 1
            prompt_messages = _messages_for_model(messages, state)
            self._record_loop(
                state,
                f"turn.{turns_used}.call",
                {
                    "message_count": len(prompt_messages),
                    "total_chars": sum(
                        len(str(m.get("content", ""))) for m in prompt_messages
                    ),
                },
            )
            try:
                response = await self._model_router.invoke(
                    capability="code.implement",
                    messages=prompt_messages,
                    ticket_id=state.ticket_key,
                    metadata={
                        "workflow_node": "implement",
                        "implementation_turn": turn_index + 1,
                        "goal_id": state.goal_id,
                        "goal_iteration": turn_index + 1,
                    },
                )
                payload = _coerce_model_payload(response)
                tool_call = _tool_call_from_payload(payload)
            except (ModelServiceError, ToolCallValidationError) as exc:
                return self._record_loop_end(
                    state,
                    _failed_implementation_result(
                        "Implementation stopped because the model returned an invalid "
                        "tool call.",
                        [str(exc)],
                        changed_files=loop.changed_files,
                        code=getattr(exc, "code", "invalid_tool_call"),
                    ),
                    turns_used=turns_used,
                )
            except Exception as exc:
                return self._record_loop_end(
                    state,
                    _failed_implementation_result(
                        "Implementation stopped because the model call failed.",
                        [f"{type(exc).__name__}: {exc}"],
                        changed_files=loop.changed_files,
                        code="model_invoke_failed",
                    ),
                    turns_used=turns_used,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": _json_for_prompt(tool_call.model_dump()),
                }
            )

            if tool_call.action == "finish":
                return self._record_loop_end(
                    state,
                    _successful_implementation_result(
                        tool_call.args["summary"],
                        loop.changed_files,
                        _optional_tool_notes(tool_call.args),
                    ),
                    turns_used=turns_used,
                )

            tool_result = self._execute_tool_call(loop, tool_call)
            self._record_loop(
                state,
                f"turn.{turns_used}.result",
                {
                    "action": tool_call.action,
                    "args": safe_tool_args(tool_call.args),
                    "ok": tool_result.get("ok"),
                    "error_code": tool_result.get("error_code"),
                    "changed_file_count": len(loop.changed_files),
                },
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "tool_result: "
                        f"{_json_for_prompt(tool_result)}"
                    ),
                }
            )

            if tool_result.get("ok") is False:
                last_failed_tool_result = tool_result
                if tool_result.get("error_code") == "path_boundary_violation":
                    return self._record_loop_end(
                        state,
                        _failed_result_for_tool_error(
                            tool_result,
                            changed_files=loop.changed_files,
                        ),
                        turns_used=turns_used,
                    )
                continue

            last_failed_tool_result = None

        if last_failed_tool_result is not None:
            return self._record_loop_end(
                state,
                _failed_result_for_tool_error(
                    last_failed_tool_result,
                    changed_files=loop.changed_files,
                ),
                turns_used=turns_used,
            )

        return self._record_loop_end(
            state,
            _failed_implementation_result(
                "Implementation stopped before the model called finish.",
                [f"max_turns exhausted after {self._max_turns} turns"],
                changed_files=loop.changed_files,
                code="max_turns_exhausted",
            ),
            turns_used=turns_used,
        )

    def _execute_tool_call(
        self,
        loop: _LoopContext,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        try:
            if tool_call.action == "read_file":
                return self._read_file(loop, tool_call)
            if tool_call.action == "list_dir":
                return self._list_dir(loop, tool_call)
            if tool_call.action == "search":
                return self._search(loop, tool_call)
            if tool_call.action == "edit_file":
                return self._edit_file(loop, tool_call)
            if tool_call.action == "run_tests":
                return self._run_tests(loop)
            return self._write_file(loop, tool_call)
        except PolicyViolationError as exc:
            return _tool_error(
                tool_call.action,
                exc,
                "policy_violation",
                self._tool_result_max_chars,
            )
        except PathBoundaryError as exc:
            return _tool_error(
                tool_call.action,
                exc,
                "path_boundary_violation",
                self._tool_result_max_chars,
            )
        except (AgentSystemError, OSError, ValueError, RuntimeError) as exc:
            return _tool_error(
                tool_call.action,
                exc,
                "tool_execution_failed",
                self._tool_result_max_chars,
            )

    def _read_file(
        self,
        loop: _LoopContext,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        path = tool_call.args["path"]
        offset = tool_call.args.get("offset")
        limit = tool_call.args.get("limit")
        try:
            content = loop.files.read_text(path)
        except FileNotFoundError:
            # A file that does not exist cannot be blindly overwritten.
            loop.complete_view_paths.add(path)
            return {
                "ok": True,
                "action": "read_file",
                "path": path,
                "missing": True,
                "content": "",
                "truncated": False,
                "complete": True,
            }

        if offset is None and limit is None:
            payload = _truncated_content(content, self._tool_result_max_chars)
            complete = not payload["truncated"]
            if complete:
                loop.complete_view_paths.add(path)
            return {
                "ok": True,
                "action": "read_file",
                "path": path,
                "complete": complete,
                **payload,
            }

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        start_index = (offset - 1) if offset else 0
        start_index = min(max(start_index, 0), total_lines)
        end_index = total_lines if limit is None else min(start_index + limit, total_lines)
        selected = "".join(lines[start_index:end_index])
        payload = _truncated_content(selected, self._tool_result_max_chars)
        complete = (
            start_index == 0 and end_index >= total_lines and not payload["truncated"]
        )
        if complete:
            loop.complete_view_paths.add(path)
        return {
            "ok": True,
            "action": "read_file",
            "path": path,
            "start_line": start_index + 1,
            "end_line": end_index,
            "total_lines": total_lines,
            "complete": complete,
            **payload,
        }

    def _list_dir(
        self,
        loop: _LoopContext,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        path = tool_call.args["path"]
        listed_files = list(loop.files.list_files(path))
        return {
            "ok": True,
            "action": "list_dir",
            "path": path,
            **_truncated_file_list(listed_files, self._tool_result_max_chars),
        }

    def _search(
        self,
        loop: _LoopContext,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        pattern = tool_call.args["pattern"]
        search_path = tool_call.args.get("path", ".")
        max_results = _search_result_cap(tool_call.args.get("max_results"))
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {
                "ok": False,
                "action": "search",
                "error": _truncate_text(str(exc), self._tool_result_max_chars),
                "error_code": "invalid_pattern",
            }

        matches: list[str] = []
        files_searched = 0
        truncated = False
        for rel_path in loop.files.list_files(search_path):
            try:
                text = loop.files.read_text(rel_path)
            except (FileNotFoundError, UnicodeDecodeError, OSError):
                continue
            if "\x00" in text:
                continue
            files_searched += 1
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    snippet = _truncate_text(line.strip(), _MAX_SEARCH_LINE_CHARS)
                    matches.append(f"{rel_path}:{lineno}:{snippet}")
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        joined = "\n".join(matches)
        if len(joined) > self._tool_result_max_chars:
            truncated = True
            while matches and len("\n".join(matches)) > self._tool_result_max_chars:
                matches.pop()
        return {
            "ok": True,
            "action": "search",
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches),
            "files_searched": files_searched,
            "truncated": truncated,
        }

    def _edit_file(
        self,
        loop: _LoopContext,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        path = tool_call.args["path"]
        old_string = tool_call.args["old_string"]
        new_string = tool_call.args["new_string"]
        replace_all = tool_call.args.get("replace_all", False)
        try:
            content = loop.files.read_text(path)
        except FileNotFoundError:
            return {
                "ok": False,
                "action": "edit_file",
                "path": path,
                "error": "file does not exist; use write_file to create it",
                "error_code": "edit_target_not_found",
            }

        occurrences = content.count(old_string)
        if occurrences == 0:
            return {
                "ok": False,
                "action": "edit_file",
                "path": path,
                "error": "old_string was not found in the file",
                "error_code": "edit_target_not_found",
            }
        if occurrences > 1 and not replace_all:
            return {
                "ok": False,
                "action": "edit_file",
                "path": path,
                "error": (
                    f"old_string matches {occurrences} times; add surrounding "
                    "context to make it unique or set replace_all to true"
                ),
                "error_code": "edit_target_ambiguous",
            }

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = occurrences
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        loop.files.write_text(path, new_content)
        # Reading the whole file to edit it means we hold a complete view.
        loop.complete_view_paths.add(path)
        if path not in loop.changed_files:
            loop.changed_files.append(path)
        return {
            "ok": True,
            "action": "edit_file",
            "path": path,
            "replacements": replacements,
        }

    def _write_file(
        self,
        loop: _LoopContext,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        path = tool_call.args["path"]
        content = tool_call.args["content"]
        exists = getattr(loop.files, "exists", None)
        if (
            callable(exists)
            and exists(path)
            and path not in loop.complete_view_paths
        ):
            return {
                "ok": False,
                "action": "write_file",
                "path": path,
                "error": (
                    "refusing to overwrite a file you have not fully read; read "
                    "the whole file first or use edit_file for a targeted change"
                ),
                "error_code": "truncated_write_rejected",
            }
        loop.files.write_text(path, content)
        # We just wrote the whole content, so we hold a complete view.
        loop.complete_view_paths.add(path)
        if path not in loop.changed_files:
            loop.changed_files.append(path)
        return {
            "ok": True,
            "action": "write_file",
            "path": path,
        }

    def _run_tests(self, loop: _LoopContext) -> dict[str, Any]:
        if loop.test_runner is None:
            return {
                "ok": False,
                "action": "run_tests",
                "error": "run_tests is not available in this run",
                "error_code": "tests_unavailable",
            }
        if loop.test_runs >= loop.max_test_runs:
            return {
                "ok": False,
                "action": "run_tests",
                "error": (
                    f"test run budget of {loop.max_test_runs} runs is exhausted"
                ),
                "error_code": "test_budget_exhausted",
            }
        loop.test_runs += 1
        try:
            result = loop.test_runner()
        except Exception as exc:  # noqa: BLE001 - surface any runner failure to model
            return {
                "ok": False,
                "action": "run_tests",
                "error": _truncate_text(
                    f"{type(exc).__name__}: {exc}",
                    self._tool_result_max_chars,
                ),
                "error_code": "test_run_failed",
            }
        return _summarized_test_result(result, self._tool_result_max_chars)


class ModelRouterReviewService:
    """Review implementation results through the internal model router."""

    def __init__(self, model_router: ModelRouterProtocol) -> None:
        self._model_router = model_router

    async def review(self, state: TicketState) -> dict[str, Any]:
        response = await self._model_router.invoke(
            capability="code.verify",
            messages=_messages_for_model(_review_messages(state), state),
            ticket_id=state.ticket_key,
            metadata={
                "workflow_node": "review",
                "goal_id": state.goal_id,
                "goal_iteration": state.implementation_attempts,
            },
        )
        payload = _coerce_model_payload(response)
        passed = _optional_bool(
            payload.get("passed", payload.get("approved")),
            False,
            "passed or approved",
        )

        issues = _optional_string_list(payload.get("issues"), "issues")
        verdicts = _parse_criteria_verdicts(payload.get("criteria_verdicts"))
        unmet = [verdict["criterion"] for verdict in verdicts if not verdict["met"]]
        if unmet:
            # An explicit unmet acceptance criterion overrides a passed verdict.
            passed = False
            issues = [
                *issues,
                *(f"Unmet acceptance criterion: {criterion}" for criterion in unmet),
            ]

        result: dict[str, Any] = {
            "passed": passed,
            "status": "approved" if passed else "rejected",
            "reasoning": _optional_string(payload.get("reasoning"), ""),
            "issues": issues,
        }
        if verdicts:
            result["criteria_verdicts"] = verdicts
        if "confidence" in payload and payload["confidence"] is not None:
            result["confidence"] = _float_field(payload["confidence"], "confidence")
        return result


def _coerce_model_payload(response: Any) -> dict[str, Any]:
    return _coerce_model_payload_inner(response, seen=set())


def _coerce_model_payload_inner(response: Any, *, seen: set[int]) -> dict[str, Any]:
    response_id = id(response)
    if response_id in seen:
        raise ModelServiceError("model response envelope is recursive")
    seen.add(response_id)

    if isinstance(response, Mapping):
        if len(response) == 1:
            for field in _MODEL_ENVELOPE_FIELDS:
                if field in response:
                    return _coerce_envelope_field(response, field, seen=seen)
        envelope_field = _mapping_envelope_field(response)
        if envelope_field is not None:
            return _coerce_envelope_field(response, envelope_field, seen=seen)
        return dict(response)

    if isinstance(response, str):
        return _extract_json_object(response)

    for field in _MODEL_ENVELOPE_FIELDS:
        if hasattr(response, field):
            value = getattr(response, field)
            if value is not None:
                return _coerce_envelope_value(value, field, seen=seen)

    raise ModelServiceError(
        f"model response has unsupported shape: {type(response).__name__}"
    )


def _mapping_envelope_field(response: Mapping[str, Any]) -> str | None:
    has_metadata = bool(_MODEL_ENVELOPE_METADATA_FIELDS.intersection(response))
    invalid_metadata_field: str | None = None

    for field in _MODEL_ENVELOPE_FIELDS:
        if field not in response or response[field] is None:
            continue
        value = response[field]
        if _is_payload_like_envelope_value(value):
            return field
        if has_metadata:
            invalid_metadata_field = field

    return invalid_metadata_field


def _is_payload_like_envelope_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return True
    if isinstance(value, str):
        return _try_extract_json_object(value) is not None
    return False


def _coerce_envelope_field(
    response: Mapping[str, Any],
    field: str,
    *,
    seen: set[int],
) -> dict[str, Any]:
    return _coerce_envelope_value(response[field], field, seen=seen)


def _coerce_envelope_value(value: Any, field: str, *, seen: set[int]) -> dict[str, Any]:
    try:
        return _coerce_model_payload_inner(value, seen=seen)
    except ModelServiceError as exc:
        raise ModelServiceError(
            f"model response envelope field {field!r} could not be parsed: {exc}"
        ) from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ModelServiceError("model response is empty")

    direct = _parse_json_dict(stripped)
    if direct is not None:
        return direct

    fenced_match = _FENCED_JSON_RE.search(stripped)
    if fenced_match is not None:
        fenced = _parse_json_dict(fenced_match.group(1).strip())
        if fenced is not None:
            return fenced

    for start_index, character in enumerate(stripped):
        if character != "{":
            continue
        candidate = _balanced_json_object(stripped, start_index)
        if candidate is None:
            continue
        parsed = _parse_json_dict(candidate)
        if parsed is not None:
            return parsed

    raise ModelServiceError("model response could not be parsed as a JSON object")


def _try_extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        return _extract_json_object(text)
    except ModelServiceError:
        return None


def _parse_json_dict(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        raise ModelServiceError("model response JSON must be an object")
    return parsed


def _balanced_json_object(text: str, start_index: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start_index, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    return None


def _tool_call_from_payload(payload: Mapping[str, Any]) -> ToolCall:
    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ToolCallValidationError(
            "invalid_tool_call",
            "tool call must include action as a string",
        )
    if action not in _TOOL_ACTIONS:
        raise ToolCallValidationError(
            "unknown_action",
            f"unknown tool action: {action!r}",
        )
    if "args" not in payload:
        raise ToolCallValidationError(
            "invalid_tool_call",
            "tool call must include args as an object",
        )
    if not isinstance(payload["args"], Mapping):
        raise ToolCallValidationError(
            "invalid_tool_call",
            "tool call args must be an object",
        )

    try:
        return ToolCall.model_validate(payload)
    except ValidationError as exc:
        raise ToolCallValidationError(
            "invalid_tool_call",
            f"invalid tool call: {exc.errors()[0]['msg']}",
        ) from exc


def _tool_arg_string(
    args: Mapping[str, Any],
    name: str,
    action: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = args.get(name)
    valid = isinstance(value, str) and (allow_empty or bool(value.strip()))
    if not valid:
        raise ValueError(f"{action}.args.{name} must be a string")
    return value


def _optional_tool_arg_int(
    args: Mapping[str, Any],
    name: str,
    action: str,
    *,
    minimum: int,
) -> int | None:
    if name not in args:
        return None
    value = args[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{action}.args.{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{action}.args.{name} must be >= {minimum}")
    return value


def _optional_tool_arg_bool(
    args: Mapping[str, Any],
    name: str,
    action: str,
) -> bool:
    value = args[name]
    if not isinstance(value, bool):
        raise ValueError(f"{action}.args.{name} must be a boolean")
    return value


def _tool_arg_string_list(value: Any, name: str, action: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{action}.args.{name} must be a list of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"{action}.args.{name}[{index}] must be a string"
            )
        result.append(item)
    return result


def _successful_implementation_result(
    summary: str,
    changed_files: Sequence[str],
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "status": "success",
        "summary": summary,
        "changed_files": list(changed_files),
        "notes": list(notes),
        "errors": [],
    }


def _failed_implementation_result(
    summary: str,
    errors: Sequence[str],
    *,
    changed_files: Sequence[str],
    code: str = "implementation_failed",
) -> dict[str, Any]:
    error_list = [error for error in errors if error]
    result: dict[str, Any] = {
        "status": "failed",
        "summary": summary,
        "changed_files": list(changed_files),
        "notes": [],
        "errors": error_list,
        "error_code": code,
    }
    if error_list:
        result["error"] = error_list[0]
    return result


def _failed_result_for_tool_error(
    tool_result: Mapping[str, Any],
    *,
    changed_files: Sequence[str],
) -> dict[str, Any]:
    return _failed_implementation_result(
        "Implementation stopped because a tool call failed.",
        [_optional_string(tool_result.get("error"), "tool call failed")],
        changed_files=changed_files,
        code=_optional_string(tool_result.get("error_code"), "tool_failed"),
    )


def _optional_tool_notes(args: Mapping[str, Any]) -> list[str]:
    notes = args.get("notes")
    if notes is None:
        return []
    return [note for note in notes if isinstance(note, str)]


def _truncated_content(content: str, max_chars: int) -> dict[str, Any]:
    truncated = _truncate_text(content, max_chars)
    return {
        "content": truncated,
        "truncated": len(truncated) != len(content),
        "original_chars": len(content),
    }


def _truncated_file_list(paths: Sequence[str], max_chars: int) -> dict[str, Any]:
    kept: list[str] = []
    used_chars = 2
    for path in paths:
        path_chars = len(path) + 4
        if kept and used_chars + path_chars > max_chars:
            break
        kept.append(path)
        used_chars += path_chars
    return {
        "files": kept,
        "truncated": len(kept) != len(paths),
        "total_files": len(paths),
    }


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    suffix = f"\n...[truncated {len(text) - max_chars} chars]"
    return text[: max(0, max_chars - len(suffix))] + suffix


def _tool_error(
    action: str,
    exc: Exception,
    error_code: str,
    max_chars: int,
) -> dict[str, Any]:
    return {
        "ok": False,
        "action": action,
        "error": _truncate_text(str(exc), max_chars),
        "error_code": error_code,
    }


def _search_result_cap(requested: Any) -> int:
    if not isinstance(requested, int) or isinstance(requested, bool):
        return _DEFAULT_SEARCH_RESULT_CAP
    return max(1, min(requested, _MAX_SEARCH_RESULT_CAP))


def _summarized_test_result(result: Any, max_chars: int) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {
            "ok": True,
            "action": "run_tests",
            "tests_passed": None,
            "output": "",
        }
    output = result.get("output") or result.get("stdout") or ""
    return {
        "ok": True,
        "action": "run_tests",
        "tests_passed": result.get("tests_passed"),
        "status": result.get("status"),
        "exit_code": result.get("exit_code"),
        "timed_out": result.get("timed_out"),
        "output": _truncate_text(str(output), max_chars),
    }


def _planning_messages(state: TicketState) -> list[dict[str, str]]:
    criteria = parse_acceptance_criteria(state.description)
    criteria_lines: list[str] = []
    if criteria:
        criteria_lines = [
            "acceptance_criteria (the plan must cover every one of these):",
            *(f"- {criterion}" for criterion in criteria),
            "Include a criteria_coverage object mapping each acceptance "
            "criterion (verbatim) to the plan step that satisfies it. Every "
            "criterion must appear as a key.",
        ]
    coverage_schema = (
        ', "criteria_coverage": {"criterion": "plan step"}'
        if criteria
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You decompose software tickets into concise execution plans. "
                "Return exactly one strict JSON object. Do not include markdown "
                "fences or prose."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "Create a concise implementation plan for this ticket.",
                    f"ticket_key: {state.ticket_key}",
                    f"summary: {state.summary}",
                    f"description: {state.description}",
                    f"repository: {state.repository or ''}",
                    f"max_attempts: {state.max_attempts}",
                    *criteria_lines,
                    "Plan only this Jira ticket's Ticket scope. Treat any "
                    "Original Slack request text as background for product "
                    "context, not as authorization to implement the whole "
                    "request.",
                    "Do not plan sibling tickets, duplicate the whole app, "
                    "invent a different framework, or create a different "
                    "repository/app path than the ticket context implies.",
                    "files_to_modify entries must be repository-relative paths "
                    "and should stay inside the ticket's scoped app/module.",
                    "Required JSON schema:",
                    (
                        '{"plan": "string", '
                        '"files_to_modify": ["relative/path.py"], '
                        '"risks": ["string"], '
                        '"complexity": "low|medium|high", '
                        '"requires_human_review": false'
                        + coverage_schema
                        + "}"
                    ),
                    "Paths must be relative, must not contain '..', and must "
                    "not be absolute.",
                    "Return JSON only. No markdown fences. No prose before or "
                    "after JSON.",
                ]
            ),
        },
    ]


def _implementation_messages(
    state: TicketState,
    repo_context: RepoContext,
) -> list[dict[str, str]]:
    failed_test_excerpt = _failed_test_excerpt(state.test_result)
    failed_implementation_excerpt = _failed_implementation_excerpt(
        state.implementation_result
    )
    rejected_review_excerpt = _rejected_review_excerpt(state)
    user_lines = [
        "Create an explicit file operation plan for this ticket.",
        f"ticket_key: {state.ticket_key}",
        f"summary: {state.summary}",
        f"description: {state.description}",
        f"decomposition: {_json_for_prompt(state.decomposition)}",
        f"previous_test_result: {_json_for_prompt(state.test_result)}",
        f"implementation_attempts: {state.implementation_attempts}",
        f"max_attempts: {state.max_attempts}",
        f"repository: {state.repository or ''}",
        f"repo_context: {_json_for_prompt(repo_context.to_prompt_dict())}",
    ]
    if failed_test_excerpt is not None:
        user_lines.append(f"previous_test_failure: {failed_test_excerpt}")
    if failed_implementation_excerpt is not None:
        user_lines.append(
            f"previous_implementation_failure: {failed_implementation_excerpt}"
        )
    if rejected_review_excerpt is not None:
        user_lines.append(f"previous_review_rejection: {rejected_review_excerpt}")
    user_lines.extend(
        [
            *_write_policy_prompt_lines(repo_context),
            "Required JSON schema:",
            (
                '{"summary": "string", "operations": ['
                '{"type": "write_file", '
                '"path": "relative/path.py", '
                '"content": "file content"}]}'
            ),
            "Implementation supports write_file only for now.",
            "Paths must be relative, must not contain '..', and must "
            "not be absolute.",
            "Return JSON only. No markdown fences. No prose before or "
            "after JSON.",
        ]
    )

    return [
        {
            "role": "system",
            "content": (
                "You produce explicit repository file write plans for a "
                "software implementation agent. Return exactly one strict JSON "
                "object. Do not include markdown fences or prose."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(user_lines),
        },
    ]


def _implementation_loop_messages(
    state: TicketState,
    repo_context: RepoContext,
    *,
    tests_available: bool = False,
) -> list[dict[str, str]]:
    failed_test_excerpt = _failed_test_excerpt(state.test_result)
    failed_implementation_excerpt = _failed_implementation_excerpt(
        state.implementation_result
    )
    rejected_review_excerpt = _rejected_review_excerpt(state)
    user_lines = [
        "Implement this ticket by returning one JSON tool call at a time.",
        f"ticket_key: {state.ticket_key}",
        f"summary: {state.summary}",
        f"description: {state.description}",
        f"decomposition_plan: {_json_for_prompt(state.decomposition)}",
        f"repo_context: {_json_for_prompt(repo_context.to_prompt_dict())}",
        f"current_implementation_attempt: {state.implementation_attempts + 1}",
        f"implementation_attempts_completed: {state.implementation_attempts}",
        f"max_attempts: {state.max_attempts}",
        f"repository: {state.repository or ''}",
    ]
    if failed_test_excerpt is not None:
        user_lines.append(f"previous_test_failure: {failed_test_excerpt}")
    if failed_implementation_excerpt is not None:
        user_lines.append(
            f"previous_implementation_failure: {failed_implementation_excerpt}"
        )
    if rejected_review_excerpt is not None:
        user_lines.append(
            f"previous_review_rejection: {rejected_review_excerpt}"
        )
        user_lines.append(
            "The previous implementation passed tests but was rejected by "
            "review. Address every listed issue and unmet acceptance "
            "criterion before you finish."
        )
    run_tests_actions = (
        ['{"action": "run_tests", "args": {}}'] if tests_available else []
    )
    user_lines.extend(
        [
            *_write_policy_prompt_lines(repo_context),
            "Tool call schema:",
            (
                '{"action": "read_file", "args": '
                '{"path": "relative/path.py", "offset": 1, "limit": 200}}'
            ),
            '{"action": "list_dir", "args": {"path": "src"}}',
            (
                '{"action": "search", "args": '
                '{"pattern": "regex", "path": "src", "max_results": 50}}'
            ),
            (
                '{"action": "edit_file", "args": '
                '{"path": "relative/path.py", "old_string": "exact text", '
                '"new_string": "replacement", "replace_all": false}}'
            ),
            (
                '{"action": "write_file", "args": '
                '{"path": "relative/path.py", "content": "complete file content"}}'
            ),
            *run_tests_actions,
            (
                '{"action": "finish", "args": '
                '{"summary": "string", "notes": ["optional string"]}}'
            ),
            (
                "Available actions are read_file, list_dir, search, edit_file, "
                "write_file"
                + (", run_tests," if tests_available else ",")
                + " and finish."
            ),
            "Use search to locate code by pattern before reading whole files; "
            "use list_dir and read_file when you need more context.",
            "read_file accepts optional 1-based offset and limit to page through "
            "large files; a result with complete=false means you have not seen "
            "the whole file yet.",
            "Prefer edit_file for changes to existing files: give an old_string "
            "that is unique in the file and the exact new_string to replace it. "
            "Set replace_all to true only when you intend to replace every "
            "occurrence.",
            "Use write_file only to create a new file or to fully rewrite a file "
            "whose complete content you have read; a blind write_file over a file "
            "you have not fully read is rejected as truncated_write_rejected.",
            "When you add imports, setup files, framework config, or test helpers "
            "that require packages, update the allowed dependency manifest "
            "such as package.json or pyproject.toml in the same attempt.",
            "For TypeScript React tests, do not import "
            "@testing-library/jest-dom/vitest unless package.json includes "
            "@testing-library/jest-dom as a dev dependency.",
            "If a tool_result reports ok=false, use the error_code and error to "
            "choose a safe allowed path or a smaller valid edit before trying "
            "again.",
            *(
                [
                    "Call run_tests to run the repo-contract test command inside "
                    "the worktree; read tests_passed in the result. Fix failures "
                    "and run again until tests pass before you finish. run_tests "
                    "is limited to a small number of runs per attempt.",
                ]
                if tests_available
                else [
                    "You cannot run tests in this run; the graph test node runs "
                    "tests after finish.",
                ]
            ),
            "Call finish only after all required file edits have been written"
            + (" and tests pass." if tests_available else "."),
            "Return exactly one strict JSON object per response.",
            "No markdown fences. No prose before or after JSON.",
        ]
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a software implementation agent controlled through "
                "provider-agnostic JSON-over-text tool calls. You do not have "
                "native tool use. Return exactly one JSON object matching the "
                "Tool call schema each turn."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(user_lines),
        },
    ]


def _write_policy_prompt_lines(repo_context: RepoContext) -> list[str]:
    contract = repo_context.repo_contract
    if contract is None:
        return []

    return [
        "Repo contract write policy:",
        (
            "Write source files only under source_dirs or test_dirs: "
            f"{_json_for_prompt(list(contract.source_dirs + contract.test_dirs))}."
        ),
        (
            "Write root-level config/static files only when the exact path is "
            "listed in config_paths_allowed: "
            f"{_json_for_prompt(list(contract.config_paths_allowed))}."
        ),
        (
            "Never write secret-bearing dotenv files such as .env, .env.local, "
            "or other .env.* files. The placeholder template .env.example may "
            "be written only when it is listed in config_paths_allowed."
        ),
        (
            "Do not invent top-level directories outside those allowed lists; "
            "choose an existing allowed source directory such as src/, app/, "
            "components/, lib/, data/, pages/, public/, or styles/ when present."
        ),
    ]


def _failed_test_excerpt(test_result: Any) -> str | None:
    if not isinstance(test_result, Mapping):
        return None
    passed = test_result.get("tests_passed")
    status = test_result.get("status")
    failed = passed is False or (isinstance(status, str) and status.lower() == "failed")
    if not failed:
        return None
    excerpt_parts: list[str] = []
    for field in ("stdout", "stderr", "summary", "output", "error", "message"):
        value = test_result.get(field)
        if isinstance(value, str) and value.strip():
            excerpt_parts.append(f"{field}: {value.strip()}")
    if not excerpt_parts:
        return _truncate_failure_excerpt(_json_for_prompt(test_result))
    return _truncate_failure_excerpt(" | ".join(excerpt_parts))


def _failed_implementation_excerpt(implementation_result: Any) -> str | None:
    if not isinstance(implementation_result, Mapping):
        return None
    status = implementation_result.get("status")
    failed = isinstance(status, str) and status.lower() in {
        "failed",
        "failure",
        "error",
    }
    if not failed and not (
        implementation_result.get("error")
        or implementation_result.get("error_code")
    ):
        return None

    excerpt_parts: list[str] = []
    for field in ("status", "error_code", "error", "summary"):
        value = implementation_result.get(field)
        if isinstance(value, str) and value.strip():
            excerpt_parts.append(f"{field}: {value.strip()}")
    errors = implementation_result.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, str):
        cleaned_errors = [
            item.strip()
            for item in errors
            if isinstance(item, str) and item.strip()
        ]
        if cleaned_errors:
            excerpt_parts.append(f"errors: {'; '.join(cleaned_errors)}")
    if not excerpt_parts:
        return _truncate_failure_excerpt(_json_for_prompt(implementation_result))
    return _truncate_failure_excerpt(" | ".join(excerpt_parts))


def _rejected_review_excerpt(state: TicketState) -> str | None:
    if state.review_passed is not False:
        return None
    result = state.verification_result
    if not isinstance(result, Mapping):
        return None

    excerpt_parts: list[str] = []
    reasoning = result.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        excerpt_parts.append(f"reasoning: {reasoning.strip()}")
    issues = result.get("issues")
    if isinstance(issues, Sequence) and not isinstance(issues, str):
        cleaned_issues = [
            item.strip() for item in issues if isinstance(item, str) and item.strip()
        ]
        if cleaned_issues:
            excerpt_parts.append(f"issues: {'; '.join(cleaned_issues)}")
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        excerpt_parts.append(f"error: {error.strip()}")
    if not excerpt_parts:
        return _truncate_failure_excerpt(_json_for_prompt(result))
    return _truncate_failure_excerpt(" | ".join(excerpt_parts))


def _truncate_failure_excerpt(excerpt: str) -> str:
    if len(excerpt) <= _MAX_FAILURE_EXCERPT_CHARS:
        return excerpt
    return excerpt[: _MAX_FAILURE_EXCERPT_CHARS - 14].rstrip() + "...[truncated]"


def _parse_criteria_verdicts(value: Any) -> list[dict[str, Any]]:
    """Normalize a model-returned criteria_verdicts array.

    Ignores malformed entries. A missing ``met`` is treated as not met so a
    reviewer cannot pass a criterion by omitting the flag.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    verdicts: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        criterion = _optional_string(entry.get("criterion"), "")
        if not criterion:
            continue
        verdicts.append(
            {
                "criterion": criterion,
                "met": entry.get("met") is True,
                "evidence": _optional_string(entry.get("evidence"), ""),
            }
        )
    return verdicts


def _review_messages(state: TicketState) -> list[dict[str, str]]:
    criteria = parse_acceptance_criteria(state.description)
    criteria_lines: list[str] = []
    verdict_schema = ""
    if criteria:
        criteria_lines = [
            "acceptance_criteria (judge each one against the actual changes):",
            *(f"- {criterion}" for criterion in criteria),
            "Return a criteria_verdicts array with one entry per criterion: "
            '{"criterion": "verbatim text", "met": true, "evidence": '
            '"one line pointing to the change or test"}. Set passed to false '
            "if any criterion is not met.",
        ]
        verdict_schema = (
            ', "criteria_verdicts": [{"criterion": "string", '
            '"met": true, "evidence": "string"}]'
        )
    return [
        {
            "role": "system",
            "content": (
                "You verify whether implementation results satisfy a ticket. "
                "Return exactly one strict JSON object. Do not include markdown "
                "fences or prose."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "Review this ticket implementation.",
                    f"ticket_key: {state.ticket_key}",
                    f"summary: {state.summary}",
                    f"description: {state.description}",
                    f"decomposition: {_json_for_prompt(state.decomposition)}",
                    (
                        "implementation_result: "
                        f"{_json_for_prompt(state.implementation_result)}"
                    ),
                    f"test_result: {_json_for_prompt(state.test_result)}",
                    f"branch_name: {state.branch_name or ''}",
                    (
                        "changed_files: "
                        f"{_json_for_prompt(_changed_files_from_state(state))}"
                    ),
                    *criteria_lines,
                    "Required JSON schema:",
                    (
                        '{"passed": true, "reasoning": "string", '
                        '"issues": ["string"], "confidence": 0.0'
                        + verdict_schema
                        + "}"
                    ),
                    "Return JSON only. No markdown fences. No prose before or "
                    "after JSON.",
                ]
            ),
        },
    ]


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _messages_for_model(
    messages: Sequence[Mapping[str, str]],
    state: TicketState,
) -> list[dict[str, str]]:
    local_paths = tuple(
        path for path in (state.worktree_path, state.repo_path) if path
    )
    return [
        {
            "role": message["role"],
            "content": redact_local_paths(message["content"], local_paths),
        }
        for message in messages
    ]


def _changed_files_from_state(state: TicketState) -> list[str]:
    if not state.implementation_result:
        return []
    changed_files = state.implementation_result.get("changed_files")
    if not isinstance(changed_files, Sequence) or isinstance(changed_files, str):
        return []
    return [path for path in changed_files if isinstance(path, str)]


def _required_file_operations(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    files = payload.get("operations", payload.get("files"))
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ModelServiceError("model response must include operations as a list")
    operations: list[Mapping[str, Any]] = []
    for index, operation in enumerate(files):
        if not isinstance(operation, Mapping):
            raise ModelServiceError(f"file operation at index {index} must be an object")
        operations.append(operation)
    return operations


def _apply_file_operation(
    files: Any,
    operation: Mapping[str, Any],
    index: int,
) -> str:
    operation_name = operation.get("type", operation.get("operation"))
    if operation_name != "write_file":
        raise ModelServiceError(
            f"unsupported file operation at index {index}: {operation_name!r}"
        )

    path = _required_string(operation.get("path"), f"files[{index}].path")
    _validate_relative_path(path, index)
    if "content" not in operation:
        raise ModelServiceError(f"files[{index}].content is required")
    content = operation["content"]
    if not isinstance(content, str):
        raise ModelServiceError(f"files[{index}].content must be a string")

    files.write_text(path, content)
    return path


def _validate_relative_path(path: str, index: int) -> None:
    path_obj = Path(path)
    if path_obj.is_absolute() or PureWindowsPath(path).is_absolute():
        raise ModelServiceError(f"files[{index}].path must be relative")

    normalized = path.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    if not parts:
        raise ModelServiceError(f"files[{index}].path must not be empty")
    if ".." in parts:
        raise ModelServiceError(f"files[{index}].path must not contain '..'")


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelServiceError(f"model response must include {field_name} as a string")
    return value


def _optional_string(value: Any, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ModelServiceError("model response field must be a string")
    return value


def _optional_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ModelServiceError(f"model response field {field_name} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ModelServiceError(
                f"model response field {field_name}[{index}] must be a string"
            )
        result.append(item)
    return result


def _optional_bool(value: Any, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ModelServiceError(f"model response field {field_name} must be a boolean")
    return value


def _float_field(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelServiceError(f"model response field {field_name} must be a number")
    return float(value)


__all__ = [
    "IterativeImplementationService",
    "ModelRouterImplementationService",
    "ModelRouterPlannerService",
    "ModelRouterProtocol",
    "ModelRouterReviewService",
    "ModelServiceError",
    "ToolCall",
]

"""ModelRouter-backed services for ticket workflow orchestration."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.orchestrator.repo_context import (
    RepoContext,
    RepoContextBuilder,
)
from ticket_agent.orchestrator.state import TicketState


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
ToolAction = Literal["read_file", "write_file", "list_dir", "finish"]

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
_TOOL_ACTIONS = frozenset({"read_file", "write_file", "list_dir", "finish"})
_DEFAULT_MAX_TOOL_RESULT_CHARS = 6000


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
    def validate_args_for_action(self) -> "ToolCall":
        if self.action not in _TOOL_ACTIONS:
            raise ValueError(f"unknown action: {self.action!r}")

        if self.action in {"read_file", "write_file", "list_dir"}:
            _tool_arg_string(self.args, "path", self.action)
        if self.action == "write_file":
            _tool_arg_string(
                self.args,
                "content",
                self.action,
                allow_empty=True,
            )
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
            messages=_planning_messages(state),
            ticket_id=state.ticket_key,
            metadata={"workflow_node": "plan"},
        )
        payload = _coerce_model_payload(response)
        plan = _required_string(
            payload.get("plan", payload.get("summary")),
            "plan or summary",
        )

        return {
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
            messages=_implementation_messages(state, repo_context),
            ticket_id=state.ticket_key,
            metadata={"workflow_node": "implement"},
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


class IterativeImplementationService:
    """Run a provider-agnostic JSON tool-call implementation loop."""

    def __init__(
        self,
        model_router: ModelRouterProtocol,
        file_adapter_factory: FileAdapterFactory | None = None,
        repo_context_builder: RepoContextBuilder | None = None,
        *,
        max_turns: int = 12,
        tool_result_max_chars: int = _DEFAULT_MAX_TOOL_RESULT_CHARS,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if tool_result_max_chars < 256:
            raise ValueError("tool_result_max_chars must be at least 256")

        self._model_router = model_router
        self._file_adapter_factory = file_adapter_factory or LocalFileAdapter
        self._repo_context_builder = repo_context_builder or RepoContextBuilder()
        self._max_turns = max_turns
        self._tool_result_max_chars = tool_result_max_chars

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

        return await self._run_loop(context.state, context.files)

    async def _run_loop(self, state: TicketState, files: Any) -> dict[str, Any]:
        repo_context = self._repo_context_builder.build(state)
        messages = _implementation_loop_messages(state, repo_context)
        changed_files: list[str] = []

        for turn_index in range(self._max_turns):
            try:
                response = await self._model_router.invoke(
                    capability="code.implement",
                    messages=messages,
                    ticket_id=state.ticket_key,
                    metadata={
                        "workflow_node": "implement",
                        "implementation_turn": turn_index + 1,
                    },
                )
                payload = _coerce_model_payload(response)
                tool_call = _tool_call_from_payload(payload)
            except (ModelServiceError, ToolCallValidationError) as exc:
                return _failed_implementation_result(
                    "Implementation stopped because the model returned an invalid "
                    "tool call.",
                    [str(exc)],
                    changed_files=changed_files,
                    code=getattr(exc, "code", "invalid_tool_call"),
                )
            except Exception as exc:
                return _failed_implementation_result(
                    "Implementation stopped because the model call failed.",
                    [f"{type(exc).__name__}: {exc}"],
                    changed_files=changed_files,
                    code="model_invoke_failed",
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": _json_for_prompt(tool_call.model_dump()),
                }
            )

            if tool_call.action == "finish":
                return _successful_implementation_result(
                    tool_call.args["summary"],
                    changed_files,
                    _optional_tool_notes(tool_call.args),
                )

            tool_result = self._execute_tool_call(files, tool_call, changed_files)
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
                return _failed_implementation_result(
                    "Implementation stopped because a tool call failed.",
                    [_optional_string(tool_result.get("error"), "tool call failed")],
                    changed_files=changed_files,
                    code=_optional_string(
                        tool_result.get("error_code"),
                        "tool_failed",
                    ),
                )

        return _failed_implementation_result(
            "Implementation stopped before the model called finish.",
            [f"max_turns exhausted after {self._max_turns} turns"],
            changed_files=changed_files,
            code="max_turns_exhausted",
        )

    def _execute_tool_call(
        self,
        files: Any,
        tool_call: ToolCall,
        changed_files: list[str],
    ) -> dict[str, Any]:
        try:
            if tool_call.action == "read_file":
                path = tool_call.args["path"]
                content = files.read_text(path)
                return {
                    "ok": True,
                    "action": "read_file",
                    "path": path,
                    **_truncated_content(content, self._tool_result_max_chars),
                }

            if tool_call.action == "list_dir":
                path = tool_call.args["path"]
                listed_files = list(files.list_files(path))
                return {
                    "ok": True,
                    "action": "list_dir",
                    "path": path,
                    **_truncated_file_list(
                        listed_files,
                        self._tool_result_max_chars,
                    ),
                }

            path = tool_call.args["path"]
            content = tool_call.args["content"]
            files.write_text(path, content)
            if path not in changed_files:
                changed_files.append(path)
            return {
                "ok": True,
                "action": "write_file",
                "path": path,
            }
        except (AgentSystemError, OSError, ValueError, RuntimeError) as exc:
            return {
                "ok": False,
                "action": tool_call.action,
                "error": _truncate_text(str(exc), self._tool_result_max_chars),
                "error_code": "tool_execution_failed",
            }


class ModelRouterReviewService:
    """Review implementation results through the internal model router."""

    def __init__(self, model_router: ModelRouterProtocol) -> None:
        self._model_router = model_router

    async def review(self, state: TicketState) -> dict[str, Any]:
        response = await self._model_router.invoke(
            capability="code.verify",
            messages=_review_messages(state),
            ticket_id=state.ticket_key,
            metadata={"workflow_node": "review"},
        )
        payload = _coerce_model_payload(response)
        passed = _optional_bool(
            payload.get("passed", payload.get("approved")),
            False,
            "passed or approved",
        )

        result: dict[str, Any] = {
            "passed": passed,
            "status": "approved" if passed else "rejected",
            "reasoning": _optional_string(payload.get("reasoning"), ""),
            "issues": _optional_string_list(payload.get("issues"), "issues"),
        }
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


def _planning_messages(state: TicketState) -> list[dict[str, str]]:
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
                    f"repo_path: {state.repo_path or ''}",
                    f"max_attempts: {state.max_attempts}",
                    "Required JSON schema:",
                    (
                        '{"plan": "string", '
                        '"files_to_modify": ["relative/path.py"], '
                        '"risks": ["string"], '
                        '"complexity": "low|medium|high", '
                        '"requires_human_review": false}'
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
        f"repo_path: {state.repo_path or ''}",
        f"worktree_path: {state.worktree_path or ''}",
        f"repo_context: {_json_for_prompt(repo_context.to_prompt_dict())}",
    ]
    if failed_test_excerpt is not None:
        user_lines.append(f"previous_test_failure: {failed_test_excerpt}")
    user_lines.extend(
        [
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
) -> list[dict[str, str]]:
    failed_test_excerpt = _failed_test_excerpt(state.test_result)
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
        f"repo_path: {state.repo_path or ''}",
        f"worktree_path: {state.worktree_path or ''}",
    ]
    if failed_test_excerpt is not None:
        user_lines.append(f"previous_test_failure: {failed_test_excerpt}")
    user_lines.extend(
        [
            "Tool call schema:",
            '{"action": "read_file", "args": {"path": "relative/path.py"}}',
            '{"action": "list_dir", "args": {"path": "src"}}',
            (
                '{"action": "write_file", "args": '
                '{"path": "relative/path.py", "content": "complete file content"}}'
            ),
            (
                '{"action": "finish", "args": '
                '{"summary": "string", "notes": ["optional string"]}}'
            ),
            "Available actions are read_file, list_dir, write_file, and finish.",
            "Use read_file and list_dir when you need more context.",
            "Use write_file with the complete replacement content for that file.",
            "Call finish only after all required file edits have been written.",
            "Do not run tests. The graph test node runs tests after finish.",
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


def _failed_test_excerpt(test_result: Any) -> str | None:
    if not isinstance(test_result, Mapping):
        return None
    passed = test_result.get("tests_passed")
    status = test_result.get("status")
    failed = passed is False or (isinstance(status, str) and status.lower() == "failed")
    if not failed:
        return None
    excerpt_parts: list[str] = []
    for field in ("stdout", "stderr", "summary", "output", "message"):
        value = test_result.get(field)
        if isinstance(value, str) and value.strip():
            excerpt_parts.append(f"{field}: {value.strip()}")
    if not excerpt_parts:
        return _json_for_prompt(test_result)
    return " | ".join(excerpt_parts)


def _review_messages(state: TicketState) -> list[dict[str, str]]:
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
                    "Required JSON schema:",
                    (
                        '{"passed": true, "reasoning": "string", '
                        '"issues": ["string"], "confidence": 0.0}'
                    ),
                    "Return JSON only. No markdown fences. No prose before or "
                    "after JSON.",
                ]
            ),
        },
    ]


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


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

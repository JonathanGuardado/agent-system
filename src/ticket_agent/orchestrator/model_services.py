"""ModelRouter-backed services for ticket workflow orchestration."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any, Protocol

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
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

_MODEL_ENVELOPE_FIELDS = ("content", "text", "message", "data")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


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
    ) -> None:
        self._model_router = model_router
        self._file_adapter_factory = file_adapter_factory or LocalFileAdapter

    async def implement(self, state: TicketState) -> dict[str, Any]:
        if not state.worktree_path:
            raise ModelServiceError("worktree_path is required to apply file writes")

        response = await self._model_router.invoke(
            capability="code.implement",
            messages=_implementation_messages(state),
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
                    return _coerce_model_payload_inner(response[field], seen=seen)
        return dict(response)

    if isinstance(response, str):
        return _extract_json_object(response)

    for field in _MODEL_ENVELOPE_FIELDS:
        if hasattr(response, field):
            value = getattr(response, field)
            if value is not None:
                return _coerce_model_payload_inner(value, seen=seen)

    raise ModelServiceError(
        f"model response has unsupported shape: {type(response).__name__}"
    )


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


def _planning_messages(state: TicketState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You decompose software tickets into concise execution plans. "
                "Return JSON only."
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
                        '{"plan": string, "files_to_modify": string[], '
                        '"risks": string[], "complexity": string, '
                        '"requires_human_review": boolean}'
                    ),
                    "Return JSON only. Do not include markdown.",
                ]
            ),
        },
    ]


def _implementation_messages(state: TicketState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You produce explicit repository file write plans for a "
                "software implementation agent. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
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
                    "Supported file operation:",
                    (
                        '{"operation": "write_file", "path": string, '
                        '"content": string}'
                    ),
                    "Required JSON schema:",
                    (
                        '{"summary": string, "files": [file_operation], '
                        '"notes": string[]}'
                    ),
                    "Return JSON only. Do not include markdown.",
                ]
            ),
        },
    ]


def _review_messages(state: TicketState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You verify whether implementation results satisfy a ticket. "
                "Return JSON only."
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
                        '{"passed": boolean, "reasoning": string, '
                        '"issues": string[], "confidence": number}'
                    ),
                    "Return JSON only. Do not include markdown.",
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
    files = payload.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ModelServiceError("model response must include files as a list")
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
    operation_name = operation.get("operation")
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
    "ModelRouterImplementationService",
    "ModelRouterPlannerService",
    "ModelRouterProtocol",
    "ModelRouterReviewService",
    "ModelServiceError",
]

"""Append-only JSONL transcripts of loop decisions.

This records *decisions*, not payloads: which node ran, which strategy was
chosen, how many turns a loop took, what shape a tool call had. Content is
deliberately excluded -- a gate's output can be megabytes and a file write can
be a secret -- so events carry sizes, keys, and references instead of values.

Three properties this module must hold, each learned from a specific failure:

**It never raises.** Observability that can break the pipeline is worse than
no observability. Every failure path warns (rate-limited) and increments
``write_failures``, which the report script surfaces so silent loss is
visible.

**Concurrent writers cannot interleave.** Each event is serialized fully and
issued as a single ``write()`` under a lock, so two pollers writing at once
produce two whole lines rather than one corrupt one.

**Externally supplied identifiers are validated before any path is built.**
``ticket_key`` and ``run_id`` reach this module from Jira and from lock ids;
both are used in a filename, which makes them a path-traversal boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
import json
import logging
from pathlib import Path
import re
from threading import RLock
from typing import Any, Protocol, Self, TextIO

from ticket_agent.redaction import redact

_LOGGER = logging.getLogger(__name__)

_TICKET_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")

#: Open file handles are capped so a long-running process cannot exhaust FDs.
_MAX_OPEN_HANDLES = 16

#: Values longer than this are truncated with an explicit marker, so a
#: transcript can never be inflated by one runaway string.
_MAX_VALUE_CHARS = 2_000
_TRUNCATION_SUFFIX = "…[truncated]"

#: Tool-call arguments safe to record verbatim. Everything else is reduced to
#: a character count -- ``content``/``old_string``/``new_string`` carry file
#: bodies and are exactly what must not land on disk here.
_SAFE_TOOL_ARG_KEYS = frozenset(
    {"path", "pattern", "offset", "limit", "replace_all", "max_results"}
)
_SIZED_TOOL_ARG_KEYS = frozenset({"content", "old_string", "new_string", "summary"})


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """One decision point in the loop."""

    ticket_key: str
    kind: str
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    goal_id: str | None = None
    iteration: int | None = None
    phase: str | None = None
    at: datetime | None = None


class TranscriptRecorder(Protocol):
    def record(self, event: TranscriptEvent) -> None: ...


class NullTranscriptRecorder:
    """The default everywhere. Keeps the 595-test baseline silent."""

    __slots__ = ()

    def record(self, event: TranscriptEvent) -> None:
        return None


def safe_record(recorder: TranscriptRecorder, event: TranscriptEvent) -> None:
    """Record an event, swallowing any failure the recorder raises.

    ``TranscriptRecorder`` is a Protocol, so the never-raises guarantee cannot
    live only inside ``JsonlTranscriptRecorder``: any other implementation --
    a test double, an in-memory collector, a future sink -- would otherwise
    propagate straight into the pipeline it is supposed to be observing. Every
    hook records through here so the guarantee holds at the boundary.
    """

    try:
        recorder.record(event)
    except Exception as exc:  # noqa: BLE001 - observability must not break the run
        _LOGGER.warning(
            "transcript recorder raised, dropping event %s/%s: %s: %s",
            event.kind,
            event.name,
            type(exc).__name__,
            exc,
        )


def safe_tool_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reduce tool arguments to shapes and sizes.

    Keeps the arguments that describe *where* an action happened and replaces
    the ones carrying file content with ``<field>_chars``.
    """

    if not isinstance(args, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, value in args.items():
        name = str(key)
        if name in _SAFE_TOOL_ARG_KEYS:
            safe[name] = value
        elif name in _SIZED_TOOL_ARG_KEYS:
            safe[f"{name}_chars"] = len(value) if isinstance(value, str) else 0
        else:
            safe[f"{name}_present"] = value is not None
    return safe


def _jsonable(value: Any, *, seen: set[int], local_paths: Sequence[str]) -> Any:
    """Normalize to JSON-safe values, redacting strings and guarding cycles.

    The recursion guard tracks ``id()`` of containers currently on the stack.
    Without it a self-referential payload -- which a model response envelope
    can produce -- turns logging into a ``RecursionError``.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(redact(value, local_paths))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return _clip(redact(str(value), local_paths))
    if isinstance(value, Enum):
        return _jsonable(value.value, seen=seen, local_paths=local_paths)

    marker = id(value)
    if marker in seen:
        return "<recursive>"

    if is_dataclass(value) and not isinstance(value, type):
        seen.add(marker)
        try:
            return _jsonable(asdict(value), seen=seen, local_paths=local_paths)
        finally:
            seen.discard(marker)

    if isinstance(value, Mapping):
        seen.add(marker)
        try:
            return {
                str(k): _jsonable(v, seen=seen, local_paths=local_paths)
                for k, v in value.items()
            }
        finally:
            seen.discard(marker)

    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(marker)
        try:
            return [
                _jsonable(item, seen=seen, local_paths=local_paths) for item in value
            ]
        finally:
            seen.discard(marker)

    return _clip(redact(str(value), local_paths))


def _clip(text: str) -> str:
    if len(text) <= _MAX_VALUE_CHARS:
        return text
    return text[:_MAX_VALUE_CHARS] + _TRUNCATION_SUFFIX


class JsonlTranscriptRecorder:
    """Write events as JSONL, one file per ticket run."""

    def __init__(
        self,
        root: str | Path,
        *,
        local_paths: Sequence[str] = (),
        max_open_handles: int = _MAX_OPEN_HANDLES,
        clock: Any = None,
    ) -> None:
        self._root = Path(root)
        self._local_paths = tuple(local_paths)
        self._max_open_handles = max(1, max_open_handles)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._handles: OrderedDict[str, TextIO] = OrderedDict()
        self._write_failures = 0
        self._warned = False

    @property
    def write_failures(self) -> int:
        """Count of dropped events, surfaced by ``report_loops.py``."""

        with self._lock:
            return self._write_failures

    def record(self, event: TranscriptEvent) -> None:
        """Append one event. Never raises."""

        try:
            self._record(event)
        except Exception as exc:  # noqa: BLE001 - observability must not break the run
            self._note_failure(exc)

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:
                    _LOGGER.warning("transcript handle close failed", exc_info=True)
            self._handles.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _record(self, event: TranscriptEvent) -> None:
        # Validate before touching the filesystem: both identifiers are
        # externally supplied and both land in a filename.
        ticket_key = str(event.ticket_key or "")
        if not _TICKET_KEY_RE.match(ticket_key):
            raise ValueError(f"invalid ticket_key for transcript: {ticket_key!r}")

        run_id = event.run_id
        if run_id is not None:
            run_id = str(run_id)[:8]
            if not _RUN_ID_RE.match(run_id):
                raise ValueError(f"invalid run_id for transcript: {run_id!r}")

        line = self._serialize(event, ticket_key=ticket_key, run_id=run_id)

        with self._lock:
            handle = self._handle_for(ticket_key, run_id)
            # One write call per event: a partial line cannot be interleaved
            # with another writer's partial line.
            handle.write(line)
            handle.flush()

    def _serialize(
        self, event: TranscriptEvent, *, ticket_key: str, run_id: str | None
    ) -> str:
        at = event.at or self._clock()
        record = {
            "at": at.isoformat(),
            "ticket_key": ticket_key,
            "run_id": run_id,
            "goal_id": event.goal_id,
            "iteration": event.iteration,
            "phase": event.phase,
            "kind": event.kind,
            "name": event.name,
            "payload": _jsonable(
                event.payload or {}, seen=set(), local_paths=self._local_paths
            ),
        }
        return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"

    def _handle_for(self, ticket_key: str, run_id: str | None) -> TextIO:
        stem = ticket_key if run_id is None else f"{ticket_key}-{run_id}"
        existing = self._handles.get(stem)
        if existing is not None:
            self._handles.move_to_end(stem)
            return existing

        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{stem}.jsonl"
        # Both components are regex-validated above, so this is belt and
        # braces -- but a containment check costs nothing and a traversal
        # would be silent.
        if path.parent.resolve() != self._root.resolve():
            raise ValueError(f"transcript path escapes root: {path}")

        handle = path.open("a", encoding="utf-8")
        self._handles[stem] = handle
        while len(self._handles) > self._max_open_handles:
            _, evicted = self._handles.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                _LOGGER.warning("evicted transcript handle close failed", exc_info=True)
        return handle

    def _note_failure(self, exc: Exception) -> None:
        with self._lock:
            self._write_failures += 1
            should_warn = not self._warned
            self._warned = True
        if should_warn:
            _LOGGER.warning(
                "transcript write failed; further failures counted only: %s: %s",
                type(exc).__name__,
                exc,
            )


__all__ = [
    "JsonlTranscriptRecorder",
    "NullTranscriptRecorder",
    "TranscriptEvent",
    "TranscriptRecorder",
    "safe_record",
    "safe_tool_args",
]

"""Durable loop telemetry: stage timestamps, gate outcomes, iteration cost.

Two design rules, both from specific failure modes:

**First timestamp wins.** Stage writes use ``COALESCE(<col>, excluded.<col>)``
so a re-run or a resumed ticket never overwrites when a stage was first
reached. Without it, "time to first PR" silently becomes "time to last PR".

**Every expected gate is retained, including the ones that never ran.**
Routing short-circuits: when tests fail, build never executes. If only
executed gates are recorded, "build pass rate" quietly means "build pass rate
among tickets whose tests passed", which is a different and much more
flattering number. ``not_run`` is stored like any other status.

Like the transcript recorder, nothing here raises into the caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Protocol

from ticket_agent.sqlite_support import connect, write_transaction

_LOGGER = logging.getLogger(__name__)

_DEFAULT_BUSY_TIMEOUT_MS = 5_000

#: Funnel stages in pipeline order. Each maps to a nullable timestamp column.
STAGES: tuple[str, ...] = (
    "claimed",
    "planned",
    "approved",
    "implemented",
    "committed",
    "verified",
    "reviewed",
    "demoed",
    "pr_opened",
    "merged",
    "escalated",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS ticket_funnel (
    ticket_key TEXT PRIMARY KEY,
    goal_id TEXT,
    {", ".join(f"{stage}_at TEXT" for stage in STAGES)},
    escalation_reason TEXT
);

CREATE TABLE IF NOT EXISTS gate_results (
    ticket_key TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    gate TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_class TEXT,
    exit_code INTEGER,
    timed_out INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (ticket_key, attempt, gate)
);

CREATE TABLE IF NOT EXISTS loop_iterations (
    goal_id TEXT NOT NULL,
    loop TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    strategy_id TEXT,
    outcome TEXT,
    fingerprint TEXT,
    tokens INTEGER,
    cost_usd REAL,
    wall_ms INTEGER,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (goal_id, loop, iteration)
);

CREATE TABLE IF NOT EXISTS observability_counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_gate_results_gate ON gate_results (gate, status);
"""


@dataclass(frozen=True, slots=True)
class GateRecord:
    ticket_key: str
    attempt: int
    gate: str
    status: str
    failure_class: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class IterationRecord:
    goal_id: str
    loop: str
    iteration: int
    strategy_id: str | None = None
    outcome: str | None = None
    fingerprint: str | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    wall_ms: int | None = None


class TelemetryRecorder(Protocol):
    def record_stage(
        self, ticket_key: str, stage: str, *, goal_id: str | None = ...
    ) -> None: ...

    def record_escalation(self, ticket_key: str, reason: str) -> None: ...

    def record_gates(self, records: Iterable[GateRecord]) -> None: ...

    def record_iteration(self, record: IterationRecord) -> None: ...

    def increment(self, name: str, delta: int = ...) -> None: ...


class NullTelemetryRecorder:
    """The default everywhere."""

    __slots__ = ()

    def record_stage(self, ticket_key: str, stage: str, *, goal_id: str | None = None) -> None:
        return None

    def record_escalation(self, ticket_key: str, reason: str) -> None:
        return None

    def record_gates(self, records: Iterable[GateRecord]) -> None:
        return None

    def record_iteration(self, record: IterationRecord) -> None:
        return None

    def increment(self, name: str, delta: int = 1) -> None:
        return None


class SQLiteTelemetryStore:
    """WAL-backed telemetry. Never raises into the caller."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        clock: Any = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._connection = connect(db_path, busy_timeout_ms)
        self._warned = False
        # executescript() implicitly commits any pending transaction, so it
        # must not run inside write_transaction(). The schema is idempotent
        # (CREATE ... IF NOT EXISTS), so it needs no transaction of its own.
        with self._lock:
            self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            try:
                self._connection.close()
            except Exception:
                _LOGGER.warning("telemetry store close failed", exc_info=True)

    # -- writes ------------------------------------------------------------

    def record_stage(
        self, ticket_key: str, stage: str, *, goal_id: str | None = None
    ) -> None:
        if stage not in STAGES:
            self._note_failure(ValueError(f"unknown funnel stage: {stage!r}"))
            return
        column = f"{stage}_at"
        sql = (
            f"INSERT INTO ticket_funnel (ticket_key, goal_id, {column}) "  # noqa: S608 - column name comes from the validated STAGES set, never from input
            "VALUES (?, ?, ?) "
            "ON CONFLICT(ticket_key) DO UPDATE SET "
            # First write wins: a resumed ticket must not reset its clock.
            f"{column} = COALESCE(ticket_funnel.{column}, excluded.{column}), "
            "goal_id = COALESCE(ticket_funnel.goal_id, excluded.goal_id)"
        )
        self._execute(sql, (ticket_key, goal_id, self._now()))

    def record_escalation(self, ticket_key: str, reason: str) -> None:
        self._execute(
            "INSERT INTO ticket_funnel (ticket_key, escalated_at, escalation_reason) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(ticket_key) DO UPDATE SET "
            "escalated_at = COALESCE(ticket_funnel.escalated_at, excluded.escalated_at), "
            "escalation_reason = COALESCE("
            "ticket_funnel.escalation_reason, excluded.escalation_reason)",
            (ticket_key, self._now(), reason),
        )

    def record_gates(self, records: Iterable[GateRecord]) -> None:
        """Write a whole gate chain, including gates that never ran.

        Later attempts overwrite the same ``(ticket, attempt, gate)`` row, so
        a re-verified attempt reports its final outcome rather than its first.
        """

        rows = [
            (
                r.ticket_key,
                r.attempt,
                r.gate,
                r.status,
                r.failure_class,
                r.exit_code,
                1 if r.timed_out else 0,
                r.duration_ms,
                self._now(),
            )
            for r in records
        ]
        if not rows:
            return
        self._executemany(
            "INSERT INTO gate_results (ticket_key, attempt, gate, status, "
            "failure_class, exit_code, timed_out, duration_ms, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(ticket_key, attempt, gate) DO UPDATE SET "
            "status = excluded.status, failure_class = excluded.failure_class, "
            "exit_code = excluded.exit_code, timed_out = excluded.timed_out, "
            "duration_ms = excluded.duration_ms, recorded_at = excluded.recorded_at",
            rows,
        )

    def record_iteration(self, record: IterationRecord) -> None:
        self._execute(
            "INSERT INTO loop_iterations (goal_id, loop, iteration, strategy_id, "
            "outcome, fingerprint, tokens, cost_usd, wall_ms, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(goal_id, loop, iteration) DO UPDATE SET "
            "strategy_id = excluded.strategy_id, outcome = excluded.outcome, "
            "fingerprint = excluded.fingerprint, tokens = excluded.tokens, "
            "cost_usd = excluded.cost_usd, wall_ms = excluded.wall_ms",
            (
                record.goal_id,
                record.loop,
                record.iteration,
                record.strategy_id,
                record.outcome,
                record.fingerprint,
                record.tokens,
                record.cost_usd,
                record.wall_ms,
                self._now(),
            ),
        )

    def increment(self, name: str, delta: int = 1) -> None:
        self._execute(
            "INSERT INTO observability_counters (name, value) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
            (name, delta),
        )

    # -- reads (used by scripts/report_loops.py) ---------------------------

    def funnel_counts(self, *, since: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stage in STAGES:
            sql = f"SELECT COUNT(*) FROM ticket_funnel WHERE {stage}_at IS NOT NULL"  # noqa: S608 - column name comes from the validated STAGES set, never from input
            params: tuple[Any, ...] = ()
            if since:
                sql += f" AND {stage}_at >= ?"
                params = (since,)
            counts[stage] = int(self._scalar(sql, params) or 0)
        return counts

    def gate_counts(self) -> list[Mapping[str, Any]]:
        return self._rows(
            "SELECT gate, status, COUNT(*) AS n FROM gate_results "
            "GROUP BY gate, status ORDER BY gate, status"
        )

    def escalation_reasons(self) -> list[Mapping[str, Any]]:
        return self._rows(
            "SELECT escalation_reason AS reason, COUNT(*) AS n FROM ticket_funnel "
            "WHERE escalation_reason IS NOT NULL GROUP BY reason ORDER BY n DESC"
        )

    def iteration_totals(self) -> list[Mapping[str, Any]]:
        return self._rows(
            "SELECT goal_id, COUNT(*) AS iterations, "
            "SUM(COALESCE(tokens, 0)) AS tokens, "
            "SUM(COALESCE(cost_usd, 0)) AS cost_usd "
            "FROM loop_iterations GROUP BY goal_id ORDER BY goal_id"
        )

    def counters(self) -> dict[str, int]:
        return {
            str(row["name"]): int(row["value"])
            for row in self._rows("SELECT name, value FROM observability_counters")
        }

    # -- internals ---------------------------------------------------------

    def _now(self) -> str:
        return self._clock().isoformat()

    def _execute(self, sql: str, params: Sequence[Any]) -> None:
        try:
            with self._lock, write_transaction(self._connection):
                self._connection.execute(sql, tuple(params))
        except Exception as exc:  # noqa: BLE001 - telemetry must not break the run
            self._note_failure(exc)

    def _executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        try:
            with self._lock, write_transaction(self._connection):
                self._connection.executemany(sql, [tuple(row) for row in rows])
        except Exception as exc:  # noqa: BLE001
            self._note_failure(exc)

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]:
        try:
            with self._lock:
                cursor = self._connection.execute(sql, tuple(params))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:  # noqa: BLE001
            self._note_failure(exc)
            return []

    def _scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        try:
            with self._lock:
                row = self._connection.execute(sql, tuple(params)).fetchone()
                return None if row is None else row[0]
        except Exception as exc:  # noqa: BLE001
            self._note_failure(exc)
            return None

    def _note_failure(self, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            _LOGGER.warning(
                "telemetry write failed; further failures are silent: %s: %s",
                type(exc).__name__,
                exc,
            )


def open_telemetry_store(db_path: str | Path) -> TelemetryRecorder:
    """Open a store, degrading to the no-op recorder if it cannot be created."""

    try:
        return SQLiteTelemetryStore(db_path)
    except sqlite3.Error as exc:
        _LOGGER.warning("telemetry unavailable, continuing without it: %s", exc)
        return NullTelemetryRecorder()


__all__ = [
    "STAGES",
    "GateRecord",
    "IterationRecord",
    "NullTelemetryRecorder",
    "SQLiteTelemetryStore",
    "TelemetryRecorder",
    "open_telemetry_store",
]

"""Durable goal state, action journal, and budget reservations in one DB."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from threading import RLock
from typing import Any

from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.identity import normalize_goal_id
from ticket_agent.goal.types import (
    ActionRecord,
    AutonomyCeiling,
    AutonomyDecision,
    AutonomyMode,
    Budgets,
    EvidenceRef,
    Finding,
    LoopState,
    NextAction,
    StrategyRef,
    action_id,
    canonical_json,
    digest,
)
from ticket_agent.sqlite_support import connect, write_transaction

_DEFAULT_BUSY_TIMEOUT_MS = 5_000


class GoalSpineError(AgentSystemError):
    """Raised when a journal replay conflicts with durable intent."""


class SQLiteGoalSpine:
    """Atomic persistence for loop state, action intent, and model budget."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS goal_loop_state (
        goal_id TEXT PRIMARY KEY,
        contract_version INTEGER NOT NULL,
        phase TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        state_digest TEXT NOT NULL,
        payload TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS action_records (
        action_id TEXT PRIMARY KEY,
        goal_id TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        kind TEXT NOT NULL,
        operation TEXT NOT NULL,
        natural_key TEXT NOT NULL,
        request_digest TEXT NOT NULL,
        state TEXT NOT NULL,
        external INTEGER NOT NULL,
        attempts INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        lease_owner TEXT,
        lease_expires_at TEXT,
        result_ref TEXT,
        result_identity TEXT,
        reserved_model_cost_usd REAL NOT NULL,
        actual_model_cost_usd REAL NOT NULL,
        error TEXT,
        error_classification TEXT,
        recovery_classification TEXT,
        UNIQUE (goal_id, operation, natural_key)
    );

    CREATE INDEX IF NOT EXISTS idx_action_records_goal_state
        ON action_records (goal_id, state);

    CREATE TABLE IF NOT EXISTS budget_reservations (
        action_id TEXT PRIMARY KEY,
        goal_id TEXT NOT NULL,
        reserved_model_cost_usd REAL NOT NULL,
        actual_model_cost_usd REAL NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS autonomy_decisions (
        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        effective_mode TEXT NOT NULL,
        decision_digest TEXT NOT NULL UNIQUE,
        payload TEXT NOT NULL,
        decided_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_autonomy_decisions_goal
        ON autonomy_decisions (goal_id, contract_version, decision_id);
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        clock: Any = None,
    ) -> None:
        self._lock = RLock()
        self._connection = connect(db_path, busy_timeout_ms)
        self._clock = clock or (lambda: datetime.now(UTC))
        with self._lock:
            self._connection.executescript(self._SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_loop_state(self, state: LoopState) -> None:
        normalize_goal_id(state.goal_id)
        now = self._clock()
        with self._lock, write_transaction(self._connection):
            self._write_loop_state(state, now)

    def load_loop_state(self, goal_id: str) -> LoopState | None:
        canonical_goal_id = normalize_goal_id(goal_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM goal_loop_state WHERE goal_id = ?",
                (canonical_goal_id,),
            ).fetchone()
        if row is None:
            return None
        return _loop_state_from_payload(json.loads(row["payload"]))

    def reserve_action(
        self,
        state: LoopState,
        action: ActionRecord,
    ) -> ActionRecord:
        """Atomically persist a state transition and its action reservation."""

        canonical_goal_id = normalize_goal_id(state.goal_id)
        if normalize_goal_id(action.goal_id) != canonical_goal_id:
            raise GoalSpineError("loop state and action must name the same goal")
        if not action.operation or not action.natural_key or not action.request_digest:
            raise GoalSpineError(
                "action operation, natural key, and request digest are required"
            )
        expected_id = action_id(
            canonical_goal_id,
            action.iteration,
            action.operation,
            action.natural_key,
        )
        if action.action_id != expected_id:
            raise GoalSpineError("action id is not deterministic for its natural key")

        now = self._clock()
        with self._lock, write_transaction(self._connection):
            existing = self._connection.execute(
                "SELECT * FROM action_records WHERE action_id = ?",
                (action.action_id,),
            ).fetchone()
            if existing is not None:
                loaded = _action_from_row(existing)
                if (
                    loaded.goal_id != canonical_goal_id
                    or loaded.operation != action.operation
                    or loaded.natural_key != action.natural_key
                    or loaded.request_digest != action.request_digest
                ):
                    raise GoalSpineError(
                        f"action replay conflicts with durable intent: {action.action_id}"
                    )
                return loaded

            durable = replace(
                action,
                created_at=action.created_at or now,
                updated_at=now,
            )
            self._write_loop_state(state, now)
            self._insert_action(durable)
            self._connection.execute(
                "INSERT INTO budget_reservations "
                "(action_id, goal_id, reserved_model_cost_usd, "
                "actual_model_cost_usd, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    durable.action_id,
                    durable.goal_id,
                    durable.reserved_model_cost_usd,
                    durable.actual_model_cost_usd,
                    _datetime_text(durable.created_at),
                    _datetime_text(durable.updated_at),
                ),
            )
            return durable

    def mark_in_flight(
        self,
        action_id_value: str,
        *,
        lease_owner: str,
        lease_seconds: int = 300,
        recovery_classification: str | None = None,
    ) -> ActionRecord:
        if not lease_owner:
            raise GoalSpineError("an in-flight action requires a lease owner")
        if lease_seconds <= 0:
            raise GoalSpineError("action lease_seconds must be positive")
        now = self._clock()
        lease_expires = now + timedelta(seconds=lease_seconds)
        with self._lock, write_transaction(self._connection):
            self._require_action(action_id_value)
            self._connection.execute(
                "UPDATE action_records SET state = 'in_flight', "
                "attempts = attempts + 1, started_at = COALESCE(started_at, ?), "
                "updated_at = ?, lease_owner = ?, lease_expires_at = ?, "
                "recovery_classification = COALESCE(?, recovery_classification) "
                "WHERE action_id = ? AND state != 'done'",
                (
                    now.isoformat(),
                    now.isoformat(),
                    lease_owner,
                    lease_expires.isoformat(),
                    recovery_classification,
                    action_id_value,
                ),
            )
            return self._load_action_locked(action_id_value)

    def mark_done(
        self,
        action_id_value: str,
        *,
        result_identity: str | None = None,
        actual_model_cost_usd: float = 0.0,
        recovery_classification: str | None = None,
    ) -> ActionRecord:
        now = self._clock()
        with self._lock, write_transaction(self._connection):
            self._require_action(action_id_value)
            self._connection.execute(
                "UPDATE action_records SET state = 'done', completed_at = ?, "
                "updated_at = ?, lease_owner = NULL, lease_expires_at = NULL, "
                "result_identity = ?, actual_model_cost_usd = ?, "
                "recovery_classification = COALESCE(?, recovery_classification), "
                "error = NULL, error_classification = NULL WHERE action_id = ?",
                (
                    now.isoformat(),
                    now.isoformat(),
                    result_identity,
                    actual_model_cost_usd,
                    recovery_classification,
                    action_id_value,
                ),
            )
            self._connection.execute(
                "UPDATE budget_reservations SET actual_model_cost_usd = ?, "
                "updated_at = ? WHERE action_id = ?",
                (actual_model_cost_usd, now.isoformat(), action_id_value),
            )
            return self._load_action_locked(action_id_value)

    def mark_failed(
        self,
        action_id_value: str,
        *,
        error: str,
        error_classification: str,
        recovery_classification: str,
        actual_model_cost_usd: float | None = None,
    ) -> ActionRecord:
        now = self._clock()
        with self._lock, write_transaction(self._connection):
            self._require_action(action_id_value)
            self._connection.execute(
                "UPDATE action_records SET state = 'failed', updated_at = ?, "
                "lease_owner = NULL, lease_expires_at = NULL, error = ?, "
                "error_classification = ?, recovery_classification = ?, "
                "actual_model_cost_usd = COALESCE(?, actual_model_cost_usd) "
                "WHERE action_id = ? AND state != 'done'",
                (
                    now.isoformat(),
                    error,
                    error_classification,
                    recovery_classification,
                    actual_model_cost_usd,
                    action_id_value,
                ),
            )
            if actual_model_cost_usd is not None:
                self._connection.execute(
                    "UPDATE budget_reservations SET actual_model_cost_usd = ?, "
                    "updated_at = ? WHERE action_id = ?",
                    (actual_model_cost_usd, now.isoformat(), action_id_value),
                )
            return self._load_action_locked(action_id_value)

    def load_action(self, action_id_value: str) -> ActionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM action_records WHERE action_id = ?",
                (action_id_value,),
            ).fetchone()
        return None if row is None else _action_from_row(row)

    def actions_for_goal(self, goal_id: str) -> tuple[ActionRecord, ...]:
        canonical_goal_id = normalize_goal_id(goal_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM action_records WHERE goal_id = ? "
                "ORDER BY created_at, action_id",
                (canonical_goal_id,),
            ).fetchall()
        return tuple(_action_from_row(row) for row in rows)

    def budget_for_action(self, action_id_value: str) -> tuple[float, float] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT reserved_model_cost_usd, actual_model_cost_usd "
                "FROM budget_reservations WHERE action_id = ?",
                (action_id_value,),
            ).fetchone()
        if row is None:
            return None
        return (
            float(row["reserved_model_cost_usd"]),
            float(row["actual_model_cost_usd"]),
        )

    def save_autonomy_decision(
        self,
        decision: AutonomyDecision,
    ) -> AutonomyDecision:
        normalize_goal_id(decision.goal_id)
        now = decision.decided_at or self._clock()
        durable = replace(decision, decided_at=now)
        with self._lock, write_transaction(self._connection):
            existing = self._connection.execute(
                "SELECT payload FROM autonomy_decisions WHERE decision_digest = ?",
                (durable.decision_digest,),
            ).fetchone()
            if existing is not None:
                return _autonomy_decision_from_payload(json.loads(existing["payload"]))
            self._connection.execute(
                "INSERT INTO autonomy_decisions (goal_id, contract_version, "
                "effective_mode, decision_digest, payload, decided_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    durable.goal_id,
                    durable.contract_version,
                    str(durable.effective_mode),
                    durable.decision_digest,
                    canonical_json(durable),
                    now.isoformat(),
                ),
            )
        return durable

    def latest_autonomy_decision(
        self,
        goal_id: str,
        *,
        contract_version: int | None = None,
    ) -> AutonomyDecision | None:
        goal_id = normalize_goal_id(goal_id)
        with self._lock:
            if contract_version is None:
                row = self._connection.execute(
                    "SELECT payload FROM autonomy_decisions WHERE goal_id = ? "
                    "ORDER BY decision_id DESC LIMIT 1",
                    (goal_id,),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT payload FROM autonomy_decisions WHERE goal_id = ? "
                    "AND contract_version = ? ORDER BY decision_id DESC LIMIT 1",
                    (goal_id, contract_version),
                ).fetchone()
        if row is None:
            return None
        return _autonomy_decision_from_payload(json.loads(row["payload"]))

    def _write_loop_state(self, state: LoopState, now: datetime) -> None:
        payload = canonical_json(state)
        self._connection.execute(
            "INSERT INTO goal_loop_state "
            "(goal_id, contract_version, phase, iteration, state_digest, "
            "payload, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(goal_id) DO UPDATE SET "
            "contract_version = excluded.contract_version, "
            "phase = excluded.phase, iteration = excluded.iteration, "
            "state_digest = excluded.state_digest, payload = excluded.payload, "
            "updated_at = excluded.updated_at",
            (
                state.goal_id,
                state.contract_version,
                state.phase,
                state.iteration,
                digest(state),
                payload,
                now.isoformat(),
            ),
        )

    def _insert_action(self, action: ActionRecord) -> None:
        self._connection.execute(
            "INSERT INTO action_records ("
            "action_id, goal_id, iteration, kind, operation, natural_key, "
            "request_digest, state, external, attempts, created_at, updated_at, "
            "started_at, completed_at, lease_owner, lease_expires_at, result_ref, "
            "result_identity, reserved_model_cost_usd, actual_model_cost_usd, "
            "error, error_classification, recovery_classification"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action.action_id,
                action.goal_id,
                action.iteration,
                action.kind,
                action.operation,
                action.natural_key,
                action.request_digest,
                action.state,
                int(action.external),
                action.attempts,
                _datetime_text(action.created_at),
                _datetime_text(action.updated_at),
                _datetime_text(action.started_at),
                _datetime_text(action.completed_at),
                action.lease_owner,
                _datetime_text(action.lease_expires_at),
                canonical_json(action.result_ref) if action.result_ref else None,
                action.result_identity,
                action.reserved_model_cost_usd,
                action.actual_model_cost_usd,
                action.error,
                action.error_classification,
                action.recovery_classification,
            ),
        )

    def _require_action(self, action_id_value: str) -> None:
        row = self._connection.execute(
            "SELECT 1 FROM action_records WHERE action_id = ?",
            (action_id_value,),
        ).fetchone()
        if row is None:
            raise GoalSpineError(f"unknown action: {action_id_value}")

    def _load_action_locked(self, action_id_value: str) -> ActionRecord:
        row = self._connection.execute(
            "SELECT * FROM action_records WHERE action_id = ?",
            (action_id_value,),
        ).fetchone()
        if row is None:
            raise GoalSpineError(f"unknown action: {action_id_value}")
        return _action_from_row(row)


def _action_from_row(row: Any) -> ActionRecord:
    result_ref = None
    if row["result_ref"]:
        result_ref = _evidence_from_payload(json.loads(row["result_ref"]))
    return ActionRecord(
        action_id=row["action_id"],
        goal_id=row["goal_id"],
        iteration=int(row["iteration"]),
        kind=row["kind"],
        state=row["state"],
        operation=row["operation"],
        natural_key=row["natural_key"],
        request_digest=row["request_digest"],
        external=bool(row["external"]),
        attempts=int(row["attempts"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
        started_at=_parse_datetime(row["started_at"]),
        completed_at=_parse_datetime(row["completed_at"]),
        lease_owner=row["lease_owner"],
        lease_expires_at=_parse_datetime(row["lease_expires_at"]),
        result_ref=result_ref,
        result_identity=row["result_identity"],
        reserved_model_cost_usd=float(row["reserved_model_cost_usd"]),
        actual_model_cost_usd=float(row["actual_model_cost_usd"]),
        error=row["error"],
        error_classification=row["error_classification"],
        recovery_classification=row["recovery_classification"],
    )


def _loop_state_from_payload(payload: dict[str, Any]) -> LoopState:
    return LoopState(
        goal_id=payload["goal_id"],
        contract_version=int(payload["contract_version"]),
        phase=payload["phase"],
        iteration=int(payload.get("iteration", 0)),
        strategy=(
            StrategyRef(**payload["strategy"]) if payload.get("strategy") else None
        ),
        hypothesis=payload.get("hypothesis", ""),
        candidate_sha=payload.get("candidate_sha"),
        evidence_refs=tuple(
            _evidence_from_payload(item) for item in payload.get("evidence_refs", ())
        ),
        verification_findings=tuple(
            _finding_from_payload(item)
            for item in payload.get("verification_findings", ())
        ),
        review_findings=tuple(
            _finding_from_payload(item) for item in payload.get("review_findings", ())
        ),
        assumptions=tuple(payload.get("assumptions", ())),
        discovered_constraints=tuple(payload.get("discovered_constraints", ())),
        failure_fingerprints=tuple(payload.get("failure_fingerprints", ())),
        no_progress_count=int(payload.get("no_progress_count", 0)),
        consumed=Budgets(**payload.get("consumed", {})),
        next_action=(
            NextAction(**payload["next_action"])
            if payload.get("next_action")
            else None
        ),
        terminal_status=payload.get("terminal_status"),
    )


def _autonomy_decision_from_payload(payload: dict[str, Any]) -> AutonomyDecision:
    return AutonomyDecision(
        goal_id=payload["goal_id"],
        contract_version=int(payload["contract_version"]),
        effective_mode=AutonomyMode(int(payload["effective_mode"])),
        ceilings=tuple(
            AutonomyCeiling(
                source=item["source"],
                mode=AutonomyMode(int(item["mode"])),
                detail=item.get("detail", ""),
            )
            for item in payload.get("ceilings", ())
        ),
        required_gates=tuple(payload.get("required_gates", ())),
        enforced_gate_names=tuple(payload.get("enforced_gate_names", ())),
        decided_at=_parse_datetime(payload.get("decided_at")),
        # A payload written before the candidate-evidence ceiling existed has
        # no version. It reads as 0 -- legacy -- rather than being promoted to
        # the current one by omission.
        schema_version=int(payload.get("schema_version", 0)),
    )


def _evidence_from_payload(payload: dict[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        kind=payload["kind"],
        sha256=payload["sha256"],
        uri=payload["uri"],
        produced_at=_require_datetime(payload["produced_at"], "produced_at"),
        produced_by=payload["produced_by"],
        candidate_sha=payload.get("candidate_sha"),
    )


def _finding_from_payload(payload: dict[str, Any]) -> Finding:
    evidence = payload.get("evidence_ref")
    return Finding(
        finding_id=payload["finding_id"],
        severity=payload["severity"],
        claim=payload["claim"],
        file=payload.get("file"),
        line=payload.get("line"),
        evidence_ref=_evidence_from_payload(evidence) if evidence else None,
        suggested_action=payload.get("suggested_action"),
    )


def _datetime_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    return None if not value else datetime.fromisoformat(value)


def _require_datetime(value: str | None, field: str) -> datetime:
    """EvidenceRef declares produced_at non-optional; a row without one is
    malformed, and passing None through would only move the failure later."""

    parsed = _parse_datetime(value)
    if parsed is None:
        raise GoalSpineError(f"evidence row is missing {field}")
    return parsed


__all__ = ["GoalSpineError", "SQLiteGoalSpine"]

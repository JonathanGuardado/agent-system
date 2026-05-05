"""Slack-driven execution approval before implementation starts."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol

from langgraph.types import Command, interrupt

from ticket_agent.orchestrator.services import ApprovalDecision
from ticket_agent.orchestrator.state import TicketState


ApprovalStatus = Literal["pending", "approved", "rejected", "expired"]
_COMMAND_RE = re.compile(
    r"^\s*(approve|reject)\s+([A-Z][A-Z0-9]*-\d+)\s*$",
    re.I,
)
_DEFAULT_TIMEOUT = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class ExecutionApproval:
    ticket_key: str
    slack_channel: str
    slack_thread_ts: str
    plan_summary: str
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class PendingApprovalResult:
    approval: ExecutionApproval
    created: bool


@dataclass(frozen=True, slots=True)
class ExecutionApprovalCommandResult:
    action: Literal["approve", "reject", "expire"]
    ticket_key: str
    approval: ExecutionApproval
    graph_result: Any


class ResumableGraph(Protocol):
    def ainvoke(
        self,
        graph_input: Any,
        config: Mapping[str, Any],
    ) -> Awaitable[Any]:
        """Resume a checkpointed graph thread."""


class SlackPoster(Protocol):
    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        """Post a plain-text reply to a Slack thread."""


class SQLiteExecutionApprovalStore:
    """Persist pending execution approvals in SQLite."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or _utcnow
        self._conn_lock = RLock()
        self._conn = _connect(db_path, busy_timeout_ms)
        self._initialize()

    def close(self) -> None:
        with self._conn_lock:
            self._conn.close()

    def get(self, ticket_key: str) -> ExecutionApproval | None:
        with self._conn_lock:
            row = self._conn.execute(
                """
                SELECT ticket_key, slack_channel, slack_thread_ts, plan_summary,
                       status, created_at, expires_at
                FROM execution_approvals
                WHERE ticket_key = ?
                """,
                (ticket_key,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_approval(row)

    def create_pending(
        self,
        *,
        ticket_key: str,
        slack_channel: str,
        slack_thread_ts: str,
        plan_summary: str,
        timeout: timedelta = _DEFAULT_TIMEOUT,
    ) -> ExecutionApproval:
        """Create a fresh pending approval for a ticket."""

        now = _ensure_aware(self._clock())
        approval = _pending_approval(
            ticket_key=ticket_key,
            slack_channel=slack_channel,
            slack_thread_ts=slack_thread_ts,
            plan_summary=plan_summary,
            now=now,
            timeout=timeout,
        )
        with self._conn_lock, _write_transaction(self._conn):
            self._upsert_approval_locked(approval)
        return approval

    def ensure_pending(
        self,
        *,
        ticket_key: str,
        slack_channel: str,
        slack_thread_ts: str,
        plan_summary: str,
        timeout: timedelta = _DEFAULT_TIMEOUT,
    ) -> PendingApprovalResult:
        now = _ensure_aware(self._clock())

        with self._conn_lock, _write_transaction(self._conn):
            row = self._conn.execute(
                """
                SELECT ticket_key, slack_channel, slack_thread_ts, plan_summary,
                       status, created_at, expires_at
                FROM execution_approvals
                WHERE ticket_key = ?
                """,
                (ticket_key,),
            ).fetchone()
            if row is not None:
                approval = _row_to_approval(row)
                if approval.status == "pending":
                    if approval.expires_at <= now:
                        approval = self._mark_status_locked(ticket_key, "expired")
                    return PendingApprovalResult(approval=approval, created=False)

                approval = _pending_approval(
                    ticket_key=ticket_key,
                    slack_channel=slack_channel,
                    slack_thread_ts=slack_thread_ts,
                    plan_summary=plan_summary,
                    now=now,
                    timeout=timeout,
                )
                self._upsert_approval_locked(approval)
                return PendingApprovalResult(approval=approval, created=True)

            approval = _pending_approval(
                ticket_key=ticket_key,
                slack_channel=slack_channel,
                slack_thread_ts=slack_thread_ts,
                plan_summary=plan_summary,
                now=now,
                timeout=timeout,
            )
            self._conn.execute(
                """
                INSERT INTO execution_approvals (
                    ticket_key, slack_channel, slack_thread_ts, plan_summary,
                    status, created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                _approval_values(approval),
            )
        return PendingApprovalResult(approval=approval, created=True)

    def approve(self, ticket_key: str) -> ExecutionApproval | None:
        return self.mark_approved(ticket_key)

    def mark_approved(self, ticket_key: str) -> ExecutionApproval | None:
        approval = self._mark_status(ticket_key, "approved", require_pending=True)
        if approval is not None and approval.status == "approved":
            return approval
        return None

    def reject(self, ticket_key: str) -> ExecutionApproval | None:
        return self.mark_rejected(ticket_key)

    def mark_rejected(self, ticket_key: str) -> ExecutionApproval | None:
        approval = self._mark_status(ticket_key, "rejected", require_pending=True)
        if approval is not None and approval.status == "rejected":
            return approval
        return None

    def is_approved(self, ticket_key: str) -> bool:
        approval = self.get(ticket_key)
        return approval is not None and approval.status == "approved"

    def mark_expired(self, ticket_key: str) -> ExecutionApproval | None:
        approval = self._mark_status(ticket_key, "expired", require_pending=True)
        if approval is not None and approval.status == "expired":
            return approval
        return None

    def expire_pending(self, now: datetime) -> int:
        now = _ensure_aware(now)
        with self._conn_lock, _write_transaction(self._conn):
            cursor = self._conn.execute(
                """
                UPDATE execution_approvals
                SET status = 'expired'
                WHERE status = 'pending' AND expires_at <= ?
                """,
                (_datetime_text(now),),
            )
        return cursor.rowcount

    def expire_due(self) -> int:
        return self.expire_pending(self._clock())

    def _mark_status(
        self,
        ticket_key: str,
        status: ApprovalStatus,
        *,
        require_pending: bool = False,
    ) -> ExecutionApproval | None:
        now = _ensure_aware(self._clock())
        with self._conn_lock, _write_transaction(self._conn):
            row = self._conn.execute(
                """
                SELECT ticket_key, slack_channel, slack_thread_ts, plan_summary,
                       status, created_at, expires_at
                FROM execution_approvals
                WHERE ticket_key = ?
                """,
                (ticket_key,),
            ).fetchone()
            if row is None:
                return None
            approval = _row_to_approval(row)
            if approval.status == "pending" and approval.expires_at <= now:
                return self._mark_status_locked(ticket_key, "expired")
            if require_pending and approval.status != "pending":
                return approval
            return self._mark_status_locked(ticket_key, status)

    def _mark_status_locked(
        self,
        ticket_key: str,
        status: ApprovalStatus,
    ) -> ExecutionApproval:
        self._conn.execute(
            "UPDATE execution_approvals SET status = ? WHERE ticket_key = ?",
            (status, ticket_key),
        )
        row = self._conn.execute(
            """
            SELECT ticket_key, slack_channel, slack_thread_ts, plan_summary,
                   status, created_at, expires_at
            FROM execution_approvals
            WHERE ticket_key = ?
            """,
            (ticket_key,),
        ).fetchone()
        assert row is not None
        return _row_to_approval(row)

    def _upsert_approval_locked(self, approval: ExecutionApproval) -> None:
        self._conn.execute(
            """
            INSERT INTO execution_approvals (
                ticket_key, slack_channel, slack_thread_ts, plan_summary,
                status, created_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_key) DO UPDATE SET
                slack_channel = excluded.slack_channel,
                slack_thread_ts = excluded.slack_thread_ts,
                plan_summary = excluded.plan_summary,
                status = excluded.status,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            _approval_values(approval),
        )

    def _initialize(self) -> None:
        with self._conn_lock, _write_transaction(self._conn):
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_approvals (
                    ticket_key TEXT PRIMARY KEY,
                    slack_channel TEXT NOT NULL,
                    slack_thread_ts TEXT NOT NULL,
                    plan_summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_execution_approvals_status
                ON execution_approvals (status, expires_at)
                """
            )


class SlackExecutionApprovalService:
    """Approval service that posts to Slack and interrupts the graph."""

    def __init__(
        self,
        *,
        store: SQLiteExecutionApprovalStore,
        slack: SlackPoster,
        default_channel: str,
        default_thread_ts: str | None = None,
        timeout: timedelta = _DEFAULT_TIMEOUT,
        poster_user_id: str = "execution-approval",
    ) -> None:
        self._store = store
        self._slack = slack
        self._default_channel = default_channel
        self._default_thread_ts = default_thread_ts
        self._timeout = timeout
        self._poster_user_id = poster_user_id

    async def request_approval(self, state: TicketState) -> ApprovalDecision:
        pending = self._store.ensure_pending(
            ticket_key=state.ticket_key,
            slack_channel=state.slack_channel or self._default_channel,
            slack_thread_ts=state.slack_thread_ts
            or self._default_thread_ts
            or state.ticket_key,
            plan_summary=_plan_summary(state),
            timeout=self._timeout,
        )
        approval = pending.approval
        if approval.status != "pending":
            return _decision_for_status(approval.status)

        if pending.created:
            await self._slack.post_thread_reply(
                approval.slack_channel,
                approval.slack_thread_ts,
                self._poster_user_id,
                _format_approval_message(approval, state),
            )

        resume_value = interrupt(
            {
                "ticket_key": approval.ticket_key,
                "status": approval.status,
                "plan_summary": approval.plan_summary,
                "expires_at": _datetime_text(approval.expires_at),
            }
        )
        resumed_status = _resume_status(resume_value)
        if resumed_status == "approved":
            marked = self._store.mark_approved(approval.ticket_key)
            return _decision_for_status(
                marked.status if marked else _stored_status(self._store, approval)
            )
        if resumed_status == "rejected":
            marked = self._store.mark_rejected(approval.ticket_key)
            return _decision_for_status(
                marked.status if marked else _stored_status(self._store, approval)
            )
        marked = self._store.mark_expired(approval.ticket_key)
        return _decision_for_status(marked.status if marked else "expired")


class ExecutionApprovalCommandHandler:
    """Handle plain-text Slack execution approval commands."""

    def __init__(
        self,
        *,
        store: SQLiteExecutionApprovalStore,
        graph: ResumableGraph,
        slack: SlackPoster | None = None,
        poster_user_id: str = "execution-approval",
    ) -> None:
        self._store = store
        self._graph = graph
        self._slack = slack
        self._poster_user_id = poster_user_id

    def matches(self, text: str) -> bool:
        return _parse_command(text) is not None

    async def handle_message(
        self,
        *,
        text: str,
        channel: str | None = None,
        thread_ts: str | None = None,
        user_id: str | None = None,
    ) -> ExecutionApprovalCommandResult | None:
        del user_id
        command = _parse_command(text)
        if command is None:
            return None

        action, ticket_key = command
        approval = (
            self._store.mark_approved(ticket_key)
            if action == "approve"
            else self._store.mark_rejected(ticket_key)
        )
        if approval is None:
            return None

        graph_result = await self._resume(ticket_key, action)
        await self._post_ack(
            approval,
            action,
            channel=channel,
            thread_ts=thread_ts,
        )
        return ExecutionApprovalCommandResult(
            action=action,
            ticket_key=ticket_key,
            approval=approval,
            graph_result=graph_result,
        )

    async def expire_pending(
        self,
        ticket_key: str,
    ) -> ExecutionApprovalCommandResult | None:
        approval = self._store.mark_expired(ticket_key)
        if approval is None:
            return None
        graph_result = await self._resume(ticket_key, "expire")
        return ExecutionApprovalCommandResult(
            action="expire",
            ticket_key=ticket_key,
            approval=approval,
            graph_result=graph_result,
        )

    async def _resume(self, ticket_key: str, action: str) -> Any:
        decision = "approved" if action == "approve" else action
        return await self._graph.ainvoke(
            Command(resume={"decision": decision}),
            config={"configurable": {"thread_id": ticket_key}},
        )

    async def _post_ack(
        self,
        approval: ExecutionApproval,
        action: str,
        *,
        channel: str | None,
        thread_ts: str | None,
    ) -> None:
        if self._slack is None:
            return
        verb = "approved" if action == "approve" else "rejected"
        await self._slack.post_thread_reply(
            channel or approval.slack_channel,
            thread_ts or approval.slack_thread_ts,
            self._poster_user_id,
            f"Execution {verb} for {approval.ticket_key}.",
        )


def is_execution_approval_command(text: str) -> bool:
    return _parse_command(text) is not None


def _parse_command(text: str) -> tuple[Literal["approve", "reject"], str] | None:
    match = _COMMAND_RE.match(text)
    if match is None:
        return None
    action = match.group(1).lower()
    ticket_key = match.group(2).upper()
    return ("approve" if action == "approve" else "reject", ticket_key)


def _decision_for_status(status: ApprovalStatus) -> ApprovalDecision:
    if status == "approved":
        return ApprovalDecision(approved=True, status="approved")
    if status == "expired":
        return ApprovalDecision(
            approved=False,
            status="expired",
            reason="execution approval expired",
        )
    return ApprovalDecision(
        approved=False,
        status="rejected",
        reason="execution approval rejected",
    )


def _stored_status(
    store: SQLiteExecutionApprovalStore,
    approval: ExecutionApproval,
) -> ApprovalStatus:
    stored = store.get(approval.ticket_key)
    if stored is None:
        return "expired"
    return stored.status


def _resume_status(value: Any) -> ApprovalStatus:
    if isinstance(value, Mapping):
        raw = value.get("decision") or value.get("status")
    else:
        raw = value
    normalized = str(raw or "").strip().lower()
    if normalized in {"approve", "approved", "yes"}:
        return "approved"
    if normalized in {"reject", "rejected", "no"}:
        return "rejected"
    return "expired"


def _plan_summary(state: TicketState) -> str:
    if state.decomposition:
        summary = state.decomposition.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        plan = state.decomposition.get("plan")
        if isinstance(plan, str) and plan.strip():
            return plan.strip()
        try:
            return json.dumps(state.decomposition, sort_keys=True)
        except TypeError:
            return str(state.decomposition)
    if state.description.strip():
        return state.description.strip()
    return state.summary


def _format_approval_message(
    approval: ExecutionApproval,
    state: TicketState | None = None,
) -> str:
    lines = [
        f"Execution approval requested for {approval.ticket_key}.",
        "",
        "Plan:",
        approval.plan_summary,
    ]
    details = _plan_details(state)
    if details:
        lines.extend(["", *details])
    lines.extend(
        [
            "",
            "Command examples:",
            f"approve {approval.ticket_key}",
            f"reject {approval.ticket_key}",
        ]
    )
    return "\n".join(lines)


def _plan_details(state: TicketState | None) -> list[str]:
    if state is None or not isinstance(state.decomposition, Mapping):
        return []

    details: list[str] = []
    files = _string_list(state.decomposition.get("files_to_modify"))
    if files:
        details.append("Files:")
        details.extend(f"- {path}" for path in files)

    risks = _string_list(state.decomposition.get("risks"))
    if risks:
        details.append("Risks:")
        details.extend(f"- {risk}" for risk in risks)
    return details


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _connect(db_path: str | Path, busy_timeout_ms: int) -> sqlite3.Connection:
    path = Path(db_path)
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def _write_transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _row_to_approval(row: sqlite3.Row) -> ExecutionApproval:
    return ExecutionApproval(
        ticket_key=str(row["ticket_key"]),
        slack_channel=str(row["slack_channel"]),
        slack_thread_ts=str(row["slack_thread_ts"]),
        plan_summary=str(row["plan_summary"]),
        status=_coerce_status(row["status"]),
        created_at=_datetime_from_text(str(row["created_at"])),
        expires_at=_datetime_from_text(str(row["expires_at"])),
    )


def _approval_values(approval: ExecutionApproval) -> tuple[str, ...]:
    return (
        approval.ticket_key,
        approval.slack_channel,
        approval.slack_thread_ts,
        approval.plan_summary,
        approval.status,
        _datetime_text(approval.created_at),
        _datetime_text(approval.expires_at),
    )


def _pending_approval(
    *,
    ticket_key: str,
    slack_channel: str,
    slack_thread_ts: str,
    plan_summary: str,
    now: datetime,
    timeout: timedelta,
) -> ExecutionApproval:
    return ExecutionApproval(
        ticket_key=ticket_key,
        slack_channel=slack_channel,
        slack_thread_ts=slack_thread_ts,
        plan_summary=plan_summary,
        status="pending",
        created_at=now,
        expires_at=now + timeout,
    )


def _coerce_status(value: object) -> ApprovalStatus:
    status = str(value)
    if status == "pending":
        return "pending"
    if status == "approved":
        return "approved"
    if status == "rejected":
        return "rejected"
    if status == "expired":
        return "expired"
    raise ValueError(f"unknown execution approval status: {status}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_text(value: datetime) -> str:
    return _ensure_aware(value).isoformat()


def _datetime_from_text(value: str) -> datetime:
    return _ensure_aware(datetime.fromisoformat(value))


__all__ = [
    "ExecutionApproval",
    "ExecutionApprovalCommandHandler",
    "ExecutionApprovalCommandResult",
    "PendingApprovalResult",
    "SQLiteExecutionApprovalStore",
    "SlackExecutionApprovalService",
    "is_execution_approval_command",
]

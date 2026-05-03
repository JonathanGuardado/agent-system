"""SQLite-backed store for active intake proposals."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from ticket_agent.domain.intake import Proposal, ProposalStatus


Clock = Callable[[], datetime]


PROPOSAL_TTL_SECONDS = 24 * 60 * 60


class ProposalStore:
    """Persist intake proposals across the Slack approval lifecycle."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        clock: Clock | None = None,
    ) -> None:
        self._connection_lock = RLock()
        self._connection = _connect(db_path, busy_timeout_ms)
        self._clock = clock or _utcnow
        self._initialize()

    def close(self) -> None:
        with self._connection_lock:
            self._connection.close()

    def save(self, proposal: Proposal) -> None:
        """Insert a new proposal row."""

        row = _proposal_to_row(proposal)
        with self._connection_lock, _write_transaction(self._connection):
            self._connection.execute(
                """
                INSERT INTO active_proposals (
                    proposal_id,
                    slack_user_id,
                    slack_thread_ts,
                    mode,
                    jira_project_key,
                    jira_epic_key,
                    status,
                    proposal_json,
                    revision_count,
                    created_at,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def update(self, proposal: Proposal) -> None:
        """Replace an existing proposal row."""

        row = _proposal_to_row(proposal)
        with self._connection_lock, _write_transaction(self._connection):
            cursor = self._connection.execute(
                """
                UPDATE active_proposals
                SET slack_user_id = ?,
                    slack_thread_ts = ?,
                    mode = ?,
                    jira_project_key = ?,
                    jira_epic_key = ?,
                    status = ?,
                    proposal_json = ?,
                    revision_count = ?,
                    created_at = ?,
                    expires_at = ?
                WHERE proposal_id = ?
                """,
                (*row[1:], row[0]),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"proposal not found: {proposal.proposal_id}")

    def mark_status(
        self,
        proposal_id: str,
        status: ProposalStatus,
    ) -> None:
        """Update the status of a proposal in the column and the JSON snapshot."""

        with self._connection_lock, _write_transaction(self._connection):
            row = self._connection.execute(
                "SELECT proposal_json FROM active_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"proposal not found: {proposal_id}")

            proposal = Proposal.model_validate_json(row["proposal_json"])
            updated = proposal.model_copy(update={"status": status})
            self._connection.execute(
                """
                UPDATE active_proposals
                SET status = ?, proposal_json = ?
                WHERE proposal_id = ?
                """,
                (
                    status.value,
                    json.dumps(updated.model_dump(mode="json")),
                    proposal_id,
                ),
            )

    def get(self, proposal_id: str) -> Proposal | None:
        with self._connection_lock:
            row = self._connection.execute(
                "SELECT proposal_json FROM active_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        return Proposal.model_validate_json(row["proposal_json"])

    def get_active_for_thread(
        self,
        slack_user_id: str,
        slack_thread_ts: str,
    ) -> Proposal | None:
        """Return the most recent non-terminal proposal for a Slack thread."""

        active_statuses = (
            ProposalStatus.DRAFTING.value,
            ProposalStatus.AWAITING_CONFIRMATION.value,
        )
        placeholders = ",".join("?" for _ in active_statuses)
        sql = (
            "SELECT proposal_json FROM active_proposals "
            "WHERE slack_user_id = ? AND slack_thread_ts = ? "
            f"AND status IN ({placeholders}) "
            "ORDER BY created_at DESC LIMIT 1"
        )
        with self._connection_lock:
            row = self._connection.execute(
                sql,
                (slack_user_id, slack_thread_ts, *active_statuses),
            ).fetchone()
        if row is None:
            return None
        return Proposal.model_validate_json(row["proposal_json"])

    def expire_old(self, now: datetime | None = None) -> int:
        """Mark proposals whose `expires_at` is in the past as expired."""

        moment = _ensure_aware(now if now is not None else self._clock())
        with self._connection_lock, _write_transaction(self._connection):
            cursor = self._connection.execute(
                """
                UPDATE active_proposals
                SET status = ?
                WHERE status IN (?, ?) AND expires_at <= ?
                """,
                (
                    ProposalStatus.EXPIRED.value,
                    ProposalStatus.DRAFTING.value,
                    ProposalStatus.AWAITING_CONFIRMATION.value,
                    _datetime_text(moment),
                ),
            )
        return cursor.rowcount

    def _initialize(self) -> None:
        with self._connection_lock, _write_transaction(self._connection):
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS active_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    slack_user_id TEXT NOT NULL,
                    slack_thread_ts TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    jira_project_key TEXT,
                    jira_epic_key TEXT,
                    status TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    revision_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_active_proposals_thread
                ON active_proposals (slack_user_id, slack_thread_ts, status)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_active_proposals_expires_at
                ON active_proposals (expires_at)
                """
            )


def _proposal_to_row(proposal: Proposal) -> tuple[object, ...]:
    payload = proposal.model_dump(mode="json")
    return (
        proposal.proposal_id,
        proposal.slack_user_id,
        proposal.slack_thread_ts,
        proposal.mode.value,
        proposal.project_key,
        proposal.epic_key,
        proposal.status.value,
        json.dumps(payload),
        proposal.revision_count,
        _datetime_text(_ensure_aware(proposal.created_at)),
        _datetime_text(_ensure_aware(proposal.expires_at)),
    )


def _connect(db_path: str | Path, busy_timeout_ms: int) -> sqlite3.Connection:
    path = Path(db_path)
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def _write_transaction(connection: sqlite3.Connection):
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("proposal timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _datetime_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


__all__ = ["PROPOSAL_TTL_SECONDS", "ProposalStore"]

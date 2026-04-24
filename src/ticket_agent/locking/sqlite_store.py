"""SQLite-backed ticket locks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from threading import RLock

from ticket_agent.domain.errors import TicketLockError
from ticket_agent.domain.execution import TicketLock


class SQLiteTicketLockStore:
    """SQLite lock store with heartbeat and expiry semantics."""

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self._db_path = Path(db_path)
        if self._db_path != Path(":memory:"):
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._connection = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def acquire(
        self,
        ticket_key: str,
        owner: str,
        ttl_seconds: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        ticket_key, owner, ttl_seconds = _validate_lock_args(
            ticket_key, owner, ttl_seconds
        )
        now_ts = _to_epoch(now)
        expires_at = now_ts + ttl_seconds

        with self._lock, self._connection:
            self._expire_stale(now_ts=now_ts, ticket_key=ticket_key)
            try:
                self._connection.execute(
                    """
                    INSERT INTO ticket_locks (
                        ticket_key,
                        owner,
                        acquired_at,
                        heartbeat_at,
                        expires_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticket_key, owner, now_ts, now_ts, expires_at),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def heartbeat(
        self,
        ticket_key: str,
        owner: str,
        ttl_seconds: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        ticket_key, owner, ttl_seconds = _validate_lock_args(
            ticket_key, owner, ttl_seconds
        )
        now_ts = _to_epoch(now)
        expires_at = now_ts + ttl_seconds

        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE ticket_locks
                SET heartbeat_at = ?, expires_at = ?
                WHERE ticket_key = ? AND owner = ? AND expires_at > ?
                """,
                (now_ts, expires_at, ticket_key, owner, now_ts),
            )
        return cursor.rowcount == 1

    def release(self, ticket_key: str, owner: str) -> bool:
        ticket_key = _validate_text("ticket_key", ticket_key)
        owner = _validate_text("owner", owner)

        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM ticket_locks WHERE ticket_key = ? AND owner = ?",
                (ticket_key, owner),
            )
        return cursor.rowcount == 1

    def current_lock(
        self,
        ticket_key: str,
        *,
        now: datetime | None = None,
    ) -> TicketLock | None:
        ticket_key = _validate_text("ticket_key", ticket_key)
        now_ts = _to_epoch(now)

        with self._lock, self._connection:
            self._expire_stale(now_ts=now_ts, ticket_key=ticket_key)
            row = self._connection.execute(
                """
                SELECT ticket_key, owner, acquired_at, heartbeat_at, expires_at
                FROM ticket_locks
                WHERE ticket_key = ?
                """,
                (ticket_key,),
            ).fetchone()

        if row is None:
            return None
        return _row_to_ticket_lock(row)

    def expire_stale(self, *, now: datetime | None = None) -> int:
        now_ts = _to_epoch(now)
        with self._lock, self._connection:
            return self._expire_stale(now_ts=now_ts)

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ticket_locks (
                    ticket_key TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ticket_locks_expires_at
                ON ticket_locks (expires_at)
                """
            )

    def _expire_stale(self, *, now_ts: float, ticket_key: str | None = None) -> int:
        if ticket_key is None:
            cursor = self._connection.execute(
                "DELETE FROM ticket_locks WHERE expires_at <= ?",
                (now_ts,),
            )
        else:
            cursor = self._connection.execute(
                "DELETE FROM ticket_locks WHERE ticket_key = ? AND expires_at <= ?",
                (ticket_key, now_ts),
            )
        return cursor.rowcount


def _validate_lock_args(
    ticket_key: str,
    owner: str,
    ttl_seconds: int,
) -> tuple[str, str, int]:
    ticket_key = _validate_text("ticket_key", ticket_key)
    owner = _validate_text("owner", owner)
    if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        raise TicketLockError("ttl_seconds must be a positive integer")
    return ticket_key, owner, ttl_seconds


def _validate_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TicketLockError(f"{name} must be a non-empty string")
    return value.strip()


def _to_epoch(value: datetime | None) -> float:
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise TicketLockError("lock timestamps must be timezone-aware")
    return value.timestamp()


def _from_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _row_to_ticket_lock(row: sqlite3.Row) -> TicketLock:
    return TicketLock(
        ticket_key=row["ticket_key"],
        owner=row["owner"],
        acquired_at=_from_epoch(row["acquired_at"]),
        heartbeat_at=_from_epoch(row["heartbeat_at"]),
        expires_at=_from_epoch(row["expires_at"]),
    )

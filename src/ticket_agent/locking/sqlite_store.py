"""SQLite-backed ticket locks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any
from uuid import uuid4

from ticket_agent.domain.errors import TicketLockError
from ticket_agent.domain.execution import TicketLock


LockIdFactory = Callable[[], str]
Clock = Callable[[], datetime]
EventEmitter = Callable[[str, Mapping[str, Any]], Any]

EVENT_LOCK_ACQUIRED = "lock.acquired"
EVENT_LOCK_RELEASED = "lock.released"
EVENT_LOCK_HEARTBEAT = "lock.heartbeat"


class SQLiteLockManager:
    """Runner-facing SQLite lock manager with lock IDs and component ownership."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        component_id: str,
        busy_timeout_ms: int = 5000,
        lock_id_factory: LockIdFactory | None = None,
        clock: Clock | None = None,
        emit: EventEmitter | None = None,
    ) -> None:
        self._component_id = _validate_text("component_id", component_id)
        self._lock_id_factory = lock_id_factory or (lambda: uuid4().hex)
        self._clock = clock or _utcnow
        self._emit = emit
        self._connection_lock = RLock()
        self._connection = _connect(db_path, busy_timeout_ms)
        self._initialize()

    def close(self) -> None:
        with self._connection_lock:
            self._connection.close()

    def acquire(self, ticket_key: str, ttl_s: int = 1800) -> TicketLock | None:
        """Acquire a ticket lock when no non-expired lock exists."""

        ticket_key = _validate_text("ticket_key", ticket_key)
        ttl_s = _validate_ttl(ttl_s)
        lock_id = _validate_text("lock_id", self._lock_id_factory())
        now = _ensure_aware(self._clock())
        expires_at = now + timedelta(seconds=ttl_s)

        with self._connection_lock, _write_transaction(self._connection):
            try:
                self._connection.execute(
                    """
                    INSERT INTO ticket_locks (
                        ticket_key,
                        lock_id,
                        component_id,
                        acquired_at,
                        expires_at,
                        last_heartbeat
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticket_key,
                        lock_id,
                        self._component_id,
                        _datetime_text(now),
                        _datetime_text(expires_at),
                        _datetime_text(now),
                    ),
                )
            except sqlite3.IntegrityError:
                return None

        lock = TicketLock(
            ticket_key=ticket_key,
            owner=self._component_id,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
            lock_id=lock_id,
        )
        self._emit_event(EVENT_LOCK_ACQUIRED, lock)
        return lock

    def heartbeat(self, lock: TicketLock, ttl_s: int = 1800) -> bool:
        """Extend an owned, non-expired ticket lock."""

        ttl_s = _validate_ttl(ttl_s)
        lock_id = _required_lock_id(lock)
        now = _ensure_aware(self._clock())
        expires_at = now + timedelta(seconds=ttl_s)

        with self._connection_lock, _write_transaction(self._connection):
            cursor = self._connection.execute(
                """
                UPDATE ticket_locks
                SET last_heartbeat = ?, expires_at = ?
                WHERE ticket_key = ?
                  AND lock_id = ?
                  AND component_id = ?
                  AND expires_at > ?
                """,
                (
                    _datetime_text(now),
                    _datetime_text(expires_at),
                    lock.ticket_key,
                    lock_id,
                    self._component_id,
                    _datetime_text(now),
                ),
            )
        succeeded = cursor.rowcount == 1
        if succeeded:
            self._emit_event(EVENT_LOCK_HEARTBEAT, lock)
        return succeeded

    def release(self, lock: TicketLock) -> None:
        """Release an owned ticket lock."""

        self._delete_owned_lock(lock)
        self._emit_event(EVENT_LOCK_RELEASED, lock)

    def has_active_lock(self, ticket_key: str) -> bool:
        """Return true when ``ticket_key`` has a non-expired lock."""

        return self.current_lock(ticket_key) is not None

    def current_lock(self, ticket_key: str) -> TicketLock | None:
        """Return the active lock for ``ticket_key``, if any."""

        ticket_key = _validate_text("ticket_key", ticket_key)
        now = _ensure_aware(self._clock())
        with self._connection_lock, _write_transaction(self._connection):
            row = self._connection.execute(
                """
                SELECT ticket_key, lock_id, component_id, acquired_at,
                       last_heartbeat, expires_at
                FROM ticket_locks
                WHERE ticket_key = ? AND expires_at > ?
                """,
                (ticket_key, _datetime_text(now)),
            ).fetchone()
        if row is None:
            return None
        return _row_to_ticket_lock(row)

    def expired_locks(self, *, limit: int | None = None) -> list[TicketLock]:
        """Return expired lock rows without deleting them."""

        now = _ensure_aware(self._clock())
        sql = """
            SELECT ticket_key, lock_id, component_id, acquired_at,
                   last_heartbeat, expires_at
            FROM ticket_locks
            WHERE expires_at <= ?
            ORDER BY expires_at ASC
        """
        params: tuple[object, ...] = (_datetime_text(now),)
        if limit is not None:
            if limit < 1:
                raise TicketLockError("limit must be positive")
            sql += " LIMIT ?"
            params = (*params, limit)

        with self._connection_lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [_row_to_ticket_lock(row) for row in rows]

    def delete_lock(self, lock: TicketLock) -> bool:
        """Delete a lock row by ticket and lock ID."""

        return self._delete_owned_lock(lock)

    def _delete_owned_lock(self, lock: TicketLock) -> bool:
        lock_id = _required_lock_id(lock)
        with self._connection_lock, _write_transaction(self._connection):
            cursor = self._connection.execute(
                """
                DELETE FROM ticket_locks
                WHERE ticket_key = ? AND lock_id = ? AND component_id = ?
                """,
                (lock.ticket_key, lock_id, lock.owner),
            )
        return cursor.rowcount == 1

    def _initialize(self) -> None:
        with self._connection_lock, _write_transaction(self._connection):
            _initialize_schema(self._connection)

    def _emit_event(self, event_name: str, lock: TicketLock) -> None:
        if self._emit is None:
            return
        self._emit(
            event_name,
            {
                "ticket_key": lock.ticket_key,
                "component_id": lock.owner,
                "lock_id": lock.lock_id,
            },
        )


class SQLiteTicketLockStore:
    """Deprecated compatibility/test-only lock store.

    Production pickup uses :class:`SQLiteLockManager`, which never deletes expired
    rows on acquire/current-lock checks. This store preserves the historical API
    for legacy unit coverage and should not be used on the active runner path.
    """

    deprecated = True

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self._connection_lock = RLock()
        self._connection = _connect(db_path, busy_timeout_ms)
        with self._connection_lock, _write_transaction(self._connection):
            _initialize_schema(self._connection)

    def close(self) -> None:
        with self._connection_lock:
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
        now_dt = _to_datetime(now)
        expires_at = now_dt + timedelta(seconds=ttl_seconds)

        with self._connection_lock, _write_transaction(self._connection):
            _delete_expired(self._connection, now=now_dt, ticket_key=ticket_key)
            try:
                self._connection.execute(
                    """
                    INSERT INTO ticket_locks (
                        ticket_key,
                        lock_id,
                        component_id,
                        acquired_at,
                        expires_at,
                        last_heartbeat
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticket_key,
                        owner,
                        owner,
                        _datetime_text(now_dt),
                        _datetime_text(expires_at),
                        _datetime_text(now_dt),
                    ),
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
        now_dt = _to_datetime(now)
        expires_at = now_dt + timedelta(seconds=ttl_seconds)

        with self._connection_lock, _write_transaction(self._connection):
            cursor = self._connection.execute(
                """
                UPDATE ticket_locks
                SET last_heartbeat = ?, expires_at = ?
                WHERE ticket_key = ?
                  AND component_id = ?
                  AND expires_at > ?
                """,
                (
                    _datetime_text(now_dt),
                    _datetime_text(expires_at),
                    ticket_key,
                    owner,
                    _datetime_text(now_dt),
                ),
            )
        return cursor.rowcount == 1

    def release(self, ticket_key: str, owner: str) -> bool:
        ticket_key = _validate_text("ticket_key", ticket_key)
        owner = _validate_text("owner", owner)

        with self._connection_lock, _write_transaction(self._connection):
            cursor = self._connection.execute(
                """
                DELETE FROM ticket_locks
                WHERE ticket_key = ? AND component_id = ?
                """,
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
        now_dt = _to_datetime(now)

        with self._connection_lock, _write_transaction(self._connection):
            _delete_expired(self._connection, now=now_dt, ticket_key=ticket_key)
            row = self._connection.execute(
                """
                SELECT ticket_key, lock_id, component_id, acquired_at,
                       last_heartbeat, expires_at
                FROM ticket_locks
                WHERE ticket_key = ?
                """,
                (ticket_key,),
            ).fetchone()

        if row is None:
            return None
        return _row_to_ticket_lock(row)

    def expire_stale(self, *, now: datetime | None = None) -> int:
        now_dt = _to_datetime(now)
        with self._connection_lock, _write_transaction(self._connection):
            return _delete_expired(self._connection, now=now_dt)


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


def _initialize_schema(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection)
    if columns and not _has_requested_schema(columns):
        _migrate_legacy_table(connection, columns)
    elif not columns:
        _create_lock_table(connection)

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ticket_locks_expires_at
        ON ticket_locks (expires_at)
        """
    )


def _create_lock_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ticket_locks (
            ticket_key TEXT PRIMARY KEY,
            lock_id TEXT NOT NULL,
            component_id TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            last_heartbeat TEXT NOT NULL
        )
        """
    )


def _table_columns(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("PRAGMA table_info(ticket_locks)").fetchall()
    return {str(row["name"]) for row in rows}


def _has_requested_schema(columns: set[str]) -> bool:
    return {
        "ticket_key",
        "lock_id",
        "component_id",
        "acquired_at",
        "expires_at",
        "last_heartbeat",
    }.issubset(columns)


def _migrate_legacy_table(
    connection: sqlite3.Connection,
    columns: set[str],
) -> None:
    rows = connection.execute("SELECT * FROM ticket_locks").fetchall()
    connection.execute("ALTER TABLE ticket_locks RENAME TO ticket_locks_legacy")
    _create_lock_table(connection)
    for row in rows:
        ticket_key = str(row["ticket_key"])
        owner = _row_value(row, columns, "component_id") or _row_value(
            row, columns, "owner"
        )
        if not owner:
            continue
        lock_id = _row_value(row, columns, "lock_id") or owner
        heartbeat = (
            _row_value(row, columns, "last_heartbeat")
            or _row_value(row, columns, "heartbeat_at")
            or _row_value(row, columns, "acquired_at")
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO ticket_locks (
                ticket_key,
                lock_id,
                component_id,
                acquired_at,
                expires_at,
                last_heartbeat
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_key,
                str(lock_id),
                str(owner),
                _coerce_datetime_text(_row_value(row, columns, "acquired_at")),
                _coerce_datetime_text(_row_value(row, columns, "expires_at")),
                _coerce_datetime_text(heartbeat),
            ),
        )
    connection.execute("DROP TABLE ticket_locks_legacy")


def _row_value(row: sqlite3.Row, columns: set[str], name: str) -> object | None:
    if name not in columns:
        return None
    return row[name]


def _delete_expired(
    connection: sqlite3.Connection,
    *,
    now: datetime,
    ticket_key: str | None = None,
) -> int:
    if ticket_key is None:
        cursor = connection.execute(
            "DELETE FROM ticket_locks WHERE expires_at <= ?",
            (_datetime_text(now),),
        )
    else:
        cursor = connection.execute(
            "DELETE FROM ticket_locks WHERE ticket_key = ? AND expires_at <= ?",
            (ticket_key, _datetime_text(now)),
        )
    return cursor.rowcount


def _validate_lock_args(
    ticket_key: str,
    owner: str,
    ttl_seconds: int,
) -> tuple[str, str, int]:
    ticket_key = _validate_text("ticket_key", ticket_key)
    owner = _validate_text("owner", owner)
    ttl_seconds = _validate_ttl(ttl_seconds)
    return ticket_key, owner, ttl_seconds


def _validate_ttl(value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise TicketLockError("ttl_seconds must be a positive integer")
    return value


def _validate_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TicketLockError(f"{name} must be a non-empty string")
    return value.strip()


def _required_lock_id(lock: TicketLock) -> str:
    if not isinstance(lock.lock_id, str) or not lock.lock_id.strip():
        raise TicketLockError("lock_id must be present")
    return lock.lock_id.strip()


def _to_datetime(value: datetime | None) -> datetime:
    if value is None:
        value = _utcnow()
    return _ensure_aware(value)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise TicketLockError("lock timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_text(value: datetime) -> str:
    return _ensure_aware(value).isoformat(timespec="microseconds")


def _coerce_datetime_text(value: object) -> str:
    if isinstance(value, (int, float)):
        return _datetime_text(datetime.fromtimestamp(float(value), tz=timezone.utc))
    if isinstance(value, str) and value.strip():
        return _datetime_text(_parse_datetime_text(value))
    return _datetime_text(_utcnow())


def _parse_datetime_text(value: object) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str):
        raise TicketLockError("lock timestamp must be a datetime string")
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    return _ensure_aware(parsed)


def _row_to_ticket_lock(row: sqlite3.Row) -> TicketLock:
    return TicketLock(
        ticket_key=row["ticket_key"],
        owner=row["component_id"],
        acquired_at=_parse_datetime_text(row["acquired_at"]),
        heartbeat_at=_parse_datetime_text(row["last_heartbeat"]),
        expires_at=_parse_datetime_text(row["expires_at"]),
        lock_id=row["lock_id"],
    )


__all__ = [
    "EVENT_LOCK_ACQUIRED",
    "EVENT_LOCK_HEARTBEAT",
    "EVENT_LOCK_RELEASED",
    "SQLiteLockManager",
    "SQLiteTicketLockStore",
]

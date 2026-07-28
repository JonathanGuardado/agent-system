"""Shared SQLite connection and write-transaction helpers.

Every SQLite-backed store in this repo needs the same two things: a connection
configured for cross-thread use with WAL and an explicit busy timeout, and a
write transaction that serializes writers via ``BEGIN IMMEDIATE``.

``isolation_level=None`` disables Python's implicit transaction management so
``BEGIN IMMEDIATE`` is explicit and visible. ``check_same_thread=False`` means
the caller must supply the mutex; every store here holds an ``RLock`` around
both reads and writes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3


def connect(db_path: str | Path, busy_timeout_ms: int) -> sqlite3.Connection:
    """Open a WAL-mode connection suitable for a lock-guarded store."""

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
def write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run a block inside ``BEGIN IMMEDIATE`` with explicit rollback."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


__all__ = ["connect", "write_transaction"]

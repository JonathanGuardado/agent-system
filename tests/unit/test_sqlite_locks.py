from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import sqlite3

from ticket_agent.locking.sqlite_store import (
    SQLiteLockManager,
    SQLiteTicketLockStore,
)


def test_sqlite_lock_store_allows_single_active_owner(tmp_path):
    store = SQLiteTicketLockStore(tmp_path / "locks.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert store.acquire("PROJ-1", "worker-a", ttl_seconds=30, now=now)
    assert not store.acquire("PROJ-1", "worker-b", ttl_seconds=30, now=now)

    current = store.current_lock("PROJ-1", now=now)
    assert current is not None
    assert current.ticket_key == "PROJ-1"
    assert current.owner == "worker-a"

    store.close()


def test_sqlite_lock_store_heartbeats_only_owned_live_lock(tmp_path):
    store = SQLiteTicketLockStore(tmp_path / "locks.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert store.acquire("PROJ-1", "worker-a", ttl_seconds=30, now=now)
    assert not store.heartbeat("PROJ-1", "worker-b", ttl_seconds=30, now=now)
    assert store.heartbeat("PROJ-1", "worker-a", ttl_seconds=60, now=now)

    current = store.current_lock("PROJ-1", now=now)
    assert current is not None
    assert current.owner == "worker-a"

    store.close()


def test_sqlite_lock_store_expires_stale_locks(tmp_path):
    store = SQLiteTicketLockStore(tmp_path / "locks.sqlite3")
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

    assert store.acquire("PROJ-1", "worker-a", ttl_seconds=30, now=start)
    assert store.current_lock("PROJ-1", now=later) is None
    assert store.acquire("PROJ-1", "worker-b", ttl_seconds=30, now=later)

    current = store.current_lock("PROJ-1", now=later)
    assert current is not None
    assert current.owner == "worker-b"

    store.close()


def test_sqlite_lock_store_releases_only_owned_lock(tmp_path):
    store = SQLiteTicketLockStore(tmp_path / "locks.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert store.acquire("PROJ-1", "worker-a", ttl_seconds=30, now=now)
    assert not store.release("PROJ-1", "worker-b")
    assert store.release("PROJ-1", "worker-a")
    assert store.current_lock("PROJ-1", now=now) is None

    store.close()


def test_sqlite_lock_manager_uses_requested_table_schema(tmp_path):
    db_path = tmp_path / "locks.sqlite3"
    manager = SQLiteLockManager(
        db_path,
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
    )

    lock = manager.acquire("PROJ-1", ttl_s=30)

    assert lock is not None
    assert lock.lock_id == "lock-1"
    assert lock.owner == "runner-1"
    manager.close()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(ticket_locks)")
        }
    assert columns == {
        "ticket_key",
        "lock_id",
        "component_id",
        "acquired_at",
        "expires_at",
        "last_heartbeat",
    }


def test_sqlite_lock_manager_allows_only_one_concurrent_acquire(tmp_path):
    db_path = tmp_path / "locks.sqlite3"
    initializer = SQLiteLockManager(db_path, component_id="initializer")
    initializer.close()

    managers = [
        SQLiteLockManager(db_path, component_id=f"runner-{index}")
        for index in range(8)
    ]
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda manager: manager.acquire("PROJ-1", ttl_s=60),
                    managers,
                )
            )
    finally:
        for manager in managers:
            manager.close()

    winners = [lock for lock in results if lock is not None]
    assert len(winners) == 1
    assert winners[0].ticket_key == "PROJ-1"


def test_sqlite_lock_manager_expired_lock_waits_for_reconciliation(tmp_path):
    current_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def clock() -> datetime:
        return current_time

    manager = SQLiteLockManager(
        tmp_path / "locks.sqlite3",
        component_id="runner-1",
        lock_id_factory=lambda: "lock-1",
        clock=clock,
    )
    try:
        assert manager.acquire("PROJ-1", ttl_s=1) is not None
        current_time = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

        assert manager.has_active_lock("PROJ-1") is False
        assert len(manager.expired_locks()) == 1
        assert manager.acquire("PROJ-1", ttl_s=60) is None
    finally:
        manager.close()

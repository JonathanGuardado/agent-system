"""Locking implementations."""

from ticket_agent.locking.sqlite_store import (
    EVENT_LOCK_ACQUIRED,
    EVENT_LOCK_HEARTBEAT,
    EVENT_LOCK_RELEASED,
    SQLiteLockManager,
    SQLiteTicketLockStore,
)
from ticket_agent.locking.reconciler import (
    EVENT_LOCK_RECONCILED,
    ExpiredLockManager,
    reconcile_expired_locks,
)

__all__ = [
    # Event constants
    "EVENT_LOCK_ACQUIRED",
    "EVENT_LOCK_HEARTBEAT",
    "EVENT_LOCK_RECONCILED",
    "EVENT_LOCK_RELEASED",
    # Protocols
    "ExpiredLockManager",
    # Classes
    "SQLiteLockManager",
    "SQLiteTicketLockStore",
    # Functions
    "reconcile_expired_locks",
]

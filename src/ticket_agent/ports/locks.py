"""Locking boundary interfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ticket_agent.domain.execution import TicketLock


class TicketLockStore(Protocol):
    """Persistence boundary for ticket execution locks."""

    def acquire(
        self,
        ticket_key: str,
        owner: str,
        ttl_seconds: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Acquire a ticket lock when no non-expired lock exists."""

    def heartbeat(
        self,
        ticket_key: str,
        owner: str,
        ttl_seconds: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Extend an owned, non-expired ticket lock."""

    def release(self, ticket_key: str, owner: str) -> bool:
        """Release a ticket lock owned by the caller."""

    def current_lock(
        self,
        ticket_key: str,
        *,
        now: datetime | None = None,
    ) -> TicketLock | None:
        """Return the active ticket lock, if one exists."""

    def expire_stale(self, *, now: datetime | None = None) -> int:
        """Remove all expired locks and return the number removed."""

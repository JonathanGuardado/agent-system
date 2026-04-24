"""Execution-domain data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TicketLock:
    """A currently held lock for a Jira ticket."""

    ticket_key: str
    owner: str
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime

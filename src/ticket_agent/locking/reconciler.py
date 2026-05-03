"""Expired SQLite lock reconciliation against Jira."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from inspect import isawaitable
from typing import Any, Protocol

from ticket_agent.domain.execution import TicketLock
from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_RETRY_COUNT,
    LABEL_AI_CLAIMED,
    STATUS_IN_PROGRESS,
    STATUS_TODO,
)
from ticket_agent.jira.models import JiraTicket


EVENT_LOCK_RECONCILED = "lock.reconciled"
EVENT_LOCK_RECONCILE_FAILED = "lock.reconcile_failed"

EventEmitter = Callable[[str, Mapping[str, Any]], Any]
_LOGGER = logging.getLogger(__name__)


class ExpiredLockManager(Protocol):
    """Lock manager boundary needed by expired lock reconciliation."""

    def expired_locks(self, *, limit: int | None = None) -> list[TicketLock]:
        """Return expired locks without deleting them."""

    def delete_lock(self, lock: TicketLock) -> bool:
        """Delete a lock row after external cleanup succeeds."""


async def reconcile_expired_locks(
    lock_manager: ExpiredLockManager,
    jira_client: JiraClient,
    *,
    limit: int | None = None,
    emit: EventEmitter | None = None,
) -> int:
    """Restore Jira state for expired locks and delete cleaned lock rows."""

    reconciled = 0
    for lock in lock_manager.expired_locks(limit=limit):
        try:
            ticket = await jira_client.get_ticket(lock.ticket_key)
            jira_cleaned = False
            if _needs_jira_cleanup(ticket):
                await _cleanup_jira_ticket(jira_client, ticket)
                jira_cleaned = True
            if lock_manager.delete_lock(lock):
                reconciled += 1
                await _emit_reconciled(emit, lock, jira_cleaned=jira_cleaned)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _emit_reconcile_failed(emit, lock, exc)
            continue
    return reconciled


def _needs_jira_cleanup(ticket: JiraTicket) -> bool:
    component = ticket.fields.get(FIELD_AGENT_ASSIGNED_COMPONENT)
    return (
        LABEL_AI_CLAIMED in ticket.labels
        or (isinstance(component, str) and bool(component.strip()))
        or ticket.status == STATUS_IN_PROGRESS
    )


async def _cleanup_jira_ticket(
    jira_client: JiraClient,
    ticket: JiraTicket,
) -> None:
    if ticket.status != STATUS_TODO:
        await jira_client.transition_ticket(ticket.key, STATUS_TODO)
    if LABEL_AI_CLAIMED in ticket.labels:
        await jira_client.remove_labels(ticket.key, [LABEL_AI_CLAIMED])
    await jira_client.update_fields(
        ticket.key,
        {
            FIELD_AGENT_ASSIGNED_COMPONENT: None,
            FIELD_AGENT_RETRY_COUNT: _retry_count(ticket) + 1,
        },
    )
    await jira_client.add_comment(
        ticket.key,
        "AI execution lock expired. The ticket was restored to To Do for retry.",
    )


def _retry_count(ticket: JiraTicket) -> int:
    value = ticket.fields.get(FIELD_AGENT_RETRY_COUNT)
    if isinstance(value, int) and value >= 0:
        return value
    return 0


async def _emit_reconciled(
    emit: EventEmitter | None,
    lock: TicketLock,
    *,
    jira_cleaned: bool,
) -> None:
    if emit is None:
        return
    await _safe_emit(
        emit,
        EVENT_LOCK_RECONCILED,
        {
            "ticket_key": lock.ticket_key,
            "component_id": lock.owner,
            "lock_id": lock.lock_id,
            "jira_cleaned": jira_cleaned,
        },
    )


async def _emit_reconcile_failed(
    emit: EventEmitter | None,
    lock: TicketLock,
    exc: BaseException,
) -> None:
    payload = {
        "ticket_key": lock.ticket_key,
        "component_id": lock.owner,
        "lock_id": lock.lock_id,
        **_error_payload(exc),
    }
    _LOGGER.warning(
        "lock_reconcile_failed",
        extra=payload,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    if emit is None:
        return
    await _safe_emit(emit, EVENT_LOCK_RECONCILE_FAILED, payload)


async def _safe_emit(
    emit: EventEmitter,
    event_name: str,
    payload: Mapping[str, Any],
) -> None:
    try:
        result = emit(event_name, payload)
        if isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception:
        _LOGGER.warning(
            "lock_reconcile_event_emit_failed",
            extra={"event_name": event_name},
            exc_info=True,
        )


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "error_type": exc.__class__.__name__,
        "error": str(exc) or exc.__class__.__name__,
    }


__all__ = [
    "EVENT_LOCK_RECONCILE_FAILED",
    "EVENT_LOCK_RECONCILED",
    "ExpiredLockManager",
    "reconcile_expired_locks",
]

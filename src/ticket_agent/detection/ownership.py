"""Deterministic ownership rules for Jira ticket pickup."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_MAX_ATTEMPTS,
    LABEL_AI_READY,
    LABEL_DO_NOT_AUTOMATE,
    STATUS_TODO,
)
from ticket_agent.jira.models import JiraTicket


LockLookup = Callable[[str], object | None]
"""Callable returning a current lock for a ticket key, or None."""


@dataclass(frozen=True, slots=True)
class OwnershipDecision:
    """Decision returned by OwnershipChecker.check."""

    eligible: bool
    reason: str = ""


class OwnershipChecker:
    """Apply deterministic rules to decide whether a ticket can be picked up."""

    def __init__(
        self,
        *,
        component_id: str,
        lock_lookup: LockLookup,
        max_retries: int = 3,
    ) -> None:
        if not component_id or not component_id.strip():
            raise ValueError("component_id must be a non-empty string")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self._component_id = component_id.strip()
        self._lock_lookup = lock_lookup
        self._max_retries = int(max_retries)

    def check(self, ticket: JiraTicket) -> OwnershipDecision:
        labels = _ticket_labels(ticket)

        # R0: do-not-automate present → skip silently with empty reason.
        if LABEL_DO_NOT_AUTOMATE in labels:
            return OwnershipDecision(eligible=False, reason="")

        # R1: human assignee set → skip.
        if _has_human_assignee(ticket):
            return OwnershipDecision(eligible=False, reason="human_assigned")

        # R2: agent component mismatch → skip.
        component = ticket.fields.get(FIELD_AGENT_ASSIGNED_COMPONENT)
        if isinstance(component, str) and component.strip():
            if component.strip() != self._component_id:
                return OwnershipDecision(
                    eligible=False,
                    reason="different_component",
                )

        # R3: ai-ready label missing → skip.
        if LABEL_AI_READY not in labels:
            return OwnershipDecision(eligible=False, reason="missing_ai_ready")

        # R4: status must be To Do.
        status = (ticket.status or "").strip()
        if status != STATUS_TODO:
            return OwnershipDecision(
                eligible=False,
                reason=f"wrong_status:{status or 'unknown'}",
            )

        # R5: active SQLite lock exists → skip.
        try:
            current_lock = self._lock_lookup(ticket.key)
        except Exception:  # noqa: BLE001 - defensive: skip on lock lookup error
            current_lock = None
        if current_lock is not None:
            return OwnershipDecision(eligible=False, reason="active_lock")

        # R6: unresolved blocking issue exists → skip.
        # TODO: JiraTicket does not yet model issue links natively; we read
        # `blocked_by` / `blocks` lists from the fields mapping. Replace with
        # the real issue-link API once the boundary supports it.
        blocking_key = _first_blocking_issue(ticket)
        if blocking_key is not None:
            return OwnershipDecision(
                eligible=False,
                reason=f"blocked_by:{blocking_key}",
            )

        # R7: retry count >= max allowed → skip.
        if _retry_limit_reached(ticket, self._max_retries):
            return OwnershipDecision(
                eligible=False,
                reason="retry_limit_reached",
            )

        return OwnershipDecision(eligible=True, reason="")


def _ticket_labels(ticket: JiraTicket) -> set[str]:
    labels = ticket.labels or []
    return {label for label in labels if isinstance(label, str)}


def _has_human_assignee(ticket: JiraTicket) -> bool:
    assignee = ticket.assignee
    if not isinstance(assignee, str):
        return False
    return bool(assignee.strip())


def _first_blocking_issue(ticket: JiraTicket) -> str | None:
    for field_name in ("blocked_by", "blocks", "blocking_issues"):
        value = ticket.fields.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                key = _coerce_blocking_key(item)
                if key is not None:
                    return key
        if isinstance(value, Mapping):
            for item in value.values():
                key = _coerce_blocking_key(item)
                if key is not None:
                    return key
    return None


def _coerce_blocking_key(item: object) -> str | None:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, Mapping):
        candidate = item.get("key") or item.get("ticket_key")
        if isinstance(candidate, str) and candidate.strip():
            resolved = item.get("resolved")
            if isinstance(resolved, bool) and resolved:
                return None
            status = item.get("status")
            if isinstance(status, str) and status.lower() in {
                "done",
                "closed",
                "resolved",
            }:
                return None
            return candidate.strip()
    return None


def _retry_limit_reached(ticket: JiraTicket, default_max_retries: int) -> bool:
    retry_count = ticket.fields.get(FIELD_AGENT_RETRY_COUNT, 0)
    if not isinstance(retry_count, int):
        return False

    max_attempts = ticket.fields.get(FIELD_MAX_ATTEMPTS)
    if isinstance(max_attempts, int) and max_attempts > 0:
        return retry_count >= max_attempts
    return retry_count >= default_max_retries


__all__ = [
    "LockLookup",
    "OwnershipChecker",
    "OwnershipDecision",
]

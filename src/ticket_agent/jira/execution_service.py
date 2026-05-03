"""Jira execution-state updates for claimed work."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from inspect import isawaitable

from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    EVENT_JIRA_COMPENSATION_COMPLETED,
    EVENT_JIRA_COMPENSATION_FAILED,
    EVENT_JIRA_COMPENSATION_STARTED,
    FIELD_AGENT_ASSIGNED_COMPONENT,
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_IN_REVIEW,
    STATUS_TODO,
)
from ticket_agent.jira.models import JiraExecutionError

_CLAIMED_LABELS = (LABEL_AI_CLAIMED,)
_FAILED_LABELS = (LABEL_AI_FAILED,)
_MARK_CLAIMED_OPERATION = "mark_claimed"
_MARK_FAILED_OPERATION = "mark_failed"
_MARK_IN_REVIEW_OPERATION = "mark_in_review"
_MARK_RELEASED_OPERATION = "mark_released"
_STEP_ADD_COMMENT = "add_comment"
_STEP_ADD_LABELS = "add_labels"
_STEP_REMOVE_LABELS = "remove_labels"
_STEP_TRANSITION_TICKET = "transition_ticket"
_STEP_UPDATE_FIELDS = "update_fields"

_LOGGER = logging.getLogger(__name__)

EventEmitter = Callable[[str, dict[str, object]], object]


class JiraExecutionService:
    """Update Jira with execution lifecycle state."""

    def __init__(
        self,
        client: JiraClient,
        component_id: str,
        emit: EventEmitter | None = None,
    ) -> None:
        self._client = client
        self._component_id = component_id
        self._event_emitter = emit

    async def mark_claimed(self, ticket_key: str) -> None:
        """Mark a Jira ticket as claimed by this component."""

        failed_step = _STEP_ADD_LABELS
        try:
            await self._call_jira(
                _MARK_CLAIMED_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.add_labels(ticket_key, list(_CLAIMED_LABELS)),
            )
            failed_step = _STEP_UPDATE_FIELDS
            await self._call_jira(
                _MARK_CLAIMED_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.update_fields(
                    ticket_key,
                    {FIELD_AGENT_ASSIGNED_COMPONENT: self._component_id},
                ),
            )
            failed_step = _STEP_TRANSITION_TICKET
            await self._call_jira(
                _MARK_CLAIMED_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.transition_ticket(
                    ticket_key,
                    STATUS_IN_PROGRESS,
                ),
            )
        except JiraExecutionError as exc:
            await self._run_compensation(
                ticket_key=ticket_key,
                operation=_MARK_CLAIMED_OPERATION,
                failed_step=failed_step,
                compensation_action="release_claim_and_restore_todo",
                actions=(
                    (
                        "remove_claimed_label",
                        lambda: self._client.remove_labels(
                            ticket_key,
                            list(_CLAIMED_LABELS),
                        ),
                    ),
                    (
                        "clear_agent_assigned_component",
                        lambda: self._client.update_fields(
                            ticket_key,
                            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
                        ),
                    ),
                    (
                        "transition_back_to_todo",
                        lambda: self._client.transition_ticket(
                            ticket_key,
                            STATUS_TODO,
                        ),
                    ),
                    (
                        "add_claim_failure_comment",
                        lambda: self._client.add_comment(
                            ticket_key,
                            _claim_compensation_comment(failed_step, exc),
                        ),
                    ),
                ),
            )
            raise

    async def mark_failed(self, ticket_key: str, reason: str) -> None:
        """Mark a Jira ticket as failed and leave an execution comment."""

        await self._call_jira(
            _MARK_FAILED_OPERATION,
            ticket_key,
            _STEP_ADD_LABELS,
            lambda: self._client.add_labels(ticket_key, list(_FAILED_LABELS)),
        )
        failed_step = _STEP_REMOVE_LABELS
        try:
            await self._call_jira(
                _MARK_FAILED_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.remove_labels(ticket_key, list(_CLAIMED_LABELS)),
            )
            failed_step = _STEP_UPDATE_FIELDS
            await self._call_jira(
                _MARK_FAILED_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.update_fields(
                    ticket_key,
                    {FIELD_AGENT_ASSIGNED_COMPONENT: None},
                ),
            )
        except JiraExecutionError as exc:
            await self._add_partial_failure_comment(
                ticket_key=ticket_key,
                operation=_MARK_FAILED_OPERATION,
                failed_step=failed_step,
                error=exc,
            )
            raise
        await self._call_jira(
            _MARK_FAILED_OPERATION,
            ticket_key,
            _STEP_ADD_COMMENT,
            lambda: self._client.add_comment(
                ticket_key,
                f"AI execution failed:\n\n{reason}",
            ),
        )

    async def mark_in_review(self, ticket_key: str, pull_request_url: str) -> None:
        """Move a Jira ticket to review after opening a pull request."""

        await self._call_jira(
            _MARK_IN_REVIEW_OPERATION,
            ticket_key,
            _STEP_TRANSITION_TICKET,
            lambda: self._client.transition_ticket(ticket_key, STATUS_IN_REVIEW),
        )
        failed_step = _STEP_REMOVE_LABELS
        try:
            await self._call_jira(
                _MARK_IN_REVIEW_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.remove_labels(ticket_key, list(_CLAIMED_LABELS)),
            )
            failed_step = _STEP_UPDATE_FIELDS
            await self._call_jira(
                _MARK_IN_REVIEW_OPERATION,
                ticket_key,
                failed_step,
                lambda: self._client.update_fields(
                    ticket_key,
                    {FIELD_AGENT_ASSIGNED_COMPONENT: None},
                ),
            )
        except JiraExecutionError as exc:
            await self._add_partial_failure_comment(
                ticket_key=ticket_key,
                operation=_MARK_IN_REVIEW_OPERATION,
                failed_step=failed_step,
                error=exc,
            )
            raise
        await self._call_jira(
            _MARK_IN_REVIEW_OPERATION,
            ticket_key,
            _STEP_ADD_COMMENT,
            lambda: self._client.add_comment(
                ticket_key,
                f"AI execution opened pull request:\n\n{pull_request_url}",
            ),
        )

    async def mark_released(self, ticket_key: str) -> None:
        """Release this component's execution claim without changing status."""

        try:
            await self._call_jira(
                _MARK_RELEASED_OPERATION,
                ticket_key,
                _STEP_REMOVE_LABELS,
                lambda: self._client.remove_labels(ticket_key, list(_CLAIMED_LABELS)),
            )
        except JiraExecutionError as exc:
            await self._run_compensation(
                ticket_key=ticket_key,
                operation=_MARK_RELEASED_OPERATION,
                failed_step=_STEP_REMOVE_LABELS,
                compensation_action="clear_agent_assigned_component",
                actions=(
                    (
                        "clear_agent_assigned_component",
                        lambda: self._client.update_fields(
                            ticket_key,
                            {FIELD_AGENT_ASSIGNED_COMPONENT: None},
                        ),
                    ),
                ),
            )
            await self._add_partial_failure_comment(
                ticket_key=ticket_key,
                operation=_MARK_RELEASED_OPERATION,
                failed_step=_STEP_REMOVE_LABELS,
                error=exc,
            )
            raise

        try:
            await self._call_jira(
                _MARK_RELEASED_OPERATION,
                ticket_key,
                _STEP_UPDATE_FIELDS,
                lambda: self._client.update_fields(
                    ticket_key,
                    {FIELD_AGENT_ASSIGNED_COMPONENT: None},
                ),
            )
        except JiraExecutionError as exc:
            await self._add_partial_failure_comment(
                ticket_key=ticket_key,
                operation=_MARK_RELEASED_OPERATION,
                failed_step=_STEP_UPDATE_FIELDS,
                error=exc,
            )
            raise

    async def _call_jira(
        self,
        method_name: str,
        ticket_key: str,
        failed_step: str,
        call: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            await call()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            raise JiraExecutionError(
                f"{method_name} failed for {ticket_key}: {error}"
            ) from exc

    async def _add_partial_failure_comment(
        self,
        *,
        ticket_key: str,
        operation: str,
        failed_step: str,
        error: JiraExecutionError,
    ) -> None:
        message = _error_message(error)
        body = (
            f"AI execution Jira update partially failed during {operation}.\n\n"
            f"Failed step: {failed_step}\n"
            f"Error: {message}"
        )
        await self._run_compensation(
            ticket_key=ticket_key,
            operation=operation,
            failed_step=failed_step,
            compensation_action="add_partial_failure_comment",
            actions=(
                (
                    "add_partial_failure_comment",
                    lambda: self._client.add_comment(ticket_key, body),
                ),
            ),
        )

    async def _run_compensation(
        self,
        *,
        ticket_key: str,
        operation: str,
        failed_step: str,
        compensation_action: str,
        actions: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
    ) -> None:
        await self._emit_compensation(
            EVENT_JIRA_COMPENSATION_STARTED,
            ticket_key=ticket_key,
            operation=operation,
            failed_step=failed_step,
            compensation_action=compensation_action,
        )

        failed = False
        for action_name, action in actions:
            try:
                await action()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failed = True
                payload = {
                    "ticket_key": ticket_key,
                    "operation": operation,
                    "failed_step": failed_step,
                    "compensation_action": action_name,
                    "error": _error_message(exc),
                }
                await self._emit_compensation(
                    EVENT_JIRA_COMPENSATION_FAILED,
                    **payload,
                )
                _LOGGER.warning(
                    "jira_compensation_failed",
                    extra=payload,
                    exc_info=True,
                )

        if failed:
            return

        await self._emit_compensation(
            EVENT_JIRA_COMPENSATION_COMPLETED,
            ticket_key=ticket_key,
            operation=operation,
            failed_step=failed_step,
            compensation_action=compensation_action,
        )

    async def _emit_compensation(
        self,
        event_name: str,
        **payload: object,
    ) -> None:
        if self._event_emitter is None:
            return
        try:
            result = self._event_emitter(event_name, payload)
            if isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            return


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _claim_compensation_comment(failed_step: str, error: JiraExecutionError) -> str:
    return (
        "AI execution could not claim this ticket cleanly.\n\n"
        f"Failed step: {failed_step}\n"
        f"Error: {_error_message(error)}\n\n"
        "Best-effort compensation was attempted: removed ai-claimed, cleared the "
        "assigned component, and moved the ticket back to To Do."
    )


__all__ = ["EventEmitter", "JiraExecutionService"]

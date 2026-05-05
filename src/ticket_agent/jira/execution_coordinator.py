"""Coordinate Jira ticket execution without coupling the runner to Jira."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from inspect import isawaitable
from typing import Protocol

from ticket_agent.jira.constants import (
    EVENT_JIRA_EXECUTION_CHECKPOINT_CLEANUP_FAILED,
    EVENT_JIRA_EXECUTION_COMPLETED,
    EVENT_JIRA_EXECUTION_FAILED,
    EVENT_JIRA_EXECUTION_FAILURE_REPORT_FAILED,
    EVENT_JIRA_EXECUTION_IN_REVIEW,
    EVENT_JIRA_EXECUTION_RELEASED_WITHOUT_PR,
    EVENT_JIRA_EXECUTION_SLACK_NOTIFICATION_FAILED,
    EVENT_JIRA_EXECUTION_STARTED,
    EVENT_JIRA_EXECUTION_WORKTREE_CLEANUP_FAILED,
)
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.orchestrator.runner import TicketClaimFailedError, TicketWorkItem
from ticket_agent.orchestrator.state import TicketState

EventEmitter = Callable[[str, dict[str, object]], object]


class TicketRunner(Protocol):
    """Minimal runner boundary needed by Jira execution coordination."""

    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        """Run a loaded ticket work item."""


class SlackPoster(Protocol):
    """Minimal Slack boundary for terminal execution notifications."""

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None: ...


class WorktreeCleaner(Protocol):
    """Terminal cleanup boundary for ticket worktrees."""

    def cleanup(self, state: TicketState) -> object:
        """Clean up local resources for a terminal ticket state."""


class CheckpointCleaner(Protocol):
    """Terminal checkpoint cleanup boundary."""

    def delete_thread(self, thread_id: str) -> object:
        """Delete persisted graph state for a ticket thread."""


class JiraExecutionCoordinator:
    """Load a Jira ticket, run it, and reflect execution state back to Jira."""

    def __init__(
        self,
        loader: JiraWorkItemLoader,
        execution_service: JiraExecutionService,
        runner: TicketRunner,
        emit: EventEmitter | None = None,
        slack: SlackPoster | None = None,
        worktree_cleaner: WorktreeCleaner | None = None,
        checkpointer: CheckpointCleaner | None = None,
        slack_poster_user_id: str = "ticket-agent",
    ) -> None:
        self._loader = loader
        self._execution_service = execution_service
        self._runner = runner
        self._event_emitter = emit
        self._slack = slack
        self._worktree_cleaner = worktree_cleaner
        self._checkpointer = checkpointer
        self._slack_poster_user_id = slack_poster_user_id

    async def run_ticket(self, ticket_key: str) -> TicketState:
        """Run one Jira ticket through the existing orchestrator runner."""

        await self._emit(EVENT_JIRA_EXECUTION_STARTED, ticket_key=ticket_key)

        work_item: TicketWorkItem | None = None
        final_state: TicketState | None = None
        try:
            work_item = await self._loader.load(ticket_key)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit_failed(ticket_key, exc)
            raise

        try:
            final_state = await self._runner.run_ticket(work_item)
        except asyncio.CancelledError:
            raise
        except TicketClaimFailedError as exc:
            await self._emit_failed(ticket_key, exc)
            await self._cleanup_terminal_state(None, work_item)
            raise
        except Exception as exc:
            await self._mark_failed(ticket_key, exc)
            await self._notify_failure(
                _state_from_work_item(work_item),
                _error_message(exc),
            )
            await self._emit_failed(ticket_key, exc)
            await self._cleanup_terminal_state(None, work_item)
            raise

        try:
            if _is_escalated(final_state):
                reason = _escalation_reason(final_state)
                if "escalate" not in final_state.visited_nodes:
                    await self._mark_failed(ticket_key, Exception(reason))
                await self._notify_failure(final_state, reason)
                await self._emit_failed(ticket_key, Exception(reason))
                return final_state

            if final_state.pull_request_url:
                await self._execution_service.mark_in_review(
                    ticket_key,
                    final_state.pull_request_url,
                )
                await self._notify_pr_success(final_state)
                await self._emit(
                    EVENT_JIRA_EXECUTION_IN_REVIEW,
                    ticket_key=ticket_key,
                    pull_request_url=final_state.pull_request_url,
                )
            else:
                await self._execution_service.mark_released(ticket_key)
                await self._emit(
                    EVENT_JIRA_EXECUTION_RELEASED_WITHOUT_PR,
                    ticket_key=ticket_key,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._mark_failed(ticket_key, exc)
            await self._notify_failure(final_state, _error_message(exc))
            await self._emit_failed(ticket_key, exc)
            raise
        finally:
            await self._cleanup_terminal_state(final_state, work_item)

        await self._emit(EVENT_JIRA_EXECUTION_COMPLETED, ticket_key=ticket_key)
        return final_state

    async def _mark_failed(self, ticket_key: str, error: BaseException) -> None:
        try:
            await self._execution_service.mark_failed(ticket_key, _error_message(error))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                EVENT_JIRA_EXECUTION_FAILURE_REPORT_FAILED,
                ticket_key=ticket_key,
                error=_error_message(exc),
            )

    async def _emit_failed(self, ticket_key: str, error: BaseException) -> None:
        await self._emit(
            EVENT_JIRA_EXECUTION_FAILED,
            ticket_key=ticket_key,
            error=_error_message(error),
        )

    async def _notify_pr_success(self, state: TicketState) -> None:
        if self._slack is None or not state.slack_thread_ts:
            return
        await self._post_slack(
            state,
            (
                f"AI execution opened a pull request for {state.ticket_key}:\n\n"
                f"{state.pull_request_url}"
            ),
        )

    async def _notify_failure(self, state: TicketState, reason: str) -> None:
        if self._slack is None or not state.slack_thread_ts:
            return
        await self._post_slack(
            state,
            f"AI execution escalated {state.ticket_key}:\n\n{reason}",
        )

    async def _post_slack(self, state: TicketState, text: str) -> None:
        try:
            result = self._slack.post_thread_reply(
                state.slack_channel,
                state.slack_thread_ts or state.ticket_key,
                self._slack_poster_user_id,
                text,
            )
            if isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                EVENT_JIRA_EXECUTION_SLACK_NOTIFICATION_FAILED,
                ticket_key=state.ticket_key,
                error=_error_message(exc),
            )

    async def _cleanup_terminal_state(
        self,
        final_state: TicketState | None,
        work_item: TicketWorkItem | None,
    ) -> None:
        state = final_state
        if state is None and work_item is not None:
            state = _state_from_work_item(work_item)
        if state is None:
            return
        await self._cleanup_worktree(state)
        await self._cleanup_checkpoint(state.ticket_key)

    async def _cleanup_worktree(self, state: TicketState) -> None:
        if self._worktree_cleaner is None:
            return
        try:
            result = self._worktree_cleaner.cleanup(state)
            if isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                EVENT_JIRA_EXECUTION_WORKTREE_CLEANUP_FAILED,
                ticket_key=state.ticket_key,
                error=_error_message(exc),
            )

    async def _cleanup_checkpoint(self, ticket_key: str) -> None:
        if self._checkpointer is None:
            return
        try:
            result = self._checkpointer.delete_thread(ticket_key)
            if isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                EVENT_JIRA_EXECUTION_CHECKPOINT_CLEANUP_FAILED,
                ticket_key=ticket_key,
                error=_error_message(exc),
            )

    async def _emit(self, event_name: str, **payload: object) -> None:
        if self._event_emitter is None:
            return
        result = self._event_emitter(event_name, payload)
        if isawaitable(result):
            await result


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _is_escalated(state: TicketState) -> bool:
    return state.workflow_status == "escalated"


def _escalation_reason(state: TicketState) -> str:
    return state.escalation_reason or state.error or "workflow escalated"


def _state_from_work_item(work_item: TicketWorkItem) -> TicketState:
    return TicketState(
        ticket_key=work_item.ticket_key,
        summary=work_item.summary,
        description=work_item.description,
        repository=work_item.repository,
        repo_path=work_item.repo_path,
        worktree_path=work_item.worktree_path,
        slack_channel=work_item.slack_channel,
        slack_thread_ts=work_item.slack_thread_ts,
        max_attempts=work_item.max_attempts,
    )


__all__ = [
    "CheckpointCleaner",
    "EventEmitter",
    "JiraExecutionCoordinator",
    "SlackPoster",
    "TicketRunner",
    "WorktreeCleaner",
]

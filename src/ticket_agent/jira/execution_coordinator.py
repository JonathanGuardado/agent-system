"""Coordinate Jira ticket execution without coupling the runner to Jira."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from inspect import isawaitable
from typing import Protocol

from ticket_agent.jira.constants import (
    EVENT_JIRA_EXECUTION_COMPLETED,
    EVENT_JIRA_EXECUTION_FAILED,
    EVENT_JIRA_EXECUTION_FAILURE_REPORT_FAILED,
    EVENT_JIRA_EXECUTION_IN_REVIEW,
    EVENT_JIRA_EXECUTION_RELEASED_WITHOUT_PR,
    EVENT_JIRA_EXECUTION_STARTED,
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


class JiraExecutionCoordinator:
    """Load a Jira ticket, run it, and reflect execution state back to Jira."""

    def __init__(
        self,
        loader: JiraWorkItemLoader,
        execution_service: JiraExecutionService,
        runner: TicketRunner,
        emit: EventEmitter | None = None,
    ) -> None:
        self._loader = loader
        self._execution_service = execution_service
        self._runner = runner
        self._event_emitter = emit

    async def run_ticket(self, ticket_key: str) -> TicketState:
        """Run one Jira ticket through the existing orchestrator runner."""

        await self._emit(EVENT_JIRA_EXECUTION_STARTED, ticket_key=ticket_key)

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
            raise
        except Exception as exc:
            await self._mark_failed(ticket_key, exc)
            await self._emit_failed(ticket_key, exc)
            raise

        try:
            if final_state.pull_request_url:
                await self._execution_service.mark_in_review(
                    ticket_key,
                    final_state.pull_request_url,
                )
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
            await self._emit_failed(ticket_key, exc)
            raise

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

    async def _emit(self, event_name: str, **payload: object) -> None:
        if self._event_emitter is None:
            return
        result = self._event_emitter(event_name, payload)
        if isawaitable(result):
            await result


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


__all__ = ["EventEmitter", "JiraExecutionCoordinator", "TicketRunner"]

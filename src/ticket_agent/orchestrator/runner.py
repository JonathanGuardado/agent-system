"""Runtime runner for executing one eligible ticket through the graph."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from inspect import isawaitable
from logging import Logger
import re
from typing import Any, Protocol

from ticket_agent.domain.errors import TicketLockError
from ticket_agent.domain.execution import TicketLock
from ticket_agent.orchestrator.state import TicketState

Lock = TicketLock
EventEmitter = Callable[[str, Mapping[str, Any]], Any]
ClaimTicket = Callable[[str], Any]

EVENT_LOCK_ACQUIRED = "lock.acquired"
EVENT_LOCK_HEARTBEAT_FAILED = "lock.heartbeat_failed"
EVENT_LOCK_RELEASED = "lock.released"
EVENT_LOCK_RELEASE_FAILED = "lock.release_failed"
EVENT_RUNNER_CLAIM_FAILED = "runner.claim_failed"
EVENT_GRAPH_CHECKPOINT_CLEARED = "graph.checkpoint_cleared"
EVENT_TICKET_STARTED = "ticket_started"
EVENT_TICKET_COMPLETED = "ticket_completed"
EVENT_TICKET_FAILED = "ticket_failed"
EVENT_TICKET_SKIPPED = "ticket_skipped"
INTERRUPT_RESULT_KEY = "__interrupt__"
_SAFE_BRANCH_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class TicketWorkItem:
    """Ticket details needed to start a workflow run."""

    ticket_key: str
    summary: str
    description: str
    repository: str
    repo_path: str | None = None
    worktree_path: str | None = None
    slack_channel: str | None = None
    slack_thread_ts: str | None = None
    max_attempts: int = 3


class LockManager(Protocol):
    """Runtime lock manager used by the orchestrator runner."""

    def acquire(self, ticket_key: str) -> Lock | None:
        """Acquire execution ownership for a ticket, if available."""

    def heartbeat(self, lock: Lock) -> bool:
        """Refresh an acquired lock."""

    def release(self, lock: Lock) -> None:
        """Release an acquired lock."""


class TicketGraph(Protocol):
    """Compiled graph boundary used by the runner."""

    def ainvoke(
        self,
        state: TicketState,
        config: Mapping[str, Any],
    ) -> Awaitable[Any]:
        """Run the workflow graph for a ticket state."""


class CheckpointCleaner(Protocol):
    """Minimal checkpoint store boundary used by the runner."""

    def delete_thread(self, thread_id: str) -> None:
        """Delete persisted graph state for a thread."""


class TicketAlreadyLockedError(TicketLockError):
    """Raised when a ticket cannot be run because another owner holds the lock."""

    def __init__(self, ticket_key: str) -> None:
        super().__init__(f"ticket is already locked: {ticket_key}")
        self.ticket_key = ticket_key


class TicketClaimFailedError(TicketLockError):
    """Raised when Jira cannot reflect a lock-backed ticket claim."""

    def __init__(self, ticket_key: str, error: BaseException) -> None:
        message = str(error) or error.__class__.__name__
        super().__init__(f"ticket claim failed for {ticket_key}: {message}")
        self.ticket_key = ticket_key
        self.original_error = error


class TicketHeartbeatLostError(TicketLockError):
    """Raised when a running graph loses ownership of its SQLite lock."""

    def __init__(self, ticket_key: str) -> None:
        super().__init__(f"ticket lock heartbeat failed for {ticket_key}")
        self.ticket_key = ticket_key


class OrchestratorRunner:
    """Acquire a ticket lock, run the graph, and release the lock."""

    def __init__(
        self,
        *,
        graph: TicketGraph,
        lock_manager: LockManager,
        component_id: str,
        logger: Logger | None = None,
        event_emitter: EventEmitter | None = None,
        claim_ticket: ClaimTicket | None = None,
        checkpointer: CheckpointCleaner | None = None,
        heartbeat_interval_s: float = 600.0,
    ) -> None:
        if heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")

        self._graph = graph
        self._lock_manager = lock_manager
        self._component_id = component_id
        self._logger = logger
        self._event_emitter = event_emitter
        self._claim_ticket = claim_ticket
        self._checkpointer = checkpointer
        self._heartbeat_interval_s = float(heartbeat_interval_s)

    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        """Run one ticket through the graph when its lock can be acquired."""

        lock = self._lock_manager.acquire(work_item.ticket_key)
        if lock is None:
            await self._emit(
                EVENT_TICKET_SKIPPED,
                ticket_key=work_item.ticket_key,
                reason="already_locked",
                component_id=self._component_id,
            )
            if self._logger is not None:
                self._logger.info("ticket already locked: %s", work_item.ticket_key)
            raise TicketAlreadyLockedError(work_item.ticket_key)

        await self._emit_lock_event(EVENT_LOCK_ACQUIRED, work_item, lock)
        await self._clear_stale_checkpoint(work_item, lock)
        state: TicketState | None = None
        graph_exception: Exception | None = None
        graph_traceback = None
        try:
            try:
                await self._claim_jira_ticket(work_item.ticket_key)
            except TicketClaimFailedError as exc:
                graph_exception = exc
                graph_traceback = exc.__traceback__
                await self._emit_claim_failed(work_item, lock, exc)
                raise
            state = self._build_initial_state(work_item, lock)
            await self._emit(
                EVENT_TICKET_STARTED,
                ticket_key=state.ticket_key,
                component_id=self._component_id,
                lock_id=state.lock_id,
            )
            result = await self._run_graph_with_heartbeat(work_item, state, lock)
            final_state = _coerce_graph_result(state, result)
            await self._emit(
                EVENT_TICKET_COMPLETED,
                ticket_key=final_state.ticket_key,
                component_id=self._component_id,
                workflow_status=final_state.workflow_status,
                lock_id=final_state.lock_id,
            )
            return final_state
        except Exception as exc:
            graph_exception = exc
            graph_traceback = exc.__traceback__
            if state is None:
                raise
            failed_state = _mark_failed(state, exc)
            await self._emit(
                EVENT_TICKET_FAILED,
                ticket_key=failed_state.ticket_key,
                component_id=self._component_id,
                error=failed_state.error,
                lock_id=failed_state.lock_id,
            )
            if self._logger is not None:
                self._logger.exception(
                    "ticket graph failed for %s",
                    work_item.ticket_key,
                )
            return failed_state
        finally:
            try:
                self._lock_manager.release(lock)
            except Exception as release_exc:
                await self._emit_lock_release_failed(work_item, lock, release_exc)
                if graph_exception is not None:
                    raise graph_exception.with_traceback(graph_traceback) from None
                raise
            else:
                await self._emit_lock_event(EVENT_LOCK_RELEASED, work_item, lock)

    def _build_initial_state(
        self,
        work_item: TicketWorkItem,
        lock: Lock,
    ) -> TicketState:
        updates: dict[str, Any] = {}
        state_fields = TicketState.model_fields
        lock_id = _lock_id(lock)

        if "lock_id" in state_fields and lock_id is not None:
            updates["lock_id"] = lock_id
        if "component_id" in state_fields:
            updates["component_id"] = self._component_id
        if "branch_name" in state_fields:
            updates["branch_name"] = _branch_name(work_item.ticket_key, lock)

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
            **updates,
        )

    async def _emit(self, event_name: str, **payload: Any) -> None:
        if self._event_emitter is None:
            return
        result = self._event_emitter(event_name, payload)
        if isawaitable(result):
            await result

    async def _claim_jira_ticket(self, ticket_key: str) -> None:
        if self._claim_ticket is None:
            return
        try:
            result = self._claim_ticket(ticket_key)
            if isawaitable(result):
                await result
        except Exception as exc:
            raise TicketClaimFailedError(ticket_key, exc) from exc

    async def _run_graph_with_heartbeat(
        self,
        work_item: TicketWorkItem,
        state: TicketState,
        lock: Lock,
    ) -> Any:
        heartbeat_task = asyncio.create_task(
            self._heartbeat_until_cancelled(work_item, lock),
            name=f"ticket-heartbeat:{work_item.ticket_key}",
        )
        graph_task = asyncio.create_task(
            self._graph.ainvoke(state, config=_graph_config(work_item.ticket_key)),
            name=f"ticket-graph:{work_item.ticket_key}",
        )

        try:
            done, _ = await asyncio.wait(
                {graph_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_error = _task_error(
                    heartbeat_task,
                    fallback=TicketHeartbeatLostError(work_item.ticket_key),
                )
                await _cancel_and_await(graph_task)
                raise heartbeat_error
            return await graph_task
        finally:
            await _cancel_and_await(heartbeat_task)

    async def _heartbeat_until_cancelled(
        self,
        work_item: TicketWorkItem,
        lock: Lock,
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_s)
            try:
                result = self._lock_manager.heartbeat(lock)
                if isawaitable(result):
                    result = await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit_heartbeat_failed(work_item, lock, exc)
                raise

            if not result:
                exc = TicketHeartbeatLostError(work_item.ticket_key)
                await self._emit_heartbeat_failed(work_item, lock, exc)
                raise exc

    async def _emit_claim_failed(
        self,
        work_item: TicketWorkItem,
        lock: Lock,
        exc: TicketClaimFailedError,
    ) -> None:
        await self._emit(
            EVENT_RUNNER_CLAIM_FAILED,
            ticket_key=work_item.ticket_key,
            component_id=self._component_id,
            lock_id=_lock_id(lock),
            error=_error_message(exc.original_error),
        )

    async def _clear_stale_checkpoint(
        self,
        work_item: TicketWorkItem,
        lock: Lock,
    ) -> None:
        if self._checkpointer is None:
            return
        self._checkpointer.delete_thread(work_item.ticket_key)
        await self._emit(
            EVENT_GRAPH_CHECKPOINT_CLEARED,
            ticket_key=work_item.ticket_key,
            component_id=self._component_id,
            lock_id=_lock_id(lock),
        )

    async def _emit_heartbeat_failed(
        self,
        work_item: TicketWorkItem,
        lock: Lock,
        exc: BaseException,
    ) -> None:
        await self._emit(
            EVENT_LOCK_HEARTBEAT_FAILED,
            ticket_key=work_item.ticket_key,
            component_id=self._component_id,
            lock_id=_lock_id(lock),
            error=_error_message(exc),
        )

    async def _emit_lock_event(
        self,
        event_name: str,
        work_item: TicketWorkItem,
        lock: Lock,
    ) -> None:
        await self._emit(
            event_name,
            ticket_key=work_item.ticket_key,
            component_id=self._component_id,
            lock_id=_lock_id(lock),
        )

    async def _emit_lock_release_failed(
        self,
        work_item: TicketWorkItem,
        lock: Lock,
        exc: Exception,
    ) -> None:
        lock_id = _lock_id(lock)
        payload = {
            "ticket_key": work_item.ticket_key,
            "component_id": self._component_id,
            "lock_id": lock_id,
            "error": str(exc) or exc.__class__.__name__,
        }
        if self._logger is not None:
            self._logger.exception(
                "lock_release_failed for %s",
                work_item.ticket_key,
                extra=payload,
            )
        try:
            await self._emit(EVENT_LOCK_RELEASE_FAILED, **payload)
        except Exception:
            if self._logger is not None:
                self._logger.exception(
                    "failed to emit lock_release_failed for %s",
                    work_item.ticket_key,
                )


def _coerce_graph_result(initial_state: TicketState, result: Any) -> TicketState:
    if isinstance(result, TicketState):
        return result
    if isinstance(result, Mapping):
        payload = initial_state.model_dump()
        payload.update(
            {key: value for key, value in result.items() if key != INTERRUPT_RESULT_KEY}
        )
        if INTERRUPT_RESULT_KEY in result:
            payload["workflow_status"] = "waiting_for_approval"
            payload["current_node"] = (
                payload.get("current_node") or "request_execution_approval"
            )
        return TicketState.model_validate(payload)
    return TicketState.model_validate(result)


def _graph_config(ticket_key: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": ticket_key}}


def _mark_failed(state: TicketState, exc: Exception) -> TicketState:
    error = _error_message(exc)
    return state.model_copy(
        update={
            "workflow_status": "escalated",
            "error": error,
            "errors": [*state.errors, error],
        }
    )


def _lock_id(lock: Lock) -> str | None:
    value = getattr(lock, "lock_id", None)
    if value is None:
        value = getattr(lock, "id", None)
    if value is None:
        return None
    return str(value)


def _branch_name(ticket_key: str, lock: Lock) -> str:
    """Build branch name following agent/{TICKET-KEY}/{short-lock-id} convention."""
    lock_id = _lock_id(lock)
    short_id = _short_safe_component(lock_id) if lock_id is not None else ""
    if not short_id:
        # Derive a short, stable ID from the lock owner when no lock_id is present.
        owner = str(getattr(lock, "owner", "") or "")
        short_id = _safe_branch_component(owner) or "run"
    safe_key = _safe_branch_component(ticket_key)
    return f"agent/{safe_key}/{short_id}"


def _short_safe_component(value: str) -> str:
    return _safe_branch_component(value)[:8]


def _safe_branch_component(value: str) -> str:
    return _SAFE_BRANCH_COMPONENT_CHARS.sub("-", value).strip("-_")


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _task_error(task: asyncio.Task[Any], *, fallback: Exception) -> Exception:
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return fallback
    if isinstance(exc, Exception):
        return exc
    return fallback


async def _cancel_and_await(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception:
        return

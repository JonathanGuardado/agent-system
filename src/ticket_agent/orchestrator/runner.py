"""Runtime runner for executing one eligible ticket through the graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from inspect import isawaitable
from logging import Logger
from typing import Any, Protocol

from ticket_agent.domain.errors import TicketLockError
from ticket_agent.domain.execution import TicketLock
from ticket_agent.orchestrator.state import TicketState

Lock = TicketLock
EventEmitter = Callable[[str, Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class TicketWorkItem:
    """Ticket details needed to start a workflow run."""

    ticket_key: str
    summary: str
    description: str
    repository: str
    repo_path: str | None = None
    worktree_path: str | None = None
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

    def ainvoke(self, state: TicketState) -> Awaitable[Any]:
        """Run the workflow graph for a ticket state."""


class TicketAlreadyLockedError(TicketLockError):
    """Raised when a ticket cannot be run because another owner holds the lock."""

    def __init__(self, ticket_key: str) -> None:
        super().__init__(f"ticket is already locked: {ticket_key}")
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
    ) -> None:
        self._graph = graph
        self._lock_manager = lock_manager
        self._component_id = component_id
        self._logger = logger
        self._event_emitter = event_emitter

    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        """Run one ticket through the graph when its lock can be acquired."""

        lock = self._lock_manager.acquire(work_item.ticket_key)
        if lock is None:
            await self._emit(
                "ticket_skipped",
                ticket_key=work_item.ticket_key,
                reason="already_locked",
                component_id=self._component_id,
            )
            if self._logger is not None:
                self._logger.info("ticket already locked: %s", work_item.ticket_key)
            raise TicketAlreadyLockedError(work_item.ticket_key)

        state: TicketState | None = None
        graph_exception: Exception | None = None
        graph_traceback = None
        try:
            state = self._build_initial_state(work_item, lock)
            await self._emit(
                "ticket_started",
                ticket_key=state.ticket_key,
                component_id=self._component_id,
                lock_id=state.lock_id,
            )
            result = await self._graph.ainvoke(state)
            final_state = _coerce_graph_result(state, result)
            await self._emit(
                "ticket_completed",
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
                "ticket_failed",
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
            max_attempts=work_item.max_attempts,
            **updates,
        )

    async def _emit(self, event_name: str, **payload: Any) -> None:
        if self._event_emitter is None:
            return
        result = self._event_emitter(event_name, payload)
        if isawaitable(result):
            await result

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
            await self._emit("lock_release_failed", **payload)
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
        payload.update(result)
        return TicketState.model_validate(payload)
    return TicketState.model_validate(result)


def _mark_failed(state: TicketState, exc: Exception) -> TicketState:
    error = str(exc) or exc.__class__.__name__
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
    short_id = _lock_id(lock)
    if short_id is None:
        # Derive a short, stable ID from the lock owner when no lock_id is present.
        owner = str(getattr(lock, "owner", "") or "")
        short_id = "".join(c for c in owner if c.isalnum() or c == "-") or "run"
    safe_key = ticket_key.replace("/", "-")
    return f"agent/{safe_key}/{short_id}"

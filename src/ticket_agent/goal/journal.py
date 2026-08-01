"""Per-operation external-effect journal and ambiguity recovery."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from inspect import isawaitable
from typing import Any, Generic, Protocol, TypeVar, cast

from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.identity import normalize_goal_id
from ticket_agent.goal.types import (
    OPERATION_POLICIES,
    ActionRecord,
    LoopState,
    OperationPolicy,
    action_id,
    digest,
)

T = TypeVar("T")
Effect = Callable[[], T | Awaitable[T]]
Probe = Callable[[], "ProbeResult[T] | Awaitable[ProbeResult[T]]"]
CrashHook = Callable[[str, ActionRecord], None]


class ActionJournalError(AgentSystemError):
    """Raised when an external action cannot be replayed within its policy."""


class InjectedJournalCrash(BaseException):
    """Test-only crash sentinel that deliberately leaves the row ambiguous."""


@dataclass(frozen=True, slots=True)
class ProbeResult(Generic[T]):
    """Result of probing whether an external effect already happened."""

    found: bool
    value: T | None = None
    result_identity: str | None = None


@dataclass(frozen=True, slots=True)
class ActionOutcome(Generic[T]):
    """Value plus durable record returned by one journaled action."""

    value: T | None
    record: ActionRecord
    recovered: bool = False
    replayed_completion: bool = False


class ActionStore(Protocol):
    def load_loop_state(self, goal_id: str) -> LoopState | None: ...

    def reserve_action(
        self,
        state: LoopState,
        action: ActionRecord,
    ) -> ActionRecord: ...

    def mark_in_flight(
        self,
        action_id_value: str,
        *,
        lease_owner: str,
        lease_seconds: int = 300,
        recovery_classification: str | None = None,
    ) -> ActionRecord: ...

    def mark_done(
        self,
        action_id_value: str,
        *,
        result_identity: str | None = None,
        actual_model_cost_usd: float = 0.0,
        recovery_classification: str | None = None,
    ) -> ActionRecord: ...

    def mark_failed(
        self,
        action_id_value: str,
        *,
        error: str,
        error_classification: str,
        recovery_classification: str,
        actual_model_cost_usd: float | None = None,
    ) -> ActionRecord: ...


class GoalActionJournal:
    """Reserve, execute, and recover effects under their concrete policy."""

    def __init__(
        self,
        store: ActionStore,
        *,
        lease_owner: str,
        lease_seconds: int = 300,
        crash_hook: CrashHook | None = None,
    ) -> None:
        if not lease_owner:
            raise ValueError("journal lease_owner is required")
        if lease_seconds <= 0:
            raise ValueError("journal lease_seconds must be positive")
        self._store = store
        self._lease_owner = lease_owner
        self._lease_seconds = lease_seconds
        self._crash_hook = crash_hook

    async def execute(
        self,
        state: LoopState,
        *,
        operation: str,
        natural_key: str,
        request: object,
        effect: Effect[T],
        probe: Probe[T] | None = None,
        result_identity: Callable[[T], str | None] | None = None,
        restore: Callable[[str], T | Awaitable[T]] | None = None,
        reserved_model_cost_usd: float = 0.0,
        actual_model_cost_usd: Callable[[T], float] | None = None,
    ) -> ActionOutcome[T]:
        policy = _policy(operation)
        goal_id = normalize_goal_id(state.goal_id)
        action_iteration = state.iteration
        durable_state = _merge_loop_state(
            self._store.load_loop_state(goal_id),
            state,
        )
        if reserved_model_cost_usd < 0:
            raise ActionJournalError("model cost reservation cannot be negative")
        if not natural_key:
            raise ActionJournalError(f"{operation} requires a natural key")

        total_model_reservation = (
            reserved_model_cost_usd * policy.max_attempts
            if policy.charge_reservation_on_ambiguity
            else reserved_model_cost_usd
        )
        record = ActionRecord(
            action_id=action_id(goal_id, action_iteration, operation, natural_key),
            goal_id=goal_id,
            iteration=action_iteration,
            kind=policy.kind,
            state="intended",
            operation=operation,
            natural_key=natural_key,
            request_digest=digest(request),
            external=True,
            reserved_model_cost_usd=total_model_reservation,
        )
        self._inject("before_reservation", record)
        durable = self._store.reserve_action(durable_state, record)
        self._inject("after_reservation", durable)

        if durable.state == "done":
            value = await _restore_completed(durable, probe=probe, restore=restore)
            return ActionOutcome(
                value=value,
                record=durable,
                replayed_completion=True,
            )

        ambiguous = durable.state in {"in_flight", "failed"} or durable.attempts > 0
        recovery = None
        if ambiguous:
            recovery = policy.ambiguity_recovery
            if (
                policy.charge_reservation_on_ambiguity
                and durable.actual_model_cost_usd
                < _per_attempt_reservation(durable, policy)
            ):
                durable = self._store.mark_failed(
                    durable.action_id,
                    error="previous model outcome is ambiguous",
                    error_classification="external_effect_ambiguous",
                    recovery_classification=recovery,
                    actual_model_cost_usd=_per_attempt_reservation(
                        durable,
                        policy,
                    ),
                )
            if policy.probe_required:
                if probe is None:
                    raise ActionJournalError(
                        f"{operation} requires its documented ambiguity probe"
                    )
                probed = await _await(probe())
                if probed.found:
                    completed = self._store.mark_done(
                        durable.action_id,
                        result_identity=(
                            probed.result_identity or durable.result_identity
                        ),
                        actual_model_cost_usd=durable.actual_model_cost_usd,
                        recovery_classification=recovery,
                    )
                    return ActionOutcome(
                        value=probed.value,
                        record=completed,
                        recovered=True,
                    )
            if durable.attempts >= policy.max_attempts:
                raise ActionJournalError(
                    f"{operation} exhausted its {policy.max_attempts} bounded attempts; "
                    f"policy allows {policy.maximum_duplicate}"
                )

        in_flight = self._store.mark_in_flight(
            durable.action_id,
            lease_owner=self._lease_owner,
            lease_seconds=self._lease_seconds,
            recovery_classification=recovery,
        )
        try:
            value = await _await(effect())
        except Exception as exc:
            # A boundary exception cannot prove whether the remote side applied
            # the request. Keep the classification conservative and make the
            # next attempt follow this operation's explicit recovery policy.
            ambiguous_cost = (
                min(
                    in_flight.reserved_model_cost_usd,
                    in_flight.actual_model_cost_usd
                    + _per_attempt_reservation(in_flight, policy),
                )
                if policy.charge_reservation_on_ambiguity
                else None
            )
            self._store.mark_failed(
                durable.action_id,
                error=str(exc) or exc.__class__.__name__,
                error_classification="external_effect_ambiguous",
                recovery_classification=policy.ambiguity_recovery,
                actual_model_cost_usd=ambiguous_cost,
            )
            raise

        self._inject("after_external_effect", in_flight)
        identity = (
            result_identity(value)
            if result_identity is not None
            else _default_result_identity(value)
        )
        actual_cost = (
            max(0.0, float(actual_model_cost_usd(value)))
            if actual_model_cost_usd is not None
            else 0.0
        )
        if policy.charge_reservation_on_ambiguity:
            actual_cost += in_flight.actual_model_cost_usd
        completed = self._store.mark_done(
            durable.action_id,
            result_identity=identity,
            actual_model_cost_usd=actual_cost,
            recovery_classification=recovery,
        )
        self._inject("after_completion", completed)
        return ActionOutcome(value=value, record=completed, recovered=ambiguous)

    def _inject(self, point: str, record: ActionRecord) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point, record)


class JournaledModelRouter:
    """Journal goal-scoped model calls while leaving intake calls available."""

    def __init__(
        self,
        delegate: Any,
        journal: GoalActionJournal,
        *,
        reservation_usd: float = 1.0,
    ) -> None:
        if reservation_usd <= 0:
            raise ValueError("model action reservation must be positive")
        self._delegate = delegate
        self._journal = journal
        self._reservation_usd = float(reservation_usd)

    async def invoke(
        self,
        capability: str,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> Any:
        raw_metadata = kwargs.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        raw_goal_id = metadata.get("goal_id")
        if raw_goal_id is None:
            return await self._delegate.invoke(capability, messages, **kwargs)
        goal_id = normalize_goal_id(raw_goal_id)
        iteration = _nonnegative_int(
            metadata.get(
                "goal_iteration",
                metadata.get("implementation_turn", 0),
            )
        )
        prompt_digest = digest(messages)
        phase = _model_phase(metadata.get("workflow_node"))

        async def effect() -> Any:
            return await self._delegate.invoke(capability, messages, **kwargs)

        outcome = await self._journal.execute(
            LoopState(
                goal_id=goal_id,
                contract_version=1,
                phase=phase,
                iteration=iteration,
            ),
            operation="model_invoke",
            natural_key=f"{goal_id}:{iteration}:{prompt_digest}",
            request={
                "goal_id": goal_id,
                "iteration": iteration,
                "capability": capability,
                "prompt_digest": prompt_digest,
            },
            effect=effect,
            result_identity=lambda response: digest(
                getattr(response, "content", response)
            ),
            reserved_model_cost_usd=self._reservation_usd,
            actual_model_cost_usd=_model_cost,
        )
        if outcome.value is None:
            # The model output itself is intentionally not copied into the
            # action row. A completed row therefore blocks a duplicate call
            # and asks checkpoint recovery to supply the already-used output.
            raise ActionJournalError(
                "completed model action has no replayable response; "
                "resume from its workflow checkpoint"
            )
        return outcome.value


def _policy(operation: str) -> OperationPolicy:
    try:
        return OPERATION_POLICIES[operation]
    except KeyError as exc:
        raise ActionJournalError(
            f"external operation {operation!r} has no recovery policy"
        ) from exc


_PHASE_ORDER = (
    "discovering",
    "implementing",
    "verifying",
    "reviewing",
    "demoing",
    "delivering",
    "integrating",
    "ready_for_promotion",
    "closed",
)


def _merge_loop_state(
    existing: LoopState | None,
    requested: LoopState,
) -> LoopState:
    """Advance the durable spine without erasing evidence from earlier actions."""

    if existing is None:
        return requested
    if existing.contract_version != requested.contract_version:
        raise ActionJournalError(
            "action contract version disagrees with durable loop state"
        )
    if existing.is_terminal:
        raise ActionJournalError("cannot reserve an action for a closed goal")
    phase = max(
        (existing.phase, requested.phase),
        key=_PHASE_ORDER.index,
    )
    return replace(
        existing,
        phase=phase,
        iteration=max(existing.iteration, requested.iteration),
        candidate_sha=requested.candidate_sha or existing.candidate_sha,
    )


async def _restore_completed(
    record: ActionRecord,
    *,
    probe: Probe[T] | None,
    restore: Callable[[str], T | Awaitable[T]] | None,
) -> T | None:
    if restore is not None and record.result_identity is not None:
        return await _await(restore(record.result_identity))
    if probe is not None:
        probed = await _await(probe())
        if probed.found:
            return probed.value
    return None


def _default_result_identity(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return digest(value)


def _model_cost(response: object) -> float:
    value = getattr(response, "estimated_cost_usd", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(response, Mapping):
        value = response.get("estimated_cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _per_attempt_reservation(
    record: ActionRecord,
    policy: OperationPolicy,
) -> float:
    return record.reserved_model_cost_usd / max(1, policy.max_attempts)


def _nonnegative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _model_phase(workflow_node: object) -> str:
    return {
        "plan": "discovering",
        "implement": "implementing",
        "review": "reviewing",
    }.get(str(workflow_node), "discovering")


async def _await(value: T | Awaitable[T]) -> T:
    if isawaitable(value):
        return await cast(Awaitable[T], value)
    return cast(T, value)


__all__ = [
    "ActionJournalError",
    "ActionOutcome",
    "GoalActionJournal",
    "InjectedJournalCrash",
    "JournaledModelRouter",
    "ProbeResult",
]

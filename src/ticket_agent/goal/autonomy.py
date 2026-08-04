"""Persisted autonomy resolution and concrete action ceilings."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any, Protocol

from ticket_agent.config.repo_contract import load_repo_contract
from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.identity import normalize_goal_id
from ticket_agent.goal.types import (
    AUTONOMY_DECISION_SCHEMA_VERSION,
    CANDIDATE_EVIDENCE_SOURCE,
    READINESS_CEILING,
    RISK_CEILING,
    AutonomyCeiling,
    AutonomyDecision,
    AutonomyMode,
    GoalContract,
    HarnessReadiness,
    resolve_autonomy,
)

LOGGER = logging.getLogger(__name__)


class AutonomyDecisionStore(Protocol):
    def save_autonomy_decision(
        self,
        decision: AutonomyDecision,
    ) -> AutonomyDecision: ...

    def latest_autonomy_decision(
        self,
        goal_id: str,
        *,
        contract_version: int | None = None,
    ) -> AutonomyDecision | None: ...


class AutonomyActionError(AgentSystemError):
    """Raised when an operation exceeds the goal's persisted ceiling."""


ACTION_MINIMUM_MODE: Mapping[str, AutonomyMode] = {
    "plan": AutonomyMode.PROPOSE,
    "proposal_publish": AutonomyMode.PROPOSE,
    "request_execution_approval": AutonomyMode.PROPOSE,
    "implement": AutonomyMode.IMPLEMENT,
    "run_tests": AutonomyMode.IMPLEMENT,
    "review": AutonomyMode.IMPLEMENT,
    "open_pull_request": AutonomyMode.DELIVER,
    "pr_create": AutonomyMode.DELIVER,
    "pr_merge": AutonomyMode.AUTONOMOUS,
}


class GoalAutonomyResolver:
    """Resolve all named ceilings and append the durable decision."""

    def __init__(
        self,
        store: AutonomyDecisionStore,
        *,
        configured_mode: object = "observe",
        contract_dir: str | Path = "config/repos",
        repo_defaults: Mapping[str, Mapping[str, str]] | None = None,
        enforced_gate_names: tuple[str, ...] = ("test",),
        candidate_evidence_ready: bool = False,
        clock: Any = None,
    ) -> None:
        self._store = store
        self._configured_mode = configured_mode
        self._contract_dir = Path(contract_dir)
        self._repo_defaults = repo_defaults or {}
        self._enforced_gate_names = tuple(sorted(set(enforced_gate_names)))
        # Derived, not configured. This stays False until isolated verification
        # of a committed SHA (P14), independent complete-diff review (P15), and
        # candidate-authorization enforcement are all live. The parameter exists
        # so those phases can flip it and so tests can exercise both sides -- it
        # is not an operator knob, and no configuration file reaches it.
        self._candidate_evidence_ready = candidate_evidence_ready
        self._clock = clock or (lambda: datetime.now(UTC))

    def decide(
        self,
        contract: GoalContract,
        *,
        sandbox_available: bool,
        per_command_approval: bool = False,
        halted: bool = False,
    ) -> AutonomyDecision:
        goal_id = normalize_goal_id(contract.goal_id)
        readiness, readiness_detail, required_gates = self._readiness(contract)
        all_required_gates_enforced = set(required_gates).issubset(
            self._enforced_gate_names
        )
        effective, _ = resolve_autonomy(
            configured=self._configured_mode,
            risk_class=contract.risk_class,
            readiness=readiness,
            sandbox_available=sandbox_available,
            per_command_approval=per_command_approval,
            all_required_gates_enforced=all_required_gates_enforced,
            candidate_evidence_ready=self._candidate_evidence_ready,
            halted=halted,
        )
        configured = AutonomyMode.parse(self._configured_mode)
        sandbox_ceiling = (
            AutonomyMode.AUTONOMOUS
            if sandbox_available
            else AutonomyMode.IMPLEMENT
            if per_command_approval
            else AutonomyMode.PROPOSE
        )
        ceilings = (
            AutonomyCeiling(
                source="env:AGENT_SYSTEM_AUTONOMY_MODE",
                mode=configured,
                detail=f"configured={self._configured_mode!r}",
            ),
            AutonomyCeiling(
                source="contract:risk_class",
                mode=RISK_CEILING[contract.risk_class],
                detail=f"risk_class={contract.risk_class}",
            ),
            AutonomyCeiling(
                source="repo:harness_readiness",
                mode=READINESS_CEILING[readiness],
                detail=readiness_detail,
            ),
            AutonomyCeiling(
                source="runtime:sandbox_available",
                mode=sandbox_ceiling,
                detail=f"sandbox_available={sandbox_available}",
            ),
            AutonomyCeiling(
                source="policy:per_command_approval",
                mode=AutonomyMode.AUTONOMOUS,
                detail=f"per_command_approval={per_command_approval}",
            ),
            AutonomyCeiling(
                source="runtime:derived_gate_enforcement",
                mode=(
                    AutonomyMode.AUTONOMOUS
                    if all_required_gates_enforced
                    else AutonomyMode.IMPLEMENT
                ),
                detail=(
                    f"required={list(required_gates)!r}; "
                    f"wired={list(self._enforced_gate_names)!r}"
                ),
            ),
            AutonomyCeiling(
                source=CANDIDATE_EVIDENCE_SOURCE,
                mode=(
                    AutonomyMode.AUTONOMOUS
                    if self._candidate_evidence_ready
                    else AutonomyMode.IMPLEMENT
                ),
                detail=(
                    "candidate_evidence_ready="
                    f"{self._candidate_evidence_ready}; "
                    "requires isolated verification of a committed SHA, "
                    "independent complete-diff review, and enforced candidate "
                    "authorization"
                ),
            ),
            AutonomyCeiling(
                source="operator:halted",
                mode=(
                    AutonomyMode.OBSERVE if halted else AutonomyMode.AUTONOMOUS
                ),
                detail=f"halted={halted}",
            ),
        )
        # Keep the explicit records honest with the canonical resolver.
        if effective != min(ceiling.mode for ceiling in ceilings):
            raise AutonomyActionError("autonomy ceiling records disagree with resolver")
        return self._store.save_autonomy_decision(
            AutonomyDecision(
                goal_id=goal_id,
                contract_version=contract.version,
                effective_mode=effective,
                ceilings=ceilings,
                required_gates=required_gates,
                enforced_gate_names=self._enforced_gate_names,
                decided_at=self._clock(),
                schema_version=AUTONOMY_DECISION_SCHEMA_VERSION,
            )
        )

    def _readiness(
        self,
        contract: GoalContract,
    ) -> tuple[HarnessReadiness, str, tuple[str, ...]]:
        readiness_order: tuple[HarnessReadiness, ...] = (
            "unready",
            "partial",
            "full",
        )
        results: list[tuple[str, HarnessReadiness, tuple[str, ...]]] = []
        required: set[str] = set()
        for repository in contract.permitted_scope.repositories:
            try:
                repo_contract = self._repo_contract(repository)
                repo_path = self._repo_path(repository)
                readiness, reasons = repo_contract.readiness(
                    repo_path,
                    executable_gates=self._enforced_gate_names,
                )
                required.update(repo_contract.required_gates)
                detail = reasons or ("ready",)
            except Exception as exc:  # noqa: BLE001 - missing harness lowers mode
                readiness = "unready"
                detail = (f"harness unavailable: {exc}",)
            results.append((repository, readiness, tuple(detail)))

        if not results:
            results.append(("(none)", "unready", ("no repository in scope",)))
        worst = min(
            (result[1] for result in results),
            key=readiness_order.index,
        )
        detail = "; ".join(
            f"{repository}={readiness} ({', '.join(reasons)})"
            for repository, readiness, reasons in results
        )
        return worst, detail, tuple(sorted(required))

    def _repo_contract(self, repository: str):
        direct = self._contract_dir / f"{repository}.yaml"
        if direct.is_file():
            return load_repo_contract(direct)
        for candidate in sorted(self._contract_dir.glob("*.yaml")):
            loaded = load_repo_contract(candidate)
            if loaded.repo.name == repository:
                return loaded
        raise FileNotFoundError(
            f"no repository contract for {repository!r} in {self._contract_dir}"
        )

    def _repo_path(self, repository: str) -> Path | None:
        matches = [
            values.get("repo_path")
            for values in self._repo_defaults.values()
            if values.get("repository") == repository and values.get("repo_path")
        ]
        if len(matches) == 1:
            return Path(matches[0]).expanduser()
        return None


class AutonomyActionGuard:
    """Enforce the latest persisted mode at each concrete effect boundary."""

    def __init__(self, store: AutonomyDecisionStore) -> None:
        self._store = store

    def check(self, goal_id: str | None, operation: str) -> AutonomyDecision:
        try:
            canonical_goal_id = normalize_goal_id(goal_id)
        except Exception as exc:
            raise AutonomyActionError(
                f"operation {operation} requires canonical goal identity"
            ) from exc
        required = ACTION_MINIMUM_MODE.get(operation)
        if required is None:
            raise AutonomyActionError(
                f"operation {operation!r} has no autonomy ceiling policy"
            )
        decision = self._store.latest_autonomy_decision(canonical_goal_id)
        if decision is None:
            raise AutonomyActionError(
                f"operation {operation} has no persisted autonomy decision"
            )
        effective = self._capped_mode(decision, operation)
        if effective < required:
            raise AutonomyActionError(
                f"operation {operation} requires {required}, "
                f"effective mode is {effective}"
            )
        return decision

    def _capped_mode(
        self,
        decision: AutonomyDecision,
        operation: str,
    ) -> AutonomyMode:
        """Re-apply the candidate-evidence ceiling at the effect boundary.

        A decision persisted by an earlier build carries whatever mode that
        build resolved -- possibly ``deliver`` -- and reading it back would
        otherwise let a pre-upgrade row authorize an external effect the
        current build would never have granted. Upgrading must not leave a
        stale grant lying in the database.
        """

        if decision.is_candidate_evidence_aware:
            return decision.effective_mode
        capped = min(decision.effective_mode, AutonomyMode.IMPLEMENT)
        if capped != decision.effective_mode:
            LOGGER.warning(
                "capping legacy autonomy decision for %s: operation=%s "
                "persisted=%s capped=%s schema_version=%s",
                decision.goal_id,
                operation,
                decision.effective_mode,
                capped,
                decision.schema_version,
            )
        return capped


__all__ = [
    "ACTION_MINIMUM_MODE",
    "AutonomyActionError",
    "AutonomyActionGuard",
    "GoalAutonomyResolver",
]

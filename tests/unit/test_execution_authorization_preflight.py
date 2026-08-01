from __future__ import annotations

import asyncio

import pytest

from ticket_agent.goal.contract import (
    Allowlist,
    GoalContractCompiler,
    SQLiteGoalContractStore,
)
from ticket_agent.goal.execution_preflight import (
    ExecutionAuthorizationError,
    ExecutionAuthorizationPreflight,
)
from ticket_agent.goal.policy import RiskPolicy
from ticket_agent.goal.semantic_check import SemanticVerdict
from ticket_agent.goal.signing import Signer, generate_key
from ticket_agent.goal.types import AutonomyDecision, AutonomyMode
from ticket_agent.orchestrator.runner import TicketWorkItem

GOAL_ID = "prop-0123456789ab"


class _SemanticChecker:
    def __init__(self, agrees: bool = True) -> None:
        self.agrees = agrees

    async def check(self, contract, *, exclude_providers=()):
        del contract, exclude_providers
        return SemanticVerdict(
            objective_matches=True,
            criteria_complete=self.agrees,
            nothing_invented=True,
            missing=() if self.agrees else ("required behavior",),
        )


class _Sandbox:
    profile = "bwrap"


class _Environment:
    def __init__(self) -> None:
        self.subjects = []

    def check(self, subject=None):
        self.subjects.append(subject)
        return _Sandbox()


class _AutonomyResolver:
    def decide(self, contract, *, sandbox_available, **kwargs):
        del sandbox_available, kwargs
        return AutonomyDecision(
            goal_id=contract.goal_id,
            contract_version=contract.version,
            effective_mode=AutonomyMode.IMPLEMENT,
            ceilings=(),
        )


def test_verified_durable_authorization_allows_scoped_execution(tmp_path):
    signer = Signer(generate_key())
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    environment = _Environment()
    item = _work_item()
    try:
        store.save_outcome(_outcome(signer))
        preflight = ExecutionAuthorizationPreflight(
            environment,
            store,
            signer,
            _AutonomyResolver(),
        )

        sandbox = preflight.check(item)

        assert sandbox.profile == "bwrap"
        assert environment.subjects == [item]
    finally:
        store.close()


@pytest.mark.parametrize(
    "item",
    (
        TicketWorkItem(
            ticket_key="AGENT-1",
            summary="Legacy",
            description="",
            repository="agent-system",
            labels=("ai-ready",),
        ),
        TicketWorkItem(
            ticket_key="AGENT-1",
            summary="Ambiguous",
            description="",
            repository="agent-system",
            goal_id=GOAL_ID,
            labels=(f"ai-goal-{GOAL_ID}", f"ai-goal-{GOAL_ID}"),
        ),
    ),
)
def test_missing_or_ambiguous_goal_identity_is_refused(tmp_path, item):
    signer = Signer(generate_key())
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(_outcome(signer))
        preflight = ExecutionAuthorizationPreflight(
            _Environment(), store, signer, _AutonomyResolver()
        )

        with pytest.raises(ExecutionAuthorizationError, match="goal identity"):
            preflight.check(item)
    finally:
        store.close()


def test_semantically_denied_goal_is_refused_from_durable_record(tmp_path):
    signer = Signer(generate_key())
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(_outcome(signer, semantic_agrees=False))
        preflight = ExecutionAuthorizationPreflight(
            _Environment(), store, signer, _AutonomyResolver()
        )

        with pytest.raises(ExecutionAuthorizationError, match="required behavior"):
            preflight.check(_work_item())
    finally:
        store.close()


def test_current_revocation_is_revalidated_and_refused(tmp_path):
    signer = Signer(generate_key())
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(_outcome(signer))
        store.append_revocation(
            GOAL_ID,
            1,
            revoked_by="operator@example.test",
            reason="operator stopped the goal",
            signer=signer,
        )
        preflight = ExecutionAuthorizationPreflight(
            _Environment(), store, signer, _AutonomyResolver()
        )

        with pytest.raises(ExecutionAuthorizationError, match="revoked"):
            preflight.check(_work_item())
    finally:
        store.close()


def test_tampered_authorization_digest_is_refused(tmp_path):
    signer = Signer(generate_key())
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(_outcome(signer))
        store._connection.execute(
            "UPDATE goal_contracts SET evidence_digest = 'tampered'"
        )
        preflight = ExecutionAuthorizationPreflight(
            _Environment(), store, signer, _AutonomyResolver()
        )

        with pytest.raises(ExecutionAuthorizationError, match="digest"):
            preflight.check(_work_item())
    finally:
        store.close()


def _outcome(signer: Signer, *, semantic_agrees: bool = True):
    compiler = GoalContractCompiler(
        policy=RiskPolicy(version=1, repositories=("agent-system",)),
        allowlist=Allowlist(
            users=frozenset({"U1"}),
            channels=frozenset({"C1"}),
        ),
        signer=signer,
        semantic_checker=_SemanticChecker(semantic_agrees),
    )
    return asyncio.run(
        compiler.compile(
            goal_id=GOAL_ID,
            original_request="Implement the required behavior",
            objective="Implement the required behavior",
            acceptance_criteria=("Required behavior works",),
            user_id="U1",
            channel="C1",
            thread_ts="1.0",
            repositories=("agent-system",),
        )
    )


def _work_item() -> TicketWorkItem:
    return TicketWorkItem(
        ticket_key="AGENT-1",
        summary="Authorized",
        description="",
        repository="agent-system",
        goal_id=GOAL_ID,
        labels=(f"ai-goal-{GOAL_ID}", "ai-ready"),
    )

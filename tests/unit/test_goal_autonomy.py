from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ticket_agent.goal.autonomy import (
    AutonomyActionError,
    AutonomyActionGuard,
    GoalAutonomyResolver,
)
from ticket_agent.goal.spine import SQLiteGoalSpine
from ticket_agent.goal.types import (
    AUTONOMY_DECISION_SCHEMA_VERSION,
    CANDIDATE_EVIDENCE_SOURCE,
    AcceptanceCriterion,
    AuthorizationContext,
    AutonomyCeiling,
    AutonomyDecision,
    AutonomyMode,
    GoalContract,
    ScopeSpec,
)

GOAL_ID = "prop-0123456789ab"


def test_unrecognized_configured_mode_persists_observe_with_named_inputs(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    _write_contract(tmp_path, required_gates=("test",))
    resolver = GoalAutonomyResolver(
        spine,
        configured_mode="typo-mode",
        contract_dir=tmp_path,
        enforced_gate_names=("test",),
    )
    try:
        decision = resolver.decide(_contract(), sandbox_available=True)

        assert decision.effective_mode is AutonomyMode.OBSERVE
        assert decision.binding_sources == (
            "env:AGENT_SYSTEM_AUTONOMY_MODE",
        )
        assert {ceiling.source for ceiling in decision.ceilings} == {
            "env:AGENT_SYSTEM_AUTONOMY_MODE",
            "contract:risk_class",
            "repo:harness_readiness",
            "runtime:sandbox_available",
            "policy:per_command_approval",
            "runtime:derived_gate_enforcement",
            CANDIDATE_EVIDENCE_SOURCE,
            "operator:halted",
        }
        assert spine.latest_autonomy_decision(GOAL_ID) == decision
    finally:
        spine.close()


def test_gate_enforcement_is_derived_from_wired_executors(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    _write_contract(tmp_path, required_gates=("test", "typecheck"))
    resolver = GoalAutonomyResolver(
        spine,
        configured_mode="autonomous",
        contract_dir=tmp_path,
        # A declared typecheck command is not an executor. Only test is wired
        # into a production node whose failure blocks progress.
        enforced_gate_names=("test",),
    )
    try:
        decision = resolver.decide(_contract(), sandbox_available=True)

        assert decision.required_gates == ("test", "typecheck")
        assert decision.enforced_gate_names == ("test",)
        # Two independent ceilings now name the same hole, and readiness is the
        # stricter of the two: a repository whose required verification cannot
        # run is unready (-> propose), not merely limited to implementing.
        assert decision.effective_mode is AutonomyMode.PROPOSE
        assert decision.binding_sources == ("repo:harness_readiness",)

        ceilings = {ceiling.source: ceiling for ceiling in decision.ceilings}
        # The gate-enforcement ceiling still records the same hole at implement;
        # readiness simply binds below it now, and says why.
        assert (
            ceilings["runtime:derived_gate_enforcement"].mode
            is AutonomyMode.IMPLEMENT
        )
        assert "no runtime executor" in ceilings["repo:harness_readiness"].detail
    finally:
        spine.close()


def test_action_ceiling_binds_each_concrete_boundary(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    guard = AutonomyActionGuard(spine)
    decided_at = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        _save_mode(spine, AutonomyMode.OBSERVE, decided_at)
        with pytest.raises(AutonomyActionError, match="requires propose"):
            guard.check(GOAL_ID, "plan")

        _save_mode(spine, AutonomyMode.PROPOSE, decided_at + timedelta(seconds=1))
        guard.check(GOAL_ID, "proposal_publish")
        with pytest.raises(AutonomyActionError, match="requires implement"):
            guard.check(GOAL_ID, "implement")

        _save_mode(spine, AutonomyMode.IMPLEMENT, decided_at + timedelta(seconds=2))
        guard.check(GOAL_ID, "run_tests")
        with pytest.raises(AutonomyActionError, match="requires deliver"):
            guard.check(GOAL_ID, "pr_create")

        _save_mode(spine, AutonomyMode.DELIVER, decided_at + timedelta(seconds=3))
        guard.check(GOAL_ID, "pr_create")
        with pytest.raises(AutonomyActionError, match="requires autonomous"):
            guard.check(GOAL_ID, "pr_merge")

        _save_mode(spine, AutonomyMode.AUTONOMOUS, decided_at + timedelta(seconds=4))
        guard.check(GOAL_ID, "pr_merge")
    finally:
        spine.close()


def _contract() -> GoalContract:
    return GoalContract(
        goal_id=GOAL_ID,
        version=1,
        schema_version=1,
        authorization=AuthorizationContext(
            requester="U1",
            slack_channel="C1",
            slack_message_ts="1.0",
            allowlisted=True,
            authorized_at=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        original_request="Implement the feature",
        objective="Implement the feature",
        acceptance_criteria=(AcceptanceCriterion("c1", "Feature works"),),
        permitted_scope=ScopeSpec(repositories=("agent-system",)),
        risk_class="low",
    )


def _save_mode(
    spine: SQLiteGoalSpine,
    mode: AutonomyMode,
    decided_at: datetime,
    *,
    candidate_evidence_ready: bool = True,
) -> None:
    """Persist a decision at ``mode``.

    Defaults to a current-capability decision so ladder assertions exercise the
    guard's mode comparison rather than the legacy cap. Pass
    ``candidate_evidence_ready=False`` to build a pre-upgrade row.
    """

    ceilings = (
        (
            AutonomyCeiling(
                source=CANDIDATE_EVIDENCE_SOURCE,
                mode=AutonomyMode.AUTONOMOUS,
                detail="candidate_evidence_ready=True",
            ),
        )
        if candidate_evidence_ready
        else ()
    )
    spine.save_autonomy_decision(
        AutonomyDecision(
            goal_id=GOAL_ID,
            contract_version=1,
            effective_mode=mode,
            ceilings=ceilings,
            decided_at=decided_at,
            schema_version=(
                AUTONOMY_DECISION_SCHEMA_VERSION if candidate_evidence_ready else 0
            ),
        )
    )


def _write_contract(tmp_path, *, required_gates: tuple[str, ...]) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    command_blocks = [
        "  test:",
        "    command: ['python', '-m', 'pytest']",
        "    timeout_seconds: 30",
        "    working_directory: .",
        "    writable_paths: ['.pytest_cache']",
        "    network: none",
    ]
    if "typecheck" in required_gates:
        command_blocks.extend(
            [
                "  typecheck:",
                "    command: ['python', '-m', 'mypy', 'src']",
                "    timeout_seconds: 30",
                "    working_directory: .",
                "    writable_paths: ['.mypy_cache']",
                "    network: none",
            ]
        )
    gates = [f"  {gate}: required" for gate in required_gates]
    content = [
        "repo:",
        "  name: agent-system",
        f"  root: {repo_path}",
        "  default_branch: main",
        "language:",
        "  primary: python",
        "  package_manager: setuptools",
        "commands:",
        *command_blocks,
        "  lint: null",
        "  install: null",
        "gates:",
        *gates,
        "trust_root:",
        "  - kind: file",
        "    path: pyproject.toml",
        "policy:",
        "  dependency_install_allowed: false",
        "  config_paths_allowed: []",
        "  protected_paths: []",
        "source_dirs: ['src/']",
        "test_dirs: ['tests/']",
    ]
    (tmp_path / "agent-system.yaml").write_text(
        "\n".join(content) + "\n",
        encoding="utf-8",
    )


def test_candidate_evidence_ceiling_caps_a_deliverable_contract(tmp_path):
    """Configuration alone cannot buy delivery.

    A standard-risk contract has `RISK_CEILING["standard"] == DELIVER`, and a
    repository requiring only gates that happen to be wired satisfies gate
    enforcement. Before this ceiling those two facts together were enough to
    reach `deliver`.
    """

    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    resolver = GoalAutonomyResolver(
        spine,
        configured_mode="autonomous",
        contract_dir=tmp_path,
        enforced_gate_names=("test",),
    )
    try:
        decision = resolver.decide(_contract(), sandbox_available=True)

        assert decision.effective_mode <= AutonomyMode.IMPLEMENT
        # Asserted as a contributed ceiling rather than a binding one: other
        # ceilings may sit lower in a given fixture, but this one must always
        # be present and must never exceed `implement`.
        ceiling = next(
            item
            for item in decision.ceilings
            if item.source == CANDIDATE_EVIDENCE_SOURCE
        )
        assert ceiling.mode is AutonomyMode.IMPLEMENT
        assert decision.schema_version == AUTONOMY_DECISION_SCHEMA_VERSION
    finally:
        spine.close()


def test_pre_upgrade_deliver_decision_cannot_open_or_push_a_candidate(tmp_path):
    """The release test for this step.

    An operator who ran an older build may already have a persisted `deliver`
    or `autonomous` row. Upgrading must not leave that grant usable: the guard
    re-applies the ceiling on read, so the stale row authorizes nothing
    external no matter what mode it recorded.
    """

    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    guard = AutonomyActionGuard(spine)
    decided_at = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        for offset, mode in enumerate(
            (AutonomyMode.DELIVER, AutonomyMode.AUTONOMOUS)
        ):
            _save_mode(
                spine,
                mode,
                decided_at + timedelta(seconds=offset),
                candidate_evidence_ready=False,
            )
            for operation in ("open_pull_request", "pr_create", "pr_merge"):
                with pytest.raises(AutonomyActionError, match="effective mode"):
                    guard.check(GOAL_ID, operation)

            # Capped, not revoked: implementation still proceeds.
            guard.check(GOAL_ID, "run_tests")
    finally:
        spine.close()


def test_legacy_decision_is_identified_by_version_not_only_by_source(tmp_path):
    """A forged ceiling record on a stale schema version is still legacy."""

    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    guard = AutonomyActionGuard(spine)
    try:
        spine.save_autonomy_decision(
            AutonomyDecision(
                goal_id=GOAL_ID,
                contract_version=1,
                effective_mode=AutonomyMode.DELIVER,
                ceilings=(
                    AutonomyCeiling(
                        source=CANDIDATE_EVIDENCE_SOURCE,
                        mode=AutonomyMode.AUTONOMOUS,
                        detail="claimed",
                    ),
                ),
                decided_at=datetime(2026, 8, 1, tzinfo=UTC),
                schema_version=0,
            )
        )

        with pytest.raises(AutonomyActionError, match="effective mode"):
            guard.check(GOAL_ID, "pr_create")
    finally:
        spine.close()


def test_candidate_evidence_ready_restores_the_full_ladder(tmp_path):
    """The ceiling lowers autonomy and nothing else.

    Guards against a cap that silently becomes permanent once the phases that
    should lift it are wired.
    """

    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    resolver = GoalAutonomyResolver(
        spine,
        configured_mode="deliver",
        contract_dir=tmp_path,
        enforced_gate_names=("test",),
        candidate_evidence_ready=True,
    )
    try:
        decision = resolver.decide(_contract(), sandbox_available=True)

        assert CANDIDATE_EVIDENCE_SOURCE not in decision.binding_sources
        assert decision.is_candidate_evidence_aware
    finally:
        spine.close()

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ticket_agent.goal.autonomy import (
    AutonomyActionError,
    AutonomyActionGuard,
    GoalAutonomyResolver,
)
from ticket_agent.goal.spine import SQLiteGoalSpine
from ticket_agent.goal.types import (
    AcceptanceCriterion,
    AuthorizationContext,
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
        assert decision.effective_mode is AutonomyMode.IMPLEMENT
        assert "runtime:derived_gate_enforcement" in decision.binding_sources
    finally:
        spine.close()


def test_action_ceiling_binds_each_concrete_boundary(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    guard = AutonomyActionGuard(spine)
    decided_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
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
            authorized_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
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
) -> None:
    spine.save_autonomy_decision(
        AutonomyDecision(
            goal_id=GOAL_ID,
            contract_version=1,
            effective_mode=mode,
            ceilings=(),
            decided_at=decided_at,
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

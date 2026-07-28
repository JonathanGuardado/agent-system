"""Tests for gate outcomes, goal contracts, and the two authorizations.

These are the fail-closed guarantees the delivery gates will rest on, so the
negative cases matter more than the positive one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ticket_agent.goal.types import (
    AcceptanceCriterion,
    AuthorizationContext,
    AutonomyMode,
    Budgets,
    CandidateAuthorization,
    CriterionOutcome,
    DigestSet,
    GoalAchievement,
    GoalContract,
    LoopState,
    ScopeSpec,
    action_id,
    digest,
    resolve_autonomy,
)
from ticket_agent.orchestrator.gates import (
    GateOutcome,
    ReviewCoverage,
    VerificationPolicy,
    VerificationRecord,
    initial_outcomes,
)

_NOW = datetime(2026, 7, 27, tzinfo=timezone.utc)


def _policy(required=("test",), **kwargs) -> VerificationPolicy:
    return VerificationPolicy(
        schema_version=1, policy_version=1, required_gates=required, **kwargs
    )


def _record(policy: VerificationPolicy, outcomes, **kwargs) -> VerificationRecord:
    return VerificationRecord(
        policy=policy, candidate_sha="a" * 40, outcomes=outcomes, **kwargs
    )


# -- gate authorization ----------------------------------------------------


def test_empty_required_gates_never_authorizes():
    record = _record(_policy(required=()), {})
    assert record.authorized is False


def test_record_with_no_outcomes_denies():
    assert _record(_policy(), {}).authorized is False


def test_all_skipped_record_denies():
    record = _record(
        _policy(required=("test", "lint")),
        {
            "test": GateOutcome(gate="test", status="skipped"),
            "lint": GateOutcome(gate="lint", status="skipped"),
        },
    )
    assert record.authorized is False


def test_not_run_gate_denies():
    record = _record(
        _policy(required=("test", "build")),
        {
            "test": GateOutcome(gate="test", status="passed"),
            "build": GateOutcome(gate="build", status="not_run"),
        },
    )
    assert record.authorized is False


def test_missing_required_gate_denies():
    record = _record(
        _policy(required=("test", "typecheck")),
        {"test": GateOutcome(gate="test", status="passed")},
    )
    assert record.authorized is False
    assert record.first_non_passing.gate == "typecheck"


def test_all_required_passed_authorizes():
    record = _record(
        _policy(required=("test", "lint")),
        {
            "test": GateOutcome(gate="test", status="passed"),
            "lint": GateOutcome(gate="lint", status="passed"),
        },
    )
    assert record.authorized is True


def test_mutation_during_verification_denies():
    record = _record(
        _policy(),
        {"test": GateOutcome(gate="test", status="passed")},
        mutation_detected=True,
    )
    assert record.authorized is False
    assert record.failure_class == "policy"


def test_initial_outcomes_seed_every_expected_gate_as_not_run():
    outcomes = initial_outcomes(("test", "lint", "build"))
    assert {o.status for o in outcomes.values()} == {"not_run"}


def test_passed_gate_cannot_carry_a_failure_class():
    with pytest.raises(ValueError):
        GateOutcome(gate="test", status="passed", failure_class="defect")


def test_gate_cannot_be_required_and_optional():
    with pytest.raises(ValueError):
        VerificationPolicy(
            schema_version=1,
            policy_version=1,
            required_gates=("test",),
            optional_gates=("test",),
        )


# -- review coverage -------------------------------------------------------


def test_partial_review_coverage_is_incomplete():
    assert not ReviewCoverage(files_total=3, files_reviewed=2).complete
    assert not ReviewCoverage(
        files_total=2, files_reviewed=2, truncated=("a.py",)
    ).complete
    assert not ReviewCoverage().complete


def test_full_review_coverage_is_complete():
    coverage = ReviewCoverage(
        files_total=2, files_reviewed=2, hunks_total=5, hunks_reviewed=5
    )
    assert coverage.complete


# -- contract --------------------------------------------------------------


def _contract(**kwargs) -> GoalContract:
    defaults = dict(
        goal_id="g1",
        version=1,
        schema_version=1,
        authorization=AuthorizationContext(
            requester="U1",
            slack_channel="C1",
            slack_message_ts="1.0",
            allowlisted=True,
            authorized_at=_NOW,
        ),
        original_request="Add a landing page",
        objective="Ship a landing page",
        acceptance_criteria=(AcceptanceCriterion("c1", "Page renders"),),
        permitted_scope=ScopeSpec(repositories=("demo",)),
        risk_class="standard",
    )
    defaults.update(kwargs)
    return GoalContract(**defaults)


def test_contract_digest_changes_when_any_field_changes():
    baseline = _contract().contract_digest
    assert _contract(objective="Ship two pages").contract_digest != baseline
    assert _contract(non_goals=("no auth",)).contract_digest != baseline
    assert _contract(risk_class="low").contract_digest != baseline
    assert _contract(budgets=Budgets(iterations=5)).contract_digest != baseline
    assert _contract().contract_digest == baseline


def test_contract_requires_criteria_and_original_request():
    with pytest.raises(ValueError):
        _contract(acceptance_criteria=())
    with pytest.raises(ValueError):
        _contract(original_request="   ")


def test_contract_rejects_duplicate_criterion_ids():
    with pytest.raises(ValueError):
        _contract(
            acceptance_criteria=(
                AcceptanceCriterion("c1", "one"),
                AcceptanceCriterion("c1", "two"),
            )
        )


def test_digest_is_order_independent_for_mappings():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_action_id_is_deterministic():
    assert action_id("g1", 2, "git_push", "LAB-30") == action_id(
        "g1", 2, "git_push", "LAB-30"
    )
    assert action_id("g1", 2, "git_push", "LAB-30") != action_id(
        "g1", 3, "git_push", "LAB-30"
    )


# -- autonomy resolution ---------------------------------------------------


def test_autonomy_is_bounded_by_every_ceiling():
    mode, binding = resolve_autonomy(
        configured="autonomous",
        risk_class="low",
        readiness="full",
        sandbox_available=True,
        all_required_gates_enforced=True,
    )
    assert mode is AutonomyMode.AUTONOMOUS

    mode, binding = resolve_autonomy(
        configured="autonomous",
        risk_class="elevated",
        readiness="full",
        sandbox_available=True,
        all_required_gates_enforced=True,
    )
    assert mode is AutonomyMode.IMPLEMENT
    assert "risk_class" in binding


def test_missing_sandbox_caps_at_propose_unless_commands_are_approved():
    without, binding = resolve_autonomy(
        configured="autonomous",
        risk_class="low",
        readiness="full",
        sandbox_available=False,
        all_required_gates_enforced=True,
    )
    assert without is AutonomyMode.PROPOSE
    assert binding == ("sandbox",)

    with_approval, _ = resolve_autonomy(
        configured="autonomous",
        risk_class="low",
        readiness="full",
        sandbox_available=False,
        per_command_approval=True,
        all_required_gates_enforced=True,
    )
    assert with_approval is AutonomyMode.IMPLEMENT


def test_gate_not_enforced_caps_below_deliver():
    mode, binding = resolve_autonomy(
        configured="autonomous",
        risk_class="low",
        readiness="full",
        sandbox_available=True,
        all_required_gates_enforced=False,
    )
    assert mode is AutonomyMode.IMPLEMENT
    assert "gate_enforcement" in binding
    assert mode < AutonomyMode.DELIVER


def test_unrecognized_mode_fails_closed_to_observe():
    mode, _ = resolve_autonomy(
        configured="turbo",
        risk_class="low",
        readiness="full",
        sandbox_available=True,
        all_required_gates_enforced=True,
    )
    assert mode is AutonomyMode.OBSERVE


def test_halt_lowers_to_observe():
    mode, binding = resolve_autonomy(
        configured="autonomous",
        risk_class="low",
        readiness="full",
        sandbox_available=True,
        all_required_gates_enforced=True,
        halted=True,
    )
    assert mode is AutonomyMode.OBSERVE
    assert "halted" in binding


# -- loop state: phases are not terminal statuses ---------------------------


def test_terminal_status_requires_closed_phase():
    with pytest.raises(ValueError):
        LoopState(
            goal_id="g1",
            contract_version=1,
            phase="ready_for_promotion",
            terminal_status="achieved",
        )


def test_closed_phase_requires_a_terminal_status():
    with pytest.raises(ValueError):
        LoopState(goal_id="g1", contract_version=1, phase="closed")


def test_ready_for_promotion_is_a_non_terminal_phase():
    state = LoopState(goal_id="g1", contract_version=1, phase="ready_for_promotion")
    assert state.is_terminal is False
    assert state.terminal_status is None


# -- the two authorizations -------------------------------------------------


def _authorization(**kwargs) -> CandidateAuthorization:
    record = _record(_policy(), {"test": GateOutcome(gate="test", status="passed")})
    defaults = dict(
        repository="demo",
        goal_id="g1",
        head_sha="a" * 40,
        tree_oid="b" * 40,
        base_branch="integration/g1",
        base_oid="c" * 40,
        merge_base_oid="c" * 40,
        verification=record,
        review_coverage=ReviewCoverage(
            files_total=1, files_reviewed=1, hunks_total=1, hunks_reviewed=1
        ),
        review_verdict="approved",
        digests=DigestSet(contract="x"),
        trust_root_untouched=True,
        secrets_clean=True,
        scope_respected=True,
        binaries_cleared=True,
    )
    defaults.update(kwargs)
    return CandidateAuthorization(**defaults)


def test_candidate_authorization_is_a_positive_conjunction():
    assert _authorization().authorized is True


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"trust_root_untouched": False}, "trust_root_touched"),
        ({"secrets_clean": False}, "secret_detected"),
        ({"scope_respected": False}, "scope_violation"),
        ({"binaries_cleared": False}, "binary_or_submodule_change"),
        ({"review_verdict": "rejected"}, "review_rejected"),
        ({"review_coverage": ReviewCoverage()}, "review_coverage_incomplete"),
        ({"head_sha": "d" * 40}, "verified_sha_mismatch"),
    ],
)
def test_any_failing_clause_denies_and_is_named(override, reason):
    authorization = _authorization(**override)
    assert authorization.authorized is False
    assert reason in authorization.denial_reasons()


def test_goal_achievement_requires_a_confirmed_promotion_pr():
    """Evidence that criteria are met is not the same as having presented it."""

    met = {"c1": CriterionOutcome("c1", met=True, oracle="deterministic")}
    ready = GoalAchievement(
        goal_id="g1",
        contract_version=1,
        criteria=met,
        non_goals_respected=True,
        integration_checkpoint_passed=True,
    )
    assert ready.ready_for_promotion is True
    assert ready.achieved is False

    opened = GoalAchievement(
        goal_id="g1",
        contract_version=1,
        criteria=met,
        non_goals_respected=True,
        integration_checkpoint_passed=True,
        promotion_pr_url="https://github.test/x/y/pull/1",
        promotion_pr_confirmed=False,
    )
    assert opened.achieved is False, "an unconfirmed PR must not mark a goal achieved"

    confirmed = GoalAchievement(
        goal_id="g1",
        contract_version=1,
        criteria=met,
        non_goals_respected=True,
        integration_checkpoint_passed=True,
        promotion_pr_url="https://github.test/x/y/pull/1",
        promotion_pr_confirmed=True,
    )
    assert confirmed.achieved is True


def test_goal_achievement_denies_on_unmet_criteria_and_failed_demo():
    base = dict(
        goal_id="g1",
        contract_version=1,
        non_goals_respected=True,
        integration_checkpoint_passed=True,
        promotion_pr_url="u",
        promotion_pr_confirmed=True,
    )
    unmet = GoalAchievement(
        criteria={"c1": CriterionOutcome("c1", met=False)}, **base
    )
    assert unmet.achieved is False

    failed_demo = GoalAchievement(
        criteria={"c1": CriterionOutcome("c1", met=True)},
        cumulative_demo_passed=False,
        **base,
    )
    assert failed_demo.achieved is False


def test_goal_achievement_detects_uncovered_criteria():
    contract = _contract(
        acceptance_criteria=(
            AcceptanceCriterion("c1", "one"),
            AcceptanceCriterion("c2", "two"),
        )
    )
    partial = GoalAchievement(
        goal_id="g1",
        contract_version=1,
        criteria={"c1": CriterionOutcome("c1", met=True)},
    )
    assert partial.covers(contract) is False


def test_empty_criteria_are_not_ready_for_promotion():
    """A goal with no evaluated criteria must not pass vacuously."""

    assert (
        GoalAchievement(
            goal_id="g1",
            contract_version=1,
            non_goals_respected=True,
            integration_checkpoint_passed=True,
        ).ready_for_promotion
        is False
    )

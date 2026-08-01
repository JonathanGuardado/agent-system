from __future__ import annotations

import asyncio

import pytest

from ticket_agent.goal.journal import (
    ActionJournalError,
    GoalActionJournal,
    InjectedJournalCrash,
    JournaledModelRouter,
    ProbeResult,
)
from ticket_agent.goal.spine import SQLiteGoalSpine
from ticket_agent.goal.types import OPERATION_POLICIES, LoopState

GOAL_ID = "prop-0123456789ab"
OPERATIONS = tuple(OPERATION_POLICIES)
CRASH_POINTS = (
    "before_reservation",
    "after_reservation",
    "after_external_effect",
    "after_completion",
)


def test_operation_policy_covers_the_complete_production_effect_map():
    assert set(OPERATION_POLICIES) == {
        "jira_write:create_issue",
        "jira_write:add_ai_ready",
        "jira_write:transition",
        "jira_write:comment",
        "slack_post",
        "worktree_create",
        "git_commit",
        "git_push",
        "pr_create",
        "pr_merge",
        "model_invoke",
        "gate_run",
    }
    assert OPERATION_POLICIES["jira_write:create_issue"].probe_required is True
    assert OPERATION_POLICIES["slack_post"].probe_required is False
    assert (
        OPERATION_POLICIES["model_invoke"].charge_reservation_on_ambiguity
        is True
    )


def test_goal_scoped_model_router_reserves_records_cost_and_blocks_replay(tmp_path):
    class Response:
        content = "model output"
        estimated_cost_usd = 0.25

    class Delegate:
        def __init__(self):
            self.calls = 0

        async def invoke(self, capability, messages, **kwargs):
            del capability, messages, kwargs
            self.calls += 1
            return Response()

    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    delegate = Delegate()
    router = JournaledModelRouter(
        delegate,
        GoalActionJournal(spine, lease_owner="worker-1"),
        reservation_usd=1.5,
    )
    kwargs = {
        "capability": "code.implement",
        "messages": [{"role": "user", "content": "implement it"}],
        "metadata": {
            "goal_id": GOAL_ID,
            "goal_iteration": 3,
            "workflow_node": "implement",
        },
    }

    try:
        response = asyncio.run(router.invoke(**kwargs))
        assert response.content == "model output"

        with pytest.raises(ActionJournalError, match="no replayable response"):
            asyncio.run(router.invoke(**kwargs))

        assert delegate.calls == 1
        action = spine.actions_for_goal(GOAL_ID)[0]
        assert action.operation == "model_invoke"
        assert spine.budget_for_action(action.action_id) == (3.0, 0.25)
    finally:
        spine.close()


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("crash_point", CRASH_POINTS)
def test_concrete_operation_crash_matrix_obeys_duplicate_and_budget_bounds(
    tmp_path,
    operation,
    crash_point,
):
    spine = SQLiteGoalSpine(tmp_path / "spine.sqlite3")
    state = LoopState(
        goal_id=GOAL_ID,
        contract_version=1,
        phase="implementing",
        iteration=1,
    )
    external = {"applied": False, "effects": 0}

    def crash_hook(point, record):
        del record
        if point == crash_point:
            raise InjectedJournalCrash(point)

    async def effect():
        external["effects"] += 1
        external["applied"] = True
        return f"result:{operation}"

    async def probe():
        return ProbeResult(
            found=bool(external["applied"]),
            value=f"result:{operation}" if external["applied"] else None,
            result_identity=f"result:{operation}" if external["applied"] else None,
        )

    kwargs = {
        "operation": operation,
        "natural_key": f"key:{operation}",
        "request": {"operation": operation, "payload": "same"},
        "effect": effect,
        "probe": probe if OPERATION_POLICIES[operation].probe_required else None,
        "reserved_model_cost_usd": 2.5 if operation == "model_invoke" else 0.0,
        "actual_model_cost_usd": (
            (lambda value: 0.25) if operation == "model_invoke" else None
        ),
    }

    try:
        crashing = GoalActionJournal(
            spine,
            lease_owner="worker-crash",
            crash_hook=crash_hook,
        )
        with pytest.raises(InjectedJournalCrash):
            asyncio.run(crashing.execute(state, **kwargs))

        recovering = GoalActionJournal(spine, lease_owner="worker-recovery")
        outcome = asyncio.run(recovering.execute(state, **kwargs))

        assert outcome.record.state == "done"
        expected_effects = (
            2
            if crash_point == "after_external_effect"
            and not OPERATION_POLICIES[operation].probe_required
            else 1
        )
        assert external["effects"] == expected_effects

        # A completed action is immutable intent: replay never calls the
        # external boundary again, regardless of probe support.
        asyncio.run(recovering.execute(state, **kwargs))
        assert external["effects"] == expected_effects

        actions = spine.actions_for_goal(GOAL_ID)
        assert len(actions) == 1
        assert actions[0].attempts <= 2
        if operation == "model_invoke":
            reserved, actual = spine.budget_for_action(actions[0].action_id)
            assert reserved == 5.0
            assert actual == (
                2.75 if crash_point == "after_external_effect" else 0.25
            )
    finally:
        spine.close()

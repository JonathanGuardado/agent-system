from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ticket_agent.goal.identity import (
    GoalIdentityError,
    goal_id_from_labels,
    goal_label,
    normalize_goal_id,
    validate_goal_labels,
)
from ticket_agent.goal.spine import GoalSpineError, SQLiteGoalSpine
from ticket_agent.goal.types import ActionRecord, LoopState, action_id, digest
from ticket_agent.intake.jira_writer import JiraWriter
from ticket_agent.jira.constants import LABEL_AI_READY
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket

GOAL_ID = "prop-0123456789ab"


def test_goal_identity_is_proposal_id_used_verbatim():
    assert normalize_goal_id(GOAL_ID) == GOAL_ID
    assert goal_label(GOAL_ID) == f"ai-goal-{GOAL_ID}"
    assert goal_id_from_labels([goal_label(GOAL_ID)]) == GOAL_ID
    assert validate_goal_labels([goal_label(GOAL_ID)], goal_id=GOAL_ID) == GOAL_ID


@pytest.mark.parametrize(
    "value",
    (
        "PROP-0123456789ab",
        "prop-0123456789a",
        "prop-0123456789AB",
        "01234567-89ab-cdef-0123-456789abcdef",
        "prop-0123456789ab-extra",
    ),
)
def test_goal_identity_rejects_noncanonical_values(value):
    with pytest.raises(GoalIdentityError):
        normalize_goal_id(value)


@pytest.mark.parametrize(
    "labels",
    (
        (),
        (f"ai-goal-{GOAL_ID}", f"ai-goal-{GOAL_ID}"),
        (f"ai-goal-{GOAL_ID}", "ai-goal-prop-abcdefabcdef"),
        (f"AI-GOAL-{GOAL_ID}",),
        ("ai-goal-prop-0123456789a",),
    ),
)
def test_goal_label_lookup_fails_closed_on_missing_duplicate_or_malformed(labels):
    with pytest.raises(GoalIdentityError):
        goal_id_from_labels(labels)


def test_goal_state_action_and_budget_reservation_share_atomic_commit(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "goal-spine.sqlite3")
    state = LoopState(
        goal_id=GOAL_ID,
        contract_version=1,
        phase="discovering",
    )
    record = _action(request_digest="request-a", reserved=1.25)

    try:
        durable = spine.reserve_action(state, record)
        assert durable.state == "intended"
        assert spine.load_loop_state(GOAL_ID) == state
        assert spine.budget_for_action(record.action_id) == (1.25, 0.0)

        in_flight = spine.mark_in_flight(
            record.action_id,
            lease_owner="worker-1",
        )
        assert in_flight.state == "in_flight"
        assert in_flight.attempts == 1
        assert in_flight.lease_owner == "worker-1"

        done = spine.mark_done(
            record.action_id,
            result_identity="AGENT-1",
            actual_model_cost_usd=0.75,
        )
        assert done.state == "done"
        assert done.result_identity == "AGENT-1"
        assert spine.budget_for_action(record.action_id) == (1.25, 0.75)
    finally:
        spine.close()


def test_conflicting_action_rolls_back_goal_state_transition(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "goal-spine.sqlite3")
    initial = LoopState(
        goal_id=GOAL_ID,
        contract_version=1,
        phase="discovering",
    )
    record = _action(request_digest="request-a")

    try:
        spine.reserve_action(initial, record)
        conflicting = replace(record, request_digest="request-b")
        advanced = LoopState(
            goal_id=GOAL_ID,
            contract_version=1,
            phase="implementing",
        )

        with pytest.raises(GoalSpineError, match="conflicts"):
            spine.reserve_action(advanced, conflicting)

        assert spine.load_loop_state(GOAL_ID) == initial
    finally:
        spine.close()


def test_ambiguous_ai_ready_write_recovers_by_probe_without_duplicate(tmp_path):
    spine = SQLiteGoalSpine(tmp_path / "goal-spine.sqlite3")
    ticket = JiraTicket(
        key="AGENT-1",
        summary="Ready",
        labels=[LABEL_AI_READY],
    )
    client = FakeJiraClient(ticket)
    writer = JiraWriter(client, goal_spine=spine, component_id="worker-1")
    record = _action_for_ticket(ticket.key)

    try:
        spine.reserve_action(
            LoopState(
                goal_id=GOAL_ID,
                contract_version=1,
                phase="discovering",
            ),
            record,
        )
        spine.mark_in_flight(record.action_id, lease_owner="crashed-worker")

        asyncio.run(
            writer._publish_ai_ready(
                JiraTicket(key=ticket.key, summary=ticket.summary, labels=[]),
                goal_id=GOAL_ID,
            )
        )

        add_calls = [call for call in client.calls if call[0] == "add_labels"]
        assert add_calls == []
        durable = spine.load_action(record.action_id)
        assert durable is not None
        assert durable.state == "done"
        assert durable.recovery_classification == "label_presence_probe"
    finally:
        spine.close()


def _action(*, request_digest: str, reserved: float = 0.0) -> ActionRecord:
    operation = "jira_write:add_ai_ready"
    natural_key = "AGENT-1"
    return ActionRecord(
        action_id=action_id(GOAL_ID, 0, operation, natural_key),
        goal_id=GOAL_ID,
        iteration=0,
        kind="jira_write",
        state="intended",
        operation=operation,
        natural_key=natural_key,
        request_digest=request_digest,
        external=True,
        reserved_model_cost_usd=reserved,
    )


def _action_for_ticket(ticket_key: str) -> ActionRecord:
    operation = "jira_write:add_ai_ready"
    return ActionRecord(
        action_id=action_id(GOAL_ID, 0, operation, ticket_key),
        goal_id=GOAL_ID,
        iteration=0,
        kind="jira_write",
        state="intended",
        operation=operation,
        natural_key=ticket_key,
        request_digest=digest(
            {"ticket_key": ticket_key, "label": LABEL_AI_READY}
        ),
        external=True,
    )

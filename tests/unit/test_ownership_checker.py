from __future__ import annotations

import pytest

from ticket_agent.detection.ownership import OwnershipChecker
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_MAX_ATTEMPTS,
    LABEL_AI_READY,
    LABEL_DO_NOT_AUTOMATE,
    STATUS_IN_PROGRESS,
    STATUS_TODO,
)
from ticket_agent.jira.models import JiraTicket


COMPONENT_ID = "agent-system"


def _ticket(**overrides) -> JiraTicket:
    base = {
        "key": "AGENT-1",
        "summary": "Test ticket",
        "description": "",
        "status": STATUS_TODO,
        "labels": [LABEL_AI_READY],
        "assignee": None,
        "fields": {},
    }
    base.update(overrides)
    return JiraTicket(**base)


def _checker(*, lock_lookup=None, max_retries=3) -> OwnershipChecker:
    return OwnershipChecker(
        component_id=COMPONENT_ID,
        lock_lookup=lock_lookup or (lambda key: None),
        max_retries=max_retries,
    )


def test_eligible_ticket_returns_eligible_decision():
    decision = _checker().check(_ticket())

    assert decision.eligible is True
    assert decision.reason == ""


def test_r0_do_not_automate_label_skips_silently():
    ticket = _ticket(labels=[LABEL_AI_READY, LABEL_DO_NOT_AUTOMATE])

    decision = _checker().check(ticket)

    assert decision.eligible is False
    assert decision.reason == ""


def test_r1_human_assignee_skips_with_reason():
    ticket = _ticket(assignee="alice@example.com")

    decision = _checker().check(ticket)

    assert decision.eligible is False
    assert decision.reason == "human_assigned"


def test_r2_different_component_skips_with_reason():
    ticket = _ticket(
        fields={FIELD_AGENT_ASSIGNED_COMPONENT: "other-component"},
    )

    decision = _checker().check(ticket)

    assert decision.eligible is False
    assert decision.reason == "different_component"


def test_r2_matching_component_does_not_skip():
    ticket = _ticket(
        fields={FIELD_AGENT_ASSIGNED_COMPONENT: COMPONENT_ID},
    )

    decision = _checker().check(ticket)

    assert decision.eligible is True


def test_r3_missing_ai_ready_label_skips_with_reason():
    ticket = _ticket(labels=[])

    decision = _checker().check(ticket)

    assert decision.eligible is False
    assert decision.reason == "missing_ai_ready"


def test_r4_wrong_status_skips_with_reason():
    ticket = _ticket(status=STATUS_IN_PROGRESS)

    decision = _checker().check(ticket)

    assert decision.eligible is False
    assert decision.reason == f"wrong_status:{STATUS_IN_PROGRESS}"


def test_r5_active_lock_skips_with_reason():
    ticket = _ticket()

    decision = _checker(
        lock_lookup=lambda key: {"owner": "other-worker"}
    ).check(ticket)

    assert decision.eligible is False
    assert decision.reason == "active_lock"


def test_r6_blocked_by_unresolved_issue_skips_with_reason():
    ticket = _ticket(fields={"blocked_by": ["AGENT-99"]})

    decision = _checker().check(ticket)

    assert decision.eligible is False
    assert decision.reason == "blocked_by:AGENT-99"


def test_r6_resolved_blocking_issue_does_not_skip():
    ticket = _ticket(
        fields={
            "blocked_by": [
                {"key": "AGENT-99", "status": "Done"},
            ]
        }
    )

    decision = _checker().check(ticket)

    assert decision.eligible is True


def test_r7_retry_limit_reached_skips_with_reason():
    ticket = _ticket(
        fields={FIELD_AGENT_RETRY_COUNT: 3},
    )

    decision = _checker(max_retries=3).check(ticket)

    assert decision.eligible is False
    assert decision.reason == "retry_limit_reached"


def test_r7_uses_ticket_max_attempts_field_when_present():
    ticket = _ticket(
        fields={FIELD_AGENT_RETRY_COUNT: 5, FIELD_MAX_ATTEMPTS: 6},
    )

    decision = _checker(max_retries=3).check(ticket)

    assert decision.eligible is True


def test_eligibility_does_not_require_ai_execution_approved_label():
    ticket = _ticket(labels=[LABEL_AI_READY])

    decision = _checker().check(ticket)

    assert decision.eligible is True


def test_invalid_component_id_raises():
    with pytest.raises(ValueError):
        OwnershipChecker(component_id="", lock_lookup=lambda k: None)


def test_lock_lookup_exception_treated_as_no_lock():
    def raising(key: str):
        raise RuntimeError("db unavailable")

    decision = _checker(lock_lookup=raising).check(_ticket())

    assert decision.eligible is True

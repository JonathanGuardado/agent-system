"""Tests for TicketState's strict-extra policy and new workflow statuses.

Pydantic v2 defaults to ``extra="ignore"``. Under that default a node that
returns an undeclared field has its update silently discarded: the graph
advances, the value is missing downstream, and the ticket escalates with no
explanation of why. ``extra="forbid"`` converts that into a loud error at the
point of the mistake.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from ticket_agent.orchestrator.state import TicketState


def _state(**kwargs) -> TicketState:
    return TicketState(ticket_key="LAB-30", summary="Do a thing", **kwargs)


def test_undeclared_field_raises_instead_of_being_silently_dropped():
    with pytest.raises(ValidationError) as excinfo:
        _state(candidate_shaa="a" * 40)

    assert "candidate_shaa" in str(excinfo.value)


def test_declared_verification_fields_round_trip():
    state = _state(
        goal_id="g1",
        candidate_sha="a" * 40,
        verification_record={"authorized": False},
        verification_attempts=2,
    )

    assert state.goal_id == "g1"
    assert state.candidate_sha == "a" * 40
    assert state.verification_record == {"authorized": False}
    assert state.verification_attempts == 2


def test_verification_attempts_defaults_to_zero():
    assert _state().verification_attempts == 0


@pytest.mark.parametrize("status", ["committing", "verifying"])
def test_new_workflow_statuses_are_accepted(status):
    assert _state(workflow_status=status).workflow_status == status


def test_unknown_workflow_status_is_rejected():
    with pytest.raises(ValidationError):
        _state(workflow_status="teleporting")

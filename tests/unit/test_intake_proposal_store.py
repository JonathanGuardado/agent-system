from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ticket_agent.domain.intake import (
    IntakeMode,
    Proposal,
    ProposalStatus,
    TicketSpec,
)
from ticket_agent.intake.proposal_store import PROPOSAL_TTL_SECONDS, ProposalStore


def _make_proposal(
    *,
    proposal_id: str = "prop-1",
    user_id: str = "U123",
    thread_ts: str = "1700000000.0001",
    status: ProposalStatus = ProposalStatus.AWAITING_CONFIRMATION,
    created_at: datetime | None = None,
    expires_in_seconds: int = PROPOSAL_TTL_SECONDS,
    project_key: str | None = "AGENT",
    revision_count: int = 0,
) -> Proposal:
    created = created_at or datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    return Proposal(
        proposal_id=proposal_id,
        slack_user_id=user_id,
        slack_thread_ts=thread_ts,
        mode=IntakeMode.NEW_FEATURE,
        project_key=project_key,
        epic_key=None,
        title="Add OAuth login",
        summary="Add OAuth login to AGENT",
        tickets=[
            TicketSpec(
                summary="Add OAuth login",
                description="Add OAuth login to AGENT",
                labels=["ai-ready"],
                capabilities_needed=["code.implement"],
                repository="agent-system",
                repo_path="/home/agent",
            )
        ],
        revision_count=revision_count,
        status=status,
        created_at=created,
        expires_at=created + timedelta(seconds=expires_in_seconds),
    )


def test_save_and_get_active_for_thread(tmp_path):
    db_path = tmp_path / "proposals.db"
    store = ProposalStore(db_path)
    proposal = _make_proposal()

    store.save(proposal)
    loaded = store.get_active_for_thread(proposal.slack_user_id, proposal.slack_thread_ts)

    assert loaded is not None
    assert loaded.proposal_id == proposal.proposal_id
    assert loaded.tickets[0].labels == ["ai-ready"]


def test_update_replaces_proposal(tmp_path):
    db_path = tmp_path / "proposals.db"
    store = ProposalStore(db_path)
    proposal = _make_proposal()
    store.save(proposal)

    revised = proposal.model_copy(
        update={"revision_count": 1, "summary": "Updated summary"}
    )
    store.update(revised)

    loaded = store.get(proposal.proposal_id)
    assert loaded is not None
    assert loaded.revision_count == 1
    assert loaded.summary == "Updated summary"


def test_update_unknown_proposal_raises(tmp_path):
    store = ProposalStore(tmp_path / "proposals.db")
    with pytest.raises(KeyError):
        store.update(_make_proposal(proposal_id="missing"))


def test_mark_status_changes_status_and_excludes_from_active(tmp_path):
    store = ProposalStore(tmp_path / "proposals.db")
    proposal = _make_proposal()
    store.save(proposal)

    store.mark_status(proposal.proposal_id, ProposalStatus.CONFIRMED)

    assert store.get_active_for_thread(proposal.slack_user_id, proposal.slack_thread_ts) is None
    loaded = store.get(proposal.proposal_id)
    assert loaded is not None
    assert loaded.status == ProposalStatus.CONFIRMED


def test_expire_old_marks_expired(tmp_path):
    created = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    store = ProposalStore(tmp_path / "proposals.db")
    proposal = _make_proposal(
        created_at=created,
        expires_in_seconds=10,
    )
    store.save(proposal)

    later = created + timedelta(seconds=11)
    expired = store.expire_old(now=later)

    assert expired == 1
    assert store.get_active_for_thread(
        proposal.slack_user_id, proposal.slack_thread_ts
    ) is None


def test_get_active_returns_none_for_terminal_status(tmp_path):
    store = ProposalStore(tmp_path / "proposals.db")
    proposal = _make_proposal(status=ProposalStatus.CANCELLED)
    store.save(proposal)

    assert (
        store.get_active_for_thread(
            proposal.slack_user_id, proposal.slack_thread_ts
        )
        is None
    )

from __future__ import annotations

import sqlite3

from scripts import system_reset


def test_system_reset_dry_run_does_not_delete_local_ticket_state(tmp_path):
    _seed_runtime_dbs(tmp_path)

    exit_code = system_reset.main(
        [
            "--data-dir",
            str(tmp_path),
            "--ticket",
            "AGENT-123",
            "--skip-jira",
        ]
    )

    assert exit_code == 0
    assert _count(tmp_path / "execution_approvals.sqlite3", "execution_approvals") == 1
    assert _count(tmp_path / "ticket_locks.sqlite3", "ticket_locks") == 1
    assert _count(tmp_path / "ticket_graph_checkpoints.sqlite3", "checkpoints") == 1
    assert (
        _count(tmp_path / "ticket_graph_checkpoints.sqlite3", "checkpoint_writes")
        == 1
    )


def test_system_reset_confirm_deletes_local_ticket_state(tmp_path):
    _seed_runtime_dbs(tmp_path)

    exit_code = system_reset.main(
        [
            "--data-dir",
            str(tmp_path),
            "--ticket",
            "AGENT-123",
            "--skip-jira",
            "--confirm",
        ]
    )

    assert exit_code == 0
    assert _count(tmp_path / "execution_approvals.sqlite3", "execution_approvals") == 0
    assert _count(tmp_path / "ticket_locks.sqlite3", "ticket_locks") == 0
    assert _count(tmp_path / "ticket_graph_checkpoints.sqlite3", "checkpoints") == 0
    assert (
        _count(tmp_path / "ticket_graph_checkpoints.sqlite3", "checkpoint_writes")
        == 0
    )


def test_system_reset_all_local_includes_proposals_only_when_requested(tmp_path):
    _seed_runtime_dbs(tmp_path)

    exit_code = system_reset.main(
        [
            "--data-dir",
            str(tmp_path),
            "--all-local",
            "--skip-jira",
            "--include-proposals",
            "--confirm",
        ]
    )

    assert exit_code == 0
    assert _count(tmp_path / "intake_proposals.sqlite3", "active_proposals") == 0


def _seed_runtime_dbs(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "execution_approvals.sqlite3") as con:
        con.execute(
            "CREATE TABLE execution_approvals ("
            "ticket_key TEXT PRIMARY KEY, slack_channel TEXT, slack_thread_ts TEXT, "
            "plan_summary TEXT, status TEXT, created_at TEXT, expires_at TEXT)"
        )
        con.execute(
            "INSERT INTO execution_approvals VALUES "
            "('AGENT-123', 'C1', 'T1', 'plan', 'pending', 'now', 'later')"
        )

    with sqlite3.connect(tmp_path / "ticket_locks.sqlite3") as con:
        con.execute(
            "CREATE TABLE ticket_locks ("
            "ticket_key TEXT PRIMARY KEY, lock_id TEXT, component_id TEXT, "
            "acquired_at TEXT, expires_at TEXT, last_heartbeat TEXT)"
        )
        con.execute(
            "INSERT INTO ticket_locks VALUES "
            "('AGENT-123', 'lock-1', 'runner', 'now', 'later', 'now')"
        )

    with sqlite3.connect(tmp_path / "ticket_graph_checkpoints.sqlite3") as con:
        con.execute(
            "CREATE TABLE checkpoints ("
            "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, "
            "metadata_type TEXT, metadata BLOB, "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
        )
        con.execute(
            "CREATE TABLE checkpoint_writes ("
            "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, blob BLOB, "
            "task_path TEXT, "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))"
        )
        con.execute(
            "INSERT INTO checkpoints VALUES "
            "('AGENT-123', '', 'cp-1', NULL, 'json', X'00', 'json', X'00')"
        )
        con.execute(
            "INSERT INTO checkpoint_writes VALUES "
            "('AGENT-123', '', 'cp-1', 'task-1', 0, 'messages', 'json', X'00', '')"
        )

    with sqlite3.connect(tmp_path / "intake_proposals.sqlite3") as con:
        con.execute(
            "CREATE TABLE active_proposals ("
            "proposal_id TEXT PRIMARY KEY, slack_user_id TEXT, slack_thread_ts TEXT, "
            "mode TEXT, jira_project_key TEXT, jira_epic_key TEXT, status TEXT, "
            "proposal_json TEXT, revision_count INTEGER, created_at TEXT, "
            "expires_at TEXT)"
        )
        con.execute(
            "INSERT INTO active_proposals VALUES "
            "('prop-1', 'U1', 'T1', 'new_feature', 'AGENT', NULL, "
            "'awaiting_confirmation', '{}', 0, 'now', 'later')"
        )


def _count(path, table: str) -> int:
    with sqlite3.connect(path) as con:
        row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from typing import Any

from ticket_agent.locking.checkpointer import SQLiteCheckpointer
from ticket_agent.orchestrator.graph import (
    TicketWorkflowNodes,
    build_persistent_ticket_graph,
    build_ticket_graph,
)
from ticket_agent.orchestrator.nodes import implement_ticket
from ticket_agent.orchestrator.state import TicketState


def test_graph_run_writes_checkpoint(tmp_path):
    checkpointer = SQLiteCheckpointer(tmp_path / "checkpoints.sqlite3")
    graph = build_ticket_graph(_nodes(), checkpointer=checkpointer)

    try:
        asyncio.run(graph.ainvoke(_state(), config=_config("AGENT-123")))

        checkpoints = list(checkpointer.list(_config("AGENT-123")))
        assert checkpoints
    finally:
        checkpointer.close()


def test_second_checkpointer_can_load_prior_checkpoint_for_same_ticket(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite3"
    checkpointer = SQLiteCheckpointer(db_path)
    graph = build_ticket_graph(_nodes(), checkpointer=checkpointer)

    try:
        asyncio.run(graph.ainvoke(_state(), config=_config("AGENT-123")))
    finally:
        checkpointer.close()

    reopened = SQLiteCheckpointer(db_path)
    try:
        assert reopened.get_tuple(_config("AGENT-123")) is not None
        assert list(reopened.list(_config("AGENT-123")))
    finally:
        reopened.close()


def test_production_graph_builder_uses_sqlite_checkpointer(tmp_path):
    db_path = tmp_path / "production-checkpoints.sqlite3"
    graph = build_persistent_ticket_graph(_nodes(), checkpoint_db_path=db_path)

    asyncio.run(graph.ainvoke(_state(), config=_config("AGENT-123")))

    reopened = SQLiteCheckpointer(db_path)
    try:
        assert reopened.get_tuple(_config("AGENT-123")) is not None
    finally:
        reopened.close()


def test_delete_thread_removes_checkpoints_and_pending_writes_for_ticket_only(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite3"
    checkpointer = SQLiteCheckpointer(db_path)
    graph = build_ticket_graph(_nodes(), checkpointer=checkpointer)

    try:
        asyncio.run(graph.ainvoke(_state("AGENT-123"), config=_config("AGENT-123")))
        asyncio.run(graph.ainvoke(_state("AGENT-999"), config=_config("AGENT-999")))

        assert _row_count(db_path, "checkpoint_writes", "AGENT-123") > 0
        assert _row_count(db_path, "checkpoint_writes", "AGENT-999") > 0

        checkpointer.delete_thread("AGENT-123")

        assert list(checkpointer.list(_config("AGENT-123"))) == []
        assert list(checkpointer.list(_config("AGENT-999")))
        assert _row_count(db_path, "checkpoints", "AGENT-123") == 0
        assert _row_count(db_path, "checkpoint_writes", "AGENT-123") == 0
        assert _row_count(db_path, "checkpoints", "AGENT-999") > 0
        assert _row_count(db_path, "checkpoint_writes", "AGENT-999") > 0
    finally:
        checkpointer.close()


def _nodes() -> TicketWorkflowNodes:
    return TicketWorkflowNodes(
        plan=_stub("plan", workflow_status="planned"),
        request_execution_approval=_stub(
            "request_execution_approval",
            execution_approved=True,
        ),
        implement=implement_ticket,
        run_tests=_stub("run_tests", tests_passed=True),
        review=_stub("review", review_passed=True),
        open_pull_request=_stub("open_pull_request", pull_request_url="https://pr"),
        report=_stub("report", workflow_status="completed"),
    )


def _state(ticket_key: str = "AGENT-123") -> TicketState:
    return TicketState(ticket_key=ticket_key, summary="Add checkpoint tests")


def _config(ticket_key: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": ticket_key}}


def _stub(name: str, **updates: Any):
    async def node(state: TicketState) -> Mapping[str, Any]:
        return {
            "current_node": name,
            "visited_nodes": [*state.visited_nodes, name],
            **updates,
        }

    return node


def _row_count(db_path, table: str, ticket_key: str) -> int:
    tables = {
        "checkpoints": "checkpoints",
        "checkpoint_writes": "checkpoint_writes",
    }
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {tables[table]} WHERE thread_id = ?",
            (ticket_key,),
        ).fetchone()
    return int(row[0])

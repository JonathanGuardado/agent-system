"""SQLite checkpointer for LangGraph workflow state persistence.

Stores checkpoints in a separate SQLite WAL database so that workflow state
survives process restarts. Designed to share the same data directory as the
ticket lock store but uses its own file to avoid write contention.

Thread-safety: all public methods are protected by an RLock.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class SQLiteCheckpointer(BaseCheckpointSaver):
    """Checkpoint saver backed by a SQLite WAL database.

    Implements the full BaseCheckpointSaver interface including sync and async
    variants. The async variants delegate to their sync counterparts since the
    underlying SQLite operations are fast and I/O is bounded by local disk.

    The ``delete_thread`` method acts as the stale-checkpoint guard: the
    OrchestratorRunner calls it after acquiring a fresh lock so that no
    checkpoint from a prior crashed run can be resumed.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
    ) -> None:
        super().__init__()
        self._conn_lock = RLock()
        self._conn = _connect(db_path, busy_timeout_ms)
        self._initialize()

    def close(self) -> None:
        with self._conn_lock:
            self._conn.close()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str | None = get_checkpoint_id(config)

        with self._conn_lock:
            if checkpoint_id:
                row = self._conn.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint,
                           metadata_type, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint,
                           metadata_type, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()

        if row is None:
            return None

        actual_id = row["checkpoint_id"]
        checkpoint = self.serde.loads_typed((row["type"], row["checkpoint"]))
        metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))
        parent_id = row["parent_checkpoint_id"]

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": actual_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=self._load_writes(thread_id, checkpoint_ns, actual_id),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        sql = """
            SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                   type, checkpoint, metadata_type, metadata
            FROM checkpoints
        """
        params: list[Any] = []
        conditions: list[str] = []

        if config:
            conditions.append("thread_id = ?")
            params.append(config["configurable"]["thread_id"])
            ns = config["configurable"].get("checkpoint_ns")
            if ns is not None:
                conditions.append("checkpoint_ns = ?")
                params.append(ns)

        if before:
            before_id = get_checkpoint_id(before)
            if before_id:
                conditions.append("checkpoint_id < ?")
                params.append(before_id)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY checkpoint_id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        with self._conn_lock:
            rows = self._conn.execute(sql, params).fetchall()

        for row in rows:
            checkpoint = self.serde.loads_typed((row["type"], row["checkpoint"]))
            metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))

            if filter and not all(
                metadata.get(k) == v for k, v in filter.items()
            ):
                continue

            tid = row["thread_id"]
            ns = row["checkpoint_ns"]
            cid = row["checkpoint_id"]
            parent_cid = row["parent_checkpoint_id"]

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_ns": ns,
                        "checkpoint_id": cid,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": tid,
                            "checkpoint_ns": ns,
                            "checkpoint_id": parent_cid,
                        }
                    }
                    if parent_cid
                    else None
                ),
                pending_writes=self._load_writes(tid, ns, cid),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        chk_type, chk_data = self.serde.dumps_typed(checkpoint)
        meta_type, meta_data = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )

        with self._conn_lock, _write_transaction(self._conn):
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                     type, checkpoint, metadata_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    chk_type,
                    chk_data,
                    meta_type,
                    meta_data,
                ),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._conn_lock, _write_transaction(self._conn):
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                w_type, w_data = self.serde.dumps_typed(value)
                if write_idx < 0:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoint_writes
                            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                             channel, type, blob, task_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id, checkpoint_ns, checkpoint_id, task_id,
                            write_idx, channel, w_type, w_data, task_path,
                        ),
                    )
                else:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO checkpoint_writes
                            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                             channel, type, blob, task_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id, checkpoint_ns, checkpoint_id, task_id,
                            write_idx, channel, w_type, w_data, task_path,
                        ),
                    )

    def get_next_version(self, current: str | None, channel: Any) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread (ticket_key).

        Called by OrchestratorRunner after acquiring a fresh lock so that any
        checkpoint from a prior crashed run is not resumed.
        """
        with self._conn_lock, _write_transaction(self._conn):
            self._conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            self._conn.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = ?",
                (thread_id,),
            )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def _initialize(self) -> None:
        with self._conn_lock, _write_transaction(self._conn):
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT NOT NULL,
                    checkpoint BLOB NOT NULL,
                    metadata_type TEXT NOT NULL,
                    metadata BLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
                ON checkpoints (thread_id, checkpoint_ns)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BLOB NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
                """
            )

    def _load_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        with self._conn_lock:
            rows = self._conn.execute(
                """
                SELECT task_id, channel, type, blob
                FROM checkpoint_writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ORDER BY idx
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
        return [
            (
                row["task_id"],
                row["channel"],
                self.serde.loads_typed((row["type"], row["blob"])),
            )
            for row in rows
        ]


def _connect(db_path: str | Path, busy_timeout_ms: int) -> sqlite3.Connection:
    path = Path(db_path)
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def _write_transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


__all__ = ["SQLiteCheckpointer"]

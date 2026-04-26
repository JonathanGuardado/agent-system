"""Git-domain data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorktreeInfo:
    """A local git worktree prepared for a ticket."""

    repo_path: Path
    worktree_path: Path
    branch_name: str
    ticket_key: str
    lock_id: str

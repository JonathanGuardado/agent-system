from __future__ import annotations

import pytest

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.domain.errors import PathBoundaryError


def test_file_adapter_reads_and_writes_inside_worktree(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    adapter = LocalFileAdapter(worktree)

    adapter.write_text("src/example.py", "print('hello')\n")

    assert adapter.exists("src/example.py")
    assert adapter.read_text("src/example.py") == "print('hello')\n"


def test_file_adapter_rejects_parent_directory_escape(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    adapter = LocalFileAdapter(worktree)

    with pytest.raises(PathBoundaryError):
        adapter.write_text("../outside.txt", "nope")


def test_file_adapter_rejects_symlink_escape(tmp_path):
    worktree = tmp_path / "worktree"
    outside = tmp_path / "outside"
    worktree.mkdir()
    outside.mkdir()
    (worktree / "link").symlink_to(outside, target_is_directory=True)
    adapter = LocalFileAdapter(worktree)

    with pytest.raises(PathBoundaryError):
        adapter.write_text("link/escaped.txt", "nope")

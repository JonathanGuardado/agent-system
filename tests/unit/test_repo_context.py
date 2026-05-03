from __future__ import annotations

import os
from pathlib import Path

import pytest

from ticket_agent.orchestrator.repo_context import (
    RepoContextBuilder,
)
from ticket_agent.orchestrator.state import TicketState


def _state(worktree: Path, **updates) -> TicketState:
    values = {
        "ticket_key": "AGENT-100",
        "summary": "Add pagination to users",
        "description": "Update src/api/users.py to support page and page_size.",
        "repository": "agent-system",
        "repo_path": str(worktree),
        "worktree_path": str(worktree),
        "max_attempts": 3,
    }
    values.update(updates)
    return TicketState(**values)


def _write(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_builder_lists_files_inside_worktree(tmp_path: Path):
    _write(tmp_path / "src" / "api" / "users.py", "def list_users(): ...\n")
    _write(tmp_path / "tests" / "test_users.py", "def test_users(): ...\n")
    _write(tmp_path / "README.md", "# Project\n")

    builder = RepoContextBuilder()
    context = builder.build(_state(tmp_path))

    assert context.total_files_listed == 3
    assert "src/api/users.py" in context.relevant_files


def test_builder_skips_noisy_directories(tmp_path: Path):
    _write(tmp_path / "src" / "users.py", "VALUE = 1\n")
    _write(tmp_path / "node_modules" / "lib" / "index.js", "x")
    _write(tmp_path / "__pycache__" / "users.cpython.pyc", "x")
    _write(tmp_path / ".venv" / "lib" / "thing.py", "x")
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / "build" / "out.txt", "x")

    builder = RepoContextBuilder()
    context = builder.build(_state(tmp_path))

    listed = context.relevant_files + list(context.file_contents.keys())
    for path in listed:
        assert not path.startswith("node_modules/")
        assert not path.startswith(".venv/")
        assert not path.startswith(".git/")
        assert not path.startswith("__pycache__/")
        assert not path.startswith("build/")


def test_builder_skips_binary_and_lock_files(tmp_path: Path):
    _write(tmp_path / "src" / "users.py", "VALUE = 1\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data.sqlite").write_bytes(b"SQLite")

    builder = RepoContextBuilder()
    context = builder.build(_state(tmp_path))

    assert "logo.png" not in context.relevant_files
    assert "package-lock.json" not in context.relevant_files
    assert "data.sqlite" not in context.relevant_files
    assert "src/users.py" in context.relevant_files


def test_builder_prioritizes_decomposition_files(tmp_path: Path):
    _write(tmp_path / "src" / "users.py", "VALUE = 1\n")
    _write(tmp_path / "src" / "orders.py", "VALUE = 2\n")
    _write(tmp_path / "src" / "billing.py", "VALUE = 3\n")

    state = _state(
        tmp_path,
        summary="Tweak something",
        description="No specific paths mentioned in this description.",
        decomposition={"files_to_modify": ["src/orders.py", "src/billing.py"]},
    )
    builder = RepoContextBuilder(max_files_read=2)
    context = builder.build(state)

    assert context.relevant_files[:2] == ["src/orders.py", "src/billing.py"]
    assert "src/orders.py" in context.file_contents
    assert "src/billing.py" in context.file_contents


def test_builder_respects_max_file_chars_and_total_chars(tmp_path: Path):
    big = "a" * 5000
    _write(tmp_path / "src" / "a.py", big)
    _write(tmp_path / "src" / "b.py", big)
    _write(tmp_path / "src" / "c.py", big)

    state = _state(
        tmp_path,
        decomposition={
            "files_to_modify": ["src/a.py", "src/b.py", "src/c.py"],
        },
    )
    builder = RepoContextBuilder(
        max_files_read=10,
        max_file_chars=2000,
        max_total_chars=3000,
    )
    context = builder.build(state)

    for content in context.file_contents.values():
        assert len(content) <= 2000
    assert context.total_chars_read <= 3000
    assert all(path in context.truncated_files for path in context.file_contents)


def test_builder_never_reads_outside_worktree(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.py"
    secret.write_text("SECRET = True\n", encoding="utf-8")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    _write(worktree / "src" / "users.py", "VALUE = 1\n")

    # Symlink-style escape: create a file inside worktree pointing to outside
    link_path = worktree / "src" / "linked.py"
    try:
        os.symlink(secret, link_path)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows etc.
        pytest.skip("symlink not supported on this platform")

    state = _state(
        worktree,
        decomposition={"files_to_modify": ["src/linked.py"]},
    )
    builder = RepoContextBuilder()
    context = builder.build(state)

    for path, content in context.file_contents.items():
        assert "SECRET = True" not in content
    # The symlink target is outside the worktree, so it must not appear at all.
    assert not any(
        "secret" in path.lower() for path in context.file_contents
    )


def test_builder_includes_previous_test_failure_output(tmp_path: Path):
    _write(tmp_path / "src" / "users.py", "VALUE = 1\n")

    state = _state(
        tmp_path,
        test_result={
            "status": "failed",
            "tests_passed": False,
            "stdout": "FAILED tests/test_users.py::test_pagination",
            "stderr": "AssertionError: pagination missing",
        },
    )
    builder = RepoContextBuilder()
    context = builder.build(state)

    assert context.previous_test_result is not None
    assert context.previous_test_result["status"] == "failed"
    assert context.previous_test_result["stderr"].startswith("AssertionError")


def test_builder_handles_missing_files_gracefully(tmp_path: Path):
    _write(tmp_path / "src" / "users.py", "VALUE = 1\n")

    state = _state(
        tmp_path,
        decomposition={
            "files_to_modify": [
                "src/users.py",
                "src/does_not_exist.py",
                "../escape.py",
                "/etc/passwd",
            ],
        },
    )
    builder = RepoContextBuilder()
    context = builder.build(state)

    assert "src/users.py" in context.relevant_files
    assert "src/does_not_exist.py" not in context.relevant_files
    assert "../escape.py" not in context.relevant_files
    assert "/etc/passwd" not in context.relevant_files


def test_builder_returns_empty_when_worktree_path_missing():
    builder = RepoContextBuilder()
    state = TicketState(
        ticket_key="AGENT-1",
        summary="No worktree",
        description="",
        worktree_path=None,
    )

    context = builder.build(state)

    assert context.total_files_listed == 0
    assert context.relevant_files == []
    assert context.file_contents == {}


def test_builder_caps_files_listed(tmp_path: Path):
    for index in range(50):
        _write(tmp_path / "src" / f"mod_{index:03d}.py", "x\n")

    builder = RepoContextBuilder(max_files_listed=10, max_files_read=2)
    context = builder.build(_state(tmp_path))

    assert context.total_files_listed == 10


def test_builder_includes_repo_contract_summary(tmp_path: Path):
    _write(tmp_path / "src" / "users.py", "VALUE = 1\n")

    from ticket_agent.config.repo_contract import (
        CommandSpec,
        ExecutionPolicy,
        LanguageInfo,
        RepoCommands,
        RepoContract,
        RepoInfo,
    )

    contract = RepoContract(
        repo=RepoInfo(name="agent-system", root=str(tmp_path), default_branch="main"),
        language=LanguageInfo(primary="python", package_manager="poetry"),
        commands=RepoCommands(
            test=CommandSpec(
                command=("pytest",),
                timeout_seconds=60,
                working_directory=".",
            ),
            lint=None,
            install=None,
        ),
        policy=ExecutionPolicy(
            dependency_install_allowed=False,
            config_paths_allowed=("config/system.yaml",),
            protected_paths=(".github/",),
        ),
        source_dirs=("src/",),
        test_dirs=("tests/",),
    )

    builder = RepoContextBuilder(repo_contract=contract)
    context = builder.build(_state(tmp_path))

    assert context.repo_contract is not None
    assert context.repo_contract.name == "agent-system"
    assert context.repo_contract.test_command == ("pytest",)
    assert "src/" in context.repo_contract.source_dirs

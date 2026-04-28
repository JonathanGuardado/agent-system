from __future__ import annotations

import pytest

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.config.repo_contract import (
    CommandSpec,
    ExecutionPolicy,
    LanguageInfo,
    RepoCommands,
    RepoContract,
    RepoInfo,
)
from ticket_agent.domain.errors import PathBoundaryError, PolicyViolationError


def _worktree(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    return worktree


def _contract(
    *,
    source_dirs: tuple[str, ...] = ("src",),
    test_dirs: tuple[str, ...] = ("tests",),
    config_paths_allowed: tuple[str, ...] = (),
) -> RepoContract:
    return RepoContract(
        repo=RepoInfo(name="example", root=".", default_branch="main"),
        language=LanguageInfo(primary="python", package_manager="pip"),
        commands=RepoCommands(
            test=CommandSpec(
                command=("python", "-m", "pytest"),
                timeout_seconds=120,
                working_directory=".",
            ),
            lint=None,
            install=None,
        ),
        policy=ExecutionPolicy(
            dependency_install_allowed=False,
            config_paths_allowed=config_paths_allowed,
            protected_paths=(),
        ),
        source_dirs=source_dirs,
        test_dirs=test_dirs,
    )


def test_file_adapter_reads_and_writes_inside_worktree(tmp_path):
    adapter = LocalFileAdapter(_worktree(tmp_path))

    adapter.write_text("src/example.py", "print('hello')\n")

    assert adapter.exists("src/example.py")
    assert adapter.read_text("src/example.py") == "print('hello')\n"


def test_file_adapter_rejects_parent_directory_escape(tmp_path):
    adapter = LocalFileAdapter(_worktree(tmp_path))

    with pytest.raises(PathBoundaryError):
        adapter.write_text("../outside.txt", "nope")


def test_file_adapter_rejects_absolute_path_outside_worktree(tmp_path):
    worktree = _worktree(tmp_path)
    outside = tmp_path / "outside.txt"
    adapter = LocalFileAdapter(worktree)

    with pytest.raises(PathBoundaryError):
        adapter.write_text(outside, "nope")


def test_file_adapter_rejects_symlink_escape(tmp_path):
    worktree = _worktree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (worktree / "link").symlink_to(outside, target_is_directory=True)
    adapter = LocalFileAdapter(worktree)

    with pytest.raises(PathBoundaryError):
        adapter.write_text("link/escaped.txt", "nope")


@pytest.mark.parametrize(
    "path",
    (
        ".github/workflows/ci.yml",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env",
        ".env.local",
        "secrets/token.txt",
        "src/secrets/token.txt",
        ".git/config",
    ),
)
def test_file_adapter_rejects_protected_write_paths(tmp_path, path):
    adapter = LocalFileAdapter(_worktree(tmp_path))

    with pytest.raises(PolicyViolationError, match=path.replace(".", r"\.")):
        adapter.write_text(path, "nope")


def test_file_adapter_rejects_root_pyproject_toml_when_not_allowed(tmp_path):
    adapter = LocalFileAdapter(_worktree(tmp_path), _contract())

    with pytest.raises(PolicyViolationError, match="pyproject.toml"):
        adapter.write_text("pyproject.toml", "[project]\nname = 'example'\n")


def test_file_adapter_allows_root_pyproject_toml_when_explicitly_allowed(tmp_path):
    worktree = _worktree(tmp_path)
    adapter = LocalFileAdapter(
        worktree,
        _contract(config_paths_allowed=("pyproject.toml",)),
    )

    adapter.write_text("pyproject.toml", "[project]\nname = 'example'\n")

    assert (worktree / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname = 'example'\n"
    )


def test_file_adapter_rejects_write_outside_contract_dirs(tmp_path):
    adapter = LocalFileAdapter(_worktree(tmp_path), _contract())

    with pytest.raises(PolicyViolationError, match="outside repo contract"):
        adapter.write_text("docs/notes.md", "notes\n")


def test_file_adapter_allows_write_inside_source_dirs(tmp_path):
    adapter = LocalFileAdapter(_worktree(tmp_path), _contract())

    adapter.write_text("src/ticket_agent/example.py", "VALUE = 1\n")

    assert adapter.read_text("src/ticket_agent/example.py") == "VALUE = 1\n"


def test_file_adapter_allows_write_inside_test_dirs(tmp_path):
    adapter = LocalFileAdapter(_worktree(tmp_path), _contract())

    adapter.write_text("tests/test_example.py", "def test_example():\n    assert True\n")

    assert adapter.read_text("tests/test_example.py") == (
        "def test_example():\n    assert True\n"
    )


def test_file_adapter_list_files_returns_relative_paths(tmp_path):
    worktree = _worktree(tmp_path)
    (worktree / "src").mkdir()
    (worktree / "src" / "example.py").write_text("print('hello')\n", encoding="utf-8")
    (worktree / "README.md").write_text("# Example\n", encoding="utf-8")
    (worktree / "src" / "nested").mkdir()

    adapter = LocalFileAdapter(worktree)

    assert adapter.list_files() == ("README.md", "src/example.py")


def test_file_adapter_list_files_excludes_git_contents(tmp_path):
    worktree = _worktree(tmp_path)
    (worktree / ".git" / "objects").mkdir(parents=True)
    (worktree / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (worktree / "src").mkdir()
    (worktree / "src" / "example.py").write_text("print('hello')\n", encoding="utf-8")

    adapter = LocalFileAdapter(worktree)

    assert adapter.list_files() == ("src/example.py",)


def test_file_adapter_blocks_empty_write_to_existing_non_empty_file(tmp_path):
    worktree = _worktree(tmp_path)
    (worktree / "src").mkdir()
    (worktree / "src" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    adapter = LocalFileAdapter(worktree)

    with pytest.raises(PolicyViolationError, match="empty write"):
        adapter.write_text("src/example.py", "")

    assert adapter.read_text("src/example.py") == "VALUE = 1\n"

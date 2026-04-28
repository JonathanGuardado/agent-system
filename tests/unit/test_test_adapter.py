from __future__ import annotations

import sys

from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.adapters.local.test_adapter import LocalTestAdapter
from ticket_agent.config.repo_contract import load_repo_contract


def _write_contract(tmp_path, commands: str):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    contract_path = worktree / "ticket-agent.yaml"
    contract_path.write_text(
        f"""
repo:
  name: example
  root: "."
  default_branch: main

language:
  primary: python
  package_manager: pip

commands:
{commands}

policy:
  dependency_install_allowed: false
  config_paths_allowed: []

source_dirs:
  - src/
test_dirs:
  - tests/
""",
        encoding="utf-8",
    )
    return worktree, load_repo_contract(contract_path)


def _shell(worktree):
    return LocalShellAdapter(worktree, allowed_commands=[(sys.executable, "-c")])


def test_test_adapter_run_tests_uses_contract_commands_test(tmp_path):
    worktree, contract = _write_contract(
        tmp_path,
        f"""
  test:
    command: [{sys.executable!r}, "-c", "print('tests ran')"]
    timeout_seconds: 5
    working_directory: "."
  lint: null
  install:
    command: [{sys.executable!r}, "-c", "print('install should not run')"]
    timeout_seconds: 5
    working_directory: "."
""",
    )
    tests = LocalTestAdapter(_shell(worktree), contract)

    result = tests.run_tests()

    assert result.ok
    assert result.stdout == "tests ran\n"


def test_test_adapter_run_lint_uses_contract_commands_lint_when_present(tmp_path):
    worktree, contract = _write_contract(
        tmp_path,
        f"""
  test:
    command: [{sys.executable!r}, "-c", "print('tests ran')"]
    timeout_seconds: 5
    working_directory: "."
  lint:
    command: [{sys.executable!r}, "-c", "print('lint ran')"]
    timeout_seconds: 5
    working_directory: "."
  install: null
""",
    )
    tests = LocalTestAdapter(_shell(worktree), contract)

    result = tests.run_lint()

    assert result is not None
    assert result.ok
    assert result.stdout == "lint ran\n"


def test_test_adapter_run_lint_returns_none_when_lint_is_null(tmp_path):
    worktree, contract = _write_contract(
        tmp_path,
        f"""
  test:
    command: [{sys.executable!r}, "-c", "print('tests ran')"]
    timeout_seconds: 5
    working_directory: "."
  lint: null
  install: null
""",
    )
    tests = LocalTestAdapter(_shell(worktree), contract)

    assert tests.run_lint() is None


def test_test_adapter_does_not_auto_detect_commands(tmp_path):
    worktree, contract = _write_contract(
        tmp_path,
        f"""
  test:
    command: [{sys.executable!r}, "-c", "print('contract tests ran')"]
    timeout_seconds: 5
    working_directory: "."
  lint: null
  install: null
""",
    )
    (worktree / "pyproject.toml").write_text(
        """
[tool.pytest.ini_options]
addopts = "-q"
""",
        encoding="utf-8",
    )
    tests = LocalTestAdapter(_shell(worktree), contract)

    result = tests.run_tests()

    assert result.ok
    assert result.stdout == "contract tests ran\n"

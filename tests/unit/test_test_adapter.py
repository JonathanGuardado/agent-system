from __future__ import annotations

import sys

from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.adapters.local.test_adapter import LocalTestAdapter
from ticket_agent.config.repo_contract import load_repo_contract


def test_test_adapter_runs_declared_repo_contract_command(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    contract_path = worktree / "ticket-agent.yaml"
    contract_path.write_text(
        f"""
test_commands:
  default:
    command: [{sys.executable!r}, "-c", "print('tests ran')"]
    timeout_seconds: 5
    working_directory: "."
shell_allowlist:
  - [{sys.executable!r}, "-c"]
""",
        encoding="utf-8",
    )
    contract = load_repo_contract(contract_path)
    shell = LocalShellAdapter(worktree, allowed_commands=contract.shell_allowlist)
    tests = LocalTestAdapter(shell, contract)

    result = tests.run_tests()

    assert result.ok
    assert result.stdout == "tests ran\n"

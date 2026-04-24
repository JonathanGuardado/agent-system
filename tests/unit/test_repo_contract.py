from __future__ import annotations

import sys

import pytest

from ticket_agent.config.repo_contract import RepoContractError, load_repo_contract


def test_load_repo_contract_validates_structured_test_commands(tmp_path):
    contract_path = tmp_path / "ticket-agent.yaml"
    contract_path.write_text(
        f"""
test_commands:
  unit:
    command: [{sys.executable!r}, "-c", "print('unit')"]
    timeout_seconds: 5
    working_directory: "."
shell_allowlist:
  - [{sys.executable!r}, "-c"]
""",
        encoding="utf-8",
    )

    contract = load_repo_contract(contract_path)

    assert contract.test_command("unit").command == (
        sys.executable,
        "-c",
        "print('unit')",
    )
    assert contract.test_command("unit").timeout_seconds == 5
    assert contract.shell_allowlist == ((sys.executable, "-c"),)


def test_load_repo_contract_rejects_string_commands(tmp_path):
    contract_path = tmp_path / "ticket-agent.yaml"
    contract_path.write_text(
        """
test_commands:
  unit:
    command: "pytest"
""",
        encoding="utf-8",
    )

    with pytest.raises(RepoContractError):
        load_repo_contract(contract_path)


def test_repo_contract_rejects_unknown_test_suite(tmp_path):
    contract_path = tmp_path / "ticket-agent.yaml"
    contract_path.write_text(
        f"""
test_commands:
  unit:
    command: [{sys.executable!r}, "-c", "print('unit')"]
""",
        encoding="utf-8",
    )
    contract = load_repo_contract(contract_path)

    with pytest.raises(RepoContractError):
        contract.test_command("integration")

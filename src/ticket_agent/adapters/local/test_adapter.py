"""Repo-contract backed local test adapter."""

from __future__ import annotations

from ticket_agent.config.repo_contract import RepoContract
from ticket_agent.ports.tools import CommandResult, ShellPort


class LocalTestAdapter:
    """Run only test commands declared in the repo contract."""

    def __init__(self, shell: ShellPort, contract: RepoContract) -> None:
        self._shell = shell
        self._contract = contract

    def run_tests(self, suite: str = "default") -> CommandResult:
        test_command = self._contract.test_command(suite)
        return self._shell.run(
            test_command.command,
            cwd=test_command.working_directory,
            timeout_seconds=test_command.timeout_seconds,
        )

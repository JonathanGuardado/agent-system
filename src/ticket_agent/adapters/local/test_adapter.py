"""Repo-contract backed local test adapter."""

from __future__ import annotations

from ticket_agent.config.repo_contract import RepoContract
from ticket_agent.domain.errors import RepoContractError
from ticket_agent.ports.tools import CommandResult, ShellPort


class LocalTestAdapter:
    """Run only test commands declared in the repo contract."""

    def __init__(self, shell: ShellPort, contract: RepoContract) -> None:
        self._shell = shell
        self._contract = contract

    def run_tests(self, suite: str = "default") -> CommandResult:
        if suite != "default":
            raise RepoContractError(
                "repo contract only declares the default test command"
            )
        test_command = self._contract.commands.test
        return self._shell.run(
            test_command.command,
            cwd=test_command.working_directory,
            timeout_seconds=test_command.timeout_seconds,
        )

    def run_lint(self) -> CommandResult | None:
        lint_command = self._contract.commands.lint
        if lint_command is None:
            return None
        return self._shell.run(
            lint_command.command,
            cwd=lint_command.working_directory,
            timeout_seconds=lint_command.timeout_seconds,
        )

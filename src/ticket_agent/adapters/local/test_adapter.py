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
        install_command = self._install_command()
        if install_command is not None:
            install_result = self._shell.run(
                install_command.command,
                cwd=install_command.working_directory,
                timeout_seconds=install_command.timeout_seconds,
            )
            if not install_result.ok:
                return install_result

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

    def _install_command(self):
        if not self._contract.policy.dependency_install_allowed:
            return None
        return self._contract.commands.install

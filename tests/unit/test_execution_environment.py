from __future__ import annotations

import asyncio

import pytest

from ticket_agent.adapters.local.sandbox import (
    NullSandbox,
    SandboxPolicy,
    SandboxUnavailableError,
)
from ticket_agent.config.repo_contract import (
    CommandSpec,
    ExecutionPolicy,
    LanguageInfo,
    RepoCommands,
    RepoContract,
    RepoInfo,
)
from ticket_agent.orchestrator.execution_environment import (
    ExecutionEnvironmentPreflight,
)
from ticket_agent.orchestrator.local_services import (
    AdapterTestService,
    RuntimeShellFactory,
    _make_contract_test_runner,
)
from ticket_agent.orchestrator.state import TicketState


def test_preflight_rejects_null_wrapper_even_when_policy_text_says_bwrap():
    misleading_policy = SandboxPolicy()
    assert misleading_policy.profile().startswith("bwrap:")
    preflight = ExecutionEnvironmentPreflight(NullSandbox)

    with pytest.raises(SandboxUnavailableError, match="actual wrapper profile"):
        preflight.check()


def test_both_contract_test_paths_use_the_same_enforcing_shell_factory(tmp_path):
    contract = _contract()
    preflight = ExecutionEnvironmentPreflight(NullSandbox)
    shell_factory = RuntimeShellFactory(preflight)
    state = TicketState(
        ticket_key="AGENT-123",
        summary="Verify shell wiring",
        repository="example",
        repo_path=str(tmp_path),
        worktree_path=str(tmp_path),
    )
    adapter_service = AdapterTestService(
        contract_loader=lambda path: contract,
        shell_factory=shell_factory,
    )
    model_runner = _make_contract_test_runner(
        tmp_path,
        contract,
        shell_factory=shell_factory,
    )

    adapter_result = asyncio.run(adapter_service.run_tests(state))
    model_result = model_runner()

    assert adapter_result["tests_passed"] is False
    assert model_result["tests_passed"] is False
    assert "actual wrapper profile" in adapter_result["error"]
    assert "actual wrapper profile" in model_result["error"]


def test_omitting_the_shell_factory_refuses_instead_of_crashing(tmp_path):
    """There is no safe default shell factory, so the default must say so.

    `_build_contract_shell` used to be the default while requiring a
    keyword-only `sandbox`, so it could not be called through the
    `ShellFactory` signature at all. Omitting the factory raised `TypeError:
    missing 1 required keyword-only argument` -- and TypeError is not among the
    `(AgentSystemError, OSError, ValueError)` this boundary catches, so it
    escaped as an unhandled crash instead of a failed test result. No test
    reached it: the ones that omit the factory all fail earlier on contract
    errors. Now it is a SandboxUnavailableError, which the boundary does catch.
    """

    result = _make_contract_test_runner(tmp_path, _contract())()

    assert result["tests_passed"] is False
    assert result["status"] == "failed"
    assert "RuntimeShellFactory" in result["error"]


def _contract() -> RepoContract:
    return RepoContract(
        repo=RepoInfo(name="example", root=".", default_branch="main"),
        language=LanguageInfo(primary="python", package_manager="pip"),
        commands=RepoCommands(
            test=CommandSpec(
                command=("/bin/true",),
                timeout_seconds=10,
                working_directory=".",
                writable_paths=(),
                network="none",
            ),
            lint=None,
            install=None,
        ),
        policy=ExecutionPolicy(
            dependency_install_allowed=False,
            config_paths_allowed=(),
            protected_paths=(),
        ),
        source_dirs=("src/",),
        test_dirs=("tests/",),
    )

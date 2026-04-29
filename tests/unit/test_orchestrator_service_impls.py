from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ticket_agent.config.repo_contract import (
    CommandSpec,
    ExecutionPolicy,
    LanguageInfo,
    RepoCommands,
    RepoContract,
    RepoInfo,
)
from ticket_agent.domain.errors import RepoContractError
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.local_services import AdapterTestService
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.ports.tools import CommandResult


def test_adapter_test_service_returns_passed_result_when_adapter_succeeds(tmp_path):
    service, calls = _service_for(
        CommandResult(
            command=("python", "-m", "pytest"),
            returncode=0,
            stdout="tests passed\n",
            stderr="",
        )
    )

    result = asyncio.run(service.run_tests(_state(tmp_path)))

    assert result == {
        "status": "passed",
        "tests_passed": True,
        "exit_code": 0,
        "output": "tests passed\n",
        "failed_tests": [],
        "timed_out": False,
        "error": None,
    }
    assert calls["contract_path"] == Path("config/repos/example.yaml")
    assert calls["worktree_path"] == tmp_path
    assert calls["adapter_calls"] == ["default"]


def test_adapter_test_service_returns_failed_result_when_adapter_fails(tmp_path):
    service, _ = _service_for(
        CommandResult(
            command=("python", "-m", "pytest"),
            returncode=1,
            stdout="1 failed\n",
            stderr="AssertionError\n",
        )
    )

    result = asyncio.run(service.run_tests(_state(tmp_path)))

    assert result["status"] == "failed"
    assert result["tests_passed"] is False
    assert result["exit_code"] == 1
    assert result["output"] == "1 failed\nAssertionError\n"
    assert result["failed_tests"] == []
    assert result["timed_out"] is False
    assert "AssertionError" in result["error"]


def test_adapter_test_service_returns_failed_result_when_adapter_raises(tmp_path):
    service, _ = _service_for(RepoContractError("suite is not configured"))

    result = asyncio.run(service.run_tests(_state(tmp_path)))

    assert result["status"] == "failed"
    assert result["tests_passed"] is False
    assert result["error"] == "test adapter failed: suite is not configured"


def test_adapter_test_service_returns_failed_result_without_worktree_path():
    loader_called = False

    def load_contract(path: Path) -> RepoContract:
        nonlocal loader_called
        loader_called = True
        return _contract()

    service = AdapterTestService(contract_loader=load_contract)

    result = asyncio.run(
        service.run_tests(TicketState(ticket_key="AGENT-123", summary="Missing path"))
    )

    assert result["status"] == "failed"
    assert result["tests_passed"] is False
    assert result["error"] == "worktree_path is required to run tests"
    assert loader_called is False


def test_adapter_test_service_loads_contract_from_repo_path_when_repository_missing(
    tmp_path,
):
    worktree_path = tmp_path / "worktree"
    repo_path = tmp_path / "agent-system"
    service, calls = _service_for(
        CommandResult(
            command=("python", "-m", "pytest"),
            returncode=0,
            stdout="tests passed\n",
            stderr="",
        )
    )
    state = TicketState(
        ticket_key="AGENT-123",
        summary="Run adapter tests",
        repo_path=str(repo_path),
        worktree_path=str(worktree_path),
    )

    result = asyncio.run(service.run_tests(state))

    assert result["tests_passed"] is True
    assert calls["contract_path"] == Path("config/repos/agent-system.yaml")


@pytest.mark.parametrize(
    "contract_error",
    [
        FileNotFoundError("ticket-agent.yaml"),
        RepoContractError("commands.test is required"),
    ],
)
def test_adapter_test_service_returns_failed_result_for_missing_or_invalid_contract(
    tmp_path,
    contract_error: Exception,
):
    def load_contract(path: Path) -> RepoContract:
        raise contract_error

    service = AdapterTestService(contract_loader=load_contract)

    result = asyncio.run(service.run_tests(_state(tmp_path)))

    assert result["status"] == "failed"
    assert result["tests_passed"] is False
    assert result["exit_code"] == 1
    assert result["error"].startswith("repo contract missing or invalid:")


@pytest.mark.parametrize(
    ("returncode", "expected_tests_passed"),
    [(0, True), (1, False)],
)
def test_ticket_node_runner_run_tests_stores_adapter_service_result(
    tmp_path,
    returncode: int,
    expected_tests_passed: bool,
):
    service, _ = _service_for(
        CommandResult(
            command=("python", "-m", "pytest"),
            returncode=returncode,
            stdout="ok\n" if expected_tests_passed else "failed\n",
            stderr="",
        )
    )
    runner = _runner(tests=service)

    initial_state = _state(tmp_path)
    state = initial_state.model_copy(
        update=asyncio.run(runner.run_tests(initial_state))
    )

    assert state.test_result is not None
    assert state.test_result["tests_passed"] is expected_tests_passed
    assert state.tests_passed is expected_tests_passed


def _service_for(
    command_result: CommandResult | Exception,
) -> tuple[AdapterTestService, dict[str, Any]]:
    calls: dict[str, Any] = {"adapter_calls": []}
    contract = _contract()

    def load_contract(path: Path) -> RepoContract:
        calls["contract_path"] = path
        return contract

    def shell_factory(worktree_path: Path, loaded_contract: RepoContract) -> _FakeShell:
        calls["worktree_path"] = worktree_path
        calls["shell_contract"] = loaded_contract
        return _FakeShell(worktree_path)

    def adapter_factory(
        shell: _FakeShell,
        loaded_contract: RepoContract,
    ) -> _FakeTestAdapter:
        calls["adapter_shell"] = shell
        calls["adapter_contract"] = loaded_contract
        return _FakeTestAdapter(command_result, calls["adapter_calls"])

    return (
        AdapterTestService(
            contract_loader=load_contract,
            shell_factory=shell_factory,
            adapter_factory=adapter_factory,
        ),
        calls,
    )


def _contract() -> RepoContract:
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
            config_paths_allowed=(),
            protected_paths=(),
        ),
        source_dirs=("src/",),
        test_dirs=("tests/",),
    )


def _state(worktree_path: Path) -> TicketState:
    return TicketState(
        ticket_key="AGENT-123",
        summary="Run adapter tests",
        repository="example",
        repo_path=str(worktree_path.parent),
        worktree_path=str(worktree_path),
    )


def _runner(**overrides: Any) -> TicketNodeRunner:
    defaults = {
        "planner": _Planner(),
        "approval": _Approval(),
        "implementation": _Implementation(),
        "tests": overrides.pop("tests"),
        "review": _Review(),
        "pull_request": _PullRequest(),
        "escalation": _Escalation(),
    }
    defaults.update(overrides)
    return TicketNodeRunner(**defaults)


class _FakeShell:
    def __init__(self, root: Path) -> None:
        self.root = root


class _FakeTestAdapter:
    def __init__(self, result: CommandResult | Exception, calls: list[str]) -> None:
        self._result = result
        self._calls = calls

    def run_tests(self, suite: str = "default") -> CommandResult:
        self._calls.append(suite)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def run_lint(self) -> CommandResult | None:
        return None


class _Planner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {}


class _Approval:
    async def request_approval(self, state: TicketState) -> bool:
        return True


class _Implementation:
    async def implement(self, state: TicketState) -> dict[str, Any]:
        return {}


class _Review:
    async def review(self, state: TicketState) -> dict[str, Any]:
        return {"status": "accepted"}


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/example/repo/pull/1"


class _Escalation:
    async def escalate(self, state: TicketState, reason: str) -> None:
        return None

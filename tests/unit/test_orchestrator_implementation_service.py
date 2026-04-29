from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ticket_agent.config.repo_contract import (
    CommandSpec,
    ExecutionPolicy,
    LanguageInfo,
    RepoCommands,
    RepoContract,
    RepoInfo,
)
from ticket_agent.domain.errors import RepoContractError, WorktreeCreationError
from ticket_agent.domain.git import WorktreeInfo
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.service_impls import (
    ImplementationContext,
    LocalImplementationService,
)
from ticket_agent.orchestrator.state import TicketState


def test_local_implementation_service_creates_worktree_and_calls_step(tmp_path):
    repo_path = tmp_path / "repo"
    worktree_path = repo_path / ".worktrees" / "AGENT-123"
    calls: dict[str, Any] = {}
    git = _FakeGit(
        WorktreeInfo(
            repo_path=repo_path,
            worktree_path=worktree_path,
            branch_name="agent/AGENT-123/12345678",
            ticket_key="AGENT-123",
            lock_id="12345678",
        )
    )

    def implementation_step(context: ImplementationContext) -> dict[str, Any]:
        calls["context"] = context
        return {"status": "implemented", "changed_files": ["src/example.py"]}

    service = LocalImplementationService(
        contract_loader=_loader(calls, _contract(repo_root=str(repo_path))),
        git=git,
        file_adapter_factory=_file_adapter_factory(calls),
        implementation_step=implementation_step,
        lock_id_factory=lambda state: "12345678",
    )

    result = asyncio.run(
        service.implement(
            TicketState(
                ticket_key="AGENT-123",
                summary="Implement feature",
                repository="example",
                repo_path=str(repo_path),
            )
        )
    )

    assert calls["contract_path"] == Path("config/repos/example.yaml")
    assert git.calls == [(repo_path, "AGENT-123", "12345678")]
    assert calls["file_adapter_args"] == (worktree_path, calls["contract"])
    assert calls["context"].files is calls["file_adapter"]
    assert result == {
        "repo_path": str(repo_path),
        "worktree_path": str(worktree_path),
        "branch_name": "agent/AGENT-123/12345678",
        "lock_id": "12345678",
        "implementation_result": {
            "status": "implemented",
            "changed_files": ["src/example.py"],
        },
    }


def test_local_implementation_service_uses_contract_repo_root_when_repo_path_missing(
    tmp_path,
):
    repo_path = tmp_path / "repo"
    worktree_path = repo_path / ".worktrees" / "AGENT-123"
    calls: dict[str, Any] = {}
    git = _FakeGit(
        WorktreeInfo(
            repo_path=repo_path,
            worktree_path=worktree_path,
            branch_name="agent/AGENT-123/abcdef12",
            ticket_key="AGENT-123",
            lock_id="abcdef12",
        )
    )
    service = LocalImplementationService(
        contract_loader=_loader(calls, _contract(repo_root=str(repo_path))),
        git=git,
        file_adapter_factory=_file_adapter_factory(calls),
        lock_id_factory=lambda state: "abcdef12",
    )

    result = asyncio.run(
        service.implement(
            TicketState(
                ticket_key="AGENT-123",
                summary="Implement feature",
                repository="example",
            )
        )
    )

    assert git.calls == [(repo_path, "AGENT-123", "abcdef12")]
    assert result["worktree_path"] == str(worktree_path)
    assert result["implementation_result"]["status"] == "prepared"


def test_local_implementation_service_reuses_existing_worktree(tmp_path):
    repo_path = tmp_path / "repo"
    worktree_path = tmp_path / "existing-worktree"
    calls: dict[str, Any] = {}
    git = _FakeGit(
        WorktreeInfo(
            repo_path=repo_path,
            worktree_path=repo_path / ".worktrees" / "unused",
            branch_name="agent/AGENT-123/unused",
            ticket_key="AGENT-123",
            lock_id="unused",
        )
    )
    service = LocalImplementationService(
        contract_loader=_loader(calls, _contract(repo_root=str(repo_path))),
        git=git,
        file_adapter_factory=_file_adapter_factory(calls),
    )

    result = asyncio.run(
        service.implement(
            TicketState(
                ticket_key="AGENT-123",
                summary="Retry feature",
                repository="example",
                repo_path=str(repo_path),
                worktree_path=str(worktree_path),
                branch_name="agent/AGENT-123/12345678",
                lock_id="12345678",
            )
        )
    )

    assert git.calls == []
    assert calls["file_adapter_args"] == (worktree_path, calls["contract"])
    assert result["worktree_path"] == str(worktree_path)
    assert result["branch_name"] == "agent/AGENT-123/12345678"
    assert result["lock_id"] == "12345678"


def test_local_implementation_service_returns_failed_result_without_repo_identity():
    loader_called = False

    def load_contract(path: Path) -> RepoContract:
        nonlocal loader_called
        loader_called = True
        return _contract()

    service = LocalImplementationService(contract_loader=load_contract)

    result = asyncio.run(
        service.implement(TicketState(ticket_key="AGENT-123", summary="No repo"))
    )

    assert result["implementation_result"]["status"] == "failed"
    assert result["error"] == (
        "repository or repo_path is required to load repo contract"
    )
    assert loader_called is False


def test_local_implementation_service_returns_failed_result_for_invalid_contract(
    tmp_path,
):
    def load_contract(path: Path) -> RepoContract:
        raise RepoContractError("commands.test is required")

    service = LocalImplementationService(contract_loader=load_contract)

    result = asyncio.run(
        service.implement(
            TicketState(
                ticket_key="AGENT-123",
                summary="Invalid contract",
                repository="example",
                repo_path=str(tmp_path / "repo"),
            )
        )
    )

    assert result["implementation_result"]["status"] == "failed"
    assert result["error"].startswith("repo contract missing or invalid:")


def test_local_implementation_service_returns_failed_result_for_worktree_error(
    tmp_path,
):
    service = LocalImplementationService(
        contract_loader=lambda path: _contract(repo_root=str(tmp_path / "repo")),
        git=_FakeGit(WorktreeCreationError("worktree already exists")),
        lock_id_factory=lambda state: "12345678",
    )

    result = asyncio.run(
        service.implement(
            TicketState(
                ticket_key="AGENT-123",
                summary="Worktree conflict",
                repository="example",
                repo_path=str(tmp_path / "repo"),
            )
        )
    )

    assert result["implementation_result"]["status"] == "failed"
    assert result["error"] == "implementation failed: worktree already exists"


def test_ticket_node_runner_implement_stores_local_implementation_update(tmp_path):
    repo_path = tmp_path / "repo"
    worktree_path = repo_path / ".worktrees" / "AGENT-123"
    service = LocalImplementationService(
        contract_loader=lambda path: _contract(repo_root=str(repo_path)),
        git=_FakeGit(
            WorktreeInfo(
                repo_path=repo_path,
                worktree_path=worktree_path,
                branch_name="agent/AGENT-123/12345678",
                ticket_key="AGENT-123",
                lock_id="12345678",
            )
        ),
        file_adapter_factory=lambda worktree, contract: _FakeFileAdapter(worktree),
        implementation_step=lambda context: {
            "status": "implemented",
            "changed_files": ["src/example.py"],
        },
        lock_id_factory=lambda state: "12345678",
    )
    runner = _runner(implementation=service)
    initial_state = TicketState(
        ticket_key="AGENT-123",
        summary="Implement feature",
        repository="example",
        repo_path=str(repo_path),
    )

    state = initial_state.model_copy(
        update=asyncio.run(runner.implement(initial_state))
    )

    assert state.implementation_attempts == 1
    assert state.worktree_path == str(worktree_path)
    assert state.branch_name == "agent/AGENT-123/12345678"
    assert state.lock_id == "12345678"
    assert state.implementation_result == {
        "status": "implemented",
        "changed_files": ["src/example.py"],
    }


def _loader(
    calls: dict[str, Any],
    contract: RepoContract,
):
    def load_contract(path: Path) -> RepoContract:
        calls["contract_path"] = path
        calls["contract"] = contract
        return contract

    return load_contract


def _file_adapter_factory(calls: dict[str, Any]):
    def build(worktree_path: Path, contract: RepoContract) -> _FakeFileAdapter:
        adapter = _FakeFileAdapter(worktree_path)
        calls["file_adapter_args"] = (worktree_path, contract)
        calls["file_adapter"] = adapter
        return adapter

    return build


def _contract(repo_root: str = ".") -> RepoContract:
    return RepoContract(
        repo=RepoInfo(name="example", root=repo_root, default_branch="main"),
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


def _runner(**overrides: Any) -> TicketNodeRunner:
    defaults = {
        "planner": _Planner(),
        "approval": _Approval(),
        "implementation": _Implementation(),
        "tests": _Tests(),
        "review": _Review(),
        "pull_request": _PullRequest(),
        "escalation": _Escalation(),
    }
    defaults.update(overrides)
    return TicketNodeRunner(**defaults)


class _FakeGit:
    def __init__(self, result: WorktreeInfo | Exception) -> None:
        self._result = result
        self.calls: list[tuple[Path, str, str]] = []

    def create_worktree(
        self,
        repo_path: str | Path,
        ticket_key: str,
        short_lock_id: str,
    ) -> WorktreeInfo:
        self.calls.append((Path(repo_path), ticket_key, short_lock_id))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeFileAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve(self, path: str | Path) -> Path:
        return self.root / path

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        return ""

    def write_text(
        self,
        path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        return None

    def exists(self, path: str | Path) -> bool:
        return False

    def list_files(self, path: str | Path = ".") -> tuple[str, ...]:
        return ()


class _Planner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {}


class _Approval:
    async def request_approval(self, state: TicketState) -> bool:
        return True


class _Implementation:
    async def implement(self, state: TicketState) -> dict[str, Any]:
        return {}


class _Tests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed"}


class _Review:
    async def review(self, state: TicketState) -> dict[str, Any]:
        return {"status": "accepted"}


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/example/repo/pull/1"


class _Escalation:
    async def escalate(self, state: TicketState, reason: str) -> None:
        return None

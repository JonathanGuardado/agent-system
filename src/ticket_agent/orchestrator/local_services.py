"""Local concrete service implementations for ticket workflow nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Protocol
from uuid import uuid4

from yaml import YAMLError

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.adapters.local.test_adapter import LocalTestAdapter
from ticket_agent.config.repo_contract import RepoContract, load_repo_contract
from ticket_agent.domain.errors import (
    AgentSystemError,
    PullRequestCreationError,
    RepoContractError,
)
from ticket_agent.domain.git import WorktreeInfo
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.ports.tools import CommandResult, FilePort, ShellPort, TestPort


TestResult = dict[str, Any]
ImplementationResult = dict[str, Any]
ContractLoader = Callable[[Path], RepoContract]
FileAdapterFactory = Callable[[Path, RepoContract], FilePort]
ShellFactory = Callable[[Path, RepoContract], ShellPort]
TestAdapterFactory = Callable[[ShellPort, RepoContract], TestPort]
ImplementationStep = Callable[["ImplementationContext"], ImplementationResult]
LockIdFactory = Callable[[TicketState], str]


class GitPullRequestPort(Protocol):
    def commit(self, worktree_path: str | Path, message: str) -> str: ...

    def push(self, worktree_path: str | Path, branch_name: str) -> None: ...


class GitWorktreePort(Protocol):
    def create_worktree(
        self,
        repo_path: str | Path,
        ticket_key: str,
        short_lock_id: str,
    ) -> WorktreeInfo: ...


class PullRequestOpener(Protocol):
    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str: ...


@dataclass(frozen=True)
class ImplementationContext:
    state: TicketState
    contract: RepoContract
    repo_path: Path
    worktree_path: Path
    branch_name: str | None
    lock_id: str | None
    files: FilePort


@dataclass(frozen=True)
class _PreparedWorktree:
    repo_path: Path
    worktree_path: Path
    branch_name: str | None
    ticket_key: str
    lock_id: str | None


_DEFAULT_CONTRACT_DIR = Path("config/repos")


class LocalImplementationService:
    """Prepare local implementation adapters without invoking an LLM."""

    def __init__(
        self,
        *,
        contract_dir: str | Path = _DEFAULT_CONTRACT_DIR,
        contract_loader: ContractLoader = load_repo_contract,
        git: GitWorktreePort | None = None,
        file_adapter_factory: FileAdapterFactory = LocalFileAdapter,
        implementation_step: ImplementationStep | None = None,
        lock_id_factory: LockIdFactory | None = None,
    ) -> None:
        self._contract_dir = Path(contract_dir)
        self._contract_loader = contract_loader
        self._git = git or GitAdapter()
        self._file_adapter_factory = file_adapter_factory
        self._implementation_step = implementation_step or _prepare_only_step
        self._lock_id_factory = lock_id_factory or _new_short_lock_id

    async def implement(self, state: TicketState) -> dict[str, Any]:
        contract_path = _contract_path(state, self._contract_dir)
        if contract_path is None:
            return _failed_implementation_update(
                "repository or repo_path is required to load repo contract"
            )

        try:
            contract = self._contract_loader(contract_path)
        except (FileNotFoundError, RepoContractError, YAMLError, OSError) as exc:
            return _failed_implementation_update(
                f"repo contract missing or invalid: {exc}"
            )

        repo_path = _repo_path(state, contract)
        if repo_path is None:
            return _failed_implementation_update(
                "repo_path or repo.root is required to create worktree"
            )

        try:
            worktree = _worktree_info(
                state,
                repo_path,
                self._git,
                self._lock_id_factory,
            )
            files = self._file_adapter_factory(worktree.worktree_path, contract)
            context = ImplementationContext(
                state=state,
                contract=contract,
                repo_path=worktree.repo_path,
                worktree_path=worktree.worktree_path,
                branch_name=worktree.branch_name,
                lock_id=worktree.lock_id,
                files=files,
            )
            implementation_result = self._implementation_step(context)
        except (AgentSystemError, OSError, ValueError) as exc:
            return _failed_implementation_update(f"implementation failed: {exc}")

        return {
            "repo_path": str(worktree.repo_path),
            "worktree_path": str(worktree.worktree_path),
            "branch_name": worktree.branch_name,
            "lock_id": worktree.lock_id,
            "implementation_result": implementation_result,
        }


class GitService:
    """Commit, push, and open a pull request for completed ticket work."""

    def __init__(
        self,
        *,
        git: GitPullRequestPort | None = None,
        pull_request_opener: PullRequestOpener | None = None,
        base_branch: str = "main",
    ) -> None:
        self._git = git or GitAdapter()
        self._pull_request_opener = pull_request_opener or GhPullRequestOpener()
        self._base_branch = base_branch

    async def open_pull_request(self, state: TicketState) -> str:
        worktree_path = _required_worktree_path(state)
        branch_name = _required_branch_name(state)

        commit_message = _commit_message(state)
        self._git.commit(worktree_path, commit_message)
        self._git.push(worktree_path, branch_name)
        return self._pull_request_opener.open_pull_request(
            worktree_path=worktree_path,
            branch_name=branch_name,
            base_branch=self._base_branch,
            title=_pull_request_title(state),
            body=_pull_request_body(state),
        )


class GhPullRequestOpener:
    """Open pull requests through the GitHub CLI."""

    def __init__(self, *, timeout_seconds: int = 300) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds

    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str:
        command = (
            "gh",
            "pr",
            "create",
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--title",
            title,
            "--body",
            body,
        )
        try:
            result = subprocess.run(
                command,
                cwd=worktree_path,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise PullRequestCreationError(
                f"gh pr create timed out after {self._timeout_seconds} seconds"
            ) from exc

        if result.returncode != 0:
            raise PullRequestCreationError(_subprocess_failure_message(result))

        url = result.stdout.strip()
        if not url:
            raise PullRequestCreationError("gh pr create did not return a PR URL")
        return url


class AdapterTestService:
    """Run repository tests through the repo-contract backed test adapter."""

    def __init__(
        self,
        *,
        contract_dir: str | Path = _DEFAULT_CONTRACT_DIR,
        contract_loader: ContractLoader = load_repo_contract,
        shell_factory: ShellFactory | None = None,
        adapter_factory: TestAdapterFactory = LocalTestAdapter,
    ) -> None:
        self._contract_dir = Path(contract_dir)
        self._contract_loader = contract_loader
        self._shell_factory = shell_factory or _build_contract_shell
        self._adapter_factory = adapter_factory

    async def run_tests(self, state: TicketState) -> TestResult:
        worktree_path = _worktree_path(state)
        if worktree_path is None:
            return _failed_result("worktree_path is required to run tests")

        contract_path = _contract_path(state, self._contract_dir)
        if contract_path is None:
            return _failed_result(
                "repository or repo_path is required to load repo contract"
            )

        try:
            contract = self._contract_loader(contract_path)
        except (FileNotFoundError, RepoContractError, YAMLError, OSError) as exc:
            return _failed_result(f"repo contract missing or invalid: {exc}")

        try:
            shell = self._shell_factory(worktree_path, contract)
            adapter = self._adapter_factory(shell, contract)
            command_result = adapter.run_tests()
        except (AgentSystemError, OSError, ValueError) as exc:
            return _failed_result(f"test adapter failed: {exc}")

        return _command_result_to_test_result(command_result)


def _worktree_path(state: TicketState) -> Path | None:
    if not state.worktree_path:
        return None
    return Path(state.worktree_path)


def _required_worktree_path(state: TicketState) -> Path:
    worktree_path = _worktree_path(state)
    if worktree_path is None:
        raise PullRequestCreationError(
            "worktree_path is required to open pull request"
        )
    return worktree_path


def _required_branch_name(state: TicketState) -> str:
    if not state.branch_name:
        raise PullRequestCreationError(
            "branch_name is required to open pull request"
        )
    return state.branch_name


def _repo_path(state: TicketState, contract: RepoContract) -> Path | None:
    if state.repo_path:
        return Path(state.repo_path).expanduser()
    if contract.repo.root:
        return Path(contract.repo.root).expanduser()
    return None


def _worktree_info(
    state: TicketState,
    repo_path: Path,
    git: GitWorktreePort,
    lock_id_factory: LockIdFactory,
) -> _PreparedWorktree:
    existing_worktree_path = _worktree_path(state)
    if existing_worktree_path is not None:
        return _PreparedWorktree(
            repo_path=repo_path,
            worktree_path=existing_worktree_path,
            branch_name=state.branch_name,
            ticket_key=state.ticket_key,
            lock_id=state.lock_id,
        )

    lock_id = lock_id_factory(state)
    created = git.create_worktree(repo_path, state.ticket_key, lock_id)
    return _PreparedWorktree(
        repo_path=created.repo_path,
        worktree_path=created.worktree_path,
        branch_name=created.branch_name,
        ticket_key=created.ticket_key,
        lock_id=created.lock_id,
    )


def _contract_path(state: TicketState, contract_dir: Path) -> Path | None:
    repo_name = _repo_name(state)
    if repo_name is None:
        return None
    return contract_dir / f"{repo_name}.yaml"


def _repo_name(state: TicketState) -> str | None:
    if state.repository:
        return _normalize_repo_name(state.repository)
    if state.repo_path:
        return Path(state.repo_path).resolve(strict=False).name
    return None


def _normalize_repo_name(repository: str) -> str:
    normalized = repository.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return Path(normalized).name


def _build_contract_shell(worktree_path: Path, contract: RepoContract) -> ShellPort:
    return LocalShellAdapter(
        worktree_path,
        allowed_commands=[contract.commands.test.command],
    )


def _prepare_only_step(context: ImplementationContext) -> ImplementationResult:
    return {
        "status": "prepared",
        "changed_files": [],
        "message": "worktree prepared; no implementation step configured",
    }


def _new_short_lock_id(state: TicketState) -> str:
    return uuid4().hex[:8]


def _commit_message(state: TicketState) -> str:
    return f"{state.ticket_key}: {state.summary}"


def _pull_request_title(state: TicketState) -> str:
    return _commit_message(state)


def _pull_request_body(state: TicketState) -> str:
    parts = [
        f"Ticket: {state.ticket_key}",
        f"Summary: {state.summary}",
    ]
    if state.description:
        parts.extend(("", state.description))
    return "\n".join(parts)


def _command_result_to_test_result(result: CommandResult) -> TestResult:
    tests_passed = result.ok
    return {
        "status": "passed" if tests_passed else "failed",
        "tests_passed": tests_passed,
        "exit_code": result.returncode,
        "output": _combined_output(result),
        "failed_tests": [],
        "timed_out": result.timed_out,
        "error": None if tests_passed else _error_output(result),
    }


def _combined_output(result: CommandResult) -> str:
    if result.stdout and result.stderr:
        separator = "" if result.stdout.endswith("\n") else "\n"
        return f"{result.stdout}{separator}{result.stderr}"
    return result.stdout or result.stderr


def _error_output(result: CommandResult) -> str | None:
    output = _combined_output(result).strip()
    return output or f"test command exited with code {result.returncode}"


def _failed_result(error: str) -> TestResult:
    return {
        "status": "failed",
        "tests_passed": False,
        "exit_code": 1,
        "output": "",
        "failed_tests": [],
        "timed_out": False,
        "error": error,
    }


def _failed_implementation_update(error: str) -> dict[str, Any]:
    return {
        "implementation_result": {
            "status": "failed",
            "changed_files": [],
            "error": error,
        },
        "error": error,
    }


def _subprocess_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    return output or f"gh pr create exited with return code {result.returncode}"

"""Local concrete service implementations for ticket workflow nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import json
import logging
from pathlib import Path
import re
import tempfile
from typing import Any, Protocol
from uuid import uuid4

from yaml import YAMLError

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.adapters.local.sandbox import Sandbox
from ticket_agent.adapters.local.shell_adapter import LocalShellAdapter
from ticket_agent.adapters.local.test_adapter import LocalTestAdapter
from ticket_agent.config.repo_contract import (
    RepoContract,
    load_repo_contract,
    scaffold_repo_contract,
)
from ticket_agent.domain.errors import (
    AgentSystemError,
    RepoContractError,
)
from ticket_agent.domain.execution import CommandExecutionPolicy, SandboxAttestation
from ticket_agent.domain.git import WorktreeInfo
from ticket_agent.goal.journal import GoalActionJournal, ProbeResult
from ticket_agent.goal.types import LoopState, canonical_json, digest
from ticket_agent.orchestrator.execution_environment import ExecutionPreflight
from ticket_agent.orchestrator.git_services import (
    GhPullRequestOpener,
    GitPullRequestPort,
    GitService,
    PullRequestOpener,
)
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.ports.tools import CommandResult, FilePort, ShellPort, TestPort

_LOGGER = logging.getLogger(__name__)

TestResult = dict[str, Any]
ImplementationResult = dict[str, Any]
ContractLoader = Callable[[Path], RepoContract]
FileAdapterFactory = Callable[[Path, RepoContract], FilePort]
ShellFactory = Callable[[Path, RepoContract], ShellPort]
TestAdapterFactory = Callable[[ShellPort, RepoContract], TestPort]
TestRunner = Callable[[], "TestResult"]
TestRunnerFactory = Callable[[Path, RepoContract], TestRunner]
ImplementationStep = Callable[
    ["ImplementationContext"],
    ImplementationResult | Awaitable[ImplementationResult],
]
LockIdFactory = Callable[[TicketState], str]


class GitWorktreePort(Protocol):
    def create_worktree(
        self,
        repo_path: str | Path,
        ticket_key: str,
        short_lock_id: str,
        base_branch: str | None = None,
    ) -> WorktreeInfo: ...


@dataclass(frozen=True)
class ImplementationContext:
    state: TicketState
    contract: RepoContract
    repo_path: Path
    worktree_path: Path
    branch_name: str | None
    lock_id: str | None
    files: FilePort
    test_runner: TestRunner | None = None


@dataclass(frozen=True)
class _PreparedWorktree:
    repo_path: Path
    worktree_path: Path
    branch_name: str | None
    ticket_key: str
    lock_id: str | None


_DEFAULT_CONTRACT_DIR = Path("config/repos")
_SAFE_LOCK_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _load_or_scaffold_contract(
    state: TicketState,
    contract_path: Path,
    contract_loader: ContractLoader,
) -> tuple[RepoContract | None, str | None]:
    """Load the contract at contract_path, scaffolding it if it doesn't exist yet.

    Returns (contract, None) on success, (None, error_message) on failure.
    """
    try:
        return contract_loader(contract_path), None
    except FileNotFoundError:
        matched = _find_matching_contract(
            state,
            contract_path.parent,
            contract_loader,
            skip_path=contract_path,
        )
        if matched is not None:
            return matched, None
    except (RepoContractError, YAMLError, OSError) as exc:
        return None, f"repo contract missing or invalid: {exc}"

    repo_name = _repo_name(state)
    if repo_name is None:
        return None, "repository or repo_path is required to scaffold repo contract"

    try:
        contract = scaffold_repo_contract(
            repo_name=repo_name,
            repo_path=state.repo_path,
            contract_path=contract_path,
        )
        return contract, None
    except (RepoContractError, OSError) as exc:
        return None, f"repo contract scaffolding failed: {exc}"


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
        test_runner_factory: TestRunnerFactory | None = None,
        shell_factory: ShellFactory | None = None,
        execution_preflight: ExecutionPreflight | None = None,
        action_journal: GoalActionJournal | None = None,
    ) -> None:
        self._contract_dir = Path(contract_dir)
        self._contract_loader = contract_loader
        self._git = git or GitAdapter()
        self._file_adapter_factory = file_adapter_factory
        self._implementation_step = implementation_step or _prepare_only_step
        self._lock_id_factory = lock_id_factory or _new_short_lock_id
        self._shell_factory = shell_factory or _build_contract_shell
        self._test_runner_factory = test_runner_factory or (
            lambda worktree, contract: _make_contract_test_runner(
                worktree,
                contract,
                shell_factory=self._shell_factory,
            )
        )
        self._execution_preflight = execution_preflight
        self._action_journal = action_journal

    async def implement(self, state: TicketState) -> dict[str, Any]:
        contract_path = _contract_path(state, self._contract_dir)
        if contract_path is None:
            return _failed_implementation_update(
                "repository or repo_path is required to load repo contract"
            )

        contract, err = _load_or_scaffold_contract(
            state,
            contract_path,
            self._contract_loader,
        )
        if err is not None:
            return _failed_implementation_update(err)

        repo_path = _repo_path(state, contract)
        if repo_path is None:
            return _failed_implementation_update(
                "repo_path or repo.root is required to create worktree"
            )

        try:
            if self._execution_preflight is not None:
                self._execution_preflight.check(state)
            worktree = await self._prepare_worktree(state, repo_path)
            files = self._file_adapter_factory(worktree.worktree_path, contract)
            test_runner = self._test_runner_factory(
                worktree.worktree_path, contract
            )
            context = ImplementationContext(
                state=state,
                contract=contract,
                repo_path=worktree.repo_path,
                worktree_path=worktree.worktree_path,
                branch_name=worktree.branch_name,
                lock_id=worktree.lock_id,
                files=files,
                test_runner=test_runner,
            )
            implementation_result = self._implementation_step(context)
            if inspect.isawaitable(implementation_result):
                implementation_result = await implementation_result
        except (AgentSystemError, OSError, ValueError) as exc:
            return _failed_implementation_update(f"implementation failed: {exc}")

        return {
            "repo_path": str(worktree.repo_path),
            "worktree_path": str(worktree.worktree_path),
            "branch_name": worktree.branch_name,
            "lock_id": worktree.lock_id,
            "implementation_result": implementation_result,
        }

    async def _prepare_worktree(
        self,
        state: TicketState,
        repo_path: Path,
    ) -> _PreparedWorktree:
        existing = _worktree_path(state)
        if existing is not None or self._action_journal is None:
            return _worktree_info(
                state,
                repo_path,
                self._git,
                self._lock_id_factory,
            )
        if state.goal_id is None:
            raise ValueError("goal_id is required to journal worktree creation")

        lock_id = self._lock_id_factory(state)
        expected_path = repo_path / ".worktrees" / state.ticket_key / lock_id
        branch_name = f"agent/{state.ticket_key}/{lock_id}"

        def effect() -> _PreparedWorktree:
            return _worktree_info(
                state,
                repo_path,
                self._git,
                lambda ignored: lock_id,
            )

        def probe() -> ProbeResult[_PreparedWorktree]:
            if not expected_path.exists():
                return ProbeResult(found=False)
            prepared = _PreparedWorktree(
                repo_path=repo_path,
                worktree_path=expected_path,
                branch_name=branch_name,
                ticket_key=state.ticket_key,
                lock_id=state.lock_id or lock_id,
            )
            return ProbeResult(
                found=True,
                value=prepared,
                result_identity=str(expected_path),
            )

        async def restore(path: str) -> _PreparedWorktree:
            restored = Path(path)
            if not restored.exists():
                raise ValueError(f"completed worktree is missing: {restored}")
            return _PreparedWorktree(
                repo_path=repo_path,
                worktree_path=restored,
                branch_name=branch_name,
                ticket_key=state.ticket_key,
                lock_id=state.lock_id or lock_id,
            )

        outcome = await self._action_journal.execute(
            LoopState(
                goal_id=state.goal_id,
                contract_version=1,
                phase="implementing",
                iteration=state.implementation_attempts + 1,
            ),
            operation="worktree_create",
            natural_key=f"{state.ticket_key}:{lock_id}",
            request={
                "repo_path": str(repo_path),
                "ticket_key": state.ticket_key,
                "lock_id": lock_id,
                "base_branch": state.pull_request_base_branch,
            },
            effect=effect,
            probe=probe,
            result_identity=lambda prepared: str(prepared.worktree_path),
            restore=restore,
        )
        if outcome.value is None:
            raise ValueError("journaled worktree creation returned no path")
        return outcome.value


class AdapterTestService:
    """Run repository tests through the repo-contract backed test adapter."""

    def __init__(
        self,
        *,
        contract_dir: str | Path = _DEFAULT_CONTRACT_DIR,
        contract_loader: ContractLoader = load_repo_contract,
        shell_factory: ShellFactory | None = None,
        adapter_factory: TestAdapterFactory = LocalTestAdapter,
        action_journal: GoalActionJournal | None = None,
    ) -> None:
        self._contract_dir = Path(contract_dir)
        self._contract_loader = contract_loader
        self._shell_factory = shell_factory or _build_contract_shell
        self._adapter_factory = adapter_factory
        self._action_journal = action_journal

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
            contract = _load_contract_for_state(
                state,
                contract_path,
                self._contract_loader,
            )
        except (FileNotFoundError, RepoContractError, YAMLError, OSError) as exc:
            return _failed_result(f"repo contract missing or invalid: {exc}")

        try:
            shell = self._shell_factory(worktree_path, contract)
            adapter = self._adapter_factory(shell, contract)
            if self._action_journal is None:
                command_result = adapter.run_tests()
            else:
                if state.goal_id is None:
                    raise ValueError("goal_id is required to journal gate execution")
                candidate = state.candidate_sha or _worktree_content_digest(
                    worktree_path
                )

                def effect() -> CommandResult:
                    return adapter.run_tests()

                outcome = await self._action_journal.execute(
                    LoopState(
                        goal_id=state.goal_id,
                        contract_version=1,
                        phase="verifying",
                        iteration=state.implementation_attempts,
                        candidate_sha=candidate,
                    ),
                    operation="gate_run",
                    natural_key=f"{candidate}:test",
                    request={
                        "candidate_sha": candidate,
                        "gate": "test",
                        "command": contract.commands.test.command,
                    },
                    effect=effect,
                    # An ambiguous gate is safe to re-run. A completed result
                    # is restored from its canonical JSON identity below.
                    probe=lambda: ProbeResult(found=False),
                    result_identity=canonical_json,
                    restore=_command_result_from_identity,
                )
                if outcome.value is None:
                    raise ValueError("journaled gate returned no result")
                command_result = outcome.value
        except (AgentSystemError, OSError, ValueError) as exc:
            return _failed_result(f"test adapter failed: {exc}")

        return _command_result_to_test_result(command_result)


def _worktree_path(state: TicketState) -> Path | None:
    if not state.worktree_path:
        return None
    return Path(state.worktree_path)


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
    if state.pull_request_base_branch:
        created = git.create_worktree(
            repo_path,
            state.ticket_key,
            lock_id,
            base_branch=state.pull_request_base_branch,
        )
    else:
        created = git.create_worktree(repo_path, state.ticket_key, lock_id)
    return _PreparedWorktree(
        repo_path=created.repo_path,
        worktree_path=created.worktree_path,
        branch_name=created.branch_name,
        ticket_key=created.ticket_key,
        lock_id=state.lock_id or created.lock_id,
    )


def _contract_path(state: TicketState, contract_dir: Path) -> Path | None:
    repo_name = _repo_name(state)
    if repo_name is None:
        return None
    return contract_dir / f"{repo_name}.yaml"


def _load_contract_for_state(
    state: TicketState,
    contract_path: Path,
    contract_loader: ContractLoader,
) -> RepoContract:
    try:
        return contract_loader(contract_path)
    except FileNotFoundError:
        matched = _find_matching_contract(
            state,
            contract_path.parent,
            contract_loader,
            skip_path=contract_path,
        )
        if matched is not None:
            return matched
        raise


def _find_matching_contract(
    state: TicketState,
    contract_dir: Path,
    contract_loader: ContractLoader,
    *,
    skip_path: Path,
) -> RepoContract | None:
    repo_name = _repo_name(state)
    repo_path = _state_repo_path(state)
    if repo_name is None and repo_path is None:
        return None

    for candidate in sorted(contract_dir.glob("*.yaml")):
        if candidate == skip_path:
            continue
        try:
            contract = contract_loader(candidate)
        except (FileNotFoundError, RepoContractError, YAMLError, OSError):
            continue
        if _contract_matches_state(
            contract,
            repo_name=repo_name,
            repo_path=repo_path,
        ):
            return contract
    return None


def _contract_matches_state(
    contract: RepoContract,
    *,
    repo_name: str | None,
    repo_path: Path | None,
) -> bool:
    if (
        repo_name is not None
        and _normalize_repo_name(contract.repo.name) == repo_name
    ):
        return True
    if repo_path is not None:
        contract_root = Path(contract.repo.root).expanduser().resolve(strict=False)
        return contract_root == repo_path
    return False


def _state_repo_path(state: TicketState) -> Path | None:
    if not state.repo_path:
        return None
    return Path(state.repo_path).expanduser().resolve(strict=False)


def _repo_name(state: TicketState) -> str | None:
    if state.repository:
        return _normalize_repo_name(state.repository)
    if state.repo_path:
        return Path(state.repo_path).resolve(strict=False).name
    return None


def _normalize_repo_name(repository: str) -> str:
    normalized = repository.rstrip("/")
    normalized = normalized.removesuffix(".git")
    return Path(normalized).name


def _build_contract_shell(
    worktree_path: Path,
    contract: RepoContract,
    *,
    sandbox: Sandbox,
) -> ShellPort:
    allowed_commands = [
        spec.command for spec in contract.commands.gate_commands().values()
    ]
    return LocalShellAdapter(
        worktree_path,
        allowed_commands=allowed_commands,
        sandbox=sandbox,
    )


class RuntimeShellFactory:
    """Build every production contract shell from the same live preflight."""

    requires_enforcing_sandbox = True

    def __init__(self, preflight: ExecutionPreflight) -> None:
        self._preflight = preflight

    def __call__(self, worktree_path: Path, contract: RepoContract) -> ShellPort:
        return _build_contract_shell(
            worktree_path,
            contract,
            sandbox=self._preflight.check(),
        )

    def probe_attestation(self) -> SandboxAttestation:
        """Execute a harmless command through the production wrapper path."""

        with tempfile.TemporaryDirectory(prefix="ticket-agent-sandbox-probe-") as raw:
            root = Path(raw)
            shell = LocalShellAdapter(
                root,
                allowed_commands=(("/bin/true",),),
                sandbox=self._preflight.check(),
            )
            result = shell.run(
                ("/bin/true",),
                policy=CommandExecutionPolicy(),
            )
        if not result.ok or result.sandbox_attestation is None:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise AgentSystemError(f"sandbox enforcement probe failed: {detail}")
        return result.sandbox_attestation


def _make_contract_test_runner(
    worktree_path: Path,
    contract: RepoContract,
    *,
    shell_factory: ShellFactory = _build_contract_shell,
) -> TestRunner:
    """Build a zero-arg runner for the repo-contract test command.

    The returned callable runs the contract's test command through the same
    TestAdapter path as AdapterTestService, so the model-callable run_tests
    action can never run an auto-detected or arbitrary command.
    """

    def run() -> TestResult:
        try:
            shell = shell_factory(worktree_path, contract)
            adapter = LocalTestAdapter(shell, contract)
            command_result = adapter.run_tests()
        except (AgentSystemError, OSError, ValueError) as exc:
            return _failed_result(f"test adapter failed: {exc}")
        return _command_result_to_test_result(command_result)

    return run


def _prepare_only_step(context: ImplementationContext) -> ImplementationResult:
    return {
        "status": "prepared",
        "changed_files": [],
        "message": "worktree prepared; no implementation step configured",
    }


def _new_short_lock_id(state: TicketState) -> str:
    if state.lock_id:
        shortened = _short_safe_id(state.lock_id)
        if shortened:
            return shortened
    return uuid4().hex[:8]


def _short_safe_id(value: str) -> str:
    cleaned = _SAFE_LOCK_ID_CHARS.sub("-", value).strip("-_")
    return cleaned[:8]


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


def _command_result_from_identity(identity: str) -> CommandResult:
    payload = json.loads(identity)
    attestation_payload = payload.get("sandbox_attestation")
    attestation = (
        SandboxAttestation(**attestation_payload) if attestation_payload else None
    )
    return CommandResult(
        command=tuple(payload["command"]),
        returncode=int(payload["returncode"]),
        stdout=str(payload.get("stdout", "")),
        stderr=str(payload.get("stderr", "")),
        timed_out=bool(payload.get("timed_out", False)),
        sandbox_attestation=attestation,
    )


def _worktree_content_digest(worktree_path: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(worktree_path.rglob("*")):
        relative = path.relative_to(worktree_path)
        if ".git" in relative.parts or ".worktrees" in relative.parts:
            continue
        if path.is_file() and not path.is_symlink():
            entries.append((relative.as_posix(), digest(path.read_bytes())))
    return digest(entries)


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


class AutoApprovalService:
    """Default MVP execution approval.

    The human approves the full plan in Slack before Jira tickets are
    created/marked ai-ready, so individual ticket execution does not require
    another manual approval.
    """

    async def request_approval(self, state: TicketState) -> bool:
        return True


def _failed_implementation_update(error: str) -> dict[str, Any]:
    return {
        "implementation_result": {
            "status": "failed",
            "changed_files": [],
            "error": error,
        },
        "error": error,
    }


__all__ = [
    "AdapterTestService",
    "AutoApprovalService",
    "GhPullRequestOpener",
    "GitPullRequestPort",
    "GitService",
    "ImplementationContext",
    "LocalImplementationService",
    "PullRequestOpener",
    "RuntimeShellFactory",
]

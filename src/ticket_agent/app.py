"""Production process composition for the Slack and Jira agent runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import Any

from ticket_agent.detection.detector import DetectionComponent
from ticket_agent.detection.jira_search import JiraDetectionSearchClient
from ticket_agent.detection.ownership import OwnershipChecker
from ticket_agent.intake.approval_flow import ApprovalFlow, SlackPoster
from ticket_agent.intake.intent_resolver import IntakeIntentResolver
from ticket_agent.intake.jira_writer import JiraWriter
from ticket_agent.intake.proposal_generator import DeterministicProposalGenerator
from ticket_agent.intake.proposal_store import ProposalStore
from ticket_agent.intake.slack_listener import (
    SlackIntakeListener,
    SlackSDKPoster,
    SlackSocketModeService,
)
from ticket_agent.jira.client import JiraClient, JiraRestClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_MAX_ATTEMPTS,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
)
from ticket_agent.jira.execution_coordinator import JiraExecutionCoordinator
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.locking.checkpointer import SQLiteCheckpointer
from ticket_agent.locking.reconciler import reconcile_expired_locks
from ticket_agent.locking.sqlite_store import SQLiteLockManager
from ticket_agent.orchestrator.execution_approval import (
    ExecutionApprovalCommandHandler,
    SQLiteExecutionApprovalStore,
    SlackExecutionApprovalService,
)
from ticket_agent.orchestrator.execution_worker import ExecutionWorker
from ticket_agent.orchestrator.git_services import GitService, WorktreeCleanupService
from ticket_agent.orchestrator.graph import build_persistent_ticket_graph
from ticket_agent.orchestrator.jira_services import JiraEscalationService
from ticket_agent.orchestrator.local_services import (
    AdapterTestService,
    LocalImplementationService,
)
from ticket_agent.orchestrator.model_services import (
    IterativeImplementationService,
    ModelRouterPlannerService,
    ModelRouterProtocol,
    ModelRouterReviewService,
)
from ticket_agent.orchestrator.node_runner import TicketNodeRunner
from ticket_agent.orchestrator.runner import OrchestratorRunner
from ticket_agent.orchestrator.services import (
    EscalationService,
    ImplementationService,
    PlannerService,
    PullRequestService,
    ReviewService,
    TestService,
)
from ticket_agent.router.factory import create_model_router


EventEmitter = Callable[[str, Mapping[str, Any]], Any]
SlackLoop = Callable[[], Awaitable[None]]

DEFAULT_DATA_DIR = Path("data")
DEFAULT_COMPONENT_ID = "agent-system"
DEFAULT_ENV_PATH = Path("~/config/agent-system.env")
ENV_PATH_VAR = "AGENT_SYSTEM_ENV_PATH"
REQUIRED_ENV_VARS = (
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "JIRA_BASE_URL",
    "JIRA_USER_EMAIL",
    "JIRA_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
)

_LOGGER = logging.getLogger(__name__)
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StartupConfigError(RuntimeError):
    """Raised when production startup configuration is invalid."""


class RuntimeLoopExited(RuntimeError):
    """Raised when a long-running runtime loop exits unexpectedly."""


class _ShutdownRequested(Exception):
    """Internal sentinel used to cancel a TaskGroup for graceful shutdown."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Configuration for composing the long-running agent process."""

    component_id: str = DEFAULT_COMPONENT_ID
    data_dir: Path = DEFAULT_DATA_DIR
    intake_channel: str | None = None
    execution_approval_channel: str | None = None
    repo_defaults: Mapping[str, Mapping[str, str]] | None = None
    poll_interval_seconds: float = 30.0
    max_backoff_seconds: float = 300.0
    heartbeat_interval_s: float = 600.0
    reconcile_interval_seconds: float = 300.0
    reconcile_batch_size: int | None = None
    contract_dir: Path = Path("config/repos")
    pull_request_base_branch: str = "main"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated production app configuration loaded from environment."""

    env_path: Path
    env_file_loaded: bool
    slack_bot_token: str
    slack_app_token: str
    jira_base_url: str
    jira_user_email: str
    jira_api_key: str
    jira_timeout_s: float
    jira_field_map: Mapping[str, str]
    runtime: RuntimeConfig


@dataclass(slots=True)
class AgentSystemRuntime:
    """Fully wired runtime components for one agent-system process."""

    listener: SlackIntakeListener
    detector: DetectionComponent
    worker: ExecutionWorker
    graph: Any
    checkpointer: SQLiteCheckpointer
    approval_store: SQLiteExecutionApprovalStore
    proposal_store: ProposalStore
    lock_manager: SQLiteLockManager
    execution_approval_handler: ExecutionApprovalCommandHandler
    queue: asyncio.Queue[str]
    jira_client: JiraClient
    config: RuntimeConfig
    database_paths: Mapping[str, Path]
    emit: EventEmitter | None = None

    async def run_execution_services(self) -> None:
        """Run detection, worker, and reconciler loops until cancelled."""

        await asyncio.gather(
            self.detector.run_forever(),
            self.worker.run_forever(),
            self.run_reconciler_loop(),
        )

    async def run_reconciler_loop(self) -> None:
        """Periodically restore Jira state for expired local locks."""

        await _reconciler_loop(
            self.lock_manager,
            self.jira_client,
            interval_s=self.config.reconcile_interval_seconds,
            limit=self.config.reconcile_batch_size,
            emit=self.emit,
        )

    def close(self) -> None:
        """Close local SQLite resources owned by the runtime."""

        self.checkpointer.close()
        self.approval_store.close()
        self.proposal_store.close()
        self.lock_manager.close()


def build_runtime(
    *,
    jira_client: JiraClient,
    slack: SlackPoster,
    config: RuntimeConfig | None = None,
    model_router: ModelRouterProtocol | None = None,
    planner: PlannerService | None = None,
    implementation: ImplementationService | None = None,
    tests: TestService | None = None,
    review: ReviewService | None = None,
    pull_request: PullRequestService | None = None,
    escalation: EscalationService | None = None,
    queue: asyncio.Queue[str] | None = None,
    emit: EventEmitter | None = None,
) -> AgentSystemRuntime:
    """Compose the production listener, graph, detector, and worker."""

    runtime_config = config or RuntimeConfig()
    _validate_runtime_config(runtime_config)
    data_dir = Path(runtime_config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    database_paths = _database_paths(data_dir)

    approval_channel = (
        runtime_config.execution_approval_channel
        or runtime_config.intake_channel
        or ""
    )

    proposal_store = ProposalStore(database_paths["proposal_store"])
    approval_store = SQLiteExecutionApprovalStore(
        database_paths["execution_approvals"]
    )
    checkpointer = SQLiteCheckpointer(database_paths["graph_checkpoints"])
    lock_manager = SQLiteLockManager(
        database_paths["ticket_locks"],
        component_id=runtime_config.component_id,
        emit=emit,
    )

    router = model_router
    if planner is None or implementation is None or review is None:
        router = router or create_model_router()
    implementation_loop = (
        None if implementation is not None else IterativeImplementationService(router)
    )
    worktree_cleaner = WorktreeCleanupService()

    execution_service = JiraExecutionService(
        jira_client,
        runtime_config.component_id,
        emit=emit,
    )
    node_runner = TicketNodeRunner(
        planner=planner or ModelRouterPlannerService(router),
        approval=SlackExecutionApprovalService(
            store=approval_store,
            slack=slack,
            default_channel=approval_channel,
        ),
        implementation=implementation
        or LocalImplementationService(
            contract_dir=runtime_config.contract_dir,
            implementation_step=implementation_loop.implement_context,
        ),
        tests=tests or AdapterTestService(contract_dir=runtime_config.contract_dir),
        review=review or ModelRouterReviewService(router),
        pull_request=pull_request
        or GitService(base_branch=runtime_config.pull_request_base_branch),
        escalation=escalation or JiraEscalationService(execution_service),
    )
    graph = build_persistent_ticket_graph(node_runner, checkpointer=checkpointer)

    runner = OrchestratorRunner(
        graph=graph,
        lock_manager=lock_manager,
        component_id=runtime_config.component_id,
        event_emitter=emit,
        claim_ticket=execution_service.mark_claimed,
        checkpointer=checkpointer,
        heartbeat_interval_s=runtime_config.heartbeat_interval_s,
    )
    detector_queue = queue or asyncio.Queue()
    detector = DetectionComponent(
        client=JiraDetectionSearchClient(jira_client),
        queue=detector_queue,
        ownership_checker=OwnershipChecker(
            component_id=runtime_config.component_id,
            lock_lookup=lock_manager.current_lock,
        ),
        poll_interval_seconds=runtime_config.poll_interval_seconds,
        max_backoff_seconds=runtime_config.max_backoff_seconds,
        emit=emit,
    )
    coordinator = _MarkDoneCoordinator(
        JiraExecutionCoordinator(
            JiraWorkItemLoader(jira_client),
            execution_service,
            runner,
            emit=emit,
            slack=slack,
            worktree_cleaner=worktree_cleaner,
            checkpointer=checkpointer,
        ),
        detector,
    )
    worker = ExecutionWorker(detector_queue, coordinator, emit=emit)

    approval_flow = ApprovalFlow(
        resolver=IntakeIntentResolver(),
        generator=DeterministicProposalGenerator(),
        store=proposal_store,
        jira_writer=JiraWriter(jira_client),
        slack=slack,
        repo_defaults=runtime_config.repo_defaults,
        emit=emit,
    )
    execution_approval_handler = ExecutionApprovalCommandHandler(
        store=approval_store,
        graph=graph,
        slack=slack,
    )
    listener = SlackIntakeListener(
        approval_flow=approval_flow,
        store=proposal_store,
        intake_channel=runtime_config.intake_channel,
        execution_approval_handler=execution_approval_handler,
        emit=emit,
    )

    return AgentSystemRuntime(
        listener=listener,
        detector=detector,
        worker=worker,
        graph=graph,
        checkpointer=checkpointer,
        approval_store=approval_store,
        proposal_store=proposal_store,
        lock_manager=lock_manager,
        execution_approval_handler=execution_approval_handler,
        queue=detector_queue,
        jira_client=jira_client,
        config=runtime_config,
        database_paths=database_paths,
        emit=emit,
    )


async def run_runtime(
    runtime: AgentSystemRuntime,
    *,
    slack_loop: SlackLoop,
    shutdown_event: asyncio.Event | None = None,
    emit: EventEmitter | None = None,
) -> None:
    """Run all production loops concurrently until failure or shutdown."""

    stop_event = shutdown_event or asyncio.Event()
    try:
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(
                _run_named_loop("slack_intake_listener", slack_loop, emit),
                name="slack-intake-listener",
            )
            task_group.create_task(
                _run_named_loop("detection_polling", runtime.detector.run_forever, emit),
                name="detection-polling",
            )
            task_group.create_task(
                _run_named_loop("execution_worker", runtime.worker.run_forever, emit),
                name="execution-worker",
            )
            task_group.create_task(
                _run_named_loop(
                    "lock_reconciler",
                    runtime.run_reconciler_loop,
                    emit,
                ),
                name="lock-reconciler",
            )
            task_group.create_task(
                _shutdown_watcher(stop_event),
                name="shutdown-watcher",
            )
    except* _ShutdownRequested:
        await _emit(emit, "app.shutdown_complete", {})


async def main(
    *,
    env: Mapping[str, str] | None = None,
    env_path: str | Path | None = None,
    jira_client: JiraClient | None = None,
    slack: SlackPoster | None = None,
    model_router: ModelRouterProtocol | None = None,
    planner: PlannerService | None = None,
    implementation: ImplementationService | None = None,
    tests: TestService | None = None,
    review: ReviewService | None = None,
    pull_request: PullRequestService | None = None,
    escalation: EscalationService | None = None,
    config: RuntimeConfig | None = None,
    slack_loop: SlackLoop | None = None,
    shutdown_event: asyncio.Event | None = None,
    emit: EventEmitter | None = None,
    install_signal_handlers: bool = True,
) -> int:
    """Start the full production runtime from environment configuration."""

    app_config = load_app_config(env=env, env_path=env_path, install=True)
    _configure_logging()
    event_emitter = emit or _logging_event_emitter
    runtime_config = config or app_config.runtime
    stop_event = shutdown_event or asyncio.Event()
    if install_signal_handlers:
        _install_signal_handlers(stop_event, event_emitter)

    runtime: AgentSystemRuntime | None = None
    await _emit(event_emitter, "app.config_loaded", _config_payload(app_config))
    try:
        if jira_client is None:
            jira_client = JiraRestClient(
                base_url=app_config.jira_base_url,
                user_email=app_config.jira_user_email,
                api_key=app_config.jira_api_key,
                timeout_s=app_config.jira_timeout_s,
                field_map=app_config.jira_field_map,
            )
        if slack is None:
            slack = _build_slack_poster(app_config, runtime_config)

        runtime = build_runtime(
            jira_client=jira_client,
            slack=slack,
            config=runtime_config,
            model_router=model_router,
            planner=planner,
            implementation=implementation,
            tests=tests,
            review=review,
            pull_request=pull_request,
            escalation=escalation,
            emit=event_emitter,
        )
        await _emit(event_emitter, "app.starting", _runtime_payload(runtime))

        if slack_loop is None:
            slack_service = SlackSocketModeService(
                runtime.listener,
                bot_token=app_config.slack_bot_token,
                app_token=app_config.slack_app_token,
            )
            slack_loop = slack_service.run_forever

        await run_runtime(
            runtime,
            slack_loop=slack_loop,
            shutdown_event=stop_event,
            emit=event_emitter,
        )
        return 0
    finally:
        if runtime is not None:
            runtime.close()
            await _emit(event_emitter, "app.closed", {})


def run() -> int:
    """Synchronous console-script entrypoint for ``ticket-agent``."""

    try:
        return asyncio.run(main())
    except StartupConfigError as exc:
        _configure_logging()
        _LOGGER.critical("startup_config_error: %s", exc)
        return 2
    except KeyboardInterrupt:
        _LOGGER.info("shutdown_requested")
        return 130
    except Exception:
        _LOGGER.exception("ticket_agent_process_failed")
        return 1


def run_process(
    runtime: AgentSystemRuntime,
    *,
    bot_token: str | None = None,
    app_token: str | None = None,
) -> None:
    """Compatibility wrapper for callers that already built a runtime."""

    bot_token = bot_token or _env_value(os.environ, "SLACK_BOT_TOKEN")
    app_token = app_token or _env_value(os.environ, "SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        raise RuntimeError(
            "run_process requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN"
        )
    slack_service = SlackSocketModeService(
        runtime.listener,
        bot_token=bot_token,
        app_token=app_token,
    )
    try:
        asyncio.run(run_runtime(runtime, slack_loop=slack_service.run_forever))
    finally:
        runtime.close()


def load_app_config(
    *,
    env: Mapping[str, str] | None = None,
    env_path: str | Path | None = None,
    install: bool = False,
) -> AppConfig:
    """Load dotenv-style config, validate required variables, and coerce types."""

    base_env = dict(os.environ if env is None else env)
    resolved_env_path = _resolve_env_path(base_env, env_path)
    file_values: dict[str, str] = {}
    env_file_loaded = False
    if resolved_env_path.exists():
        file_values = _parse_env_file(resolved_env_path)
        env_file_loaded = True
    elif env_path is not None or base_env.get(ENV_PATH_VAR):
        raise StartupConfigError(f"env file does not exist: {resolved_env_path}")

    merged_env = dict(base_env)
    for key, value in file_values.items():
        if not merged_env.get(key):
            merged_env[key] = value

    missing = [name for name in REQUIRED_ENV_VARS if not merged_env.get(name)]
    if missing:
        raise StartupConfigError(
            "missing required environment variables: "
            + ", ".join(missing)
            + f" (checked env file: {resolved_env_path})"
        )

    if install:
        for key, value in merged_env.items():
            os.environ[str(key)] = str(value)

    runtime_config = RuntimeConfig(
        component_id=_env_value(merged_env, "AGENT_SYSTEM_COMPONENT_ID")
        or DEFAULT_COMPONENT_ID,
        data_dir=Path(_env_value(merged_env, "AGENT_SYSTEM_DATA_DIR") or DEFAULT_DATA_DIR),
        intake_channel=_first_env(
            merged_env,
            "AGENT_SYSTEM_INTAKE_CHANNEL",
            "SLACK_INTAKE_CHANNEL",
            "INTAKE_CHANNEL",
            "SLACK_DEFAULT_CHANNEL",
        ),
        execution_approval_channel=_first_env(
            merged_env,
            "AGENT_SYSTEM_EXECUTION_APPROVAL_CHANNEL",
            "SLACK_EXECUTION_APPROVAL_CHANNEL",
            "SLACK_DEFAULT_CHANNEL",
            "INTAKE_CHANNEL",
        ),
        poll_interval_seconds=_float_env(
            merged_env,
            "AGENT_SYSTEM_POLL_INTERVAL_SECONDS",
            default=30.0,
        ),
        max_backoff_seconds=_float_env(
            merged_env,
            "AGENT_SYSTEM_MAX_BACKOFF_SECONDS",
            default=300.0,
        ),
        heartbeat_interval_s=_float_env(
            merged_env,
            "AGENT_SYSTEM_HEARTBEAT_INTERVAL_SECONDS",
            default=600.0,
        ),
        reconcile_interval_seconds=_float_env(
            merged_env,
            "AGENT_SYSTEM_RECONCILE_INTERVAL_SECONDS",
            default=300.0,
        ),
        reconcile_batch_size=_optional_int_env(
            merged_env,
            "AGENT_SYSTEM_RECONCILE_BATCH_SIZE",
        ),
        contract_dir=Path(
            _first_env(
                merged_env,
                "AGENT_SYSTEM_REPO_CONFIG_PATH",
                "AGENT_SYSTEM_CONTRACT_DIR",
            )
            or "config/repos"
        ),
        pull_request_base_branch=_env_value(
            merged_env,
            "AGENT_SYSTEM_PULL_REQUEST_BASE_BRANCH",
        )
        or "main",
    )
    _validate_runtime_config(runtime_config)

    return AppConfig(
        env_path=resolved_env_path,
        env_file_loaded=env_file_loaded,
        slack_bot_token=_required(merged_env, "SLACK_BOT_TOKEN"),
        slack_app_token=_required(merged_env, "SLACK_APP_TOKEN"),
        jira_base_url=_required(merged_env, "JIRA_BASE_URL"),
        jira_user_email=_required(merged_env, "JIRA_USER_EMAIL"),
        jira_api_key=_required(merged_env, "JIRA_API_KEY"),
        jira_timeout_s=_float_env(merged_env, "JIRA_TIMEOUT_SECONDS", default=30.0),
        jira_field_map=_jira_field_map(merged_env),
        runtime=runtime_config,
    )


class _MarkDoneCoordinator:
    def __init__(
        self,
        coordinator: JiraExecutionCoordinator,
        detector: DetectionComponent,
    ) -> None:
        self._coordinator = coordinator
        self._detector = detector

    async def run_ticket(self, ticket_key: str) -> Any:
        try:
            return await self._coordinator.run_ticket(ticket_key)
        finally:
            self._detector.mark_done(ticket_key)


async def _reconciler_loop(
    lock_manager: SQLiteLockManager,
    jira_client: JiraClient,
    *,
    interval_s: float,
    limit: int | None = None,
    emit: EventEmitter | None = None,
) -> None:
    while True:
        await reconcile_expired_locks(
            lock_manager,
            jira_client,
            limit=limit,
            emit=emit,
        )
        await asyncio.sleep(interval_s)


async def _run_named_loop(
    name: str,
    loop: SlackLoop,
    emit: EventEmitter | None,
) -> None:
    await _emit(emit, "app.loop_started", {"loop": name})
    try:
        await loop()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _emit(
            emit,
            "app.loop_failed",
            {
                "loop": name,
                "error_type": exc.__class__.__name__,
                "error": str(exc) or exc.__class__.__name__,
            },
        )
        raise
    else:
        exc = RuntimeLoopExited(f"runtime loop exited unexpectedly: {name}")
        await _emit(
            emit,
            "app.loop_failed",
            {
                "loop": name,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        raise exc
    finally:
        await _emit(emit, "app.loop_stopped", {"loop": name})


async def _shutdown_watcher(shutdown_event: asyncio.Event) -> None:
    await shutdown_event.wait()
    raise _ShutdownRequested()


async def _emit(
    emit: EventEmitter | None,
    event_name: str,
    payload: Mapping[str, Any],
) -> None:
    if emit is None:
        return
    result = emit(event_name, payload)
    if isawaitable(result):
        await result


def _build_slack_poster(
    app_config: AppConfig,
    runtime_config: RuntimeConfig,
) -> SlackSDKPoster:
    try:
        from slack_sdk.web import WebClient
    except ImportError as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError("slack_sdk is required to run ticket-agent") from exc
    return SlackSDKPoster(
        WebClient(token=app_config.slack_bot_token),
        default_channel=runtime_config.intake_channel
        or runtime_config.execution_approval_channel,
    )


def _install_signal_handlers(
    shutdown_event: asyncio.Event,
    emit: EventEmitter,
) -> None:
    loop = asyncio.get_running_loop()

    def _request_shutdown(signal_name: str) -> None:
        if shutdown_event.is_set():
            return
        result = emit("app.shutdown_requested", {"signal": signal_name})
        if isawaitable(result):
            loop.create_task(result)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except (NotImplementedError, RuntimeError, ValueError):
            try:
                signal.signal(
                    sig,
                    lambda _signum, _frame, name=sig.name: loop.call_soon_threadsafe(
                        _request_shutdown,
                        name,
                    ),
                )
            except (RuntimeError, ValueError):
                continue


def _database_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "proposal_store": data_dir / "intake_proposals.sqlite3",
        "execution_approvals": data_dir / "execution_approvals.sqlite3",
        "graph_checkpoints": data_dir / "ticket_graph_checkpoints.sqlite3",
        "ticket_locks": data_dir / "ticket_locks.sqlite3",
    }


def _validate_runtime_config(config: RuntimeConfig) -> None:
    if not str(config.component_id).strip():
        raise StartupConfigError("AGENT_SYSTEM_COMPONENT_ID must not be blank")
    _positive("AGENT_SYSTEM_POLL_INTERVAL_SECONDS", config.poll_interval_seconds)
    _positive("AGENT_SYSTEM_MAX_BACKOFF_SECONDS", config.max_backoff_seconds)
    _positive("AGENT_SYSTEM_HEARTBEAT_INTERVAL_SECONDS", config.heartbeat_interval_s)
    _positive(
        "AGENT_SYSTEM_RECONCILE_INTERVAL_SECONDS",
        config.reconcile_interval_seconds,
    )
    if config.max_backoff_seconds < config.poll_interval_seconds:
        raise StartupConfigError(
            "AGENT_SYSTEM_MAX_BACKOFF_SECONDS must be >= "
            "AGENT_SYSTEM_POLL_INTERVAL_SECONDS"
        )
    if config.reconcile_batch_size is not None and config.reconcile_batch_size < 1:
        raise StartupConfigError("AGENT_SYSTEM_RECONCILE_BATCH_SIZE must be positive")


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise StartupConfigError(f"{name} must be positive")


def _resolve_env_path(
    env: Mapping[str, str],
    env_path: str | Path | None,
) -> Path:
    raw = str(env_path) if env_path is not None else _env_value(env, ENV_PATH_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_ENV_PATH.expanduser()


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parsed = _parse_env_line(line, path=path, lineno=lineno)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def _parse_env_line(
    line: str,
    *,
    path: Path,
    lineno: int,
) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    key, separator, raw_value = stripped.partition("=")
    key = key.strip()
    if not separator or not _ENV_KEY_RE.match(key):
        raise StartupConfigError(
            f"invalid env file line {lineno} in {path}: expected KEY=VALUE"
        )
    return key, _parse_env_value(raw_value.strip())


def _parse_env_value(raw_value: str) -> str:
    if (
        len(raw_value) >= 2
        and raw_value[0] == raw_value[-1]
        and raw_value[0] in {"'", '"'}
    ):
        return raw_value[1:-1]
    return raw_value.split(" #", maxsplit=1)[0].strip()


def _required(env: Mapping[str, str], name: str) -> str:
    value = _env_value(env, name)
    if value is None:
        raise StartupConfigError(f"missing required environment variable: {name}")
    return value


def _env_value(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _first_env(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = _env_value(env, name)
        if value:
            return value
    return None


def _float_env(env: Mapping[str, str], name: str, *, default: float) -> float:
    raw = _env_value(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise StartupConfigError(f"{name} must be a number") from exc
    if value <= 0:
        raise StartupConfigError(f"{name} must be positive")
    return value


def _optional_int_env(env: Mapping[str, str], name: str) -> int | None:
    raw = _env_value(env, name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise StartupConfigError(f"{name} must be an integer") from exc
    if value < 1:
        raise StartupConfigError(f"{name} must be positive")
    return value


def _jira_field_map(env: Mapping[str, str]) -> Mapping[str, str]:
    mapping = {
        logical: value
        for env_name, logical in _JIRA_FIELD_ENV_NAMES.items()
        if (value := _env_value(env, env_name))
    }
    raw_json = _env_value(env, "JIRA_FIELD_MAP_JSON")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise StartupConfigError("JIRA_FIELD_MAP_JSON must be valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise StartupConfigError("JIRA_FIELD_MAP_JSON must be a JSON object")
        for logical, jira_field in parsed.items():
            if not isinstance(logical, str) or not isinstance(jira_field, str):
                raise StartupConfigError(
                    "JIRA_FIELD_MAP_JSON keys and values must be strings"
                )
            mapping[logical] = jira_field
    return mapping


_JIRA_FIELD_ENV_NAMES = {
    "JIRA_FIELD_AGENT_ASSIGNED_COMPONENT": FIELD_AGENT_ASSIGNED_COMPONENT,
    "JIRA_FIELD_AGENT_RETRY_COUNT": FIELD_AGENT_RETRY_COUNT,
    "JIRA_FIELD_AGENT_CAPABILITIES_NEEDED": FIELD_AGENT_CAPABILITIES_NEEDED,
    "JIRA_FIELD_REPOSITORY": FIELD_REPOSITORY,
    "JIRA_FIELD_REPO_PATH": FIELD_REPO_PATH,
    "JIRA_FIELD_MAX_ATTEMPTS": FIELD_MAX_ATTEMPTS,
}


def _configure_logging() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _logging_event_emitter(event_name: str, payload: Mapping[str, Any]) -> None:
    _LOGGER.info(
        json.dumps(
            {
                "event": event_name,
                **_jsonable_mapping(payload),
            },
            sort_keys=True,
        )
    )


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            result[str(key)] = str(value)
        elif isinstance(value, Mapping):
            result[str(key)] = _jsonable_mapping(value)
        else:
            result[str(key)] = value
    return result


def _config_payload(app_config: AppConfig) -> dict[str, Any]:
    return {
        "env_path": str(app_config.env_path),
        "env_file_loaded": app_config.env_file_loaded,
    }


def _runtime_payload(runtime: AgentSystemRuntime) -> dict[str, Any]:
    config = runtime.config
    return {
        "component_id": config.component_id,
        "data_dir": str(config.data_dir),
        "db_paths": {
            name: str(path) for name, path in runtime.database_paths.items()
        },
        "poll_interval_seconds": config.poll_interval_seconds,
        "max_backoff_seconds": config.max_backoff_seconds,
        "heartbeat_interval_seconds": config.heartbeat_interval_s,
        "reconcile_interval_seconds": config.reconcile_interval_seconds,
        "repo_config_path": str(config.contract_dir),
        "pull_request_base_branch": config.pull_request_base_branch,
        "intake_channel_configured": bool(config.intake_channel),
        "execution_approval_channel_configured": bool(
            config.execution_approval_channel
        ),
    }


__all__ = [
    "AgentSystemRuntime",
    "AppConfig",
    "REQUIRED_ENV_VARS",
    "RuntimeConfig",
    "RuntimeLoopExited",
    "StartupConfigError",
    "build_runtime",
    "load_app_config",
    "main",
    "run",
    "run_process",
    "run_runtime",
]

"""Process composition for the Slack intake and ticket execution runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    load_slack_env,
    run_socket_mode,
)
from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.execution_coordinator import JiraExecutionCoordinator
from ticket_agent.jira.execution_service import JiraExecutionService
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.locking.checkpointer import SQLiteCheckpointer
from ticket_agent.locking.sqlite_store import SQLiteLockManager
from ticket_agent.orchestrator.execution_approval import (
    ExecutionApprovalCommandHandler,
    SQLiteExecutionApprovalStore,
    SlackExecutionApprovalService,
)
from ticket_agent.orchestrator.execution_worker import ExecutionWorker
from ticket_agent.orchestrator.git_services import GitService
from ticket_agent.orchestrator.graph import build_persistent_ticket_graph
from ticket_agent.orchestrator.jira_services import JiraEscalationService
from ticket_agent.orchestrator.local_services import (
    AdapterTestService,
    LocalImplementationService,
)
from ticket_agent.orchestrator.model_services import (
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

DEFAULT_DATA_DIR = Path("data")
DEFAULT_COMPONENT_ID = "agent-system"


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
    contract_dir: Path = Path("config/repos")
    pull_request_base_branch: str = "main"


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

    async def run_execution_services(self) -> None:
        """Run detection and execution workers until cancelled."""

        await asyncio.gather(
            self.detector.run_forever(),
            self.worker.run_forever(),
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
    data_dir = Path(runtime_config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    approval_channel = (
        runtime_config.execution_approval_channel
        or runtime_config.intake_channel
    )
    if not approval_channel:
        raise ValueError(
            "execution_approval_channel or intake_channel is required"
        )

    proposal_store = ProposalStore(data_dir / "intake_proposals.sqlite3")
    approval_store = SQLiteExecutionApprovalStore(
        data_dir / "execution_approvals.sqlite3"
    )
    checkpointer = SQLiteCheckpointer(data_dir / "ticket_graph_checkpoints.sqlite3")
    lock_manager = SQLiteLockManager(
        data_dir / "ticket_locks.sqlite3",
        component_id=runtime_config.component_id,
        emit=emit,
    )

    router = model_router
    if planner is None or review is None:
        router = router or create_model_router()

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
        or LocalImplementationService(contract_dir=runtime_config.contract_dir),
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
    )


def run_process(
    runtime: AgentSystemRuntime,
    *,
    bot_token: str | None = None,
    app_token: str | None = None,
) -> None:
    """Run the execution loop and blocking Slack Socket Mode listener."""

    background = _ExecutionServiceTask(runtime.run_execution_services)
    background.start()
    try:
        run_socket_mode(runtime.listener, bot_token=bot_token, app_token=app_token)
    finally:
        background.stop()
        runtime.close()


def main(*, jira_client: JiraClient | None = None) -> int:
    """Start the agent-system process with a caller-supplied Jira client."""

    if jira_client is None:
        raise RuntimeError(
            "ticket_agent.app.main requires a JiraClient implementation. "
            "Use build_runtime(..., jira_client=...) from the deployment layer."
        )

    bot_token, app_token, intake_channel = load_slack_env()

    try:
        from slack_sdk.web import WebClient
    except ImportError as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "slack_sdk is required to run the Slack process"
        ) from exc

    slack = SlackSDKPoster(
        WebClient(token=bot_token),
        default_channel=intake_channel,
    )
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            intake_channel=intake_channel,
            execution_approval_channel=intake_channel,
        ),
    )
    run_process(runtime, bot_token=bot_token, app_token=app_token)
    return 0


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


class _ExecutionServiceTask:
    def __init__(self, coro_factory: Callable[[], Any]) -> None:
        self._coro_factory = coro_factory
        self._task: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        import threading

        thread = threading.Thread(target=self._run, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        if self._loop is None or self._task is None:
            return
        self._loop.call_soon_threadsafe(self._task.cancel)
        self._thread.join(timeout=10)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._task = loop.create_task(self._coro_factory())
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()


__all__ = [
    "AgentSystemRuntime",
    "RuntimeConfig",
    "build_runtime",
    "main",
    "run_process",
]

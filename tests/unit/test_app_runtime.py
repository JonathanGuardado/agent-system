from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.app import (
    DEFAULT_ENV_PATH,
    REQUIRED_ENV_VARS,
    RuntimeConfig,
    StartupConfigError,
    build_runtime,
    load_app_config,
    main,
    run_runtime,
)
from ticket_agent.intake.slack_listener import SlackEvent
from ticket_agent.jira.constants import FIELD_AGENT_ASSIGNED_COMPONENT
from ticket_agent.jira.constants import (
    LABEL_AI_CLAIMED,
    LABEL_AI_EXECUTION_APPROVED,
    LABEL_AI_READY,
    STATUS_IN_REVIEW,
    STATUS_TODO,
)
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket
from ticket_agent.orchestrator.state import TicketState


def test_build_runtime_wires_execution_approval_commands_into_listener(tmp_path):
    slack = _FakeSlack()
    jira_client = FakeJiraClient(
        JiraTicket(
            key="AGENT-123",
            summary="Exercise runtime composition",
            status=STATUS_TODO,
            labels=[LABEL_AI_READY],
        )
    )
    implementation = _Implementation()
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
        ),
        planner=_Planner(),
        implementation=implementation,
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )

    try:
        result = asyncio.run(
            runtime.graph.ainvoke(
                TicketState(
                    ticket_key="AGENT-123",
                    summary="Exercise runtime composition",
                    slack_channel="C-INTAKE",
                    slack_thread_ts="execution-thread",
                ),
                config={"configurable": {"thread_id": "AGENT-123"}},
            )
        )
        assert "__interrupt__" in result

        routed = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve AGENT-123",
                    channel="C-INTAKE",
                    thread_ts="execution-thread",
                )
            )
        )

        assert routed is None
        assert implementation.calls == ["AGENT-123"]
        approval = runtime.approval_store.get("AGENT-123")
        assert approval is not None
        assert approval.status == "approved"
        assert jira_client.ticket("AGENT-123").status == STATUS_IN_REVIEW
    finally:
        runtime.close()


def test_build_runtime_wires_question_answering_without_creating_proposal(tmp_path):
    slack = _FakeSlack()
    model_router = _QuestionRouter(
        "Model runtime response for AGENT-321. No Jira ticket was created."
    )
    jira_client = FakeJiraClient(
        JiraTicket(
            key="AGENT-321",
            summary="Add Slack Q&A mode",
            status=STATUS_TODO,
        )
    )
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
        ),
        planner=_Planner(),
        implementation=_Implementation(),
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
        model_router=model_router,
    )

    try:
        routed = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="how is AGENT-321 going?",
                    channel="C-INTAKE",
                    thread_ts="question-thread",
                )
            )
        )

        assert routed is None
        active = runtime.proposal_store.get_active_for_thread(
            "U1",
            "question-thread",
        )
        assert active is None
        assert jira_client.created_keys == []
        assert slack.messages[-1][3] == (
            "Model runtime response for AGENT-321. No Jira ticket was created."
        )
        assert model_router.calls == ["how is AGENT-321 going?"]
    finally:
        runtime.close()


def test_fake_slack_to_jira_to_execution_pr_path(tmp_path):
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    implementation = _Implementation()
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
            repo_defaults={
                "AGENT": {
                    "repository": "agent-system",
                    "repo_path": "/home/jguardado/repos/agent-system",
                }
            },
            poll_interval_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
        planner=_Planner(),
        implementation=implementation,
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )

    try:
        request = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text=(
                        "Break this AGENT epic into Jira tickets:\n"
                        "- Add login API\n"
                        "- Add login UI"
                    ),
                    channel="C-INTAKE",
                    thread_ts="intake-thread",
                )
            )
        )
        assert request is not None

        approved = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve",
                    channel="C-INTAKE",
                    thread_ts="intake-thread",
                )
            )
        )
        assert approved is not None
        assert approved.write_result is not None
        assert approved.write_result.created_epic_key == "AGENT-1"
        assert approved.write_result.created_ticket_keys == ("AGENT-2", "AGENT-3")

        assert asyncio.run(runtime.detector.poll_once()) == 2
        assert asyncio.run(runtime.worker.run_once()) is True

        pending = runtime.approval_store.get("AGENT-2")
        assert pending is not None
        assert pending.status == "pending"
        assert implementation.calls == []

        routed = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve AGENT-2",
                    channel="C-INTAKE",
                    thread_ts="intake-thread",
                )
            )
        )

        assert routed is None
        assert implementation.calls == ["AGENT-2"]
        assert runtime.approval_store.get("AGENT-2").status == "approved"
        assert jira_client.ticket("AGENT-2").status == STATUS_IN_REVIEW
        assert any(
            "https://github.test/acme/repo/pull/1" in message
            for _channel, _thread, _user, message in slack.messages
        )
    finally:
        runtime.close()


def test_runtime_dry_run_execution_approval_stops_before_implementation(tmp_path):
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    implementation = _Implementation()
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
            repo_defaults={
                "AGENT": {
                    "repository": "agent-system",
                    "repo_path": "/home/jguardado/repos/agent-system",
                }
            },
            poll_interval_seconds=0.01,
            max_backoff_seconds=0.01,
            execution_mode="dry_run",
        ),
        planner=_Planner(),
        implementation=implementation,
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )

    try:
        asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="Add OAuth login to AGENT",
                    channel="C-INTAKE",
                    thread_ts="dry-run-thread",
                )
            )
        )
        approved = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve",
                    channel="C-INTAKE",
                    thread_ts="dry-run-thread",
                )
            )
        )
        assert approved is not None
        ticket_key = approved.write_result.created_ticket_keys[0]

        assert asyncio.run(runtime.detector.poll_once()) == 1
        assert asyncio.run(runtime.worker.run_once()) is True
        ticket = jira_client.ticket(ticket_key)
        assert LABEL_AI_CLAIMED in ticket.labels

        routed = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text=f"approve {ticket_key}",
                    channel="C-INTAKE",
                    thread_ts="dry-run-thread",
                )
            )
        )

        assert routed is None
        assert implementation.calls == []
        assert LABEL_AI_EXECUTION_APPROVED in ticket.labels
        assert LABEL_AI_CLAIMED not in ticket.labels
        assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
        assert runtime.approval_store.get(ticket_key).status == "approved"
        assert jira_client.comments_for(ticket_key) == [
            "AI execution dry-run approved. No code changes were attempted."
        ]
        assert any(
            "Dry-run execution approved" in message
            for _channel, _thread, _user, message in slack.messages
        )
    finally:
        runtime.close()


def test_runtime_dry_run_reject_releases_and_marks_failed(tmp_path):
    slack = _FakeSlack()
    jira_client = FakeJiraClient([])
    runtime = build_runtime(
        jira_client=jira_client,
        slack=slack,
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
            repo_defaults={
                "AGENT": {
                    "repository": "agent-system",
                    "repo_path": "/home/jguardado/repos/agent-system",
                }
            },
            poll_interval_seconds=0.01,
            max_backoff_seconds=0.01,
            execution_mode="dry_run",
        ),
        planner=_Planner(),
        implementation=_Implementation(),
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )

    try:
        asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="Add OAuth login to AGENT",
                    channel="C-INTAKE",
                    thread_ts="dry-run-reject-thread",
                )
            )
        )
        approved = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="approve",
                    channel="C-INTAKE",
                    thread_ts="dry-run-reject-thread",
                )
            )
        )
        assert approved is not None
        ticket_key = approved.write_result.created_ticket_keys[0]
        assert asyncio.run(runtime.detector.poll_once()) == 1
        assert asyncio.run(runtime.worker.run_once()) is True

        routed = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text=f"reject {ticket_key}",
                    channel="C-INTAKE",
                    thread_ts="dry-run-reject-thread",
                )
            )
        )

        ticket = jira_client.ticket(ticket_key)
        assert routed is None
        assert LABEL_AI_CLAIMED not in ticket.labels
        assert "ai-failed" in ticket.labels
        assert ticket.fields[FIELD_AGENT_ASSIGNED_COMPONENT] is None
        assert runtime.approval_store.get(ticket_key).status == "rejected"
        assert jira_client.comments_for(ticket_key) == [
            "AI execution failed:\n\nexecution approval rejected in dry-run mode"
        ]
    finally:
        runtime.close()


def test_build_runtime_derives_repo_defaults_from_single_repo_contract(tmp_path):
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    contract_dir.joinpath("agent-system.yaml").write_text(
        "\n".join(
            [
                "repo:",
                "  name: agent-system",
                f"  root: {tmp_path / 'repo'}",
                "  default_branch: main",
                "language:",
                "  primary: python",
                "  package_manager: setuptools",
                "commands:",
                "  test:",
                "    command: ['python', '-m', 'pytest']",
                "    timeout_seconds: 30",
                "    working_directory: .",
                "  lint: null",
                "  install: null",
                "policy:",
                "  dependency_install_allowed: false",
                "source_dirs: ['src/']",
                "test_dirs: ['tests/']",
            ]
        ),
        encoding="utf-8",
    )
    runtime = build_runtime(
        jira_client=FakeJiraClient([]),
        slack=_FakeSlack(),
        config=RuntimeConfig(
            data_dir=tmp_path / "data",
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
            contract_dir=contract_dir,
            jira_target_projects=("AGENT",),
        ),
        planner=_Planner(),
        implementation=_Implementation(),
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
    )

    try:
        result = asyncio.run(
            runtime.listener.handle_event(
                SlackEvent(
                    user_id="U1",
                    text="Add OAuth login to AGENT",
                    channel="C-INTAKE",
                    thread_ts="contract-thread",
                )
            )
        )

        assert result is not None
        assert result.proposal is not None
        assert result.proposal.tickets[0].repository == "agent-system"
        assert result.proposal.tickets[0].repo_path == str(tmp_path / "repo")
    finally:
        runtime.close()


def test_load_app_config_missing_required_env_fails_fast(tmp_path):
    env_path = tmp_path / "agent-system.env"
    env_path.write_text("", encoding="utf-8")

    with pytest.raises(StartupConfigError) as exc_info:
        load_app_config(env={}, env_path=env_path)

    message = str(exc_info.value)
    assert "missing required environment variables" in message
    assert "SLACK_BOT_TOKEN" in message
    assert "GEMINI_API_KEY" in message


def test_default_env_path_is_repo_dotenv():
    assert str(DEFAULT_ENV_PATH) == ".env"


def test_load_app_config_reads_env_file_and_runtime_options(tmp_path):
    env_path = tmp_path / "agent-system.env"
    data_dir = tmp_path / "runtime-data"
    env_path.write_text(
        "\n".join(
            [
                "SLACK_BOT_TOKEN=xoxb-unit",
                "SLACK_APP_TOKEN=xapp-unit",
                "AGENT_SYSTEM_INTAKE_CHANNEL=C-INTAKE",
                "AGENT_SYSTEM_EXECUTION_APPROVAL_CHANNEL=C-EXEC",
                "JIRA_BASE_URL=https://jira.example.test",
                "JIRA_USER_EMAIL=agent@example.test",
                "JIRA_API_KEY=jira-key",
                "DEEPSEEK_API_KEY=deepseek-key",
                "GEMINI_API_KEY=gemini-key",
                "AGENT_SYSTEM_JIRA_TARGET_PROJECTS=AGENT,OPS",
                "JIRA_FIELD_AGENT_ASSIGNED_COMPONENT=customfield_10001",
                f"AGENT_SYSTEM_DATA_DIR={data_dir}",
                "AGENT_SYSTEM_COMPONENT_ID=runner-1",
                "AGENT_SYSTEM_POLL_INTERVAL_SECONDS=0.25",
                "AGENT_SYSTEM_RECONCILE_INTERVAL_SECONDS=0.5",
                "AGENT_SYSTEM_REPO_CONFIG_PATH=config/test-repos",
                "AGENT_SYSTEM_EXECUTION_MODE=dry_run",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(env={}, env_path=env_path)

    assert config.env_file_loaded is True
    assert config.runtime.data_dir == data_dir
    assert config.runtime.component_id == "runner-1"
    assert config.runtime.intake_channel == "C-INTAKE"
    assert config.runtime.execution_approval_channel == "C-EXEC"
    assert config.runtime.poll_interval_seconds == 0.25
    assert config.runtime.reconcile_interval_seconds == 0.5
    assert str(config.runtime.contract_dir) == "config/test-repos"
    assert config.runtime.jira_target_projects == ("AGENT", "OPS")
    assert config.runtime.execution_mode == "dry_run"
    assert config.jira_target_projects == ("AGENT", "OPS")
    assert config.jira_field_map[FIELD_AGENT_ASSIGNED_COMPONENT] == "customfield_10001"


def test_load_app_config_rejects_unknown_execution_mode(tmp_path):
    env_path = tmp_path / "agent-system.env"
    env_path.write_text("", encoding="utf-8")
    env = {name: f"{name.lower()}-value" for name in REQUIRED_ENV_VARS}
    env["JIRA_BASE_URL"] = "https://jira.example.test"
    env["JIRA_USER_EMAIL"] = "agent@example.test"
    env["AGENT_SYSTEM_EXECUTION_MODE"] = "surprise"

    with pytest.raises(StartupConfigError, match="AGENT_SYSTEM_EXECUTION_MODE"):
        load_app_config(env=env, env_path=env_path)


def test_run_runtime_starts_all_loops_and_shuts_down_cleanly(tmp_path):
    shutdown = asyncio.Event()
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event_name: str, payload: dict[str, Any]) -> None:
        events.append((event_name, dict(payload)))
        started = {
            item["loop"]
            for name, item in events
            if name == "app.loop_started"
        }
        if started == {
            "slack_intake_listener",
            "detection_polling",
            "execution_worker",
            "lock_reconciler",
        }:
            shutdown.set()

    runtime = build_runtime(
        jira_client=FakeJiraClient([]),
        slack=_FakeSlack(),
        config=RuntimeConfig(
            data_dir=tmp_path,
            intake_channel="C-INTAKE",
            execution_approval_channel="C-INTAKE",
            poll_interval_seconds=0.01,
            max_backoff_seconds=0.01,
            reconcile_interval_seconds=0.01,
        ),
        planner=_Planner(),
        implementation=_Implementation(),
        tests=_Tests(),
        review=_Review(),
        pull_request=_PullRequest(),
        escalation=_Escalation(),
        emit=emit,
    )

    async def slack_loop() -> None:
        while True:
            await asyncio.sleep(1)

    try:
        asyncio.run(
            run_runtime(
                runtime,
                slack_loop=slack_loop,
                shutdown_event=shutdown,
                emit=emit,
            )
        )
    finally:
        runtime.close()

    assert ("app.shutdown_complete", {}) in events
    stopped = {
        payload["loop"]
        for name, payload in events
        if name == "app.loop_stopped"
    }
    assert stopped == {
        "slack_intake_listener",
        "detection_polling",
        "execution_worker",
        "lock_reconciler",
    }


def test_main_uses_injected_services_without_live_network(tmp_path):
    env_path = tmp_path / "agent-system.env"
    env_path.write_text("", encoding="utf-8")
    env = {name: f"{name.lower()}-value" for name in REQUIRED_ENV_VARS}
    env["JIRA_BASE_URL"] = "https://jira.example.test"
    env["JIRA_USER_EMAIL"] = "agent@example.test"
    shutdown = asyncio.Event()
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event_name: str, payload: dict[str, Any]) -> None:
        events.append((event_name, dict(payload)))
        if event_name == "app.loop_started" and payload["loop"] == "lock_reconciler":
            shutdown.set()

    async def slack_loop() -> None:
        while True:
            await asyncio.sleep(1)

    exit_code = asyncio.run(
        main(
            env=env,
            env_path=env_path,
            jira_client=FakeJiraClient([]),
            slack=_FakeSlack(),
            config=RuntimeConfig(
                data_dir=tmp_path / "data",
                intake_channel="C-INTAKE",
                execution_approval_channel="C-INTAKE",
                poll_interval_seconds=0.01,
                max_backoff_seconds=0.01,
                reconcile_interval_seconds=0.01,
            ),
            planner=_Planner(),
            implementation=_Implementation(),
            tests=_Tests(),
            review=_Review(),
            pull_request=_PullRequest(),
            escalation=_Escalation(),
            slack_loop=slack_loop,
            shutdown_event=shutdown,
            emit=emit,
            install_signal_handlers=False,
        )
    )

    assert exit_code == 0
    assert any(name == "app.starting" for name, _ in events)
    assert any(name == "app.closed" for name, _ in events)


class _FakeSlack:
    def __init__(self) -> None:
        self.messages: list[tuple[str | None, str, str, str]] = []

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        self.messages.append((channel, thread_ts, user_id, text))


class _Planner:
    async def plan(self, state: TicketState) -> dict[str, Any]:
        return {"summary": "Approve the runtime plan before implementation."}


class _Implementation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def implement(self, state: TicketState) -> dict[str, Any]:
        self.calls.append(state.ticket_key)
        return {"implementation_result": {"status": "implemented"}}


class _Tests:
    async def run_tests(self, state: TicketState) -> dict[str, Any]:
        return {"status": "passed", "tests_passed": True}


class _Review:
    async def review(self, state: TicketState) -> dict[str, Any]:
        return {"status": "approved", "review_passed": True}


class _PullRequest:
    async def open_pull_request(self, state: TicketState) -> str:
        return "https://github.test/acme/repo/pull/1"


class _Escalation:
    async def escalate(self, state: TicketState, reason: str) -> None:
        del state, reason


class _QuestionRouter:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []
        self.capabilities: list[str] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del kwargs
        self.capabilities.append(capability)
        user_message = messages[-1]["content"].splitlines()[0]
        self.calls.append(user_message.removeprefix("user_message: "))
        return self._response

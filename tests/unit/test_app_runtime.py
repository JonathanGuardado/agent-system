from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ticket_agent.app import (
    REQUIRED_ENV_VARS,
    RuntimeConfig,
    StartupConfigError,
    build_runtime,
    load_app_config,
    main,
    run_runtime,
)
from ticket_agent.intake.slack_listener import SlackEvent
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.orchestrator.state import TicketState


def test_build_runtime_wires_execution_approval_commands_into_listener(tmp_path):
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


def test_load_app_config_reads_env_file_and_runtime_options(tmp_path):
    env_path = tmp_path / "agent-system.env"
    data_dir = tmp_path / "runtime-data"
    env_path.write_text(
        "\n".join(
            [
                "SLACK_BOT_TOKEN=xoxb-unit",
                "SLACK_APP_TOKEN=xapp-unit",
                "JIRA_BASE_URL=https://jira.example.test",
                "JIRA_USER_EMAIL=agent@example.test",
                "JIRA_API_KEY=jira-key",
                "DEEPSEEK_API_KEY=deepseek-key",
                "GEMINI_API_KEY=gemini-key",
                f"AGENT_SYSTEM_DATA_DIR={data_dir}",
                "AGENT_SYSTEM_COMPONENT_ID=runner-1",
                "AGENT_SYSTEM_POLL_INTERVAL_SECONDS=0.25",
                "AGENT_SYSTEM_RECONCILE_INTERVAL_SECONDS=0.5",
                "AGENT_SYSTEM_REPO_CONFIG_PATH=config/test-repos",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(env={}, env_path=env_path)

    assert config.env_file_loaded is True
    assert config.runtime.data_dir == data_dir
    assert config.runtime.component_id == "runner-1"
    assert config.runtime.poll_interval_seconds == 0.25
    assert config.runtime.reconcile_interval_seconds == 0.5
    assert str(config.runtime.contract_dir) == "config/test-repos"


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

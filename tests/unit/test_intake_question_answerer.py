from __future__ import annotations

import asyncio
from typing import Any

from ticket_agent.domain.model import ModelResponse
from ticket_agent.intake.question_answerer import (
    JiraQuestionAnswerHandler,
    is_question_text,
)
from ticket_agent.jira.fake_client import FakeJiraClient
from ticket_agent.jira.models import JiraTicket


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


def test_question_answerer_reads_explicit_ticket_key():
    slack = _FakeSlack()
    router = _Router("Model says AGENT-123 is in progress. No Jira ticket was created.")
    jira_client = FakeJiraClient(
        JiraTicket(
            key="AGENT-123",
            summary="Add Slack Q&A",
            status="In Progress",
        )
    )
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=slack,
        model_router=router,
    )

    result = asyncio.run(
        handler.handle_message(
            text="how is AGENT-123 going?",
            channel="C-INTAKE",
            thread_ts="t1",
            user_id="U1",
        )
    )

    assert result.answered is True
    assert result.message == "Model says AGENT-123 is in progress. No Jira ticket was created."
    assert slack.messages == [("C-INTAKE", "t1", "U1", result.message)]
    assert jira_client.calls[0] == ("get_ticket", "AGENT-123", None)
    assert router.calls[0]["capability"] == "trivial.respond"
    assert '"kind": "jira_ticket"' in router.calls[0]["messages"][1]["content"]
    assert '"key": "AGENT-123"' in router.calls[0]["messages"][1]["content"]


def test_question_answerer_searches_jira_for_ticket_question():
    router = _Router("Model found matching OAuth tickets. No Jira ticket was created.")
    jira_client = FakeJiraClient(
        [
            JiraTicket(key="AGENT-1", summary="Add OAuth login", status="To Do"),
            JiraTicket(key="AGENT-2", summary="Document OAuth setup", status="Done"),
        ]
    )
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=_FakeSlack(),
        model_router=router,
    )

    result = asyncio.run(handler.answer("is there a ticket for OAuth login?"))

    assert result.message == "Model found matching OAuth tickets. No Jira ticket was created."
    search_call = jira_client.calls[0]
    assert search_call[0] == "search_issues"
    assert search_call[2]["jql"] == 'text ~ "OAuth login" ORDER BY updated DESC'
    assert '"kind": "jira_search"' in router.calls[0]["messages"][1]["content"]
    assert '"summary": "Add OAuth login"' in router.calls[0]["messages"][1]["content"]


def test_question_answerer_replies_to_dm_style_greeting_without_jira():
    jira_client = FakeJiraClient([])
    router = _Router("Model says hello from the assistant. No Jira ticket was created.")
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=_FakeSlack(),
        model_router=router,
    )

    result = asyncio.run(handler.answer("hi are you able to reply"))

    assert result.message == "Model says hello from the assistant. No Jira ticket was created."
    assert jira_client.calls == []
    assert router.calls[0]["capability"] == "trivial.respond"
    assert '"kind": "general_dm"' in router.calls[0]["messages"][1]["content"]


def test_question_answerer_replies_to_punctuated_dm_presence_check_without_jira():
    jira_client = FakeJiraClient([])
    router = _Router("Model says yes, I am here. No Jira ticket was created.")
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=_FakeSlack(),
        model_router=router,
    )

    result = asyncio.run(handler.answer("hi, are you there?"))

    assert result.message == "Model says yes, I am here. No Jira ticket was created."
    assert jira_client.calls == []
    assert router.calls[0]["capability"] == "trivial.respond"
    assert '"kind": "general_dm"' in router.calls[0]["messages"][1]["content"]


def test_question_answerer_redirects_dm_task_request_without_jira():
    jira_client = FakeJiraClient([])
    router = _Router("Model redirects the task to the intake channel. No Jira ticket was created.")
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=_FakeSlack(),
        model_router=router,
    )

    result = asyncio.run(handler.answer("please fix OAuth login in AGENT"))

    assert result.message == "Model redirects the task to the intake channel. No Jira ticket was created."
    assert jira_client.calls == []
    assert '"kind": "general_dm"' in router.calls[0]["messages"][1]["content"]


def test_question_answerer_routes_basic_help_question_without_jira_search():
    jira_client = FakeJiraClient([])
    router = _Router("Model explains what it can do. No Jira ticket was created.")
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=_FakeSlack(),
        model_router=router,
    )

    result = asyncio.run(handler.answer("what can you do?"))

    assert result.message == "Model explains what it can do. No Jira ticket was created."
    assert jira_client.calls == []
    assert router.calls[0]["capability"] == "trivial.respond"
    assert '"kind": "general_dm"' in router.calls[0]["messages"][1]["content"]


def test_question_answerer_returns_model_metadata():
    jira_client = FakeJiraClient([])
    router = _Router(
        ModelResponse(
            content="Model says yes. No Jira ticket was created.",
            model="gpt-4.1-mini",
            provider="openai",
            capability="trivial.respond",
            fallback_used=True,
        )
    )
    handler = JiraQuestionAnswerHandler(
        jira_client=jira_client,
        slack=_FakeSlack(),
        model_router=router,
    )

    result = asyncio.run(handler.answer("hi, are you there?"))

    assert result.message == "Model says yes. No Jira ticket was created."
    assert result.model == "gpt-4.1-mini"
    assert result.provider == "openai"
    assert result.capability == "trivial.respond"
    assert result.fallback_used is True


def test_question_answerer_has_safe_fallback_when_model_router_is_unavailable():
    jira_client = FakeJiraClient(
        JiraTicket(
            key="AGENT-123",
            summary="Add Slack Q&A",
            status="In Progress",
        )
    )
    handler = JiraQuestionAnswerHandler(jira_client=jira_client, slack=_FakeSlack())

    result = asyncio.run(handler.answer("how is AGENT-123 going?"))

    assert "`AGENT-123` is `In Progress`: Add Slack Q&A" in result.message
    assert "No Jira ticket was created" in result.message


def test_question_classifier_keeps_action_requests_in_intake_flow():
    assert is_question_text("is there a ticket for OAuth login?") is True
    assert is_question_text("how is AGENT-123 going?") is True
    assert is_question_text("ask: what is blocking OAuth?") is True
    assert is_question_text("hello") is True
    assert is_question_text("hi are you able to reply") is True
    assert is_question_text("what can you do?") is True
    assert is_question_text("can you implement OAuth login?") is False
    assert is_question_text("please fix OAuth login") is False


class _Router:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self,
        capability: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> object:
        self.calls.append(
            {
                "capability": capability,
                "messages": messages,
                "kwargs": dict(kwargs),
            }
        )
        return self.response

"""Slack Q&A path for non-ticket intake questions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ticket_agent.intake.approval_flow import SlackPoster
from ticket_agent.jira.models import JiraTicket


class JiraQuestionClient(Protocol):
    """Jira boundary used by Slack Q&A."""

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        """Return one Jira ticket by key."""

    async def search_issues(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> Sequence[JiraTicket | Mapping[str, Any]] | Mapping[str, Any]:
        """Return Jira issues matching a JQL query."""


class ModelRouterProtocol(Protocol):
    """Model router boundary used to compose assistant replies."""

    async def invoke(
        self,
        capability: str,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> object:
        """Return a model response for ``messages``."""


@dataclass(frozen=True)
class QuestionAnswerResult:
    """Outcome of answering one Slack question."""

    answered: bool
    message: str
    model: str | None = None
    provider: str | None = None
    capability: str | None = None
    fallback_used: bool | None = None


@dataclass(frozen=True)
class QuestionConversationMessage:
    """One ephemeral prior message in a Slack Q&A conversation."""

    role: str
    content: str


@dataclass(frozen=True)
class _ModelReply:
    message: str
    model: str | None = None
    provider: str | None = None
    capability: str | None = None
    fallback_used: bool | None = None


class InMemoryQuestionConversationStore:
    """Bounded, process-local memory for Slack Q&A follow-up context."""

    def __init__(self, *, max_messages: int = 12, max_chars: int = 6000) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_messages = max_messages
        self._max_chars = max_chars
        self._messages: dict[str, list[QuestionConversationMessage]] = {}

    def get(self, key: str | None) -> tuple[QuestionConversationMessage, ...]:
        if key is None:
            return ()
        return tuple(self._messages.get(key, ()))

    def append(self, key: str | None, *, user: str, assistant: str) -> None:
        if key is None:
            return
        user = user.strip()
        assistant = assistant.strip()
        if not user or not assistant:
            return
        messages = [
            *self._messages.get(key, ()),
            QuestionConversationMessage(role="user", content=user),
            QuestionConversationMessage(role="assistant", content=assistant),
        ]
        self._messages[key] = self._trim(messages)

    def _trim(
        self,
        messages: list[QuestionConversationMessage],
    ) -> list[QuestionConversationMessage]:
        trimmed = messages[-self._max_messages :]
        while _conversation_chars(trimmed) > self._max_chars and len(trimmed) > 2:
            trimmed = trimmed[1:]
        return trimmed


class JiraQuestionAnswerHandler:
    """Answer Slack questions without creating Jira proposals or tickets."""

    def __init__(
        self,
        *,
        jira_client: JiraQuestionClient,
        slack: SlackPoster,
        model_router: ModelRouterProtocol | None = None,
        conversation_store: InMemoryQuestionConversationStore | None = None,
        max_results: int = 5,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self._jira_client = jira_client
        self._slack = slack
        self._model_router = model_router
        self._conversation_store = (
            conversation_store or InMemoryQuestionConversationStore()
        )
        self._max_results = max_results

    def matches(self, text: str) -> bool:
        """Return true when ``text`` looks like a non-ticket question."""

        return is_question_text(text)

    async def handle_message(
        self,
        *,
        text: str,
        channel: str | None = None,
        thread_ts: str | None = None,
        user_id: str | None = None,
    ) -> QuestionAnswerResult:
        """Post an answer to Slack and return it for tests/telemetry."""

        conversation_key = _conversation_key(
            channel=channel,
            thread_ts=thread_ts,
            user_id=user_id,
        )
        answer = await self.answer(text, conversation_key=conversation_key)
        if thread_ts is not None and user_id is not None:
            await self._slack.post_thread_reply(
                channel,
                thread_ts,
                user_id,
                answer.message,
            )
        return answer

    async def answer(
        self,
        text: str,
        *,
        conversation_key: str | None = None,
    ) -> QuestionAnswerResult:
        """Answer one Jira-oriented question."""

        question = _strip_question_prefix(text)
        history = self._conversation_store.get(conversation_key)
        context = await self._answer_context(question, has_history=bool(history))
        reply = await self._model_reply(question, context, history)
        result = QuestionAnswerResult(
            answered=True,
            message=reply.message,
            model=reply.model,
            provider=reply.provider,
            capability=reply.capability,
            fallback_used=reply.fallback_used,
        )
        self._conversation_store.append(
            conversation_key,
            user=question,
            assistant=result.message,
        )
        return result

    async def _answer_context(
        self,
        question: str,
        *,
        has_history: bool = False,
    ) -> dict[str, object]:
        ticket_key = _extract_ticket_key(question)
        if ticket_key is not None:
            try:
                ticket = await self._jira_client.get_ticket(ticket_key)
            except Exception as exc:
                return {
                    "kind": "jira_ticket_error",
                    "ticket_key": ticket_key,
                    "error": _error_text(exc),
                }
            return {
                "kind": "jira_ticket",
                "ticket": _ticket_payload(ticket),
            }

        if _is_general_dm_question(question) or (
            has_history and _is_contextual_follow_up(question)
        ):
            return {"kind": "general_dm"}

        terms = _search_terms(question)
        if (
            not terms
            or _is_conversational_search_terms(terms)
            or not is_question_text(question)
        ):
            return {"kind": "general_dm"}

        jql = _search_jql(question, terms)
        try:
            result = await self._jira_client.search_issues(
                jql,
                fields=("summary", "status", "labels", "assignee", "updated"),
            )
        except Exception as exc:
            return {
                "kind": "jira_search_error",
                "terms": terms,
                "jql": jql,
                "error": _error_text(exc),
            }

        tickets = _coerce_tickets(result)[: self._max_results]
        return {
            "kind": "jira_search",
            "terms": terms,
            "jql": jql,
            "tickets": [_ticket_payload(ticket) for ticket in tickets],
        }

    async def _model_reply(
        self,
        question: str,
        context: Mapping[str, object],
        history: Sequence[QuestionConversationMessage],
    ) -> _ModelReply:
        if self._model_router is None:
            return _ModelReply(message=_fallback_reply(context))

        try:
            response = await self._model_router.invoke(
                "trivial.respond",
                _reply_messages(question, context, history),
                ticket_id=_context_ticket_id(context),
                metadata={"workflow_node": "intake_question_answer"},
            )
            content = _model_content(response)
        except Exception:
            return _ModelReply(message=_fallback_reply(context))

        return _reply_from_model_response(response, content or _fallback_reply(context))


def is_question_text(text: str) -> bool:
    """Return true for Slack text that should be answered, not proposed."""

    normalized = " ".join(_strip_question_prefix(text).lower().split())
    if not normalized:
        return False
    if _has_question_prefix(text):
        return True
    if _is_general_dm_question(normalized):
        return True
    if normalized.startswith(_ACTION_STARTS):
        return False
    return normalized.endswith("?") or normalized.startswith(_QUESTION_STARTS)


def _strip_question_prefix(text: str) -> str:
    stripped = text.strip()
    return _QUESTION_PREFIX_PATTERN.sub("", stripped, count=1).strip()


def _has_question_prefix(text: str) -> bool:
    return _QUESTION_PREFIX_PATTERN.match(text.strip()) is not None


def _extract_ticket_key(text: str) -> str | None:
    match = _TICKET_KEY_PATTERN.search(text)
    return match.group(1).upper() if match is not None else None


def _search_terms(text: str) -> str:
    cleaned = _TICKET_KEY_PATTERN.sub(" ", text)
    cleaned = _QUESTION_PREFIX_PATTERN.sub(" ", cleaned)
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cleaned)
        if word.lower() not in _STOP_WORDS and not word.isdigit()
    ]
    return " ".join(words[:8]).strip()


def _conversation_key(
    *,
    channel: str | None,
    thread_ts: str | None,
    user_id: str | None,
) -> str | None:
    if channel is not None and channel.startswith("D") and user_id:
        return f"dm:{channel}:{user_id}"
    if channel is not None and thread_ts:
        return f"thread:{channel}:{thread_ts}"
    return None


def _conversation_chars(messages: Sequence[QuestionConversationMessage]) -> int:
    return sum(len(message.content) for message in messages)


def _is_general_dm_question(text: str) -> bool:
    normalized = " ".join(_strip_question_prefix(text).lower().split())
    normalized = normalized.strip(".,!?;: ")
    if not normalized:
        return False
    if normalized in _GREETING_TEXTS:
        return True
    if normalized in _GENERAL_DM_TEXTS:
        return True
    if any(phrase in normalized for phrase in _CAPABILITY_PHRASES):
        return True
    return normalized.startswith(_GENERAL_DM_STARTS)


def _is_contextual_follow_up(text: str) -> bool:
    normalized = " ".join(_strip_question_prefix(text).lower().split())
    normalized = normalized.strip(".,!?;: ")
    if not normalized or normalized.startswith(_ACTION_STARTS):
        return False
    if _extract_ticket_key(normalized) is not None:
        return False
    if any(phrase in normalized for phrase in _FOLLOW_UP_PHRASES):
        return True
    words = set(re.findall(r"[a-z0-9][a-z0-9_.-]*", normalized))
    return bool(words & _FOLLOW_UP_WORDS)


def _is_conversational_search_terms(terms: str) -> bool:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", terms.lower())
    return bool(words) and all(word in _CONVERSATIONAL_TERMS for word in words)


def _search_jql(text: str, terms: str) -> str:
    project_key = _extract_project_key(text)
    clauses: list[str] = []
    if project_key is not None:
        clauses.append(f"project = {project_key}")
    clauses.append(f'text ~ "{_escape_jql_string(terms)}"')
    return " AND ".join(clauses) + " ORDER BY updated DESC"


def _extract_project_key(text: str) -> str | None:
    for match in _PROJECT_KEY_PATTERN.finditer(text):
        candidate = match.group(1).upper()
        if candidate.lower() not in _PROJECT_STOP_WORDS:
            return candidate
    return None


def _escape_jql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _coerce_tickets(
    result: Sequence[JiraTicket | Mapping[str, Any]] | Mapping[str, Any],
) -> list[JiraTicket]:
    if isinstance(result, Mapping):
        issues = result.get("issues")
        if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes)):
            return [_normalize_ticket(issue) for issue in issues]
        return [_normalize_ticket(result)]
    return [_normalize_ticket(issue) for issue in result]


def _normalize_ticket(issue: JiraTicket | Mapping[str, Any]) -> JiraTicket:
    if isinstance(issue, JiraTicket):
        return issue

    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        fields = {}

    status = fields.get("status")
    status_name = ""
    if isinstance(status, Mapping):
        status_name = str(status.get("name") or "")
    elif status is not None:
        status_name = str(status)

    labels = fields.get("labels")
    label_values = (
        [str(label) for label in labels]
        if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes))
        else []
    )

    return JiraTicket(
        key=str(issue.get("key") or ""),
        summary=str(fields.get("summary") or ""),
        status=status_name,
        labels=label_values,
    )


def _ticket_payload(ticket: JiraTicket) -> dict[str, object]:
    return {
        "key": ticket.key,
        "summary": ticket.summary,
        "status": ticket.status,
        "labels": list(ticket.labels),
        "assignee": ticket.assignee,
        "fields": dict(ticket.fields),
    }


def _reply_messages(
    question: str,
    context: Mapping[str, object],
    history: Sequence[QuestionConversationMessage] = (),
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are Agentic System, a Slack assistant for Jira-backed "
                "software work. Reply conversationally and concisely. Use the "
                "provided Jira context when present; treat it as fresher than "
                "conversation history. Use prior conversation only to resolve "
                "follow-up references. Do not invent ticket status, ticket keys, "
                "work progress, PRs, blockers, or created tickets. You cannot "
                "create Jira tickets from direct messages. "
                "If the user asks you to implement, fix, create, or change "
                "work, tell them to post the task in the configured intake "
                "channel for proposal review. Always make clear that no Jira "
                "ticket was created by this answer."
            ),
        },
        *(_history_message(message) for message in history),
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"user_message: {question}",
                    "jira_context_json:",
                    json.dumps(context, sort_keys=True),
                ]
            ),
        },
    ]


def _history_message(message: QuestionConversationMessage) -> dict[str, str]:
    role = "assistant" if message.role == "assistant" else "user"
    return {"role": role, "content": message.content}


def _model_content(response: object) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, Mapping):
        for key in ("content", "text", "message"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    return ""


def _reply_from_model_response(response: object, message: str) -> _ModelReply:
    return _ModelReply(
        message=message,
        model=_response_string(response, "model"),
        provider=_response_string(response, "provider"),
        capability=_response_string(response, "capability"),
        fallback_used=_response_bool(response, "fallback_used"),
    )


def _response_string(response: object, key: str) -> str | None:
    value = response.get(key) if isinstance(response, Mapping) else getattr(response, key, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _response_bool(response: object, key: str) -> bool | None:
    value = response.get(key) if isinstance(response, Mapping) else getattr(response, key, None)
    return value if isinstance(value, bool) else None


def _context_ticket_id(context: Mapping[str, object]) -> str | None:
    ticket = context.get("ticket")
    if isinstance(ticket, Mapping):
        key = ticket.get("key")
        if isinstance(key, str) and key.strip():
            return key.strip()
    ticket_key = context.get("ticket_key")
    if isinstance(ticket_key, str) and ticket_key.strip():
        return ticket_key.strip()
    return None


def _fallback_reply(context: Mapping[str, object]) -> str:
    kind = context.get("kind")
    if kind == "jira_ticket":
        ticket = context.get("ticket")
        if isinstance(ticket, Mapping):
            key = str(ticket.get("key") or "ticket")
            status = str(ticket.get("status") or "unknown status")
            summary = str(ticket.get("summary") or "(no summary)")
            return f"`{key}` is `{status}`: {summary}. No Jira ticket was created."
    if kind == "jira_search":
        tickets = context.get("tickets")
        if isinstance(tickets, Sequence) and not isinstance(tickets, (str, bytes)):
            normalized = [
                ticket
                for ticket in tickets
                if isinstance(ticket, Mapping)
            ]
            if normalized:
                lines = [
                    "I found matching Jira ticket(s). No Jira ticket was created.",
                    *(
                        _format_ticket(
                            JiraTicket(
                                key=str(ticket.get("key") or ""),
                                summary=str(ticket.get("summary") or ""),
                                status=str(ticket.get("status") or ""),
                            )
                        )
                        for ticket in normalized
                    ),
                ]
                return "\n".join(lines)
        terms = str(context.get("terms") or "that")
        return f"I did not find a Jira ticket matching `{terms}`. No Jira ticket was created."
    if kind == "jira_ticket_error":
        key = str(context.get("ticket_key") or "that ticket")
        error = str(context.get("error") or "unknown error")
        return f"I could not read `{key}` from Jira: {error}. No Jira ticket was created."
    if kind == "jira_search_error":
        terms = str(context.get("terms") or "that")
        error = str(context.get("error") or "unknown error")
        return f"I could not search Jira for `{terms}`: {error}. No Jira ticket was created."
    return (
        "I can answer Jira questions here. For task requests, post in the "
        "intake channel for proposal review. No Jira ticket was created."
    )


def _format_ticket(ticket: JiraTicket) -> str:
    status = ticket.status or "unknown status"
    summary = ticket.summary or "(no summary)"
    return f"- `{ticket.key}` - {summary} ({status})"


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


_QUESTION_PREFIX_PATTERN = re.compile(r"^(?:ask|question|q|status|find)\s*:\s*", re.I)
_TICKET_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d+)\b", re.I)
_PROJECT_KEY_PATTERN = re.compile(r"\b(?:project\s+)?([A-Z][A-Z0-9]{1,9})(?:-\d+)?\b")
_QUESTION_STARTS = (
    "are there ",
    "can you check ",
    "do we have ",
    "find ",
    "how is ",
    "how's ",
    "is there ",
    "status of ",
    "what is ",
    "what's ",
    "where is ",
)
_GREETING_TEXTS = {
    "hello",
    "hey",
    "hi",
    "hi there",
}
_CAPABILITY_PHRASES = (
    "able to reply",
    "are you there",
    "are you working",
    "can you reply",
    "do you work",
)
_GENERAL_DM_TEXTS = (
    "can you help",
    "help",
    "help me",
)
_GENERAL_DM_STARTS = (
    "what can you do",
    "what do you do",
    "who are you",
)
_CONVERSATIONAL_TERMS = {
    "able",
    "bot",
    "hello",
    "help",
    "hey",
    "hi",
    "reply",
    "there",
    "you",
}
_FOLLOW_UP_PHRASES = (
    "for that",
    "for it",
    "intake channel",
    "tag you",
)
_FOLLOW_UP_WORDS = {
    "that",
    "this",
    "it",
}
_ACTION_STARTS = (
    "add ",
    "build ",
    "can you add ",
    "can you build ",
    "can you change ",
    "can you create ",
    "can you fix ",
    "can you implement ",
    "can you make ",
    "can you refactor ",
    "can you update ",
    "change ",
    "create ",
    "fix ",
    "implement ",
    "make ",
    "please add ",
    "please build ",
    "please create ",
    "please fix ",
    "please implement ",
    "refactor ",
    "update ",
)
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "check",
    "do",
    "does",
    "feature",
    "find",
    "for",
    "going",
    "have",
    "hi",
    "how",
    "is",
    "issue",
    "it",
    "me",
    "please",
    "q",
    "question",
    "status",
    "the",
    "there",
    "this",
    "ticket",
    "we",
    "what",
    "where",
    "you",
}
_PROJECT_STOP_WORDS = _STOP_WORDS | {
    "api",
    "cli",
    "json",
    "oauth",
    "saas",
    "saml",
    "sso",
    "ui",
    "yaml",
}


__all__ = [
    "InMemoryQuestionConversationStore",
    "JiraQuestionAnswerHandler",
    "QuestionConversationMessage",
    "QuestionAnswerResult",
    "is_question_text",
]

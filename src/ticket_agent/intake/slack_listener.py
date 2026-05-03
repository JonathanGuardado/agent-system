"""Slack Socket Mode entry point for the intake pipeline.

The listener is split into two layers:

* :class:`SlackIntakeListener` — pure routing: takes a parsed
  :class:`SlackEvent`, decides whether the message is a new request or a
  reply to an active proposal, and dispatches into :class:`ApprovalFlow`.
  This layer is fully testable without the Slack SDK.
* :func:`run_socket_mode` — thin runtime wrapper that imports
  ``slack_sdk`` lazily and forwards events into the listener.

Tests should construct the listener with a fake :class:`SlackClient` and a
fake :class:`SlackPoster` (typically the :class:`ApprovalFlow`'s slack
adapter) and feed messages directly via :meth:`handle_event`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ticket_agent.intake.approval_flow import (
    ApprovalFlow,
    ApprovalResult,
    SlackPoster,
)
from ticket_agent.intake.proposal_store import ProposalStore


@dataclass(frozen=True)
class SlackEvent:
    """A normalized Slack event passed into the listener."""

    user_id: str | None
    text: str
    channel: str | None
    thread_ts: str
    is_bot: bool = False


class SlackClient(Protocol):
    """Minimal Slack boundary used by the listener and approval flow."""

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None: ...


class SlackIntakeListener:
    """Route Slack messages to :class:`ApprovalFlow`.

    The listener treats a message as a *reply* if the thread already has an
    active proposal in the store (drafting or awaiting confirmation),
    otherwise it treats the message as a new request.
    """

    def __init__(
        self,
        *,
        approval_flow: ApprovalFlow,
        store: ProposalStore,
        intake_channel: str | None = None,
        emit: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._approval_flow = approval_flow
        self._store = store
        self._intake_channel = intake_channel
        self._emit = emit

    async def handle_event(self, event: SlackEvent) -> ApprovalResult | None:
        """Process one Slack event. Returns ``None`` for ignored messages."""

        if event.is_bot:
            self._emit_event("intake.slack_ignored", {"reason": "bot_message"})
            return None
        if event.user_id is None or not event.user_id.strip():
            self._emit_event("intake.slack_ignored", {"reason": "missing_user_id"})
            return None
        if not event.text.strip():
            self._emit_event("intake.slack_ignored", {"reason": "empty_text"})
            return None
        if (
            self._intake_channel is not None
            and event.channel is not None
            and event.channel != self._intake_channel
        ):
            self._emit_event(
                "intake.slack_ignored",
                {"reason": "wrong_channel", "channel": event.channel},
            )
            return None

        active = self._store.get_active_for_thread(event.user_id, event.thread_ts)
        if active is not None:
            return await self._approval_flow.handle_reply(
                user_id=event.user_id,
                thread_ts=event.thread_ts,
                text=event.text,
                channel=event.channel,
            )
        return await self._approval_flow.handle_new_request(
            user_id=event.user_id,
            thread_ts=event.thread_ts,
            text=event.text,
            channel=event.channel,
        )

    def _emit_event(self, name: str, payload: dict[str, object]) -> None:
        if self._emit is None:
            return
        self._emit(name, payload)


def event_from_slack_payload(payload: dict[str, Any]) -> SlackEvent:
    """Translate a raw Slack message event payload into :class:`SlackEvent`."""

    user_id = payload.get("user")
    text = str(payload.get("text") or "")
    channel = payload.get("channel")
    thread_ts = payload.get("thread_ts") or payload.get("ts") or ""
    bot_id = payload.get("bot_id")
    subtype = payload.get("subtype")
    is_bot = bool(bot_id) or subtype == "bot_message"

    return SlackEvent(
        user_id=str(user_id) if user_id else None,
        text=text,
        channel=str(channel) if channel else None,
        thread_ts=str(thread_ts),
        is_bot=is_bot,
    )


class SlackSDKPoster:
    """Adapter that fulfills :class:`SlackPoster` via slack_sdk's WebClient."""

    def __init__(self, web_client: Any, *, default_channel: str | None = None) -> None:
        self._client = web_client
        self._default_channel = default_channel

    async def post_thread_reply(
        self,
        channel: str | None,
        thread_ts: str,
        user_id: str,
        text: str,
    ) -> None:
        del user_id  # Slack threads are addressed by channel + thread_ts.
        target = channel or self._default_channel
        if not target:
            raise ValueError("post_thread_reply requires a channel")
        # slack_sdk's WebClient methods are sync; offload off the event loop
        # in real deployments. Tests use a fake poster instead.
        self._client.chat_postMessage(
            channel=target,
            thread_ts=thread_ts,
            text=text,
        )


def load_slack_env() -> tuple[str, str, str]:
    """Read ``SLACK_BOT_TOKEN``, ``SLACK_APP_TOKEN`` and ``INTAKE_CHANNEL``."""

    bot_token = _required_env("SLACK_BOT_TOKEN")
    app_token = _required_env("SLACK_APP_TOKEN")
    intake_channel = _required_env("INTAKE_CHANNEL")
    return bot_token, app_token, intake_channel


def run_socket_mode(
    listener: SlackIntakeListener,
    *,
    bot_token: str | None = None,
    app_token: str | None = None,
) -> None:  # pragma: no cover - requires slack_sdk + network
    """Run the Slack Socket Mode loop, dispatching events into ``listener``.

    Imports ``slack_sdk`` lazily so unit tests don't need the dependency.
    """

    try:
        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse
        from slack_sdk.web import WebClient
    except ImportError as exc:
        raise RuntimeError(
            "slack_sdk is required to run socket mode; install slack_sdk"
        ) from exc

    import asyncio

    bot_token = bot_token or _required_env("SLACK_BOT_TOKEN")
    app_token = app_token or _required_env("SLACK_APP_TOKEN")

    web_client = WebClient(token=bot_token)
    socket_client = SocketModeClient(app_token=app_token, web_client=web_client)

    def _on_request(client: SocketModeClient, req: SocketModeRequest) -> None:
        if req.type != "events_api":
            client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
            return
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        payload = (req.payload or {}).get("event") or {}
        if payload.get("type") != "message":
            return
        event = event_from_slack_payload(payload)
        asyncio.run(listener.handle_event(event))

    socket_client.socket_mode_request_listeners.append(_on_request)
    socket_client.connect()
    try:
        import time as _time

        while True:
            _time.sleep(1)
    finally:
        socket_client.close()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is required")
    return value


__all__ = [
    "SlackClient",
    "SlackEvent",
    "SlackIntakeListener",
    "SlackPoster",
    "SlackSDKPoster",
    "event_from_slack_payload",
    "load_slack_env",
    "run_socket_mode",
]

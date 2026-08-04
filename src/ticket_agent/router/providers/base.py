from __future__ import annotations

from typing import Protocol

from ticket_agent.domain.model import ProviderResponse

#: One chat message: ``{"role": ..., "content": ...}``. Every producer in
#: src/ builds exactly that shape, and every provider serializes it straight
#: to JSON, so the values are strings rather than anything richer.
type ChatMessage = dict[str, str]


class ProviderClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        timeout_s: int,
    ) -> ProviderResponse:
        ...

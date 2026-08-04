from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from ticket_agent.domain.model import ProviderResponse

#: One chat message: ``{"role": ..., "content": ...}``. Every producer in
#: src/ builds exactly that shape, and every provider serializes it straight
#: to JSON, so the values are strings rather than anything richer.
#:
#: A Mapping rather than a dict because nothing downstream mutates a message,
#: and requiring dict specifically is what stopped ModelRouter from satisfying
#: the router protocol, whose messages are Mappings.
type ChatMessage = Mapping[str, str]


class ProviderClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        timeout_s: int,
    ) -> ProviderResponse:
        ...

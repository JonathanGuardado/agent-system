from __future__ import annotations

from typing import Protocol

from ticket_agent.domain.model import ProviderResponse


class ProviderClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        ...

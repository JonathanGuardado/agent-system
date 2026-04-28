from __future__ import annotations

from dataclasses import dataclass

from ticket_agent.domain.model import ProviderResponse


@dataclass(frozen=True, slots=True)
class StaticProviderClient:
    """Tiny provider stub for tests and local wiring experiments."""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None

    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        del model, messages, timeout_s
        return ProviderResponse(
            content=self.content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
        )


@dataclass(frozen=True, slots=True)
class FailingProviderClient:
    """Tiny provider stub that always fails."""

    error: str = "provider failed"

    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        del model, messages, timeout_s
        raise RuntimeError(self.error)

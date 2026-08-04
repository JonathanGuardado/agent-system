"""Domain models for internal model routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelAttempt:
    model: str
    provider: str
    success: bool
    error: str | None = None
    latency_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    model: str
    provider: str
    capability: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    fallback_used: bool = False
    attempts: tuple[ModelAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None


class ModelRouterProtocol(Protocol):
    """The router boundary every model-backed component calls through.

    One definition, in the layer they all already depend on. There were three
    of this name -- in orchestrator.model_services, intake.proposal_generator,
    and intake.question_answerer -- differing in whether the extra arguments
    were `**kwargs: Any`, `**kwargs: object`, or named, and in whether the
    return was `Any` or `object`. They were interchangeable only while nothing
    checked them against each other.

    `ticket_id` and `metadata` are named rather than absorbed into `**kwargs`
    because they are what every caller passes and what the journaling wrapper
    reads.
    """

    async def invoke(
        self,
        capability: str,
        messages: Sequence[Mapping[str, str]],
        ticket_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any: ...

"""Domain models for internal model routing."""

from __future__ import annotations

from dataclasses import dataclass


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

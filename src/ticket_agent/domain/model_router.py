from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ticket_agent.domain.model import ModelAttempt, ModelResponse, ProviderResponse


__all__ = [
    "ModelAttempt",
    "ModelResponse",
    "ProviderResponse",
    "ModelRouterResponse",
    "ModelAttemptFailure",
]


@dataclass(frozen=True, slots=True)
class ModelRouterResponse:
    """Legacy response shape from the old HTTP router client."""

    capability: str
    model_used: str
    provider_used: str
    deployment_used: str
    fallback_used: bool
    content: str
    raw_response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelAttemptFailure:
    """Legacy failed-attempt shape from the old HTTP router client."""

    model: str
    provider: str
    deployment: str
    error: str

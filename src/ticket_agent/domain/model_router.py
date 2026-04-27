from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelRouterResponse:
    capability: str
    model_used: str
    provider_used: str
    deployment_used: str
    fallback_used: bool
    content: str
    raw_response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelAttemptFailure:
    model: str
    provider: str
    deployment: str
    error: str

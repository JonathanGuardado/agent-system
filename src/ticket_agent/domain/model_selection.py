from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    selection_tier: str
    provider: str
    model_name: str
    deployment_name: str
    invocation: str = "openai_chat"


@dataclass(frozen=True, slots=True)
class ModelSelection:
    capability: str
    primary: ModelEndpoint
    fallbacks: tuple[ModelEndpoint, ...]
    intent_confidence: float | None = None
    intent_debug: tuple[str, ...] = ()
    selector_debug: tuple[str, ...] = ()

    @property
    def primary_model(self) -> str:
        return self.primary.selection_tier

    @property
    def primary_deployment(self) -> str:
        return self.primary.deployment_name

    @property
    def fallback_models(self) -> tuple[str, ...]:
        return tuple(fallback.selection_tier for fallback in self.fallbacks)

    @property
    def fallback_deployments(self) -> tuple[str, ...]:
        return tuple(fallback.deployment_name for fallback in self.fallbacks)

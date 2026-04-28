from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from ticket_agent.domain.errors import AllBackendsFailedError
from ticket_agent.domain.model import ModelAttempt, ModelResponse
from ticket_agent.router.providers import ProviderClient


class ModelRouter:
    def __init__(
        self,
        selector: Any | None = None,
        providers: Mapping[str, ProviderClient] | None = None,
        timeout_s: int = 120,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._selector = selector if selector is not None else _load_default_selector()
        self._providers = dict(providers or {})
        self._timeout_s = timeout_s

    async def invoke(
        self,
        capability: str,
        messages: list[dict],
        ticket_id: str | None = None,
        metadata: dict | None = None,
    ) -> ModelResponse:
        del ticket_id, metadata

        decision = _select_for_capability(self._selector, capability)
        selected_models = (decision.primary, *decision.fallbacks)
        attempts: list[ModelAttempt] = []

        for index, selected_model in enumerate(selected_models):
            provider_name = str(selected_model.provider)
            model_name = _execution_model_name(selected_model)
            started = time.monotonic()
            provider = self._providers.get(provider_name)

            if provider is None:
                attempts.append(
                    ModelAttempt(
                        model=model_name,
                        provider=provider_name,
                        success=False,
                        error=f"provider not configured: {provider_name}",
                        latency_ms=_elapsed_ms(started),
                    )
                )
                continue

            try:
                provider_response = await provider.chat(
                    model_name,
                    messages,
                    self._timeout_s,
                )
            except Exception as exc:
                attempts.append(
                    ModelAttempt(
                        model=model_name,
                        provider=provider_name,
                        success=False,
                        error=_error_message(exc),
                        latency_ms=_elapsed_ms(started),
                    )
                )
                continue

            attempts.append(
                ModelAttempt(
                    model=model_name,
                    provider=provider_name,
                    success=True,
                    latency_ms=_elapsed_ms(started),
                )
            )
            return ModelResponse(
                content=provider_response.content,
                model=model_name,
                provider=provider_name,
                capability=decision.capability,
                input_tokens=provider_response.input_tokens,
                output_tokens=provider_response.output_tokens,
                estimated_cost_usd=provider_response.estimated_cost_usd,
                fallback_used=index > 0,
                attempts=tuple(attempts),
            )

        raise AllBackendsFailedError(attempts)


def _load_default_selector() -> Any:
    from ticket_agent.router.selector_config import load_model_selector

    return load_model_selector()


def _select_for_capability(selector: Any, capability: str) -> Any:
    resolver = getattr(selector, "resolver", None)
    deterministic_selector = getattr(selector, "selector", None)
    if resolver is not None and deterministic_selector is not None:
        resolution = resolver.resolve(capability)
        context = _build_request_context(resolution)
        return deterministic_selector.select(context)

    if hasattr(selector, "resolve") and hasattr(selector, "select"):
        resolution = selector.resolve(capability)
        context = _build_request_context(resolution)
        return selector.select(context)

    if hasattr(selector, "resolve_intent") and hasattr(selector, "select"):
        resolution = selector.resolve_intent(capability)
        context = _build_request_context(resolution)
        return selector.select(context)

    if hasattr(selector, "select"):
        return selector.select(capability)

    raise TypeError("selector must expose resolver/select or select(capability)")


def _build_request_context(resolution: Any) -> Any:
    from ai_model_selector import build_request_context

    return build_request_context(resolution)


def _execution_model_name(selected_model: Any) -> str:
    deployment_name = getattr(selected_model, "deployment_name", None)
    if deployment_name:
        return str(deployment_name)

    model_name = getattr(selected_model, "model_name", None)
    if model_name:
        return str(model_name)

    raise ValueError("selected model must include deployment_name or model_name")


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__

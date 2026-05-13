from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from ticket_agent.domain.errors import AllBackendsFailedError
from ticket_agent.domain.model import ModelAttempt, ModelResponse
from ticket_agent.router.providers import ProviderClient

_LOGGER = logging.getLogger(__name__)


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
        decision = _select_for_capability(self._selector, capability)
        selected_models = (decision.primary, *decision.fallbacks)
        attempts: list[ModelAttempt] = []

        for index, selected_model in enumerate(selected_models):
            provider_name = str(selected_model.provider)
            model_name = _execution_model_name(selected_model)
            started = time.monotonic()
            provider = self._providers.get(provider_name)
            event_context = {
                "capability": decision.capability,
                "requested_capability": capability,
                "provider": provider_name,
                "model": model_name,
                "fallback_index": index,
                "fallback_used": index > 0,
                "ticket_id": ticket_id,
                "metadata": metadata or {},
            }
            _log_model_event("model.invoke_attempt_started", event_context)

            if provider is None:
                error = f"provider not configured: {provider_name}"
                attempts.append(
                    ModelAttempt(
                        model=model_name,
                        provider=provider_name,
                        success=False,
                        error=error,
                        latency_ms=_elapsed_ms(started),
                    )
                )
                _log_model_event(
                    "model.invoke_attempt_failed",
                    {**event_context, "error": error, "latency_ms": _elapsed_ms(started)},
                    level=logging.WARNING,
                )
                continue

            try:
                provider_response = await provider.chat(
                    model_name,
                    messages,
                    self._timeout_s,
                )
            except Exception as exc:
                error = _error_message(exc)
                attempts.append(
                    ModelAttempt(
                        model=model_name,
                        provider=provider_name,
                        success=False,
                        error=error,
                        latency_ms=_elapsed_ms(started),
                    )
                )
                _log_model_event(
                    "model.invoke_attempt_failed",
                    {**event_context, "error": error, "latency_ms": _elapsed_ms(started)},
                    level=logging.WARNING,
                )
                continue

            latency_ms = _elapsed_ms(started)
            attempts.append(
                ModelAttempt(
                    model=model_name,
                    provider=provider_name,
                    success=True,
                    latency_ms=latency_ms,
                )
            )
            _log_model_event(
                "model.invoke_attempt_completed",
                {
                    **event_context,
                    "latency_ms": latency_ms,
                    "input_tokens": provider_response.input_tokens,
                    "output_tokens": provider_response.output_tokens,
                    "estimated_cost_usd": provider_response.estimated_cost_usd,
                },
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

        _log_model_event(
            "model.invoke_all_backends_failed",
            {
                "capability": decision.capability,
                "requested_capability": capability,
                "ticket_id": ticket_id,
                "metadata": metadata or {},
                "attempts": [
                    {
                        "provider": attempt.provider,
                        "model": attempt.model,
                        "success": attempt.success,
                        "error": attempt.error,
                        "latency_ms": attempt.latency_ms,
                    }
                    for attempt in attempts
                ],
            },
            level=logging.ERROR,
        )
        raise AllBackendsFailedError(attempts)


def _load_default_selector() -> Any:
    from ticket_agent.router.selector_config import load_model_selector

    return load_model_selector()


def _select_for_capability(selector: Any, capability: str) -> Any:
    if hasattr(selector, "capability_definitions") and hasattr(selector, "select"):
        return selector.select(capability)

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


def _log_model_event(
    event_name: str,
    payload: Mapping[str, Any],
    *,
    level: int = logging.INFO,
) -> None:
    _LOGGER.log(
        level,
        json.dumps({"event": event_name, **_jsonable_mapping(payload)}, sort_keys=True),
    )


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, str | int | float | bool):
            result[str(key)] = value
        elif isinstance(value, Mapping):
            result[str(key)] = _jsonable_mapping(value)
        elif isinstance(value, list | tuple):
            result[str(key)] = [_jsonable_value(item) for item in value]
        else:
            result[str(key)] = str(value)
    return result


def _jsonable_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, list | tuple):
        return [_jsonable_value(item) for item in value]
    return str(value)

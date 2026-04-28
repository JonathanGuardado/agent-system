from __future__ import annotations

import asyncio

import pytest

from ticket_agent.domain.errors import AllBackendsFailedError
from ticket_agent.domain.model import ProviderResponse
from ticket_agent.domain.model_selection import ModelEndpoint, ModelSelection
from ticket_agent.router.model_router import ModelRouter


def test_primary_model_succeeds():
    selector = FakeSelector(_selection())
    provider = RecordingProvider(
        ProviderResponse(
            content="done",
            input_tokens=10,
            output_tokens=4,
            estimated_cost_usd=0.001,
        )
    )

    response = asyncio.run(
        ModelRouter(
            selector=selector,
            providers={"deepseek": provider},
            timeout_s=30,
        ).invoke(
            capability="code.implement",
            messages=[{"role": "user", "content": "implement OAuth login"}],
            ticket_id="ABC-123",
            metadata={"source": "unit-test"},
        )
    )

    assert selector.capabilities == ["code.implement"]
    assert provider.calls == [
        (
            "deepseek-v4-pro",
            [{"role": "user", "content": "implement OAuth login"}],
            30,
        )
    ]
    assert response.content == "done"
    assert response.model == "deepseek-v4-pro"
    assert response.provider == "deepseek"
    assert response.capability == "code.implement"
    assert response.input_tokens == 10
    assert response.output_tokens == 4
    assert response.estimated_cost_usd == 0.001
    assert response.fallback_used is False
    attempts = [
        (attempt.model, attempt.provider, attempt.success, attempt.error)
        for attempt in response.attempts
    ]
    assert attempts == [("deepseek-v4-pro", "deepseek", True, None)]


def test_primary_fails_and_fallback_succeeds():
    response = asyncio.run(
        ModelRouter(
            selector=FakeSelector(_selection()),
            providers={
                "deepseek": RecordingProvider(error="primary boom"),
                "gemini": RecordingProvider(
                    ProviderResponse(content="fallback answer")
                ),
            },
        ).invoke(
            capability="code.implement",
            messages=[{"role": "user", "content": "implement OAuth login"}],
        )
    )

    assert response.content == "fallback answer"
    assert response.model == "gemini-2.5-flash"
    assert response.provider == "gemini"
    assert [attempt.model for attempt in response.attempts] == [
        "deepseek-v4-pro",
        "gemini-2.5-flash",
    ]
    assert [attempt.success for attempt in response.attempts] == [False, True]
    assert response.attempts[0].error == "primary boom"


def test_fallback_used_is_true_when_fallback_succeeds():
    response = asyncio.run(
        ModelRouter(
            selector=FakeSelector(_selection()),
            providers={
                "deepseek": RecordingProvider(error="primary boom"),
                "gemini": RecordingProvider(
                    ProviderResponse(content="fallback answer")
                ),
            },
        ).invoke(
            capability="code.implement",
            messages=[{"role": "user", "content": "implement OAuth login"}],
        )
    )

    assert response.fallback_used is True


def test_all_providers_fail_raises_all_backends_failed():
    with pytest.raises(AllBackendsFailedError) as exc_info:
        asyncio.run(
            ModelRouter(
                selector=FakeSelector(_selection()),
                providers={
                    "deepseek": RecordingProvider(error="primary boom"),
                    "gemini": RecordingProvider(error="gemini boom"),
                    "ollama": RecordingProvider(error="ollama boom"),
                },
            ).invoke(
                capability="code.implement",
                messages=[{"role": "user", "content": "implement OAuth login"}],
            )
        )

    attempts = [
        (attempt.model, attempt.provider, attempt.success, attempt.error)
        for attempt in exc_info.value.attempts
    ]
    assert attempts == [
        ("deepseek-v4-pro", "deepseek", False, "primary boom"),
        ("gemini-2.5-flash", "gemini", False, "gemini boom"),
        ("qwen3.5:9b", "ollama", False, "ollama boom"),
    ]


def test_missing_provider_raises_and_records_failed_attempt():
    with pytest.raises(AllBackendsFailedError) as exc_info:
        asyncio.run(
            ModelRouter(
                selector=FakeSelector(_selection(fallbacks=())),
                providers={},
            ).invoke(
                capability="code.implement",
                messages=[{"role": "user", "content": "implement OAuth login"}],
            )
        )

    attempts = [
        (attempt.model, attempt.provider, attempt.success, attempt.error)
        for attempt in exc_info.value.attempts
    ]
    assert attempts == [
        (
            "deepseek-v4-pro",
            "deepseek",
            False,
            "provider not configured: deepseek",
        ),
    ]


def test_attempts_include_model_provider_success_error_and_latency():
    response = asyncio.run(
        ModelRouter(
            selector=FakeSelector(_selection()),
            providers={
                "deepseek": RecordingProvider(error="primary boom"),
                "gemini": RecordingProvider(
                    ProviderResponse(content="fallback answer")
                ),
            },
        ).invoke(
            capability="code.implement",
            messages=[{"role": "user", "content": "implement OAuth login"}],
        )
    )

    assert response.attempts[0].model == "deepseek-v4-pro"
    assert response.attempts[0].provider == "deepseek"
    assert response.attempts[0].success is False
    assert response.attempts[0].error == "primary boom"
    assert isinstance(response.attempts[0].latency_ms, int)
    assert response.attempts[1].model == "gemini-2.5-flash"
    assert response.attempts[1].provider == "gemini"
    assert response.attempts[1].success is True
    assert response.attempts[1].error is None
    assert isinstance(response.attempts[1].latency_ms, int)


class FakeSelector:
    def __init__(self, selection: ModelSelection) -> None:
        self._selection = selection
        self.capabilities: list[str] = []

    def select(self, capability: str) -> ModelSelection:
        self.capabilities.append(capability)
        return self._selection


class RecordingProvider:
    def __init__(
        self,
        response: ProviderResponse | None = None,
        *,
        error: str | None = None,
    ) -> None:
        self._response = response or ProviderResponse(content="ok")
        self._error = error
        self.calls: list[tuple[str, list[dict], int]] = []

    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        self.calls.append((model, messages, timeout_s))
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._response


def _selection(
    *,
    fallbacks: tuple[ModelEndpoint, ...] | None = None,
) -> ModelSelection:
    return ModelSelection(
        capability="code.implement",
        primary=ModelEndpoint(
            selection_tier="deepseek-v4-pro",
            provider="deepseek",
            model_name="deepseek-v4-pro",
            deployment_name="deepseek-v4-pro",
        ),
        fallbacks=(
            ModelEndpoint(
                selection_tier="gemini-flash",
                provider="gemini",
                model_name="gemini-flash",
                deployment_name="gemini-2.5-flash",
            ),
            ModelEndpoint(
                selection_tier="qwen-local",
                provider="ollama",
                model_name="qwen-local",
                deployment_name="qwen3.5:9b",
            ),
        )
        if fallbacks is None
        else fallbacks,
    )

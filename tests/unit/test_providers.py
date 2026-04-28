from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from ticket_agent.domain.errors import ProviderError
from ticket_agent.router import providers
from ticket_agent.router.providers import DeepSeekProvider, GeminiProvider, OllamaProvider


def test_deepseek_provider_sends_expected_request(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-api-key")
    fake_client = _install_fake_async_client(
        monkeypatch,
        _response(
            {
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        ),
    )
    messages = [{"role": "user", "content": "hello"}]

    response = asyncio.run(
        DeepSeekProvider().chat(
            model="deepseek-v4-pro",
            messages=messages,
            timeout_s=42,
        )
    )

    assert response.content == "done"
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert fake_client.instances[0].timeout == 42
    assert fake_client.calls == [
        {
            "url": "https://api.deepseek.com/v1/chat/completions",
            "headers": {"Authorization": "Bearer unit-test-api-key"},
            "json": {"model": "deepseek-v4-pro", "messages": messages},
        }
    ]


def test_deepseek_provider_parses_content(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-api-key")
    _install_fake_async_client(
        monkeypatch,
        _response({"choices": [{"message": {"content": "deepseek answer"}}]}),
    )

    response = asyncio.run(DeepSeekProvider().chat("deepseek-v4-pro", [], 30))

    assert response.content == "deepseek answer"


def test_deepseek_provider_maps_usage_tokens(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-api-key")
    _install_fake_async_client(
        monkeypatch,
        _response(
            {
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45},
            }
        ),
    )

    response = asyncio.run(DeepSeekProvider().chat("deepseek-v4-pro", [], 30))

    assert response.input_tokens == 123
    assert response.output_tokens == 45
    assert response.estimated_cost_usd is None


def test_deepseek_provider_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="missing DEEPSEEK_API_KEY"):
        asyncio.run(DeepSeekProvider().chat("deepseek-v4-pro", [], 30))


def test_gemini_provider_sends_expected_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-api-key")
    fake_client = _install_fake_async_client(
        monkeypatch,
        _response({"choices": [{"message": {"content": "done"}}]}),
    )
    messages = [{"role": "user", "content": "hello"}]

    response = asyncio.run(
        GeminiProvider().chat(
            model="gemini-2.5-flash",
            messages=messages,
            timeout_s=17,
        )
    )

    assert response.content == "done"
    assert fake_client.instances[0].timeout == 17
    assert fake_client.calls == [
        {
            "url": (
                "https://generativelanguage.googleapis.com/v1beta/openai"
                "/chat/completions"
            ),
            "headers": {"Authorization": "Bearer unit-test-api-key"},
            "json": {"model": "gemini-2.5-flash", "messages": messages},
        }
    ]


def test_gemini_provider_parses_content(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-api-key")
    _install_fake_async_client(
        monkeypatch,
        _response({"choices": [{"message": {"content": "gemini answer"}}]}),
    )

    response = asyncio.run(GeminiProvider().chat("gemini-2.5-flash", [], 30))

    assert response.content == "gemini answer"


def test_gemini_provider_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="missing GEMINI_API_KEY"):
        asyncio.run(GeminiProvider().chat("gemini-2.5-flash", [], 30))


def test_ollama_provider_sends_stream_and_think_false(monkeypatch):
    fake_client = _install_fake_async_client(
        monkeypatch,
        _response({"message": {"content": "local answer"}}),
    )
    messages = [{"role": "user", "content": "hello"}]

    response = asyncio.run(
        OllamaProvider().chat(
            model="qwen3.5:9b",
            messages=messages,
            timeout_s=8,
        )
    )

    assert response.content == "local answer"
    assert fake_client.instances[0].timeout == 8
    assert fake_client.calls == [
        {
            "url": "http://localhost:11434/api/chat",
            "headers": None,
            "json": {
                "model": "qwen3.5:9b",
                "messages": messages,
                "stream": False,
                "think": False,
            },
        }
    ]


def test_ollama_provider_sends_to_api_chat(monkeypatch):
    fake_client = _install_fake_async_client(
        monkeypatch,
        _response({"message": {"content": "local answer"}}),
    )

    asyncio.run(OllamaProvider().chat("qwen3.5:9b", [], 30))

    assert fake_client.calls[0]["url"] == "http://localhost:11434/api/chat"


def test_ollama_provider_parses_content(monkeypatch):
    _install_fake_async_client(
        monkeypatch,
        _response({"message": {"content": "ollama answer"}}),
    )

    response = asyncio.run(OllamaProvider().chat("qwen3.5:9b", [], 30))

    assert response.content == "ollama answer"


@pytest.mark.parametrize(
    "provider",
    (
        DeepSeekProvider(),
        GeminiProvider(),
        OllamaProvider(),
    ),
)
def test_non_2xx_response_raises_provider_error(monkeypatch, provider):
    _set_provider_key(monkeypatch, provider)
    _install_fake_async_client(monkeypatch, _response({"error": "nope"}, status_code=500))

    with pytest.raises(ProviderError, match="HTTP 500"):
        asyncio.run(provider.chat("model", [], 30))


@pytest.mark.parametrize(
    "provider,response_data",
    (
        (DeepSeekProvider(), {"choices": []}),
        (GeminiProvider(), {"choices": [{"message": {}}]}),
        (OllamaProvider(), {"message": {}}),
    ),
)
def test_malformed_response_raises_provider_error(monkeypatch, provider, response_data):
    _set_provider_key(monkeypatch, provider)
    _install_fake_async_client(monkeypatch, _response(response_data))

    with pytest.raises(ProviderError):
        asyncio.run(provider.chat("model", [], 30))


@pytest.mark.parametrize(
    "provider",
    (
        DeepSeekProvider(),
        GeminiProvider(),
        OllamaProvider(),
    ),
)
def test_timeout_raises_provider_error(monkeypatch, provider):
    _set_provider_key(monkeypatch, provider)
    _install_fake_async_client(monkeypatch, httpx.TimeoutException("slow"))

    with pytest.raises(ProviderError, match="timed out"):
        asyncio.run(provider.chat("model", [], 30))


def test_provider_error_messages_do_not_include_raw_api_key(monkeypatch):
    api_key = "unit-test-api-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    _install_fake_async_client(monkeypatch, httpx.ConnectError(api_key))

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(DeepSeekProvider().chat("deepseek-v4-pro", [], 30))

    assert api_key not in str(exc_info.value)


class FakeResponse:
    def __init__(self, data: Any, *, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _response(data: Any, *, status_code: int = 200) -> FakeResponse:
    return FakeResponse(data, status_code=status_code)


def _install_fake_async_client(monkeypatch, result):
    class FakeAsyncClient:
        calls: list[dict[str, Any]] = []
        instances: list["FakeAsyncClient"] = []

        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout
            self.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url: str, *, headers=None, json=None):
            self.calls.append({"url": url, "headers": headers, "json": json})
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(providers.httpx, "AsyncClient", FakeAsyncClient)
    return FakeAsyncClient


def _set_provider_key(monkeypatch, provider) -> None:
    if isinstance(provider, DeepSeekProvider):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-api-key")
    if isinstance(provider, GeminiProvider):
        monkeypatch.setenv("GEMINI_API_KEY", "unit-test-api-key")

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ticket_agent.domain.errors import ProviderError
from ticket_agent.domain.model import ProviderResponse


class ProviderClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        ...


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


@dataclass(frozen=True, slots=True)
class DeepSeekProvider:
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"

    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise ProviderError(f"missing {self.api_key_env}")

        return await _post_openai_chat(
            provider_name="deepseek",
            url=_join_url(self.base_url, "/v1/chat/completions"),
            api_key=api_key,
            model=model,
            messages=messages,
            timeout_s=timeout_s,
        )


@dataclass(frozen=True, slots=True)
class GeminiProvider:
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key_env: str = "GEMINI_API_KEY"

    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise ProviderError(f"missing {self.api_key_env}")

        return await _post_openai_chat(
            provider_name="gemini",
            url=_join_url(self.base_url, "/chat/completions"),
            api_key=api_key,
            model=model,
            messages=messages,
            timeout_s=timeout_s,
        )


@dataclass(frozen=True, slots=True)
class OllamaProvider:
    base_url: str = "http://localhost:11434"

    async def chat(
        self,
        model: str,
        messages: list[dict],
        timeout_s: int,
    ) -> ProviderResponse:
        url = _join_url(self.base_url, "/api/chat")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(_sanitize_error("ollama request failed", exc)) from exc

        _raise_for_status("ollama", response)
        return _parse_ollama_chat_response(_response_json("ollama", response))


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_base: str = ""
    api_key_env: str = ""
    local: bool = False


PROVIDER_DEFAULTS = {
    "ollama": ProviderConfig("ollama", local=True),
    "gemini": ProviderConfig(
        "gemini",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    "deepseek": ProviderConfig(
        "deepseek",
        api_base="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    ),
}


def load_provider(provider_name: str) -> ProviderConfig:
    default = PROVIDER_DEFAULTS.get(provider_name, ProviderConfig(provider_name))
    prefix = provider_name.upper().replace("-", "_")
    return ProviderConfig(
        name=provider_name,
        api_base=os.getenv(f"{prefix}_API_BASE", default.api_base),
        api_key_env=os.getenv(
            f"{prefix}_API_KEY_ENV",
            default.api_key_env or f"{prefix}_API_KEY",
        ),
        local=default.local,
    )


def load_providers() -> dict[str, ProviderConfig]:
    return {name: load_provider(name) for name in PROVIDER_DEFAULTS}


async def _post_openai_chat(
    *,
    provider_name: str,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    timeout_s: int,
) -> ProviderResponse:
    payload = {"model": model, "messages": messages}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                url,
                headers=_auth_header(api_key),
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{provider_name} request timed out") from exc
    except httpx.HTTPError as exc:
        message = _sanitize_error(f"{provider_name} request failed", exc, api_key)
        raise ProviderError(message) from exc

    _raise_for_status(provider_name, response)
    return _parse_openai_chat_response(
        provider_name,
        _response_json(provider_name, response),
    )


def _auth_header(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _parse_openai_chat_response(
    provider_name: str,
    data: dict[str, Any],
) -> ProviderResponse:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{provider_name} response missing message content") from exc

    if not isinstance(content, str):
        raise ProviderError(f"{provider_name} message content must be a string")

    usage = data.get("usage")
    input_tokens = output_tokens = None
    if isinstance(usage, dict):
        input_tokens = _optional_int(usage.get("prompt_tokens"))
        output_tokens = _optional_int(usage.get("completion_tokens"))

    return ProviderResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _parse_ollama_chat_response(data: dict[str, Any]) -> ProviderResponse:
    try:
        content = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ProviderError("ollama response missing message content") from exc

    if not isinstance(content, str):
        raise ProviderError("ollama message content must be a string")
    return ProviderResponse(content=content)


def _response_json(provider_name: str, response: Any) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider_name} response was not valid JSON") from exc

    if not isinstance(data, dict):
        raise ProviderError(f"{provider_name} response JSON root must be an object")
    return data


def _raise_for_status(provider_name: str, response: Any) -> None:
    status_code = int(getattr(response, "status_code", 0))
    if status_code < 200 or status_code >= 300:
        raise ProviderError(f"{provider_name} returned HTTP {status_code}")


def _sanitize_error(
    message: str,
    exc: Exception | None = None,
    *secrets: str,
) -> str:
    detail = f"{message}: {exc.__class__.__name__}" if exc is not None else message
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return detail


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None

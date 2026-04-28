from __future__ import annotations

from typing import Any

import httpx

from ticket_agent.domain.errors import ProviderError
from ticket_agent.domain.model import ProviderResponse


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
        detail = _redact_secret(detail, secret)
    return detail


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _redact_secret(message: str, secret: str) -> str:
    if not secret:
        return message
    return message.replace(secret, "[redacted]")


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None

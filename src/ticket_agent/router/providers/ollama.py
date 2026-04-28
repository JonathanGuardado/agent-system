from __future__ import annotations

from dataclasses import dataclass

import httpx

from ticket_agent.domain.errors import ProviderError
from ticket_agent.domain.model import ProviderResponse
from ticket_agent.router.providers.http import (
    _join_url,
    _parse_ollama_chat_response,
    _raise_for_status,
    _response_json,
    _sanitize_error,
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

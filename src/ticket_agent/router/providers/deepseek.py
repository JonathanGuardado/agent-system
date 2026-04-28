from __future__ import annotations

import os
from dataclasses import dataclass

from ticket_agent.domain.errors import ProviderError
from ticket_agent.domain.model import ProviderResponse
from ticket_agent.router.providers.http import _join_url, _post_openai_chat


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

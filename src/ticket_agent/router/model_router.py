from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ticket_agent.domain.errors import AllModelsFailedError, ModelCallError
from ticket_agent.domain.model_selection import ModelEndpoint
from ticket_agent.domain.model_router import ModelAttemptFailure, ModelRouterResponse
from ticket_agent.router.selector_config import select_model_for_capability


class ModelRouter:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def invoke(
        self,
        capability_or_text: str,
        messages: list[dict[str, str]],
    ) -> ModelRouterResponse:
        selection = select_model_for_capability(capability_or_text)
        endpoints = (selection.primary, *selection.fallbacks)
        failures: list[ModelAttemptFailure] = []

        for index, endpoint in enumerate(endpoints):
            try:
                raw_response = self._post_chat_completion(endpoint, messages)
                content = self._extract_content(endpoint.selection_tier, raw_response)
            except ModelCallError as exc:
                failures.append(
                    ModelAttemptFailure(
                        model=endpoint.selection_tier,
                        provider=endpoint.provider,
                        deployment=endpoint.deployment_name,
                        error=str(exc),
                    )
                )
                continue

            return ModelRouterResponse(
                capability=selection.capability,
                model_used=endpoint.selection_tier,
                provider_used=endpoint.provider,
                deployment_used=endpoint.deployment_name,
                fallback_used=index > 0,
                content=content,
                raw_response=raw_response,
            )

        raise AllModelsFailedError(failures)

    def _post_chat_completion(
        self,
        endpoint: ModelEndpoint,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = {"model": endpoint.deployment_name, "messages": messages}
        headers = {"Content-Type": "application/json"}
        headers["X-Model-Selection-Tier"] = endpoint.selection_tier
        headers["X-Model-Provider"] = endpoint.provider
        headers["X-Model-Invocation"] = endpoint.invocation
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                status = int(getattr(response, "status", 200))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelCallError(endpoint.selection_tier, f"HTTP {exc.code}: {detail}") from exc
        except (OSError, TimeoutError, URLError) as exc:
            raise ModelCallError(endpoint.selection_tier, str(exc)) from exc

        if status >= 400:
            detail = body.decode("utf-8", errors="replace")
            raise ModelCallError(endpoint.selection_tier, f"HTTP {status}: {detail}")

        try:
            raw = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelCallError(endpoint.selection_tier, f"invalid JSON response: {exc}") from exc

        if not isinstance(raw, dict):
            raise ModelCallError(endpoint.selection_tier, "response JSON root must be an object")
        return raw

    def _extract_content(self, model: str, raw_response: dict[str, Any]) -> str:
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelCallError(
                model,
                "response missing choices[0].message.content",
            ) from exc

        if not isinstance(content, str):
            raise ModelCallError(
                model,
                "choices[0].message.content must be a string",
            )
        return content

from __future__ import annotations

from ticket_agent.router.model_router import ModelRouter
from ticket_agent.router.providers import (
    DeepSeekProvider,
    GeminiProvider,
    OllamaProvider,
    ProviderClient,
    load_provider,
)
from ticket_agent.router.selector_config import load_model_selector


def create_model_router(*, timeout_s: int = 120) -> ModelRouter:
    deepseek = load_provider("deepseek")
    gemini = load_provider("gemini")
    ollama = load_provider("ollama")

    providers: dict[str, ProviderClient] = {
        "deepseek": DeepSeekProvider(
            base_url=deepseek.api_base,
            api_key_env=deepseek.api_key_env,
        ),
        "gemini": GeminiProvider(
            base_url=gemini.api_base,
            api_key_env=gemini.api_key_env,
        ),
        "ollama": (
            OllamaProvider(base_url=ollama.api_base)
            if ollama.api_base
            else OllamaProvider()
        ),
    }

    return ModelRouter(
        selector=load_model_selector(),
        providers=providers,
        timeout_s=timeout_s,
    )

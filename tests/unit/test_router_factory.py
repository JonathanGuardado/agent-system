from __future__ import annotations

from ticket_agent.router import factory
from ticket_agent.router.factory import create_model_router
from ticket_agent.router.model_router import ModelRouter
from ticket_agent.router.providers import (
    DeepSeekProvider,
    GeminiProvider,
    OllamaProvider,
    ProviderConfig,
)


def test_create_model_router_wires_selector_and_mvp_providers(monkeypatch):
    selector = object()
    loaded_providers: list[str] = []
    configs = {
        "deepseek": ProviderConfig(
            "deepseek",
            api_base="https://deepseek.test",
            api_key_env="DEEPSEEK_TEST_KEY",
        ),
        "gemini": ProviderConfig(
            "gemini",
            api_base="https://gemini.test/openai",
            api_key_env="GEMINI_TEST_KEY",
        ),
        "ollama": ProviderConfig(
            "ollama",
            api_base="http://ollama.test",
            local=True,
        ),
    }

    def load_provider(provider_name: str) -> ProviderConfig:
        loaded_providers.append(provider_name)
        return configs[provider_name]

    monkeypatch.setattr(factory, "load_model_selector", lambda: selector)
    monkeypatch.setattr(factory, "load_provider", load_provider)

    router = create_model_router(timeout_s=9)

    assert isinstance(router, ModelRouter)
    assert router._selector is selector
    assert router._timeout_s == 9
    assert loaded_providers == ["deepseek", "gemini", "ollama"]

    deepseek = router._providers["deepseek"]
    gemini = router._providers["gemini"]
    ollama = router._providers["ollama"]

    assert isinstance(deepseek, DeepSeekProvider)
    assert deepseek.base_url == "https://deepseek.test"
    assert deepseek.api_key_env == "DEEPSEEK_TEST_KEY"
    assert isinstance(gemini, GeminiProvider)
    assert gemini.base_url == "https://gemini.test/openai"
    assert gemini.api_key_env == "GEMINI_TEST_KEY"
    assert isinstance(ollama, OllamaProvider)
    assert ollama.base_url == "http://ollama.test"


def test_create_model_router_preserves_default_ollama_base_url(monkeypatch):
    monkeypatch.setattr(factory, "load_model_selector", lambda: object())
    monkeypatch.setattr(
        factory,
        "load_provider",
        lambda provider_name: ProviderConfig(provider_name),
    )

    router = create_model_router()

    assert router._providers["ollama"].base_url == "http://localhost:11434"

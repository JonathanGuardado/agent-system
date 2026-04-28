from __future__ import annotations

import os
from dataclasses import dataclass


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


def load_provider_configs() -> dict[str, ProviderConfig]:
    return {name: load_provider(name) for name in PROVIDER_DEFAULTS}


load_providers = load_provider_configs

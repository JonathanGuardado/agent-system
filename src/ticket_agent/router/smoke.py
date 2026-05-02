from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Mapping

from ticket_agent.domain.model_selection import ModelEndpoint, ModelSelection
from ticket_agent.router import create_model_router

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPABILITY = "trivial.respond"
MESSAGES = [
    {"role": "system", "content": "You are a concise test responder."},
    {"role": "user", "content": "Reply with exactly: router smoke ok"},
]

REMOTE_MODELS = {
    "gemini": ModelEndpoint(
        selection_tier="gemini-flash",
        provider="gemini",
        model_name="gemini-flash",
        deployment_name="gemini-2.5-flash",
    ),
    "deepseek": ModelEndpoint(
        selection_tier="deepseek-v4-pro",
        provider="deepseek",
        model_name="deepseek-v4-pro",
        deployment_name="deepseek-v4-pro",
    ),
}


def main(env: Mapping[str, str] | None = None) -> int:
    _load_dotenv_if_available()
    env = os.environ if env is None else env

    try:
        router = create_model_router(timeout_s=60)
    except Exception as exc:
        print(
            "Router construction failed: "
            f"{exc.__class__.__name__}: {_safe_message(exc, env)}"
        )
        return 1

    providers = _available_providers(router)
    has_gemini_key = _has_env(env, "GEMINI_API_KEY")
    has_deepseek_key = _has_env(env, "DEEPSEEK_API_KEY")
    ollama_configured = _ollama_base_url_configured(router)

    print("Model router smoke test")
    print("router created: yes")
    print(f"available providers: {', '.join(providers) if providers else '(none)'}")
    print(f"DEEPSEEK_API_KEY present: {_yes_no(has_deepseek_key)}")
    print(f"GEMINI_API_KEY present: {_yes_no(has_gemini_key)}")
    print(f"Ollama base URL configured: {_yes_no(ollama_configured)}")

    selected_provider = _select_remote_provider(
        has_gemini_key=has_gemini_key,
        has_deepseek_key=has_deepseek_key,
    )
    if selected_provider is None:
        if ollama_configured:
            print("Ollama is configured but not tested by default.")
        print(
            "No remote provider API keys found. Router construction smoke test "
            "passed; skipping real LLM call."
        )
        return 0

    _select_only_remote_provider(router, selected_provider)
    print(f"selected remote provider for smoke call: {selected_provider}")

    try:
        response = asyncio.run(router.invoke(capability=CAPABILITY, messages=MESSAGES))
    except Exception as exc:
        message = (
            f"Provider call failed: {exc.__class__.__name__}: "
            f"{_safe_message(exc, env)}"
        )
        print(
            message,
            file=sys.stderr,
        )
        return 1

    print(f"selected model: {response.model or '(unavailable)'}")
    print(f"provider: {response.provider or '(unavailable)'}")
    print(f"response text: {response.content}")
    print(f"token usage: {_format_token_usage(response)}")
    return 0


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    # Check canonical key store first, then repo-local .env for dev convenience.
    config_env = Path.home() / "config" / "agent-system.env"
    if config_env.exists():
        load_dotenv(config_env)
    load_dotenv(REPO_ROOT / ".env")


def _available_providers(router: object) -> list[str]:
    providers = getattr(router, "_providers", {})
    if not isinstance(providers, dict):
        return []
    return sorted(str(provider) for provider in providers)


def _ollama_base_url_configured(router: object) -> bool:
    providers = getattr(router, "_providers", {})
    if not isinstance(providers, dict):
        return False
    ollama = providers.get("ollama")
    return bool(getattr(ollama, "base_url", ""))


def _select_remote_provider(
    *,
    has_gemini_key: bool,
    has_deepseek_key: bool,
) -> str | None:
    if has_gemini_key:
        return "gemini"
    if has_deepseek_key:
        return "deepseek"
    return None


def _select_only_remote_provider(router: object, provider: str) -> None:
    setattr(router, "_selector", _SingleProviderSelector(provider))


def _has_env(env: Mapping[str, str], key: str) -> bool:
    return bool(env.get(key))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _format_token_usage(response: object) -> str:
    input_tokens = getattr(response, "input_tokens", None)
    output_tokens = getattr(response, "output_tokens", None)
    if input_tokens is None and output_tokens is None:
        return "unavailable"
    input_text = input_tokens if input_tokens is not None else "?"
    output_text = output_tokens if output_tokens is not None else "?"
    return f"input={input_text} output={output_text}"


def _safe_message(exc: Exception, env: Mapping[str, str]) -> str:
    message = str(exc) or exc.__class__.__name__
    for key, value in env.items():
        if "KEY" in key.upper() or "TOKEN" in key.upper() or "SECRET" in key.upper():
            if value:
                message = message.replace(value, "[redacted]")
    return message


class _SingleProviderSelector:
    def __init__(self, provider: str) -> None:
        self._provider = provider

    def select(self, capability: str) -> ModelSelection:
        endpoint = REMOTE_MODELS[self._provider]
        return ModelSelection(
            capability=capability,
            primary=endpoint,
            fallbacks=(),
        )

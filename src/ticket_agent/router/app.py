#!/usr/bin/env python3
"""
OpenAI-compatible model execution router.

This file intentionally does not classify prompts. It asks ai-model-selector for
the primary model endpoint and fallbacks, then executes those endpoints.

Required env:
  ROUTER_PORT=8080
  OLLAMA_HOST=http://localhost:11434
  SELECTOR_CONFIG_DIR=/home/jguardado/repos/agent-system/config

Provider env:
  GEMINI_API_KEY=...
  MINIMAX_API_KEY=...
  GLM_API_KEY=...

Custom providers use:
  MY_PROVIDER_API_BASE=https://...
  MY_PROVIDER_API_KEY_ENV=MY_PROVIDER_API_KEY
  MY_PROVIDER_API_KEY=...
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_model_selector.config_loader import (
    load_capability_definitions,
    load_model_tiers,
)
from ai_model_selector.intent.models import CapabilityDefinition
from ai_model_selector.models import (
    ModelSelection,
    ModelTier,
    RequestContext,
)
from ai_model_selector.selector import DeterministicSelector
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import ollama
import uvicorn

from ticket_agent.router.providers import ProviderConfig, load_provider, load_providers


load_dotenv(Path.home() / "config" / "router.env")

ROUTER_PORT = int(os.getenv("ROUTER_PORT", "8080"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
SELECTOR_CONFIG_DIR = Path(
    os.getenv("SELECTOR_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))
).expanduser()
CAPABILITIES_PATH = Path(
    os.getenv("SELECTOR_CAPABILITIES_PATH", str(SELECTOR_CONFIG_DIR / "capabilities.yaml"))
).expanduser()
MODELS_PATH = Path(
    os.getenv("SELECTOR_MODELS_PATH", str(SELECTOR_CONFIG_DIR / "models.yaml"))
).expanduser()
TASK_PROFILES_PATH = Path(
    os.getenv("SELECTOR_TASK_PROFILES_PATH", str(SELECTOR_CONFIG_DIR / "task_profiles.yaml"))
).expanduser()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / "logs" / "router.log"),
    ],
)
log = logging.getLogger("router")


@dataclass(frozen=True)
class SelectionPlan:
    capability: str
    endpoints: tuple[ModelSelection, ...]
    intent_confidence: float | None = None
    intent_debug: tuple[str, ...] = ()
    selector_debug: tuple[str, ...] = ()
    selector_ms: float = 0.0
    override: str | None = None


PROVIDERS = load_providers()

OVERRIDE_ALIASES = {
    "router-auto": "auto",
    "auto": "auto",
    "minimax-coder": "dev",
    "gemini-flash": "browsing",
    "glm-premium": "critical",
    "local-qwen": "trivial",
}


@lru_cache(maxsize=1)
def selector_components() -> tuple[
    tuple[CapabilityDefinition, ...],
    tuple[ModelTier, ...],
    DeterministicSelector,
]:
    capabilities = load_capability_definitions(CAPABILITIES_PATH)
    model_tiers = load_model_tiers(MODELS_PATH)
    selector = DeterministicSelector.from_yaml(
        MODELS_PATH,
        TASK_PROFILES_PATH,
        CAPABILITIES_PATH,
    )
    return capabilities, model_tiers, selector


def get_provider(provider_name: str) -> ProviderConfig:
    if provider_name in PROVIDERS:
        return PROVIDERS[provider_name]
    return load_provider(provider_name)


def provider_configured(provider: ProviderConfig) -> bool:
    return provider.local or bool(os.getenv(provider.api_key_env, ""))


def model_selection_from_tier(model: ModelTier) -> ModelSelection:
    return ModelSelection(
        selection_tier=model.selection_tier,
        provider=model.provider,
        model_name=model.model_name,
        deployment_name=model.deployment_name,
        invocation=model.invocation,
    )


def selection_from_override(
    model_override: str,
    *,
    selection_tier_hint: str | None = None,
) -> SelectionPlan | None:
    requested = OVERRIDE_ALIASES.get(model_override, model_override)
    if requested == "auto":
        return None

    _, model_tiers, _ = selector_components()
    if selection_tier_hint:
        for model in model_tiers:
            if selection_tier_hint == model.selection_tier and requested in {
                model.selection_tier,
                model.model_name,
                model.deployment_name,
            }:
                return SelectionPlan(
                    capability=f"override.{model.selection_tier}",
                    endpoints=(model_selection_from_tier(model),),
                    selector_debug=(
                        f"explicit_model_override:{model_override}",
                        f"selection_tier_hint:{selection_tier_hint}",
                    ),
                    override=model_override,
                )

    for model in model_tiers:
        if requested in {
            model.selection_tier,
            model.model_name,
            model.deployment_name,
        }:
            return SelectionPlan(
                capability=f"override.{model.selection_tier}",
                endpoints=(model_selection_from_tier(model),),
                selector_debug=(f"explicit_model_override:{model_override}",),
                override=model_override,
            )
    return None


def select_for_prompt(prompt: str) -> SelectionPlan:
    t0 = time.monotonic()
    capabilities, _, selector = selector_components()
    capability_names = {definition.name for definition in capabilities}

    if prompt in capability_names:
        context = RequestContext(capability=prompt)
        decision = selector.select(context)
        capability = decision.capability or prompt
        intent_confidence = None
        intent_debug: tuple[str, ...] = ()
    else:
        resolution = selector.resolve_intent(prompt)
        decision = selector.select_prompt(prompt)
        capability = resolution.capability
        intent_confidence = resolution.confidence
        intent_debug = tuple(resolution.debug)

    selector_ms = (time.monotonic() - t0) * 1000
    return SelectionPlan(
        capability=decision.capability or capability,
        endpoints=(decision.primary, *decision.fallbacks),
        intent_confidence=intent_confidence,
        intent_debug=intent_debug,
        selector_debug=tuple(decision.debug_reasons),
        selector_ms=selector_ms,
    )


def extract_last_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        return str(content)
    return ""


def normalize_messages_for_ollama(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        normalized.append({"role": role, "content": str(content)})
    return normalized


def router_metadata(
    endpoint: ModelSelection,
    plan: SelectionPlan,
    *,
    fallback_used: bool,
    fallback_reason: str | None,
    model_ms: float,
    total_ms: float,
    error: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "capability": plan.capability,
        "selection_tier": endpoint.selection_tier,
        "provider": endpoint.provider,
        "model": endpoint.deployment_name,
        "model_name": endpoint.model_name,
        "invocation": endpoint.invocation,
        "selector": {
            "intent_confidence": (
                round(plan.intent_confidence, 4)
                if plan.intent_confidence is not None
                else None
            ),
            "intent_debug": list(plan.intent_debug),
            "debug": list(plan.selector_debug),
            "override": plan.override,
        },
        "policy": {
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
        "timing": {
            "selector_ms": round(plan.selector_ms),
            "model_ms": round(model_ms),
            "total_ms": round(total_ms),
        },
    }
    if error is not None:
        metadata["error"] = error
    return metadata


def handle_local(messages: list[dict[str, Any]], model: str) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=normalize_messages_for_ollama(messages),
        think=False,
        options={
            "temperature": 0.7,
            "num_predict": 4096,
            "num_ctx": 8192,
        },
    )
    return response["message"]["content"]


async def proxy_to_provider(
    endpoint: ModelSelection,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    provider = get_provider(endpoint.provider)
    if provider.local:
        raise ValueError(f"{endpoint.selection_tier} is local and cannot be proxied")

    api_key = os.getenv(provider.api_key_env, "")
    if not api_key:
        raise ValueError(
            f"API key not configured for provider {provider.name} ({provider.api_key_env})"
        )
    if not provider.api_base:
        raise ValueError(f"API base not configured for provider {provider.name}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Model-Selection-Tier": endpoint.selection_tier,
        "X-Model-Provider": endpoint.provider,
        "X-Model-Invocation": endpoint.invocation,
    }
    payload = {
        "model": endpoint.deployment_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{provider.api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return data


def openai_chat_response(content: str, endpoint: ModelSelection) -> dict[str, Any]:
    return {
        "id": f"router-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": endpoint.deployment_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


app = FastAPI(title="LLM Router", version="3.0.0")


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    _, model_tiers, _ = selector_components()
    models = [{"id": "router-auto", "object": "model", "owned_by": "selector"}]
    for model in model_tiers:
        models.append(
            {
                "id": model.selection_tier,
                "object": "model",
                "owned_by": model.provider,
                "resolved_model": model.deployment_name,
                "model_name": model.model_name,
                "invocation": model.invocation,
            }
        )
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_start = time.monotonic()
    body = await request.json()
    messages = body.get("messages", [])
    model_override = body.get("model", "router-auto")
    selection_tier_hint = request.headers.get("X-Model-Selection-Tier")
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided"})

    try:
        prompt = extract_last_user_message(messages)
        plan = selection_from_override(
            model_override,
            selection_tier_hint=selection_tier_hint,
        ) or select_for_prompt(prompt)
    except Exception as exc:
        log.exception("Model selection failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"model selection failed: {exc}"},
        )

    if not plan.endpoints:
        return JSONResponse(status_code=500, content={"error": "selector returned no endpoints"})

    errors: list[str] = []
    primary_tier = plan.endpoints[0].selection_tier

    for index, endpoint in enumerate(plan.endpoints):
        provider = get_provider(endpoint.provider)
        fallback_used = index > 0
        fallback_reason = f"{primary_tier}_failed" if fallback_used else None

        try:
            if not provider_configured(provider):
                raise ValueError(
                    f"provider {provider.name} is not configured ({provider.api_key_env})"
                )

            t0 = time.monotonic()
            if provider.local:
                content = handle_local(messages, endpoint.deployment_name)
                model_ms = (time.monotonic() - t0) * 1000
                total_ms = (time.monotonic() - request_start) * 1000
                response = openai_chat_response(content, endpoint)
                response["_router"] = router_metadata(
                    endpoint,
                    plan,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    model_ms=model_ms,
                    total_ms=total_ms,
                )
                return response

            data = await proxy_to_provider(
                endpoint, messages, max_tokens, temperature, stream
            )
            model_ms = (time.monotonic() - t0) * 1000
            total_ms = (time.monotonic() - request_start) * 1000
            data["_router"] = router_metadata(
                endpoint,
                plan,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                model_ms=model_ms,
                total_ms=total_ms,
            )
            return data

        except Exception as exc:
            message = f"{endpoint.selection_tier} -> {endpoint.provider}/{endpoint.deployment_name}: {exc}"
            log.error("Model attempt failed: %s", message)
            errors.append(message)

    return JSONResponse(
        status_code=502,
        content={
            "error": "all selected model endpoints failed",
            "attempts": errors,
            "_router": {
                "capability": plan.capability,
                "selector": {
                    "intent_confidence": plan.intent_confidence,
                    "intent_debug": list(plan.intent_debug),
                    "debug": list(plan.selector_debug),
                    "override": plan.override,
                },
            },
        },
    )


@app.get("/router/status")
async def router_status() -> dict[str, Any]:
    ollama_ok = False
    try:
        ollama.Client(host=OLLAMA_HOST).list()
        ollama_ok = True
    except Exception:
        pass

    selector_ok = False
    selector_error = None
    model_tiers: tuple[ModelTier, ...] = ()
    try:
        _, model_tiers, _ = selector_components()
        selector_ok = True
    except Exception as exc:
        selector_error = str(exc)

    provider_names = set(PROVIDERS) | {model.provider for model in model_tiers}
    providers = {
        name: {
            "configured": provider_configured(get_provider(name)),
            "api_base": get_provider(name).api_base,
            "api_key_env": None if get_provider(name).local else get_provider(name).api_key_env,
        }
        for name in sorted(provider_names)
    }

    return {
        "status": "running",
        "selector": {
            "connected": selector_ok,
            "error": selector_error,
            "capabilities_path": str(CAPABILITIES_PATH),
            "models_path": str(MODELS_PATH),
            "task_profiles_path": str(TASK_PROFILES_PATH),
        },
        "ollama": {"connected": ollama_ok, "host": OLLAMA_HOST},
        "providers": providers,
        "models": [
            {
                "selection_tier": model.selection_tier,
                "provider": model.provider,
                "deployment_name": model.deployment_name,
                "model_name": model.model_name,
                "invocation": model.invocation,
            }
            for model in model_tiers
        ],
    }


def main() -> None:
    log.info("Starting LLM Router on port %s", ROUTER_PORT)
    log.info("Selector config: %s", SELECTOR_CONFIG_DIR)
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_PORT, log_level="info")


if __name__ == "__main__":
    main()

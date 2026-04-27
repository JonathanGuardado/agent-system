from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_model_selector.config_loader import load_capability_definitions
from ai_model_selector.intent.models import CapabilityDefinition
from ai_model_selector.models import ModelSelection as SelectorModelSelection
from ai_model_selector.models import RequestContext
from ai_model_selector.selector import DeterministicSelector

from ticket_agent.domain.model_selection import ModelEndpoint, ModelSelection

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
CAPABILITIES_PATH = CONFIG_DIR / "capabilities.yaml"
MODELS_PATH = CONFIG_DIR / "models.yaml"
TASK_PROFILES_PATH = CONFIG_DIR / "task_profiles.yaml"


@lru_cache(maxsize=1)
def _selector_components() -> tuple[
    tuple[CapabilityDefinition, ...],
    DeterministicSelector,
]:
    capability_definitions = load_capability_definitions(CAPABILITIES_PATH)
    selector = DeterministicSelector.from_yaml(
        MODELS_PATH,
        TASK_PROFILES_PATH,
        CAPABILITIES_PATH,
    )
    return capability_definitions, selector


def select_model_for_capability(capability_or_text: str) -> ModelSelection:
    capability_definitions, selector = _selector_components()
    capability_names = {definition.name for definition in capability_definitions}

    if capability_or_text in capability_names:
        context = RequestContext(capability=capability_or_text)
        decision = selector.select(context)
        intent_confidence = None
        intent_debug: tuple[str, ...] = ()
    else:
        resolution = selector.resolve_intent(capability_or_text)
        decision = selector.select_prompt(capability_or_text)
        intent_confidence = resolution.confidence
        intent_debug = tuple(resolution.debug)

    return ModelSelection(
        capability=decision.capability,
        primary=_to_model_endpoint(decision.primary),
        fallbacks=tuple(_to_model_endpoint(fallback) for fallback in decision.fallbacks),
        intent_confidence=intent_confidence,
        intent_debug=intent_debug,
        selector_debug=tuple(decision.debug_reasons),
    )


def _to_model_endpoint(selection: SelectorModelSelection) -> ModelEndpoint:
    return ModelEndpoint(
        selection_tier=selection.selection_tier,
        provider=selection.provider,
        model_name=selection.model_name,
        deployment_name=selection.deployment_name,
        invocation=getattr(selection, "invocation", "openai_chat"),
    )

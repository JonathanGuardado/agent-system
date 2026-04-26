from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_model_selector.config_loader import load_capability_definitions
from ai_model_selector.intent import IntentResolver, build_request_context
from ai_model_selector.intent.models import CapabilityDefinition
from ai_model_selector.models import RequestContext
from ai_model_selector.selector import DeterministicSelector

from ticket_agent.domain.model_selection import ModelSelection

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
CAPABILITIES_PATH = CONFIG_DIR / "capabilities.yaml"
MODELS_PATH = CONFIG_DIR / "models.yaml"
TASK_PROFILES_PATH = CONFIG_DIR / "task_profiles.yaml"


@lru_cache(maxsize=1)
def _selector_components() -> tuple[
    tuple[CapabilityDefinition, ...],
    IntentResolver,
    DeterministicSelector,
]:
    capability_definitions = load_capability_definitions(CAPABILITIES_PATH)
    resolver = IntentResolver(capability_definitions)
    selector = DeterministicSelector.from_yaml(MODELS_PATH, TASK_PROFILES_PATH)
    return capability_definitions, resolver, selector


def select_model_for_capability(capability_or_text: str) -> ModelSelection:
    capability_definitions, resolver, selector = _selector_components()
    capability_names = {definition.name for definition in capability_definitions}

    if capability_or_text in capability_names:
        context = RequestContext(capability=capability_or_text)
    else:
        context = build_request_context(resolver.resolve(capability_or_text))

    decision = selector.select(context)
    return ModelSelection(
        capability=decision.capability,
        primary_model=decision.primary.selection_tier,
        fallback_models=decision.fallback_selection_tiers,
    )

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ai_model_selector import build_request_context
from ai_model_selector.config_loader import load_capability_definitions
from ai_model_selector.intent.models import CapabilityDefinition
from ai_model_selector.intent.resolver import IntentResolver
from ai_model_selector.models import ModelSelection as SelectorModelSelection
from ai_model_selector.models import SelectionDecision
from ai_model_selector.selector import DeterministicSelector

from ticket_agent.domain.model_selection import ModelEndpoint, ModelSelection

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
CAPABILITIES_PATH = CONFIG_DIR / "capabilities.yaml"
MODELS_PATH = CONFIG_DIR / "models.yaml"
TASK_PROFILES_PATH = CONFIG_DIR / "task_profiles.yaml"


@dataclass(frozen=True, slots=True)
class SelectorComponents:
    resolver: IntentResolver
    selector: DeterministicSelector

    def select(self, capability: str) -> SelectionDecision:
        resolution = self.resolver.resolve(capability)
        context = build_request_context(resolution)
        return self.selector.select(context)


@lru_cache(maxsize=1)
def _selector_components() -> tuple[
    tuple[CapabilityDefinition, ...],
    IntentResolver,
    DeterministicSelector,
]:
    capability_definitions = load_capability_definitions(CAPABILITIES_PATH)
    resolver = IntentResolver(capability_definitions)
    selector = DeterministicSelector.from_yaml(
        MODELS_PATH,
        TASK_PROFILES_PATH,
        CAPABILITIES_PATH,
    )
    return capability_definitions, resolver, selector


@lru_cache(maxsize=1)
def load_model_selector() -> SelectorComponents:
    _, resolver, selector = _selector_components()
    return SelectorComponents(resolver=resolver, selector=selector)


def select_model_for_capability(capability_or_text: str) -> ModelSelection:
    _, resolver, selector = _selector_components()
    resolution = resolver.resolve(capability_or_text)
    context = build_request_context(resolution)
    decision = selector.select(context)
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

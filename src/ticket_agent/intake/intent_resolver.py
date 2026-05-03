"""Deterministic intake intent resolution.

Uses ai-model-selector to resolve a Slack message to an `IntakeMode` and a
model decision. No LLM calls; no network access.
"""

from __future__ import annotations

import re
from typing import Any

from ticket_agent.domain.intake import IntakeMode, IntakeResolution
from ticket_agent.router.selector_config import (
    SelectorComponents,
    load_model_selector,
)


CAPABILITY_TO_MODE: dict[str, IntakeMode] = {
    "architecture.design": IntakeMode.NEW_PROJECT,
    "code.implement": IntakeMode.NEW_FEATURE,
    "ticket.decompose": IntakeMode.NEW_TICKETS,
    "code.verify": IntakeMode.DIRECT_TICKET,
    "trivial.respond": IntakeMode.DIRECT_TICKET,
}


_PROJECT_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})(?:-\d+)?\b")


class IntakeIntentResolver:
    """Resolve free-form Slack text into an :class:`IntakeResolution`.

    The resolver uses ai-model-selector deterministically to map text to a
    capability, then maps the capability to an :class:`IntakeMode`. It also
    inspects the text for context (project keys, repo references) and asks
    for clarification when required for the chosen mode.
    """

    def __init__(self, selector: SelectorComponents | None = None) -> None:
        self._selector = selector if selector is not None else load_model_selector()

    def resolve(self, text: str) -> IntakeResolution:
        """Return an :class:`IntakeResolution` for the supplied Slack text."""

        normalized = text.strip()
        if not normalized:
            raise ValueError("intake text must not be empty")

        decision = self._selector.select(normalized)
        capability = str(decision.capability)
        mode = CAPABILITY_TO_MODE.get(capability, IntakeMode.DIRECT_TICKET)

        primary = _model_name(decision.primary)
        fallbacks = tuple(_model_name(item) for item in decision.fallbacks)

        clarification = _clarification_for(mode, normalized)

        return IntakeResolution(
            mode=mode,
            capability=capability,
            model_primary=primary,
            model_fallbacks=fallbacks,
            requires_clarification=clarification is not None,
            clarification_question=clarification,
        )


def _model_name(selection: Any) -> str:
    deployment = getattr(selection, "deployment_name", None)
    if isinstance(deployment, str) and deployment.strip():
        return deployment.strip()
    name = getattr(selection, "model_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    tier = getattr(selection, "selection_tier", None)
    if isinstance(tier, str) and tier.strip():
        return tier.strip()
    raise ValueError("model selection is missing a deployment or model name")


def _clarification_for(mode: IntakeMode, text: str) -> str | None:
    if mode == IntakeMode.NEW_FEATURE:
        if not _has_project_reference(text) and not _has_repository_reference(text):
            return (
                "Which Jira project and repository should this feature be added "
                "to? Please reply with the project key (e.g. AGENT) and repo."
            )
        return None

    if mode == IntakeMode.NEW_TICKETS:
        if not _has_project_reference(text) and not _has_epic_reference(text):
            return (
                "Where should I attach these tickets? Reply with the Jira "
                "project key or the epic to attach them under."
            )
        return None

    if mode == IntakeMode.NEW_PROJECT:
        if not _has_project_reference(text):
            return (
                "What Jira project key should I use for this new initiative? "
                "Note: creating brand-new Jira projects is not yet supported "
                "in v1."
            )
        return None

    return None


def _has_project_reference(text: str) -> bool:
    matches = _PROJECT_KEY_PATTERN.findall(text)
    if any(match for match in matches):
        return True
    lowered = text.lower()
    return "project" in lowered and any(ch.isupper() for ch in text)


def _has_epic_reference(text: str) -> bool:
    return "epic" in text.lower() or bool(_PROJECT_KEY_PATTERN.search(text))


def _has_repository_reference(text: str) -> bool:
    lowered = text.lower()
    return "repo" in lowered or "repository" in lowered or "/" in text


__all__ = ["CAPABILITY_TO_MODE", "IntakeIntentResolver"]

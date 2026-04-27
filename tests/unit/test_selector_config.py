from __future__ import annotations

import pytest

from ticket_agent.router.app import selection_from_override
from ticket_agent.router.selector_config import select_model_for_capability


@pytest.mark.parametrize(
    ("text", "capability", "primary", "fallbacks", "provider", "model_name", "deployment"),
    (
        (
            "build a new SaaS app from scratch",
            "architecture.design",
            "design",
            ("dev", "browsing"),
            "minimax",
            "MiniMax-M2.7",
            "MiniMax-M2.7",
        ),
        (
            "break this feature into Jira tickets",
            "ticket.decompose",
            "design",
            ("dev", "browsing"),
            "minimax",
            "MiniMax-M2.7",
            "MiniMax-M2.7",
        ),
        (
            "implement OAuth login",
            "code.implement",
            "dev",
            ("browsing",),
            "deepseek",
            "deepseek-v4-pro",
            "deepseek-v4-pro",
        ),
        (
            "review this bug fix",
            "code.verify",
            "browsing",
            ("design",),
            "gemini",
            "gemini-flash",
            "gemini-2.5-flash",
        ),
        (
            "hello",
            "trivial.respond",
            "trivial",
            ("browsing",),
            "local",
            "qwen3.5:9b",
            "qwen3.5:9b",
        ),
    ),
)
def test_select_model_for_capability_resolves_expected_policy(
    text: str,
    capability: str,
    primary: str,
    fallbacks: tuple[str, ...],
    provider: str,
    model_name: str,
    deployment: str,
):
    selection = select_model_for_capability(text)

    assert selection.capability == capability
    assert selection.primary_model == primary
    assert selection.fallback_models == fallbacks
    assert selection.primary.selection_tier == primary
    assert selection.primary.provider == provider
    assert selection.primary.model_name == model_name
    assert selection.primary.deployment_name == deployment
    assert selection.primary.invocation == "openai_chat"
    assert tuple(fallback.selection_tier for fallback in selection.fallbacks) == fallbacks


def test_selection_override_prefers_selection_tier_hint_for_duplicate_deployments():
    plan = selection_from_override(
        "MiniMax-M2.7",
        selection_tier_hint="design",
    )

    assert plan is not None
    assert plan.capability == "override.design"
    assert plan.endpoints[0].selection_tier == "design"
    assert plan.endpoints[0].deployment_name == "MiniMax-M2.7"


def test_selection_override_ignores_inconsistent_selection_tier_hint():
    plan = selection_from_override(
        "gemini-2.5-flash",
        selection_tier_hint="design",
    )

    assert plan is not None
    assert plan.endpoints[0].selection_tier == "browsing"

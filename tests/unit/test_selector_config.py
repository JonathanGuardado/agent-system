from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ticket_agent.router.providers import PROVIDER_DEFAULTS
from ticket_agent.router.selector_config import select_model_for_capability

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.mark.parametrize(
    (
        "text",
        "capability",
        "primary",
        "fallbacks",
        "provider",
        "model_name",
        "deployment",
    ),
    (
        (
            "implement OAuth login",
            "code.implement",
            "deepseek-v4-pro",
            ("gemini-flash", "qwen-local"),
            "deepseek",
            "deepseek-v4-pro",
            "deepseek-v4-pro",
        ),
        (
            "review this bug fix",
            "code.verify",
            "gemini-flash",
            ("deepseek-v4-pro", "qwen-local"),
            "gemini",
            "gemini-flash",
            "gemini-2.5-flash",
        ),
        (
            "break this feature into Jira tickets",
            "ticket.decompose",
            "deepseek-v4-pro",
            ("gemini-flash", "qwen-local"),
            "deepseek",
            "deepseek-v4-pro",
            "deepseek-v4-pro",
        ),
        (
            "hello",
            "trivial.respond",
            "qwen-local",
            ("gemini-flash",),
            "ollama",
            "qwen-local",
            "qwen3.5:9b",
        ),
    ),
)
def test_select_model_for_capability_resolves_v1_policy(
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


def test_qwen_local_uses_ollama_provider_mapping():
    selection = select_model_for_capability("hello")
    qwen = selection.primary

    assert qwen.selection_tier == "qwen-local"
    assert qwen.provider == "ollama"
    assert qwen.model_name == "qwen-local"
    assert qwen.deployment_name == "qwen3.5:9b"


def test_v1_provider_defaults_are_deepseek_gemini_and_ollama_only():
    assert set(PROVIDER_DEFAULTS) == {"deepseek", "gemini", "ollama"}


def test_task_profiles_do_not_use_cost_weighted_routing():
    task_profiles = yaml.safe_load(
        (CONFIG_DIR / "task_profiles.yaml").read_text(encoding="utf-8")
    )["task_profiles"]

    for profile in task_profiles:
        weights = profile.get("scoring_weights", {})
        assert "cost" not in weights

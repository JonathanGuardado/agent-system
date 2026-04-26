from __future__ import annotations

import pytest

from ticket_agent.router.selector_config import select_model_for_capability


@pytest.mark.parametrize(
    ("text", "capability", "primary", "fallbacks"),
    (
        (
            "build a new SaaS app from scratch",
            "architecture.design",
            "minimax-m2.5",
            ("deepseek-v4", "gemini-flash"),
        ),
        (
            "break this feature into Jira tickets",
            "ticket.decompose",
            "minimax-m2.5",
            ("deepseek-v4", "gemini-flash"),
        ),
        (
            "implement OAuth login",
            "code.implement",
            "deepseek-v4",
            ("minimax-m2.5", "gemini-flash"),
        ),
        (
            "review this bug fix",
            "code.verify",
            "gemini-flash",
            ("minimax-m2.5",),
        ),
        (
            "hello",
            "trivial.respond",
            "router-auto",
            ("gemini-flash",),
        ),
    ),
)
def test_select_model_for_capability_resolves_expected_policy(
    text: str,
    capability: str,
    primary: str,
    fallbacks: tuple[str, ...],
):
    selection = select_model_for_capability(text)

    assert selection.capability == capability
    assert selection.primary_model == primary
    assert selection.fallback_models == fallbacks

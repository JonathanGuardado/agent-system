from __future__ import annotations

import pytest

from ticket_agent.domain.intake import IntakeMode
from ticket_agent.intake.intent_resolver import (
    CAPABILITY_TO_MODE,
    IntakeIntentResolver,
)


@pytest.fixture(scope="module")
def resolver() -> IntakeIntentResolver:
    return IntakeIntentResolver()


def test_capability_to_mode_table_is_complete():
    for capability in (
        "architecture.design",
        "code.implement",
        "ticket.decompose",
        "code.verify",
        "trivial.respond",
    ):
        assert capability in CAPABILITY_TO_MODE


def test_resolve_code_implement_maps_to_new_feature(resolver: IntakeIntentResolver):
    resolution = resolver.resolve("implement OAuth login in AGENT")

    assert resolution.capability == "code.implement"
    assert resolution.mode == IntakeMode.NEW_FEATURE
    assert resolution.model_primary
    assert resolution.requires_clarification is False


def test_resolve_architecture_design_maps_to_new_project(resolver: IntakeIntentResolver):
    resolution = resolver.resolve("design a scalable notification system")

    assert resolution.capability == "architecture.design"
    assert resolution.mode == IntakeMode.NEW_PROJECT


def test_resolve_ticket_decompose_maps_to_new_tickets(resolver: IntakeIntentResolver):
    resolution = resolver.resolve("break this feature into Jira tickets for AGENT")

    assert resolution.capability == "ticket.decompose"
    assert resolution.mode == IntakeMode.NEW_TICKETS


def test_resolve_code_verify_maps_to_direct_ticket(resolver: IntakeIntentResolver):
    resolution = resolver.resolve("review this bug fix")

    assert resolution.capability == "code.verify"
    assert resolution.mode == IntakeMode.DIRECT_TICKET


def test_resolve_trivial_respond_maps_to_direct_ticket(resolver: IntakeIntentResolver):
    resolution = resolver.resolve("hello")

    assert resolution.capability == "trivial.respond"
    assert resolution.mode == IntakeMode.DIRECT_TICKET


def test_new_feature_without_project_or_repo_asks_for_clarification(
    resolver: IntakeIntentResolver,
):
    resolution = resolver.resolve("implement OAuth login")

    assert resolution.mode == IntakeMode.NEW_FEATURE
    assert resolution.requires_clarification is True
    assert resolution.clarification_question
    assert "project" in resolution.clarification_question.lower()


def test_new_tickets_without_project_or_epic_asks_for_clarification(
    resolver: IntakeIntentResolver,
):
    resolution = resolver.resolve("break this feature into Jira tickets")

    assert resolution.mode == IntakeMode.NEW_TICKETS
    assert resolution.requires_clarification is True
    assert resolution.clarification_question


def test_empty_text_raises(resolver: IntakeIntentResolver):
    with pytest.raises(ValueError):
        resolver.resolve("   ")

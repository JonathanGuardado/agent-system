"""Canonical goal identity shared by intake, storage, and execution lookup."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ticket_agent.domain.errors import AgentSystemError

GOAL_ID_PATTERN = re.compile(r"prop-[0-9a-f]{12}\Z")
GOAL_LABEL_PREFIX = "ai-goal-"


class GoalIdentityError(AgentSystemError):
    """Raised when goal identity is missing, malformed, or ambiguous."""


def normalize_goal_id(value: object) -> str:
    """Validate and return the proposal id verbatim; never repair it."""

    if not isinstance(value, str) or GOAL_ID_PATTERN.fullmatch(value) is None:
        raise GoalIdentityError(
            "goal id must match exactly 'prop-[0-9a-f]{12}'"
        )
    return value


def goal_label(goal_id: object) -> str:
    return f"{GOAL_LABEL_PREFIX}{normalize_goal_id(goal_id)}"


def goal_id_from_labels(labels: Iterable[object]) -> str:
    """Extract one canonical label, refusing every malformed lookalike."""

    candidates = [
        label
        for label in labels
        if isinstance(label, str)
        and label.lower().startswith(GOAL_LABEL_PREFIX)
    ]
    if len(candidates) != 1:
        raise GoalIdentityError(
            "ticket must carry exactly one canonical ai-goal-* label"
        )
    label = candidates[0]
    if not label.startswith(GOAL_LABEL_PREFIX):
        raise GoalIdentityError("goal label must be lowercase and canonical")
    goal_id = label.removeprefix(GOAL_LABEL_PREFIX)
    if label != goal_label(goal_id):
        raise GoalIdentityError("goal label must be canonical")
    return goal_id


def validate_goal_labels(labels: Iterable[object], *, goal_id: object) -> str:
    """Require labels to identify exactly the expected goal."""

    expected = normalize_goal_id(goal_id)
    actual = goal_id_from_labels(labels)
    if actual != expected:
        raise GoalIdentityError(
            f"goal label identifies {actual!r}, expected {expected!r}"
        )
    return actual


__all__ = [
    "GOAL_ID_PATTERN",
    "GOAL_LABEL_PREFIX",
    "GoalIdentityError",
    "goal_id_from_labels",
    "goal_label",
    "normalize_goal_id",
    "validate_goal_labels",
]

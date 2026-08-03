"""Shared format for ticket acceptance criteria.

Acceptance criteria are the contract between intake, planning, and review.
They are rendered into the Jira ticket description under a stable heading by
the proposal generator, then parsed back out of the description at execution
time so the planner and reviewer can reason about each criterion explicitly.

``render_acceptance_criteria`` and ``parse_acceptance_criteria`` are inverses
for the bullet formats this module emits; keeping both here means the render
and parse sides can never drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

ACCEPTANCE_HEADING = "Acceptance Criteria:"

_HEADING_KEY = ACCEPTANCE_HEADING.rstrip(":").strip().lower()
_BULLET_RE = re.compile(r"^(?:[-*]|\d+[.)])\s+(.*)$")


def render_acceptance_criteria(criteria: Sequence[str]) -> str:
    """Render criteria as a heading followed by ``- `` bullets.

    Returns an empty string when there are no non-empty criteria so callers
    can omit the section entirely.
    """

    items = [item.strip() for item in criteria if item and item.strip()]
    if not items:
        return ""
    return "\n".join([ACCEPTANCE_HEADING, *(f"- {item}" for item in items)])


def parse_acceptance_criteria(text: str) -> list[str]:
    """Extract the criteria bullets from a ticket description.

    Finds the acceptance-criteria heading (case-insensitive, trailing colon
    optional), then collects consecutive bullet lines until a blank line or a
    non-bullet line ends the section. Returns an empty list when no heading is
    present.
    """

    if not text:
        return []

    criteria: list[str] = []
    collecting = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not collecting:
            if line.rstrip(":").strip().lower() == _HEADING_KEY:
                collecting = True
            continue
        if not line:
            break
        match = _BULLET_RE.match(line)
        if match is None:
            break
        item = match.group(1).strip()
        if item:
            criteria.append(item)
    return criteria

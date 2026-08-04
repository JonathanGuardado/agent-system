#!/usr/bin/env python3
"""Assert the pinned selector SHA agrees between pyproject.toml and uv.lock.

`uv sync --frozen` fails when the lock is stale relative to the *declared*
requirement, but the requirement is a git URL and the lock records a resolved
rev: the two can be edited independently, and a lock left pointing at an older
selector commit is exactly the failure that would make a recorded digest
untrustworthy without anything looking wrong.

Runs with no third-party imports so CI can call it before any sync.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib

DEPENDENCY = "ai-model-selector"
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Matches the trailing `@<sha>` of a PEP 508 git URL requirement.
_PYPROJECT_REV = re.compile(r"git\+[^@\s]+@([0-9a-f]{40})\b")

#: Matches the `?rev=<sha>` uv records in the resolved source URL.
_LOCK_REV = re.compile(r"[?&]rev=([0-9a-f]{40})\b")


def _pyproject_rev(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = data["project"]["dependencies"]
    matches = [req for req in requirements if req.startswith(f"{DEPENDENCY} @")]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {DEPENDENCY} requirement, found {len(matches)}"
        )
    found = _PYPROJECT_REV.search(matches[0])
    if found is None:
        raise SystemExit(
            f"{DEPENDENCY} is not pinned to a full 40-character commit SHA: "
            f"{matches[0]!r}. A branch or tag can move under a recorded digest."
        )
    return found.group(1)


def _lock_revs(lock: Path) -> set[str]:
    """Every rev the lock records for the dependency, across all mentions."""
    text = lock.read_text(encoding="utf-8")
    revs: set[str] = set()
    for line in text.splitlines():
        if DEPENDENCY in line or f"{DEPENDENCY}.git" in line:
            revs.update(_LOCK_REV.findall(line))
    return revs


def main() -> int:
    pyproject = REPO_ROOT / "pyproject.toml"
    lock = REPO_ROOT / "uv.lock"
    if not lock.exists():
        raise SystemExit("uv.lock is missing; run `uv lock`")

    declared = _pyproject_rev(pyproject)
    locked = _lock_revs(lock)

    if not locked:
        raise SystemExit(f"uv.lock records no rev for {DEPENDENCY}")
    if locked != {declared}:
        raise SystemExit(
            f"{DEPENDENCY} pin disagrees: pyproject.toml declares {declared}, "
            f"uv.lock records {sorted(locked)}. Run `uv lock` after changing "
            f"the pin."
        )

    print(f"{DEPENDENCY} pinned at {declared} in both pyproject.toml and uv.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())

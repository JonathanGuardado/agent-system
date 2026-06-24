"""Redact machine-local filesystem paths from externally shared text."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

_GENERIC_REPO_PATH_RE = re.compile(
    r"(?<![\w./-])(?:/home/[^/\s:]+|/Users/[^/\s:]+)/repos/[^/\s:]+"
    r"(?:/\.worktrees/[^/\s:/]+/[^/\s:/]+)?"
)
_GENERIC_HOME_PATH_RE = re.compile(
    r"(?<![\w./-])(?:/home/[^/\s:]+|/Users/[^/\s:]+)"
)
_GENERIC_WINDOWS_HOME_PATH_RE = re.compile(
    r"(?<![\w.\\/-])[A-Za-z]:\\Users\\[^\\\s:]+",
    re.IGNORECASE,
)


def normalize_local_paths(
    paths: Iterable[str | os.PathLike[str]],
) -> tuple[str, ...]:
    """Return unique local paths ordered longest-first for safe replacement."""

    normalized: set[str] = set()
    for path in paths:
        value = os.fspath(path).strip()
        candidate = os.path.expanduser(value).rstrip("/\\")
        if candidate and not re.fullmatch(r"[A-Za-z]:", candidate):
            normalized.add(candidate)
    return tuple(sorted(normalized, key=len, reverse=True))


def redact_local_paths(
    text: str,
    local_paths: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Replace configured and common user-local paths with stable placeholders."""

    sanitized = text
    for path in normalize_local_paths(local_paths):
        escaped = re.escape(path)
        sanitized = re.sub(
            rf"{escaped}[\\/]\.worktrees[\\/][^\s:/\\]+[\\/][^\s:/\\]+",
            "<repo>",
            sanitized,
        )
        sanitized = sanitized.replace(path, "<repo>")
    sanitized = _GENERIC_REPO_PATH_RE.sub("<repo>", sanitized)
    sanitized = _GENERIC_HOME_PATH_RE.sub("<home>", sanitized)
    sanitized = _GENERIC_WINDOWS_HOME_PATH_RE.sub("<home>", sanitized)
    return sanitized


__all__ = ["normalize_local_paths", "redact_local_paths"]

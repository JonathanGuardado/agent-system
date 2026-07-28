"""Redact machine-local paths and credential material from shared text.

Two independent concerns live here because both are egress filters applied to
the same strings: local filesystem paths leak machine layout, and credential
material leaks authority. Transcripts, prompts, logs, and PR bodies all pass
through these.

Secret patterns are deliberately high-confidence and prefix-anchored. A loose
pattern that redacts ordinary code is worse than useless: it hides the very
diff a reviewer needs to read, and it trains people to ignore the placeholder.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

SECRET_PLACEHOLDER = "<redacted-secret>"

# Prefix-anchored provider token formats. Each is specific enough that a match
# is a credential rather than a coincidence.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{35,}"),
    re.compile(r"\bASIA[A-Z0-9]{16}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    # PEM private key blocks, header through footer.
    re.compile(
        r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Bearer/Basic credentials in an Authorization header.
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,}"),
)

# Long opaque values assigned to a secret-sounding name. Only the value is
# replaced so the assignment stays readable in a diff.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)"
    r"(\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*\b"
    r"\s*[:=]\s*)"
    r"(['\"]?)([A-Za-z0-9+/_-]{16,}={0,2})\2"
)

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


def redact_secrets(text: str) -> str:
    """Replace credential material with a stable placeholder."""

    sanitized = text
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(SECRET_PLACEHOLDER, sanitized)
    sanitized = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{SECRET_PLACEHOLDER}{m.group(2)}",
        sanitized,
    )
    return sanitized


def redact(
    text: str,
    local_paths: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Apply every egress filter: credentials first, then local paths.

    Secrets are removed before paths so a token embedded in a path-like string
    is still caught.
    """

    return redact_local_paths(redact_secrets(text), local_paths)


__all__ = [
    "SECRET_PLACEHOLDER",
    "normalize_local_paths",
    "redact",
    "redact_local_paths",
    "redact_secrets",
]

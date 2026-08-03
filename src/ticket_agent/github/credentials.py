"""Role-based GitHub credentials for subprocess invocations.

The agent system uses two GitHub identities:

- ``admin``: a personal account token (e.g. ``GH_ADMIN_TOKEN``) used for
  account-scoped operations such as creating repositories or managing
  collaborators.
- ``bot``: a service account token (e.g. ``GH_BOT_TOKEN``) used for
  code-writing operations such as cloning, committing, pushing, and
  opening pull requests.

Subprocesses receive the right token via environment variables instead
of command-line arguments so the token never lands in ``ps`` output or
git config files. ``gh`` reads ``GH_TOKEN`` natively; ``git`` is wired
via the ``GIT_CONFIG_*`` env vars to inject an ``http.extraheader`` for
github.com without persisting it to repo config.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Literal

GitHubRole = Literal["admin", "bot"]

# Annotated rather than inferred: without this the constants are plain `str`,
# so every call that passes one loses the distinction between the admin and bot
# tokens at the type level -- which is the distinction the whole module exists
# to keep.
GH_ROLE_ADMIN: GitHubRole = "admin"
GH_ROLE_BOT: GitHubRole = "bot"

_GIT_GITHUB_EXTRAHEADER_KEY = "http.https://github.com/.extraheader"
_PASSTHROUGH_ENV_VARS = ("PATH", "HOME", "VIRTUAL_ENV", "LANG", "LC_ALL")


@dataclass(frozen=True, slots=True)
class GitHubCredentials:
    """Container for per-role GitHub PATs.

    Either token may be ``None`` when not configured. In that case the
    helper produces no env override for that role. Code-writing GitHub
    operations treat a missing bot token as an error instead of falling
    back to whatever ``gh``/``git`` credentials are configured on the host.
    """

    admin_token: str | None = None
    bot_token: str | None = None

    def token_for(self, role: GitHubRole) -> str | None:
        if role == GH_ROLE_ADMIN:
            return self.admin_token
        if role == GH_ROLE_BOT:
            return self.bot_token
        raise ValueError(f"unknown github role: {role!r}")

    def has_token_for(self, role: GitHubRole) -> bool:
        return bool(self.token_for(role))

    def gh_env(
        self,
        role: GitHubRole,
        *,
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str] | None:
        """Return env vars for a ``gh`` invocation.

        Returns ``None`` when no token is configured for the role so the
        caller can leave subprocess env unchanged (inherit parent env).
        """

        token = self.token_for(role)
        if not token:
            return None
        env = _minimal_env(base_env)
        env["GH_TOKEN"] = token
        return env

    def git_env(
        self,
        role: GitHubRole,
        *,
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str] | None:
        """Return env vars for a ``git`` invocation that needs GitHub auth.

        Injects a GitHub smart-HTTP Basic authorization header for
        ``x-access-token:<PAT>`` via the ``GIT_CONFIG_COUNT`` /
        ``GIT_CONFIG_KEY_<n>`` / ``GIT_CONFIG_VALUE_<n>`` protocol, so the
        token is not stored in repo config or visible on the command line.
        Returns ``None`` when no token is configured.
        """

        token = self.token_for(role)
        if not token:
            return None
        env = _minimal_env(base_env)
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = _GIT_GITHUB_EXTRAHEADER_KEY
        encoded_token = base64.b64encode(
            f"x-access-token:{token}".encode()
        ).decode("ascii")
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {encoded_token}"
        env["GH_TOKEN"] = token
        return env


def _minimal_env(base_env: Mapping[str, str] | None) -> dict[str, str]:
    source = dict(base_env) if base_env is not None else dict(os.environ)
    env: dict[str, str] = {}
    for name in _PASSTHROUGH_ENV_VARS:
        value = source.get(name)
        if value:
            env[name] = value
    return env


__all__ = [
    "GH_ROLE_ADMIN",
    "GH_ROLE_BOT",
    "GitHubCredentials",
    "GitHubRole",
]

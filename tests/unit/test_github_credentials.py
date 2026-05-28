"""Tests for the GitHubCredentials role-based token helper."""

from __future__ import annotations

import pytest

from ticket_agent.github import GH_ROLE_ADMIN, GH_ROLE_BOT, GitHubCredentials


def test_token_for_returns_role_specific_token():
    credentials = GitHubCredentials(admin_token="admin-pat", bot_token="bot-pat")
    assert credentials.token_for(GH_ROLE_ADMIN) == "admin-pat"
    assert credentials.token_for(GH_ROLE_BOT) == "bot-pat"


def test_token_for_missing_returns_none():
    credentials = GitHubCredentials()
    assert credentials.token_for(GH_ROLE_ADMIN) is None
    assert credentials.token_for(GH_ROLE_BOT) is None


def test_token_for_unknown_role_raises():
    credentials = GitHubCredentials(admin_token="x", bot_token="y")
    with pytest.raises(ValueError, match="unknown github role"):
        credentials.token_for("owner")  # type: ignore[arg-type]


def test_has_token_for_reflects_presence():
    credentials = GitHubCredentials(admin_token="x")
    assert credentials.has_token_for(GH_ROLE_ADMIN) is True
    assert credentials.has_token_for(GH_ROLE_BOT) is False


def test_gh_env_returns_none_when_token_missing():
    credentials = GitHubCredentials(admin_token="x")
    assert credentials.gh_env(GH_ROLE_BOT) is None


def test_gh_env_populates_gh_token_and_passthrough_path():
    credentials = GitHubCredentials(bot_token="bot-pat")
    env = credentials.gh_env(
        GH_ROLE_BOT,
        base_env={
            "PATH": "/usr/bin",
            "HOME": "/home/bot",
            "GH_TOKEN": "should-be-overwritten",
            "AWS_SECRET": "secret",
        },
    )
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/bot",
        "GH_TOKEN": "bot-pat",
    }


def test_git_env_returns_none_when_token_missing():
    credentials = GitHubCredentials()
    assert credentials.git_env(GH_ROLE_BOT) is None


def test_git_env_injects_extraheader_for_github_only():
    credentials = GitHubCredentials(bot_token="bot-pat")
    env = credentials.git_env(
        GH_ROLE_BOT,
        base_env={"PATH": "/usr/bin", "HOME": "/home/bot"},
    )
    assert env is not None
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == (
        "AUTHORIZATION: basic eC1hY2Nlc3MtdG9rZW46Ym90LXBhdA=="
    )
    assert env["GH_TOKEN"] == "bot-pat"
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/bot"


def test_env_helpers_strip_unrelated_variables():
    credentials = GitHubCredentials(admin_token="admin-pat")
    env = credentials.gh_env(
        GH_ROLE_ADMIN,
        base_env={"PATH": "/usr/bin", "HOME": "/h", "SECRET": "x", "DEEPSEEK_API_KEY": "k"},
    )
    assert env is not None
    assert "SECRET" not in env
    assert "DEEPSEEK_API_KEY" not in env

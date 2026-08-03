import pytest
from tests.constants import FAKE_OFERTAS_SV_REPO_PATH, FAKE_USER_HOME

from ticket_agent.redaction import (
    SECRET_PLACEHOLDER,
    redact,
    redact_local_paths,
    redact_secrets,
)


def test_redact_local_paths_covers_repo_worktree_and_home_paths():
    repo_path = FAKE_OFERTAS_SV_REPO_PATH
    text = (
        f"Repo: {repo_path}\n"
        f"File: {repo_path}/.worktrees/LAB-55/3934cc2e/src/app.ts\n"
        f"Cache: {FAKE_USER_HOME}/.cache/vitest\n"
        r"Windows: C:\Users\jonathan\repos\ofertas-sv"
    )

    redacted = redact_local_paths(text, [repo_path])

    assert FAKE_USER_HOME not in redacted
    assert r"C:\Users\jonathan" not in redacted
    assert ".worktrees" not in redacted
    assert "File: <repo>/src/app.ts" in redacted
    assert "Cache: <home>/.cache/vitest" in redacted
    assert r"Windows: <home>\repos\ofertas-sv" in redacted


def test_redact_local_paths_ignores_filesystem_roots():
    assert redact_local_paths("relative/path.py", ["/", "C:\\"]) == (
        "relative/path.py"
    )


FAKE_SLACK_BOT_TOKEN_PARTS = ("xoxb", "123456789012", "abcdefghijklmnop")
FAKE_SLACK_BOT_TOKEN = "-".join(FAKE_SLACK_BOT_TOKEN_PARTS)


SECRET_SAMPLES = [
    ("github classic", "ghp_abcdefghij0123456789ABCDEFGHIJ"),
    ("github fine-grained", "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz0123"),
    ("openai", "sk-proj-abcdefghijklmnopqrstuvwxyz0123"),
    ("google", "AIzaSyA1234567890abcdefghijklmnopqrstuv"),
    ("slack bot", FAKE_SLACK_BOT_TOKEN),
    ("aws access key", "AKIAIOSFODNN7EXAMPLE"),
    ("aws session key", "ASIAIOSFODNN7EXAMPLE"),
]


@pytest.mark.parametrize("label,secret", SECRET_SAMPLES, ids=[s[0] for s in SECRET_SAMPLES])
def test_redact_secrets_removes_known_token_formats(label, secret):
    redacted = redact_secrets(f"value={secret} trailing")

    assert secret not in redacted
    assert SECRET_PLACEHOLDER in redacted


def test_redact_secrets_removes_pem_private_key_blocks():
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAsecretmaterial\n"
        "-----END RSA PRIVATE KEY-----"
    )

    assert "secretmaterial" not in redact_secrets(text)


def test_redact_secrets_removes_authorization_headers():
    redacted = redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6Ikp9")

    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6Ikp9" not in redacted


def test_redact_secrets_masks_values_assigned_to_secret_names():
    redacted = redact_secrets('DEEPSEEK_API_KEY = "abcdefghijklmnopqrstuvwxyz012345"')

    assert "abcdefghijklmnopqrstuvwxyz012345" not in redacted
    # The assignment stays readable so a diff remains reviewable.
    assert "DEEPSEEK_API_KEY" in redacted


@pytest.mark.parametrize(
    "benign",
    [
        "def read_file(path): return path",
        "my_key = value_from_env(name)",
        "TOKEN = short",
        "import os, sys, json",
        "assert response.status_code == 200",
    ],
)
def test_redact_secrets_leaves_ordinary_code_untouched(benign):
    """A pattern loose enough to hide real code would make reviews useless."""

    assert redact_secrets(benign) == benign


def test_redact_applies_both_filters():
    text = f"token=ghp_abcdefghij0123456789ABCDEFGHIJ at {FAKE_OFERTAS_SV_REPO_PATH}/x.ts"

    redacted = redact(text, [FAKE_OFERTAS_SV_REPO_PATH])

    assert "ghp_" not in redacted
    assert FAKE_OFERTAS_SV_REPO_PATH not in redacted
    assert "<repo>/x.ts" in redacted

"""Shared, non-personal filesystem paths for tests."""

FAKE_USER_HOME = "/home/test-user"
FAKE_AGENT_SYSTEM_REPO_PATH = f"{FAKE_USER_HOME}/repos/agent-system"
FAKE_OFERTAS_SV_REPO_PATH = f"{FAKE_USER_HOME}/repos/ofertas-sv"

__all__ = [
    "FAKE_AGENT_SYSTEM_REPO_PATH",
    "FAKE_OFERTAS_SV_REPO_PATH",
    "FAKE_USER_HOME",
]

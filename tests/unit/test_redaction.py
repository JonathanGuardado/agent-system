from ticket_agent.redaction import redact_local_paths
from tests.constants import FAKE_OFERTAS_SV_REPO_PATH, FAKE_USER_HOME


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

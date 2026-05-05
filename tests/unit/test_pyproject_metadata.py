from __future__ import annotations

import tomllib
from pathlib import Path


PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_pyproject_exposes_internal_router_smoke_script():
    pyproject = _load_pyproject()

    assert pyproject["project"]["scripts"]["ticket-agent"] == "ticket_agent.app:run"
    assert pyproject["project"]["scripts"]["ticket-agent-smoke-model-router"] == (
        "ticket_agent.router.smoke:main"
    )


def test_pyproject_does_not_depend_on_external_router_service():
    pyproject = _load_pyproject()
    dependencies = {
        dependency.split(" ", maxsplit=1)[0].lower()
        for dependency in pyproject["project"]["dependencies"]
    }

    assert "fastapi" not in dependencies
    assert "uvicorn" not in dependencies


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ticket_agent.adapters.local.file_adapter import LocalFileAdapter
from ticket_agent.orchestrator.model_services import (
    ModelRouterImplementationService,
    ModelRouterPlannerService,
    ModelRouterReviewService,
    ModelServiceError,
)
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.router import create_model_router


def main() -> int:
    loaded_env = _load_dev_environment()
    if loaded_env is None:
        print("No env file found at ~/config/agent-system.env or repo-local .env.")
    else:
        print(f"Loaded env from: {loaded_env}")

    try:
        router = create_model_router(timeout_s=120)
    except Exception as exc:
        print(f"Router construction failed: {_format_exception(exc)}", file=sys.stderr)
        return 1

    missing_env = _missing_provider_env_vars(router, os.environ)
    if missing_env:
        print(
            "Missing required provider env vars for model service smoke: "
            f"{', '.join(missing_env)}",
            file=sys.stderr,
        )
        print(
            "Add them to ~/config/agent-system.env or repo-local .env, then rerun.",
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_run_smoke(router))
    except ModelServiceError as exc:
        print(f"Model service failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Smoke failed: {_format_exception(exc)}", file=sys.stderr)
        return 1


async def _run_smoke(router: Any) -> int:
    planner = ModelRouterPlannerService(router)
    implementation = ModelRouterImplementationService(router)
    review = ModelRouterReviewService(router)

    with tempfile.TemporaryDirectory(prefix="agent-system-model-smoke-") as worktree:
        print(f"Temp worktree path: {worktree}")
        state = TicketState(
            ticket_key="SMOKE-1",
            summary="Create a hello file",
            description=(
                'Create src/hello.py with a hello() function returning "hello".'
            ),
            repository="agent-system",
            repo_path=str(REPO_ROOT),
            worktree_path=worktree,
            max_attempts=1,
        )

        planner_result = await planner.plan(state)
        print("Planner result:")
        print(_json(planner_result))
        if not _valid_planner_result(planner_result):
            print("Planner did not return a valid decomposition.", file=sys.stderr)
            return 1

        state = state.model_copy(update={"decomposition": planner_result})
        implementation_result_wrapper = await implementation.implement(state)
        print("Implementation result:")
        print(_json(implementation_result_wrapper))

        implementation_result = implementation_result_wrapper.get(
            "implementation_result"
        )
        if not isinstance(implementation_result, Mapping):
            print("Implementation did not return implementation_result.", file=sys.stderr)
            return 1

        changed_files = implementation_result.get("changed_files")
        if not isinstance(changed_files, list) or not changed_files:
            print("Implementation wrote no files.", file=sys.stderr)
            return 1

        files_written = list(LocalFileAdapter(worktree).list_files())
        print("Files written:")
        print(_json(files_written))
        if not files_written:
            print("No files were found in the temp worktree.", file=sys.stderr)
            return 1

        state = state.model_copy(
            update={
                "implementation_result": dict(implementation_result),
                "test_result": {"status": "not_run", "tests_passed": None},
            }
        )
        review_result = await review.review(state)
        print("Review result:")
        print(_json(review_result))
        if not _valid_review_result(review_result):
            print("Review did not return a valid parsed result.", file=sys.stderr)
            return 1

    return 0


def _load_dev_environment() -> Path | None:
    config_env = Path.home() / "config" / "agent-system.env"
    repo_env = REPO_ROOT / ".env"
    if config_env.exists():
        _load_env_file(config_env)
        return config_env
    if repo_env.exists():
        _load_env_file(repo_env)
        return repo_env
    return None


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _missing_provider_env_vars(router: Any, env: Mapping[str, str]) -> list[str]:
    providers = getattr(router, "_providers", {})
    defaults = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}
    required: list[str] = []
    if isinstance(providers, dict):
        for provider_name in ("deepseek", "gemini"):
            provider = providers.get(provider_name)
            api_key_env = getattr(provider, "api_key_env", None) or defaults[
                provider_name
            ]
            required.append(str(api_key_env))
    else:
        required.extend(defaults.values())

    return sorted({key for key in required if not env.get(key)})


def _valid_planner_result(result: Mapping[str, Any]) -> bool:
    return (
        isinstance(result.get("plan"), str)
        and bool(result["plan"].strip())
        and isinstance(result.get("files_to_modify"), list)
        and isinstance(result.get("risks"), list)
        and isinstance(result.get("complexity"), str)
        and isinstance(result.get("requires_human_review"), bool)
    )


def _valid_review_result(result: Mapping[str, Any]) -> bool:
    return (
        isinstance(result.get("passed"), bool)
        and isinstance(result.get("reasoning"), str)
        and isinstance(result.get("issues"), list)
    )


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _format_exception(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())

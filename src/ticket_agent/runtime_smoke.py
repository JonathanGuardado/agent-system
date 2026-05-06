"""Non-mutating smoke checks for the Slack/Jira/GitHub MVP runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from ticket_agent.app import AppConfig, StartupConfigError, load_app_config
from ticket_agent.config.repo_contract import load_repo_contract
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_EPIC_LINK,
    FIELD_MAX_ATTEMPTS,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    FIELD_SLACK_CHANNEL,
    FIELD_SLACK_THREAD_TS,
)


SmokeStatus = Literal["pass", "fail", "skip"]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class SmokeCheck:
    name: str
    status: SmokeStatus
    detail: str


_REQUIRED_JIRA_FIELD_MAP_KEYS = (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    FIELD_SLACK_CHANNEL,
    FIELD_SLACK_THREAD_TS,
)
_OPTIONAL_JIRA_FIELD_MAP_KEYS = (
    FIELD_MAX_ATTEMPTS,
    FIELD_EPIC_LINK,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    run_command: CommandRunner | None = None,
) -> int:
    args = _parse_args(argv)
    checks = asyncio.run(
        collect_smoke_checks(
            env=env,
            env_path=args.env_path,
            skip_network=args.skip_network,
            run_command=run_command,
        )
    )
    print(_format_checks(checks))
    return 1 if any(check.status == "fail" for check in checks) else 0


async def collect_smoke_checks(
    *,
    env: Mapping[str, str] | None = None,
    env_path: str | Path | None = None,
    skip_network: bool = False,
    run_command: CommandRunner | None = None,
) -> list[SmokeCheck]:
    checks: list[SmokeCheck] = []
    app_config = _load_config_check(checks, env=env, env_path=env_path)
    checks.append(_gh_auth_check(run_command or subprocess.run))

    if app_config is None:
        checks.append(
            SmokeCheck(
                "repo_contracts",
                "skip",
                "startup config failed before contract path could be resolved",
            )
        )
        checks.append(
            SmokeCheck(
                "jira_field_map",
                "skip",
                "startup config failed before Jira field map could be checked",
            )
        )
        checks.append(
            SmokeCheck(
                "network_auth",
                "skip",
                "startup config failed before auth endpoints could be checked",
            )
        )
        return checks

    checks.append(_repo_contracts_check(app_config.runtime.contract_dir))
    checks.append(_jira_field_map_check(app_config))
    checks.extend(_model_env_checks())
    if skip_network:
        checks.append(
            SmokeCheck("slack_auth", "skip", "network checks skipped")
        )
        checks.append(
            SmokeCheck("jira_auth", "skip", "network checks skipped")
        )
    else:
        checks.append(await _slack_auth_check(app_config.slack_bot_token))
        checks.append(
            await _jira_auth_check(
                app_config.jira_base_url,
                app_config.jira_user_email,
                app_config.jira_api_key,
                app_config.jira_timeout_s,
            )
        )
    return checks


def _load_config_check(
    checks: list[SmokeCheck],
    *,
    env: Mapping[str, str] | None,
    env_path: str | Path | None,
) -> AppConfig | None:
    try:
        app_config = load_app_config(env=env, env_path=env_path, install=False)
    except StartupConfigError as exc:
        checks.append(SmokeCheck("startup_config", "fail", str(exc)))
        return None
    checks.append(
        SmokeCheck(
            "startup_config",
            "pass",
            f"loaded env file: {app_config.env_file_loaded}",
        )
    )
    return app_config


def _repo_contracts_check(contract_dir: Path) -> SmokeCheck:
    if not contract_dir.exists():
        return SmokeCheck(
            "repo_contracts",
            "fail",
            f"contract directory does not exist: {contract_dir}",
        )
    contract_paths = sorted(contract_dir.glob("*.yaml"))
    if not contract_paths:
        return SmokeCheck(
            "repo_contracts",
            "fail",
            f"no repo contracts found in {contract_dir}",
        )
    try:
        for path in contract_paths:
            load_repo_contract(path)
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        return SmokeCheck(
            "repo_contracts",
            "fail",
            f"{path}: {exc}",
        )
    return SmokeCheck(
        "repo_contracts",
        "pass",
        f"loaded {len(contract_paths)} contract(s)",
    )


def _jira_field_map_check(app_config: AppConfig) -> SmokeCheck:
    configured = set(app_config.jira_field_map)
    missing = [
        field for field in _REQUIRED_JIRA_FIELD_MAP_KEYS if field not in configured
    ]
    optional_missing = [
        field for field in _OPTIONAL_JIRA_FIELD_MAP_KEYS if field not in configured
    ]
    if missing:
        return SmokeCheck(
            "jira_field_map",
            "fail",
            "missing required logical field mappings: " + ", ".join(missing),
        )
    detail = "required mappings configured"
    if optional_missing:
        detail += "; optional mappings missing: " + ", ".join(optional_missing)
    return SmokeCheck("jira_field_map", "pass", detail)


def _model_env_checks() -> list[SmokeCheck]:
    checks: list[SmokeCheck] = []
    for name in ("DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
        checks.append(
            SmokeCheck(
                name.lower(),
                "pass",
                "present in validated startup config",
            )
        )
    return checks


def _gh_auth_check(run_command: CommandRunner) -> SmokeCheck:
    try:
        result = run_command(
            ("gh", "auth", "status"),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return SmokeCheck("github_auth", "fail", "gh CLI is not installed")
    except subprocess.TimeoutExpired:
        return SmokeCheck("github_auth", "fail", "gh auth status timed out")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return SmokeCheck("github_auth", "fail", detail or "gh auth failed")
    return SmokeCheck("github_auth", "pass", "gh auth status succeeded")


async def _slack_auth_check(bot_token: str) -> SmokeCheck:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {bot_token}"},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        return SmokeCheck("slack_auth", "fail", _error_message(exc))
    if payload.get("ok") is not True:
        return SmokeCheck(
            "slack_auth",
            "fail",
            str(payload.get("error") or "Slack auth failed"),
        )
    return SmokeCheck("slack_auth", "pass", "Slack auth.test succeeded")


async def _jira_auth_check(
    base_url: str,
    user_email: str,
    api_key: str,
    timeout_s: float,
) -> SmokeCheck:
    url = base_url.rstrip("/") + "/rest/api/3/myself"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(
                url,
                auth=(user_email, api_key),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        return SmokeCheck("jira_auth", "fail", _error_message(exc))
    return SmokeCheck("jira_auth", "pass", "Jira /myself succeeded")


def _format_checks(checks: Sequence[SmokeCheck]) -> str:
    return json.dumps(
        [
            {"name": check.name, "status": check.status, "detail": check.detail}
            for check in checks
        ],
        indent=2,
        sort_keys=True,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-mutating agent-system runtime smoke checks.",
    )
    parser.add_argument("--env-path", default=None)
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip Slack and Jira auth endpoint calls.",
    )
    return parser.parse_args(argv)


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

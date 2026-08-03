"""Non-mutating smoke checks for the Slack/Jira/GitHub MVP runtime."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

import httpx

from ticket_agent.adapters.local.sandbox import BubblewrapSandbox
from ticket_agent.app import (
    AppConfig,
    RuntimeConfig,
    StartupConfigError,
    load_app_config,
)
from ticket_agent.config.repo_contract import load_repo_contract
from ticket_agent.goal.policy import load_risk_policy
from ticket_agent.goal.signing import SigningError, load_signer
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_EPIC_LINK,
    FIELD_MAX_ATTEMPTS,
    FIELD_REPO_PATH,
    FIELD_REPOSITORY,
    FIELD_SLACK_CHANNEL,
    FIELD_SLACK_THREAD_TS,
)
from ticket_agent.orchestrator.execution_environment import (
    ExecutionEnvironmentPreflight,
)
from ticket_agent.orchestrator.local_services import RuntimeShellFactory

_JIRA_PROJECT_ISSUE_TYPES_REQUIRED = frozenset({"Epic", "Task"})


SmokeStatus = Literal["pass", "fail", "skip", "warn"]
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
    runner = run_command or subprocess.run
    checks.extend(_github_auth_checks(app_config, runner))

    if app_config is None:
        _skip_all = "startup config failed before auth endpoints could be checked"
        checks.append(SmokeCheck("repo_contracts", "skip", "startup config failed before contract path could be resolved"))
        checks.append(SmokeCheck("jira_field_map", "skip", "startup config failed before Jira field map could be checked"))
        checks.append(SmokeCheck("slack_auth", "skip", _skip_all))
        checks.append(SmokeCheck("jira_auth", "skip", _skip_all))
        checks.append(SmokeCheck("jira_project_metadata", "skip", _skip_all))
        checks.append(SmokeCheck("jira_epic_link_field", "skip", _skip_all))
        return checks

    checks.append(_repo_contracts_check(app_config.runtime.contract_dir))
    checks.extend(_sandbox_checks())
    checks.append(_goal_authorization_check(app_config.runtime))
    checks.extend(_harness_readiness_checks(app_config.runtime.contract_dir))
    checks.append(_jira_field_map_check(app_config))
    checks.extend(_model_env_checks())
    if skip_network:
        checks.append(SmokeCheck("slack_auth", "skip", "network checks skipped"))
        checks.append(SmokeCheck("jira_auth", "skip", "network checks skipped"))
        checks.append(SmokeCheck("jira_project_metadata", "skip", "network checks skipped"))
        checks.append(SmokeCheck("jira_epic_link_field", "skip", "network checks skipped"))
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
        checks.extend(await _jira_metadata_checks(app_config))
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


def _sandbox_checks() -> tuple[SmokeCheck, SmokeCheck, SmokeCheck]:
    """Report capability, production configuration, and wrapper evidence.

    Attempts a real unshare rather than checking that bwrap exists. Ubuntu
    24.04 sets kernel.apparmor_restrict_unprivileged_userns=1, under which a
    perfectly healthy bwrap still cannot create a user namespace -- so a
    presence check reports such a host as sandbox-ready when it is not.
    """

    available = BubblewrapSandbox.available()
    if available:
        capability = SmokeCheck(
            "sandbox_host_capability",
            "pass",
            "bwrap can create a user namespace",
        )
    else:
        capability = SmokeCheck(
            "sandbox_host_capability",
            "warn",
            "no working bwrap user namespace; repository execution is refused. "
            "On Ubuntu 24.04 this is usually the AppArmor userns restriction.",
        )

    configured = SmokeCheck(
        "sandbox_runtime_configuration",
        "pass" if RuntimeShellFactory.requires_enforcing_sandbox else "fail",
        "all production repo-contract shell paths require live bwrap preflight",
    )

    if not available:
        evidence = SmokeCheck(
            "sandbox_command_enforcement",
            "skip",
            "host capability unavailable, so no enforcing command attestation "
            "can be produced",
        )
    else:
        try:
            attestation = RuntimeShellFactory(
                ExecutionEnvironmentPreflight()
            ).probe_attestation()
        except Exception as exc:  # noqa: BLE001 - smoke boundary
            evidence = SmokeCheck(
                "sandbox_command_enforcement",
                "fail",
                str(exc),
            )
        else:
            evidence = SmokeCheck(
                "sandbox_command_enforcement",
                "pass" if attestation.sandbox_profile == "bwrap" else "fail",
                "wrapper=bwrap "
                f"policy={attestation.command_policy_digest[:12]} "
                f"launch={attestation.launch_digest[:12]}",
            )
    return capability, configured, evidence


def _goal_authorization_check(config: RuntimeConfig) -> SmokeCheck:
    """Whether a Slack request could be authorized into a goal contract.

    Reported as `warn` rather than `fail` because the pipeline still runs
    without it -- it just cannot record what was authorized, so nothing can
    later verify a merge against it.
    """

    missing: list[str] = []
    if not getattr(config, "goal_allowlist_users", ()):
        missing.append("AGENT_SYSTEM_GOAL_ALLOWLIST_USERS is empty (nobody may authorize)")
    if getattr(config, "signing_key_path", None) is None:
        missing.append("AGENT_SYSTEM_SIGNING_KEY_PATH is unset (contracts cannot be signed)")
    else:
        try:
            load_signer(config.signing_key_path, data_dir=Path(config.data_dir))
        except SigningError as exc:
            return SmokeCheck("goal_authorization", "fail", str(exc))

    try:
        policy = load_risk_policy(config.risk_policy_path)
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        return SmokeCheck("goal_authorization", "fail", f"risk policy: {exc}")
    if not policy.repositories:
        missing.append("risk policy names no repositories (permits nothing)")

    if missing:
        return SmokeCheck("goal_authorization", "warn", "; ".join(missing))
    return SmokeCheck(
        "goal_authorization",
        "pass",
        f"policy v{policy.version} digest {policy.policy_digest[:12]}, "
        f"{len(config.goal_allowlist_users)} allowlisted user(s)",
    )


def _harness_readiness_checks(contract_dir: Path) -> list[SmokeCheck]:
    """Report each repository's readiness, which caps its autonomy."""

    if not contract_dir.exists():
        return []
    results: list[SmokeCheck] = []
    for path in sorted(contract_dir.glob("*.yaml")):
        try:
            contract = load_repo_contract(path)
            root = Path(contract.repo.root).expanduser()
            readiness, reasons = contract.readiness(root)
        except Exception as exc:  # noqa: BLE001 - smoke boundary
            results.append(
                SmokeCheck(f"harness_readiness[{path.stem}]", "fail", str(exc))
            )
            continue
        status = {"full": "pass", "partial": "warn", "unready": "warn"}[readiness]
        detail = readiness if not reasons else f"{readiness}: " + "; ".join(reasons)
        results.append(SmokeCheck(f"harness_readiness[{path.stem}]", status, detail))
    return results


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


def _github_auth_checks(
    app_config: AppConfig | None,
    run_command: CommandRunner,
) -> list[SmokeCheck]:
    if app_config is None:
        return [_gh_auth_check(run_command)]
    credentials = app_config.github_credentials()
    if not credentials.admin_token and not credentials.bot_token:
        return [_gh_auth_check(run_command)]
    checks: list[SmokeCheck] = []
    for role, name in (("admin", "github_admin_auth"), ("bot", "github_bot_auth")):
        if credentials.has_token_for(role):
            checks.append(
                _gh_token_login_check(name, credentials.gh_env(role), run_command)
            )
        else:
            checks.append(SmokeCheck(name, "skip", f"{role} token not configured"))
    return checks


def _gh_token_login_check(
    name: str,
    env: Mapping[str, str] | None,
    run_command: CommandRunner,
) -> SmokeCheck:
    try:
        result = run_command(
            ("gh", "api", "user", "--jq", ".login"),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(env) if env is not None else None,
        )
    except FileNotFoundError:
        return SmokeCheck(name, "fail", "gh CLI is not installed")
    except subprocess.TimeoutExpired:
        return SmokeCheck(name, "fail", "gh api user timed out")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return SmokeCheck(name, "fail", detail or "gh api user failed")
    login = (result.stdout or "").strip()
    if not login:
        return SmokeCheck(name, "fail", "gh api user returned empty login")
    return SmokeCheck(name, "pass", f"authenticated as {login}")


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


async def _jira_metadata_checks(app_config: AppConfig) -> list[SmokeCheck]:
    """Run Jira project-level metadata checks for each configured target project."""
    target_projects = app_config.jira_target_projects
    if not target_projects:
        return [
            SmokeCheck(
                "jira_project_metadata",
                "skip",
                "no target projects configured; set AGENT_SYSTEM_JIRA_TARGET_PROJECTS",
            ),
            SmokeCheck(
                "jira_epic_link_field",
                "skip",
                "no target projects configured",
            ),
        ]

    checks: list[SmokeCheck] = []
    for project_key in target_projects:
        checks.extend(
            await _jira_project_checks(
                app_config.jira_base_url,
                app_config.jira_user_email,
                app_config.jira_api_key,
                app_config.jira_timeout_s,
                project_key,
            )
        )
    checks.append(
        await _jira_epic_link_field_check(
            app_config.jira_base_url,
            app_config.jira_user_email,
            app_config.jira_api_key,
            app_config.jira_timeout_s,
            app_config.jira_field_map,
        )
    )
    return checks


async def _jira_project_checks(
    base_url: str,
    user_email: str,
    api_key: str,
    timeout_s: float,
    project_key: str,
) -> list[SmokeCheck]:
    """Verify a Jira project exists and has the required issue types."""
    url = base_url.rstrip("/") + f"/rest/api/3/project/{project_key}"
    proj_name = f"jira_project_{project_key.lower()}"
    types_name = f"jira_issue_types_{project_key.lower()}"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(
                url,
                auth=(user_email, api_key),
                headers={"Accept": "application/json"},
            )
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        msg = _error_message(exc)
        return [
            SmokeCheck(proj_name, "fail", msg),
            SmokeCheck(types_name, "skip", f"{proj_name} failed"),
        ]

    if response.status_code == 404:
        return [
            SmokeCheck(proj_name, "fail", f"project {project_key} not found in Jira"),
            SmokeCheck(types_name, "skip", f"project {project_key} not found"),
        ]

    try:
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        return [
            SmokeCheck(proj_name, "fail", _error_message(exc)),
            SmokeCheck(types_name, "skip", f"{proj_name} failed"),
        ]

    issue_types = data.get("issueTypes") or []
    present = {t.get("name", "") for t in issue_types if isinstance(t, dict)}
    missing = sorted(_JIRA_PROJECT_ISSUE_TYPES_REQUIRED - present)

    project_check = SmokeCheck(proj_name, "pass", f"project {project_key} found")
    if missing:
        types_check = SmokeCheck(
            types_name,
            "fail",
            f"project {project_key} missing issue types: {', '.join(missing)}",
        )
    else:
        types_check = SmokeCheck(
            types_name,
            "pass",
            f"project {project_key} has Epic and Task issue types",
        )
    return [project_check, types_check]


async def _jira_epic_link_field_check(
    base_url: str,
    user_email: str,
    api_key: str,
    timeout_s: float,
    field_map: Mapping[str, str],
) -> SmokeCheck:
    """Verify the configured Epic Link field ID exists in Jira, or skip if unconfigured."""
    epic_link_field_id = field_map.get(FIELD_EPIC_LINK)
    if not epic_link_field_id:
        return SmokeCheck(
            "jira_epic_link_field",
            "skip",
            "FIELD_EPIC_LINK not configured; parent key approach will be used",
        )

    url = base_url.rstrip("/") + "/rest/api/3/field"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(
                url,
                auth=(user_email, api_key),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            fields = response.json()
    except Exception as exc:  # noqa: BLE001 - smoke boundary
        return SmokeCheck("jira_epic_link_field", "fail", _error_message(exc))

    if not isinstance(fields, list):
        return SmokeCheck("jira_epic_link_field", "fail", "Jira /field response was not a list")

    field_ids = {f.get("id", "") for f in fields if isinstance(f, dict)}
    if epic_link_field_id not in field_ids:
        return SmokeCheck(
            "jira_epic_link_field",
            "fail",
            f"Epic Link field {epic_link_field_id!r} not found in Jira fields",
        )
    return SmokeCheck(
        "jira_epic_link_field",
        "pass",
        f"Epic Link field {epic_link_field_id!r} found",
    )


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

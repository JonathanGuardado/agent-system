"""Repo contract loading for execution-safe local commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ticket_agent.domain.errors import RepoContractError


DEFAULT_PROTECTED_PATHS = (
    ".github/",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env",
    "secrets/",
)


@dataclass(frozen=True)
class RepoInfo:
    """Repository identity and checkout defaults."""

    name: str
    root: str
    default_branch: str


@dataclass(frozen=True)
class LanguageInfo:
    """Primary language and package manager for the repository."""

    primary: str
    package_manager: str


@dataclass(frozen=True)
class CommandSpec:
    """A structured argv command declared by the target repository."""

    command: tuple[str, ...]
    timeout_seconds: int
    working_directory: str


@dataclass(frozen=True)
class RepoCommands:
    """Execution commands allowed by the repository contract."""

    test: CommandSpec
    lint: CommandSpec | None
    install: CommandSpec | None


@dataclass(frozen=True)
class ExecutionPolicy:
    """Policy values future local adapters must enforce."""

    dependency_install_allowed: bool
    config_paths_allowed: tuple[str, ...]
    protected_paths: tuple[str, ...]


@dataclass(frozen=True)
class RepoContract:
    """Contract declared by a repository for safe local execution."""

    repo: RepoInfo
    language: LanguageInfo
    commands: RepoCommands
    policy: ExecutionPolicy
    source_dirs: tuple[str, ...]
    test_dirs: tuple[str, ...]

    def test_command(self, suite: str = "default") -> CommandSpec:
        """Return the default test command for current local adapter compatibility."""

        if suite != "default":
            raise RepoContractError(
                "repo contract only declares the default test command"
            )
        return self.commands.test


def load_repo_contract(path: str | Path) -> RepoContract:
    """Load and validate a repo contract YAML file."""

    contract_path = Path(path)
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RepoContractError("repo contract must be a YAML mapping")

    repo = _parse_repo(raw.get("repo"))
    language = _parse_language(raw.get("language"))
    commands = _parse_commands(raw.get("commands"))
    policy = _parse_policy(raw.get("policy", {}))
    source_dirs = _parse_string_list(
        raw.get("source_dirs"),
        "source_dirs",
        required=True,
        allow_empty=False,
    )
    test_dirs = _parse_string_list(
        raw.get("test_dirs", ["tests/"]),
        "test_dirs",
        required=False,
        allow_empty=False,
    )

    return RepoContract(
        repo=repo,
        language=language,
        commands=commands,
        policy=policy,
        source_dirs=source_dirs,
        test_dirs=test_dirs,
    )


def _parse_repo(raw: Any) -> RepoInfo:
    if not isinstance(raw, dict):
        raise RepoContractError("repo must be a mapping")
    return RepoInfo(
        name=_parse_required_string(raw.get("name"), "repo.name"),
        root=_parse_required_string(raw.get("root"), "repo.root"),
        default_branch=_parse_required_string(
            raw.get("default_branch"), "repo.default_branch"
        ),
    )


def _parse_language(raw: Any) -> LanguageInfo:
    if not isinstance(raw, dict):
        raise RepoContractError("language must be a mapping")
    return LanguageInfo(
        primary=_parse_required_string(raw.get("primary"), "language.primary"),
        package_manager=_parse_required_string(
            raw.get("package_manager"), "language.package_manager"
        ),
    )


def _parse_commands(raw: Any) -> RepoCommands:
    if not isinstance(raw, dict):
        raise RepoContractError("commands must be a mapping")
    if "test" not in raw:
        raise RepoContractError("commands.test is required")
    return RepoCommands(
        test=_parse_command_spec(raw.get("test"), "commands.test"),
        lint=_parse_optional_command_spec(raw.get("lint"), "commands.lint"),
        install=_parse_optional_command_spec(raw.get("install"), "commands.install"),
    )


def _parse_optional_command_spec(raw: Any, label: str) -> CommandSpec | None:
    if raw is None:
        return None
    return _parse_command_spec(raw, label)


def _parse_command_spec(raw: Any, label: str) -> CommandSpec:
    if not isinstance(raw, dict):
        raise RepoContractError(f"{label} must be a mapping or null")

    command = _parse_command(raw.get("command"), f"{label}.command")
    timeout_seconds = raw.get("timeout_seconds", 300)
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise RepoContractError(
            f"{label}.timeout_seconds must be a positive integer"
        )

    working_directory = raw.get("working_directory", ".")
    if not isinstance(working_directory, str) or not working_directory:
        raise RepoContractError(
            f"{label}.working_directory must be a non-empty string"
        )

    return CommandSpec(
        command=command,
        timeout_seconds=timeout_seconds,
        working_directory=working_directory,
    )


def _parse_policy(raw: Any) -> ExecutionPolicy:
    if not isinstance(raw, dict):
        raise RepoContractError("policy must be a mapping")

    dependency_install_allowed = raw.get("dependency_install_allowed", False)
    if not isinstance(dependency_install_allowed, bool):
        raise RepoContractError("policy.dependency_install_allowed must be a boolean")

    config_paths_allowed = _parse_string_list(
        raw.get("config_paths_allowed", []),
        "policy.config_paths_allowed",
        required=False,
        allow_empty=True,
    )
    protected_paths = _parse_string_list(
        raw.get("protected_paths", list(DEFAULT_PROTECTED_PATHS)),
        "policy.protected_paths",
        required=False,
        allow_empty=True,
    )

    return ExecutionPolicy(
        dependency_install_allowed=dependency_install_allowed,
        config_paths_allowed=config_paths_allowed,
        protected_paths=protected_paths,
    )


def _parse_command(raw: Any, label: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        raise RepoContractError(
            f"{label} must be a structured argv list, not a string"
        )
    if not isinstance(raw, list):
        raise RepoContractError(f"{label} must be a non-empty list of strings")
    if not raw:
        raise RepoContractError(f"{label} must not be empty")
    if not all(isinstance(part, str) and part for part in raw):
        raise RepoContractError(f"{label} parts must be non-empty strings")
    return tuple(raw)


def _parse_string_list(
    raw: Any,
    label: str,
    *,
    required: bool,
    allow_empty: bool,
) -> tuple[str, ...]:
    if raw is None and required:
        raise RepoContractError(f"{label} is required")
    if not isinstance(raw, list):
        raise RepoContractError(f"{label} must be a list of non-empty strings")
    if not raw and not allow_empty:
        raise RepoContractError(f"{label} must not be empty")
    if not all(isinstance(item, str) and item for item in raw):
        raise RepoContractError(f"{label} entries must be non-empty strings")
    return tuple(raw)


def _parse_required_string(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise RepoContractError(f"{label} must be a non-empty string")
    return raw

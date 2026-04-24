"""Repo contract loading for execution-safe test commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ticket_agent.domain.errors import RepoContractError


@dataclass(frozen=True)
class TestCommand:
    """A named test command declared by the target repository."""

    name: str
    command: tuple[str, ...]
    timeout_seconds: int
    working_directory: str = "."


@dataclass(frozen=True)
class RepoContract:
    """Contract declared by a repository for safe local execution."""

    test_commands: Mapping[str, TestCommand]
    shell_allowlist: tuple[tuple[str, ...], ...]

    def test_command(self, suite: str = "default") -> TestCommand:
        try:
            return self.test_commands[suite]
        except KeyError as exc:
            available = ", ".join(sorted(self.test_commands)) or "<none>"
            raise RepoContractError(
                f"unknown test suite {suite!r}; available suites: {available}"
            ) from exc


def load_repo_contract(path: str | Path) -> RepoContract:
    """Load and validate a repo contract YAML file."""

    contract_path = Path(path)
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RepoContractError("repo contract must be a YAML mapping")

    test_commands = _parse_test_commands(raw.get("test_commands"))
    shell_allowlist = _parse_shell_allowlist(raw.get("shell_allowlist", ()))
    return RepoContract(test_commands=test_commands, shell_allowlist=shell_allowlist)


def _parse_test_commands(raw: Any) -> Mapping[str, TestCommand]:
    if not isinstance(raw, dict) or not raw:
        raise RepoContractError("repo contract requires non-empty test_commands mapping")

    parsed: dict[str, TestCommand] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not name:
            raise RepoContractError("test command names must be non-empty strings")
        if not isinstance(entry, dict):
            raise RepoContractError(f"test command {name!r} must be a mapping")

        command = _parse_command(entry.get("command"), f"test command {name!r}")
        timeout_seconds = entry.get("timeout_seconds", 300)
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise RepoContractError(
                f"test command {name!r} timeout_seconds must be a positive integer"
            )

        working_directory = entry.get("working_directory", ".")
        if not isinstance(working_directory, str) or not working_directory:
            raise RepoContractError(
                f"test command {name!r} working_directory must be a non-empty string"
            )

        parsed[name] = TestCommand(
            name=name,
            command=command,
            timeout_seconds=timeout_seconds,
            working_directory=working_directory,
        )

    return parsed


def _parse_shell_allowlist(raw: Any) -> tuple[tuple[str, ...], ...]:
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list):
        raise RepoContractError("shell_allowlist must be a list of command prefixes")
    return tuple(_parse_command(entry, "shell_allowlist entry") for entry in raw)


def _parse_command(raw: Any, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise RepoContractError(f"{label} command must be a non-empty list of strings")
    if not all(isinstance(part, str) and part for part in raw):
        raise RepoContractError(f"{label} command parts must be non-empty strings")
    return tuple(raw)

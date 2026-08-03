"""Repo contract loading for execution-safe local commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from ticket_agent.config.trust_root import (
    TrustRootClosure,
    TrustRootEntry,
    parse_trust_root,
    resolve_closure,
)
from ticket_agent.domain.errors import RepoContractError
from ticket_agent.domain.execution import CommandNetworkMode

_LOGGER = logging.getLogger(__name__)


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
    writable_paths: tuple[str, ...] = ()
    network: CommandNetworkMode = "none"


@dataclass(frozen=True)
class RepoCommands:
    """Execution commands allowed by the repository contract.

    ``typecheck`` and ``build`` are separate gates, not one "build" step: a
    type error and a bundling failure demand different fixes, and collapsing
    them loses the more precise diagnostic. Both default to ``None`` and are
    declared last so the six test modules that build this by keyword keep
    working unchanged.
    """

    test: CommandSpec
    lint: CommandSpec | None
    install: CommandSpec | None
    typecheck: CommandSpec | None = None
    build: CommandSpec | None = None

    def gate_commands(self) -> dict[str, CommandSpec]:
        """Declared gate commands, keyed by gate name."""

        declared = {
            "test": self.test,
            "lint": self.lint,
            "typecheck": self.typecheck,
            "build": self.build,
            "install": self.install,
        }
        return {name: spec for name, spec in declared.items() if spec is not None}


@dataclass(frozen=True)
class ExecutionPolicy:
    """Policy values future local adapters must enforce."""

    dependency_install_allowed: bool
    config_paths_allowed: tuple[str, ...]
    protected_paths: tuple[str, ...]


GateRequirement = Literal["required", "optional"]

#: Gate requirements when no ``gates:`` block is declared. ``test`` is
#: required because a repository that cannot state what "working" means
#: cannot be verified at all.
DEFAULT_GATES: Mapping[str, GateRequirement] = {
    "test": "required",
    "lint": "optional",
    "typecheck": "optional",
    "build": "optional",
}


@dataclass(frozen=True)
class RepoContract:
    """Contract declared by a repository for safe local execution."""

    repo: RepoInfo
    language: LanguageInfo
    commands: RepoCommands
    policy: ExecutionPolicy
    source_dirs: tuple[str, ...]
    test_dirs: tuple[str, ...]
    gates: Mapping[str, GateRequirement] = field(default_factory=lambda: DEFAULT_GATES)
    trust_root: tuple[TrustRootEntry, ...] = ()

    @property
    def required_gates(self) -> tuple[str, ...]:
        return tuple(
            name for name, req in self.gates.items() if req == "required"
        )

    @property
    def optional_gates(self) -> tuple[str, ...]:
        return tuple(
            name for name, req in self.gates.items() if req == "optional"
        )

    def trust_root_closure(self, repo_root: Path | None = None) -> TrustRootClosure:
        """Resolve ``derived`` trust-root entries against the gate commands."""

        return resolve_closure(
            self.trust_root,
            commands={
                name: spec.command
                for name, spec in self.commands.gate_commands().items()
            },
            repo_root=repo_root,
        )

    def readiness(self, repo_root: Path | None = None) -> tuple[str, tuple[str, ...]]:
        """Harness readiness and the reasons holding it back.

        Readiness is a ceiling on autonomy, not a warning: a repository that
        cannot describe how it is verified must not be driven unattended.
        """

        reasons: list[str] = []

        if repo_root is not None and not Path(repo_root).exists():
            reasons.append(f"repository root does not exist: {repo_root}")

        if not self.required_gates:
            reasons.append("no required gates declared")

        for gate in self.required_gates:
            if self.commands.gate_commands().get(gate) is None:
                reasons.append(f"gate {gate!r} is required but no command is declared")

        for gate, spec in self.commands.gate_commands().items():
            if _is_vacuous_command(spec.command):
                reasons.append(
                    f"gate {gate!r} can succeed without running anything"
                )

        closure = self.trust_root_closure(repo_root)
        reasons.extend(f"trust root unresolved -- {item}" for item in closure.unresolved)

        if not closure.entries:
            reasons.append("no trust root declared")

        if not reasons:
            return "full", ()
        # An unresolved trust root or a vacuous gate is a hole in the
        # evidence, not an absence of tooling: cap harder than a missing
        # optional gate would.
        blocking = any(
            "unresolved" in reason
            or "without running anything" in reason
            or "does not exist" in reason
            or "no required gates" in reason
            for reason in reasons
        )
        return ("unready" if blocking else "partial"), tuple(reasons)

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
        gates=_parse_gates(raw.get("gates"), commands),
        trust_root=parse_trust_root(raw.get("trust_root")),
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


def _is_vacuous_command(argv: Sequence[str]) -> bool:
    """Whether a command can report success without verifying anything.

    Two shapes matter in practice. A bare ``echo`` is a placeholder that
    always exits 0. And a shell guard whose else-branch merely echoes --
    ``if [ -f package.json ]; then npm test; else echo 'no package.json'; fi``
    -- exits 0 on a repository with no application at all, which is exactly
    the state a greenfield goal starts in.
    """

    if not argv:
        return True
    if Path(argv[0]).name == "echo":
        return True
    joined = " ".join(argv)
    if "else" in joined and "exit 1" not in joined and "echo" in joined:
        return True
    return "--if-present" in joined


def _parse_gates(raw: Any, commands: RepoCommands) -> Mapping[str, GateRequirement]:
    if raw is None:
        return DEFAULT_GATES
    if not isinstance(raw, dict):
        raise RepoContractError("gates must be a mapping")

    gates: dict[str, GateRequirement] = {}
    for name, requirement in raw.items():
        if name == "install":
            raise RepoContractError(
                "gates.install is not declarable; installation is governed by "
                "policy.dependency_install_allowed"
            )
        if name not in DEFAULT_GATES:
            raise RepoContractError(f"unknown gate: {name!r}")
        if requirement not in ("required", "optional"):
            raise RepoContractError(
                f"gates.{name} must be 'required' or 'optional'"
            )
        gates[name] = requirement

    if gates.get("test") == "optional":
        raise RepoContractError(
            "gates.test cannot be optional; a contract that need not run its "
            "tests cannot authorize anything"
        )
    for name, requirement in gates.items():
        if requirement == "required" and commands.gate_commands().get(name) is None:
            raise RepoContractError(
                f"gates.{name} is required but commands.{name} is not declared"
            )
    return {**DEFAULT_GATES, **gates}


def _parse_commands(raw: Any) -> RepoCommands:
    if not isinstance(raw, dict):
        raise RepoContractError("commands must be a mapping")
    if "test" not in raw:
        raise RepoContractError("commands.test is required")
    return RepoCommands(
        test=_parse_command_spec(raw.get("test"), "commands.test"),
        lint=_parse_optional_command_spec(raw.get("lint"), "commands.lint"),
        install=_parse_optional_command_spec(raw.get("install"), "commands.install"),
        typecheck=_parse_optional_command_spec(
            raw.get("typecheck"), "commands.typecheck"
        ),
        build=_parse_optional_command_spec(raw.get("build"), "commands.build"),
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
    _validate_relative_path(working_directory, f"{label}.working_directory")

    if "writable_paths" not in raw:
        raise RepoContractError(f"{label}.writable_paths is required")
    writable_paths = _parse_string_list(
        raw.get("writable_paths"),
        f"{label}.writable_paths",
        required=True,
        allow_empty=True,
    )
    for index, writable_path in enumerate(writable_paths):
        _validate_relative_path(
            writable_path,
            f"{label}.writable_paths[{index}]",
        )

    if "network" not in raw:
        raise RepoContractError(f"{label}.network is required")
    network = raw.get("network")
    if network not in ("none", "install"):
        raise RepoContractError(f"{label}.network must be 'none' or 'install'")
    if network == "install" and label != "commands.install":
        raise RepoContractError(
            f"{label}.network may be 'install' only for commands.install"
        )

    return CommandSpec(
        command=command,
        timeout_seconds=timeout_seconds,
        working_directory=working_directory,
        writable_paths=writable_paths,
        network=network,
    )


def _validate_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RepoContractError(f"{label} must stay within the repository root")


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


# ---------------------------------------------------------------------------
# Auto-scaffolding
# ---------------------------------------------------------------------------


def scaffold_repo_contract(
    *,
    repo_name: str,
    repo_path: str | None,
    contract_path: Path,
) -> RepoContract:
    """Generate a minimal repo contract, write it to contract_path, and return it.

    Detects language from the repo on disk when it already exists; falls back
    to Python/pip defaults so brand-new repos get a usable starting point.
    """
    effective_root = repo_path or f"~/repos/{repo_name}"
    resolved_root = Path(effective_root).expanduser()

    language, package_manager = _detect_language(resolved_root)
    test_cmd = _scaffold_test_command(language, package_manager)
    lint_cmd = _scaffold_lint_command(language)
    source_dirs, test_dirs = _scaffold_dirs(language)

    contract = RepoContract(
        repo=RepoInfo(
            name=repo_name,
            root=effective_root,
            default_branch="main",
        ),
        language=LanguageInfo(
            primary=language,
            package_manager=package_manager,
        ),
        commands=RepoCommands(
            test=test_cmd,
            lint=lint_cmd,
            install=None,
        ),
        policy=ExecutionPolicy(
            dependency_install_allowed=False,
            config_paths_allowed=(),
            protected_paths=DEFAULT_PROTECTED_PATHS,
        ),
        source_dirs=source_dirs,
        test_dirs=test_dirs,
    )

    contract_path.parent.mkdir(parents=True, exist_ok=True)
    _write_contract_yaml(contract, contract_path)
    _LOGGER.warning(
        "auto-scaffolded repo contract for %r at %s — review and edit as needed",
        repo_name,
        contract_path,
    )
    return contract


def _detect_language(repo_root: Path) -> tuple[str, str]:
    if not repo_root.exists():
        return "python", "pip"
    if (repo_root / "package.json").exists():
        if (repo_root / "pnpm-lock.yaml").exists():
            return "javascript", "pnpm"
        if (repo_root / "yarn.lock").exists():
            return "javascript", "yarn"
        return "javascript", "npm"
    if (repo_root / "pyproject.toml").exists():
        return "python", "poetry"
    if (repo_root / "setup.py").exists() or (repo_root / "requirements.txt").exists():
        return "python", "pip"
    if (repo_root / "go.mod").exists():
        return "go", "go"
    if (repo_root / "Cargo.toml").exists():
        return "rust", "cargo"
    return "python", "pip"


def _scaffold_test_command(language: str, package_manager: str) -> CommandSpec:
    if language == "python":
        return CommandSpec(
            command=("python", "-m", "pytest", "tests/", "-x", "-q"),
            timeout_seconds=120,
            working_directory=".",
            writable_paths=(".pytest_cache",),
            network="none",
        )
    if language == "javascript":
        return CommandSpec(
            command=(package_manager, "test"),
            timeout_seconds=120,
            working_directory=".",
            writable_paths=("node_modules/.cache",),
            network="none",
        )
    if language == "go":
        return CommandSpec(
            command=("go", "test", "./..."),
            timeout_seconds=120,
            working_directory=".",
            writable_paths=(),
            network="none",
        )
    if language == "rust":
        return CommandSpec(
            command=("cargo", "test"),
            timeout_seconds=300,
            working_directory=".",
            writable_paths=("target",),
            network="none",
        )
    return CommandSpec(
        command=("echo", "no-tests-configured"),
        timeout_seconds=10,
        working_directory=".",
        writable_paths=(),
        network="none",
    )


def _scaffold_lint_command(language: str) -> CommandSpec | None:
    if language == "python":
        return CommandSpec(
            command=("python", "-m", "ruff", "check", "src/"),
            timeout_seconds=60,
            working_directory=".",
            writable_paths=(".ruff_cache",),
            network="none",
        )
    return None


def _scaffold_dirs(language: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if language == "go":
        return ((".",), (".",))
    return (("src/",), ("tests/",))


def _write_contract_yaml(contract: RepoContract, path: Path) -> None:
    lines = [
        "# Auto-scaffolded by agent-system — review and edit as needed.",
        "repo:",
        f"  name: {contract.repo.name}",
        f"  root: {contract.repo.root}",
        f"  default_branch: {contract.repo.default_branch}",
        "",
        "language:",
        f"  primary: {contract.language.primary}",
        f"  package_manager: {contract.language.package_manager}",
        "",
        "commands:",
        "  test:",
        f"    command: {list(contract.commands.test.command)}",
        f"    timeout_seconds: {contract.commands.test.timeout_seconds}",
        f"    working_directory: \"{contract.commands.test.working_directory}\"",
        f"    writable_paths: {list(contract.commands.test.writable_paths)}",
        f"    network: {contract.commands.test.network}",
    ]
    if contract.commands.lint:
        lines += [
            "  lint:",
            f"    command: {list(contract.commands.lint.command)}",
            f"    timeout_seconds: {contract.commands.lint.timeout_seconds}",
            f"    working_directory: \"{contract.commands.lint.working_directory}\"",
            f"    writable_paths: {list(contract.commands.lint.writable_paths)}",
            f"    network: {contract.commands.lint.network}",
        ]
    else:
        lines.append("  lint: null")
    lines.append("  install: null")
    lines += [
        "",
        "policy:",
        "  dependency_install_allowed: false",
        "  config_paths_allowed: []",
        "  protected_paths:",
    ]
    for p in contract.policy.protected_paths:
        lines.append(f"    - {p}")
    lines += [
        "",
        "source_dirs:",
    ]
    for d in contract.source_dirs:
        lines.append(f"  - {d}")
    lines += [
        "",
        "test_dirs:",
    ]
    for d in contract.test_dirs:
        lines.append(f"  - {d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

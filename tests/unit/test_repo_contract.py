from __future__ import annotations

import pytest

from ticket_agent.config.repo_contract import (
    DEFAULT_PROTECTED_PATHS,
    RepoContractError,
    load_repo_contract,
)


def _write_contract(tmp_path, body: str):
    contract_path = tmp_path / "ticket-agent.yaml"
    contract_path.write_text(body, encoding="utf-8")
    return contract_path


def _valid_contract(**overrides: str) -> str:
    values = {
        "commands": """
  test:
    command: ["python", "-m", "pytest", "tests/", "-x", "-q"]
    timeout_seconds: 120
    working_directory: "."
  lint:
    command: ["python", "-m", "ruff", "check", "src/"]
    timeout_seconds: 60
    working_directory: "."
  install: null
""",
        "policy": """
  dependency_install_allowed: false
  config_paths_allowed:
    - pyproject.toml
  protected_paths:
    - .github/
    - Dockerfile
    - docker-compose.yml
    - .env
    - secrets/
""",
        "source_dirs": """
  - src/
""",
        "test_dirs": """
  - tests/
""",
    }
    values.update(overrides)
    return f"""
repo:
  name: my-project
  root: ~/repos/my-project
  default_branch: main

language:
  primary: python
  package_manager: poetry

commands:
{values["commands"]}
policy:
{values["policy"]}
source_dirs:
{values["source_dirs"]}
test_dirs:
{values["test_dirs"]}
"""


def test_loads_valid_new_schema(tmp_path):
    contract = load_repo_contract(_write_contract(tmp_path, _valid_contract()))

    assert contract.repo.name == "my-project"
    assert contract.repo.root == "~/repos/my-project"
    assert contract.repo.default_branch == "main"
    assert contract.language.primary == "python"
    assert contract.language.package_manager == "poetry"
    assert contract.commands.test.command == (
        "python",
        "-m",
        "pytest",
        "tests/",
        "-x",
        "-q",
    )
    assert contract.commands.test.timeout_seconds == 120
    assert contract.commands.test.working_directory == "."
    assert contract.commands.lint is not None
    assert contract.commands.lint.command == (
        "python",
        "-m",
        "ruff",
        "check",
        "src/",
    )
    assert contract.commands.install is None
    assert contract.policy.dependency_install_allowed is False
    assert contract.policy.config_paths_allowed == ("pyproject.toml",)
    assert contract.policy.protected_paths == (
        ".github/",
        "Dockerfile",
        "docker-compose.yml",
        ".env",
        "secrets/",
    )
    assert contract.source_dirs == ("src/",)
    assert contract.test_dirs == ("tests/",)


def test_rejects_string_command(tmp_path):
    commands = """
  test:
    command: "pytest tests/"
    timeout_seconds: 120
    working_directory: "."
"""
    contract_path = _write_contract(tmp_path, _valid_contract(commands=commands))

    with pytest.raises(RepoContractError, match="structured argv list"):
        load_repo_contract(contract_path)


def test_rejects_missing_commands_test(tmp_path):
    commands = """
  lint: null
  install: null
"""
    contract_path = _write_contract(tmp_path, _valid_contract(commands=commands))

    with pytest.raises(RepoContractError, match="commands.test is required"):
        load_repo_contract(contract_path)


def test_rejects_empty_source_dirs(tmp_path):
    contract_path = _write_contract(tmp_path, _valid_contract(source_dirs=" []"))

    with pytest.raises(RepoContractError, match="source_dirs must not be empty"):
        load_repo_contract(contract_path)


def test_applies_protected_path_defaults(tmp_path):
    policy = """
  dependency_install_allowed: false
"""
    contract = load_repo_contract(
        _write_contract(tmp_path, _valid_contract(policy=policy))
    )

    assert contract.policy.protected_paths == DEFAULT_PROTECTED_PATHS


def test_applies_dependency_install_allowed_default_false(tmp_path):
    policy = """
  config_paths_allowed: []
"""
    contract = load_repo_contract(
        _write_contract(tmp_path, _valid_contract(policy=policy))
    )

    assert contract.policy.dependency_install_allowed is False


def test_loads_config_paths_allowed(tmp_path):
    policy = """
  config_paths_allowed:
    - pyproject.toml
    - poetry.lock
"""
    contract = load_repo_contract(
        _write_contract(tmp_path, _valid_contract(policy=policy))
    )

    assert contract.policy.config_paths_allowed == ("pyproject.toml", "poetry.lock")


def test_applies_test_dirs_default(tmp_path):
    contract_path = _write_contract(
        tmp_path,
        """
repo:
  name: my-project
  root: ~/repos/my-project
  default_branch: main

language:
  primary: python
  package_manager: poetry

commands:
  test:
    command: ["python", "-m", "pytest"]
    timeout_seconds: 120
    working_directory: "."

source_dirs:
  - src/
""",
    )

    contract = load_repo_contract(contract_path)

    assert contract.test_dirs == ("tests/",)


def test_rejects_command_with_non_string_part(tmp_path):
    commands = """
  test:
    command: ["python", "-m", "pytest", 3]
    timeout_seconds: 120
    working_directory: "."
"""
    contract_path = _write_contract(tmp_path, _valid_contract(commands=commands))

    with pytest.raises(RepoContractError, match="parts must be non-empty strings"):
        load_repo_contract(contract_path)


def test_rejects_empty_command_list(tmp_path):
    commands = """
  test:
    command: []
    timeout_seconds: 120
    working_directory: "."
"""
    contract_path = _write_contract(tmp_path, _valid_contract(commands=commands))

    with pytest.raises(RepoContractError, match="must not be empty"):
        load_repo_contract(contract_path)


def test_rejects_invalid_yaml_root_type(tmp_path):
    contract_path = _write_contract(tmp_path, "- not\n- a\n- mapping\n")

    with pytest.raises(
        RepoContractError,
        match="repo contract must be a YAML mapping",
    ):
        load_repo_contract(contract_path)

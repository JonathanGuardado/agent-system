"""Tests for trust-root declaration, closure, and harness readiness."""

from __future__ import annotations

from pathlib import Path

import pytest

from ticket_agent.config.trust_root import (
    TrustRootEntry,
    json_pointers_changed,
    parse_trust_root,
    resolve_closure,
)
from ticket_agent.domain.errors import RepoContractError


# -- matching --------------------------------------------------------------


def test_tree_entry_covers_paths_beneath_it():
    entry = TrustRootEntry(kind="tree", path=".github/")

    assert entry.covers(".github/workflows/ci.yml")
    assert entry.covers(".github")
    assert not entry.covers(".githubbed/x.yml")
    assert not entry.covers("src/app.py")


def test_dotfile_paths_match_their_entries():
    """Regression: lstrip('./') strips characters, not a prefix.

    It turned '.github/workflows/ci.yml' into 'github/workflows/ci.yml', so
    every dotfile trust-root entry silently failed to match -- and dotfiles
    are precisely the paths that carry CI and delivery authority.
    """

    assert TrustRootEntry(kind="tree", path=".github/").covers(
        ".github/workflows/release.yml"
    )
    assert TrustRootEntry(kind="file", path=".env.example").covers(".env.example")
    assert TrustRootEntry(kind="file", path="CODEOWNERS").covers("./CODEOWNERS")


def test_json_pointer_entry_matches_the_file_so_the_caller_can_compare_keys():
    entry = TrustRootEntry(
        kind="json_pointer", path="package.json", pointers=("/scripts",)
    )

    # Path matching cannot tell which key changed; answering False here would
    # let a scripts.test edit through unnoticed.
    assert entry.covers("package.json")


# -- parsing ---------------------------------------------------------------


def test_parse_rejects_incomplete_entries():
    with pytest.raises(RepoContractError):
        parse_trust_root([{"kind": "file"}])
    with pytest.raises(RepoContractError):
        parse_trust_root([{"kind": "json_pointer", "path": "package.json"}])
    with pytest.raises(RepoContractError):
        parse_trust_root([{"kind": "derived"}])
    with pytest.raises(RepoContractError):
        parse_trust_root([{"kind": "nonsense", "path": "x"}])


def test_parse_accepts_a_full_declaration():
    entries = parse_trust_root(
        [
            {"kind": "file", "path": "CODEOWNERS"},
            {"kind": "tree", "path": ".github/"},
            {"kind": "json_pointer", "path": "package.json", "pointers": ["/scripts"]},
            {"kind": "derived", "source": "verification_commands"},
        ]
    )

    assert [e.kind for e in entries] == ["file", "tree", "json_pointer", "derived"]


# -- closure ---------------------------------------------------------------


def test_closure_derives_the_script_a_command_delegates_to():
    closure = resolve_closure(
        [TrustRootEntry(kind="derived", source="verification_commands")],
        commands={"test": ("npm", "run", "test"), "build": ("npm", "run", "build")},
    )

    pointers = {p for e in closure.entries for p in e.pointers}
    assert "/scripts/test" in pointers
    assert "/scripts/build" in pointers
    assert closure.complete


def test_shell_commands_are_unresolvable_and_say_so():
    """A shell program can compute its target at runtime.

    The honest answer is to report the hole, not to digest a closure that
    silently omits whatever the script actually runs.
    """

    closure = resolve_closure(
        [TrustRootEntry(kind="derived", source="verification_commands")],
        commands={"test": ("bash", "-lc", "if [ -f package.json ]; then npm test; fi")},
    )

    assert not closure.complete
    assert any("bash" in reason for reason in closure.unresolved)


def test_package_manager_subcommands_are_not_holes():
    """`npm ci` runs the manager, not a repo script -- nothing to resolve."""

    closure = resolve_closure(
        [TrustRootEntry(kind="derived", source="verification_commands")],
        commands={"install": ("npm", "ci", "--ignore-scripts")},
    )

    assert closure.complete


def test_system_binaries_carry_no_in_repo_authority():
    closure = resolve_closure(
        [TrustRootEntry(kind="derived", source="verification_commands")],
        commands={"typecheck": ("npx", "--no-install", "tsc", "--noEmit")},
    )

    assert closure.complete


def test_touched_by_reports_only_trust_root_paths():
    closure = resolve_closure(
        [
            TrustRootEntry(kind="tree", path=".github/"),
            TrustRootEntry(kind="file", path="package.json"),
        ],
        commands={},
    )

    touched = closure.touched_by(
        ["src/app.py", ".github/workflows/ci.yml", "package.json", "README.md"]
    )

    assert touched == (".github/workflows/ci.yml", "package.json")


# -- json pointer comparison -----------------------------------------------


def test_only_watched_keys_count_as_a_trust_root_change():
    entry = TrustRootEntry(
        kind="json_pointer", path="package.json", pointers=("/scripts/test",)
    )
    before = '{"scripts": {"test": "vitest run"}, "version": "1.0.0"}'

    ordinary = '{"scripts": {"test": "vitest run"}, "version": "1.1.0"}'
    privileged = '{"scripts": {"test": "exit 0"}, "version": "1.0.0"}'

    assert not json_pointers_changed(entry, before, ordinary)
    assert json_pointers_changed(entry, before, privileged)


def test_unparseable_json_counts_as_changed():
    """A file the resolver cannot read is one whose authority it cannot vouch for."""

    entry = TrustRootEntry(
        kind="json_pointer", path="package.json", pointers=("/scripts/test",)
    )

    assert json_pointers_changed(entry, '{"scripts": {"test": "x"}}', "{ not json")


# -- readiness -------------------------------------------------------------


def _contract(tmp_path: Path, body: str):
    from ticket_agent.config.repo_contract import load_repo_contract

    path = tmp_path / "contract.yaml"
    path.write_text(body)
    return load_repo_contract(path)


_BASE = """
repo:
  name: demo
  root: {root}
  default_branch: main
language:
  primary: typescript
  package_manager: npm
commands:
  test:
    command: {test_command}
    timeout_seconds: 60
    working_directory: "."
    writable_paths: []
    network: none
policy:
  dependency_install_allowed: false
source_dirs: ["src/"]
test_dirs: ["tests/"]
{extra}
"""


def test_vacuous_gate_makes_a_repository_unready(tmp_path):
    contract = _contract(
        tmp_path,
        _BASE.format(
            root=str(tmp_path),
            test_command='["bash", "-lc", "if [ -f package.json ]; then npm test; else echo none; fi"]',
            extra='trust_root:\n  - {kind: file, path: CODEOWNERS}\n',
        ),
    )

    readiness, reasons = contract.readiness(tmp_path)

    assert readiness == "unready"
    assert any("without running anything" in reason for reason in reasons)


def test_missing_trust_root_is_partial_not_unready(tmp_path):
    """Missing declaration limits autonomy; it does not make the repo broken."""

    contract = _contract(
        tmp_path,
        _BASE.format(
            root=str(tmp_path), test_command='["npm", "run", "test"]', extra=""
        ),
    )

    readiness, reasons = contract.readiness(tmp_path)

    assert readiness == "partial"
    assert any("no trust root" in reason for reason in reasons)


def test_fully_declared_contract_is_ready(tmp_path):
    contract = _contract(
        tmp_path,
        _BASE.format(
            root=str(tmp_path),
            test_command='["npm", "run", "test"]',
            extra=(
                "trust_root:\n"
                "  - {kind: file, path: CODEOWNERS}\n"
                "  - {kind: derived, source: verification_commands}\n"
            ),
        ),
    )
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest"}}')

    readiness, reasons = contract.readiness(tmp_path)

    assert readiness == "full", reasons


def test_gate_declared_required_without_a_command_is_rejected(tmp_path):
    with pytest.raises(RepoContractError, match="build"):
        _contract(
            tmp_path,
            _BASE.format(
                root=str(tmp_path),
                test_command='["npm", "run", "test"]',
                extra="gates:\n  build: required\n",
            ),
        )


def test_test_gate_cannot_be_optional(tmp_path):
    with pytest.raises(RepoContractError, match="optional"):
        _contract(
            tmp_path,
            _BASE.format(
                root=str(tmp_path),
                test_command='["npm", "run", "test"]',
                extra="gates:\n  test: optional\n",
            ),
        )


def test_install_is_not_a_declarable_gate(tmp_path):
    with pytest.raises(RepoContractError, match="install"):
        _contract(
            tmp_path,
            _BASE.format(
                root=str(tmp_path),
                test_command='["npm", "run", "test"]',
                extra="gates:\n  install: required\n",
            ),
        )

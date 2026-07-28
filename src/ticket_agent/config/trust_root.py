"""What holds verification authority in a repository, and its closure.

A path glob list is the obvious way to say "the model may not touch this",
and it is both over- and under-inclusive. ``package.json`` matters because
``scripts.test`` is what ``npm test`` delegates to -- the rest of the file is
ordinary. And a *newly created* file can acquire authority without appearing
on any list anyone wrote.

So authority is declared by kind:

``file``          the whole file is authoritative
``tree``          everything under a directory
``json_pointer``  only named keys of a JSON file
``derived``       whatever the trusted commands actually resolve to

The last one cannot be computed in general. ``bash -lc`` with interpolation,
an npm script that shells out, a config that requires another file -- all can
repoint verification at runtime. Rather than digest a closure that silently
excludes them, unresolved commands **lower the repository's readiness**, which
lowers the autonomy ceiling. A hole in the trust root is reported, not hidden.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from ticket_agent.domain.errors import RepoContractError

TrustRootKind = Literal["file", "tree", "json_pointer", "derived"]
DerivedSource = Literal["verification_commands"]

#: Package-manager launchers whose first script argument names a target we can
#: resolve one hop further, into the manifest key that defines it.
_SCRIPT_RUNNERS: Mapping[str, str] = {
    "npm": "package.json",
    "pnpm": "package.json",
    "yarn": "package.json",
    "bun": "package.json",
}

#: Interpreters whose argument is a shell program rather than a program path.
#: Anything routed through these is not statically resolvable.
_SHELL_INTERPRETERS: frozenset[str] = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "fish"}
)


@dataclass(frozen=True)
class TrustRootEntry:
    """One declared unit of verification authority."""

    kind: TrustRootKind
    path: str = ""
    pointers: tuple[str, ...] = ()
    source: DerivedSource | None = None

    def __post_init__(self) -> None:
        if self.kind == "derived":
            if self.source is None:
                raise RepoContractError("trust_root derived entries require a source")
        elif not self.path:
            raise RepoContractError(f"trust_root {self.kind} entries require a path")
        if self.kind == "json_pointer" and not self.pointers:
            raise RepoContractError(
                "trust_root json_pointer entries require at least one pointer"
            )

    def covers(self, changed_path: str) -> bool:
        """Whether a changed path falls under this entry.

        ``json_pointer`` returns True for the file: path-level matching cannot
        tell which key changed, so the caller must compare pointer values to
        decide. Answering False here would let a ``scripts.test`` edit through.
        """

        # removeprefix, not lstrip: lstrip("./") strips any leading '.' or '/'
        # characters, which turns ".github/workflows/ci.yml" into
        # "github/workflows/ci.yml" and silently fails to match every dotfile
        # trust-root entry -- exactly the paths that most need to match.
        candidate = changed_path.strip().removeprefix("./")
        declared = self.path.strip().removeprefix("./")
        if self.kind == "tree":
            prefix = declared.rstrip("/") + "/"
            return candidate == declared.rstrip("/") or candidate.startswith(prefix)
        return candidate == declared


@dataclass(frozen=True)
class TrustRootClosure:
    """The resolved trust root, plus everything that could not be resolved."""

    entries: tuple[TrustRootEntry, ...] = ()
    unresolved: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.unresolved

    def covers(self, changed_path: str) -> bool:
        return any(entry.covers(changed_path) for entry in self.entries)

    def touched_by(self, changed_paths: Iterable[str]) -> tuple[str, ...]:
        """Every changed path that lands on the trust root."""

        return tuple(sorted({p for p in changed_paths if self.covers(p)}))


def parse_trust_root(raw: Any) -> tuple[TrustRootEntry, ...]:
    """Parse the declared ``trust_root`` block."""

    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RepoContractError("trust_root must be a list")

    entries: list[TrustRootEntry] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RepoContractError(f"trust_root[{index}] must be a mapping")
        kind = item.get("kind")
        if kind not in ("file", "tree", "json_pointer", "derived"):
            raise RepoContractError(
                f"trust_root[{index}].kind must be file, tree, json_pointer, or derived"
            )
        pointers = item.get("pointers", [])
        if not isinstance(pointers, list) or not all(
            isinstance(p, str) for p in pointers
        ):
            raise RepoContractError(
                f"trust_root[{index}].pointers must be a list of strings"
            )
        entries.append(
            TrustRootEntry(
                kind=kind,
                path=str(item.get("path", "")),
                pointers=tuple(pointers),
                source=item.get("source"),
            )
        )
    return tuple(entries)


def resolve_closure(
    declared: Sequence[TrustRootEntry],
    *,
    commands: Mapping[str, Sequence[str]],
    repo_root: Path | None = None,
) -> TrustRootClosure:
    """Expand ``derived`` entries by following trusted commands one hop.

    Every command that cannot be followed to a concrete artifact is recorded
    in ``unresolved`` rather than dropped, so readiness can reflect the gap.
    """

    resolved: list[TrustRootEntry] = [e for e in declared if e.kind != "derived"]
    unresolved: list[str] = []
    wants_derived = any(
        e.kind == "derived" and e.source == "verification_commands" for e in declared
    )

    if wants_derived:
        for gate, argv in sorted(commands.items()):
            entry, problem = _resolve_command(gate, tuple(argv), repo_root)
            if problem is not None:
                unresolved.append(problem)
            if entry is not None and entry not in resolved:
                resolved.append(entry)

    return TrustRootClosure(entries=tuple(resolved), unresolved=tuple(unresolved))


def _resolve_command(
    gate: str,
    argv: Sequence[str],
    repo_root: Path | None,
) -> tuple[TrustRootEntry | None, str | None]:
    """Follow one command one hop toward the artifact that defines it."""

    if not argv:
        return None, f"{gate}: empty command"

    program = Path(argv[0]).name

    if program in _SHELL_INTERPRETERS:
        # `bash -lc '<script>'` can compute its target at runtime. There is no
        # honest static answer here, so say so.
        return None, (
            f"{gate}: runs through {program!r}; a shell program's target "
            "cannot be resolved statically"
        )

    manifest = _SCRIPT_RUNNERS.get(program)
    if manifest is not None:
        if _is_non_script_subcommand(argv):
            # `npm ci` / `npm install` run the package manager itself, not a
            # repo-defined script. Nothing to resolve, and no hole either.
            return None, None
        script = _script_target(argv)
        if script is None:
            return None, f"{gate}: {program} invocation names no resolvable script"
        entry = TrustRootEntry(
            kind="json_pointer", path=manifest, pointers=(f"/scripts/{script}",)
        )
        if repo_root is not None and repo_root.exists() and not (
            repo_root / manifest
        ).exists():
            # Only meaningful when the repository itself is present; otherwise
            # every entry would report absent and drown the real reason.
            return entry, f"{gate}: {manifest} is declared but absent"
        return entry, None

    # A plain program path inside the repo is itself authoritative.
    candidate = argv[0]
    if candidate.startswith("./") or (repo_root and (repo_root / candidate).is_file()):
        return TrustRootEntry(kind="file", path=candidate.lstrip("./")), None

    # A system binary (python, tsc, ruff) carries no in-repo authority.
    return None, None


#: Package-manager subcommands that run the manager itself rather than a
#: script defined by the repository.
_NON_SCRIPT_SUBCOMMANDS: frozenset[str] = frozenset(
    {"ci", "install", "i", "exec", "dlx", "add", "prune", "audit"}
)


def _is_non_script_subcommand(argv: Sequence[str]) -> bool:
    parts = [p for p in argv[1:] if not p.startswith("-")]
    return bool(parts) and parts[0] in _NON_SCRIPT_SUBCOMMANDS


def _script_target(argv: Sequence[str]) -> str | None:
    """The script name in `npm test` / `npm run <name>` / `yarn <name>`."""

    parts = [p for p in argv[1:] if not p.startswith("-")]
    if not parts:
        return None
    if parts[0] == "run":
        return parts[1] if len(parts) > 1 else None
    if parts[0] in {"test", "lint", "build", "start"}:
        return parts[0]
    return parts[0]


def pointer_values(document: Any, pointers: Sequence[str]) -> dict[str, Any]:
    """Read JSON-pointer values, for comparing a file across two revisions."""

    return {pointer: _read_pointer(document, pointer) for pointer in pointers}


def _read_pointer(document: Any, pointer: str) -> Any:
    node = document
    for token in [t for t in pointer.split("/") if t]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    return node


def json_pointers_changed(
    entry: TrustRootEntry,
    before: str | None,
    after: str | None,
) -> bool:
    """Whether a json_pointer entry's watched keys actually changed.

    Unparseable JSON counts as changed: a file the resolver cannot read is a
    file whose authority it cannot vouch for.
    """

    if entry.kind != "json_pointer":
        return before != after

    def load(text: str | None) -> Any:
        if text is None:
            return None
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return _UNPARSEABLE

    before_doc, after_doc = load(before), load(after)
    if before_doc is _UNPARSEABLE or after_doc is _UNPARSEABLE:
        return True
    return pointer_values(before_doc, entry.pointers) != pointer_values(
        after_doc, entry.pointers
    )


_UNPARSEABLE = object()


__all__ = [
    "DerivedSource",
    "TrustRootClosure",
    "TrustRootEntry",
    "TrustRootKind",
    "json_pointers_changed",
    "parse_trust_root",
    "pointer_values",
    "resolve_closure",
]

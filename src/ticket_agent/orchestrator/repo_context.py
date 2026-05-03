"""Bounded repository context for model-backed implementation services."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ticket_agent.config.repo_contract import RepoContract
from ticket_agent.orchestrator.state import TicketState


DEFAULT_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "coverage",
        ".next",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
        ".cache",
        ".gradle",
        ".terraform",
        "target",
        "out",
    }
)


DEFAULT_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        # images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".ico",
        ".svg",
        # videos / audio
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        # archives
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # binary / compiled
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".o",
        ".obj",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        # databases / locks
        ".db",
        ".sqlite",
        ".sqlite3",
        ".lock",
        # docs binary
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        # misc
        ".dat",
        ".npy",
        ".npz",
        ".pkl",
        ".pickle",
    }
)


DEFAULT_SKIP_BASENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "go.sum",
    }
)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PATH_HINT_RE = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6}")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_COMMON_TOKENS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "when",
        "then",
        "should",
        "must",
        "will",
        "have",
        "has",
        "are",
        "was",
        "were",
        "but",
        "not",
        "you",
        "your",
        "our",
        "their",
        "use",
        "uses",
        "using",
        "add",
        "fix",
        "bug",
        "ticket",
        "feature",
        "user",
        "users",
        "test",
        "tests",
        "code",
        "file",
        "files",
        "function",
        "method",
        "class",
        "support",
        "implement",
        "implementation",
        "make",
        "ensure",
        "endpoint",
        "service",
        "module",
    }
)


@dataclass(slots=True)
class RepoContractSummary:
    """Compact, prompt-friendly view of a repo contract."""

    name: str | None
    primary_language: str | None
    package_manager: str | None
    source_dirs: tuple[str, ...]
    test_dirs: tuple[str, ...]
    test_command: tuple[str, ...] | None
    config_paths_allowed: tuple[str, ...]
    protected_paths: tuple[str, ...]


@dataclass(slots=True)
class RepoContext:
    """Bounded snapshot of repository state passed to the implementation model."""

    ticket_key: str
    summary: str
    description: str
    repository: str | None = None
    repo_path: str | None = None
    worktree_path: str | None = None
    decomposition: dict[str, Any] | None = None
    files_to_modify: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)
    truncated_files: list[str] = field(default_factory=list)
    repo_contract: RepoContractSummary | None = None
    previous_test_result: dict[str, Any] | None = None
    previous_implementation_summary: str | None = None
    implementation_attempts: int = 0
    max_attempts: int = 3
    total_files_listed: int = 0
    total_chars_read: int = 0

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable view safe to embed in a prompt."""

        return {
            "ticket_key": self.ticket_key,
            "summary": self.summary,
            "description": self.description,
            "repository": self.repository,
            "repo_path": self.repo_path,
            "worktree_path": self.worktree_path,
            "decomposition": self.decomposition,
            "files_to_modify": list(self.files_to_modify),
            "relevant_files": list(self.relevant_files),
            "file_contents": dict(self.file_contents),
            "truncated_files": list(self.truncated_files),
            "repo_contract": _contract_summary_to_dict(self.repo_contract),
            "previous_test_result": self.previous_test_result,
            "previous_implementation_summary": self.previous_implementation_summary,
            "implementation_attempts": self.implementation_attempts,
            "max_attempts": self.max_attempts,
            "total_files_listed": self.total_files_listed,
            "total_chars_read": self.total_chars_read,
        }


class RepoContextBuilder:
    """Builds a bounded RepoContext from a ticket state and a worktree."""

    def __init__(
        self,
        *,
        max_files_listed: int = 300,
        max_files_read: int = 12,
        max_file_chars: int = 12_000,
        max_total_chars: int = 50_000,
        skip_dirs: Iterable[str] = DEFAULT_SKIP_DIRS,
        skip_extensions: Iterable[str] = DEFAULT_SKIP_EXTENSIONS,
        skip_basenames: Iterable[str] = DEFAULT_SKIP_BASENAMES,
        repo_contract: RepoContract | None = None,
    ) -> None:
        self._max_files_listed = max(0, int(max_files_listed))
        self._max_files_read = max(0, int(max_files_read))
        self._max_file_chars = max(0, int(max_file_chars))
        self._max_total_chars = max(0, int(max_total_chars))
        self._skip_dirs = frozenset(skip_dirs)
        self._skip_extensions = frozenset(ext.lower() for ext in skip_extensions)
        self._skip_basenames = frozenset(skip_basenames)
        self._repo_contract = repo_contract

    def build(self, state: TicketState) -> RepoContext:
        decomposition = (
            dict(state.decomposition) if isinstance(state.decomposition, Mapping) else None
        )
        files_to_modify = _string_list(
            (decomposition or {}).get("files_to_modify")
        )

        previous_test_result = (
            dict(state.test_result) if isinstance(state.test_result, Mapping) else None
        )
        previous_implementation_summary = _previous_implementation_summary(state)

        context = RepoContext(
            ticket_key=state.ticket_key,
            summary=state.summary,
            description=state.description or "",
            repository=state.repository,
            repo_path=state.repo_path,
            worktree_path=state.worktree_path,
            decomposition=decomposition,
            files_to_modify=files_to_modify,
            previous_test_result=previous_test_result,
            previous_implementation_summary=previous_implementation_summary,
            implementation_attempts=state.implementation_attempts,
            max_attempts=state.max_attempts,
        )
        if self._repo_contract is not None:
            context.repo_contract = _contract_summary(self._repo_contract)

        worktree_root = _resolve_worktree_root(state.worktree_path)
        if worktree_root is None:
            return context

        listed_files = self._list_files(worktree_root)
        context.total_files_listed = len(listed_files)

        relevant_files = self._select_relevant_files(
            listed_files=listed_files,
            files_to_modify=files_to_modify,
            ticket_text=f"{state.summary}\n{state.description or ''}",
        )
        context.relevant_files = relevant_files

        contents, truncated, total_chars = self._read_files(
            worktree_root=worktree_root,
            paths=relevant_files,
        )
        context.file_contents = contents
        context.truncated_files = truncated
        context.total_chars_read = total_chars
        return context

    def _list_files(self, worktree_root: Path) -> list[str]:
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(worktree_root, topdown=True):
            current = Path(dirpath)
            current_resolved = _safe_resolve(current)
            if current_resolved is None or not _is_within(current_resolved, worktree_root):
                dirnames[:] = []
                continue

            kept: list[str] = []
            for name in dirnames:
                if name in self._skip_dirs:
                    continue
                if name.startswith("."):
                    continue
                kept.append(name)
            dirnames[:] = sorted(kept)

            for filename in sorted(filenames):
                if filename in self._skip_basenames:
                    continue
                suffix = Path(filename).suffix.lower()
                if suffix in self._skip_extensions:
                    continue
                full = current / filename
                resolved = _safe_resolve(full)
                if resolved is None or not _is_within(resolved, worktree_root):
                    continue
                if not resolved.is_file():
                    continue
                relative = resolved.relative_to(worktree_root).as_posix()
                results.append(relative)
                if len(results) >= self._max_files_listed:
                    return results
        return results

    def _select_relevant_files(
        self,
        *,
        listed_files: Sequence[str],
        files_to_modify: Sequence[str],
        ticket_text: str,
    ) -> list[str]:
        listed_set = set(listed_files)
        ordered: list[str] = []
        seen: set[str] = set()

        # 1. decomposition.files_to_modify, in order
        for raw in files_to_modify:
            normalized = _normalize_relative(raw)
            if normalized is None:
                continue
            if normalized in listed_set and normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)

        # 2. paths and tokens mentioned in ticket text
        path_hints = _extract_path_hints(ticket_text)
        for hint in path_hints:
            for path in listed_files:
                if path in seen:
                    continue
                if hint == path or path.endswith("/" + hint) or path.endswith(hint):
                    ordered.append(path)
                    seen.add(path)

        tokens = _extract_tokens(ticket_text)
        for token in tokens:
            for path in listed_files:
                if path in seen:
                    continue
                if token.lower() in path.lower():
                    ordered.append(path)
                    seen.add(path)
                    if len(ordered) >= self._max_files_read * 4:
                        break
            if len(ordered) >= self._max_files_read * 4:
                break

        # 3. test files related to selected source files
        related_tests: list[str] = []
        for path in list(ordered):
            related_tests.extend(_related_test_paths(path, listed_files, seen))
        for path in related_tests:
            if path not in seen:
                ordered.append(path)
                seen.add(path)

        return ordered[: self._max_files_read]

    def _read_files(
        self,
        *,
        worktree_root: Path,
        paths: Sequence[str],
    ) -> tuple[dict[str, str], list[str], int]:
        contents: dict[str, str] = {}
        truncated: list[str] = []
        total_chars = 0

        for path in paths:
            if total_chars >= self._max_total_chars:
                break
            full = (worktree_root / path).resolve()
            if not _is_within(full, worktree_root) or not full.is_file():
                continue
            try:
                raw = full.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                continue
            remaining_total = self._max_total_chars - total_chars
            limit = min(self._max_file_chars, remaining_total)
            if limit <= 0:
                break
            if len(raw) > limit:
                contents[path] = raw[:limit]
                truncated.append(path)
                total_chars += limit
            else:
                contents[path] = raw
                total_chars += len(raw)

        return contents, truncated, total_chars


def _resolve_worktree_root(worktree_path: str | None) -> Path | None:
    if not worktree_path:
        return None
    try:
        root = Path(worktree_path).resolve()
    except (OSError, RuntimeError):
        return None
    if not root.exists() or not root.is_dir():
        return None
    return root


def _safe_resolve(path: Path) -> Path | None:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _previous_implementation_summary(state: TicketState) -> str | None:
    result = state.implementation_result
    if not isinstance(result, Mapping):
        return None
    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    return None


def _normalize_relative(path: str) -> str | None:
    if not isinstance(path, str):
        return None
    cleaned = path.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("/"):
        return None
    parts = tuple(p for p in cleaned.split("/") if p)
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def _extract_path_hints(text: str) -> list[str]:
    if not text:
        return []
    hints: list[str] = []
    seen: set[str] = set()

    for match in _BACKTICK_RE.finditer(text):
        candidate = match.group(1).strip()
        normalized = _normalize_relative(candidate)
        if normalized is None or normalized in seen:
            continue
        if "/" not in normalized and "." not in normalized:
            continue
        hints.append(normalized)
        seen.add(normalized)

    for match in _PATH_HINT_RE.finditer(text):
        candidate = match.group(0)
        normalized = _normalize_relative(candidate)
        if normalized is None or normalized in seen:
            continue
        hints.append(normalized)
        seen.add(normalized)
    return hints


def _extract_tokens(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        if lowered in _COMMON_TOKENS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        tokens.append(word)
    return tokens


def _related_test_paths(
    source_path: str,
    listed_files: Sequence[str],
    seen: set[str],
) -> list[str]:
    posix = PurePosixPath(source_path)
    stem = posix.stem
    if not stem:
        return []
    if stem.startswith("test_") or stem.endswith("_test"):
        return []
    candidates: list[str] = []
    for path in listed_files:
        if path in seen:
            continue
        candidate = PurePosixPath(path)
        candidate_stem = candidate.stem
        is_test_file = (
            candidate_stem.startswith("test_") or candidate_stem.endswith("_test")
        )
        if not is_test_file:
            continue
        if stem in candidate_stem:
            candidates.append(path)
    return candidates


def _contract_summary(contract: RepoContract) -> RepoContractSummary:
    test_command: tuple[str, ...] | None
    try:
        test_command = tuple(contract.commands.test.command)
    except Exception:  # noqa: BLE001 - defensive: best-effort summary
        test_command = None
    return RepoContractSummary(
        name=getattr(contract.repo, "name", None),
        primary_language=getattr(contract.language, "primary", None),
        package_manager=getattr(contract.language, "package_manager", None),
        source_dirs=tuple(contract.source_dirs),
        test_dirs=tuple(contract.test_dirs),
        test_command=test_command,
        config_paths_allowed=tuple(contract.policy.config_paths_allowed),
        protected_paths=tuple(contract.policy.protected_paths),
    )


def _contract_summary_to_dict(
    summary: RepoContractSummary | None,
) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "name": summary.name,
        "primary_language": summary.primary_language,
        "package_manager": summary.package_manager,
        "source_dirs": list(summary.source_dirs),
        "test_dirs": list(summary.test_dirs),
        "test_command": list(summary.test_command) if summary.test_command else None,
        "config_paths_allowed": list(summary.config_paths_allowed),
        "protected_paths": list(summary.protected_paths),
    }


__all__ = [
    "DEFAULT_SKIP_BASENAMES",
    "DEFAULT_SKIP_DIRS",
    "DEFAULT_SKIP_EXTENSIONS",
    "RepoContext",
    "RepoContextBuilder",
    "RepoContractSummary",
]

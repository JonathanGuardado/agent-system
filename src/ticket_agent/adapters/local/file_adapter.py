"""Worktree-scoped local file adapter."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from ticket_agent.config.repo_contract import RepoContract
from ticket_agent.domain.errors import PathBoundaryError, PolicyViolationError


_ROOT_CONFIG_SUFFIXES = frozenset(
    {".yml", ".yaml", ".toml", ".json", ".ini", ".cfg"}
)


class LocalFileAdapter:
    """Read and write files without escaping a configured worktree root."""

    def __init__(
        self,
        worktree_root: str | Path,
        repo_contract: RepoContract | None = None,
    ) -> None:
        self._root = Path(worktree_root).resolve()
        self._repo_contract = repo_contract

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate

        resolved = candidate.resolve()
        if not _is_relative_to(resolved, self._root):
            raise PathBoundaryError(resolved, self._root)
        return resolved

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        return self.resolve(path).read_text(encoding=encoding)

    def write_text(
        self,
        path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        resolved = self.resolve(path)
        relative_path = _relative_posix(resolved, self._root)
        self._enforce_write_policy(relative_path, resolved, content)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)

    def exists(self, path: str | Path) -> bool:
        return self.resolve(path).exists()

    def list_files(self, path: str | Path = ".") -> tuple[str, ...]:
        resolved = self.resolve(path)
        if not resolved.exists():
            return ()

        if resolved.is_file():
            relative_path = _relative_posix(resolved, self._root)
            if _is_git_path(relative_path):
                return ()
            return (relative_path,)

        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(resolved, topdown=True):
            current_dir = Path(dirpath)
            current_resolved = current_dir.resolve()
            if not _is_relative_to(current_resolved, self._root):
                raise PathBoundaryError(current_resolved, self._root)

            kept_dirnames: list[str] = []
            for dirname in dirnames:
                child = current_dir / dirname
                if dirname == ".git":
                    continue
                child_resolved = child.resolve()
                if not _is_relative_to(child_resolved, self._root):
                    raise PathBoundaryError(child_resolved, self._root)
                kept_dirnames.append(dirname)
            dirnames[:] = kept_dirnames

            for filename in filenames:
                child = current_dir / filename
                relative_path = _relative_posix(child, self._root)
                if _is_git_path(relative_path):
                    continue
                child_resolved = child.resolve()
                if not _is_relative_to(child_resolved, self._root):
                    raise PathBoundaryError(child_resolved, self._root)
                if child_resolved.is_file():
                    files.append(relative_path)

        return tuple(sorted(files))

    def _enforce_write_policy(
        self,
        relative_path: str,
        resolved: Path,
        content: str,
    ) -> None:
        if content == "" and resolved.exists() and resolved.stat().st_size > 0:
            raise PolicyViolationError(
                relative_path,
                "empty write would delete existing non-empty file contents",
            )

        protected_reason = _protected_path_reason(relative_path)
        if protected_reason is not None:
            raise PolicyViolationError(relative_path, protected_reason)

        if _is_root_config_file(relative_path) and not self._is_allowed_config_path(
            relative_path
        ):
            raise PolicyViolationError(
                relative_path,
                "root-level config file requires explicit repo contract permission",
            )

        if self._repo_contract is not None and not self._is_contract_allowed_path(
            relative_path
        ):
            raise PolicyViolationError(
                relative_path,
                "write is outside repo contract source_dirs, test_dirs, "
                "and config_paths_allowed",
            )

    def _is_allowed_config_path(self, relative_path: str) -> bool:
        if self._repo_contract is None:
            return False
        return any(
            _path_matches_spec(relative_path, spec)
            for spec in self._repo_contract.policy.config_paths_allowed
        )

    def _is_contract_allowed_path(self, relative_path: str) -> bool:
        if self._repo_contract is None:
            return True

        return (
            any(
                _path_matches_directory_spec(relative_path, spec)
                for spec in self._repo_contract.source_dirs
                + self._repo_contract.test_dirs
            )
            or any(
                _path_matches_spec(relative_path, spec)
                for spec in self._repo_contract.policy.config_paths_allowed
            )
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _protected_path_reason(relative_path: str) -> str | None:
    parts = PurePosixPath(relative_path).parts
    if not parts:
        return None

    if parts[0] == ".github":
        return "writes under .github/ are blocked"
    if parts[0] == ".git" or ".git" in parts:
        return "writes under .git/ are blocked"
    if parts[0] == "secrets" or "secrets" in parts:
        return "writes under secrets/ are blocked"

    name = parts[-1]
    if len(parts) == 1 and name in {
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
    }:
        return f"writes to {name} are blocked"
    if name == ".env" or name.startswith(".env."):
        return "writes to .env files are blocked"

    return None


def _is_root_config_file(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return len(path.parts) == 1 and path.suffix in _ROOT_CONFIG_SUFFIXES


def _is_git_path(relative_path: str) -> bool:
    return PurePosixPath(relative_path).parts[:1] == (".git",)


def _path_matches_spec(relative_path: str, spec: str) -> bool:
    normalized, is_directory = _normalize_policy_spec(spec)
    if normalized == ".":
        return True
    if is_directory:
        return relative_path == normalized or relative_path.startswith(
            f"{normalized}/"
        )
    return relative_path == normalized


def _path_matches_directory_spec(relative_path: str, spec: str) -> bool:
    normalized, _ = _normalize_policy_spec(spec)
    if normalized == ".":
        return True
    return relative_path == normalized or relative_path.startswith(f"{normalized}/")


def _normalize_policy_spec(spec: str) -> tuple[str, bool]:
    normalized = spec.replace("\\", "/")
    is_directory = normalized.endswith("/")
    normalized = PurePosixPath(normalized).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.rstrip("/")
    if normalized == "":
        normalized = "."
    return normalized, is_directory

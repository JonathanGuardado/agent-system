"""Worktree-scoped local file adapter."""

from __future__ import annotations

from pathlib import Path

from ticket_agent.domain.errors import PathBoundaryError


class LocalFileAdapter:
    """Read and write files without escaping a configured worktree root."""

    def __init__(self, worktree_root: str | Path) -> None:
        self._root = Path(worktree_root).resolve(strict=True)

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate

        resolved = candidate.resolve(strict=False)
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
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)

    def exists(self, path: str | Path) -> bool:
        return self.resolve(path).exists()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

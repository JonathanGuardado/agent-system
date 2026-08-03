"""Local git adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import subprocess

from ticket_agent.domain.errors import (
    GitAdapterError,
    NoChangesToCommitError,
    PushError,
    WorktreeCleanupError,
    WorktreeCreationError,
)
from ticket_agent.domain.git import WorktreeInfo
from ticket_agent.github import GH_ROLE_ADMIN, GH_ROLE_BOT, GitHubCredentials

_SAFE_REF_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SAFE_GITHUB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PROTECTED_BRANCHES = frozenset({"main", "master", "develop"})
_GENERATED_COMMIT_EXCLUDES = (
    "node_modules",
    ".next",
    "dist",
    "build",
    "coverage",
)


class GitAdapter:
    """Run local git operations for ticket worktrees."""

    def __init__(
        self,
        *,
        default_timeout_seconds: int = 300,
        credentials: GitHubCredentials | None = None,
    ) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        self._default_timeout_seconds = default_timeout_seconds
        self._credentials = credentials

    def create_worktree(
        self,
        repo_path: str | Path,
        ticket_key: str,
        short_lock_id: str,
        base_branch: str | None = None,
    ) -> WorktreeInfo:
        repo = Path(repo_path).resolve(strict=True)
        _validate_safe_ref_component(ticket_key, "ticket_key")
        _validate_safe_ref_component(short_lock_id, "short_lock_id")
        if base_branch is not None:
            _validate_integration_branch(base_branch, WorktreeCreationError)
            self._prepare_integration_branch(repo, base_branch)

        branch_name = f"agent/{ticket_key}/{short_lock_id}"
        worktree_path = repo / ".worktrees" / ticket_key / short_lock_id
        try:
            _validate_worktree_path(repo, worktree_path)
        except WorktreeCleanupError as exc:
            raise WorktreeCreationError(str(exc)) from exc
        if worktree_path.exists() or worktree_path.is_symlink():
            raise WorktreeCreationError(f"worktree already exists: {worktree_path}")

        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        result = self._add_worktree(
            repo,
            worktree_path,
            branch_name,
            base_branch=base_branch,
        )
        if result.returncode != 0:
            raise WorktreeCreationError(_failure_message(result))

        return WorktreeInfo(
            repo_path=repo,
            worktree_path=worktree_path,
            branch_name=branch_name,
            ticket_key=ticket_key,
            lock_id=short_lock_id,
        )

    def create_worktree_for_branch(
        self,
        repo_path: str | Path,
        ticket_key: str,
        branch_name: str,
        short_lock_id: str,
    ) -> WorktreeInfo:
        repo = Path(repo_path).resolve(strict=True)
        _validate_safe_ref_component(ticket_key, "ticket_key")
        _validate_push_branch(branch_name)
        _validate_safe_ref_component(short_lock_id, "short_lock_id")

        worktree_path = repo / ".worktrees" / ticket_key / short_lock_id
        try:
            _validate_worktree_path(repo, worktree_path)
        except WorktreeCleanupError as exc:
            raise WorktreeCreationError(str(exc)) from exc
        if worktree_path.exists() or worktree_path.is_symlink():
            raise WorktreeCreationError(f"worktree already exists: {worktree_path}")

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_local_branch(repo, branch_name)

        result = self._run_git(
            ("worktree", "add", str(worktree_path), branch_name),
            cwd=repo,
        )
        if result.returncode != 0:
            raise WorktreeCreationError(_failure_message(result))

        return WorktreeInfo(
            repo_path=repo,
            worktree_path=worktree_path,
            branch_name=branch_name,
            ticket_key=ticket_key,
            lock_id=short_lock_id,
        )

    def commit(self, worktree_path: str | Path, message: str) -> str:
        worktree = Path(worktree_path).resolve(strict=True)

        add_result = self._run_git(("add", "-A"), cwd=worktree)
        if add_result.returncode != 0:
            raise GitAdapterError(_failure_message(add_result))

        _unstage_generated_paths(worktree, timeout_seconds=self._default_timeout_seconds)

        diff_result = self._run_git(("diff", "--cached", "--quiet"), cwd=worktree)
        if diff_result.returncode == 0:
            raise NoChangesToCommitError("no changes to commit")
        if diff_result.returncode != 1:
            raise GitAdapterError(_failure_message(diff_result))

        commit_result = self._run_git(("commit", "-m", message), cwd=worktree)
        if commit_result.returncode != 0:
            raise GitAdapterError(_failure_message(commit_result))

        sha_result = self._run_git(("rev-parse", "HEAD"), cwd=worktree)
        if sha_result.returncode != 0:
            raise GitAdapterError(_failure_message(sha_result))
        return sha_result.stdout.strip()

    def staged_tree_digest(self, worktree_path: str | Path) -> str:
        """Stage the candidate and return Git's content-addressed tree id."""

        worktree = Path(worktree_path).resolve(strict=True)
        add_result = self._run_git(("add", "-A"), cwd=worktree)
        if add_result.returncode != 0:
            raise GitAdapterError(_failure_message(add_result))
        _unstage_generated_paths(worktree, timeout_seconds=self._default_timeout_seconds)
        tree_result = self._run_git(("write-tree",), cwd=worktree)
        if tree_result.returncode != 0:
            raise GitAdapterError(_failure_message(tree_result))
        return tree_result.stdout.strip()

    def current_head(self, worktree_path: str | Path) -> tuple[str, str]:
        """Return ``(commit SHA, tree id)`` for the current branch head."""

        worktree = Path(worktree_path).resolve(strict=True)
        result = self._run_git(("show", "-s", "--format=%H%n%T", "HEAD"), cwd=worktree)
        if result.returncode != 0:
            raise GitAdapterError(_failure_message(result))
        lines = result.stdout.strip().splitlines()
        if len(lines) != 2 or not all(lines):
            raise GitAdapterError("git show did not return a commit and tree id")
        return lines[0], lines[1]

    def remote_ref_sha(
        self,
        worktree_path: str | Path,
        branch_name: str,
    ) -> str | None:
        """Probe the bot-visible origin ref without mutating it."""

        worktree = Path(worktree_path).resolve(strict=True)
        _validate_push_branch(branch_name)
        bot_env = self._required_git_bot_env()
        _ensure_origin_remote(
            worktree,
            timeout_seconds=self._default_timeout_seconds,
            credentials=self._credentials,
        )
        result = self._run_git(
            ("ls-remote", "--heads", "origin", f"refs/heads/{branch_name}"),
            cwd=worktree,
            env=bot_env,
        )
        if result.returncode != 0:
            raise PushError(_failure_message(result))
        output = result.stdout.strip()
        if not output:
            return None
        return output.split(maxsplit=1)[0]

    def push(self, worktree_path: str | Path, branch_name: str) -> None:
        worktree = Path(worktree_path).resolve(strict=True)
        _validate_push_branch(branch_name)
        bot_env = self._required_git_bot_env()
        _ensure_origin_remote(
            worktree,
            timeout_seconds=self._default_timeout_seconds,
            credentials=self._credentials,
        )

        result = self._run_git(
            ("push", "origin", branch_name),
            cwd=worktree,
            env=bot_env,
        )
        if result.returncode != 0:
            raise PushError(_failure_message(result))

    def ensure_pull_request_base(
        self,
        worktree_path: str | Path,
        branch_name: str,
    ) -> None:
        worktree = Path(worktree_path).resolve(strict=True)
        _validate_integration_branch(branch_name, PushError)
        bot_env = self._required_git_bot_env()
        _ensure_origin_remote(
            worktree,
            timeout_seconds=self._default_timeout_seconds,
            credentials=self._credentials,
        )
        result = self._run_git(
            ("push", "origin", f"{branch_name}:{branch_name}"),
            cwd=worktree,
            env=bot_env,
        )
        if result.returncode != 0:
            raise PushError(_failure_message(result))

    def cleanup_worktree(self, repo_path: str | Path, worktree_path: str | Path) -> None:
        repo = Path(repo_path).resolve(strict=True)
        resolved_worktree = Path(worktree_path).resolve(strict=False)
        _validate_worktree_path(repo, resolved_worktree)
        if not resolved_worktree.exists() and not resolved_worktree.is_symlink():
            return

        result = self._run_git(
            ("worktree", "remove", "--force", str(resolved_worktree)),
            cwd=repo,
        )
        if result.returncode != 0:
            raise WorktreeCleanupError(_failure_message(result))
        _remove_empty_parents(resolved_worktree, repo / ".worktrees")

    def _add_worktree(
        self,
        repo: Path,
        worktree_path: Path,
        branch_name: str,
        *,
        base_branch: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        branch_result = self._run_git(
            ("rev-parse", "--verify", f"refs/heads/{branch_name}"),
            cwd=repo,
        )
        if branch_result.returncode == 0:
            return self._run_git(
                ("worktree", "add", str(worktree_path), branch_name),
                cwd=repo,
            )
        return self._run_git(
            (
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_path),
                *((base_branch,) if base_branch is not None else ()),
            ),
            cwd=repo,
        )

    def _prepare_integration_branch(self, repo: Path, branch_name: str) -> None:
        branch_result = self._run_git(
            ("rev-parse", "--verify", f"refs/heads/{branch_name}"),
            cwd=repo,
        )
        fetch_result = self._run_git(
            (
                "fetch",
                "origin",
                f"+refs/heads/{branch_name}:refs/heads/{branch_name}",
            ),
            cwd=repo,
            env=self._git_bot_env(),
        )
        if fetch_result.returncode == 0 or branch_result.returncode == 0:
            return
        create_result = self._run_git(("branch", branch_name, "HEAD"), cwd=repo)
        if create_result.returncode != 0:
            raise WorktreeCreationError(_failure_message(create_result))

    def _ensure_local_branch(self, repo: Path, branch_name: str) -> None:
        branch_result = self._run_git(
            ("rev-parse", "--verify", f"refs/heads/{branch_name}"),
            cwd=repo,
        )
        if branch_result.returncode == 0:
            return

        fetch_result = self._run_git(
            ("fetch", "origin", f"{branch_name}:{branch_name}"),
            cwd=repo,
        )
        if fetch_result.returncode != 0:
            raise WorktreeCreationError(_failure_message(fetch_result))

    def _run_git(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        kwargs: dict[str, object] = {
            "cwd": cwd,
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": self._default_timeout_seconds,
        }
        if env is not None:
            kwargs["env"] = dict(env)
        return subprocess.run(("git", *args), **kwargs)

    def _git_bot_env(self) -> Mapping[str, str] | None:
        if self._credentials is None:
            return None
        return self._credentials.git_env(GH_ROLE_BOT)

    def _required_git_bot_env(self) -> Mapping[str, str]:
        env = self._git_bot_env()
        if env is None:
            raise PushError(
                "GH_BOT_TOKEN is required to push code as the system user"
            )
        return env


def _failure_message(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    return output or f"git exited with return code {result.returncode}"


def _unstage_generated_paths(worktree: Path, *, timeout_seconds: int) -> None:
    existing = [path for path in _GENERATED_COMMIT_EXCLUDES if (worktree / path).exists()]
    if not existing:
        return
    result = _run_command(
        ("git", "reset", "--quiet", "--", *existing),
        cwd=worktree,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise GitAdapterError(_failure_message(result))


def _ensure_origin_remote(
    worktree: Path,
    *,
    timeout_seconds: int,
    credentials: GitHubCredentials | None = None,
) -> None:
    result = _run_command(
        ("git", "remote", "get-url", "origin"),
        cwd=worktree,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode == 0 and result.stdout.strip():
        return
    if (
        credentials is None
        or not credentials.has_token_for(GH_ROLE_ADMIN)
        or not credentials.has_token_for(GH_ROLE_BOT)
    ):
        raise PushError(
            "GH_ADMIN_TOKEN and GH_BOT_TOKEN are required to create a missing "
            "GitHub origin without using ambient local credentials"
        )
    repo_root = _primary_repo_root(worktree, timeout_seconds=timeout_seconds)
    repo_name = _github_repo_name(repo_root)
    admin_env = credentials.gh_env(GH_ROLE_ADMIN)
    create_result = _run_command(
        (
            "gh",
            "repo",
            "create",
            repo_name,
            "--private",
            "--source",
            str(repo_root),
            "--remote",
            "origin",
        ),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
        env=admin_env,
    )
    if create_result.returncode != 0:
        raise PushError(
            "git remote 'origin' is not configured for "
            f"{worktree}, and automatic GitHub repo creation failed: "
            f"{_failure_message(create_result)}"
        )

    _authorize_bot_for_created_repo(
        repo_root,
        timeout_seconds=timeout_seconds,
        credentials=credentials,
    )

    default_branch = _default_branch(repo_root, timeout_seconds=timeout_seconds)
    push_env = credentials.git_env(GH_ROLE_BOT)
    push_default_result = _run_command(
        ("git", "push", "-u", "origin", default_branch),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
        env=push_env,
    )
    if push_default_result.returncode != 0:
        raise PushError(_failure_message(push_default_result))


def _authorize_bot_for_created_repo(
    repo_root: Path,
    *,
    timeout_seconds: int,
    credentials: GitHubCredentials,
) -> None:
    admin_env = credentials.gh_env(GH_ROLE_ADMIN)
    bot_env = credentials.gh_env(GH_ROLE_BOT)
    repo_result = _run_command(
        ("gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
        env=admin_env,
    )
    repo_full_name = repo_result.stdout.strip()
    if repo_result.returncode != 0 or not _is_github_repo_full_name(repo_full_name):
        raise PushError(
            "GitHub repository was created, but automatic bot collaborator "
            f"setup could not identify it: {_failure_message(repo_result)}"
        )

    bot_result = _run_command(
        ("gh", "api", "user", "--jq", ".login"),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
        env=bot_env,
    )
    bot_login = bot_result.stdout.strip()
    if bot_result.returncode != 0 or _SAFE_GITHUB_NAME.fullmatch(bot_login) is None:
        raise PushError(
            "GitHub repository was created, but automatic bot collaborator "
            f"setup could not identify the bot account: {_failure_message(bot_result)}"
        )

    owner_login = repo_full_name.split("/", 1)[0]
    if owner_login.casefold() != bot_login.casefold():
        add_result = _run_command(
            (
                "gh",
                "api",
                "--method",
                "PUT",
                f"repos/{repo_full_name}/collaborators/{bot_login}",
                "--jq",
                ".id",
            ),
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
            env=admin_env,
        )
        if add_result.returncode != 0:
            raise PushError(
                "GitHub repository was created, but automatic bot collaborator "
                f"setup failed while adding {bot_login}: {_failure_message(add_result)}"
            )

        invitation_id = add_result.stdout.strip()
        if invitation_id:
            if not invitation_id.isdigit():
                raise PushError(
                    "GitHub repository was created, but automatic bot collaborator "
                    "setup returned an invalid repository invitation id"
                )
            accept_result = _run_command(
                (
                    "gh",
                    "api",
                    "--method",
                    "PATCH",
                    f"user/repository_invitations/{invitation_id}",
                ),
                cwd=repo_root,
                timeout_seconds=timeout_seconds,
                env=bot_env,
            )
            if accept_result.returncode != 0:
                raise PushError(
                    "GitHub repository was created, but the bot could not accept "
                    f"its collaborator invitation: {_failure_message(accept_result)}"
                )

    # Bot PAT authentication is sent as an HTTPS header, not an SSH credential.
    remote_result = _run_command(
        (
            "git",
            "remote",
            "set-url",
            "origin",
            f"https://github.com/{repo_full_name}.git",
        ),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    if remote_result.returncode != 0:
        raise PushError(
            "GitHub repository was created, but automatic bot collaborator "
            f"setup could not configure its HTTPS remote: {_failure_message(remote_result)}"
        )


def _is_github_repo_full_name(value: str) -> bool:
    if value.count("/") != 1:
        return False
    return all(_SAFE_GITHUB_NAME.fullmatch(part) is not None for part in value.split("/"))


def _primary_repo_root(worktree: Path, *, timeout_seconds: int) -> Path:
    result = _run_command(
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        cwd=worktree,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise PushError(_failure_message(result))
    git_common_dir = Path(result.stdout.strip())
    if git_common_dir.name == ".git":
        return git_common_dir.parent.resolve(strict=True)
    return worktree


def _github_repo_name(repo_root: Path) -> str:
    repo_name = repo_root.name.strip()
    if not repo_name or repo_name in {".", ".."}:
        raise PushError(f"cannot infer GitHub repository name from {repo_root}")
    return repo_name


def _default_branch(repo_root: Path, *, timeout_seconds: int) -> str:
    result = _run_command(
        ("git", "symbolic-ref", "--quiet", "--short", "HEAD"),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise PushError(
            "cannot infer default branch to initialize GitHub repository: "
            f"{_failure_message(result)}"
        )
    return branch


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {
        "cwd": cwd,
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": timeout_seconds,
    }
    if env is not None:
        kwargs["env"] = dict(env)
    return subprocess.run(tuple(command), **kwargs)


def _validate_safe_ref_component(value: str, label: str) -> None:
    if _SAFE_REF_COMPONENT.fullmatch(value) is None:
        raise WorktreeCreationError(f"unsafe {label}: {value}")


def _validate_push_branch(branch_name: str) -> None:
    if branch_name in _PROTECTED_BRANCHES:
        raise PushError(f"refusing to push protected branch: {branch_name}")
    if not branch_name.startswith("agent/"):
        raise PushError(f"refusing to push non-agent branch: {branch_name}")

    parts = branch_name.split("/")
    if len(parts) != 3 or parts[0] != "agent":
        raise PushError(f"unsafe agent branch name: {branch_name}")
    for label, value in (("ticket_key", parts[1]), ("short_lock_id", parts[2])):
        if _SAFE_REF_COMPONENT.fullmatch(value) is None:
            raise PushError(f"unsafe {label} in branch name: {branch_name}")


def _validate_integration_branch(
    branch_name: str,
    error_type: type[WorktreeCreationError | PushError],
) -> None:
    parts = branch_name.split("/")
    if (
        len(parts) != 2
        or parts[0] != "integration"
        or _SAFE_REF_COMPONENT.fullmatch(parts[1]) is None
    ):
        raise error_type(f"unsafe integration branch name: {branch_name}")


def _validate_worktree_path(repo: Path, worktree_path: Path) -> None:
    worktrees_root = repo / ".worktrees"
    if worktrees_root.is_symlink():
        raise WorktreeCleanupError(
            f"worktrees root must not be a symlink: {worktrees_root}"
        )
    resolved_worktrees_root = worktrees_root.resolve(strict=False)
    resolved_worktree_path = worktree_path.resolve(strict=False)
    try:
        relative = resolved_worktree_path.relative_to(resolved_worktrees_root)
    except ValueError as exc:
        raise WorktreeCleanupError(
            f"worktree path is outside {worktrees_root}: {worktree_path}"
        ) from exc
    if not relative.parts:
        raise WorktreeCleanupError(f"refusing to remove worktrees root: {worktree_path}")


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    stop = stop_at.resolve(strict=False)
    current = path.parent.resolve(strict=False)
    while current != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent

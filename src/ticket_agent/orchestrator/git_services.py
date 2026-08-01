"""Git-backed pull request service implementations."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Protocol

from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.domain.errors import PullRequestCreationError
from ticket_agent.goal.journal import GoalActionJournal, ProbeResult
from ticket_agent.goal.types import LoopState, digest
from ticket_agent.github import GH_ROLE_BOT, GitHubCredentials
from ticket_agent.orchestrator.state import TicketState
from ticket_agent.redaction import redact_local_paths


class GitPullRequestPort(Protocol):
    def commit(self, worktree_path: str | Path, message: str) -> str: ...

    def push(self, worktree_path: str | Path, branch_name: str) -> None: ...

    def ensure_pull_request_base(
        self,
        worktree_path: str | Path,
        branch_name: str,
    ) -> None: ...

    def staged_tree_digest(self, worktree_path: str | Path) -> str: ...

    def current_head(self, worktree_path: str | Path) -> tuple[str, str]: ...

    def remote_ref_sha(
        self,
        worktree_path: str | Path,
        branch_name: str,
    ) -> str | None: ...


class GitWorktreeCleanupPort(Protocol):
    def cleanup_worktree(
        self,
        repo_path: str | Path,
        worktree_path: str | Path,
    ) -> None: ...


class PullRequestOpener(Protocol):
    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str: ...

    def find_open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
    ) -> str | None: ...


class GitService:
    """Commit, push, and open a pull request for completed ticket work."""

    def __init__(
        self,
        *,
        git: GitPullRequestPort | None = None,
        pull_request_opener: PullRequestOpener | None = None,
        base_branch: str = "main",
        credentials: GitHubCredentials | None = None,
        action_journal: GoalActionJournal | None = None,
    ) -> None:
        self._git = git or GitAdapter(credentials=credentials)
        self._pull_request_opener = pull_request_opener or GhPullRequestOpener(
            credentials=credentials,
        )
        self._base_branch = base_branch
        self._action_journal = action_journal

    async def open_pull_request(self, state: TicketState) -> str:
        if state.pull_request_url:
            return state.pull_request_url

        worktree_path = _required_worktree_path(state)
        branch_name = _required_branch_name(state)

        commit_message = _commit_message(state)
        base_branch = state.pull_request_base_branch or self._base_branch
        if self._action_journal is None:
            self._git.commit(worktree_path, commit_message)
            if state.pull_request_base_branch:
                self._git.ensure_pull_request_base(worktree_path, base_branch)
            self._git.push(worktree_path, branch_name)
            return self._pull_request_opener.open_pull_request(
                worktree_path=worktree_path,
                branch_name=branch_name,
                base_branch=base_branch,
                title=_pull_request_title(state),
                body=_pull_request_body(state),
            )
        if state.goal_id is None:
            raise PullRequestCreationError(
                "goal_id is required to journal Git delivery"
            )

        loop_state = LoopState(
            goal_id=state.goal_id,
            contract_version=1,
            phase="delivering",
            iteration=state.implementation_attempts,
        )
        tree_digest = self._git.staged_tree_digest(worktree_path)

        def probe_commit() -> ProbeResult[str]:
            sha, current_tree = self._git.current_head(worktree_path)
            return ProbeResult(
                found=current_tree == tree_digest,
                value=sha if current_tree == tree_digest else None,
                result_identity=sha if current_tree == tree_digest else None,
            )

        commit = await self._action_journal.execute(
            loop_state,
            operation="git_commit",
            natural_key=f"{branch_name}:{tree_digest}",
            request={
                "branch": branch_name,
                "tree_digest": tree_digest,
                "message": commit_message,
            },
            effect=lambda: self._git.commit(worktree_path, commit_message),
            probe=probe_commit,
            restore=lambda sha: sha,
        )
        if commit.value is None:
            raise PullRequestCreationError("journaled commit returned no SHA")
        candidate_sha = commit.value

        if state.pull_request_base_branch:
            self._git.ensure_pull_request_base(worktree_path, base_branch)

        def push_effect() -> str:
            self._git.push(worktree_path, branch_name)
            return candidate_sha

        def probe_push() -> ProbeResult[str]:
            remote_sha = self._git.remote_ref_sha(worktree_path, branch_name)
            return ProbeResult(
                found=remote_sha == candidate_sha,
                value=candidate_sha if remote_sha == candidate_sha else None,
                result_identity=candidate_sha if remote_sha == candidate_sha else None,
            )

        await self._action_journal.execute(
            loop_state,
            operation="git_push",
            natural_key=f"origin:{branch_name}:{candidate_sha}",
            request={
                "remote": "origin",
                "branch": branch_name,
                "sha": candidate_sha,
            },
            effect=push_effect,
            probe=probe_push,
            restore=lambda sha: sha,
        )

        title = _pull_request_title(state)
        body = _pull_request_body(state)

        def open_pr() -> str:
            return self._pull_request_opener.open_pull_request(
                worktree_path=worktree_path,
                branch_name=branch_name,
                base_branch=base_branch,
                title=title,
                body=body,
            )

        def probe_pr() -> ProbeResult[str]:
            url = self._pull_request_opener.find_open_pull_request(
                worktree_path=worktree_path,
                branch_name=branch_name,
                base_branch=base_branch,
            )
            return ProbeResult(
                found=url is not None,
                value=url,
                result_identity=url,
            )

        pr = await self._action_journal.execute(
            loop_state,
            operation="pr_create",
            natural_key=f"{state.repository}:{branch_name}",
            request={
                "repository": state.repository,
                "head": branch_name,
                "base": base_branch,
                "title_digest": digest(title),
                "body_digest": digest(body),
            },
            effect=open_pr,
            probe=probe_pr,
            restore=lambda url: url,
        )
        if pr.value is None:
            raise PullRequestCreationError("journaled PR creation returned no URL")
        return pr.value


class WorktreeCleanupService:
    """Remove terminal ticket worktrees from the local repository."""

    def __init__(
        self,
        *,
        git: GitWorktreeCleanupPort | None = None,
        credentials: GitHubCredentials | None = None,
    ) -> None:
        self._git = git or GitAdapter(credentials=credentials)

    def cleanup(self, state: TicketState) -> None:
        repo_path = _worktree_cleanup_repo_path(state)
        worktree_path = _worktree_path(state)
        if repo_path is None or worktree_path is None:
            return
        self._git.cleanup_worktree(repo_path, worktree_path)


class GhPullRequestOpener:
    """Open pull requests through the GitHub CLI."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 300,
        credentials: GitHubCredentials | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._credentials = credentials

    def open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str:
        title = redact_local_paths(title, [worktree_path])
        body = redact_local_paths(body, [worktree_path])
        existing_url = self._existing_pull_request_url(
            worktree_path=worktree_path,
            branch_name=branch_name,
            base_branch=base_branch,
        )
        if existing_url is not None:
            return existing_url

        command = (
            "gh",
            "pr",
            "create",
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--title",
            title,
            "--body",
            body,
        )
        try:
            result = subprocess.run(
                command,
                cwd=worktree_path,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                **self._gh_env_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise PullRequestCreationError(
                f"gh pr create timed out after {self._timeout_seconds} seconds"
            ) from exc

        if result.returncode != 0:
            raise PullRequestCreationError(_subprocess_failure_message(result))

        url = result.stdout.strip()
        if not url:
            raise PullRequestCreationError("gh pr create did not return a PR URL")
        return url

    def find_open_pull_request(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
    ) -> str | None:
        return self._existing_pull_request_url(
            worktree_path=worktree_path,
            branch_name=branch_name,
            base_branch=base_branch,
        )

    def _existing_pull_request_url(
        self,
        *,
        worktree_path: Path,
        branch_name: str,
        base_branch: str,
    ) -> str | None:
        command = (
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--json",
            "url",
            "--jq",
            ".[0].url",
        )
        try:
            result = subprocess.run(
                command,
                cwd=worktree_path,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                **self._gh_env_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None
        url = result.stdout.strip()
        if not url or url == "null":
            return None
        return url

    def _gh_env_kwargs(self) -> dict[str, Any]:
        env = (
            None
            if self._credentials is None
            else self._credentials.gh_env(GH_ROLE_BOT)
        )
        if env is None:
            raise PullRequestCreationError(
                "GH_BOT_TOKEN is required to open pull requests as the system user"
            )
        return {"env": env}


def _worktree_path(state: TicketState) -> Path | None:
    if not state.worktree_path:
        return None
    return Path(state.worktree_path)


def _worktree_cleanup_repo_path(state: TicketState) -> Path | None:
    if not state.repo_path:
        return None
    return Path(state.repo_path)


def _required_worktree_path(state: TicketState) -> Path:
    worktree_path = _worktree_path(state)
    if worktree_path is None:
        raise PullRequestCreationError(
            "worktree_path is required to open pull request"
        )
    return worktree_path


def _required_branch_name(state: TicketState) -> str:
    if not state.branch_name:
        raise PullRequestCreationError(
            "branch_name is required to open pull request"
        )
    return state.branch_name


def _commit_message(state: TicketState) -> str:
    return redact_local_paths(
        f"{state.ticket_key}: {state.summary}",
        _state_local_paths(state),
    )


def _pull_request_title(state: TicketState) -> str:
    return _commit_message(state)


def _pull_request_body(state: TicketState) -> str:
    parts = [
        f"Ticket: {state.ticket_key}",
        f"Summary: {state.summary}",
    ]
    if state.description:
        parts.extend(("", state.description))
    return redact_local_paths("\n".join(parts), _state_local_paths(state))


def _state_local_paths(state: TicketState) -> tuple[str, ...]:
    return tuple(
        path for path in (state.worktree_path, state.repo_path) if path
    )


def _subprocess_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    return output or f"gh pr create exited with return code {result.returncode}"


__all__ = [
    "GhPullRequestOpener",
    "GitPullRequestPort",
    "GitWorktreeCleanupPort",
    "GitService",
    "PullRequestOpener",
    "WorktreeCleanupService",
]

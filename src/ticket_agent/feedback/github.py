"""GitHub PR feedback polling and ticket follow-up execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace
from hashlib import sha256
from inspect import isawaitable
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any, Protocol

from ticket_agent.adapters.local.git_adapter import GitAdapter
from ticket_agent.jira.work_item_loader import JiraWorkItemLoader
from ticket_agent.orchestrator.git_services import WorktreeCleanupService
from ticket_agent.orchestrator.runner import TicketWorkItem
from ticket_agent.orchestrator.state import TicketState

EventEmitter = Callable[[str, dict[str, Any]], Any]

EVENT_FEEDBACK_POLL_STARTED = "feedback.poll_started"
EVENT_FEEDBACK_ENQUEUED = "feedback.enqueued"
EVENT_FEEDBACK_POLL_COMPLETED = "feedback.poll_completed"
EVENT_FEEDBACK_POLL_FAILED = "feedback.poll_failed"
EVENT_FEEDBACK_WORKER_STARTED = "feedback.worker_started"
EVENT_FEEDBACK_WORKER_COMPLETED = "feedback.worker_completed"
EVENT_FEEDBACK_WORKER_FAILED = "feedback.worker_failed"

_TICKET_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]*-\d+)\b")


@dataclass(frozen=True, slots=True)
class FeedbackItem:
    """Actionable feedback collected from an open agent pull request."""

    ticket_key: str
    repo_path: str
    branch_name: str
    pull_request_url: str
    feedback_text: str
    fingerprint: str


class FeedbackClient(Protocol):
    """Boundary that finds actionable PR feedback."""

    def find_feedback(self) -> Sequence[FeedbackItem]:
        """Return feedback items that should be addressed by the agent."""


class FeedbackStore(Protocol):
    """Persistence boundary for feedback fingerprints."""

    def seen(self, fingerprint: str) -> bool:
        """Return True when feedback has already been enqueued or handled."""

    def mark_seen(self, fingerprint: str) -> None:
        """Persist a feedback fingerprint."""


class TicketRunner(Protocol):
    async def run_ticket(self, work_item: TicketWorkItem) -> TicketState:
        """Run one prepared work item."""


class FeedbackWorktreeFactory(Protocol):
    def create_worktree_for_branch(
        self,
        repo_path: str | Path,
        ticket_key: str,
        branch_name: str,
        short_lock_id: str,
    ) -> Any:
        """Create a worktree for an existing PR branch."""


class SQLiteFeedbackStore:
    """SQLite store for processed GitHub feedback fingerprints."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_fingerprints (
                fingerprint TEXT PRIMARY KEY
            )
            """
        )
        self._connection.commit()

    def seen(self, fingerprint: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM feedback_fingerprints WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def mark_seen(self, fingerprint: str) -> None:
        self._connection.execute(
            "INSERT OR IGNORE INTO feedback_fingerprints (fingerprint) VALUES (?)",
            (fingerprint,),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class GitHubFeedbackPoller:
    """Poll GitHub feedback and enqueue unseen items."""

    def __init__(
        self,
        *,
        client: FeedbackClient,
        store: FeedbackStore,
        queue: asyncio.Queue[FeedbackItem],
        poll_interval_seconds: float = 60.0,
        emit: EventEmitter | None = None,
        clock: Callable[[float], Any] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._client = client
        self._store = store
        self._queue = queue
        self._poll_interval = float(poll_interval_seconds)
        self._emit_fn = emit
        self._clock = clock or asyncio.sleep

    async def poll_once(self) -> int:
        await self._emit(EVENT_FEEDBACK_POLL_STARTED)
        try:
            items = self._client.find_feedback()
        except Exception as exc:
            await self._emit(EVENT_FEEDBACK_POLL_FAILED, error=_error_message(exc))
            raise

        enqueued = 0
        for item in items:
            if self._store.seen(item.fingerprint):
                continue
            await self._queue.put(item)
            self._store.mark_seen(item.fingerprint)
            enqueued += 1
            await self._emit(
                EVENT_FEEDBACK_ENQUEUED,
                ticket_key=item.ticket_key,
                pull_request_url=item.pull_request_url,
                branch_name=item.branch_name,
            )

        await self._emit(
            EVENT_FEEDBACK_POLL_COMPLETED,
            considered=len(items),
            enqueued=enqueued,
        )
        return enqueued

    async def run_forever(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await self._sleep(self._poll_interval)

    async def _sleep(self, seconds: float) -> None:
        result = self._clock(seconds)
        if isawaitable(result):
            await result

    async def _emit(self, event_name: str, **payload: Any) -> None:
        if self._emit_fn is None:
            return
        result = self._emit_fn(event_name, payload)
        if isawaitable(result):
            await result


class FeedbackWorker:
    """Consume GitHub feedback items and run ticket follow-ups."""

    def __init__(
        self,
        queue: asyncio.Queue[FeedbackItem],
        coordinator: "FeedbackExecutionCoordinator",
        *,
        emit: EventEmitter | None = None,
        stop_on_error: bool = False,
    ) -> None:
        self._queue = queue
        self._coordinator = coordinator
        self._emit_fn = emit
        self._stop_on_error = stop_on_error

    async def run_once(self) -> bool:
        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False

        try:
            await self._emit(
                EVENT_FEEDBACK_WORKER_STARTED,
                ticket_key=item.ticket_key,
                pull_request_url=item.pull_request_url,
            )
            try:
                await self._coordinator.run_feedback(item)
            except Exception as exc:
                await self._emit(
                    EVENT_FEEDBACK_WORKER_FAILED,
                    ticket_key=item.ticket_key,
                    pull_request_url=item.pull_request_url,
                    error=_error_message(exc),
                )
                if self._stop_on_error:
                    raise
            else:
                await self._emit(
                    EVENT_FEEDBACK_WORKER_COMPLETED,
                    ticket_key=item.ticket_key,
                    pull_request_url=item.pull_request_url,
                )
            return True
        finally:
            self._queue.task_done()

    async def run_forever(self, poll_interval_s: float = 1.0) -> None:
        while True:
            processed = await self.run_once()
            if not processed:
                await asyncio.sleep(poll_interval_s)

    async def _emit(self, event_name: str, **payload: Any) -> None:
        if self._emit_fn is None:
            return
        result = self._emit_fn(event_name, payload)
        if isawaitable(result):
            await result


class FeedbackExecutionCoordinator:
    """Run the normal ticket graph on the existing PR branch with feedback context."""

    def __init__(
        self,
        *,
        loader: JiraWorkItemLoader,
        runner: TicketRunner,
        worktree_factory: FeedbackWorktreeFactory | None = None,
        worktree_cleaner: WorktreeCleanupService | None = None,
    ) -> None:
        self._loader = loader
        self._runner = runner
        self._worktree_factory = worktree_factory or GitAdapter()
        self._worktree_cleaner = worktree_cleaner or WorktreeCleanupService()

    async def run_feedback(self, item: FeedbackItem) -> TicketState:
        base = await self._loader.load(item.ticket_key)
        short_id = item.fingerprint[:8]
        worktree = self._worktree_factory.create_worktree_for_branch(
            item.repo_path,
            item.ticket_key,
            item.branch_name,
            short_id,
        )
        work_item = replace(
            base,
            description=_description_with_feedback(base, item),
            worktree_path=str(worktree.worktree_path),
            branch_name=item.branch_name,
        )
        try:
            return await self._runner.run_ticket(work_item)
        finally:
            self._worktree_cleaner.cleanup(
                TicketState(
                    ticket_key=item.ticket_key,
                    summary=base.summary,
                    repository=base.repository,
                    repo_path=item.repo_path,
                    worktree_path=str(worktree.worktree_path),
                )
            )


class GhCliFeedbackClient:
    """Find feedback on open GitHub PRs through the authenticated gh user."""

    def __init__(
        self,
        *,
        repo_paths: Sequence[str | Path],
        ignore_self_comments: bool = False,
        timeout_seconds: int = 120,
    ) -> None:
        self._repo_paths = tuple(Path(path) for path in repo_paths)
        self._ignore_self_comments = ignore_self_comments
        self._timeout_seconds = timeout_seconds

    def find_feedback(self) -> Sequence[FeedbackItem]:
        actor = self._current_login() if self._ignore_self_comments else ""
        items: list[FeedbackItem] = []
        for repo_path in self._repo_paths:
            for pr in self._open_agent_prs(repo_path):
                feedback = self._feedback_for_pr(repo_path, pr, actor)
                if not feedback:
                    continue
                ticket_key = _ticket_key(pr)
                branch_name = str(pr.get("headRefName") or "")
                url = str(pr.get("url") or "")
                text = _format_feedback(pr, feedback)
                items.append(
                    FeedbackItem(
                        ticket_key=ticket_key,
                        repo_path=str(repo_path),
                        branch_name=branch_name,
                        pull_request_url=url,
                        feedback_text=text,
                        fingerprint=_fingerprint(url, branch_name, text),
                    )
                )
        return items

    def _current_login(self) -> str:
        result = self._run(("gh", "api", "user", "--jq", ".login"), cwd=None)
        return result.stdout.strip()

    def _open_agent_prs(self, repo_path: Path) -> list[dict[str, Any]]:
        result = self._run(
            (
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--json",
                "number,title,headRefName,url",
            ),
            cwd=repo_path,
        )
        data = json.loads(result.stdout or "[]")
        if not isinstance(data, list):
            return []
        return [
            pr
            for pr in data
            if isinstance(pr, dict)
            and str(pr.get("headRefName") or "").startswith("agent/")
            and _ticket_key(pr)
        ]

    def _feedback_for_pr(
        self,
        repo_path: Path,
        pr: dict[str, Any],
        actor: str,
    ) -> list[dict[str, str]]:
        number = str(pr.get("number"))
        view = self._run(
            ("gh", "pr", "view", number, "--json", "comments,reviews"),
            cwd=repo_path,
        )
        payload = json.loads(view.stdout or "{}")
        feedback = _top_level_feedback(payload, actor)

        repo = self._run(
            ("gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"),
            cwd=repo_path,
        ).stdout.strip()
        if repo:
            comments = self._run_json(
                ("gh", "api", f"repos/{repo}/pulls/{number}/comments"),
                cwd=repo_path,
            )
            feedback.extend(_review_comment_feedback(comments, actor))
        return feedback

    def _run_json(self, command: Sequence[str], *, cwd: Path) -> Any:
        result = self._run(command, cwd=cwd)
        return json.loads(result.stdout or "null")

    def _run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            tuple(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(_subprocess_failure_message(result))
        return result


def _description_with_feedback(base: TicketWorkItem, item: FeedbackItem) -> str:
    return "\n\n".join(
        part
        for part in (
            base.description,
            (
                "Pull request feedback to address:\n"
                f"PR: {item.pull_request_url}\n"
                f"Branch: {item.branch_name}\n\n"
                f"{item.feedback_text}"
            ),
        )
        if part
    )


def _ticket_key(pr: dict[str, Any]) -> str:
    for value in (pr.get("title"), pr.get("headRefName"), pr.get("url")):
        match = _TICKET_KEY_RE.search(str(value or ""))
        if match:
            return match.group(1)
    return ""


def _top_level_feedback(payload: dict[str, Any], actor: str) -> list[dict[str, str]]:
    feedback: list[dict[str, str]] = []
    for comment in payload.get("comments") or []:
        item = _feedback_entry(comment, actor)
        if item is not None:
            feedback.append(item)
    for review in payload.get("reviews") or []:
        item = _feedback_entry(review, actor)
        if item is not None:
            feedback.append(item)
    return feedback


def _review_comment_feedback(payload: Any, actor: str) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    feedback: list[dict[str, str]] = []
    for comment in payload:
        item = _feedback_entry(comment, actor)
        if item is not None:
            path = str(comment.get("path") or "")
            if path:
                item["location"] = path
            feedback.append(item)
    return feedback


def _feedback_entry(payload: dict[str, Any], actor: str) -> dict[str, str] | None:
    body = str(payload.get("body") or "").strip()
    if not body:
        return None
    author = payload.get("author") or payload.get("user") or {}
    login = str(author.get("login") or "")
    if login and login == actor:
        return None
    return {
        "author": login or "unknown",
        "body": body,
    }


def _format_feedback(pr: dict[str, Any], feedback: Sequence[dict[str, str]]) -> str:
    lines = [f"PR title: {pr.get('title')}", ""]
    for index, item in enumerate(feedback, start=1):
        location = f" ({item['location']})" if item.get("location") else ""
        lines.append(f"{index}. {item['author']}{location}:")
        lines.append(item["body"])
        lines.append("")
    return "\n".join(lines).strip()


def _fingerprint(url: str, branch_name: str, text: str) -> str:
    return sha256(f"{url}\0{branch_name}\0{text}".encode("utf-8")).hexdigest()


def _subprocess_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip() or (
        f"{result.args!r} exited with {result.returncode}"
    )


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


__all__ = [
    "EVENT_FEEDBACK_ENQUEUED",
    "EVENT_FEEDBACK_POLL_COMPLETED",
    "EVENT_FEEDBACK_POLL_FAILED",
    "EVENT_FEEDBACK_POLL_STARTED",
    "EVENT_FEEDBACK_WORKER_COMPLETED",
    "EVENT_FEEDBACK_WORKER_FAILED",
    "EVENT_FEEDBACK_WORKER_STARTED",
    "FeedbackExecutionCoordinator",
    "FeedbackItem",
    "FeedbackWorker",
    "GhCliFeedbackClient",
    "GitHubFeedbackPoller",
    "SQLiteFeedbackStore",
]

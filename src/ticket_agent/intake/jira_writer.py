"""Write a confirmed :class:`Proposal` to Jira."""

from __future__ import annotations

from dataclasses import dataclass, field

from ticket_agent.domain.intake import IntakeMode, Proposal, TicketSpec
from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_REPOSITORY,
    FIELD_REPO_PATH,
    FIELD_SLACK_THREAD_TS,
    LABEL_AI_READY,
)
from ticket_agent.jira.models import JiraTicket


@dataclass(frozen=True, slots=True)
class JiraWriteFailure:
    """Per-ticket write failure surfaced by :class:`JiraWriter`."""

    spec: TicketSpec
    reason: str


@dataclass(frozen=True, slots=True)
class JiraWriteResult:
    """Aggregate result of attempting to write a proposal to Jira."""

    project_key: str | None
    created_ticket_keys: tuple[str, ...] = ()
    failed_items: tuple[JiraWriteFailure, ...] = ()
    partial: bool = False
    unsupported_reason: str | None = None

    @property
    def success(self) -> bool:
        return (
            self.unsupported_reason is None
            and not self.failed_items
            and bool(self.created_ticket_keys)
        )


@dataclass
class _WriteContext:
    project_key: str
    created: list[str] = field(default_factory=list)
    failures: list[JiraWriteFailure] = field(default_factory=list)


class JiraWriter:
    """Persist a confirmed proposal as Jira issues with the ai-ready label."""

    def __init__(self, client: JiraClient) -> None:
        self._client = client

    async def write(self, proposal: Proposal) -> JiraWriteResult:
        if proposal.mode == IntakeMode.NEW_PROJECT:
            return JiraWriteResult(
                project_key=proposal.project_key,
                unsupported_reason="creating new Jira projects is not supported in v1",
                partial=True,
                failed_items=tuple(
                    JiraWriteFailure(spec=spec, reason="new_project_unsupported")
                    for spec in proposal.tickets
                ),
            )

        project_key = _required_project_key(proposal)
        if not proposal.tickets:
            return JiraWriteResult(
                project_key=project_key,
                unsupported_reason="proposal has no ticket specs",
            )

        context = _WriteContext(project_key=project_key)
        for spec in proposal.tickets:
            await self._write_one(spec, proposal, context)

        partial = bool(context.created) and bool(context.failures)
        return JiraWriteResult(
            project_key=project_key,
            created_ticket_keys=tuple(context.created),
            failed_items=tuple(context.failures),
            partial=partial,
        )

    async def _write_one(
        self,
        spec: TicketSpec,
        proposal: Proposal,
        context: _WriteContext,
    ) -> None:
        labels = _normalize_labels(spec.labels)
        fields = _build_fields(spec, proposal)

        try:
            ticket = await self._client.create_issue(
                context.project_key,
                summary=spec.summary,
                description=spec.description,
                issue_type=spec.issue_type,
                priority=spec.priority,
                labels=labels,
                fields=fields,
                parent_key=proposal.epic_key,
            )
        except Exception as exc:  # noqa: BLE001 - boundary call
            context.failures.append(
                JiraWriteFailure(spec=spec, reason=_error_message(exc))
            )
            return

        await _ensure_ai_ready_label(self._client, ticket, labels)
        context.created.append(ticket.key)


def _required_project_key(proposal: Proposal) -> str:
    if not proposal.project_key:
        raise ValueError(
            f"proposal {proposal.proposal_id} cannot be written to Jira "
            "without a project_key"
        )
    return proposal.project_key


def _normalize_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in [*labels, LABEL_AI_READY]:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _build_fields(spec: TicketSpec, proposal: Proposal) -> dict[str, object]:
    fields: dict[str, object] = {
        FIELD_AGENT_RETRY_COUNT: 0,
        FIELD_AGENT_CAPABILITIES_NEEDED: list(spec.capabilities_needed),
        FIELD_SLACK_THREAD_TS: proposal.slack_thread_ts,
    }
    if spec.repository:
        fields[FIELD_REPOSITORY] = spec.repository
    if spec.repo_path:
        fields[FIELD_REPO_PATH] = spec.repo_path
    return fields


async def _ensure_ai_ready_label(
    client: JiraClient,
    ticket: JiraTicket,
    expected_labels: list[str],
) -> None:
    missing = [label for label in expected_labels if label not in ticket.labels]
    if not missing:
        return
    await client.add_labels(ticket.key, missing)


def _error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


__all__ = [
    "JiraWriteFailure",
    "JiraWriteResult",
    "JiraWriter",
]

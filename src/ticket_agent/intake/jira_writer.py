"""Write a confirmed :class:`Proposal` to Jira."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ticket_agent.domain.intake import Proposal, TicketSpec
from ticket_agent.jira.client import JiraClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_CAPABILITIES_NEEDED,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_REPOSITORY,
    FIELD_SLACK_CHANNEL,
    FIELD_SLACK_THREAD_TS,
    LABEL_AI_READY,
    LABEL_AI_SEQUENCE_PREFIX,
    LABEL_AI_STEP_PREFIX,
)
from ticket_agent.jira.models import JiraTicket
from ticket_agent.goal.identity import (
    GoalIdentityError,
    goal_label,
    normalize_goal_id,
    validate_goal_labels,
)
from ticket_agent.goal.types import ActionRecord, LoopState, action_id, digest


class GoalSpine(Protocol):
    def reserve_action(
        self,
        state: LoopState,
        action: ActionRecord,
    ) -> ActionRecord: ...

    def mark_in_flight(
        self,
        action_id_value: str,
        *,
        lease_owner: str,
        lease_seconds: int = 300,
        recovery_classification: str | None = None,
    ) -> ActionRecord: ...

    def mark_done(
        self,
        action_id_value: str,
        *,
        result_identity: str | None = None,
        actual_model_cost_usd: float = 0.0,
        recovery_classification: str | None = None,
    ) -> ActionRecord: ...


@dataclass(frozen=True, slots=True)
class JiraWriteFailure:
    """Per-ticket write failure surfaced by :class:`JiraWriter`."""

    spec: TicketSpec
    reason: str


@dataclass(frozen=True, slots=True)
class JiraWriteResult:
    """Aggregate result of attempting to write a proposal to Jira."""

    project_key: str | None
    created_epic_key: str | None = None
    created_ticket_keys: tuple[str, ...] = ()
    execution_ready_ticket_keys: tuple[str, ...] = ()
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
    created_epic_key: str | None = None
    created: list[str] = field(default_factory=list)
    execution_ready: list[str] = field(default_factory=list)
    failures: list[JiraWriteFailure] = field(default_factory=list)
    optional_fields_disabled: bool = False
    unsupported_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _SequentialDelivery:
    sequence_label: str
    base_branch: str
    total_steps: int


class JiraWriter:
    """Persist a confirmed proposal as Jira issues with the ai-ready label."""

    def __init__(
        self,
        client: JiraClient,
        *,
        goal_spine: GoalSpine | None = None,
        component_id: str = "ticket-agent",
    ) -> None:
        self._client = client
        self._goal_spine = goal_spine
        self._component_id = component_id

    async def write(self, proposal: Proposal) -> JiraWriteResult:
        try:
            goal_id = normalize_goal_id(proposal.proposal_id)
            canonical_goal_label = goal_label(goal_id)
            for spec in proposal.tickets:
                validate_goal_labels(
                    [*spec.labels, canonical_goal_label],
                    goal_id=goal_id,
                )
        except GoalIdentityError as exc:
            return JiraWriteResult(
                project_key=_optional_project_key(proposal),
                unsupported_reason=f"invalid goal identity: {exc}",
                failed_items=tuple(
                    JiraWriteFailure(spec=spec, reason="invalid_goal_identity")
                    for spec in proposal.tickets
                ),
            )

        project_key = _optional_project_key(proposal)
        if project_key is None:
            return JiraWriteResult(
                project_key=None,
                unsupported_reason=(
                    "an existing Jira project key is required; creating brand-new "
                    "Jira projects is not supported in v1"
                ),
                failed_items=tuple(
                    JiraWriteFailure(spec=spec, reason="missing_project_key")
                    for spec in proposal.tickets
                ),
            )

        if not proposal.tickets:
            return JiraWriteResult(
                project_key=project_key,
                unsupported_reason="proposal has no ticket specs",
            )

        context = _WriteContext(project_key=project_key)
        parent_key = proposal.epic_key
        if parent_key is None and len(proposal.tickets) > 1:
            parent_key = await self._create_epic(
                proposal,
                context,
                goal_label_value=canonical_goal_label,
            )
            if parent_key is None:
                return JiraWriteResult(
                    project_key=project_key,
                    created_epic_key=context.created_epic_key,
                    failed_items=tuple(context.failures),
                    partial=bool(context.created_epic_key),
                    unsupported_reason=context.unsupported_reason,
                )

        delivery = _sequential_delivery(parent_key, proposal)
        for index, spec in enumerate(proposal.tickets):
            await self._write_one(
                spec,
                proposal,
                context,
                parent_key=parent_key,
                execution_ready=_ticket_starts_execution_ready(proposal, index),
                delivery=delivery,
                step_index=index,
                goal_id=goal_id,
                goal_label_value=canonical_goal_label,
            )
            if context.unsupported_reason is not None:
                break

        partial = bool(context.created or context.created_epic_key) and bool(
            context.failures
        )
        return JiraWriteResult(
            project_key=project_key,
            created_epic_key=context.created_epic_key,
            created_ticket_keys=tuple(context.created),
            execution_ready_ticket_keys=tuple(context.execution_ready),
            failed_items=tuple(context.failures),
            partial=partial,
            unsupported_reason=context.unsupported_reason,
        )

    async def _create_epic(
        self,
        proposal: Proposal,
        context: _WriteContext,
        *,
        goal_label_value: str,
    ) -> str | None:
        summary = proposal.epic_summary or proposal.title
        description = proposal.epic_description or proposal.summary
        try:
            epic = await self._client.create_issue(
                context.project_key,
                summary=summary,
                description=description,
                issue_type="Epic",
                labels=[goal_label_value],
                fields={},
            )
        except Exception as exc:  # noqa: BLE001 - boundary call
            reason = f"epic_create_failed: {_error_message(exc)}"
            if _is_invalid_project_create_error(exc):
                context.unsupported_reason = _invalid_project_reason(
                    context.project_key
                )
            context.failures.extend(
                JiraWriteFailure(spec=spec, reason=reason)
                for spec in proposal.tickets
            )
            return None
        context.created_epic_key = epic.key
        return epic.key

    async def _write_one(
        self,
        spec: TicketSpec,
        proposal: Proposal,
        context: _WriteContext,
        *,
        parent_key: str | None,
        execution_ready: bool,
        delivery: _SequentialDelivery | None,
        step_index: int,
        goal_id: str,
        goal_label_value: str,
    ) -> None:
        labels = _normalize_labels(
            [
                *spec.labels,
                *_delivery_labels(delivery, step_index),
                goal_label_value,
            ],
            execution_ready=False,
        )
        description = _description_with_delivery(spec.description, delivery, step_index)
        fields = (
            {}
            if context.optional_fields_disabled
            else _build_fields(spec, proposal)
        )

        try:
            ticket = await self._client.create_issue(
                context.project_key,
                summary=spec.summary,
                description=description,
                issue_type=spec.issue_type,
                priority=spec.priority,
                labels=labels,
                fields=fields,
                parent_key=parent_key,
            )
        except Exception as exc:  # noqa: BLE001 - boundary call
            if _is_invalid_project_create_error(exc):
                context.unsupported_reason = _invalid_project_reason(
                    context.project_key
                )
                context.failures.append(
                    JiraWriteFailure(spec=spec, reason=_error_message(exc))
                )
                return
            if fields and _is_optional_field_create_error(exc):
                context.optional_fields_disabled = True
                try:
                    ticket = await self._client.create_issue(
                        context.project_key,
                        summary=spec.summary,
                        description=description,
                        issue_type=spec.issue_type,
                        priority=spec.priority,
                        labels=labels,
                        fields={},
                        parent_key=parent_key,
                    )
                except Exception as retry_exc:  # noqa: BLE001 - boundary call
                    context.failures.append(
                        JiraWriteFailure(spec=spec, reason=_error_message(retry_exc))
                    )
                    return
            else:
                context.failures.append(
                    JiraWriteFailure(spec=spec, reason=_error_message(exc))
                )
                return

        await _ensure_expected_labels(self._client, ticket, labels)
        context.created.append(ticket.key)
        if execution_ready:
            await self._publish_ai_ready(ticket, goal_id=goal_id)
            context.execution_ready.append(ticket.key)

    async def _publish_ai_ready(self, ticket: JiraTicket, *, goal_id: str) -> None:
        if self._goal_spine is None:
            if LABEL_AI_READY not in ticket.labels:
                await self._client.add_labels(ticket.key, [LABEL_AI_READY])
            return

        natural_key = ticket.key
        record = ActionRecord(
            action_id=action_id(
                goal_id,
                0,
                "jira_write:add_ai_ready",
                natural_key,
            ),
            goal_id=goal_id,
            iteration=0,
            kind="jira_write",
            state="intended",
            operation="jira_write:add_ai_ready",
            natural_key=natural_key,
            request_digest=digest(
                {"ticket_key": ticket.key, "label": LABEL_AI_READY}
            ),
            external=True,
        )
        durable = self._goal_spine.reserve_action(
            LoopState(
                goal_id=goal_id,
                contract_version=1,
                phase="discovering",
            ),
            record,
        )
        if durable.state == "done":
            return

        recovery = None
        current = ticket
        if durable.state in {"in_flight", "failed"} or durable.attempts:
            recovery = "label_presence_probe"
            current = await self._client.get_ticket(ticket.key)
        if LABEL_AI_READY in current.labels:
            self._goal_spine.mark_done(
                durable.action_id,
                result_identity=ticket.key,
                recovery_classification=recovery,
            )
            return

        self._goal_spine.mark_in_flight(
            durable.action_id,
            lease_owner=self._component_id,
            recovery_classification=recovery,
        )
        await self._client.add_labels(ticket.key, [LABEL_AI_READY])
        self._goal_spine.mark_done(
            durable.action_id,
            result_identity=ticket.key,
            recovery_classification=recovery,
        )


def _is_optional_field_create_error(exc: BaseException) -> bool:
    message = _error_message(exc).lower()
    return (
        "cannot be set" in message
        and ("appropriate screen" in message or "unknown" in message)
    )


def _is_invalid_project_create_error(exc: BaseException) -> bool:
    message = _error_message(exc).lower()
    return (
        "valid project is required" in message
        or "no project could be found" in message
    )


def _invalid_project_reason(project_key: str) -> str:
    return (
        f"Jira project `{project_key}` is not valid or not accessible. "
        "Create that Jira Software project first, or reply with edits using "
        "an existing Jira project key."
    )


def _optional_project_key(proposal: Proposal) -> str | None:
    if not proposal.project_key:
        return None
    return proposal.project_key


def _normalize_labels(labels: list[str], *, execution_ready: bool = True) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    requested_labels = [*labels, LABEL_AI_READY] if execution_ready else labels
    for label in requested_labels:
        if label == LABEL_AI_READY and not execution_ready:
            continue
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
    if proposal.slack_channel:
        fields[FIELD_SLACK_CHANNEL] = proposal.slack_channel
    if spec.repository:
        fields[FIELD_REPOSITORY] = spec.repository
    return fields


def _ticket_starts_execution_ready(proposal: Proposal, index: int) -> bool:
    return len(proposal.tickets) <= 1 or index == 0


def _sequential_delivery(
    parent_key: str | None,
    proposal: Proposal,
) -> _SequentialDelivery | None:
    if parent_key is None or len(proposal.tickets) <= 1:
        return None
    normalized_key = parent_key.lower()
    return _SequentialDelivery(
        sequence_label=f"{LABEL_AI_SEQUENCE_PREFIX}{normalized_key}",
        base_branch=f"integration/{normalized_key}",
        total_steps=len(proposal.tickets),
    )


def _delivery_labels(
    delivery: _SequentialDelivery | None,
    step_index: int,
) -> list[str]:
    if delivery is None:
        return []
    return [
        delivery.sequence_label,
        f"{LABEL_AI_STEP_PREFIX}{step_index + 1:04d}",
    ]


def _description_with_delivery(
    description: str,
    delivery: _SequentialDelivery | None,
    step_index: int,
) -> str:
    if delivery is None:
        return description
    metadata = (
        "Delivery workflow:\n"
        f"- Pull request base branch: {delivery.base_branch}\n"
        f"- Delivery sequence: {step_index + 1} of {delivery.total_steps}\n"
        "- Delivery gate: merge this pull request before the next step is released."
    )
    return "\n\n".join(part for part in (description, metadata) if part)


async def _ensure_expected_labels(
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

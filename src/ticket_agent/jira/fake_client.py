"""In-memory Jira client for tests and local execution demos."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from ticket_agent.jira.models import JiraTicket

JiraClientCall = tuple[str, str, object]
JiraOperationFailure = BaseException | Sequence[BaseException | None]
_ConfiguredFailure = BaseException | list[BaseException | None]


class FakeJiraClient:
    """Mutable in-memory implementation of the Jira client protocol."""

    def __init__(
        self,
        tickets: JiraTicket | Iterable[JiraTicket] | Mapping[str, JiraTicket],
        *,
        fail_on: Mapping[str, JiraOperationFailure] | None = None,
    ) -> None:
        self.tickets = _coerce_tickets(tickets)
        self.fail_on = _coerce_failures(fail_on)
        self.calls: list[JiraClientCall] = []
        self.comments: dict[str, list[str]] = {
            ticket_key: [] for ticket_key in self.tickets
        }

    async def get_ticket(self, ticket_key: str) -> JiraTicket:
        """Return a stored ticket by key."""

        self.calls.append(("get_ticket", ticket_key, None))
        self._raise_if_configured("get_ticket")
        return self._ticket(ticket_key)

    async def search_issues(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> Sequence[JiraTicket]:
        """Return stored tickets matching the subset of JQL used in tests."""

        payload = {"jql": jql, "fields": None if fields is None else list(fields)}
        self.calls.append(("search_issues", "", payload))
        self._raise_if_configured("search_issues")
        return [
            ticket
            for ticket in self.tickets.values()
            if _matches_basic_jql(ticket, jql)
        ]

    async def transition_ticket(self, ticket_key: str, status: str) -> None:
        """Set a ticket's current workflow status."""

        self.calls.append(("transition_ticket", ticket_key, status))
        self._raise_if_configured("transition_ticket")
        self._ticket(ticket_key).status = status

    async def add_labels(self, ticket_key: str, labels: list[str]) -> None:
        """Add labels while preserving the existing label order."""

        labels_to_add = list(labels)
        self.calls.append(("add_labels", ticket_key, labels_to_add))
        self._raise_if_configured("add_labels")

        ticket = self._ticket(ticket_key)
        for label in labels_to_add:
            if label not in ticket.labels:
                ticket.labels.append(label)

    async def remove_labels(self, ticket_key: str, labels: list[str]) -> None:
        """Remove labels if present."""

        labels_to_remove = list(labels)
        self.calls.append(("remove_labels", ticket_key, labels_to_remove))
        self._raise_if_configured("remove_labels")

        ticket = self._ticket(ticket_key)
        ticket.labels = [
            label for label in ticket.labels if label not in labels_to_remove
        ]

    async def update_fields(self, ticket_key: str, fields: dict[str, object]) -> None:
        """Merge field updates into a ticket's field mapping."""

        fields_to_update = dict(fields)
        self.calls.append(("update_fields", ticket_key, fields_to_update))
        self._raise_if_configured("update_fields")
        self._ticket(ticket_key).fields.update(fields_to_update)

    async def add_comment(self, ticket_key: str, body: str) -> None:
        """Append a comment to a ticket."""

        self.calls.append(("add_comment", ticket_key, body))
        self._raise_if_configured("add_comment")
        self.comments.setdefault(ticket_key, []).append(body)

    def ticket(self, ticket_key: str) -> JiraTicket:
        """Return the mutable ticket stored under ``ticket_key``."""

        return self._ticket(ticket_key)

    def comments_for(self, ticket_key: str) -> list[str]:
        """Return comments written to one ticket."""

        return self.comments.setdefault(ticket_key, [])

    def configure_failure(
        self,
        operation: str,
        failure: JiraOperationFailure,
    ) -> None:
        """Configure failures for a Jira operation."""

        self.fail_on[operation] = _coerce_failure(failure)

    def _ticket(self, ticket_key: str) -> JiraTicket:
        try:
            return self.tickets[ticket_key]
        except KeyError as exc:
            raise KeyError(f"fake Jira ticket not found: {ticket_key}") from exc

    def _raise_if_configured(self, operation: str) -> None:
        configured = self.fail_on.get(operation)
        if isinstance(configured, list):
            if not configured:
                return
            exc = configured.pop(0)
            if exc is not None:
                raise exc
            return
        if configured is not None:
            raise configured


def _coerce_tickets(
    tickets: JiraTicket | Iterable[JiraTicket] | Mapping[str, JiraTicket],
) -> dict[str, JiraTicket]:
    if isinstance(tickets, JiraTicket):
        return {tickets.key: tickets}
    if isinstance(tickets, Mapping):
        return dict(tickets)
    return {ticket.key: ticket for ticket in tickets}


def _coerce_failures(
    fail_on: Mapping[str, JiraOperationFailure] | None,
) -> dict[str, _ConfiguredFailure]:
    if fail_on is None:
        return {}
    return {
        operation: _coerce_failure(failure)
        for operation, failure in fail_on.items()
    }


def _coerce_failure(failure: JiraOperationFailure) -> _ConfiguredFailure:
    if isinstance(failure, BaseException):
        return failure
    return list(failure)


def _matches_basic_jql(ticket: JiraTicket, jql: str) -> bool:
    normalized = " ".join(jql.split())
    if 'status = "To Do"' in normalized and ticket.status != "To Do":
        return False

    required_labels = _quoted_terms_after(normalized, "labels = ")
    if any(label not in ticket.labels for label in required_labels):
        return False

    excluded_labels = _quoted_terms_after(normalized, "labels != ")
    return not any(label in ticket.labels for label in excluded_labels)


def _quoted_terms_after(text: str, marker: str) -> list[str]:
    values: list[str] = []
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return values
        quote_start = text.find('"', index + len(marker))
        if quote_start < 0:
            return values
        quote_end = text.find('"', quote_start + 1)
        if quote_end < 0:
            return values
        values.append(text[quote_start + 1 : quote_end])
        start = quote_end + 1


__all__ = [
    "FakeJiraClient",
    "JiraClientCall",
    "JiraOperationFailure",
]

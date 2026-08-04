"""Pre-mutation enforcement of the repository command environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from ticket_agent.adapters.local.sandbox import (
    ENFORCING_SANDBOX_PROFILE,
    Sandbox,
    SandboxUnavailableError,
    build_sandbox,
    is_enforcing_sandbox,
)

if TYPE_CHECKING:
    from ticket_agent.goal.types import AutonomyDecision


class ExecutionPreflight[CheckedT_co](Protocol):
    """Guard shared by every production execution entry point.

    Generic in what a passing check yields, because the two implementations
    yield different evidence and their consumers need different amounts of it.
    `ExecutionEnvironmentPreflight` returns the `Sandbox` alone; the
    authorization preflight returns that plus the autonomy decision it just
    persisted. Declaring only the narrower one forced consumers of the wider
    one to reach for the extra field through `getattr`, which is a type-level
    hole in the guard the roadmap treats as a safety boundary.
    """

    def check(self, subject: object | None = None) -> CheckedT_co:
        """Return the execution evidence, or refuse before any mutation."""


class ExecutionEnvironmentPreflight:
    """Require the actual runtime sandbox, independent of authorization."""

    def __init__(
        self,
        sandbox_factory: Callable[[], Sandbox] | None = None,
    ) -> None:
        self._sandbox_factory = sandbox_factory or (
            lambda: build_sandbox(required=True)
        )

    def check(self, subject: object | None = None) -> Sandbox:
        del subject
        sandbox = self._sandbox_factory()
        # A configured policy string is not evidence. Consult the wrapper
        # object that will actually build the launch argv.
        if not is_enforcing_sandbox(sandbox):
            raise SandboxUnavailableError(
                "production repository commands require an enforcing "
                f"{ENFORCING_SANDBOX_PROFILE} sandbox; actual wrapper profile "
                f"is {sandbox.profile!r}"
            )
        return sandbox


class AuthorizedExecution(Protocol):
    """What an authorizing preflight yields: sandbox plus what it decided.

    Structural rather than an import of `AuthorizedExecutionContext`, so the
    orchestrator, feedback, and jira packages can state what they require of a
    preflight without depending on the goal package that implements it.
    """

    @property
    def sandbox(self) -> Sandbox: ...

    @property
    def autonomy_decision(self) -> AutonomyDecision: ...


#: A preflight that also settles authority, not only the environment. Every
#: production entry point takes this; only the shell factory, which needs the
#: sandbox and nothing else, takes `ExecutionPreflight[Sandbox]`.
type AuthorizingPreflight = ExecutionPreflight[AuthorizedExecution]


__all__ = [
    "AuthorizedExecution",
    "AuthorizingPreflight",
    "ExecutionEnvironmentPreflight",
    "ExecutionPreflight",
]

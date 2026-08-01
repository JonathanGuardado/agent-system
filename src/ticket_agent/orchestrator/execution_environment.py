"""Pre-mutation enforcement of the repository command environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ticket_agent.adapters.local.sandbox import (
    Sandbox,
    SandboxUnavailableError,
    build_sandbox,
)


class ExecutionPreflight(Protocol):
    """Guard shared by every production execution entry point."""

    def check(self) -> Sandbox:
        """Return the enforcing sandbox or refuse before execution mutation."""


class ExecutionEnvironmentPreflight:
    """Require the actual runtime sandbox, independent of authorization."""

    def __init__(
        self,
        sandbox_factory: Callable[[], Sandbox] | None = None,
    ) -> None:
        self._sandbox_factory = sandbox_factory or (
            lambda: build_sandbox(required=True)
        )

    def check(self) -> Sandbox:
        sandbox = self._sandbox_factory()
        # A configured policy string is not evidence. Consult the wrapper
        # object that will actually build the launch argv.
        if sandbox.profile != "bwrap":
            raise SandboxUnavailableError(
                "production repository commands require an enforcing bwrap "
                f"sandbox; actual wrapper profile is {sandbox.profile!r}"
            )
        return sandbox


__all__ = ["ExecutionEnvironmentPreflight", "ExecutionPreflight"]

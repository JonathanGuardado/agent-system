"""Durable execution authority composed with the sandbox boundary."""

from __future__ import annotations

from dataclasses import dataclass

from ticket_agent.adapters.local.sandbox import Sandbox, is_enforcing_sandbox
from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.autonomy import GoalAutonomyResolver
from ticket_agent.goal.contract import SQLiteGoalContractStore
from ticket_agent.goal.identity import (
    GoalIdentityError,
    goal_id_from_labels,
    normalize_goal_id,
)
from ticket_agent.goal.signing import NullSigner, Signer
from ticket_agent.goal.types import AutonomyDecision, AutonomyMode
from ticket_agent.orchestrator.execution_environment import (
    ExecutionEnvironmentPreflight,
)


class ExecutionAuthorizationError(AgentSystemError):
    """Raised when durable goal authority does not permit execution."""


@dataclass(frozen=True, slots=True)
class AuthorizedExecutionContext:
    """Verified sandbox plus the freshly persisted autonomy decision."""

    sandbox: Sandbox
    autonomy_decision: AutonomyDecision

    @property
    def profile(self) -> str:
        return self.sandbox.profile

    # There was a `wrap` here that forwarded to `self.sandbox.wrap`. It was
    # never called from src/ or tests/, and it passed a CommandExecutionPolicy
    # where Sandbox.wrap requires a SandboxPolicy -- the conversion
    # SandboxPolicy.from_execution_policy exists and this one skipped it. An
    # uncalled, untested shortcut through the sandbox boundary is worse than no
    # shortcut; callers use `.sandbox` and go through the real one.


class ExecutionAuthorizationPreflight:
    """Compose sandbox readiness with current durable execution authority."""

    def __init__(
        self,
        environment: ExecutionEnvironmentPreflight,
        authorization_store: SQLiteGoalContractStore,
        signer: Signer | NullSigner,
        autonomy_resolver: GoalAutonomyResolver,
    ) -> None:
        self._environment = environment
        self._authorization_store = authorization_store
        self._signer = signer
        self._autonomy_resolver = autonomy_resolver

    def check(self, subject: object | None = None) -> AuthorizedExecutionContext:
        sandbox = self._environment.check(subject)
        if subject is None:
            raise ExecutionAuthorizationError(
                "execution subject is required for authorization preflight"
            )

        raw_goal_id = getattr(subject, "goal_id", None)
        try:
            goal_id = normalize_goal_id(raw_goal_id)
            labels = getattr(subject, "labels", None)
            if labels is not None:
                label_goal_id = goal_id_from_labels(labels)
                if label_goal_id != goal_id:
                    raise GoalIdentityError(
                        "ticket goal label disagrees with loaded goal id"
                    )
        except GoalIdentityError as exc:
            raise ExecutionAuthorizationError(
                f"invalid or missing goal identity: {exc}"
            ) from exc

        effective = self._authorization_store.effective_authorization(
            goal_id,
            self._signer,
        )
        if not effective.authorized or effective.record is None:
            reason = "; ".join(effective.reasons) or "authorization denied"
            raise ExecutionAuthorizationError(
                f"goal {goal_id} is not executable: {reason}"
            )

        repository = getattr(subject, "repository", None)
        allowed_repositories = effective.record.contract.permitted_scope.repositories
        if repository and repository not in allowed_repositories:
            raise ExecutionAuthorizationError(
                f"repository {repository!r} is outside goal {goal_id} scope"
            )
        # Observe the wrapper object that will actually build the launch argv,
        # for the same reason ``ExecutionEnvironmentPreflight`` does: a
        # configured policy string is not evidence. ``check`` above refuses a
        # non-enforcing sandbox today, so this is currently always true -- but
        # asserting a constant makes the guarantee depend on a raise forty
        # lines away, and any injected preflight satisfying the protocol
        # without that raise would silently record availability it never had.
        autonomy = self._autonomy_resolver.decide(
            effective.record.contract,
            sandbox_available=is_enforcing_sandbox(sandbox),
        )
        if autonomy.effective_mode < AutonomyMode.IMPLEMENT:
            raise ExecutionAuthorizationError(
                f"goal {goal_id} effective autonomy {autonomy.effective_mode} "
                "does not permit implementation"
            )
        return AuthorizedExecutionContext(sandbox, autonomy)


__all__ = [
    "AuthorizedExecutionContext",
    "ExecutionAuthorizationError",
    "ExecutionAuthorizationPreflight",
]

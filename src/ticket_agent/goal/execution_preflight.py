"""Durable execution authority composed with the sandbox boundary."""

from __future__ import annotations

from ticket_agent.adapters.local.sandbox import Sandbox
from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.contract import SQLiteGoalContractStore
from ticket_agent.goal.identity import (
    GoalIdentityError,
    goal_id_from_labels,
    normalize_goal_id,
)
from ticket_agent.goal.signing import NullSigner, Signer
from ticket_agent.goal.types import AutonomyMode
from ticket_agent.orchestrator.execution_environment import (
    ExecutionEnvironmentPreflight,
)


class ExecutionAuthorizationError(AgentSystemError):
    """Raised when durable goal authority does not permit execution."""


class ExecutionAuthorizationPreflight:
    """Compose sandbox readiness with current durable execution authority."""

    def __init__(
        self,
        environment: ExecutionEnvironmentPreflight,
        authorization_store: SQLiteGoalContractStore,
        signer: Signer | NullSigner,
    ) -> None:
        self._environment = environment
        self._authorization_store = authorization_store
        self._signer = signer

    def check(self, subject: object | None = None) -> Sandbox:
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
        if effective.record.contract.autonomy_ceiling < AutonomyMode.IMPLEMENT:
            raise ExecutionAuthorizationError(
                f"goal {goal_id} autonomy ceiling does not permit implementation"
            )
        return sandbox


__all__ = [
    "ExecutionAuthorizationError",
    "ExecutionAuthorizationPreflight",
]

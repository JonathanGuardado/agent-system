"""Compiling, authorizing, signing, and storing goal contracts.

One approval, at intake. A request from an allowlisted Slack user in an
allowlisted channel authorizes ordinary in-policy work; there is no second
per-proposal approval. That is the shift from steering by tickets to steering
by goals: the human sets the objective, the agent plans freely underneath it.

Authorization is a conjunction of three independent judgements, and any one of
them can send the request to a human:

  allowlist        who asked, and where
  risk policy      deterministic rules over what was asked for
  semantic check   a different model comparing the compilation to the verbatim
                   request

They are independent on purpose. The allowlist knows nothing about content;
the rules know nothing about meaning; the checker knows nothing about policy.
Each covers what the others cannot see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from typing import Any

from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.policy import Decision, RiskPolicy, stricter
from ticket_agent.goal.semantic_check import (
    NullSemanticChecker,
    SemanticChecker,
    SemanticVerdict,
)
from ticket_agent.goal.signing import NullSigner, Signature, Signer, SigningError
from ticket_agent.goal.types import (
    AcceptanceCriterion,
    AuthorizationContext,
    Budgets,
    GoalContract,
    RiskClass,
    ScopeSpec,
    canonical_json,
)
from ticket_agent.sqlite_support import connect, write_transaction

_DEFAULT_BUSY_TIMEOUT_MS = 5_000


class GoalContractError(AgentSystemError):
    """Raised when a contract cannot be compiled, amended, or stored."""


@dataclass(frozen=True, slots=True)
class Allowlist:
    """Who may authorize work, and from where.

    Empty means empty: a system with no configured allowlist authorizes
    nobody. Treating "unconfigured" as "everyone" is how an internal tool
    becomes an open one.
    """

    users: frozenset[str] = frozenset()
    channels: frozenset[str] = frozenset()

    def permits(self, *, user_id: str, channel: str | None) -> tuple[bool, str]:
        if not self.users:
            return False, "no Slack users are allowlisted for goal authorization"
        if user_id not in self.users:
            return False, f"user {user_id} is not allowlisted"
        if self.channels and (channel or "") not in self.channels:
            return False, f"channel {channel or '(none)'} is not allowlisted"
        return True, ""


@dataclass(frozen=True, slots=True)
class AuthorizationOutcome:
    """A compiled contract plus why it may or may not proceed unattended."""

    contract: GoalContract
    signature: Signature | None
    risk: Decision
    semantic: SemanticVerdict
    allowlisted: bool
    allowlist_reason: str = ""

    @property
    def authorized(self) -> bool:
        """Whether work may start without a further human decision."""

        return (
            self.allowlisted
            and self.signature is not None
            and self.semantic.agrees
            and self.contract.risk_class in ("low", "standard")
        )

    def escalation_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.allowlisted:
            reasons.append(self.allowlist_reason or "not allowlisted")
        if self.signature is None:
            reasons.append("contract could not be signed")
        if self.contract.risk_class not in ("low", "standard"):
            reasons.append(f"risk class {self.contract.risk_class}")
            reasons.extend(self.risk.reasons)
        reasons.extend(self.semantic.disagreements())
        return tuple(reasons)


class GoalContractCompiler:
    """Turns an authorized Slack request into a signed, immutable contract."""

    def __init__(
        self,
        *,
        policy: RiskPolicy,
        allowlist: Allowlist,
        signer: Signer | NullSigner | None = None,
        semantic_checker: SemanticChecker | None = None,
        harness_digest: str = "",
        trust_root_digest: str = "",
        clock: Any = None,
    ) -> None:
        self._policy = policy
        self._allowlist = allowlist
        self._signer = signer or NullSigner()
        self._checker = semantic_checker or NullSemanticChecker()
        self._harness_digest = harness_digest
        self._trust_root_digest = trust_root_digest
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def compile(
        self,
        *,
        goal_id: str,
        original_request: str,
        objective: str,
        acceptance_criteria: Sequence[str | AcceptanceCriterion],
        user_id: str,
        channel: str | None,
        thread_ts: str,
        repositories: Sequence[str],
        non_goals: Sequence[str] = (),
        allowed_paths: Sequence[str] = (),
        denied_paths: Sequence[str] = (),
        budgets: Budgets | None = None,
        compiler_provider: str = "",
    ) -> AuthorizationOutcome:
        """Compile and judge a request. Never raises on a denial."""

        allowlisted, allowlist_reason = self._allowlist.permits(
            user_id=user_id, channel=channel
        )

        scope = ScopeSpec(
            repositories=tuple(repositories),
            allowed_paths=tuple(allowed_paths),
            denied_paths=tuple(denied_paths),
        )
        risk = self._policy.classify_request(
            repositories=repositories, scope=scope, budgets=budgets
        )

        risk_class: RiskClass = risk.risk
        if not allowlisted:
            risk_class = stricter(risk_class, "human_only")

        contract = GoalContract(
            goal_id=goal_id,
            version=1,
            schema_version=1,
            authorization=AuthorizationContext(
                requester=user_id,
                slack_channel=channel or "",
                slack_message_ts=thread_ts,
                allowlisted=allowlisted,
                authorized_at=self._clock(),
            ),
            original_request=original_request,
            objective=objective,
            acceptance_criteria=_normalize_criteria(acceptance_criteria),
            non_goals=tuple(non_goals),
            permitted_scope=scope,
            risk_class=risk_class,
            budgets=budgets or Budgets(),
            harness_digest=self._harness_digest,
            trust_root_digest=self._trust_root_digest,
            policy_digest=self._policy.policy_digest,
        )

        # The checker must not resolve to the provider that did the compiling.
        exclude = (compiler_provider,) if compiler_provider else ()
        semantic = await self._checker.check(contract, exclude_providers=exclude)

        signature: Signature | None
        try:
            signature = self._signer.sign(_signed_view(contract))
        except SigningError:
            signature = None

        return AuthorizationOutcome(
            contract=contract,
            signature=signature,
            risk=risk,
            semantic=semantic,
            allowlisted=allowlisted,
            allowlist_reason=allowlist_reason,
        )

    def amend(
        self,
        contract: GoalContract,
        *,
        authorized_by: str,
        reason: str,
        objective: str | None = None,
        add_criteria: Sequence[str | AcceptanceCriterion] = (),
        add_non_goals: Sequence[str] = (),
        scope: ScopeSpec | None = None,
        budgets: Budgets | None = None,
    ) -> tuple[GoalContract, Signature | None]:
        """Produce a new signed version. Never mutates in place.

        Amendment is a human act by construction: `authorized_by` and `reason`
        are required, and the result is a new version rather than an edit, so
        the record of what was originally agreed survives.
        """

        if not authorized_by.strip():
            raise GoalContractError("an amendment requires an authorizing human")
        if not reason.strip():
            raise GoalContractError("an amendment requires a reason")

        amended = replace(
            contract,
            version=contract.version + 1,
            objective=objective or contract.objective,
            acceptance_criteria=contract.acceptance_criteria
            + _normalize_criteria(add_criteria, start=len(contract.acceptance_criteria)),
            non_goals=contract.non_goals + tuple(add_non_goals),
            permitted_scope=scope or contract.permitted_scope,
            budgets=budgets or contract.budgets,
            authorization=replace(
                contract.authorization,
                requester=authorized_by,
                authorized_at=self._clock(),
            ),
        )
        try:
            signature = self._signer.sign(_signed_view(amended))
        except SigningError:
            signature = None
        return amended, signature

    def verify(self, contract: GoalContract, signature: Signature | str) -> bool:
        return self._signer.verify(_signed_view(contract), signature)


def _signed_view(contract: GoalContract) -> Mapping[str, Any]:
    """What the signature covers.

    Everything that defines authority. Kept explicit rather than signing the
    dataclass wholesale so that adding an incidental field later cannot
    silently invalidate every stored signature.
    """

    return {
        "goal_id": contract.goal_id,
        "version": contract.version,
        "schema_version": contract.schema_version,
        "original_request": contract.original_request,
        "objective": contract.objective,
        "acceptance_criteria": [
            {"id": c.criterion_id, "text": c.text} for c in contract.acceptance_criteria
        ],
        "non_goals": list(contract.non_goals),
        "permitted_scope": {
            "repositories": list(contract.permitted_scope.repositories),
            "allowed_paths": list(contract.permitted_scope.allowed_paths),
            "denied_paths": list(contract.permitted_scope.denied_paths),
        },
        "risk_class": contract.risk_class,
        "requester": contract.authorization.requester,
        "authorized_at": contract.authorization.authorized_at,
        "harness_digest": contract.harness_digest,
        "trust_root_digest": contract.trust_root_digest,
        "policy_digest": contract.policy_digest,
    }


def _normalize_criteria(
    criteria: Sequence[str | AcceptanceCriterion], *, start: int = 0
) -> tuple[AcceptanceCriterion, ...]:
    normalized: list[AcceptanceCriterion] = []
    for offset, item in enumerate(criteria):
        if isinstance(item, AcceptanceCriterion):
            normalized.append(item)
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(
            AcceptanceCriterion(criterion_id=f"c{start + offset + 1}", text=text)
        )
    return tuple(normalized)


class SQLiteGoalContractStore:
    """Versioned contract storage, keyed on ``(goal_id, version)``."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS goal_contracts (
        goal_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        contract_digest TEXT NOT NULL,
        signature TEXT,
        payload TEXT NOT NULL,
        risk_class TEXT NOT NULL,
        authorized_at TEXT NOT NULL,
        PRIMARY KEY (goal_id, version)
    );
    """

    def __init__(
        self, db_path: str | Path, *, busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS
    ) -> None:
        self._lock = RLock()
        self._connection = connect(db_path, busy_timeout_ms)
        with self._lock:
            self._connection.executescript(self._SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save(self, contract: GoalContract, signature: Signature | None) -> None:
        """Store a version. Re-saving the same version is an error, not an
        overwrite -- an immutable record that can be replaced is not one."""

        with self._lock, write_transaction(self._connection):
            existing = self._connection.execute(
                "SELECT contract_digest FROM goal_contracts "
                "WHERE goal_id = ? AND version = ?",
                (contract.goal_id, contract.version),
            ).fetchone()
            if existing is not None:
                if existing["contract_digest"] == contract.contract_digest:
                    return
                raise GoalContractError(
                    f"refusing to overwrite {contract.goal_id} v{contract.version} "
                    "with different content"
                )
            self._connection.execute(
                "INSERT INTO goal_contracts (goal_id, version, contract_digest, "
                "signature, payload, risk_class, authorized_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    contract.goal_id,
                    contract.version,
                    contract.contract_digest,
                    str(signature) if signature is not None else None,
                    canonical_json(_signed_view(contract)),
                    contract.risk_class,
                    contract.authorization.authorized_at.isoformat(),
                ),
            )

    def latest_version(self, goal_id: str) -> int | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT MAX(version) AS v FROM goal_contracts WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
        return None if row is None or row["v"] is None else int(row["v"])

    def stored_signature(self, goal_id: str, version: int) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT signature FROM goal_contracts WHERE goal_id = ? AND version = ?",
                (goal_id, version),
            ).fetchone()
        return None if row is None else row["signature"]

    def stored_payload(self, goal_id: str, version: int) -> Mapping[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM goal_contracts WHERE goal_id = ? AND version = ?",
                (goal_id, version),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def verify_stored(
        self, goal_id: str, version: int, signer: Signer | NullSigner
    ) -> bool:
        """Whether the stored row still matches its signature."""

        payload = self.stored_payload(goal_id, version)
        signature = self.stored_signature(goal_id, version)
        if payload is None or signature is None:
            return False
        return signer.verify(payload, signature)


__all__ = [
    "Allowlist",
    "AuthorizationOutcome",
    "GoalContractCompiler",
    "GoalContractError",
    "SQLiteGoalContractStore",
]

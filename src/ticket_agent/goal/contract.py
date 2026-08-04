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
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import RLock
from typing import Any

from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.identity import GoalIdentityError, normalize_goal_id
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
    digest,
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
    evidence_signature: Signature | None = None

    @property
    def authorized(self) -> bool:
        """Whether work may start without a further human decision."""

        return (
            self.allowlisted
            and self.signature is not None
            and self.evidence_signature is not None
            and self.semantic.agrees
            and self.contract.risk_class in ("low", "standard")
        )

    def escalation_reasons(self) -> tuple[str, ...]:
        reasons = list(_authorization_denial_reasons(self))
        if self.signature is not None and self.evidence_signature is None:
            reasons.append("authorization evidence could not be signed")
        return tuple(reasons)


def _authorization_denial_reasons(
    outcome: AuthorizationOutcome,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not outcome.allowlisted:
        reasons.append(outcome.allowlist_reason or "not allowlisted")
    if outcome.signature is None:
        reasons.append("contract could not be signed")
    if outcome.contract.risk_class not in ("low", "standard"):
        reasons.append(f"risk class {outcome.contract.risk_class}")
        reasons.extend(outcome.risk.reasons)
    reasons.extend(outcome.semantic.disagreements())
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
        self._clock = clock or (lambda: datetime.now(UTC))

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

        goal_id = normalize_goal_id(goal_id)

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

        unsigned_outcome = AuthorizationOutcome(
            contract=contract,
            signature=signature,
            risk=risk,
            semantic=semantic,
            allowlisted=allowlisted,
            allowlist_reason=allowlist_reason,
        )
        try:
            evidence_signature = self._signer.sign(
                _authorization_evidence(unsigned_outcome)
            )
        except SigningError:
            evidence_signature = None
        return replace(
            unsigned_outcome,
            evidence_signature=evidence_signature,
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
        "slack_channel": contract.authorization.slack_channel,
        "slack_message_ts": contract.authorization.slack_message_ts,
        "allowlisted": contract.authorization.allowlisted,
        "authorized_at": contract.authorization.authorized_at,
        "harness_digest": contract.harness_digest,
        "trust_root_digest": contract.trust_root_digest,
        "policy_digest": contract.policy_digest,
    }


def _authorization_evidence(outcome: AuthorizationOutcome) -> Mapping[str, Any]:
    """Decision evidence signed independently from the immutable contract."""

    return {
        "goal_id": outcome.contract.goal_id,
        "contract_version": outcome.contract.version,
        "contract_digest": outcome.contract.contract_digest,
        "contract_signature": (
            str(outcome.signature) if outcome.signature is not None else None
        ),
        "decision": "authorized" if _base_authorized(outcome) else "denied",
        "semantic_verdict": asdict(outcome.semantic),
        "denial_reasons": list(_authorization_denial_reasons(outcome)),
    }


def _base_authorized(outcome: AuthorizationOutcome) -> bool:
    """Authorization clauses excluding the evidence signature itself."""

    return (
        outcome.allowlisted
        and outcome.contract.authorization.allowlisted
        and outcome.signature is not None
        and outcome.semantic.agrees
        and outcome.contract.risk_class in ("low", "standard")
    )


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


@dataclass(frozen=True, slots=True)
class StoredAuthorization:
    """One immutable durable authorization outcome."""

    contract: GoalContract
    stored_contract_digest: str
    signed_payload: Mapping[str, Any]
    contract_signature: str | None
    decision: str
    semantic: SemanticVerdict
    denial_reasons: tuple[str, ...]
    evidence_digest: str | None
    evidence_signature: str | None
    evidence_payload: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class EffectiveAuthorization:
    """Verified current authority, including append-only revocation state."""

    authorized: bool
    reasons: tuple[str, ...]
    record: StoredAuthorization | None = None
    revoked_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RevocationRecord:
    revocation_id: int
    goal_id: str
    contract_version: int
    revoked_at: datetime
    revoked_by: str
    reason: str
    evidence_digest: str
    provenance: str
    signature: str


class SQLiteGoalContractStore:
    """Versioned contract storage, keyed on ``(goal_id, version)``."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS goal_contracts (
        goal_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        contract_digest TEXT NOT NULL,
        signature TEXT,
        payload TEXT NOT NULL,
        contract_payload TEXT,
        risk_class TEXT NOT NULL,
        authorized_at TEXT NOT NULL,
        decision TEXT NOT NULL DEFAULT 'denied',
        semantic_payload TEXT NOT NULL DEFAULT '{}',
        denial_reasons TEXT NOT NULL DEFAULT '[]',
        evidence_digest TEXT,
        evidence_signature TEXT,
        evidence_payload TEXT,
        PRIMARY KEY (goal_id, version)
    );

    CREATE TABLE IF NOT EXISTS goal_revocations (
        revocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        revoked_at TEXT NOT NULL,
        revoked_by TEXT NOT NULL,
        reason TEXT NOT NULL,
        evidence_digest TEXT NOT NULL,
        provenance TEXT NOT NULL,
        signature TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_goal_revocations_contract
        ON goal_revocations (goal_id, contract_version, revocation_id);
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        clock: Any = None,
    ) -> None:
        self._lock = RLock()
        self._connection = connect(db_path, busy_timeout_ms)
        self._clock = clock or (lambda: datetime.now(UTC))
        with self._lock:
            self._connection.executescript(self._SCHEMA)
            self._migrate_authorization_columns()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save(self, contract: GoalContract, signature: Signature | None) -> None:
        """Store a version. Re-saving the same version is an error, not an
        overwrite -- an immutable record that can be replaced is not one."""

        goal_id = normalize_goal_id(contract.goal_id)
        with self._lock, write_transaction(self._connection):
            existing = self._connection.execute(
                "SELECT contract_digest FROM goal_contracts "
                "WHERE goal_id = ? AND version = ?",
                (goal_id, contract.version),
            ).fetchone()
            if existing is not None:
                if existing["contract_digest"] == contract.contract_digest:
                    return
                raise GoalContractError(
                    f"refusing to overwrite {contract.goal_id} v{contract.version} "
                    "with different content"
                )
            self._insert_authorization(
                contract=contract,
                signature=signature,
                decision="denied",
                semantic=SemanticVerdict(
                    objective_matches=False,
                    criteria_complete=False,
                    nothing_invented=False,
                    error="durable authorization outcome unavailable",
                ),
                denial_reasons=("durable authorization outcome unavailable",),
                evidence_payload=None,
                evidence_signature=None,
            )

    def save_outcome(self, outcome: AuthorizationOutcome) -> None:
        """Persist an affirmative or denied result without erasing either."""

        contract = outcome.contract
        goal_id = normalize_goal_id(contract.goal_id)
        evidence_payload = _authorization_evidence(outcome)
        evidence_digest = digest(evidence_payload)
        with self._lock, write_transaction(self._connection):
            existing = self._connection.execute(
                "SELECT contract_digest, evidence_digest FROM goal_contracts "
                "WHERE goal_id = ? AND version = ?",
                (goal_id, contract.version),
            ).fetchone()
            if existing is not None:
                if (
                    existing["contract_digest"] == contract.contract_digest
                    and existing["evidence_digest"] == evidence_digest
                ):
                    return
                raise GoalContractError(
                    f"refusing to overwrite {goal_id} v{contract.version} "
                    "with different authorization evidence"
                )
            self._insert_authorization(
                contract=contract,
                signature=outcome.signature,
                decision="authorized" if outcome.authorized else "denied",
                semantic=outcome.semantic,
                denial_reasons=outcome.escalation_reasons(),
                evidence_payload=evidence_payload,
                evidence_signature=outcome.evidence_signature,
            )

    def latest_version(self, goal_id: str) -> int | None:
        goal_id = normalize_goal_id(goal_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT MAX(version) AS v FROM goal_contracts WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
        return None if row is None or row["v"] is None else int(row["v"])

    def stored_signature(self, goal_id: str, version: int) -> str | None:
        goal_id = normalize_goal_id(goal_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT signature FROM goal_contracts WHERE goal_id = ? AND version = ?",
                (goal_id, version),
            ).fetchone()
        return None if row is None else row["signature"]

    def stored_payload(self, goal_id: str, version: int) -> Mapping[str, Any] | None:
        goal_id = normalize_goal_id(goal_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM goal_contracts WHERE goal_id = ? AND version = ?",
                (goal_id, version),
            ).fetchone()
        if row is None:
            return None
        payload: Mapping[str, Any] = json.loads(row["payload"])
        return payload

    def load_authorization(
        self,
        goal_id: str,
        version: int | None = None,
    ) -> StoredAuthorization | None:
        goal_id = normalize_goal_id(goal_id)
        with self._lock:
            if version is None:
                row = self._connection.execute(
                    "SELECT * FROM goal_contracts WHERE goal_id = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (goal_id,),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT * FROM goal_contracts WHERE goal_id = ? AND version = ?",
                    (goal_id, version),
                ).fetchone()
        if row is None:
            return None
        try:
            contract_payload = row["contract_payload"]
            if not contract_payload:
                raise GoalContractError("legacy row has no full contract payload")
            contract = _contract_from_payload(json.loads(contract_payload))
            signed_payload = json.loads(row["payload"])
            semantic = _semantic_from_payload(json.loads(row["semantic_payload"]))
            evidence_payload = (
                json.loads(row["evidence_payload"])
                if row["evidence_payload"]
                else None
            )
            denial_reasons = tuple(json.loads(row["denial_reasons"]))
        except (
            GoalIdentityError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise GoalContractError(
                f"stored authorization payload is invalid for {goal_id}: {exc}"
            ) from exc
        return StoredAuthorization(
            contract=contract,
            stored_contract_digest=row["contract_digest"],
            signed_payload=signed_payload,
            contract_signature=row["signature"],
            decision=row["decision"],
            semantic=semantic,
            denial_reasons=denial_reasons,
            evidence_digest=row["evidence_digest"],
            evidence_signature=row["evidence_signature"],
            evidence_payload=evidence_payload,
        )

    def effective_authorization(
        self,
        goal_id: str,
        signer: Signer | NullSigner,
        *,
        version: int | None = None,
    ) -> EffectiveAuthorization:
        """Calculate current authority from verified evidence and revocations."""

        canonical_goal_id = normalize_goal_id(goal_id)
        try:
            record = self.load_authorization(canonical_goal_id, version)
        except GoalContractError as exc:
            return EffectiveAuthorization(False, (str(exc),))
        if record is None:
            return EffectiveAuthorization(False, ("authorization record is missing",))

        reasons = _verify_authorization_record(record, signer)
        if reasons:
            return EffectiveAuthorization(False, reasons, record)

        revocations = self.revocations_for(
            canonical_goal_id,
            record.contract.version,
        )
        latest_revocation: RevocationRecord | None = None
        for revocation in revocations:
            payload = _revocation_payload(
                goal_id=revocation.goal_id,
                contract_version=revocation.contract_version,
                revoked_at=revocation.revoked_at,
                revoked_by=revocation.revoked_by,
                reason=revocation.reason,
                evidence_digest=revocation.evidence_digest,
                provenance=revocation.provenance,
            )
            if (
                revocation.provenance != "configured_signing_authority"
                or not signer.verify(payload, revocation.signature)
            ):
                return EffectiveAuthorization(
                    False,
                    ("revocation provenance or signature is unverifiable",),
                    record,
                )
            if revocation.evidence_digest != record.evidence_digest:
                return EffectiveAuthorization(
                    False,
                    ("revocation evidence digest does not match authorization",),
                    record,
                )
            if revocation.revoked_at < record.contract.authorization.authorized_at:
                return EffectiveAuthorization(
                    False,
                    ("revocation predates the authorization it claims to revoke",),
                    record,
                )
            latest_revocation = revocation
        if latest_revocation is not None:
            return EffectiveAuthorization(
                False,
                (f"authorization revoked: {latest_revocation.reason}",),
                record,
                revoked_at=latest_revocation.revoked_at,
            )
        return EffectiveAuthorization(True, (), record)

    def append_revocation(
        self,
        goal_id: str,
        version: int,
        *,
        revoked_by: str,
        reason: str,
        signer: Signer | NullSigner,
        provenance: str = "configured_signing_authority",
    ) -> RevocationRecord:
        """Append a signed operator revocation; never mutate authorization."""

        goal_id = normalize_goal_id(goal_id)
        if not revoked_by.strip() or not reason.strip():
            raise GoalContractError("revocation requires revoked_by and reason")
        if provenance != "configured_signing_authority":
            raise GoalContractError("untrusted revocation authority provenance")
        record = self.load_authorization(goal_id, version)
        if record is None or not record.evidence_digest:
            raise GoalContractError("cannot revoke missing authorization evidence")
        verification_reasons = _verify_authorization_record(record, signer)
        if verification_reasons:
            raise GoalContractError(
                "cannot revoke unverifiable authorization: "
                + "; ".join(verification_reasons)
            )
        revoked_at = self._clock()
        payload = _revocation_payload(
            goal_id=goal_id,
            contract_version=version,
            revoked_at=revoked_at,
            revoked_by=revoked_by,
            reason=reason,
            evidence_digest=record.evidence_digest,
            provenance=provenance,
        )
        try:
            signature = signer.sign(payload)
        except SigningError as exc:
            raise GoalContractError(
                "configured signing authority is required to revoke"
            ) from exc
        with self._lock, write_transaction(self._connection):
            cursor = self._connection.execute(
                "INSERT INTO goal_revocations (goal_id, contract_version, "
                "revoked_at, revoked_by, reason, evidence_digest, provenance, "
                "signature) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    goal_id,
                    version,
                    revoked_at.isoformat(),
                    revoked_by,
                    reason,
                    record.evidence_digest,
                    provenance,
                    str(signature),
                ),
            )
            if cursor.lastrowid is None:
                raise GoalContractError("revocation insert returned no row id")
            revocation_id = int(cursor.lastrowid)
        return RevocationRecord(
            revocation_id=revocation_id,
            goal_id=goal_id,
            contract_version=version,
            revoked_at=revoked_at,
            revoked_by=revoked_by,
            reason=reason,
            evidence_digest=record.evidence_digest,
            provenance=provenance,
            signature=str(signature),
        )

    def revocations_for(
        self,
        goal_id: str,
        version: int,
    ) -> tuple[RevocationRecord, ...]:
        goal_id = normalize_goal_id(goal_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM goal_revocations WHERE goal_id = ? "
                "AND contract_version = ? ORDER BY revocation_id",
                (goal_id, version),
            ).fetchall()
        return tuple(_revocation_from_row(row) for row in rows)

    def verify_stored(
        self, goal_id: str, version: int, signer: Signer | NullSigner
    ) -> bool:
        """Whether the stored row still matches its signature."""

        payload = self.stored_payload(goal_id, version)
        signature = self.stored_signature(goal_id, version)
        if payload is None or signature is None:
            return False
        return signer.verify(payload, signature)

    def _insert_authorization(
        self,
        *,
        contract: GoalContract,
        signature: Signature | None,
        decision: str,
        semantic: SemanticVerdict,
        denial_reasons: Sequence[str],
        evidence_payload: Mapping[str, Any] | None,
        evidence_signature: Signature | None,
    ) -> None:
        self._connection.execute(
            "INSERT INTO goal_contracts (goal_id, version, contract_digest, "
            "signature, payload, contract_payload, risk_class, authorized_at, "
            "decision, semantic_payload, denial_reasons, evidence_digest, "
            "evidence_signature, evidence_payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                contract.goal_id,
                contract.version,
                contract.contract_digest,
                str(signature) if signature is not None else None,
                canonical_json(_signed_view(contract)),
                canonical_json(contract),
                contract.risk_class,
                contract.authorization.authorized_at.isoformat(),
                decision,
                canonical_json(semantic),
                canonical_json(tuple(denial_reasons)),
                digest(evidence_payload) if evidence_payload is not None else None,
                (
                    str(evidence_signature)
                    if evidence_signature is not None
                    else None
                ),
                (
                    canonical_json(evidence_payload)
                    if evidence_payload is not None
                    else None
                ),
            ),
        )

    def _migrate_authorization_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(goal_contracts)"
            ).fetchall()
        }
        additions = {
            "contract_payload": "TEXT",
            "decision": "TEXT NOT NULL DEFAULT 'denied'",
            "semantic_payload": "TEXT NOT NULL DEFAULT '{}'",
            "denial_reasons": "TEXT NOT NULL DEFAULT '[]'",
            "evidence_digest": "TEXT",
            "evidence_signature": "TEXT",
            "evidence_payload": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE goal_contracts ADD COLUMN {name} {declaration}"
                )


def _verify_authorization_record(
    record: StoredAuthorization,
    signer: Signer | NullSigner,
) -> tuple[str, ...]:
    reasons: list[str] = []
    contract = record.contract
    if record.stored_contract_digest != contract.contract_digest:
        reasons.append("stored contract digest does not match contract payload")
    expected_signed_payload = _signed_view(contract)
    if canonical_json(record.signed_payload) != canonical_json(expected_signed_payload):
        reasons.append("stored signed payload does not match contract payload")
    if record.contract_signature is None or not signer.verify(
        expected_signed_payload,
        record.contract_signature,
    ):
        reasons.append("contract signature is missing or invalid")

    expected_decision = (
        contract.authorization.allowlisted
        and record.contract_signature is not None
        and record.semantic.agrees
        and contract.risk_class in ("low", "standard")
    )
    expected_decision_text = "authorized" if expected_decision else "denied"
    if record.decision != expected_decision_text:
        reasons.append("stored decision disagrees with signed authorization clauses")

    expected_evidence = {
        "goal_id": contract.goal_id,
        "contract_version": contract.version,
        "contract_digest": contract.contract_digest,
        "contract_signature": record.contract_signature,
        "decision": expected_decision_text,
        "semantic_verdict": asdict(record.semantic),
        "denial_reasons": list(record.denial_reasons),
    }
    if record.evidence_payload is None:
        reasons.append("authorization evidence payload is missing")
    elif canonical_json(record.evidence_payload) != canonical_json(expected_evidence):
        reasons.append("authorization evidence payload is inconsistent")
    if (
        record.evidence_digest is None
        or record.evidence_digest != digest(expected_evidence)
    ):
        reasons.append("authorization evidence digest is missing or mismatched")
    if record.evidence_signature is None or not signer.verify(
        expected_evidence,
        record.evidence_signature,
    ):
        reasons.append("authorization evidence signature is missing or invalid")
    if record.decision != "authorized":
        reasons.extend(record.denial_reasons or ("authorization was denied",))
    return tuple(dict.fromkeys(reasons))


def _contract_from_payload(payload: Mapping[str, Any]) -> GoalContract:
    authorization = payload["authorization"]
    scope = payload.get("permitted_scope", {})
    budgets = payload.get("budgets", {})
    return GoalContract(
        goal_id=normalize_goal_id(payload["goal_id"]),
        version=int(payload["version"]),
        schema_version=int(payload["schema_version"]),
        authorization=AuthorizationContext(
            requester=authorization["requester"],
            slack_channel=authorization["slack_channel"],
            slack_message_ts=authorization["slack_message_ts"],
            allowlisted=bool(authorization["allowlisted"]),
            authorized_at=datetime.fromisoformat(authorization["authorized_at"]),
        ),
        original_request=payload["original_request"],
        objective=payload["objective"],
        acceptance_criteria=tuple(
            AcceptanceCriterion(
                criterion_id=item["criterion_id"],
                text=item["text"],
                oracle=item.get("oracle"),
                demo_required=bool(item.get("demo_required", False)),
            )
            for item in payload["acceptance_criteria"]
        ),
        non_goals=tuple(payload.get("non_goals", ())),
        permitted_scope=ScopeSpec(
            repositories=tuple(scope.get("repositories", ())),
            allowed_paths=tuple(scope.get("allowed_paths", ())),
            denied_paths=tuple(scope.get("denied_paths", ())),
        ),
        risk_class=payload["risk_class"],
        budgets=Budgets(**budgets),
        human_required=tuple(payload.get("human_required", ())),
        harness_digest=payload.get("harness_digest", ""),
        trust_root_digest=payload.get("trust_root_digest", ""),
        policy_digest=payload.get("policy_digest", ""),
    )


def _semantic_from_payload(payload: Mapping[str, Any]) -> SemanticVerdict:
    return SemanticVerdict(
        objective_matches=bool(payload["objective_matches"]),
        criteria_complete=bool(payload["criteria_complete"]),
        nothing_invented=bool(payload["nothing_invented"]),
        missing=tuple(payload.get("missing", ())),
        invented=tuple(payload.get("invented", ())),
        notes=str(payload.get("notes", "")),
        error=payload.get("error"),
    )


def _revocation_payload(
    *,
    goal_id: str,
    contract_version: int,
    revoked_at: datetime,
    revoked_by: str,
    reason: str,
    evidence_digest: str,
    provenance: str,
) -> Mapping[str, Any]:
    return {
        "goal_id": goal_id,
        "contract_version": contract_version,
        "revoked_at": revoked_at,
        "revoked_by": revoked_by,
        "reason": reason,
        "evidence_digest": evidence_digest,
        "provenance": provenance,
    }


def _revocation_from_row(row: Any) -> RevocationRecord:
    return RevocationRecord(
        revocation_id=int(row["revocation_id"]),
        goal_id=row["goal_id"],
        contract_version=int(row["contract_version"]),
        revoked_at=datetime.fromisoformat(row["revoked_at"]),
        revoked_by=row["revoked_by"],
        reason=row["reason"],
        evidence_digest=row["evidence_digest"],
        provenance=row["provenance"],
        signature=row["signature"],
    )


__all__ = [
    "Allowlist",
    "AuthorizationOutcome",
    "EffectiveAuthorization",
    "GoalContractCompiler",
    "GoalContractError",
    "RevocationRecord",
    "SQLiteGoalContractStore",
    "StoredAuthorization",
]

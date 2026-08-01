"""Goal-pursuit vocabulary: contracts, run state, and the two authorizations.

The system pursues a *goal*, not a ticket queue. Three distinctions in this
module carry the weight:

**Phases are not terminal statuses.** A run is always in a ``GoalPhase``. It
carries a ``TerminalStatus`` only once the phase is ``closed``. Collapsing
these into one enum is what lets "the criteria look met" read as "done";
``ready_for_promotion`` is a phase a run can still leave.

**Candidate authorization is not goal achievement.** ``CandidateAuthorization``
asks *may this SHA merge* and is scoped to one commit. ``GoalAchievement``
asks *is the objective met* and is scoped to the whole contract. Ticket 1 of 5
is routinely authorized while the goal is nowhere near achieved; requiring
achievement to merge would deadlock the sequence.

**Digests bind evidence to the policy it was judged under.** Any edit to a
contract, harness, or policy changes its digest, which invalidates every
downstream attestation by construction rather than by invalidation logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import IntEnum
from hashlib import sha256
import json
from typing import Any, Literal

from ticket_agent.orchestrator.gates import ReviewCoverage, VerificationRecord


# --------------------------------------------------------------------------
# Autonomy
# --------------------------------------------------------------------------


class AutonomyMode(IntEnum):
    """Ordered capability ladder. Ordering is the point: resolution takes a
    ``min`` across every ceiling, so each input can only ever lower autonomy.
    """

    OBSERVE = 0
    PROPOSE = 1
    IMPLEMENT = 2
    DELIVER = 3
    AUTONOMOUS = 4

    @classmethod
    def parse(cls, value: object) -> "AutonomyMode":
        """Fail closed: anything unrecognized resolves to ``OBSERVE``."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                return cls.OBSERVE
        return cls.OBSERVE

    def __str__(self) -> str:
        return self.name.lower()


EnforcementLevel = Literal["off", "shadow", "enforce"]

RiskClass = Literal["low", "standard", "elevated", "human_only"]

#: Ceiling each risk class imposes on autonomy.
RISK_CEILING: Mapping[RiskClass, AutonomyMode] = {
    "low": AutonomyMode.AUTONOMOUS,
    "standard": AutonomyMode.DELIVER,
    "elevated": AutonomyMode.IMPLEMENT,
    "human_only": AutonomyMode.PROPOSE,
}

HarnessReadiness = Literal["unready", "partial", "full"]

READINESS_CEILING: Mapping[HarnessReadiness, AutonomyMode] = {
    "unready": AutonomyMode.PROPOSE,
    "partial": AutonomyMode.IMPLEMENT,
    "full": AutonomyMode.AUTONOMOUS,
}


# --------------------------------------------------------------------------
# Phases and terminal statuses -- deliberately separate types
# --------------------------------------------------------------------------


GoalPhase = Literal[
    "discovering",
    "implementing",
    "verifying",
    "reviewing",
    "demoing",
    "delivering",
    "integrating",
    "ready_for_promotion",
    "closed",
]

TerminalStatus = Literal[
    "achieved",
    "safe_stop",
    "policy_blocked",
    "human_judgment_required",
    "budget_exhausted",
    "cancelled",
    "failed_infrastructure",
]


# --------------------------------------------------------------------------
# Canonical serialization and digests
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic JSON for digesting: sorted keys, no incidental spacing.

    Tuples serialize as lists and datetimes as ISO strings so that a value
    round-tripped through storage digests identically to the original.
    """

    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    return str(value)


def digest(value: Any) -> str:
    """SHA-256 over the canonical serialization."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    criterion_id: str
    text: str
    #: How a pass is decided. ``None`` means no oracle is declared, which makes
    #: the criterion not-runnable rather than passing on a maker's say-so.
    oracle: Literal["deterministic", "independent_checker"] | None = None
    demo_required: bool = False


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    repositories: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Budgets:
    wall_seconds: int | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    iterations: int | None = None
    max_changed_files: int | None = None
    max_diff_bytes: int | None = None

    def exceeded_by(self, consumed: "Budgets") -> tuple[str, ...]:
        """Names of every budget ``consumed`` has met or passed."""

        breached: list[str] = []
        for name in (
            "wall_seconds",
            "tokens",
            "cost_usd",
            "iterations",
            "max_changed_files",
            "max_diff_bytes",
        ):
            limit = getattr(self, name)
            used = getattr(consumed, name)
            if limit is not None and used is not None and used >= limit:
                breached.append(name)
        return tuple(breached)


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Who authorized this goal, and by what evidence."""

    requester: str
    slack_channel: str
    slack_message_ts: str
    allowlisted: bool
    authorized_at: datetime


@dataclass(frozen=True, slots=True)
class GoalContract:
    """The authorized scope of one goal. Immutable once compiled.

    Agents revise the plan and the Jira decomposition freely beneath this;
    they may not widen ``permitted_scope``, move an item out of ``non_goals``,
    or drop a criterion. Doing any of those requires a new version carrying a
    fresh human authorization.
    """

    goal_id: str
    version: int
    schema_version: int
    authorization: AuthorizationContext
    #: The requester's verbatim words. The semantic check validates the
    #: compiled contract against *this*, never against an intermediate
    #: proposal or the compiler's own summary.
    original_request: str
    objective: str
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    non_goals: tuple[str, ...] = ()
    permitted_scope: ScopeSpec = field(default_factory=ScopeSpec)
    risk_class: RiskClass = "human_only"
    budgets: Budgets = field(default_factory=Budgets)
    human_required: tuple[str, ...] = ()
    harness_digest: str = ""
    trust_root_digest: str = ""
    policy_digest: str = ""

    def __post_init__(self) -> None:
        if not self.acceptance_criteria:
            raise ValueError("a goal contract requires at least one criterion")
        if not self.original_request.strip():
            raise ValueError("a goal contract requires the original request text")
        ids = [c.criterion_id for c in self.acceptance_criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("acceptance criterion ids must be unique")

    @property
    def contract_digest(self) -> str:
        return digest(self)

    @property
    def autonomy_ceiling(self) -> AutonomyMode:
        return RISK_CEILING[self.risk_class]


# --------------------------------------------------------------------------
# Evidence and actions
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A pointer to stored evidence, never the evidence itself.

    Content-addressed, and *current* only for the SHA and digest set it was
    produced under -- which is what makes a new push invalidate prior evidence
    without any explicit invalidation step.
    """

    kind: str
    sha256: str
    uri: str
    produced_at: datetime
    produced_by: str
    candidate_sha: str | None = None


ActionKind = Literal[
    "worktree_create",
    "model_invoke",
    "gate_run",
    "git_commit",
    "git_push",
    "jira_write",
    "slack_post",
    "pr_create",
    "pr_merge",
]

ActionState = Literal["intended", "in_flight", "done", "failed", "abandoned"]

#: Kinds whose remote effect can be probed or deduplicated, and which may
#: therefore claim effectively-once. Everything else is at-least-once with a
#: recorded duplicate risk -- notably ``model_invoke``, which has neither an
#: idempotency key nor a probe, so its spend is bounded rather than exact.
PROBEABLE_ACTIONS: frozenset[ActionKind] = frozenset(
    {"jira_write", "git_push", "pr_create", "pr_merge", "git_commit"}
)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    action_id: str
    goal_id: str
    iteration: int
    kind: ActionKind
    state: ActionState
    operation: str = ""
    natural_key: str = ""
    request_digest: str = ""
    external: bool = False
    attempts: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    result_ref: EvidenceRef | None = None
    result_identity: str | None = None
    reserved_model_cost_usd: float = 0.0
    actual_model_cost_usd: float = 0.0
    error: str | None = None
    error_classification: str | None = None
    recovery_classification: str | None = None

    @property
    def effectively_once(self) -> bool:
        """Whether replay of this action can be deduplicated remotely."""

        return self.kind in PROBEABLE_ACTIONS


def action_id(goal_id: str, iteration: int, kind: str, natural_key: str) -> str:
    """Deterministic id so a replayed action reuses its journal row."""

    return digest([goal_id, iteration, kind, natural_key])[:32]


# --------------------------------------------------------------------------
# Loop state
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyRef:
    strategy_id: str
    description: str


@dataclass(frozen=True, slots=True)
class NextAction:
    """What we will do next, and why it differs from last time.

    ``difference`` is required for a retry. A retry that cannot name what new
    evidence or strategy makes it different is refused, which is the mechanism
    that stops the loop from re-issuing an identical prompt.
    """

    description: str
    difference: str | None = None


@dataclass(frozen=True, slots=True)
class Finding:
    finding_id: str
    severity: Literal["blocking", "advisory"]
    claim: str
    file: str | None = None
    line: int | None = None
    evidence_ref: EvidenceRef | None = None
    suggested_action: str | None = None


@dataclass(frozen=True, slots=True)
class LoopState:
    """Durable intent. Recovery resumes from this, never from logs or labels."""

    goal_id: str
    contract_version: int
    phase: GoalPhase
    iteration: int = 0
    strategy: StrategyRef | None = None
    hypothesis: str = ""
    candidate_sha: str | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    verification_findings: tuple[Finding, ...] = ()
    review_findings: tuple[Finding, ...] = ()
    assumptions: tuple[str, ...] = ()
    discovered_constraints: tuple[str, ...] = ()
    failure_fingerprints: tuple[str, ...] = ()
    no_progress_count: int = 0
    consumed: Budgets = field(default_factory=Budgets)
    next_action: NextAction | None = None
    terminal_status: TerminalStatus | None = None

    def __post_init__(self) -> None:
        # The invariant that keeps "looks done" from reading as "done".
        if self.terminal_status is not None and self.phase != "closed":
            raise ValueError(
                "terminal_status is only valid in the 'closed' phase; "
                f"got phase={self.phase!r}"
            )
        if self.phase == "closed" and self.terminal_status is None:
            raise ValueError("the 'closed' phase requires a terminal_status")

    @property
    def is_terminal(self) -> bool:
        return self.phase == "closed"


# --------------------------------------------------------------------------
# The two authorizations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DigestSet:
    """Every digest a candidate was judged under, compared at merge time."""

    contract: str = ""
    harness: str = ""
    policy: str = ""
    trust_root: str = ""

    def matches(self, other: "DigestSet") -> bool:
        return self == other


@dataclass(frozen=True, slots=True)
class CandidateAuthorization:
    """May *this SHA* merge? Scoped to one commit, not to the goal."""

    repository: str
    goal_id: str
    head_sha: str
    tree_oid: str
    base_branch: str
    base_oid: str
    merge_base_oid: str
    verification: VerificationRecord
    review_coverage: ReviewCoverage
    review_verdict: Literal["approved", "rejected", "insufficient_evidence"]
    digests: DigestSet
    trust_root_untouched: bool = False
    secrets_clean: bool = False
    scope_respected: bool = False
    binaries_cleared: bool = False

    @property
    def authorized(self) -> bool:
        """Positive conjunction. Every clause must be affirmatively true."""

        return (
            self.verification.authorized
            and self.review_verdict == "approved"
            and self.review_coverage.complete
            and self.trust_root_untouched
            and self.secrets_clean
            and self.scope_respected
            and self.binaries_cleared
            and self.head_sha == self.verification.candidate_sha
        )

    def denial_reasons(self) -> tuple[str, ...]:
        """Every failing clause, for a decision log that explains itself."""

        reasons: list[str] = []
        if not self.verification.authorized:
            reasons.append("verification_not_authorized")
        if self.review_verdict != "approved":
            reasons.append(f"review_{self.review_verdict}")
        if not self.review_coverage.complete:
            reasons.append("review_coverage_incomplete")
        if not self.trust_root_untouched:
            reasons.append("trust_root_touched")
        if not self.secrets_clean:
            reasons.append("secret_detected")
        if not self.scope_respected:
            reasons.append("scope_violation")
        if not self.binaries_cleared:
            reasons.append("binary_or_submodule_change")
        if self.head_sha != self.verification.candidate_sha:
            reasons.append("verified_sha_mismatch")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class CriterionOutcome:
    criterion_id: str
    met: bool
    oracle: Literal["deterministic", "independent_checker"] | None = None
    evidence_ref: EvidenceRef | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class GoalAchievement:
    """Is the *goal* met? Scoped to the contract, evaluated by the epic loop.

    ``achieved`` deliberately requires ``promotion_pr_confirmed``: evidence
    that the criteria are met is not the same as having presented the work.
    The run reaches ``ready_for_promotion`` on evidence, and only becomes
    ``achieved`` after the promotion PR has been read back from GitHub.
    """

    goal_id: str
    contract_version: int
    criteria: Mapping[str, CriterionOutcome] = field(default_factory=dict)
    non_goals_respected: bool = False
    integration_checkpoint_passed: bool = False
    cumulative_demo_passed: bool | None = None
    merged_candidates: tuple[str, ...] = ()
    promotion_pr_url: str | None = None
    promotion_pr_confirmed: bool = False

    def covers(self, contract: GoalContract) -> bool:
        """Every declared criterion has an outcome -- no silent omissions."""

        declared = {c.criterion_id for c in contract.acceptance_criteria}
        return declared.issubset(set(self.criteria))

    @property
    def ready_for_promotion(self) -> bool:
        if not self.criteria:
            return False
        return (
            all(outcome.met for outcome in self.criteria.values())
            and self.non_goals_respected
            and self.integration_checkpoint_passed
            and self.cumulative_demo_passed is not False
        )

    @property
    def achieved(self) -> bool:
        return (
            self.ready_for_promotion
            and self.promotion_pr_url is not None
            and self.promotion_pr_confirmed
        )


def resolve_autonomy(
    *,
    configured: object,
    risk_class: RiskClass,
    readiness: HarnessReadiness,
    sandbox_available: bool,
    per_command_approval: bool = False,
    all_required_gates_enforced: bool = False,
    halted: bool = False,
) -> tuple[AutonomyMode, tuple[str, ...]]:
    """Resolve the effective mode and name every ceiling that bound it.

    Monotone by construction: each input contributes a ceiling and the result
    is their minimum, so no input can ever raise autonomy. Reporting the
    binding ceilings is what makes "why did nothing merge" answerable.
    """

    ceilings: list[tuple[str, AutonomyMode]] = [
        ("configured", AutonomyMode.parse(configured)),
        ("risk_class", RISK_CEILING[risk_class]),
        ("harness_readiness", READINESS_CEILING[readiness]),
    ]

    if not sandbox_available:
        # Without isolation, repository commands may run only under
        # per-command human approval; unattended execution is forbidden.
        ceilings.append(
            (
                "sandbox",
                AutonomyMode.IMPLEMENT if per_command_approval else AutonomyMode.PROPOSE,
            )
        )
    if not all_required_gates_enforced:
        # A gate in off or shadow authorizes nothing.
        ceilings.append(("gate_enforcement", AutonomyMode.IMPLEMENT))
    if halted:
        ceilings.append(("halted", AutonomyMode.OBSERVE))

    effective = min(mode for _, mode in ceilings)
    binding = tuple(name for name, mode in ceilings if mode == effective)
    return effective, binding


__all__ = [
    "READINESS_CEILING",
    "RISK_CEILING",
    "AcceptanceCriterion",
    "ActionKind",
    "ActionRecord",
    "ActionState",
    "AuthorizationContext",
    "AutonomyMode",
    "Budgets",
    "CandidateAuthorization",
    "CriterionOutcome",
    "DigestSet",
    "EnforcementLevel",
    "EvidenceRef",
    "Finding",
    "GoalAchievement",
    "GoalContract",
    "GoalPhase",
    "HarnessReadiness",
    "LoopState",
    "NextAction",
    "PROBEABLE_ACTIONS",
    "RiskClass",
    "ScopeSpec",
    "StrategyRef",
    "TerminalStatus",
    "action_id",
    "canonical_json",
    "digest",
    "resolve_autonomy",
]

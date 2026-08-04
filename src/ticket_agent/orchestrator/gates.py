"""Verification gate outcomes and the record that authorizes a candidate.

Authorization here is *positive*: a candidate is authorized only when every
required gate produced an affirmative ``passed``. Absence of evidence never
authorizes, which is why ``GateStatus`` distinguishes four different ways a
gate can fail to produce a pass:

``failed``        the gate ran and the code is wrong
``not_runnable``  the gate was attempted and could not execute
``skipped``       policy said not to run it
``not_run``       we never reached it, because routing short-circuited

Collapsing ``not_run`` into ``skipped`` is what lets a partial record look
complete. Every expected gate is therefore initialized to ``not_run`` before
the chain starts, so a missing outcome is impossible rather than silent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

GateName = Literal["install", "test", "lint", "typecheck", "build"]
GateStatus = Literal["passed", "failed", "not_runnable", "skipped", "not_run"]
FailureClass = Literal["defect", "flake", "transient", "policy", "unknown"]

#: Declaration order is execution order: cheapest strongest signal first.
GATE_ORDER: tuple[GateName, ...] = (
    "install",
    "test",
    "lint",
    "typecheck",
    "build",
)

#: ``install`` can fail and must be representable, but it is not declarable in
#: a repo contract -- it is governed by ``policy.dependency_install_allowed``.
DECLARABLE_GATES: frozenset[GateName] = frozenset(
    {"test", "lint", "typecheck", "build"}
)

#: The gates the runtime actually executes. A contract may declare a command
#: for any gate in DECLARABLE_GATES, but only these reach an executor -- the
#: `run_tests` node runs `test` and nothing runs the rest. A required gate
#: outside this set is a hole in the evidence, not a preference, so readiness
#: and the autonomy resolver both read it from here rather than repeating a
#: literal that could drift from what is wired.
EXECUTABLE_GATES: frozenset[GateName] = frozenset({"test"})


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """The result of one gate. ``status`` is the only field routing reads."""

    gate: GateName
    status: GateStatus
    failure_class: FailureClass | None = None
    exit_code: int | None = None
    output: str = ""
    timed_out: bool = False
    error: str | None = None
    command: tuple[str, ...] = ()
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        if self.status == "passed" and self.failure_class is not None:
            raise ValueError("a passed gate must not carry a failure_class")

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def not_run(gate: GateName) -> GateOutcome:
    """The initial outcome for every expected gate."""

    return GateOutcome(gate=gate, status="not_run")


def initial_outcomes(gates: Iterable[GateName]) -> dict[GateName, GateOutcome]:
    """Seed a full outcome map so absence is never silence."""

    return {gate: not_run(gate) for gate in gates}


@dataclass(frozen=True, slots=True)
class ReviewCoverage:
    """How much of the diff the reviewer actually saw.

    Partial coverage denies authorization. A reviewer that was shown 90% of a
    change reviewed a different change than the one being shipped.
    """

    files_total: int = 0
    files_reviewed: int = 0
    hunks_total: int = 0
    hunks_reviewed: int = 0
    bytes_reviewed: int = 0
    chunk_count: int = 0
    excluded: tuple[str, ...] = ()
    truncated: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True only when every material file and hunk was reviewed intact."""

        return (
            self.files_total > 0
            and self.files_reviewed == self.files_total
            and self.hunks_reviewed == self.hunks_total
            and not self.truncated
        )


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """The policy a verification run was judged under.

    Carried alongside the evidence so a later merge gate evaluates against the
    policy in force at verification time, and can reject on a digest mismatch
    rather than silently re-reading a contract that has since changed.
    """

    schema_version: int
    policy_version: int
    required_gates: tuple[GateName, ...]
    optional_gates: tuple[GateName, ...] = ()
    contract_digest: str = ""
    harness_digest: str = ""
    sandbox_profile: str = ""

    def __post_init__(self) -> None:
        overlap = set(self.required_gates) & set(self.optional_gates)
        if overlap:
            raise ValueError(
                f"gates cannot be both required and optional: {sorted(overlap)}"
            )

    @property
    def expected_gates(self) -> tuple[GateName, ...]:
        expected = set(self.required_gates) | set(self.optional_gates)
        return tuple(gate for gate in GATE_ORDER if gate in expected)


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """Immutable evidence that a specific commit was verified under a policy."""

    policy: VerificationPolicy
    candidate_sha: str
    outcomes: Mapping[GateName, GateOutcome] = field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    mutation_detected: bool = False

    @property
    def authorized(self) -> bool:
        """Every required gate present and exactly ``passed``.

        An empty policy denies. A record with no outcomes denies. An
        all-``skipped`` record denies. Any ``not_run`` denies. There is no
        path by which absent evidence authorizes.
        """

        if not self.policy.required_gates:
            return False
        if self.mutation_detected:
            return False
        return all(
            (outcome := self.outcomes.get(gate)) is not None and outcome.passed
            for gate in self.policy.required_gates
        )

    @property
    def first_non_passing(self) -> GateOutcome | None:
        """The earliest gate in execution order that did not pass."""

        for gate in GATE_ORDER:
            outcome = self.outcomes.get(gate)
            if outcome is not None and not outcome.passed:
                return outcome
        for gate in self.policy.required_gates:
            if gate not in self.outcomes:
                return not_run(gate)
        return None

    @property
    def failure_class(self) -> FailureClass | None:
        if self.mutation_detected:
            return "policy"
        outcome = self.first_non_passing
        return None if outcome is None else outcome.failure_class


__all__ = [
    "DECLARABLE_GATES",
    "GATE_ORDER",
    "FailureClass",
    "GateName",
    "GateOutcome",
    "GateStatus",
    "ReviewCoverage",
    "VerificationPolicy",
    "VerificationRecord",
    "initial_outcomes",
    "not_run",
]

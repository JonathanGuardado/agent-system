"""Deterministic risk and scope classification.

Risk is decided by rules, never by a model. A system that lets the agent grade
its own risk has no risk classification -- the same reasoning that makes a
model an unsuitable reviewer of its own diff makes it unsuitable as its own
risk assessor.

The rules are *data*, versioned into ``policy_digest`` and carried on the goal
contract, so a later gate can tell whether the policy that authorized a change
is still the policy in force.

Two entry points share one rule set on purpose:

``classify_request``  at intake, over what was asked for
``classify_changes``  at authorization, over what git actually shows changed

They must not drift, because the interesting failure is precisely when they
disagree: a request that classified as ``standard`` producing a diff that
touches CI config. Sharing the rules means the second call can catch it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.types import (
    RISK_CEILING,
    AutonomyMode,
    Budgets,
    RiskClass,
    ScopeSpec,
    digest,
)

#: Ordered least to most restrictive, so `max` picks the stricter of two.
_RISK_ORDER: tuple[RiskClass, ...] = ("low", "standard", "elevated", "human_only")


class RiskPolicyError(AgentSystemError):
    """Raised when the risk policy cannot be loaded or applied."""


def _rank(risk: RiskClass) -> int:
    return _RISK_ORDER.index(risk)


def stricter(left: RiskClass, right: RiskClass) -> RiskClass:
    return left if _rank(left) >= _rank(right) else right


@dataclass(frozen=True, slots=True)
class ChangeClassRule:
    """A path pattern and the risk it carries."""

    name: str
    patterns: tuple[str, ...]
    risk: RiskClass
    reason: str = ""

    def matches(self, path: str) -> bool:
        candidate = path.strip().removeprefix("./")
        return any(
            fnmatch(candidate, pattern) or candidate.startswith(pattern.rstrip("*"))
            for pattern in self.patterns
        )


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Size limits above which a change is treated as more risky."""

    max_changed_files: int | None = None
    max_diff_bytes: int | None = None
    max_cost_usd: float | None = None
    over_threshold_risk: RiskClass = "elevated"


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one classification, with every reason recorded."""

    risk: RiskClass
    reasons: tuple[str, ...] = ()
    matched_rules: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()

    @property
    def in_scope(self) -> bool:
        return not self.out_of_scope

    @property
    def ceiling(self) -> AutonomyMode:
        return RISK_CEILING[self.risk]


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """Versioned rule data. The only thing that assigns a risk class."""

    version: int
    repositories: tuple[str, ...] = ()
    change_classes: tuple[ChangeClassRule, ...] = ()
    thresholds: Thresholds = field(default_factory=Thresholds)
    #: Risk assigned when nothing matches and everything is in scope.
    baseline_risk: RiskClass = "standard"
    #: Risk assigned when the rules cannot reach a conclusion. Deliberately
    #: the most restrictive value -- an unclassifiable request is not a safe
    #: request, it is an unknown one.
    unclassifiable_risk: RiskClass = "human_only"

    @property
    def policy_digest(self) -> str:
        return digest(self)

    # -- entry points ------------------------------------------------------

    def classify_request(
        self,
        *,
        repositories: Sequence[str],
        scope: ScopeSpec | None = None,
        budgets: Budgets | None = None,
    ) -> Decision:
        """Classify what was asked for, before any code exists."""

        reasons: list[str] = []
        risk: RiskClass = self.baseline_risk

        if not repositories:
            return Decision(
                risk=self.unclassifiable_risk,
                reasons=("no target repository could be determined",),
            )

        unknown = [r for r in repositories if not self._repo_allowed(r)]
        if unknown:
            reasons.append(f"repository not in the allowlist: {', '.join(unknown)}")
            risk = stricter(risk, self.unclassifiable_risk)

        if scope is not None and scope.allowed_paths:
            path_decision = self._classify_paths(scope.allowed_paths)
            risk = stricter(risk, path_decision.risk)
            reasons.extend(path_decision.reasons)

        if budgets is not None:
            budget_decision = self._classify_budgets(budgets)
            risk = stricter(risk, budget_decision.risk)
            reasons.extend(budget_decision.reasons)

        return Decision(risk=risk, reasons=tuple(reasons))

    def classify_changes(
        self,
        changed_paths: Iterable[str],
        *,
        scope: ScopeSpec | None = None,
        diff_bytes: int | None = None,
    ) -> Decision:
        """Classify what actually changed.

        Run at candidate authorization. Intake classified an *intention*; this
        classifies the artifact, and the two diverge exactly when it matters.
        """

        paths = [p for p in changed_paths if p and p.strip()]
        if not paths:
            return Decision(
                risk=self.unclassifiable_risk,
                reasons=("no changed files were reported",),
            )

        decision = self._classify_paths(paths)
        risk = stricter(self.baseline_risk, decision.risk)
        reasons = list(decision.reasons)
        out_of_scope: list[str] = []

        if scope is not None:
            out_of_scope = [p for p in paths if not _within_scope(p, scope)]
            if out_of_scope:
                shown = ", ".join(sorted(out_of_scope)[:5])
                reasons.append(f"paths outside the permitted scope: {shown}")
                risk = stricter(risk, "human_only")

        threshold_decision = self._classify_size(
            changed_files=len(paths), diff_bytes=diff_bytes
        )
        risk = stricter(risk, threshold_decision.risk)
        reasons.extend(threshold_decision.reasons)

        return Decision(
            risk=risk,
            reasons=tuple(reasons),
            matched_rules=decision.matched_rules,
            out_of_scope=tuple(sorted(out_of_scope)),
        )

    # -- internals ---------------------------------------------------------

    def _repo_allowed(self, repository: str) -> bool:
        if not self.repositories:
            # An empty allowlist permits nothing. Fail closed: an operator who
            # has not said which repositories may be touched has not
            # authorized any of them.
            return False
        return any(fnmatch(repository, pattern) for pattern in self.repositories)

    def _classify_paths(self, paths: Sequence[str]) -> Decision:
        risk: RiskClass = "low"
        reasons: list[str] = []
        matched: list[str] = []
        for rule in self.change_classes:
            hits = sorted({p for p in paths if rule.matches(p)})
            if not hits:
                continue
            matched.append(rule.name)
            risk = stricter(risk, rule.risk)
            detail = rule.reason or f"change class {rule.name!r}"
            reasons.append(f"{detail}: {', '.join(hits[:5])}")
        return Decision(risk=risk, reasons=tuple(reasons), matched_rules=tuple(matched))

    def _classify_budgets(self, budgets: Budgets) -> Decision:
        return self._classify_size(
            changed_files=budgets.max_changed_files,
            diff_bytes=budgets.max_diff_bytes,
            cost_usd=budgets.cost_usd,
        )

    def _classify_size(
        self,
        *,
        changed_files: int | None = None,
        diff_bytes: int | None = None,
        cost_usd: float | None = None,
    ) -> Decision:
        reasons: list[str] = []
        risk: RiskClass = "low"
        checks = (
            ("changed files", changed_files, self.thresholds.max_changed_files),
            ("diff bytes", diff_bytes, self.thresholds.max_diff_bytes),
            ("cost", cost_usd, self.thresholds.max_cost_usd),
        )
        for label, value, limit in checks:
            if limit is not None and value is not None and value > limit:
                reasons.append(f"{label} {value} exceeds threshold {limit}")
                risk = stricter(risk, self.thresholds.over_threshold_risk)
        return Decision(risk=risk, reasons=tuple(reasons))


def _within_scope(path: str, scope: ScopeSpec) -> bool:
    candidate = path.strip().removeprefix("./")
    for denied in scope.denied_paths:
        if fnmatch(candidate, denied) or candidate.startswith(denied.rstrip("*")):
            return False
    if not scope.allowed_paths:
        return True
    return any(
        fnmatch(candidate, allowed) or candidate.startswith(allowed.rstrip("*"))
        for allowed in scope.allowed_paths
    )


DEFAULT_POLICY = RiskPolicy(
    version=1,
    repositories=(),
    change_classes=(
        ChangeClassRule(
            name="ci",
            patterns=(".github/*", ".gitlab-ci.yml", "Jenkinsfile"),
            risk="human_only",
            reason="continuous integration config decides what verification runs",
        ),
        ChangeClassRule(
            name="delivery-config",
            patterns=("config/repos/*", "config/policy/*", "CODEOWNERS"),
            risk="human_only",
            reason="delivery configuration governs the agent itself",
        ),
        ChangeClassRule(
            name="dependency-manifest",
            patterns=(
                "package.json",
                "package-lock.json",
                "pyproject.toml",
                "poetry.lock",
                "requirements*.txt",
                "go.mod",
                "Cargo.toml",
            ),
            risk="elevated",
            reason="dependency changes introduce third-party code",
        ),
        ChangeClassRule(
            name="infrastructure",
            patterns=("Dockerfile", "docker-compose*.yml", "*.tf", "k8s/*"),
            risk="elevated",
            reason="infrastructure changes affect the running environment",
        ),
        ChangeClassRule(
            name="migration",
            patterns=("**/migrations/*", "supabase/migrations/*"),
            risk="elevated",
            reason="schema migrations are difficult to reverse",
        ),
        ChangeClassRule(
            name="secrets",
            patterns=(".env", ".env.*", "secrets/*"),
            risk="human_only",
            reason="credential material must never be agent-authored",
        ),
    ),
    thresholds=Thresholds(
        max_changed_files=40,
        max_diff_bytes=200_000,
        over_threshold_risk="elevated",
    ),
)


def load_risk_policy(path: str | Path | None = None) -> RiskPolicy:
    """Load rule data, falling back to the built-in defaults.

    The defaults deliberately have an **empty repository allowlist**, which
    permits nothing. An operator must name the repositories the agent may
    touch; silence is not consent.
    """

    if path is None:
        return DEFAULT_POLICY
    policy_path = Path(path)
    if not policy_path.exists():
        return DEFAULT_POLICY
    try:
        raw = yaml.safe_load(policy_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise RiskPolicyError(f"could not parse risk policy {policy_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RiskPolicyError(f"risk policy must be a mapping: {policy_path}")

    return RiskPolicy(
        version=int(raw.get("version", 1)),
        repositories=tuple(str(r) for r in raw.get("repositories", ())),
        change_classes=tuple(
            _parse_rule(index, item) for index, item in enumerate(raw.get("change_classes", ()))
        )
        or DEFAULT_POLICY.change_classes,
        thresholds=_parse_thresholds(raw.get("thresholds")),
        baseline_risk=_parse_risk(raw.get("baseline_risk", "standard"), "baseline_risk"),
        unclassifiable_risk=_parse_risk(
            raw.get("unclassifiable_risk", "human_only"), "unclassifiable_risk"
        ),
    )


def _parse_rule(index: int, raw: Any) -> ChangeClassRule:
    if not isinstance(raw, Mapping):
        raise RiskPolicyError(f"change_classes[{index}] must be a mapping")
    patterns = raw.get("patterns", ())
    if not isinstance(patterns, Sequence) or isinstance(patterns, str) or not patterns:
        raise RiskPolicyError(f"change_classes[{index}].patterns must be a non-empty list")
    return ChangeClassRule(
        name=str(raw.get("name") or f"rule-{index}"),
        patterns=tuple(str(p) for p in patterns),
        risk=_parse_risk(raw.get("risk", "elevated"), f"change_classes[{index}].risk"),
        reason=str(raw.get("reason", "")),
    )


def _parse_thresholds(raw: Any) -> Thresholds:
    if raw is None:
        return DEFAULT_POLICY.thresholds
    if not isinstance(raw, Mapping):
        raise RiskPolicyError("thresholds must be a mapping")
    return Thresholds(
        max_changed_files=raw.get("max_changed_files"),
        max_diff_bytes=raw.get("max_diff_bytes"),
        max_cost_usd=raw.get("max_cost_usd"),
        over_threshold_risk=_parse_risk(
            raw.get("over_threshold_risk", "elevated"), "thresholds.over_threshold_risk"
        ),
    )


def _parse_risk(value: Any, label: str) -> RiskClass:
    text = str(value)
    if text not in _RISK_ORDER:
        raise RiskPolicyError(
            f"{label} must be one of {', '.join(_RISK_ORDER)}; got {text!r}"
        )
    return text


__all__ = [
    "DEFAULT_POLICY",
    "ChangeClassRule",
    "Decision",
    "RiskPolicy",
    "RiskPolicyError",
    "Thresholds",
    "load_risk_policy",
    "stricter",
]

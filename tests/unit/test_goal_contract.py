"""Tests for goal-contract compilation, risk policy, signing, and the checker.

The theme throughout: every way of *not* knowing something must deny. An
unclassifiable request, an unreachable checker, an unparseable response, and a
missing signature are all different from approval, and none of them may be
mistaken for it.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ticket_agent.goal.contract import (
    Allowlist,
    GoalContractCompiler,
    GoalContractError,
    SQLiteGoalContractStore,
)
from ticket_agent.goal.identity import GoalIdentityError
from ticket_agent.goal.policy import (
    DEFAULT_POLICY,
    ChangeClassRule,
    RiskPolicy,
    RiskPolicyError,
    load_risk_policy,
    stricter,
)
from ticket_agent.goal.semantic_check import (
    ModelSemanticChecker,
    NullSemanticChecker,
    SemanticVerdict,
    parse_verdict,
)
from ticket_agent.goal.signing import (
    NullSigner,
    SigningError,
    generate_key,
    load_signer,
)
from ticket_agent.goal.types import Budgets, ScopeSpec

_POLICY = replace(DEFAULT_POLICY, repositories=("ofertas-sv",))
_SCOPE = ScopeSpec(repositories=("ofertas-sv",), allowed_paths=("src/*", "tests/*"))


# -- risk policy -----------------------------------------------------------


def test_empty_repository_allowlist_permits_nothing():
    """Silence is not consent."""

    decision = DEFAULT_POLICY.classify_request(repositories=["anything"])

    assert decision.risk == "human_only"


def test_unknown_repository_is_human_only():
    assert _POLICY.classify_request(repositories=["sketchy"]).risk == "human_only"


def test_known_repository_gets_the_baseline():
    assert _POLICY.classify_request(repositories=["ofertas-sv"]).risk == "standard"


def test_a_request_with_no_repository_cannot_be_classified():
    decision = _POLICY.classify_request(repositories=[])

    assert decision.risk == "human_only"
    assert any("no target repository" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    "path,expected_rule",
    [
        (".github/workflows/ci.yml", "ci"),
        ("config/repos/lab.yaml", "delivery-config"),
        ("package.json", "dependency-manifest"),
        ("Dockerfile", "infrastructure"),
        (".env.production", "secrets"),
    ],
)
def test_change_classes_match_their_paths(path, expected_rule):
    decision = _POLICY.classify_changes([path])

    assert expected_rule in decision.matched_rules


def test_ordinary_source_changes_stay_at_the_baseline():
    decision = _POLICY.classify_changes(
        ["src/app.tsx", "tests/app.test.ts"], scope=_SCOPE
    )

    assert decision.risk == "standard"
    assert decision.in_scope


def test_paths_outside_scope_require_a_human():
    decision = _POLICY.classify_changes(["src/app.tsx", "infra/main.tf"], scope=_SCOPE)

    assert decision.risk == "human_only"
    assert decision.out_of_scope == ("infra/main.tf",)


def test_denied_paths_win_over_allowed_paths():
    scope = ScopeSpec(allowed_paths=("src/*",), denied_paths=("src/secrets/*",))

    decision = _POLICY.classify_changes(["src/secrets/keys.ts"], scope=scope)

    assert not decision.in_scope


def test_an_empty_change_set_cannot_be_classified():
    """No reported changes is not the same as no risky changes."""

    assert _POLICY.classify_changes([]).risk == "human_only"


def test_size_thresholds_raise_the_class():
    decision = _POLICY.classify_changes(
        [f"src/file{n}.ts" for n in range(60)], scope=_SCOPE
    )

    assert decision.risk == "elevated"


def test_request_and_change_classification_share_one_rule_set():
    """The interesting failure is when the two disagree, so they must not drift."""

    requested = _POLICY.classify_request(repositories=["ofertas-sv"], scope=_SCOPE)
    actual = _POLICY.classify_changes([".github/workflows/ci.yml"], scope=_SCOPE)

    assert requested.risk == "standard"
    assert actual.risk == "human_only"


def test_stricter_picks_the_more_restrictive():
    assert stricter("low", "elevated") == "elevated"
    assert stricter("human_only", "low") == "human_only"


def test_policy_digest_changes_with_the_rules():
    baseline = _POLICY.policy_digest
    changed = replace(
        _POLICY,
        change_classes=_POLICY.change_classes
        + (ChangeClassRule(name="extra", patterns=("x/*",), risk="low"),),
    )

    assert changed.policy_digest != baseline
    assert replace(_POLICY).policy_digest == baseline


def test_invalid_risk_value_is_rejected_at_load(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("version: 1\nbaseline_risk: mostly_fine\n")

    with pytest.raises(RiskPolicyError, match="baseline_risk"):
        load_risk_policy(path)


def test_missing_policy_file_falls_back_to_defaults(tmp_path):
    assert load_risk_policy(tmp_path / "absent.yaml") is DEFAULT_POLICY


def test_policy_can_be_loaded_from_data(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "version: 3\n"
        "repositories: ['demo']\n"
        "change_classes:\n"
        "  - name: docs\n"
        "    patterns: ['docs/*']\n"
        "    risk: low\n"
    )

    policy = load_risk_policy(path)

    assert policy.version == 3
    assert policy.classify_request(repositories=["demo"]).risk == "standard"
    assert "docs" in policy.classify_changes(["docs/readme.md"]).matched_rules


# -- signing ---------------------------------------------------------------


@pytest.fixture()
def signer(tmp_path):
    key = tmp_path / "key"
    key.write_bytes(generate_key())
    key.chmod(0o600)
    return load_signer(key, data_dir=tmp_path / "data")


def test_signature_verifies_and_detects_tampering(signer):
    payload = {"goal": "ship it"}
    signature = signer.sign(payload)

    assert signer.verify(payload, signature)
    assert not signer.verify({"goal": "ship something else"}, signature)


@pytest.mark.parametrize(
    "bad", ["nonsense", "1:hmac-sha256:" + "0" * 64, "99:hmac-sha256:abc", "a:b:c"]
)
def test_malformed_or_wrong_signatures_return_false_not_raise(signer, bad):
    """'Could not check' must never be reachable as 'verified'."""

    assert signer.verify({"goal": "ship it"}, bad) is False


def test_key_must_not_be_group_or_world_accessible(tmp_path):
    key = tmp_path / "key"
    key.write_bytes(generate_key())
    key.chmod(0o644)

    with pytest.raises(SigningError, match="world-accessible"):
        load_signer(key, data_dir=tmp_path / "data")


def test_key_may_not_live_inside_the_directory_it_protects(tmp_path):
    """A key stored beside the rows is readable by anything that can forge them."""

    data = tmp_path / "data"
    data.mkdir()
    key = data / "key"
    key.write_bytes(generate_key())
    key.chmod(0o600)

    with pytest.raises(SigningError, match="outside the data directory"):
        load_signer(key, data_dir=data)


def test_short_key_is_refused(tmp_path):
    key = tmp_path / "key"
    key.write_bytes(b"tooshort")
    key.chmod(0o600)

    with pytest.raises(SigningError, match="at least"):
        load_signer(key, data_dir=tmp_path / "data")


def test_null_signer_refuses_rather_than_producing_unsigned_records():
    with pytest.raises(SigningError):
        NullSigner().sign({"goal": "x"})
    assert NullSigner().verify({"goal": "x"}, "1:hmac-sha256:abc") is False


# -- semantic check --------------------------------------------------------


def test_verdict_requires_literal_true():
    """A checker must not pass a question by answering 'yes' or 1."""

    verdict = parse_verdict(
        '{"objective_matches": "yes", "criteria_complete": 1, "nothing_invented": true}'
    )

    assert not verdict.agrees


def test_verdict_parses_fenced_json():
    verdict = parse_verdict(
        '```json\n{"objective_matches": true, "criteria_complete": true, '
        '"nothing_invented": true}\n```'
    )

    assert verdict.agrees


@pytest.mark.parametrize(
    "content",
    ["not json at all", "", '{"objective_matches": true}', "{}"],
)
def test_unusable_responses_deny(content):
    assert not parse_verdict(content).agrees


def test_null_checker_denies_rather_than_approving():
    """A forgotten wiring must surface as 'needs a human', not as approval."""

    verdict = asyncio.run(NullSemanticChecker().check(_contract()))

    assert not verdict.agrees
    assert "no semantic checker configured" in verdict.error


def test_checker_falling_back_onto_the_compiler_fails_closed():
    """A compiler outage must not quietly make the compiler its own checker."""

    class _Response:
        provider = "deepseek"
        content = (
            '{"objective_matches": true, "criteria_complete": true, '
            '"nothing_invented": true}'
        )

    class _Router:
        async def invoke(self, **kwargs):
            return _Response()

    verdict = asyncio.run(
        ModelSemanticChecker(_Router()).check(
            _contract(), exclude_providers=("deepseek",)
        )
    )

    assert not verdict.agrees
    assert "own work" in (verdict.error or "")


def test_checker_on_a_different_provider_is_accepted():
    class _Response:
        provider = "gemini"
        content = (
            '{"objective_matches": true, "criteria_complete": true, '
            '"nothing_invented": true}'
        )

    class _Router:
        async def invoke(self, **kwargs):
            return _Response()

    verdict = asyncio.run(
        ModelSemanticChecker(_Router()).check(
            _contract(), exclude_providers=("deepseek",)
        )
    )

    assert verdict.agrees


def test_router_failure_denies():
    class _Router:
        async def invoke(self, **kwargs):
            raise RuntimeError("provider down")

    verdict = asyncio.run(ModelSemanticChecker(_Router()).check(_contract()))

    assert not verdict.agrees
    assert "provider down" in (verdict.error or "")


def test_checker_reads_the_verbatim_request_not_a_summary():
    captured: dict = {}

    class _Router:
        async def invoke(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            raise RuntimeError("stop here")

    asyncio.run(ModelSemanticChecker(_Router()).check(_contract()))
    prompt = captured["messages"][-1]["content"]

    assert "Build a landing page in Spanish" in prompt
    assert "data to analyse" in captured["messages"][0]["content"]


# -- compilation and authorization -----------------------------------------


class _Agreeing:
    async def check(self, contract, *, exclude_providers=()):
        return SemanticVerdict(True, True, True)


class _Objecting:
    async def check(self, contract, *, exclude_providers=()):
        return SemanticVerdict(True, False, True, missing=("Spanish copy",))


def _contract():
    outcome = asyncio.run(_compile())
    return outcome.contract


def _compiler(signer=None, checker=None, users=("U1",)):
    return GoalContractCompiler(
        policy=_POLICY,
        allowlist=Allowlist(users=frozenset(users), channels=frozenset({"C1"})),
        signer=signer,
        semantic_checker=checker or _Agreeing(),
        harness_digest="h1",
        trust_root_digest="t1",
    )


async def _compile(signer=None, checker=None, **overrides):
    request = dict(
        goal_id="prop-000000000001",
        original_request="Build a landing page in Spanish",
        objective="Ship a Spanish landing page",
        acceptance_criteria=["Page renders", "Copy is in Spanish"],
        user_id="U1",
        channel="C1",
        thread_ts="1.0",
        repositories=["ofertas-sv"],
    )
    users = overrides.pop("users", ("U1",))
    request.update(overrides)
    return await _compiler(signer, checker, users).compile(**request)


def test_allowlisted_in_policy_request_is_authorized(signer):
    outcome = asyncio.run(_compile(signer))

    assert outcome.authorized
    assert outcome.contract.risk_class == "standard"
    assert outcome.escalation_reasons() == ()


def test_non_allowlisted_user_is_not_authorized(signer):
    outcome = asyncio.run(_compile(signer, user_id="U9"))

    assert not outcome.authorized
    assert outcome.contract.risk_class == "human_only"


def test_non_allowlisted_channel_is_not_authorized(signer):
    outcome = asyncio.run(_compile(signer, channel="C-random"))

    assert not outcome.authorized


def test_empty_allowlist_authorizes_nobody(signer):
    outcome = asyncio.run(_compile(signer, users=()))

    assert not outcome.authorized


def test_semantic_disagreement_blocks_and_says_why(signer):
    outcome = asyncio.run(_compile(signer, checker=_Objecting()))

    assert not outcome.authorized
    assert any("Spanish copy" in reason for reason in outcome.escalation_reasons())


def test_unsigned_contract_is_not_authorized():
    """No signing key means no authority, not implicit trust."""

    outcome = asyncio.run(_compile(NullSigner()))

    assert outcome.signature is None
    assert not outcome.authorized
    assert any("signed" in reason for reason in outcome.escalation_reasons())


def test_elevated_risk_is_not_authorized_even_when_everything_else_passes(signer):
    outcome = asyncio.run(
        _compile(signer, budgets=Budgets(max_changed_files=999, max_diff_bytes=999_999))
    )

    assert outcome.contract.risk_class == "elevated"
    assert not outcome.authorized


def test_contract_records_the_verbatim_request(signer):
    outcome = asyncio.run(_compile(signer))

    assert outcome.contract.original_request == "Build a landing page in Spanish"


# -- amendment and storage -------------------------------------------------


def test_amendment_creates_a_new_version_and_leaves_the_original_intact(signer):
    outcome = asyncio.run(_compile(signer))
    compiler = _compiler(signer)

    amended, signature = compiler.amend(
        outcome.contract,
        authorized_by="U1",
        reason="agreed in thread",
        add_criteria=["Mobile layout"],
    )

    assert amended.version == 2
    assert len(amended.acceptance_criteria) == 3
    assert len(outcome.contract.acceptance_criteria) == 2
    assert compiler.verify(amended, signature)


@pytest.mark.parametrize("field", ["authorized_by", "reason"])
def test_amendment_requires_a_human_and_a_reason(signer, field):
    outcome = asyncio.run(_compile(signer))
    kwargs = {"authorized_by": "U1", "reason": "because", field: "  "}

    with pytest.raises(GoalContractError):
        _compiler(signer).amend(outcome.contract, **kwargs)


def test_store_round_trips_and_detects_a_tampered_row(tmp_path, signer):
    outcome = asyncio.run(_compile(signer))
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save(outcome.contract, outcome.signature)
        assert store.verify_stored("prop-000000000001", 1, signer)

        store._connection.execute(
            "UPDATE goal_contracts SET payload = replace(payload, 'Spanish', 'English')"
        )
        assert not store.verify_stored("prop-000000000001", 1, signer)
    finally:
        store.close()


def test_store_refuses_to_overwrite_a_version_with_different_content(tmp_path, signer):
    """An immutable record that can be replaced is not immutable."""

    outcome = asyncio.run(_compile(signer))
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save(outcome.contract, outcome.signature)
        store.save(outcome.contract, outcome.signature)  # identical: fine

        different = replace(outcome.contract, objective="Something else entirely")
        with pytest.raises(GoalContractError, match="refusing to overwrite"):
            store.save(different, outcome.signature)
    finally:
        store.close()


def test_store_tracks_the_latest_version(tmp_path, signer):
    outcome = asyncio.run(_compile(signer))
    compiler = _compiler(signer)
    amended, signature = compiler.amend(
        outcome.contract, authorized_by="U1", reason="more scope"
    )
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save(outcome.contract, outcome.signature)
        store.save(amended, signature)

        assert store.latest_version("prop-000000000001") == 2
        with pytest.raises(GoalIdentityError):
            store.latest_version("unknown")
    finally:
        store.close()


def test_durable_authorization_round_trips_and_verifies(tmp_path, signer):
    outcome = asyncio.run(_compile(signer))
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(outcome)

        stored = store.load_authorization("prop-000000000001")
        effective = store.effective_authorization(
            "prop-000000000001",
            signer,
        )

        assert stored is not None
        assert stored.contract == outcome.contract
        assert stored.decision == "authorized"
        assert stored.semantic.agrees is True
        assert stored.evidence_digest
        assert effective.authorized is True
        assert effective.reasons == ()
    finally:
        store.close()


def test_denied_semantic_decision_remains_auditable(tmp_path, signer):
    outcome = asyncio.run(_compile(signer, checker=_Objecting()))
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(outcome)

        stored = store.load_authorization("prop-000000000001", 1)
        effective = store.effective_authorization(
            "prop-000000000001",
            signer,
        )

        assert stored is not None
        assert stored.decision == "denied"
        assert stored.semantic.agrees is False
        assert any("Spanish copy" in reason for reason in stored.denial_reasons)
        assert effective.authorized is False
        assert any("Spanish copy" in reason for reason in effective.reasons)
    finally:
        store.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE goal_contracts SET decision = 'denied'",
        "UPDATE goal_contracts SET contract_digest = 'tampered'",
        "UPDATE goal_contracts SET evidence_digest = 'tampered'",
        "UPDATE goal_contracts SET evidence_signature = '1:hmac-sha256:bad'",
        "UPDATE goal_contracts SET semantic_payload = "
        "replace(semantic_payload, 'true', 'false')",
    ),
)
def test_tampered_durable_authorization_fails_closed(tmp_path, signer, mutation):
    outcome = asyncio.run(_compile(signer))
    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    try:
        store.save_outcome(outcome)
        store._connection.execute(mutation)

        effective = store.effective_authorization(
            "prop-000000000001",
            signer,
        )

        assert effective.authorized is False
        assert effective.reasons
    finally:
        store.close()


def test_revocation_is_append_only_and_removes_effective_authority(
    tmp_path,
    signer,
):
    outcome = asyncio.run(_compile(signer))
    store = SQLiteGoalContractStore(
        tmp_path / "contracts.sqlite3",
        clock=lambda: datetime(2027, 7, 29, tzinfo=timezone.utc),
    )
    try:
        store.save_outcome(outcome)
        original_payload = store.stored_payload("prop-000000000001", 1)

        revoked = store.append_revocation(
            "prop-000000000001",
            1,
            revoked_by="operator@example.test",
            reason="scope withdrawn",
            signer=signer,
        )
        effective = store.effective_authorization(
            "prop-000000000001",
            signer,
        )

        assert revoked.reason == "scope withdrawn"
        assert len(store.revocations_for("prop-000000000001", 1)) == 1
        assert store.stored_payload("prop-000000000001", 1) == original_payload
        assert effective.authorized is False
        assert effective.revoked_at == revoked.revoked_at
        assert effective.reasons == ("authorization revoked: scope withdrawn",)
    finally:
        store.close()


def test_latest_valid_revocation_explains_effective_decision(tmp_path, signer):
    outcome = asyncio.run(_compile(signer))
    revocation_times = iter(
        (
            datetime(2027, 7, 29, tzinfo=timezone.utc),
            datetime(2027, 7, 30, tzinfo=timezone.utc),
        )
    )
    store = SQLiteGoalContractStore(
        tmp_path / "contracts.sqlite3",
        clock=lambda: next(revocation_times),
    )
    try:
        store.save_outcome(outcome)
        store.append_revocation(
            "prop-000000000001",
            1,
            revoked_by="operator@example.test",
            reason="initial stop",
            signer=signer,
        )
        latest = store.append_revocation(
            "prop-000000000001",
            1,
            revoked_by="operator@example.test",
            reason="confirmed stop",
            signer=signer,
        )

        effective = store.effective_authorization(
            "prop-000000000001",
            signer,
        )

        assert len(store.revocations_for("prop-000000000001", 1)) == 2
        assert effective.revoked_at == latest.revoked_at
        assert effective.reasons == ("authorization revoked: confirmed stop",)
    finally:
        store.close()


# -- intake wiring ---------------------------------------------------------


def test_approval_records_a_signed_contract(tmp_path, signer):
    """The authorizer must actually fire from the approval path.

    A module nothing calls looks identical to a working one, so this drives it
    through ApprovalFlow rather than calling the authorizer directly.
    """

    import asyncio as _asyncio

    from ticket_agent.domain.intake import IntakeMode, Proposal, ProposalStatus, TicketSpec
    from ticket_agent.goal.authorizer import ProposalGoalAuthorizer
    from ticket_agent.intake.approval_flow import ApprovalFlow
    from datetime import datetime, timezone

    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    authorizer = ProposalGoalAuthorizer(_compiler(signer), store)

    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    proposal = Proposal(
        proposal_id="prop-000000000001",
        slack_user_id="U1",
        slack_channel="C1",
        slack_thread_ts="1.0",
        mode=IntakeMode.NEW_FEATURE,
        epic_key="LAB-30",
        title="Landing page",
        summary="Ship a Spanish landing page",
        original_request="Build a landing page in Spanish",
        tickets=[
            TicketSpec(
                summary="Page",
                repository="ofertas-sv",
                acceptance_criteria=["Page renders", "Copy is in Spanish"],
            )
        ],
        status=ProposalStatus.AWAITING_CONFIRMATION,
        created_at=now,
        expires_at=now,
    )

    from ticket_agent.intake.jira_writer import JiraWriteResult

    result = JiraWriteResult(
        project_key="LAB",
        created_epic_key="LAB-30",
        created_ticket_keys=("LAB-31",),
        execution_ready_ticket_keys=("LAB-31",),
    )

    events: list[tuple[str, dict]] = []
    writer = _JiraWriterStub(result)
    flow = ApprovalFlow(
        resolver=object(),
        generator=object(),
        store=_ProposalStoreStub(),
        jira_writer=writer,
        slack=_SlackStub(),
        emit=lambda name, payload: events.append((name, payload)),
        goal_authorizer=authorizer,
    )

    _asyncio.run(flow._approve(proposal, "C1"))

    recorded = [payload for name, payload in events if name == "goal.contract_recorded"]
    assert recorded, f"no contract recorded; events were {[n for n, _ in events]}"
    assert recorded[0]["goal_id"] == "prop-000000000001"
    assert recorded[0]["authorized"] is True
    assert recorded[0]["signed"] is True
    assert writer.publish_ai_ready is True
    assert store.verify_stored("prop-000000000001", 1, signer)
    store.close()


def test_missing_authorizer_never_publishes_ai_ready():
    from ticket_agent.intake.approval_flow import ApprovalFlow

    writer = _JiraWriterStub(_successful_write_result())
    flow = ApprovalFlow(
        resolver=object(),
        generator=object(),
        store=_ProposalStoreStub(),
        jira_writer=writer,
        slack=_SlackStub(),
    )

    asyncio.run(flow._approve(_approval_proposal(), "C1"))

    assert writer.publish_ai_ready is False


def test_signed_semantic_disagreement_is_stored_but_not_published(
    tmp_path,
    signer,
):
    from ticket_agent.goal.authorizer import ProposalGoalAuthorizer
    from ticket_agent.intake.approval_flow import ApprovalFlow

    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    writer = _JiraWriterStub(_successful_write_result())
    flow = ApprovalFlow(
        resolver=object(),
        generator=object(),
        store=_ProposalStoreStub(),
        jira_writer=writer,
        slack=_SlackStub(),
        goal_authorizer=ProposalGoalAuthorizer(
            _compiler(signer, checker=_Objecting()),
            store,
        ),
    )
    try:
        asyncio.run(flow._approve(_approval_proposal(), "C1"))

        stored = store.load_authorization("prop-000000000001", 1)
        assert writer.publish_ai_ready is False
        assert stored is not None
        assert stored.decision == "denied"
        assert any("Spanish copy" in reason for reason in stored.denial_reasons)
    finally:
        store.close()


def test_unsigned_decision_is_stored_but_not_published(tmp_path):
    from ticket_agent.goal.authorizer import ProposalGoalAuthorizer
    from ticket_agent.intake.approval_flow import ApprovalFlow

    store = SQLiteGoalContractStore(tmp_path / "contracts.sqlite3")
    writer = _JiraWriterStub(_successful_write_result())
    flow = ApprovalFlow(
        resolver=object(),
        generator=object(),
        store=_ProposalStoreStub(),
        jira_writer=writer,
        slack=_SlackStub(),
        goal_authorizer=ProposalGoalAuthorizer(_compiler(NullSigner()), store),
    )
    try:
        asyncio.run(flow._approve(_approval_proposal(), "C1"))

        stored = store.load_authorization("prop-000000000001", 1)
        assert writer.publish_ai_ready is False
        assert stored is not None
        assert stored.decision == "denied"
        assert stored.contract_signature is None
    finally:
        store.close()


def test_authorizer_uses_the_verbatim_request_not_the_summary(signer):
    """The whole value of the semantic check is the original wording."""

    from ticket_agent.goal.authorizer import _verbatim_request
    from ticket_agent.domain.intake import IntakeMode, Proposal, ProposalStatus
    from datetime import datetime, timezone

    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    base = dict(
        proposal_id="prop-000000000001",
        slack_user_id="U1",
        slack_thread_ts="1.0",
        mode=IntakeMode.NEW_FEATURE,
        title="t",
        summary="A model-written summary",
        status=ProposalStatus.AWAITING_CONFIRMATION,
        created_at=now,
        expires_at=now,
    )

    verbatim = Proposal(**base, original_request="the user's own words")
    assert _verbatim_request(verbatim) == "the user's own words"

    # Older proposals have no verbatim text. The fallback is a summary, and it
    # must announce itself rather than pass as the real thing.
    legacy = Proposal(**base)
    assert "verbatim request unavailable" in _verbatim_request(legacy)


class _ProposalStoreStub:
    def mark_status(self, proposal_id, status):
        return None


class _JiraWriterStub:
    def __init__(self, result):
        self._result = result

    async def write(
        self,
        proposal,
        *,
        publish_ai_ready=False,
        autonomy_mode=None,
    ):
        del autonomy_mode
        self.publish_ai_ready = publish_ai_ready
        return self._result


class _SlackStub:
    async def post_thread_reply(self, channel, thread_ts, user_id, text):
        return None


def _approval_proposal():
    from ticket_agent.domain.intake import (
        IntakeMode,
        Proposal,
        ProposalStatus,
        TicketSpec,
    )

    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    return Proposal(
        proposal_id="prop-000000000001",
        slack_user_id="U1",
        slack_channel="C1",
        slack_thread_ts="1.0",
        mode=IntakeMode.NEW_FEATURE,
        project_key="LAB",
        title="Landing page",
        summary="Ship a Spanish landing page",
        original_request="Build a landing page in Spanish",
        tickets=[
            TicketSpec(
                summary="Page",
                repository="ofertas-sv",
                acceptance_criteria=["Page renders", "Copy is in Spanish"],
            )
        ],
        status=ProposalStatus.AWAITING_CONFIRMATION,
        created_at=now,
        expires_at=now,
    )


def _successful_write_result():
    from ticket_agent.intake.jira_writer import JiraWriteResult

    return JiraWriteResult(
        project_key="LAB",
        created_ticket_keys=("LAB-31",),
    )


# -- startup reporting -----------------------------------------------------


def test_startup_reports_when_nobody_can_authorize(tmp_path):
    from ticket_agent.app import RuntimeConfig
    from ticket_agent.runtime_smoke import _goal_authorization_check

    check = _goal_authorization_check(
        RuntimeConfig(data_dir=tmp_path, risk_policy_path=tmp_path / "absent.yaml")
    )

    assert check.status == "warn"
    assert "nobody may authorize" in check.detail
    assert "cannot be signed" in check.detail


def test_startup_reports_a_usable_configuration(tmp_path):
    from ticket_agent.app import RuntimeConfig
    from ticket_agent.runtime_smoke import _goal_authorization_check

    key = tmp_path / "key"
    key.write_bytes(generate_key())
    key.chmod(0o600)
    policy = tmp_path / "risk.yaml"
    policy.write_text("version: 7\nrepositories: ['demo']\n")

    check = _goal_authorization_check(
        RuntimeConfig(
            data_dir=tmp_path / "data",
            risk_policy_path=policy,
            signing_key_path=key,
            goal_allowlist_users=("U1",),
        )
    )

    assert check.status == "pass"
    assert "policy v7" in check.detail


def test_startup_fails_loudly_on_an_unusable_key(tmp_path):
    from ticket_agent.app import RuntimeConfig
    from ticket_agent.runtime_smoke import _goal_authorization_check

    key = tmp_path / "key"
    key.write_bytes(generate_key())
    key.chmod(0o644)

    check = _goal_authorization_check(
        RuntimeConfig(
            data_dir=tmp_path / "data",
            signing_key_path=key,
            goal_allowlist_users=("U1",),
        )
    )

    assert check.status == "fail"

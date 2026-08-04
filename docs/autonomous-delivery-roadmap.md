# Autonomous delivery roadmap (P6–P18)

**This document is the canonical roadmap for `agent-system`.** It is tracked in
the repository, so a fresh clone contains everything needed to understand and
implement P6–P18 with no external files.

Authority chain:

| Document | Authority over |
|---|---|
| `docs/autonomous-delivery-roadmap.md` (this file) | The detailed roadmap: phase specs, status, order, exit criteria |
| [`../CLAUDE.md`](../CLAUDE.md) | Codebase operating rules and a concise current-phase summary |
| [`../README.md`](../README.md) | User-facing overview and roadmap summary |

The two summaries link here. If a phase's status appears to differ between
documents, this file is correct and the other should be updated.

---

## Current architecture

**This is what the runtime does today.** It is a bounded, human-reviewed
ticket-processing system — *not* unattended autonomous delivery.

```txt
Slack request
  → model-assisted proposal
  → human intake approval
  → authorized signed GoalContract + durable goal spine
  → Jira epic/tasks as the goal projection
  → ai-ready detection
  → authorization/sandbox preflight
  → SQLite ticket lock/checkpoint + action journal
  → LangGraph planning
  → execution approval (automatic or Slack, depending on policy)
  → implementation
  → worktree tests
  → summary-based model review
  → ── candidate-evidence ceiling: unreachable in production ──
  → commit/push
  → pull request
  → Jira/Slack reporting
```

**The last three steps are code paths, not current behavior.** The Step 0.5
candidate-evidence ceiling caps effective autonomy at `implement` until isolated
verification (P14), independent complete-diff review (P15), and
candidate-authorization enforcement are all live. `open_pull_request` and
`pr_create` require `deliver` (`goal/autonomy.py:53-56`), and
`CANDIDATE_EVIDENCE_CAPABILITIES_LIVE = False` (`app.py:136`, consumed at
`:424`) is a code constant no configuration reaches. A production run today
stops after review. The delivery tail becomes reachable when
`CandidateAuthorization` lands — see the near-term plan below.

The execution-approval node always runs. Whether it requires a human depends on
which `ApprovalService` is wired: the default `AutoApprovalService`
(`orchestrator/local_services.py:492`) approves unconditionally, because a
human already approved the plan at intake; `JiraLabelApprovalService` and the
Slack-driven interrupt require an explicit action.

Facts that follow from the code as it stands:

- **The authorized `GoalContract` is scope authority; Jira is its work-queue
  projection.** Detection still reads Jira and the orchestrator still works
  from Jira tickets, but the shared preflight refuses missing, invalid,
  revoked, out-of-scope, or insufficient-autonomy goal authority before a
  lock, claim, worktree, or resume mutation.
- **Goal identity and action recovery are durable.** `goal_id` reaches active
  graph state, transcripts, funnel events, and loop-iteration telemetry. The
  SQLite action journal reserves budgets before concrete side effects and
  applies operation-specific probe and ambiguity policy on recovery.
- **Verification is not bound to the committed SHA.** Tests run in the
  worktree during `run_tests`; the commit happens later, inside
  `PullRequestService.open_pull_request` (`orchestrator/git_services.py:65-77`).
- **Review does not consume the complete real diff.**
  `ModelRouterReviewService.review` (`orchestrator/model_services.py:825-860`)
  sends `state.summary` and `state.implementation_result`, both written by the
  implementing model.
- **Production contract commands are sandboxed before mutation.** P13.3a wires
  one enforcing preflight through the runner, Jira and feedback coordinators,
  approval/restart resume, worktree creation, and both contract-test paths.
  Each real wrapper launch emits a `SandboxAttestation`. P13 remains partial
  because context assembly / the knowledge map is still open.

**The system never merges pull requests automatically.** The runtime opens
PRs; it does not merge them. Any review or merge is performed by a human, as
required by current policy.

Read that as a behavioral boundary, not as operational evidence. It says what
the code will and will not do; it is not a claim that every PR the system has
generated was in fact reviewed and merged. See "Operational validation" under
[Known limitations](#known-limitations-and-configuration-debt).

## Remaining target architecture (P14–P18)

The authorized contract, spine, journal, and Jira projection at the start of
this flow are current foundations. The remaining P14–P18 steps begin at the
candidate commit and must not be described as active until their phases reach
✅.

```txt
Slack
  → authorized signed GoalContract
  → durable goal spine / action journal
  → Jira projection
  → implementation
  → candidate commit
  → isolated verification of that SHA
  → independent complete-diff review
  → attestation / outbox
  → trusted CI / integration delivery
  → human-reviewed promotion PR
  → bounded review-comment rework
```

The shape of the change is that **evidence replaces assertion**. Today a model
says "I implemented it and tests pass" and the pipeline believes it. In the
target, only an artifact — a real diff, a real test run at a real commit —
advances the work.

### Ideas the target rests on

**A claim is not a fact.** This is why the reviewer gets `git diff` and why
gates run against a committed SHA.

**Green is not done.** A passing command proves one command passed. A goal is
achieved when its acceptance criteria are met, its non-goals respected, the
integration still works, and a demo shows the user-visible behavior.

**Absence of evidence is not evidence.** A gate that never ran is not a pass,
and is also not "we chose to skip it." Three facts, three statuses
(`not_run`, `skipped`, `not_runnable`), any of which denies authorization.

**"Can this merge?" and "is the goal done?" are different questions.** Ticket 1
of 5 should merge while the goal is nowhere near finished. One gate answering
both either blocks every intermediate merge or ships an unfinished goal — hence
`CandidateAuthorization` (per SHA) and `GoalAchievement` (per contract).

**The thing doing the work cannot also check it.** Implementer and reviewer
must be different models, and that must hold when the reviewer's provider is
down and the router falls back.

**Failures come in kinds, and the kind decides what to do.** A type error means
retry the code. A registry outage means retry the *verification*, with no new
commit and no model call. A config violation means stop and get a human.

### Vocabulary

| Term | Means |
|---|---|
| **GoalContract** | What the human authorized: objective, acceptance criteria, non-goals, scope, budgets. Immutable once agreed. Agents plan freely *underneath* it and may not widen it. |
| **CandidateAuthorization** | "May this specific commit merge?" Scoped to one SHA. |
| **GoalAchievement** | "Is the objective met?" Scoped to the whole contract. Only this can mark a goal done. |
| **Trust root** | The files and config that decide how verification happens — test commands, CI workflows, gate config. The agent may never change these to make its own work pass. |
| **Harness readiness** | Whether a target repository has described itself well enough to be worked on unattended. Caps how much autonomy it gets. |

### Autonomy ladder

Five levels, each including those below:

| Level | May |
|---|---|
| `observe` | observation only; no external effects |
| `propose` | planning and Jira/Slack proposal effects; no code edits |
| `implement` | edits and verification; no pull request creation |
| `deliver` | pull request creation; no later autonomous delivery action |
| `autonomous` | later in-policy autonomous actions as their executors land |

The level in force is the **lowest** of six ceilings:

```txt
effective = min(
    configured,        # env var, default "observe"
    risk class,        # from the goal contract
    harness readiness, # how well the target repo describes itself
    sandbox,           # no isolation -> propose
    gate enforcement,  # any gate not enforcing -> implement
    halt,              # halted -> observe
)
```

Everything can only *lower* it, and an unrecognized value yields `observe`, so
failures make the system more cautious rather than less. The production shared
preflight resolves and persists this decision before execution, and concrete
action boundaries recheck the latest durable ceiling as defense in depth.

---

## Legend and maturity definitions

| Mark | Meaning |
|---|---|
| ✅ | **Complete** — implemented, wired into the production path, enforced, and tested |
| 🔶 | **Partial** — meaningful components exist, but the end-to-end invariant is not enforced |
| ⬜ | **Not started** |
| ⏸️ | **Deferred by dependency** — blocked on a prerequisite, not on effort |
| ↪ | **Absorbed** — requirements moved into another phase and are no longer scheduled separately |

Three maturity levels are routinely confused, and the distinction is the whole
point of the status table:

1. **Designed / schema landed** — types, tables, and helper classes exist.
2. **Runtime wired and enforced** — the production path constructs it, and the
   invariant it protects cannot be bypassed.
3. **Operationally validated** — it has run on real work and the evidence was
   reviewed.

**A capability is never ✅ because its types exist.** Level 1 alone is 🔶.
Goal authority and P13.3a sandboxing are runtime-enforced, but their parent
phases remain partial until P12's non-convergence/live-recovery evidence and
P13's context-assembly / knowledge-map work land.

## Phase status

Phase numbers are **stable identities, not execution order.** They are cited by
commits, tests, and prior discussion, so they are never renumbered or reused.

| Phase | State | Reality |
|---|---|---|
| P0–P7 | ✅ | Bounded ticket-processing MVP: infrastructure, Slack intake, detection, SQLite locking, adapters, ModelRouter, MVP wiring, implementation loop, acceptance criteria |
| P8 | ⏸️ | Bug-fix work profile. Deferred until P15/P16 provide trustworthy feedback |
| P9 | ↪ | Diff-based review + lint gate. **Absorbed into P14/P15**; not scheduled separately |
| P10 | 🔶 | Observability foundations exist; later producers and operational evidence are missing |
| P11 | ✅ | Agent-system owns selector config; the library copy is documented example-only and resolution is pinned by test |
| P12 | 🔶 | Durable authority, revocation, autonomy, and the action spine are enforced; non-convergence logic and live recovery evidence remain |
| P13 | 🔶 | Sandbox preflight and per-command evidence are enforced; context assembly / knowledge map remains |
| P14 | ⬜ | Immutable-SHA verification |
| P15 | ⬜ | Independent complete-diff review |
| P16 | ⬜ | Repeatable evaluation and demonstration evidence |
| P17 | ⬜ | Durable delivery, outbox, attestation, trusted CI, promotion |
| P18 | ⬜ | Bounded rework from promotion-PR change requests |

**P0–P7 are implemented and unit-tested.** They are not claimed to be
operationally validated: external Slack/Jira/GitHub end-to-end verification
remains manual, and no recorded live-run evidence is held in the repository.

## Recommended execution order

Dependency-driven, and deliberately not numeric.

```txt
P11 → P13 → P12 → P14 → P15 → P16 → P17 → P18 → revisit P8
```

| # | Phase | Why here |
|---|---|---|
| 1 | **P11** — configuration ownership | Cheapest, and everything downstream reads selector and policy config. Two copies that can drift make every later digest suspect. |
| 2 | **P13** — sandbox runtime enforcement | Runtime sandbox enforcement must precede unattended execution, so every later phase rests on an active isolation boundary. |
| 3 | **P12** — durable spine + propagation | Nothing can be resumed, budgeted, or attributed to a goal until `goal_id` reaches graph state and the journal exists. P14–P17 all record evidence *against a goal*. |
| 4 | **P14** — verify an immutable SHA | Evidence is meaningless until it binds to the artifact that will ship. |
| 5 | **P15** — independent complete-diff review | Needs P14's committed SHA to diff against, and P12's spine to record findings. |
| 6 | **P16** — evaluation + demo evidence | Needs P14 and P15 to have something trustworthy to measure. |
| 7 | **P17** — durable delivery | Merging requires every prior gate to be real. Last by construction. |
| 8 | **P18** — bounded rework | Acts on comments left on the promotion PR that P17 opens. |
| 9 | **P8** — revisit strategy work | Only once P15/P16 can tell whether a strategy helped. |

P11, P13's sandbox work, and P12's authority work are done. For what to do next
— including one milestone that is not a phase — see
[Near-term plan (M1–M4)](#near-term-plan-m1m4).

---

## Cross-phase invariants

These bind more than one phase, so they are stated once here rather than
duplicated into each phase specification. **Every invariant is added here before
the step that implements it**, and this section is authoritative when an
implementation plan disagrees with it.

### Engineering baseline

The digests every later phase binds — contract, policy, harness, trust root —
are only as trustworthy as the toolchain that computed them. That toolchain is
therefore itself an invariant, not housekeeping.

- Third-party and first-party Git dependencies are **pinned to a commit SHA
  reachable from the dependency's default branch**. A pin to an unpushed commit
  reproduces nothing and must fail CI.
- A dependency lock file exists, and a clean clone reproduces the environment
  that produced any recorded digest.
- Lint and type checking are enforced at **zero accepted violations** — no
  baseline file, no suppression list. Individual `# noqa` is permitted only with
  a stated reason, and an unused one is itself a violation.
- **Runtime readiness verifies that every declared gate command is executable**,
  not merely declared. A gate naming a tool that is absent makes readiness
  *unready*; it must never be reported as passing or silently skipped.
- Declaring an empty or unexecutable gate to obtain a gate is worse than
  declaring none: it reports `passed` for work nobody did.

### Autonomy ceiling

- Effective autonomy is capped at `implement` while **candidate evidence is not
  demonstrably available** — that is, until isolated verification (P14),
  independent complete-diff review (P15), and candidate-authorization
  enforcement are all live. The cap is derived, not configured.
- **Configuration alone can never raise effective autonomy** to push or open a
  PR. A repository contract that happens to declare few required gates must not
  widen authority as a side effect.
- Autonomy decisions carry a schema/capability version. The action guard
  **rejects or caps decisions that predate the candidate-evidence source**, so a
  decision persisted before an upgrade cannot retain `deliver` after it.
- The ceiling is re-evaluated **inside the concrete Git effects** — immediately
  before push and before PR creation — not only at graph routing. A direct
  service call must be as protected as a routed one.

> **A gate promoted to `required` may cap autonomy as a side effect, because
> `required_gates ⊆ enforced_gate_names` stops holding. That is an accident of
> configuration, not a safeguard.** It must never be treated as a substitute for
> the derived ceiling above. `config/repos/lab.yaml` is capped today only by this
> accident.

### Candidate authorization

A construction-and-validation service that **fails closed** unless all of:

- Repository and goal identities match the contract under which work began.
- Head SHA and tree OID match the actual commit; base and merge-base OIDs match
  the review record.
- Verification and review name the **same** candidate SHA.
- The full-diff digest matches the diff that was actually reviewed.
- Contract, harness, policy, and trust-root digests are **non-empty and agree**
  across current policy, the verification record, the review record, and the
  authorization itself.
- Expected chunk IDs and digests exactly equal the reviewed ones.

Further:

- Deterministic checks are recorded as **typed evidence records, never aggregate
  booleans**. Missing, stale, malformed, mismatched, or legacy evidence denies.
- `TicketState` carries record IDs and digests only; **delivery reloads the
  authorization from durable storage** rather than trusting graph state.
- Authorization is persisted **append-only**, with explicit expiration and
  invalidation rules. Modifying any bound field invalidates it.
- **Both `git_push` and `pr_create` are guarded immediately before their
  external effect**, pushing an explicit candidate-SHA refspec and reading the
  remote head back.
- A denial records **every** failed condition, not the first.

Scope note: candidate authorization answers whether an *intermediate* candidate
may be delivered as a PR. It is not goal achievement — a goal may remain
incomplete while a candidate is authorized.

### Schema and evidence durability

- Every SQLite database carries an **explicit schema version**. Migrations are
  forward-only; an **unknown future version fails closed** rather than opening
  the file.
- A backup is taken before migrating, and an interrupted migration leaves the
  database recoverable.
- WAL, busy-timeout, foreign-key, and transaction settings are applied
  uniformly across every database.
- Every table is classified as **mutable state or immutable evidence**.
  Evidence tables are append-only and reject update and delete.
- Success must never overwrite failure: **every attempt stays queryable**. One
  mutable row per action cannot express attempt history.
- Signed artifacts stay deserializable and signature-verifiable under their
  original canonical form across schema transitions. A legacy or missing field
  is surfaced explicitly — **never silently upgraded**.

---

## Near-term plan (M1–M4)

The phase order above is dependency-driven and unchanged. This section is the
**near-term execution plan layered on top of it**: what to do next, in what
order, verified against the code on 2026-08-04. Where a milestone maps onto an
existing phase or milestone-plan step, it says so rather than restating the
specification.

### Verified state on 2026-08-04 (after M1)

| Check | Result |
|---|---|
| `pytest tests/unit/ -q` | 913 passed |
| `ruff check .` | clean — `scripts/` now in scope |
| `mypy` (strict, `files = ["src", "scripts"]`) | clean |
| `ai-model-selector` pin | `c9ab8e3`, reachable from `origin/main`, and asserted to agree between `pyproject.toml` and `uv.lock` |
| Dependency lock file | `uv.lock`, CI syncs with `--frozen` |
| `.github/workflows/ci.yml` | tests + coverage floor, ruff, mypy, smoke, packaging, pinned-SHA check |
| SQLite schema versioning | none — no `PRAGMA user_version` in `src/` |
| Graph topology | `plan → approval → implement → run_tests → review → open_pull_request → escalate → report` (`orchestrator/graph.py`) — no `commit`, `verify`, or `authorize_candidate` node |
| Host prerequisites | `bwrap` 0.9.0, `gh`, and all Slack/Jira/DeepSeek/Gemini/GitHub credentials present |
| Runtime databases | **all empty** — see "Operational validation" below |

### M1 — Engineering baseline (Milestone B step 0e) — ✅ complete

Every digest the later phases bind is only as trustworthy as the toolchain that
computed it, which is why this is an invariant rather than housekeeping — see
[Engineering baseline](#engineering-baseline).

1. ✅ **Dependency lock file** — `uv.lock`. uv rather than pip-tools because
   `pip-compile --generate-hashes` refuses VCS dependencies, which would have
   left the selector pin — the one that matters most — unlocked.
2. ✅ **mypy to zero**, 98 → 0, across `src/` and `scripts/`.
3. ✅ **`scripts/` into the lint scope** — was 22 findings checked by nothing.
4. ✅ **CI** under `.github/workflows/`.
5. ✅ **Readiness verifies gate executability** — a required gate with no
   runtime executor is now a blocking readiness reason, so `lab.yaml` reports
   `unready` and says why.

**`tests/` is not in the typecheck scope.** It needs a packaging decision first
— no `__init__.py`, so mypy resolves `constants.py` under two module names —
and behind that sit 1106 errors, 907 of them bare `no-untyped-def` on test
functions. Whether test bodies carry annotations is its own call, not part of
taking `src/` to zero.

What the sweep found, beyond the annotations — these were latent defects, not
style:

- **The test suite only passed under one invocation.** Tests import
  `tests.constants` and `scripts.system_reset`, neither an installed package.
  `python -m pytest` puts the working directory on `sys.path`; bare `pytest`
  does not. CI used the second and seven modules failed to import. Fixed by
  declaring `pythonpath = ["src", "."]`.
- **A default shell factory that could never be called.**
  `_build_contract_shell` was the default for `ShellFactory` in three places
  while requiring a keyword-only `sandbox`, so reaching it raised `TypeError` —
  which the test-runner boundary does not catch, so it would have escaped as a
  crash. Nothing reached it because the tests that omit the factory all fail
  earlier.
- **The preflight protocol did not describe what it returns.** Six entry points
  took `ExecutionPreflight` (returns `Sandbox`) while being handed
  `ExecutionAuthorizationPreflight` (returns sandbox *plus* the autonomy
  decision), and the runner reached the extra field through `getattr(...,
  None)`. A test stub returned a bare `object()`, so `autonomy_mode` and
  `autonomy_decision_digest` recorded as absent without complaint.
- **`AuthorizedExecutionContext.wrap` was dead and wrong** — uncalled, and
  passing a `CommandExecutionPolicy` where `Sandbox.wrap` needs a
  `SandboxPolicy`. Removed.
- **Duplicate protocols that had drifted**: three `ModelRouterProtocol`, two
  `SlackPoster`, two `TicketNode`. Each pair was interchangeable only while
  nothing compared them.
- **Unvalidated SQL identifiers** in `scripts/system_reset.py`, safe by
  assumption rather than by check.

Promoting `agent-system.yaml`'s `lint` gate to `required` and adding `typecheck`
is now possible — but doing so lowers autonomy as a side effect of
configuration, which is the fragility the derived ceiling exists to replace. It
is not a substitute for it. See [Autonomy ceiling](#autonomy-ceiling).

### M2 — First operationally validated run

**Rationale.** No phase in this document has ever reached maturity level 3. The
runtime databases are empty, so every capability marked ✅ rests on unit tests
alone. Building P14–P16 first would stack four more phases of evidence machinery
on a loop whose evidence is entirely synthetic. Every host prerequisite is
already installed; the only reason no run exists is that none has been
attempted.

This pulls Milestone B step 0d's scratch repository forward. The scratch target
is app-agnostic by construction — satisfying the rule that nothing in `src/` may
be shaped around a particular target — and serves as the hard-kill fixture later.

- **Scratch target repository**: a real Git repository with a contract under
  `config/repos/`, real `test` and `lint` gates, declared `writable_paths`, and
  a trust root. Not `ofertas-sv`; not `agent-system` itself.
- **Drive one ticket end to end** on the live path: Slack request → proposal →
  intake approval → `GoalContract` + spine → Jira epic/tasks → detection →
  preflight → lock → plan → implement → `run_tests` → review.
- **The run stops before delivery, by design.** The candidate-evidence ceiling
  blocks `open_pull_request`. Inspect the produced worktree diff by hand.
  Reaching a pull request is M4's `CandidateAuthorization` step, not this one.
- **Enable transcripts** (`AGENT_SYSTEM_TRANSCRIPTS_ENABLED=true`) and review
  one. The sink existing demonstrates nothing.
- **Deliverable — a committed run report**: funnel rows, spine and journal rows,
  model attempts and cost, where the run needed help, and what was surprising.
  This is the project's first level-3 evidence.

Expect defects that 908 unit tests could not surface. That is the reason this
milestone precedes P14 rather than following the hard-kill validation.

### M3 — Schema and evidence durability (Milestone B step 0b)

Specified under [Schema and evidence durability](#schema-and-evidence-durability)
and in the P14/P15 sections; sequenced here, after M2, so there is real data to
migrate rather than empty files.

- Version the eight existing databases: `locking/checkpointer.py`,
  `feedback/github.py`, `orchestrator/execution_approval.py`,
  `goal/contract.py`, `goal/spine.py`, `intake/proposal_store.py`,
  `locking/sqlite_store.py`, `observability/telemetry.py`.
- Add the new evidence tables and the content-addressed evidence store.
- Fix the two identity defects while the schema is open: `action_records`'
  `UNIQUE (goal_id, operation, natural_key)` (`goal/spine.py:76`) collides on an
  A → B → A rework, and one mutable row per action (`spine.py:53`) cannot
  preserve attempt history.

### M4 — Milestone B as specified

`P14 → P12 closure → P15 → CandidateAuthorization → P16 → supervised hard-kill
validation`. Specifications and exit criteria are unchanged and live in the
phase sections below. `deliver` first becomes reachable at
`CandidateAuthorization`; `autonomous` remains P17's to unlock.

### Deferred, as decisions rather than open questions

- **Greenfield repository creation** — deferred past Milestone B. See
  [Greenfield behavior](#greenfield-behavior).
- **P8** — unchanged; waits on P15/P16 feedback.
- **P13 remainder** — context assembly / knowledge map, still unscheduled.
- **`lab.yaml` naming** and **`risk.yaml` shipping one operator's allowlist** —
  real debt, cosmetic priority; fold into M1 if convenient.

---

## Detailed phase specifications

Contiguous and in numeric order for lookup. **This is not the order to work
them in** — see "Recommended execution order" above.

### P6 — Implementation loop upgrade

Goal: `IterativeImplementationService` in
`src/ticket_agent/orchestrator/model_services.py` currently exposes only
`read_file`, `list_dir`, `write_file`, `finish`. That caps success at small
changes in small repos. Extend the JSON tool-call contract to:

```txt
read_file:  {path, offset?, limit?}   # offset = 1-based start line, limit = line count
list_dir:   {path}
search:     {pattern, path?, max_results?}
edit_file:  {path, old_string, new_string, replace_all?}
write_file: {path, content}
run_tests:  {}
finish:     {summary, notes?}
```

Behavior rules:

- `read_file` without `offset`/`limit` keeps current behavior (truncated at
  `tool_result_max_chars`). With `offset`/`limit` it returns only those
  lines. The result must state whether the returned view is complete.
- `search` is read-only regex search over worktree text files. It must only
  visit paths the FileAdapter would allow, skip binary files, cap results
  (default 50 matches), and return `path:line:matched-line` entries,
  truncated to `tool_result_max_chars`.
- `edit_file` performs exact-string replacement. Fail with error code
  `edit_target_not_found` when `old_string` is absent, and
  `edit_target_ambiguous` when it matches more than once without
  `replace_all: true`. Successful edits append the path to `changed_files`.
- Truncated-write guard: track, per path, within one loop run, whether the
  model has ever seen a complete view of the file (untruncated read, or the
  file did not previously exist). If not, reject `write_file` for that path
  with error code `truncated_write_rejected` and instruct the model to use
  `edit_file`. This prevents silent destruction of unseen file content.
- `run_tests` executes the repo-contract test command through the existing
  TestAdapter inside the worktree. Never auto-detect the command. Budget:
  max 5 runs per implementation attempt; further calls fail with
  `test_budget_exhausted`. Return the structured TestAdapter result
  (pass/fail, truncated stdout/stderr summary, timeout flag).
- Wiring: `LocalImplementationService` (orchestrator/local_services.py)
  already prepares the worktree context; extend the context it hands to
  `implement_context(...)` with a test runner so
  `IterativeImplementationService` never constructs adapters itself.
- Update the system prompt in `_implementation_loop_messages` to document
  every action, the edit-over-rewrite preference, and the run-tests-before-
  finish convention.

P6 checklist:

- [x] `search` action with boundary, binary-skip, and cap tests
- [x] `edit_file` action with not-found / ambiguous / replace_all tests
- [x] paginated `read_file` with complete-view flag
- [x] truncated-write guard with negative test
- [x] `run_tests` action via TestAdapter with budget test
- [x] prompt update in `_implementation_loop_messages`
- [x] no shell/git actions exposed to the model (negative test)

P6 landed. Notes for later phases: the model-callable `run_tests` action is
wired through `ImplementationContext.test_runner`, built by
`_make_contract_test_runner` in `orchestrator/local_services.py` (contract
test command via `LocalTestAdapter` only). The per-run loop state lives in
`_LoopContext` in `orchestrator/model_services.py`; `complete_view_paths`
tracks which files may be safely rewritten. P6 tests are in
`tests/unit/test_model_services_p6_actions.py`.

### P7 — Acceptance criteria pipeline

Goal: make testable acceptance criteria flow from intake through planning to
review, so review has something falsifiable to check.

- ProposalGenerator (`intake/proposal_generator.py`): each proposed ticket
  gains `acceptance_criteria: list[str]` (1–7 short, testable statements).
  Write them into the Jira description under an `Acceptance Criteria`
  heading via JiraWriter.
- Clarifying questions: when the intake model judges a request too ambiguous
  to propose, it may return a clarifying-question payload instead of a
  proposal. The Slack listener posts the question in the intake thread and
  treats the human reply as a revision input (reuse the existing proposal
  revision flow in `intake/approval_flow.py`). One clarifying round max,
  then propose with stated assumptions.
- Planning: `_planning_messages` includes the criteria; the plan payload
  must include a `criteria_coverage` mapping of criterion → planned step.
- Review: `_review_messages` includes the criteria; the review payload must
  include a per-criterion verdict (`met: bool` plus one-line evidence). Any
  unmet criterion routes to `rejected`.

P7 checklist:

- [x] proposal schema + Jira description rendering
- [x] clarifying-question round in Slack intake (single round)
- [x] plan prompt + `criteria_coverage` validation
- [x] review per-criterion verdicts drive accept/reject routing

P7 landed. Notes for later phases: the acceptance-criteria format lives in
`domain/acceptance.py` (`render_acceptance_criteria` / `parse_acceptance_criteria`
are inverses). Criteria are the source of truth in the Jira description under
the `Acceptance Criteria:` heading — there is no Jira custom field for them;
the planner and reviewer re-parse them from `state.description`. The planner
emits `criteria_coverage` and the reviewer emits `criteria_verdicts`; any
verdict with `met` != true forces the review to `rejected`. A rejected review
routes back to IMPLEMENT (rework) while `implementation_attempts` <
`max_attempts`, with the rejection reasoning/issues injected into the retry
prompt as `previous_review_rejection`; once attempts are exhausted it
escalates. The single-round
clarification stores a 0-ticket DRAFTING placeholder (see
`_clarification_placeholder`) so the user's answer returns through `_revise`
with clarification disabled. Foundation-first ordering is instructed in the
proposal prompt; the first ticket establishes the shared scaffold/types/schema
and later tickets receive sibling scopes as coordination context. P7 tests are
in `tests/unit/test_p7_acceptance_criteria.py`.

### P8 — Bug-fix work profile

Goal: bugs get a reproduce-first workflow instead of the generic feature
flow.

- Profile detection: Jira issue type `Bug` or label `bug` sets
  `work_profile: "bug"` on `TicketState` (default `"feature"`).
- Bug planning: the plan must include a reproduction section — expected
  behavior, actual behavior, and where the regression test will live.
- Bug implementation prompt: write the failing test first, call `run_tests`
  to confirm it fails for the expected reason, then fix, then `run_tests`
  green. The `finish` summary must name the regression test.
- Review for bugs additionally checks that a regression test exists in
  `changed_files`.
- Out of scope for P8: auto-filing bugs from CI or PR feedback (deferred).

P8 checklist:

- [ ] `work_profile` on TicketState + detection from Jira fields
- [ ] bug-specific plan and implementation prompts
- [ ] review requires regression test for bug profile
- [ ] tests for profile routing and prompt selection

### P9 — Diff-based review + lint gate

> **↪ Absorbed — do not schedule this phase.** Its two requirements are
> delivered by **P15** (real-diff review, plus git-derived file lists and
> maker ≠ checker) and **P14** (lint as a first-class gate with a `GateStatus`
> and failure class). Building it separately would mean writing a diff
> reviewer twice: once against the worktree, once against a committed SHA.
> The design notes below are retained because P14/P15 inherit them.
>
> P9 is *absorbed*, not merely deferred: there is no future point at which it
> becomes schedulable on its own. `LocalTestAdapter.run_lint()` exists at
> `adapters/local/test_adapter.py:39` with zero callers, which is the honest
> marker of its status — stubbed, not active.

Goal: review judges the real diff, and lint stops style/static regressions
before a PR opens.

- GitAdapter (`adapters/local/git_adapter.py`) gains read-only
  `diff_stat()` and `diff_text(max_chars_per_file)` against the base branch
  inside the worktree. No new write operations.
- The review node input includes the real diff (per-file truncation, keep
  the `changed_files` list) instead of only model-produced summaries.
- Lint gate: after tests pass in the RUN_TESTS node, run the repo-contract
  `lint` command when the contract defines one (it is currently dead
  config). Lint failure routes exactly like test failure
  (`retry → IMPLEMENT`). Reuse the AdapterTestService pattern; keep lint
  output in state for the implementation retry prompt.

P9 checklist:

- [ ] GitAdapter diff methods with truncation tests
- [ ] review prompt consumes real diff
- [ ] lint execution from repo contract + routing test
- [ ] contract without lint command skips the gate cleanly

### P10 — Run transcripts + funnel metrics

Goal: make runs debuggable and improvement measurable.

- TranscriptRecorder: append-only JSONL per ticket run at
  `.agent-system-data/transcripts/{TICKET-KEY}-{lock8}.jsonl`. Record node
  enter/exit, every ModelRouter attempt (capability, provider, model,
  tokens, latency, success/error), and every implementation tool call and
  truncated result. All recorded content must pass through the existing
  `redaction` module. Inject the recorder through the node runner and
  router factory; default to a no-op recorder so tests stay silent.
- Funnel metrics: SQLite table `ticket_funnel` (ticket_key, claimed_at,
  implemented_at, tests_passed_at, pr_opened_at, merged_at, escalated_at,
  escalation_reason). Written from orchestrator nodes and the merged-
  delivery poller. Add `scripts/report_funnel.py` to print stage counts and
  conversion rates.

P10.1 checklist — **landed**:

- [x] TranscriptRecorder + redaction coverage test
- [x] recorder wired into node runner, router, and implementation loop
      (no-op default everywhere)
- [x] `ticket_funnel` writes at each stage that exists today —
      `claimed` from `OrchestratorRunner` (the funnel's denominator, recorded
      where the lock is already held), and `planned`/`approved`/`implemented`/
      `pr_opened`/`escalated` from `TicketNodeRunner._record_stage`.
- [x] `scripts/report_loops.py` (supersedes the planned `report_funnel.py`)

P10.2 checklist — **partial**, which is why P10 is 🔶:

- [ ] `committed`/`verified`/`reviewed`/`demoed` stages — they cannot be
      written before P14 introduces the COMMIT/VERIFY topology.
- [ ] `merged` stage — declared in `STAGES` (`telemetry.py:46`) and never
      written by anything; needs the delivery poller from P17.
- [ ] **a producer for `gate_results`.** `SQLiteTelemetryStore.record_gates`
      has zero production callers, so the table is created and never written.
      Arrives with P14's `VerificationRecord`.
- [x] a production `loop_iterations` producer at each implementation attempt,
      carrying goal id, iteration, outcome, failure fingerprint, available
      token/cost data, and wall time
- [x] canonical `goal_id` propagated into production funnel and iteration rows
- [ ] **operational measurement.** No funnel has been reviewed over a real run
      of tickets.

**Opt-in is not operational proof.** Transcripts are gated behind
`AGENT_SYSTEM_TRANSCRIPTS_ENABLED` (default **false**). A dark feature that
*can* be switched on has not demonstrated anything about how the system
behaves. Do not cite the existence of the transcript sink as evidence that
runs are observable; cite a reviewed transcript from a real run.

The zero-valued stages above are honest zeros for stages that do not exist
yet, not broken wiring — but they are also not "done".

A node running is **not** the same as its stage being reached: approval only
counts when granted, and a PR only when a URL came back. Recording on node
entry regardless would make every conversion rate read 100%.

P10.1 landed. Notes for later phases: the sink is process-wide and routed by
`ticket_key`, not a per-run object — services are built once in `app.py` and
`ModelRouter.invoke` already receives `ticket_id`, so `invoke()`'s documented
signature is unchanged. Transcripts are opt-in via
`AGENT_SYSTEM_TRANSCRIPTS_ENABLED` (default false) and every hook takes a
recorder defaulting to `NullTranscriptRecorder`.

Three rules that are load-bearing rather than stylistic:

- **Record through `safe_record`, never `recorder.record` directly.**
  `TranscriptRecorder` is a Protocol, so the never-raises guarantee cannot
  live only in `JsonlTranscriptRecorder`; any other implementation would
  otherwise propagate into the pipeline it is meant to observe.
- **Shapes and sizes, never content.** `_safe_tool_args` keeps
  `path`/`pattern`/`offset`/`limit` and reduces `content`/`old_string`/
  `new_string` to `<field>_chars`. A gate's output can be megabytes and a
  file write can be a secret.
- **`redaction.py` now filters credentials as well as paths**, and the router's
  log path is redacted too — previously only prompts were.

P10 tests are in `tests/unit/test_observability.py` and
`tests/unit/test_observability_hooks.py`.

### P11 — Configuration ownership

Scope is exactly configuration ownership: which repository owns each selector
config file, and preventing the two copies from drifting. It does **not**
include CI, dependency locking, Ruff installation, or test-workflow fixes.

Goal: stop `capabilities.yaml` / `models.yaml` / `task_profiles.yaml` from
drifting between this repo and `ai-model-selector`.

- This repo's `config/` is canonical. Verify
  `router/selector_config.py` always loads selector config from explicit
  paths in this repo — never from the `ai-model-selector` package's bundled
  `config/`.
- In the `ai-model-selector` repo, mark its `config/` as examples only
  (README note). No selector code changes.

**P11 is complete.**
`selector_config.py:17-20` resolves `CONFIG_DIR` to this repo explicitly:

```python
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
```

and passes `CAPABILITIES_PATH` / `MODELS_PATH` / `TASK_PROFILES_PATH` into
`load_capability_definitions` and `DeterministicSelector.from_yaml`. Nothing
reads the package's bundled config. That is the load-bearing half.

`tests/unit/test_selector_config.py` pins all three resolved paths to this
repository and proves none sits beneath the installed selector package. The
`ai-model-selector` README labels its three bundled files as examples and
directs applications to pass their own explicit paths.

P11 checklist:

- [x] selector config paths resolved from this repo, never from the package
- [x] a test pinning that resolution, so a refactor cannot silently
      re-point `CONFIG_DIR` at the installed package
- [x] `ai-model-selector` README marks its bundled config example-only
- [x] a documented statement of which repo owns each config file

Exit criteria for P11:

- Exactly one canonical location per config file, stated in both repos.
- The selector's bundled config cannot be loaded by this system, and a test
  proves it.

Canonical ownership is per file, not per loader:

| File | Canonical owner | `ai-model-selector/config/` copy |
|---|---|---|
| `capabilities.yaml` | `agent-system/config/capabilities.yaml` | Example only |
| `models.yaml` | `agent-system/config/models.yaml` | Example only |
| `task_profiles.yaml` | `agent-system/config/task_profiles.yaml` | Example only |

The selector library owns the schema and deterministic selection behavior. It
does not own this deployment's capability, model, or task-profile values.

### P12 — Goal contract + durable spine

> **🔶 Partial. P12.2 durable authority, revocation, autonomy, and the complete
> per-operation action journal are wired and unit-tested.** P12 remains partial
> until non-convergence policy lands and process-kill/live-run recovery evidence
> is reviewed against the full exit criteria.

Goal: pursue a *goal*, not a ticket queue. The schema layer landed first so
later phases import types instead of forward-referencing them.

Landed in `goal/types.py` and `orchestrator/gates.py`:

- **Phases are not terminal statuses.** A `LoopState` always carries a
  `GoalPhase`; it carries a `TerminalStatus` only when the phase is `closed`,
  and the constructor rejects any other combination. `ready_for_promotion` is
  a *phase* a run can still leave. Collapsing these into one enum is what lets
  "the criteria look met" read as "done".
- **`GateStatus` has five members, not four.** `not_runnable` (tried, could
  not run), `skipped` (policy said no), and **`not_run`** (never reached,
  routing short-circuited) are different facts. Every expected gate is seeded
  to `not_run`, so a partial record can never masquerade as complete, and
  `VerificationRecord.authorized` denies on any of them.
- **Two authorizations, deliberately separate.** `CandidateAuthorization`
  answers *may this SHA merge* and is scoped to one commit.
  `GoalAchievement` answers *is the objective met* and is scoped to the
  contract. Ticket 1 of 5 is routinely authorized while the goal is nowhere
  near achieved — requiring achievement to merge would deadlock the sequence.
- **`GoalAchievement.achieved` requires a confirmed promotion PR.** Evidence
  that the criteria are met is not the same as having presented the work.
- **Autonomy resolution is monotone and fail-closed.**
  `resolve_autonomy(...)` takes a `min` across configured mode, risk class,
  harness readiness, sandbox availability, gate enforcement, and halt state,
  so no input can ever *raise* autonomy. An unrecognized mode yields
  `observe`. A required gate below `enforce` caps at `implement` — a gate in
  shadow authorizes nothing. No sandbox caps at `propose` unless each command
  is human-approved.

#### P12 runtime status

The proposal id is now durable goal identity on every created Jira issue and
flows through `TicketWorkItem`, `TicketState`, transcripts, funnel metrics, and
iteration telemetry. Affirmative signed evidence is stored before `ai-ready`
publication. Every production execution entry point shares a preflight that
revalidates identity, signature/digests, semantic decision, current revocation,
repository scope, sandbox readiness, and effective autonomy before mutation.

The action spine and its budget reservations share one SQLite transaction.
Concrete operation policies cover Jira, Slack, worktrees, Git/PR delivery,
model calls, and gates; crash tests exercise all four ambiguity points. This is
not yet operational proof: no recorded real process-kill recovery or sustained
live goal run has been reviewed, and repeated-failure/non-convergence policy is
still absent.

P12 checklist:

- [x] schema vocabulary with canonical-JSON digests
- [x] `TicketState` `extra="forbid"` (see below) + `committing`/`verifying`
- [x] `GoalContract` compilation, signing, and intake authorization
- [x] `goal_id` on `TicketWorkItem`, populated from the canonical Jira goal
      label and threaded into `TicketState`
- [x] execution refuses to start when the goal has no authorized contract and
      autonomous execution was requested
- [x] `resolve_autonomy` consulted on the execution path and every action ceiling
- [x] durable spine: loop state and Jira publication intent share SQLite
- [x] action journal (`intended → in-flight → done`) with bounded duplicate
      spend on ambiguous recovery
- [ ] non-convergence detection over repeated failure fingerprints and strategy
      outcomes
- [ ] aggregate goal budget covering wall-clock, iterations, verification
      retries, and rework rounds — not model cost alone

#### P12 non-convergence and aggregate budget

Something has to stop the `implement → commit → verify → review → implement`
loop. Two mechanisms, and the first does not depend on the second.

**Structural detection** fires regardless of whether any budget is declared, so
a missing or generous limit can never buy an unbounded loop:

- Fingerprints are computed over the **candidate tree OID**, not a diff string.
  P14 already computes it for the commit natural key.
- Detect: an identical tree repeated · **alternation between previously seen
  trees (A → B → A)** · repeated verification failure with the same gate and
  the same classification · a review objection already addressed once · no
  material change between iterations.
- **An unchanged candidate consumes no iteration budget.**
- Non-convergence is a **deny-and-escalate terminal, never a retry.**
  Escalation carries an evidence summary naming the repeated fingerprints,
  their attempt IDs, and the budget line that tripped, if any.

**Declared limits** bound the loops that are *not* structurally detectable —
where each attempt differs genuinely but never converges:

| Limit | Value |
|---|---|
| `iterations` (implementation attempts per goal) | **5** |
| `verification_retries` (flake/transient re-runs) | **3** |
| `rework_rounds` (after human PR review comments) | **3** |

- These are **deployment defaults injected at
  `GoalAuthorizer(default_budgets=…)` (`goal/authorizer.py:51`), never constants
  in the enforcement path.** The resolver reads `GoalContract.budgets`; the
  numbers only decide what an authorized contract carries when the human did not
  override them.
- A goal may declare **tighter** limits freely. **Widening a limit is a scope
  change** and requires the same authority as any other budget change —
  otherwise an agent can grant itself more attempts.
- `authorizer.py:51` currently defaults to `Budgets()`, every field `None`, and
  `None` means unlimited. **That is a fail-open.** An unattended goal with an
  unset limit **denies at authorization** rather than resolving to unlimited.

**Aggregate budget.** `Budgets` (`goal/types.py:213-219`) already declares
`wall_seconds`, `tokens`, `cost_usd`, and `iterations`; add
`verification_retries` and `rework_rounds`. The gap is enforcement, not
vocabulary — the `budget_reservations` ledger (`goal/spine.py:80`,
`reserve_action` at `spine.py:139`) reserves and settles **model cost only**.
**Extend that ledger; do not replace it.** Concurrent actions cannot overspend
one goal budget, restart does not reset consumed budget, and exhaustion produces
`budget_exhausted` with an escalation rather than a silent stop.

Exit criteria for P12:

- **Autonomous execution is impossible without an authorized contract.** Not
  logged, not warned — refused.
- `goal_id` is present on every funnel row, transcript event, and iteration
  record produced by a real run.
- A process killed mid-run resumes from the spine, and the ambiguous model
  charge is counted pessimistically rather than lost.
- Replaying a completed journal entry performs no second side effect.

P12.1 landed. Authorization is a conjunction of three **independent** judgements
in `goal/`, and any one of them sends the request to a human:

- **`policy.py` — who and what, by rule.** Risk is decided by versioned *data*
  (`config/policy/risk.yaml`), hashed into `policy_digest` and carried on the
  contract. No model assigns a risk class: a system that lets the agent grade
  its own risk has no risk classification. `classify_request` (at intake) and
  `classify_changes` (at authorization, PR 7) deliberately share one rule set,
  because the interesting failure is when the two disagree.
- **`semantic_check.py` — meaning, by a different model.** Rules answer "is
  this in policy"; they cannot answer "is this what the person asked for". The
  checker compares the compiled contract to the **verbatim request**, on a
  provider disjoint from the compiler's, and can only *flag*, never widen. If
  the router falls back onto the compiler's own provider it fails closed.
- **`contract.py` — the allowlist.** Empty means empty. Treating "unconfigured"
  as "everyone" is how an internal tool becomes an open one.

Fail-closed throughout: an unclassifiable request, an unreachable checker, an
unparseable response, and a missing signature are four different things, and
none of them may read as approval.

**Signing (`signing.py`) is tamper-evidence, not tamper-prevention.** It makes a
forged or hand-edited row loud. On a single-host deployment the agent runs as
the operator's user and can read the key, so a compromised agent can still mint
valid signatures. Real prevention needs a separate signing principal — a
deployment change, not a code change. Do not let the presence of an HMAC imply
more than this.

`Proposal` gained `original_request` because `_original_request_from_proposal`
falls back to `proposal.summary`, which is *model-written*. Checking a compiled
contract against a model's summary of the request is the blind-reviewer defect
in a new costume.

Env: `AGENT_SYSTEM_GOAL_ALLOWLIST_USERS`, `AGENT_SYSTEM_GOAL_ALLOWLIST_CHANNELS`,
`AGENT_SYSTEM_SIGNING_KEY_PATH`, `AGENT_SYSTEM_RISK_POLICY_PATH`. All unset by
default, which authorizes nothing; `runtime_smoke` reports the posture. Contracts
are recorded before executable publication and revalidated by the shared
execution preflight. Tests: `tests/unit/test_goal_contract.py`.

**`TicketState` now sets `model_config = ConfigDict(extra="forbid")`.** Under
pydantic's default `extra="ignore"`, a node returning an undeclared field had
its update *silently dropped*: the graph advanced, the value was missing
downstream, and the ticket escalated with no explanation. Adding a node field
now requires declaring it in `state.py`, and adding a workflow status requires
extending the `WorkflowStatus` Literal.

#### P12.2 — durable execution authority and action spine

P12.2 lands in five increments. Infrastructure may land alone, but durable
publication and durable consumption are one safety boundary:

```txt
P12.2a                  spine/journal kernel + canonical goal identity
P12.2b + P12.2c         smallest safe release unit
P12.2d                  autonomy decision propagation and action ceilings
P12.2e                  expanded per-operation journal coverage and recovery
```

If P12.2b and P12.2c are not released together, both must remain behind the
same default-off flag. Controlling publication alone is insufficient while
legacy `ai-ready` tickets can still enter through an unchecked execution path.

**Canonical goal identity.** The proposal id is the goal id, used verbatim. It
has the exact form `prop-[0-9a-f]{12}`. Every created ticket carries exactly
one `ai-goal-<proposal-id>` label. A shared normalization function validates
the value at write, storage, and lookup; it never lowercases, truncates, or
repairs malformed input. Missing, duplicate, uppercase, malformed, or multiple
`ai-goal-*` labels fail closed. Epic keys and `ai-sequence-*` labels remain
display/sequencing metadata and are never identity. Migration is deliberately
fail closed: a legacy `ai-ready` ticket without one valid goal label is not
executable unattended.

**P12.2a — spine/journal kernel.** Persist canonical goal state and an
`ActionRecord` journal in one SQLite database. Action ids are deterministic;
the initial operation is `jira_write:add_ai_ready`, with states
`intended → in_flight → done`. Goal-state transition, action reservation, and
budget reservation commit in one transaction. An ambiguous readiness-label
write is recovered by re-reading labels before retry, so the effect is not
duplicated. This increment is infrastructure only: it adds no authorization
gate and makes no refusal claim.

Loop state, journal rows, and model-budget reservations share that database so
their invariants can be committed atomically. The LangGraph checkpointer stays
separate and owns node-resume payloads only. The ticket-lock database stays
separate and owns cross-worker execution leases only.

Each persisted action records its natural key, request digest, attempt count,
timestamps, lease owner and expiry, result identity, reserved and actual model
cost, error classification, and recovery classification. Idempotency and
recovery policy are selected per concrete operation, never merely per broad
`ActionKind`.

P12.2a checklist:

- [x] canonical proposal-id normalization at Jira write, contract storage, and
      lookup; canonical goal label on every created epic and ticket
- [x] atomic loop-state, action-intent, and budget reservation in one SQLite
      goal-spine transaction
- [x] deterministic `jira_write:add_ai_ready` journal entry with label-presence
      recovery after an ambiguous write
- [x] durable authorization, shared execution preflight, and autonomy ceilings
      (P12.2b–d)
- [x] remaining per-operation journal coverage and crash matrix (P12.2e)

**P12.2b — durable authorization and revocation.** Store the authorization
decision, semantic verdict, and every denial reason; denied rows are retained
for audit. A stored row authorizes nothing by its existence: consumers verify
the signature and evidence digest and recompute the semantic decision.
`ai-ready` is published only after an affirmative, signed, semantically
accepted decision. Missing, denied, unsigned, digest-mismatched, or semantic-
disagreement records never publish executable work.

Revocation is append-only and never overwrites or deletes the original
authorization. Each record carries goal id, contract version, `revoked_at`,
`revoked_by`, reason, decision/evidence digest, and a signature or explicit
trusted-authority provenance. The effective decision is the newest valid
authorization for the contract version minus any later valid revocation of
that exact evidence. Revocation is revalidated before execution and before
checkpoint resume; an old checkpoint is not permission.

The revocation authority is deliberately narrower than intake approval: only
an operator acting through the configured signing authority (or an equivalent
separately configured trusted authority) may revoke. Slack allowlist
membership by itself is not revocation authority. Unknown, unsigned, or
unverifiable provenance fails closed.

**P12.2c — shared execution preflight.** One service is consumed by every
production entry point and composes P13.3a sandbox readiness with canonical
identity, verified durable authorization, current revocation state, and
autonomy. It runs before every execution mutation. Jira execution, feedback
execution, Slack approval resume, and restart/checkpoint resume all use it.
Resume reacquires or explicitly adopts the execution lock, then revalidates
authorization, revocation, autonomy, and sandbox readiness; it no longer calls
`graph.ainvoke` as an unguarded shortcut. Action-boundary checks remain as
defense in depth.

P12.2b+c checklist:

- [x] persist signed authorization evidence, decision, semantic verdict, and
      denial reasons; retain denied and legacy rows as non-authorizing audit
- [x] publish `ai-ready` only after affirmative evidence is durably stored
- [x] append separately signed operator revocations without altering the
      authorization row; calculate the latest effective decision fail closed
- [x] shared preflight verifies identity, contract/evidence digests and
      signatures, current revocation, repository scope, autonomy floor, and
      sandbox readiness
- [x] Jira, feedback, approval resume, runner, local implementation, and
      restart resume consume the shared guard before execution mutation
- [x] approval resume loads current Jira identity and returns through the
      lock-owning runner with heartbeat instead of invoking LangGraph directly
- [x] persisted `AutonomyDecision` and named action ceilings (P12.2d)
- [x] remaining per-operation journal coverage and crash matrix (P12.2e)

**P12.2d — autonomy decisions and ceilings.** Each goal persists an
`AutonomyDecision` containing effective mode and every binding ceiling, with a
named source for every `resolve_autonomy` input. `AGENT_SYSTEM_AUTONOMY_MODE`
provides the configured ceiling; an unrecognized mode becomes `observe`.
Action boundaries enforce: `observe` permits no effects; `propose` permits
planning/proposal effects only; `implement` permits edits and verification but
no PR; `deliver` permits PR creation but no later autonomous delivery action;
`autonomous` permits the later in-policy autonomous actions.

Gate enforcement is derived, not asserted by configuration:
`required_gates ⊆ enforced_gate_names`, and a gate enters
`enforced_gate_names` only when a production executor is wired and its failure
blocks the action. Configuration may select off/shadow/enforce but cannot
claim enforcement without that executor. Today only `test` has a wired
executor; therefore the derived ceiling cannot exceed `implement` until P14
wires every required gate. This honest cap is preferable to narrowing the
contract to match incomplete execution coverage.

P12.2d checklist:

- [x] parse the environment ceiling fail closed and persist the effective mode,
      every named input ceiling, binding sources, and executor-derived gate set
- [x] recompute autonomy in the shared execution preflight and propagate the
      decision digest and effective mode into ticket state
- [x] enforce planning/proposal, edit/verification, PR creation, and later
      autonomous-action boundaries against the latest durable decision
- [x] prove unknown configuration becomes `observe` and missing production
      gate executors lower the result independently of configured mode

**P12.2e — complete journal coverage.** Add a real `record_iteration`
producer, then cover every production effect with the following concrete
operation policy:

| Operation | Natural key | Idempotency / ambiguity recovery | Maximum duplicate |
|---|---|---|---|
| `jira_write:create_issue` | goal id + step index | Probe by goal label + step before retry; detect and reconcile if the post-create key was not persisted | 0 when probe succeeds; otherwise detected |
| `jira_write:add_ai_ready` | ticket key | Re-read label presence | 0 |
| `jira_write:transition` | ticket key + target status | Re-read status | 0 |
| `jira_write:comment` | ticket key + body digest | Search comment-body digest before retry | 0–1, detected |
| `slack_post` | channel + thread + body digest | At-least-once; re-post permitted | 1, declared |
| `worktree_create` | ticket key + lock id | Reuse existing path or clean and recreate | 0 |
| `git_commit` | branch + tree digest | Probe branch head/commit tree | 0 |
| `git_push` | remote + branch + SHA | Probe remote ref | 0 |
| `pr_create` | repo + head branch | Probe open PR by head before retry | 0 |
| `pr_merge` | repo + PR number + SHA | Probe merge state | 0 |
| `model_invoke` | goal + iteration + prompt digest | Reserve before call; charge the full reservation when outcome is ambiguous | 1 call, bounded spend |
| `gate_run` | candidate SHA + gate name | Results are SHA-bound; a retry may re-run | Idempotent by SHA |

Crash tests exercise each operation before reservation, after reservation,
after the external effect, and after completion persistence. Completed actions
never replay; ambiguous actions obey their declared duplicate and budget
bounds.

P12.2e checklist:

- [x] select idempotency, probe, recovery classification, attempt bound, and
      duplicate allowance from the concrete operation rather than `ActionKind`
- [x] journal production Jira, Slack, worktree, Git/push/PR, model, and gate
      boundaries; keep `pr_merge` policy ready while no merge executor exists
- [x] reserve two bounded model attempts atomically, charge an ambiguous call
      at its full per-attempt reservation, and preserve actual successful cost
- [x] produce `loop_iterations` from real implementation attempts
- [x] run the 12-operation crash matrix at all four crash points, including
      duplicate Jira-create detection and completed-action replay refusal

### P13 — Harness manifest, trust root, sandbox

> **🔶 Partial. P13.3a enforces the sandbox in the production shell path and
> before execution mutation.** Context assembly / the knowledge map remains,
> so the whole phase is not complete.

Goal: declare what may run, where it may write, and what holds verification
authority — then isolate execution so untrusted repository code cannot reach
past it.

**Trust root is declared by capability, not by path glob.** A glob list is
both over- and under-inclusive: `package.json` matters only because
`scripts.test` is what `npm test` delegates to, and a *newly created* file can
acquire authority without appearing on any list. Entries are `file`, `tree`,
`json_pointer` (sub-file), and `derived` (whatever the trusted commands
resolve to, computed at load).

**Unresolvable commands lower readiness rather than being dropped.** A
`bash -lc` program can compute its target at runtime, so its closure cannot be
computed honestly. Readiness is a ceiling on autonomy — `unready` → `propose`,
`partial` → `implement`, `full` → `autonomous` — so a hole in the trust root
reduces what the system may do instead of being silently excluded from the
digest.

**Trusted commands must be structured argv.** `_parse_command` already
rejected string commands; that rule now extends to every gate, which retired
`lab.yaml`'s three `bash -lc` guards. Those guards exited 0 when
`package.json` was absent — so on the greenfield state every new goal starts
from, all three gates passed vacuously.

Sandbox, in `adapters/local/sandbox.py`. Four decisions worth keeping:

- **No `preexec_fn`.** It is not async-signal-safe and this process runs
  asyncio TaskGroups plus `asyncio.to_thread`, where it can deadlock the
  child. Limits are applied by `prlimit(1)` in the child's own argv, after
  `exec`.
- **`--clearenv` is required.** Measured, not assumed: `bwrap --unshare-all`
  inherits the parent environment, leaking 14 credential-ish variables in a
  polluted-env test.
- **Worktree is `--ro-bind` with declared writable mounts.** Default deny; a
  gate that can rewrite the source it tests is not a gate.
- **`available()` attempts a real unshare.** Ubuntu 24.04 sets
  `kernel.apparmor_restrict_unprivileged_userns=1`, under which a healthy
  `bwrap` still fails; a presence check reports such a host as sandbox-ready
  when it is not. The probe binds exactly what `wrap()` binds, because
  `/lib64` is a symlink into `/usr` and a smaller bind set fails for an
  unrelated reason.

Setup on Ubuntu 24.04 requires `sudo apt install bubblewrap` plus an AppArmor
profile granting `userns` to `/usr/bin/bwrap` (same shape as the shipped
`keybase` profile). Without it the system still runs, capped at `propose`.

**Residual risk:** when install scripts are not ignored they run with network
inside the sandbox. The sandbox bounds the damage; it does not eliminate
supply-chain execution. `lab.yaml` therefore uses `npm ci --ignore-scripts`.

#### P13.3a landed — runtime boundary, not merely host capability

`ExecutionEnvironmentPreflight` now refuses a non-`bwrap` wrapper before lock
acquisition, Jira claim, feedback/local worktree creation, or graph resume. The
production runtime injects it independently into the runner, Jira and feedback
coordinators, approval resume, restart resume, and implementation service.
Construction stays lazy, so an incapable host can still run intake/proposal
work while repository execution is refused.

`RuntimeShellFactory` is the single production factory used by
`AdapterTestService` and the model-callable contract test runner. The port
accepts adapter-independent `CommandExecutionPolicy`; the adapter translates
it to `SandboxPolicy`. `CommandSpec` requires explicit write mounts and network
mode, Bubblewrap binds the repository root separately from the nested working
directory, and every actual wrapper launch returns a complete
`SandboxAttestation` on `CommandResult`.

Smoke no longer conflates three facts. It reports host capability, hard-wired
runtime enforcement, and evidence from a harmless command sent through the
production wrapper path as separate checks.

P13.1/P13.2 checklist — **landed**:

- [x] sandbox with PID, memory, timeout, filesystem, and credential isolation
- [x] process-group termination (fixes a live orphan bug: `subprocess.run`
      killed only the direct child, so a timed-out `npm` left grandchildren)
- [x] trust root with closure and honest unresolved reporting
- [x] readiness ladder surfaced by `runtime_smoke`

P13.3a makes the sandbox a real boundary independently of P12. It leaves
P13.3's context assembly / knowledge map open, so P13 remains 🔶.

**Execution preflight.** `ExecutionEnvironmentPreflight` probes actual sandbox
availability and requires an enforcing sandbox unconditionally for production
repo-contract commands. It rejects `NullSandbox` even if adjacent policy text
claims `bwrap`. It runs before lock acquisition, Jira claim/transition,
worktree creation, or graph resume in `JiraExecutionCoordinator`,
`FeedbackExecutionCoordinator`, `ExecutionApprovalCommandHandler`, restart /
checkpoint resume, and `OrchestratorRunner`. Intake and proposal-only paths,
where no repository command can run, remain available. The guard stays in
place when unavailable; there is no later shell-construction fallback.

**Port and schema boundary.** The port/domain type is named
`CommandExecutionPolicy` (distinct from the existing repo-contract
`ExecutionPolicy`), and `ShellPort.run` accepts it. `ports/` never imports an
adapter type; adapter-side `SandboxPolicy` translates the domain policy.
`CommandSpec` declares `writable_paths` and `network`. `install` alone may
request network; `test`, `lint`, `typecheck`, and `build` may not. Validation
rejects escaping writable paths and network requested on a non-install
command; neither value is inferred from the language.

**Root and working directory are distinct.** Sandbox wrapping receives both
the repository root and the command working directory. Bubblewrap read-only
binds the whole root and changes directory within it, so a nested command can
still read the rest of the repository. Writable mounts must resolve beneath
the root.

**One runtime-bound shell factory.** Both `AdapterTestService` and the model-
callable `_make_contract_test_runner` use the same injected production shell
factory. Neither may bypass sandbox construction.

**Per-command evidence.** The real wrapper path emits a structured
`SandboxAttestation` carrying `Sandbox.profile` (`none` or `bwrap`), command-
policy digest, repository root, command working directory, network mode,
writable mounts, and a digest of the actual launch/wrapper argv. Smoke reports
three distinct facts: host capability, whether runtime wiring is configured to
enforce, and per-command enforcement evidence.

P13.3a checklist:

- [x] sandbox preflight guards every production entry point before mutation
- [x] `CommandExecutionPolicy` crosses the port without adapter imports
- [x] command schema declares and validates writes/network
- [x] nested working directories bind the repository root and chdir within it
- [x] both production shell paths enforce the same runtime-bound sandbox
- [x] real wrapper emits complete `SandboxAttestation`
- [x] smoke separates capability, configuration, and enforcement evidence
- [x] controlled-local-endpoint tests prove install network and non-install
      isolation without relying on the public internet
- [ ] context assembly / knowledge map (separate P13.3 remainder)

Exit criteria for P13:

- Requesting autonomous execution on a host with no working sandbox is
  **refused**, not downgraded silently.
- No production code path can construct a shell that runs contract commands
  outside the sandbox.
- The autonomy ceiling reflects actual isolation in force, not host capability.

### P14 — Verification bound to an immutable committed SHA

Status: ⬜ not started. Nothing below is implemented.

Objective: make verification evidence describe the artifact that will actually
ship, rather than a working directory that resembles it.

Today the graph is `plan → request_execution_approval → implement → run_tests
→ review → open_pull_request` (`orchestrator/graph.py:64-122`), and the commit
happens *inside* `PullRequestService.open_pull_request`
(`orchestrator/git_services.py:65-77`) — **after** tests and review. So gates
run against the worktree, which still holds ignored files, stale
`node_modules`, and generated output that is not in any commit. A green run
proves something about a directory, not about a candidate.

Required changes:

- Topology becomes `implement → commit → verify → review → open PR`; add a
  COMMIT node and its `visited_nodes` expectations.
- Verification runs in a **fresh checkout of the commit**, never the worktree.
- Install once per verification, not once per gate. Today
  `_build_contract_shell` is called per gate, which means up to five full
  installs per attempt.
- Classify failures: `defect` → retry the code · `flake` → re-run once ·
  `transient` → retry verification only, with no new commit and no model call
  · `policy` → stop and escalate.
- Detect gates that mutate the tree they are testing.

#### P14 invariants

**Digest bindings.** `VerificationPolicy` (`orchestrator/gates.py:111-125`)
carries `contract_digest`, `harness_digest`, and `sandbox_profile` **all
defaulting to `""`**, and has **no `policy_digest`** at all, despite the exit
criteria below promising digest-mismatch rejection. Add `policy_digest`, and
make every digest binding **required and non-empty**. An empty digest set denies;
it must not read as "matches".

**Run and attempt identity.**

- Verification run ID, verification attempt, and gate attempt are **first-class
  identity**, persisted **before** dispatch, never derived afterwards.
- **An intentional retry gets a new attempt identity; crash recovery reuses the
  existing one.** Conflating the two either loses evidence or double-charges.
- Every attempt — failed and successful — **remains queryable**. This requires
  new append-only tables; `action_records` is one mutable row per action
  (`goal/spine.py:53`) whose `state`, `attempts`, and `error` are updated in
  place, so success overwrites failure.
- Exactly **one deterministic classification owner**. `flake` is an *observed*
  fail-then-pass, never an initial assertion. `unknown` fails closed.
- Backoff is **durable, via a next-attempt timestamp** — never an in-process
  sleep, which a restart silently discards.
- Attempt and wall-time budgets bound every retry path.
- **Install is a prerequisite that runs once per verification**, and a failed
  install must **prevent required gates from appearing to pass**.
- Required versus optional gate failure behavior is stated explicitly, not
  inferred.

**Commit identity.** `action_records` carries
`UNIQUE (goal_id, operation, natural_key)` (`goal/spine.py:76`) — **independent
of iteration**, even though `action_id` includes iteration
(`goal/types.py:500-503`). With `git_commit`'s natural key `branch:tree_digest`
(`orchestrator/git_services.py:133`), an **A → B → A rework reproduces tree A's
digest, collides with the original intent, and restores the old SHA.**

- The natural key becomes **`branch + expected parent OID + target tree OID`**.
- Crash recovery must prove the recovered commit has **exactly** the expected
  parent and the expected tree.

**Mutation detection.** State the scope for tracked, staged, ignored, and
untracked content. Verification runs from a **read-only source checkout with
ephemeral mounts reusing `CommandSpec.writable_paths`** — which already exists
and is already required (`config/repo_contract.py:59`) and already drives
Bubblewrap mounts. Specify temporary backing-directory creation and cleanup,
whether install output persists across gates within one verification,
per-command versus verification-wide writable paths, integrity-manifest
exclusion rules, and symlink and path-boundary validation. Integrity is checked
**before install and after every gate**, with before/after manifests recorded,
and it must be **provable that declared writable output cannot mask tracked
source mutation**.

**Invalidation.** Any new implementation or commit **invalidates all downstream
verification, review, and authorization state.** Stale evidence is not merely
ignored; it is cleared.

Exit criteria for P14:

- **Every verification record binds four identifiers: the candidate SHA, the
  contract digest, the policy digest, and the harness digest.** A record that
  cannot name all four does not authorize anything.
- Evidence collected against SHA *A* never authorizes SHA *B*.
- `gate_results` has a real producer, so a gate's status and failure class are
  queryable after the fact.
- No gate can pass vacuously on a repository with no application in it.

### P15 — Independent complete-diff review

Status: ⬜ not started.

Objective: a reviewer that reads code rather than the implementer's account of
it, and that cannot become the implementer.

Today `ModelRouterReviewService.review`
(`orchestrator/model_services.py:825-860`) sends `state.summary` and
`state.implementation_result` — both written by the implementing model. The
reviewer grades a self-report. There is no diff helper in the codebase at all.

Required changes:

- Supply `git diff base...candidate`, derived from git, fenced and treated as
  data rather than instructions.
- Derive the changed-file list from git, never from a model's summary.
- Treat filenames as hostile: null-delimited output, no parsing of
  human-formatted text.
- Block rather than warn on a detected secret, a trust-root change, or
  incomplete coverage. Binary and submodule changes require a human, because a
  text diff cannot represent them.
- Absorb P9's lint requirement here only insofar as review needs it; the gate
  itself belongs to P14.

#### P15 invariants

**Contributor provenance.** `implementation_result` is
`{"status", "changed_files", "summary"}` (`orchestrator/model_services.py:243-245`)
— it names **no contributor providers**, so there is no exclusion set to build.
Persist provenance for **every response that causally contributed a
candidate-changing tool call**, not merely the last one.

**Exclusion happens inside the router.** `ModelRouter.invoke`
(`router/model_router.py:86`) has no exclusion mechanism; `exclude_providers` is
ignored metadata (`goal/semantic_check.py:175`) and enforcement is **post-hoc
rejection** (`goal/semantic_check.py:181-188`) — the wrong provider is **called
and paid for**, then discarded.

- The **complete contributor set** is excluded **inside the router's selection
  loop**, so an excluded provider is **never invoked**. "Excluded" must mean not
  called, not called-and-rejected.
- An exhausted filtered chain **fails closed** and escalates. It never falls
  back to the implementer.
- Record the originally selected chain, the filtered chain, the attempted chain,
  the actual provider and model **per chunk**, and proof that excluded providers
  were never called.
- Provider exclusions and the routing-policy/selector digest are part of the
  **journaled model-call request identity**.

**Deterministic checks precede model disclosure.** Secret, scope, trust-root,
binary, and submodule checks run **before any diff content reaches an external
model** — a detected secret must not be exfiltrated to a provider in the act of
asking whether it is a secret.

**Chunking and coverage.** Deterministic chunking with **no truncation**; every
chunk proven reviewed; missing or duplicated chunks deny. Review binds base OID,
merge-base OID, candidate SHA, and full-diff digest. **Base movement invalidates
or forces recomputation.**

**Budgets before dispatch.** `max_diff_bytes`, token, model-cost, wall-time, and
chunk-count budgets are enforced **before** dispatch, and oversized files or
hunks deny or require human review. **"No truncation" must not mean unlimited
model calls.**

**Safe diff extraction.** External diff drivers and `textconv` are **disabled**;
metadata is null-delimited and machine-readable; **filenames are treated as
hostile and never used to build shell commands**; renames, mode changes,
symlinks, binaries, and submodules are recorded explicitly; the base is bound to
a fetched explicit snapshot. Prompt fencing is **defense in depth only** — pair
it with structured prompts, tool restriction, strict output validation, and
adversarial tests.

Exit criteria for P15:

- **Review consumes the complete real diff and records its coverage** — which
  files and hunks were actually examined, so partial review is visible rather
  than assumed.
- **Maker ≠ checker holds across provider fallbacks, not just at startup.**
  The router walks `(primary, *fallbacks)` and returns whichever answers, so a
  reviewer outage otherwise makes the implementer its own reviewer silently.
  The implementer's provider is passed as an exclusion set, and the check
  fails closed when the chain is exhausted.
- A truncated or unparseable review is a denial, never a pass.

### P16 — Evaluation and demonstration evidence

Status: ⬜ not started.

Objective: prove the user-visible behavior works, and be able to say whether
the system is getting better or worse over time.

Required changes:

- Start the app in the sandbox, exercise each user-visible acceptance
  criterion, capture artifacts, compare against declared expectations.
- **Name the pass oracle.** Either a declared deterministic assertion (exit
  code, probe output, snapshot hash) or an independent checker examining the
  artifact. The maker's own say-so is not evidence. A criterion with neither
  is `not_runnable`, not a pass.
- Route failures by kind: wrong output → defect · startup timeout → transient
  · missing driver → policy · a required demo that cannot run → deny without
  retry.

#### P16 invariants

**The oracle must be specifiable.** `AcceptanceCriterion.oracle`
(`goal/types.py:201`) names only a *kind* — it cannot say what would constitute
passing. An `OracleSpec` carries checker/assertion identity, expected result,
inputs, timeout, and environment/tool digest.

**A criterion has four outcomes, not two.** `CriterionOutcome.met: bool`
(`goal/types.py:657`) cannot express `not_runnable`, so a criterion nobody could
evaluate is indistinguishable from one that failed — or, worse, defaults to
passing. Replace it with explicit status: `passed` · `failed` · `not_runnable` ·
`not_run`. **A criterion without an `OracleSpec` is `not_runnable`, never a
pass**, and a legacy string or missing oracle is `not_runnable` — never silently
upgraded.

**Corpus inputs are versioned**: base commits, contracts, policies, harnesses,
environment image and tool versions, oracle definitions, budgets, and model
configuration. Results are attributable to a candidate SHA, and the corpus
re-runs with comparable results. **Graduation thresholds are recorded before
execution**, not chosen afterwards to fit the outcome.

**Relationship to candidate authorization.** P16 demonstration evidence
contributes to `GoalAchievement`; candidate authorization governs whether an
intermediate candidate may ship as a PR. Where a criterion is `demo_required`
**and** must gate its own candidate, its demo evidence is collected and bound
*before* authorization. Which criteria those are is a declared decision, not a
default.

Exit criteria for P16:

- **A repeatable evaluation corpus exists** — a fixed set of tasks that can be
  re-run against a changed system to compare outcomes.
- **Graduation thresholds are measurable and written down** before they are
  used to justify raising autonomy, not chosen afterwards to fit the result.
- Demo evidence is attributable to a specific candidate SHA and criterion.

### P17 — Durable delivery: outbox, attestation, trusted CI, promotion

Status: ⬜ not started.

Objective: make external side effects crash-safe, and merge only on evidence
that a machine can check.

Required changes:

- Outbox for every external write, built on P12's action journal.
- Attestation binds the goal, all digests, and the SHA/tree/base OIDs, keyed
  on `(repo, head SHA)` so a new push invalidates prior approval automatically.
- Sequence cursor with compare-and-swap on `(goal, plan version, step,
  predecessor, merged SHA)`, so duplicate, out-of-order, replayed, and
  superseded-plan events each fail on a different field.
- Read CI through `gh api graphql`. **Verified: `gh pr view --json
  statusCheckRollup` cannot do this** — it exposes no app identity and no
  per-check commit OID. Take the latest run for the exact commit; earlier runs
  are superseded, because a rerun is how a human clears a flake. A check from
  an unexpected app is blocked, never waited on. A repo with no checks
  configured never merges, making this opt-in per repository.
- Fix both early returns in `open_pull_request` that return a PR URL without
  pushing, so the rerun path cannot show reviewers stale code.

Exit criteria for P17:

- **Opening a PR is not achievement.** `GoalAchievement.achieved` requires the
  promotion PR to have been created *and read back from GitHub*; the correct
  sequence is criteria met → `ready_for_promotion` → open PR → read back →
  `achieved`. Evidence that criteria are met is not the same as having
  presented the work.
- Replaying any journal entry produces no second side effect, and this is
  demonstrated against duplicate, out-of-order, and superseded events.
- Nothing merges into `main`. `integration/*` only, and only on trusted CI.
- A hand-edited attestation fails signature verification.

### P18 — Bounded rework from promotion-PR change requests

Status: ⬜ not started. Full specification below — nothing about P18 lives
outside this repository.

Objective: you request changes on the promotion PR; the system fixes them and
the PR updates itself.

#### The insight that makes this cheap

The promotion PR's head branch **is** `integration/x`. So anything merged into
`integration/x` updates that PR automatically — nothing needs to re-create or
re-point it. And `create_worktree(..., base_branch="integration/x")` already
produces a pushable `agent/KEY/lock` branch cut from the integration branch.
The existing pipeline can do this end to end. The only thing genuinely missing
is a Jira ticket key to hang a run on, because every lock, branch, and graph
thread is keyed on one.

#### The flow

```txt
you click "Request changes"
  → poller finds the review
  → scope-check it against the GoalContract
  → file a rework ticket (sequence label, integration/x as its base)
  → the ORDINARY pipeline runs it: agent/KEY/lock from integration/x
    → PR into integration/x → verified, reviewed, merged
  → promotion PR updates itself
  → bot comments "fixed in LAB-31"
```

#### Design decisions

**Trigger on `state == "CHANGES_REQUESTED"` only.** Plain comments and
approvals do nothing. One review — with all its inline comments — becomes one
rework ticket. GitHub's `PullRequestReview` exposes `state`, `id`, and
`submittedAt`, which is everything the trigger needs.

**Dedupe on review ID, not on comment text.** This also fixes a live bug in the
existing ticket-PR feedback path: `_fingerprint` hashes the *entire
concatenated comment blob*, so one new comment changes the hash and re-sends
every earlier comment along with it. On a long review thread that gets
expensive and repetitive fast.

**Scope-check before filing.** Run the requested change through P12's
deterministic scope rules against the `GoalContract`. In scope → file with
`ai-ready` and it runs. Out of scope ("while you're here, add auth") → file
*without* `ai-ready` and post a Slack question. The contract stays the
boundary; a PR comment must not become a way to authorize unbudgeted work.

**Bound the loop.** Rework rounds per promotion PR are budgeted, and a review
that repeats an objection already addressed is a non-convergence signal (P12),
not another attempt.

#### What to build

- **`PromotionFeedbackPoller`** — sibling of `GitHubFeedbackPoller`. Finds open
  PRs with head `integration/*` and base == the promotion base. The existing
  poller cannot be reused: `_open_agent_prs` (`feedback/github.py:518-540`)
  hard-filters to `headRefName.startswith("agent/")`.
- **Integration branch → epic key.** `integration/lab-30` → `ai-sequence-lab-30`
  → `LAB-30`. Note the epic key is **lowercased** at
  `intake/jira_writer.py:307`, and `_TICKET_KEY_RE` requires uppercase — which
  is one reason the promotion PR is invisible to the current path.
- **Rework ticket creation** via the existing
  `JiraClient.create_issue(project_key, *, summary, description, labels,
  parent_key)`. Nothing files follow-up tickets today; this is the one
  genuinely new capability.
- ⚠️ **The description must contain the line
  `- Pull request base branch: integration/lab-30`.** That exact format is what
  `_pull_request_base_branch` (`jira/work_item_loader.py:160-166`) parses, and
  it is what makes the rework ticket behave like an ordinary sequence ticket
  with no other changes. This is the whole trick — reuse, not new plumbing.
- **Each inline comment becomes one acceptance criterion**, so P15's reviewer
  has something falsifiable to check the fix against.
- **Post back on the promotion PR** naming the ticket, so a reviewer is not
  left wondering whether anything happened.

**Reuse as-is:** `SQLiteFeedbackStore`, the poller/queue/worker skeleton,
`_feedback_entry` / `_format_feedback`, the `gh` comment-fetch commands
including `gh api repos/{repo}/pulls/{n}/comments` for inline comments, and
`GhCliFeedbackClient._run`'s credential plumbing.

#### Implementation notes verified against the code

- `_open_agent_prs` (`feedback/github.py:518-540`) filters to
  `headRefName.startswith("agent/")`, so the promotion PR is invisible to it.
  A sibling poller is needed, not a tweak.
- `_ticket_key` (`feedback/github.py:664-669`) needs an **uppercase**
  `ABC-123`. The promotion PR's title, branch, and body carry only the
  lowercased epic key, so it returns `""`.
- `create_worktree_for_branch` (`git_adapter.py:89-125`) calls
  `_validate_push_branch` and so refuses an `integration/*` head — but
  `create_worktree(..., base_branch=...)` accepts one as a **base** and cuts a
  pushable `agent/*` branch from it. That is the seam.
- `ai-step-*` labels are written (`jira_writer.py:315-324`) and **never read**;
  ordering is by `ORDER BY created ASC`. A rework ticket therefore sorts last
  naturally, which is what we want.
- `GhCliFeedbackClient` has **no test coverage at all** — not the `agent/`
  filter, not `_ticket_key`, not the self-comment filter. Anything built beside
  it must not assume that code is exercised.

**Interaction to check:** when the rework PR merges into `integration/x`,
`advance_after_merge` fires and looks for the next queued ticket. The rework
ticket carries `ai-ready` from creation, so the `labels != ai-ready` clause
excludes it; with nothing else queued it takes the promotion-PR branch, where
`_existing_pull_request_url` returns the PR that already exists. That should be
a no-op — worth an explicit test rather than an assumption.

Exit criteria for P18:

- Only `CHANGES_REQUESTED` triggers work; approvals and plain comments do not.
- Dedupe on review ID, not comment text: re-submitting the same review verbatim
  files no second ticket and re-sends no comments already addressed.
- A request outside the `GoalContract` parks with a Slack question instead of
  running — a PR comment must never become a way to authorize unbudgeted work.
- Rework rounds are budgeted, and a repeated objection is a non-convergence
  signal rather than another attempt.
- The promotion PR updates itself; no second promotion PR is ever opened.

---

## Known limitations and configuration debt

Stated precisely, because earlier drafts of this material overstated several of
these.

### Target repositories are inputs; their code changes are outputs

A **target repository** is an external input to the agent runtime, described by
a `config/repos/*.yaml` contract. The **application changes produced inside it**
are the runtime's outputs. Nothing in `src/` may be shaped around any particular
target.

### `config/repos/lab.yaml` is app-specific and badly named

It is named `lab` but declares `repo.name: ofertas-sv`, and its `source_dirs`,
`config_paths_allowed`, and trust root are specific to that Next.js/Supabase
application.

**It is not a generic fallback for arbitrary repositories.**
`_find_matching_contract` (`orchestrator/local_services.py:332-357`) iterates
`config/repos/*.yaml` and returns a contract only when
`_contract_matches_state` (`:360`) finds that the contract's `repo.name`
normalizes to the state's repository name, or its `repo.root` resolves to the
state's repo path. A repository that matches neither falls through to
`scaffold_repo_contract` (`local_services.py:119-127`), which writes a fresh
contract from the detected language and package manager.

So the real problem is naming and shipped-example hygiene — a file called
`lab.yaml` that silently means `ofertas-sv` is confusing to read and to
maintain — not silent inheritance by unrelated repositories.

### `config/policy/risk.yaml` is deployment-specific policy data

Its `repositories:` allowlist naming `ofertas-sv` is **not inherently an
architecture leak**: an allowlist exists precisely to name the repositories a
deployment permits unattended work in, and an empty list permits nothing, which
is the correct fail-closed default.

The real issue is narrower: this file is *deployment* configuration that is
currently shipped in the repository as though it were a sensible default. A
fresh clone inherits one operator's allowlist. It should ship either empty or
as a documented example, with the operative copy supplied per deployment via
`AGENT_SYSTEM_RISK_POLICY_PATH`.

### Greenfield behavior

Stated carefully, because the distinction matters:

- **The system can create a remote GitHub repository** for a local Git
  repository that already exists. `_ensure_origin_remote`
  (`adapters/local/git_adapter.py:319-382`) detects a missing `origin` and runs
  `gh repo create <name> --private --source <repo_root> --remote origin`,
  then authorizes the bot and pushes the default branch. It requires both
  `GH_ADMIN_TOKEN` and `GH_BOT_TOKEN`, and raises `PushError` rather than
  falling back to ambient local credentials.
- **It does not create or scaffold an application repository from nothing**
  before worktree creation. `gh repo create --source` requires `repo_root` to
  be an existing local Git repository, and there is no `git init` or
  application-scaffold step anywhere in `src/`.

So a request naming a target that has no local repository at all stalls: the
repo contract scaffolder can describe a repository, but nothing brings one into
existence. Harness readiness reports `unready`, which correctly caps autonomy
at `propose`. It is a prerequisite for "build me a new application" requests.

**Decision (2026-08-04): deferred past Milestone B**, as an accepted limitation
rather than an open question. Work on repositories that already exist is the
path with specifications behind it; greenfield adds a new class of authority
questions — what a human authorized when no contract, trust root, or default
branch yet exists — on top of a loop that has no operational evidence. Revisit
once M2 has produced a run report and the delivery tail is reachable.

### Operational validation

P0–P7 are implemented and covered by unit tests. **They are not claimed to be
operationally validated.** External Slack, Jira, and GitHub end-to-end
verification remains manual, and the repository holds no recorded live-run
evidence. Where this document marks something ✅, read it as "implemented,
wired, enforced, and unit-tested" — level 2 of the maturity scale, not level 3.

**Stated exactly, as measured on 2026-08-04:** every runtime database is empty
and no run has ever been recorded.

```txt
execution_approvals: 0    intake_proposals: 0    ticket_locks: 0
checkpoints: 0            checkpoint_writes: 0   feedback_fingerprints: 0
.agent-system-data/transcripts/   does not exist
goal contract and spine DBs       never created
```

**Nothing in this project has reached maturity level 3.** Not one ticket has
been processed end to end. This is not a gap in a particular phase; it applies
to every ✅ in the status table. Closing it is [M2](#m2--first-operationally-validated-run),
which is scheduled ahead of P14 for that reason.

Related: transcripts are gated behind `AGENT_SYSTEM_TRANSCRIPTS_ENABLED`,
default **false**. A feature that *can* be switched on has demonstrated nothing
about how the system behaves; cite a reviewed transcript from a real run, never
the existence of the sink.

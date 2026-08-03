"""Tests for transcripts, telemetry, and the loop-observability hooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import threading

import pytest

from ticket_agent.observability.telemetry import (
    GateRecord,
    IterationRecord,
    NullTelemetryRecorder,
    SQLiteTelemetryStore,
)
from ticket_agent.observability.transcripts import (
    JsonlTranscriptRecorder,
    NullTranscriptRecorder,
    TranscriptEvent,
    safe_tool_args,
)


@dataclass(frozen=True)
class _Sample:
    name: str
    when: datetime


def _recorder(tmp_path: Path, **kwargs) -> JsonlTranscriptRecorder:
    return JsonlTranscriptRecorder(tmp_path / "transcripts", **kwargs)


def _lines(root: Path, stem: str) -> list[dict]:
    path = root / "transcripts" / f"{stem}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- transcripts -----------------------------------------------------------


def test_transcript_filename_uses_ticket_key_and_run_id(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record(
        TranscriptEvent(ticket_key="LAB-30", kind="node", name="plan", run_id="abcd1234")
    )
    recorder.close()

    assert (tmp_path / "transcripts" / "LAB-30-abcd1234.jsonl").exists()


def test_transcript_appends_across_recorder_instances(tmp_path):
    for name in ("plan", "implement"):
        recorder = _recorder(tmp_path)
        recorder.record(
            TranscriptEvent(
                ticket_key="LAB-30", kind="node", name=name, run_id="abcd1234"
            )
        )
        recorder.close()

    assert [row["name"] for row in _lines(tmp_path, "LAB-30-abcd1234")] == [
        "plan",
        "implement",
    ]


def test_transcript_redacts_secrets_and_paths_recursively(tmp_path):
    recorder = _recorder(tmp_path, local_paths=["/home/someone/repos/demo"])
    recorder.record(
        TranscriptEvent(
            ticket_key="LAB-30",
            kind="node",
            name="implement",
            run_id="abcd1234",
            payload={
                "nested": {
                    "list": [
                        {"token": "ghp_abcdefghij0123456789ABCDEF"},
                        "/home/someone/repos/demo/src/app.py",
                    ]
                },
                "env": "DEEPSEEK_API_KEY=abcdefghijklmnopqrstuvwxyz012345",
            },
        )
    )
    recorder.close()

    raw = (tmp_path / "transcripts" / "LAB-30-abcd1234.jsonl").read_text()
    assert "ghp_abcdefghij0123456789ABCDEF" not in raw
    assert "abcdefghijklmnopqrstuvwxyz012345" not in raw
    assert "/home/someone" not in raw
    assert "<redacted-secret>" in raw
    assert "<repo>/src/app.py" in raw


def test_transcript_truncates_long_values(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record(
        TranscriptEvent(
            ticket_key="LAB-30",
            kind="node",
            name="implement",
            run_id="abcd1234",
            payload={"blob": "x" * 10_000},
        )
    )
    recorder.close()

    blob = _lines(tmp_path, "LAB-30-abcd1234")[0]["payload"]["blob"]
    assert blob.endswith("[truncated]")
    assert len(blob) < 10_000


def test_transcript_normalizes_paths_datetimes_and_dataclasses(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record(
        TranscriptEvent(
            ticket_key="LAB-30",
            kind="node",
            name="plan",
            run_id="abcd1234",
            payload={
                "path": Path("relative/file.py"),
                "when": datetime(2026, 7, 27, tzinfo=UTC),
                "sample": _Sample(name="s", when=datetime(2026, 7, 27, tzinfo=UTC)),
            },
        )
    )
    recorder.close()

    payload = _lines(tmp_path, "LAB-30-abcd1234")[0]["payload"]
    assert payload["path"] == "relative/file.py"
    assert payload["when"].startswith("2026-07-27")
    assert payload["sample"]["name"] == "s"


def test_transcript_terminates_on_recursive_payload(tmp_path):
    recorder = _recorder(tmp_path)
    payload: dict = {"a": 1}
    payload["self"] = payload

    recorder.record(
        TranscriptEvent(
            ticket_key="LAB-30",
            kind="node",
            name="plan",
            run_id="abcd1234",
            payload=payload,
        )
    )
    recorder.close()

    assert _lines(tmp_path, "LAB-30-abcd1234")[0]["payload"]["self"] == "<recursive>"


@pytest.mark.parametrize(
    "ticket_key",
    ["../../etc/passwd", "lab-30", "LAB30", "", "LAB-30/../../x"],
)
def test_transcript_rejects_invalid_ticket_keys_before_touching_disk(
    tmp_path, ticket_key
):
    recorder = _recorder(tmp_path)
    recorder.record(TranscriptEvent(ticket_key=ticket_key, kind="node", name="plan"))
    recorder.close()

    # Validation happens before any filesystem call, so the root is never even
    # created -- not merely empty of the traversal target.
    assert list(tmp_path.rglob("*")) == []
    assert recorder.write_failures == 1


def test_transcript_rejects_traversal_in_run_id(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record(
        TranscriptEvent(
            ticket_key="LAB-30", kind="node", name="plan", run_id="../../etc"
        )
    )
    recorder.close()

    assert recorder.write_failures == 1
    assert not list(tmp_path.rglob("*etc*"))


def test_transcript_caps_open_file_handles(tmp_path):
    recorder = _recorder(tmp_path, max_open_handles=2)
    for index in range(6):
        recorder.record(
            TranscriptEvent(
                ticket_key=f"LAB-{index}", kind="node", name="plan", run_id="abcd1234"
            )
        )

    assert len(recorder._handles) == 2
    recorder.close()
    assert len(list((tmp_path / "transcripts").glob("*.jsonl"))) == 6


def test_transcript_concurrent_writes_produce_whole_lines(tmp_path):
    recorder = _recorder(tmp_path)
    threads = 8
    per_thread = 40

    def write(worker: int) -> None:
        for index in range(per_thread):
            recorder.record(
                TranscriptEvent(
                    ticket_key="LAB-99",
                    kind="turn",
                    name=f"turn.{worker}.{index}",
                    run_id="deadbeef",
                    payload={"filler": "y" * 200},
                )
            )

    workers = [threading.Thread(target=write, args=(n,)) for n in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    recorder.close()

    rows = _lines(tmp_path, "LAB-99-deadbeef")
    assert len(rows) == threads * per_thread
    assert recorder.write_failures == 0


def test_transcript_sink_failure_warns_once_and_counts(tmp_path, caplog):
    recorder = _recorder(tmp_path)

    class _Exploding:
        def write(self, _line: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            raise OSError("disk full")

    recorder._handles["LAB-30-abcd1234"] = _Exploding()  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        for _ in range(3):
            recorder.record(
                TranscriptEvent(
                    ticket_key="LAB-30", kind="node", name="plan", run_id="abcd1234"
                )
            )

    assert recorder.write_failures == 3
    assert sum("transcript write failed" in r.message for r in caplog.records) == 1


def test_null_recorder_writes_nothing(tmp_path):
    NullTranscriptRecorder().record(
        TranscriptEvent(ticket_key="LAB-30", kind="node", name="plan")
    )
    assert not list(tmp_path.iterdir())


# -- tool argument shaping -------------------------------------------------


def test_safe_tool_args_keeps_shapes_and_sizes_not_content():
    safe = safe_tool_args(
        {
            "path": "src/app.py",
            "offset": 10,
            "limit": 5,
            "content": "s" * 400,
            "old_string": "abc",
            "new_string": "",
            "secret_blob": "value",
        }
    )

    assert safe["path"] == "src/app.py"
    assert safe["offset"] == 10
    assert safe["content_chars"] == 400
    assert safe["old_string_chars"] == 3
    assert safe["new_string_chars"] == 0
    assert "content" not in safe
    assert "secret_blob" not in safe
    assert safe["secret_blob_present"] is True


def test_safe_tool_args_tolerates_missing_args():
    assert safe_tool_args(None) == {}


# -- telemetry -------------------------------------------------------------


def test_telemetry_first_stage_timestamp_wins(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    store.record_stage("LAB-30", "claimed", goal_id="g1")
    first = store._rows("SELECT claimed_at FROM ticket_funnel")[0]["claimed_at"]
    store.record_stage("LAB-30", "claimed", goal_id="g1")
    second = store._rows("SELECT claimed_at FROM ticket_funnel")[0]["claimed_at"]
    store.close()

    assert first == second


def test_telemetry_retains_gates_that_never_ran(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    store.record_gates(
        [
            GateRecord("LAB-30", 1, "test", "failed", failure_class="defect"),
            GateRecord("LAB-30", 1, "lint", "not_run"),
            GateRecord("LAB-30", 1, "build", "not_run"),
        ]
    )
    counts = {(r["gate"], r["status"]): r["n"] for r in store.gate_counts()}
    store.close()

    # Without not_run, lint would look like a 0-sample gate rather than one
    # that was skipped because an earlier gate short-circuited routing.
    assert counts[("lint", "not_run")] == 1
    assert counts[("build", "not_run")] == 1
    assert counts[("test", "failed")] == 1


def test_telemetry_never_raises_on_bad_input(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    store.record_stage("LAB-30", "not_a_stage")
    store.record_gates([])
    store.increment("transcript_write_failures", 2)
    counters = store.counters()
    store.close()

    assert counters["transcript_write_failures"] == 2


def test_telemetry_iteration_rows_upsert(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.sqlite3")
    store.record_iteration(IterationRecord("g1", "implement", 1, outcome="retry"))
    store.record_iteration(IterationRecord("g1", "implement", 1, outcome="success"))
    totals = store.iteration_totals()
    store.close()

    assert len(totals) == 1
    assert totals[0]["iterations"] == 1


def test_null_telemetry_recorder_is_inert():
    recorder = NullTelemetryRecorder()
    recorder.record_stage("LAB-30", "claimed")
    recorder.record_escalation("LAB-30", "reason")
    recorder.record_gates([GateRecord("LAB-30", 1, "test", "passed")])
    recorder.record_iteration(IterationRecord("g1", "implement", 1))
    recorder.increment("x")


# -- import hygiene --------------------------------------------------------


def test_observability_modules_import_standalone():
    """Guard against reintroducing an import cycle.

    telemetry.py once imported the shared SQLite helper from inside the
    ``locking`` package, whose ``__init__`` pulls in the reconciler, jira, and
    orchestrator -- which imports telemetry back. It only failed when
    telemetry happened to be imported first, so the full suite stayed green
    while a plain ``import ticket_agent.observability.telemetry`` blew up.
    """

    import subprocess
    import sys

    for module in (
        "ticket_agent.observability.telemetry",
        "ticket_agent.observability.transcripts",
        "ticket_agent.sqlite_support",
    ):
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{module} failed to import: {result.stderr}"

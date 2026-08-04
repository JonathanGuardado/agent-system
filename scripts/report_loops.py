"""Report loop telemetry: funnel, gate outcomes, and observability health.

Usage:
    python scripts/report_loops.py [--data-dir DIR] [--since ISO8601] [--json]

Reports the metric set the roadmap will eventually grade the system on, even
while most values are zero -- so the numbers exist before there is pressure to
make them look good. Two are worth reading carefully:

``not_run`` gate counts distinguish "the build gate failed" from "the build
gate never ran because tests failed first". Without that split, a pass rate
computed over executed gates only is measuring a biased sample.

``transcript_write_failures`` is dropped observability. A nonzero value means
the transcripts are incomplete and any conclusion drawn from them is suspect.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ticket_agent.observability.telemetry import (  # noqa: E402
    STAGES,
    SQLiteTelemetryStore,
)

DEFAULT_DATA_DIR = REPO_ROOT / ".agent-system-data"
TELEMETRY_DB = "loop_telemetry.sqlite3"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.data_dir) / TELEMETRY_DB
    if not db_path.exists():
        print(f"no telemetry database at {db_path}", file=sys.stderr)
        print("run with AGENT_SYSTEM_TRANSCRIPTS_ENABLED=true to collect it")
        return 1

    store = SQLiteTelemetryStore(db_path)
    try:
        report = _build_report(store, since=args.since)
    finally:
        store.close()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def _build_report(store: SQLiteTelemetryStore, *, since: str | None) -> dict[str, Any]:
    funnel = store.funnel_counts(since=since)
    claimed = funnel.get("claimed", 0)

    gates: dict[str, dict[str, int]] = {}
    for row in store.gate_counts():
        gates.setdefault(str(row["gate"]), {})[str(row["status"])] = int(row["n"])

    return {
        "funnel": funnel,
        "conversion": _conversion(funnel, claimed),
        "gates": gates,
        "escalations": {
            str(row["reason"]): int(row["n"]) for row in store.escalation_reasons()
        },
        "iterations": [dict(row) for row in store.iteration_totals()],
        "counters": store.counters(),
    }


def _conversion(funnel: dict[str, int], claimed: int) -> dict[str, float | None]:
    # Every rate is relative to tickets claimed, so stages are comparable to
    # each other rather than to whichever stage happened to precede them.
    if not claimed:
        return dict.fromkeys(STAGES)
    return {stage: round(funnel.get(stage, 0) / claimed, 4) for stage in STAGES}


def _print_report(report: dict[str, Any]) -> None:
    print("== funnel ==")
    for stage in STAGES:
        count = report["funnel"].get(stage, 0)
        rate = report["conversion"].get(stage)
        suffix = "" if rate is None else f"  ({rate:.0%} of claimed)"
        print(f"  {stage:<12} {count:>6}{suffix}")

    print("\n== gates ==")
    if not report["gates"]:
        print("  (none recorded)")
    for gate, statuses in sorted(report["gates"].items()):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
        print(f"  {gate:<12} {parts}")

    print("\n== escalations ==")
    if not report["escalations"]:
        print("  (none)")
    for reason, count in sorted(
        report["escalations"].items(), key=lambda kv: -kv[1]
    ):
        # Group by prefix: reasons carry a "code: detail" shape and the code
        # is the part worth counting.
        print(f"  {count:>4}  {reason.split(':')[0]}")

    counters = report["counters"]
    print("\n== observability health ==")
    failures = counters.get("transcript_write_failures", 0)
    print(f"  transcript_write_failures  {failures}")
    if failures:
        print("  WARNING: transcripts are incomplete; treat loop analysis as partial")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report agent-system loop telemetry.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="directory holding the telemetry database",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp; only count stages reached at or after it",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

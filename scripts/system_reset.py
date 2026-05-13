from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ticket_agent.app import load_app_config
from ticket_agent.jira.client import JiraRestClient
from ticket_agent.jira.constants import (
    FIELD_AGENT_ASSIGNED_COMPONENT,
    FIELD_AGENT_RETRY_COUNT,
    FIELD_MAX_ATTEMPTS,
    LABEL_AI_CLAIMED,
    LABEL_AI_EXECUTION_APPROVED,
    LABEL_AI_FAILED,
    LABEL_AI_READY,
)

DEFAULT_DATA_DIR = REPO_ROOT / ".agent-system-data"
AUTOMATION_LABELS = (
    LABEL_AI_READY,
    LABEL_AI_CLAIMED,
    LABEL_AI_FAILED,
    LABEL_AI_EXECUTION_APPROVED,
)
LOCAL_TABLES = {
    "execution_approvals.sqlite3": ("execution_approvals", "ticket_key"),
    "ticket_locks.sqlite3": ("ticket_locks", "ticket_key"),
    "ticket_graph_checkpoints.sqlite3": ("checkpoints", "thread_id"),
    "ticket_graph_checkpoints.sqlite3#checkpoint_writes": (
        "checkpoint_writes",
        "thread_id",
    ),
}
PROPOSAL_TABLE = ("intake_proposals.sqlite3", "active_proposals")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.ticket and not args.jql and not args.all_local:
        print("Nothing selected. Use --ticket, --jql, or --all-local.", file=sys.stderr)
        return 2
    if args.delete_jira_issues and not args.confirm:
        print("--delete-jira-issues requires --confirm.", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    tickets = _ordered_unique(args.ticket)
    if args.jql:
        tickets.extend(asyncio.run(_ticket_keys_from_jql(args.jql)))
        tickets = _ordered_unique(tickets)

    dry_run = not args.confirm
    print("ticket-agent system reset")
    print(f"mode: {'dry-run' if dry_run else 'confirmed'}")
    if tickets:
        print(f"tickets: {', '.join(tickets)}")
    if args.all_local:
        print("local sqlite: all rows selected")

    local_report = _reset_local_state(
        data_dir=data_dir,
        tickets=tickets,
        all_local=args.all_local,
        include_proposals=args.include_proposals,
        dry_run=dry_run,
    )
    for line in local_report:
        print(line)

    if tickets and not args.skip_jira:
        asyncio.run(
            _reset_jira(
                tickets=tickets,
                dry_run=dry_run,
                delete_issues=args.delete_jira_issues,
                transition_to=args.transition_to,
                clear_fields=args.clear_fields,
            )
        )

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reset ticket-agent local SQLite runtime state and optional Jira "
            "automation markers. Dry-run by default."
        )
    )
    parser.add_argument(
        "--ticket",
        action="append",
        default=[],
        help="Jira ticket key to reset. Can be provided more than once.",
    )
    parser.add_argument(
        "--jql",
        help="Jira JQL whose matching issue keys should be reset.",
    )
    parser.add_argument(
        "--all-local",
        action="store_true",
        help="Delete all local SQLite runtime rows, including all tickets.",
    )
    parser.add_argument(
        "--include-proposals",
        action="store_true",
        help="When used with --all-local, also delete all intake proposals.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Runtime data directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--skip-jira",
        action="store_true",
        help="Only clean local SQLite state; do not call Jira.",
    )
    parser.add_argument(
        "--clear-fields",
        action="store_true",
        help=(
            "Best-effort clear of automation custom fields. Jira may reject this "
            "if fields are not on the screen."
        ),
    )
    parser.add_argument(
        "--transition-to",
        help="Optional Jira status name to transition matching tickets to.",
    )
    parser.add_argument(
        "--delete-jira-issues",
        action="store_true",
        help="Delete matching Jira issues. Requires --confirm.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform mutations. Without this flag, the script is dry-run.",
    )
    return parser.parse_args(argv)


def _reset_local_state(
    *,
    data_dir: Path,
    tickets: Sequence[str],
    all_local: bool,
    include_proposals: bool,
    dry_run: bool,
) -> list[str]:
    report: list[str] = []
    for db_key, (table, key_column) in LOCAL_TABLES.items():
        db_name = db_key.split("#", maxsplit=1)[0]
        db_path = data_dir / db_name
        if all_local:
            report.append(
                _delete_all_rows(db_path, table, dry_run=dry_run)
            )
            continue
        for ticket in tickets:
            report.append(
                _delete_rows(
                    db_path,
                    table,
                    f"{key_column} = ?",
                    (ticket,),
                    dry_run=dry_run,
                )
            )

    if include_proposals:
        db_name, table = PROPOSAL_TABLE
        db_path = data_dir / db_name
        if all_local:
            report.append(_delete_all_rows(db_path, table, dry_run=dry_run))
        else:
            report.append(
                "intake_proposals.sqlite3: skipped proposals "
                "(targeted proposal cleanup needs --all-local --include-proposals)"
            )
    return report


def _delete_all_rows(db_path: Path, table: str, *, dry_run: bool) -> str:
    if not db_path.exists():
        return f"{db_path.name}.{table}: database missing"
    count = _count_rows(db_path, table)
    if not dry_run:
        with sqlite3.connect(db_path) as con:
            con.execute(f"DELETE FROM {table}")
            con.commit()
    return f"{db_path.name}.{table}: {'would delete' if dry_run else 'deleted'} {count}"


def _delete_rows(
    db_path: Path,
    table: str,
    where_sql: str,
    params: tuple[object, ...],
    *,
    dry_run: bool,
) -> str:
    if not db_path.exists():
        return f"{db_path.name}.{table}: database missing"
    count = _count_rows(db_path, table, where_sql, params)
    if not dry_run:
        with sqlite3.connect(db_path) as con:
            con.execute(f"DELETE FROM {table} WHERE {where_sql}", params)
            con.commit()
    return f"{db_path.name}.{table}: {'would delete' if dry_run else 'deleted'} {count}"


def _count_rows(
    db_path: Path,
    table: str,
    where_sql: str | None = None,
    params: tuple[object, ...] = (),
) -> int:
    query = f"SELECT COUNT(*) FROM {table}"
    if where_sql is not None:
        query += f" WHERE {where_sql}"
    with sqlite3.connect(db_path) as con:
        row = con.execute(query, params).fetchone()
    return int(row[0] if row is not None else 0)


async def _ticket_keys_from_jql(jql: str) -> list[str]:
    client = _jira_client()
    result = await client.search_issues(jql, fields=["summary"])
    issues = result.get("issues", []) if isinstance(result, Mapping) else result
    keys: list[str] = []
    for issue in issues:
        if isinstance(issue, Mapping):
            key = issue.get("key")
            if isinstance(key, str) and key.strip():
                keys.append(key.strip())
    return _ordered_unique(keys)


async def _reset_jira(
    *,
    tickets: Sequence[str],
    dry_run: bool,
    delete_issues: bool,
    transition_to: str | None,
    clear_fields: bool,
) -> None:
    client = _jira_client()
    for ticket in tickets:
        if dry_run:
            print(f"jira {ticket}: would remove labels {list(AUTOMATION_LABELS)}")
            if clear_fields:
                print(f"jira {ticket}: would clear automation fields")
            if transition_to:
                print(f"jira {ticket}: would transition to {transition_to!r}")
            if delete_issues:
                print(f"jira {ticket}: would delete issue")
            continue

        try:
            await client.remove_labels(ticket, list(AUTOMATION_LABELS))
            print(f"jira {ticket}: removed automation labels")
        except Exception as exc:  # noqa: BLE001 - reset should keep going
            print(f"jira {ticket}: label cleanup failed: {_error(exc)}")

        if clear_fields:
            try:
                await client.update_fields(ticket, _empty_fields())
                print(f"jira {ticket}: cleared automation fields")
            except Exception as exc:  # noqa: BLE001 - Jira screens can reject fields
                print(f"jira {ticket}: field cleanup failed: {_error(exc)}")

        if transition_to:
            try:
                await client.transition_ticket(ticket, transition_to)
                print(f"jira {ticket}: transitioned to {transition_to!r}")
            except Exception as exc:  # noqa: BLE001 - reset should report all tickets
                print(f"jira {ticket}: transition failed: {_error(exc)}")

        if delete_issues:
            try:
                await client._request(  # noqa: SLF001 - script-only Jira reset path
                    "DELETE",
                    f"/rest/api/3/issue/{ticket}",
                    expect_json=False,
                )
                print(f"jira {ticket}: deleted issue")
            except Exception as exc:  # noqa: BLE001
                print(f"jira {ticket}: delete failed: {_error(exc)}")


def _jira_client() -> JiraRestClient:
    config = load_app_config()
    return JiraRestClient(
        base_url=config.jira_base_url,
        user_email=config.jira_user_email,
        api_key=config.jira_api_key,
        timeout_s=config.jira_timeout_s,
        field_map=config.jira_field_map,
    )


def _empty_fields() -> dict[str, object]:
    return {
        FIELD_AGENT_ASSIGNED_COMPONENT: None,
        FIELD_AGENT_RETRY_COUNT: None,
        FIELD_MAX_ATTEMPTS: None,
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _error(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

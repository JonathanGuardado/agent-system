"""Manual Slack/Jira vertical-slice smoke instructions."""

from __future__ import annotations

import os
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    intake_channel = os.environ.get("AGENT_SYSTEM_INTAKE_CHANNEL", "<intake-channel>")
    approval_channel = os.environ.get(
        "AGENT_SYSTEM_EXECUTION_APPROVAL_CHANNEL",
        intake_channel,
    )
    project = _first_project(os.environ.get("AGENT_SYSTEM_JIRA_TARGET_PROJECTS"))
    message = (
        f"Break this {project} request into Jira tickets for agent-system:\n"
        "- Add a tiny vertical-slice smoke marker\n"
        "- Verify the Slack/Jira handoff"
    )

    print("ticket-agent manual E2E smoke")
    print()
    print("Before starting:")
    print("- Set AGENT_SYSTEM_EXECUTION_MODE=dry_run for a safe first pass.")
    print("- Start the runtime with: ticket-agent")
    print()
    print("Manual steps:")
    print(f"1. In Slack channel {intake_channel}, post this message:")
    print(message)
    print("2. In the same Slack thread, reply:")
    print("approve")
    print("3. In Jira, wait for created Task issue(s) in the existing project.")
    print("4. Confirm at least one issue has labels ai-ready and then ai-claimed.")
    print(f"5. In Slack channel/thread {approval_channel}, wait for execution approval.")
    print("6. Reply with one of:")
    print(f"approve {project}-<number>")
    print(f"reject {project}-<number>")
    print()
    print("Check in Jira:")
    print("- Created ticket has repository, repo_path, Slack channel/thread, retry count, and capabilities fields when configured.")
    print("- Detection moves an eligible ticket to In Progress and adds ai-claimed.")
    print("- In dry_run mode, approve records a dry-run comment and releases the claim.")
    print("- In dry_run mode, reject marks failure, comments, and releases the claim.")
    print()
    print("Check in Slack:")
    print("- The intake thread shows the proposal and the created Jira ticket key(s).")
    print("- The execution approval message contains approve/reject TICKET-KEY examples.")
    print("- The final reply clearly says whether dry-run execution was approved or rejected.")
    return 0


def _first_project(raw: str | None) -> str:
    if not raw:
        return "AGENT"
    return next((part.strip().upper() for part in raw.split(",") if part.strip()), "AGENT")


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - console convenience
    raise SystemExit(main())

"""Hardcoded Jira execution values for the MVP integration boundary."""

LABEL_AI_READY = "ai-ready"
LABEL_AI_CLAIMED = "ai-claimed"
LABEL_AI_FAILED = "ai-failed"
LABEL_DO_NOT_AUTOMATE = "do-not-automate"

STATUS_TODO = "To Do"
STATUS_IN_PROGRESS = "In Progress"
STATUS_IN_REVIEW = "In Review"

FIELD_AGENT_ASSIGNED_COMPONENT = "agent_assigned_component"
FIELD_AGENT_RETRY_COUNT = "agent_retry_count"
FIELD_AGENT_CAPABILITIES_NEEDED = "agent_capabilities_needed"
FIELD_REPOSITORY = "repository"
FIELD_REPO_PATH = "repo_path"
FIELD_MAX_ATTEMPTS = "max_attempts"

__all__ = [
    "FIELD_AGENT_ASSIGNED_COMPONENT",
    "FIELD_AGENT_CAPABILITIES_NEEDED",
    "FIELD_AGENT_RETRY_COUNT",
    "FIELD_MAX_ATTEMPTS",
    "FIELD_REPOSITORY",
    "FIELD_REPO_PATH",
    "LABEL_AI_CLAIMED",
    "LABEL_AI_FAILED",
    "LABEL_AI_READY",
    "LABEL_DO_NOT_AUTOMATE",
    "STATUS_IN_PROGRESS",
    "STATUS_IN_REVIEW",
    "STATUS_TODO",
]

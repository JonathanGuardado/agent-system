"""Execution-domain data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Literal

CommandNetworkMode = Literal["none", "install"]


@dataclass(frozen=True, slots=True)
class CommandExecutionPolicy:
    """Adapter-independent limits declared for one repository command."""

    network: CommandNetworkMode = "none"
    writable_paths: tuple[str, ...] = ()
    cpu_seconds: int = 600
    memory_bytes: int = 2 * 1024 * 1024 * 1024
    max_processes: int = 512
    max_file_bytes: int = 512 * 1024 * 1024

    @property
    def digest(self) -> str:
        payload = {
            "cpu_seconds": self.cpu_seconds,
            "max_file_bytes": self.max_file_bytes,
            "max_processes": self.max_processes,
            "memory_bytes": self.memory_bytes,
            "network": self.network,
            "writable_paths": list(self.writable_paths),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SandboxAttestation:
    """Evidence emitted by the wrapper actually used for one command."""

    sandbox_profile: str
    command_policy_digest: str
    repository_root: str
    command_working_directory: str
    network_mode: CommandNetworkMode
    writable_mounts: tuple[str, ...]
    launch_digest: str


@dataclass(frozen=True)
class TicketLock:
    """A currently held lock for a Jira ticket."""

    ticket_key: str
    owner: str
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    lock_id: str | None = None

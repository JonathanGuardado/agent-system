"""Execution isolation for untrusted repository commands.

Tests, builds, and dependency installers run model-authored and
dependency-authored code. Without isolation they inherit the operator's
``PATH``, full filesystem read access, unrestricted network, and a shared
writable ``HOME`` -- and ``npm install`` alone executes arbitrary
``postinstall`` scripts from the whole dependency graph.

Mechanism, and why each piece is here rather than the obvious alternative:

**No ``preexec_fn``.** The obvious way to set rlimits is
``subprocess.Popen(preexec_fn=...)``. It is not async-signal-safe, and this
process runs asyncio TaskGroups plus ``asyncio.to_thread``; CPython documents
it as unsafe in the presence of threads, where it can deadlock the child. So
limits are applied by ``prlimit(1)`` *in the child's own argv*, after
``exec``, where thread-safety is not in play.

**Explicit ``--clearenv``.** Measured, not assumed: ``bwrap --unshare-all``
inherits the parent environment. A polluted parent leaked 14 credential-ish
variables into the sandbox. Isolation of credentials requires ``--clearenv``
plus an explicit allowlist.

**Read-only worktree with declared writable mounts.** Default-deny. A gate
that can rewrite the source it is testing is not a gate, so writability is
enumerated (build and dependency output) rather than granted wholesale.

**Resolver files bound only when the network is shared.** ``--share-net``
alone leaves DNS broken without ``/etc/resolv.conf``; binding them while
``--unshare-net`` is active still blocks the network, so they are safe to
bind unconditionally, but are bound only for install to keep the argv honest
about intent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Literal, Protocol

from ticket_agent.domain.errors import AgentSystemError

NetworkPolicy = Literal["none", "install"]

#: Read-only system paths every command needs to run at all.
_SYSTEM_ROBINDS: tuple[str, ...] = ("/usr", "/lib", "/lib64", "/bin", "/sbin")

#: Bound only when the network is shared, so name resolution works for
#: installs. Harmless with --unshare-net, which still blocks the network.
_RESOLVER_FILES: tuple[str, ...] = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ssl",
    "/etc/ca-certificates.conf",
)

#: The only variables a repository command may see. Everything else is
#: dropped, including anything matching TOKEN/SECRET/KEY/PASSWORD/COOKIE and
#: every GH_*/JIRA_*/SLACK_*/DEEPSEEK_*/GEMINI_*/AWS_* variable.
ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "HOME", "LANG", "LC_ALL", "CI", "TERM")

_SANDBOX_HOME = "/tmp/agent-home"


class SandboxUnavailableError(AgentSystemError):
    """Raised when a sandbox was required but cannot be constructed."""


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Limits applied to one sandboxed command."""

    network: NetworkPolicy = "none"
    cpu_seconds: int = 600
    memory_bytes: int = 2 * 1024 * 1024 * 1024
    max_processes: int = 512
    max_file_bytes: int = 512 * 1024 * 1024
    #: Paths inside the worktree the command may write. The worktree itself is
    #: mounted read-only; these are re-bound writable on top of it.
    writable_paths: tuple[Path, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def profile(self) -> str:
        """Short stable string recorded in the verification policy."""

        return (
            f"bwrap:net={self.network};cpu={self.cpu_seconds};"
            f"mem={self.memory_bytes};nproc={self.max_processes}"
        )


class Sandbox(Protocol):
    def wrap(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        policy: SandboxPolicy,
    ) -> tuple[str, ...]:
        """Return the argv that runs ``argv`` under this sandbox."""

    @property
    def profile(self) -> str:
        """Identifier recorded alongside verification evidence."""


class NullSandbox:
    """Passthrough. Permitted only when a human supervises each command."""

    __slots__ = ()

    def wrap(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        policy: SandboxPolicy,
    ) -> tuple[str, ...]:
        return tuple(argv)

    @property
    def profile(self) -> str:
        return "none"


class BubblewrapSandbox:
    """Namespace isolation via ``bwrap``, with rlimits applied post-exec."""

    def __init__(self, bwrap_path: str = "bwrap", prlimit_path: str = "prlimit") -> None:
        self._bwrap = bwrap_path
        self._prlimit = prlimit_path

    @property
    def profile(self) -> str:
        return "bwrap"

    @staticmethod
    def available(bwrap_path: str = "bwrap") -> bool:
        """Whether a sandbox can actually be created here.

        Deliberately attempts a real unshare rather than checking that the
        binary exists or running ``--version``. On Ubuntu 24.04
        ``kernel.apparmor_restrict_unprivileged_userns=1`` lets a perfectly
        healthy ``bwrap`` fail at ``unshare(CLONE_NEWUSER)``, so a
        presence check reports such a host as sandbox-ready when it is not.
        """

        if shutil.which(bwrap_path) is None:
            return False
        # The probe must bind exactly what wrap() binds. Binding a smaller set
        # fails for an unrelated reason -- on Ubuntu /lib64 is a symlink into
        # /usr, so omitting it leaves the dynamic loader unresolvable and even
        # /bin/true cannot exec -- which would report a working host as
        # unavailable. Sharing _SYSTEM_ROBINDS keeps probe and production from
        # drifting apart.
        probe = [bwrap_path, "--unshare-all"]
        for system_path in _SYSTEM_ROBINDS:
            if Path(system_path).exists():
                probe += ["--ro-bind", system_path, system_path]
        probe += [
            "--proc", "/proc",
            "--dev", "/dev",
            "--die-with-parent",
            "--new-session",
            "--",
            "/bin/true",
        ]
        try:
            result = subprocess.run(
                probe,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def wrap(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        policy: SandboxPolicy,
    ) -> tuple[str, ...]:
        if not argv:
            raise SandboxUnavailableError("cannot sandbox an empty command")

        worktree = Path(cwd).resolve()
        command: list[str] = [self._bwrap, "--unshare-all"]

        if policy.network == "install":
            command.append("--share-net")

        command += [
            "--die-with-parent",
            # Blocks TIOCSTI terminal-injection back into the parent's tty.
            "--new-session",
            # Measured, not assumed: without this the parent environment is
            # inherited wholesale.
            "--clearenv",
        ]

        for env_name, env_value in self._child_env(policy).items():
            command += ["--setenv", env_name, env_value]

        for system_path in _SYSTEM_ROBINDS:
            if Path(system_path).exists():
                command += ["--ro-bind", system_path, system_path]

        if policy.network == "install":
            for resolver_path in _RESOLVER_FILES:
                if Path(resolver_path).exists():
                    command += ["--ro-bind", resolver_path, resolver_path]
        elif Path("/etc/ssl").exists():
            command += ["--ro-bind", "/etc/ssl", "/etc/ssl"]

        command += ["--tmpfs", "/tmp", "--dir", _SANDBOX_HOME]

        # Default deny: the worktree is read-only, and only declared paths
        # are re-bound writable on top of it.
        command += ["--ro-bind", str(worktree), str(worktree)]
        for writable in policy.writable_paths:
            resolved = Path(writable)
            if not resolved.is_absolute():
                resolved = worktree / resolved
            resolved = resolved.resolve()
            if not _is_within(resolved, worktree):
                raise SandboxUnavailableError(
                    f"writable path escapes the worktree: {resolved}"
                )
            resolved.mkdir(parents=True, exist_ok=True)
            command += ["--bind", str(resolved), str(resolved)]

        command += ["--proc", "/proc", "--dev", "/dev", "--chdir", str(worktree), "--"]
        command += self._rlimit_prefix(policy)
        command += list(argv)
        return tuple(command)

    def _child_env(self, policy: SandboxPolicy) -> dict[str, str]:
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": _SANDBOX_HOME,
            "LANG": "C.UTF-8",
            "CI": "true",
        }
        for key, value in policy.env.items():
            if key in ENV_ALLOWLIST:
                env[key] = value
        return env

    def _rlimit_prefix(self, policy: SandboxPolicy) -> list[str]:
        """Apply rlimits in the child's argv, never via ``preexec_fn``."""

        return [
            self._prlimit,
            f"--cpu={policy.cpu_seconds}",
            f"--as={policy.memory_bytes}",
            f"--nproc={policy.max_processes}",
            f"--fsize={policy.max_file_bytes}",
            "--core=0",
            "--",
        ]


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def build_sandbox(*, required: bool) -> Sandbox:
    """Return the best available sandbox.

    ``required=True`` refuses to degrade: unattended execution of repository
    commands without isolation is the thing this module exists to prevent, so
    it raises rather than silently returning a passthrough.
    """

    if BubblewrapSandbox.available():
        return BubblewrapSandbox()
    if required:
        raise SandboxUnavailableError(
            "bubblewrap is unavailable or cannot create a user namespace; "
            "unattended repository commands are forbidden without isolation"
        )
    return NullSandbox()


__all__ = [
    "ENV_ALLOWLIST",
    "BubblewrapSandbox",
    "NetworkPolicy",
    "NullSandbox",
    "Sandbox",
    "SandboxPolicy",
    "SandboxUnavailableError",
    "build_sandbox",
]

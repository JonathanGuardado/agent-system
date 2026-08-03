"""Signing for goal authorization and delivery evidence.

What this buys, stated precisely so nobody later assumes more:

Merge authority ends up resting on rows in a local SQLite file. Anything with
write access to the data directory can edit one. Signing makes that edit
**loud** -- a forged or altered record fails verification and parks the run.

What it does **not** buy: prevention. On a single-host deployment the agent
runs as the operator's user, so a compromised agent process can read the key
and mint valid signatures. This is tamper-*evidence*, not tamper-*prevention*.
Real prevention needs a separate signing principal, which is a deployment
change rather than a code change. Do not let a later reader infer otherwise
from the presence of an HMAC.

The key therefore lives outside the data directory it protects, is required to
be mode 0600, and is excluded from the sandbox environment allowlist and from
every transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import os
from pathlib import Path
import stat
from typing import Any

from ticket_agent.domain.errors import AgentSystemError
from ticket_agent.goal.types import canonical_json

#: Bumped if the signed representation ever changes, so old signatures are
#: rejected rather than silently misread under new rules.
SIGNATURE_VERSION = 1

_MIN_KEY_BYTES = 32


class SigningError(AgentSystemError):
    """Raised when a signing key is unusable or a signature does not verify."""


@dataclass(frozen=True, slots=True)
class Signature:
    version: int
    algorithm: str
    value: str

    def __str__(self) -> str:
        return f"{self.version}:{self.algorithm}:{self.value}"

    @classmethod
    def parse(cls, text: str) -> Signature:
        parts = str(text).split(":")
        if len(parts) != 3:
            raise SigningError(f"malformed signature: {text!r}")
        version, algorithm, value = parts
        if not version.isdigit():
            raise SigningError(f"malformed signature version: {text!r}")
        return cls(version=int(version), algorithm=algorithm, value=value)


class Signer:
    """HMAC-SHA256 over the canonical serialization of a payload."""

    def __init__(self, key: bytes) -> None:
        if len(key) < _MIN_KEY_BYTES:
            raise SigningError(
                f"signing key must be at least {_MIN_KEY_BYTES} bytes; got {len(key)}"
            )
        self._key = key

    def sign(self, payload: Any) -> Signature:
        return Signature(
            version=SIGNATURE_VERSION,
            algorithm="hmac-sha256",
            value=self._digest(payload),
        )

    def verify(self, payload: Any, signature: Signature | str) -> bool:
        """Constant-time verification. Returns False rather than raising on
        an unknown version or algorithm, so a caller cannot accidentally treat
        'could not check' as 'verified'."""

        try:
            parsed = (
                signature
                if isinstance(signature, Signature)
                else Signature.parse(signature)
            )
        except SigningError:
            return False
        if parsed.version != SIGNATURE_VERSION or parsed.algorithm != "hmac-sha256":
            return False
        return hmac.compare_digest(parsed.value, self._digest(payload))

    def _digest(self, payload: Any) -> str:
        message = canonical_json(payload).encode("utf-8")
        return hmac.new(self._key, message, sha256).hexdigest()


class NullSigner:
    """Refuses to sign. The default, so signing must be configured on purpose.

    It does not silently produce unverifiable records -- an unsigned
    authorization would otherwise look identical to one whose key was never
    set up.
    """

    __slots__ = ()

    def sign(self, payload: Any) -> Signature:
        raise SigningError(
            "no signing key configured; set AGENT_SYSTEM_SIGNING_KEY_PATH to a "
            "mode-0600 file outside the data directory"
        )

    def verify(self, payload: Any, signature: Signature | str) -> bool:
        return False


def load_signer(
    key_path: str | Path | None,
    *,
    data_dir: Path | None = None,
    require_permissions: bool = True,
) -> Signer | NullSigner:
    """Load a signing key, refusing configurations that defeat the point."""

    if key_path is None:
        return NullSigner()

    path = Path(key_path).expanduser()
    if not path.is_file():
        raise SigningError(f"signing key file does not exist: {path}")

    if data_dir is not None:
        try:
            path.resolve().relative_to(Path(data_dir).resolve())
        except ValueError:
            pass
        else:
            # A key stored inside the directory it protects is readable by
            # anything that can already forge the rows.
            raise SigningError(
                f"signing key must live outside the data directory: {path}"
            )

    if require_permissions:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise SigningError(
                f"signing key must not be group- or world-accessible "
                f"(found mode {mode:o}): {path}"
            )

    key = path.read_bytes().strip()
    if not key:
        raise SigningError(f"signing key file is empty: {path}")
    return Signer(key)


def generate_key() -> bytes:
    """A fresh 32-byte key, hex-encoded for safe storage in a text file."""

    return os.urandom(32).hex().encode("ascii")


__all__ = [
    "SIGNATURE_VERSION",
    "NullSigner",
    "Signature",
    "Signer",
    "SigningError",
    "generate_key",
    "load_signer",
]

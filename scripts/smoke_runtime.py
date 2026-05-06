from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ticket_agent.runtime_smoke import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

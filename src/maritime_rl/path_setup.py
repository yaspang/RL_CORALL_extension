# help make clean imports, no matter environment running on 

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
CORALL_SRC = ROOT / "third_party" / "CORALL" / "src"
MY_SRC = ROOT / "src"

def ensure_paths() -> None:
    for p in [str(CORALL_SRC), str(MY_SRC)]:
        if p not in sys.path:
            sys.path.insert(0, p)

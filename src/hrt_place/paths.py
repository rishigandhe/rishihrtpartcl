"""Repo layout: challenge deps and `src/` root for imports."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
HRT_SRC: Path = _PKG.parent
REPO_ROOT: Path = HRT_SRC.parent

DEPS_ICCAD = (
    REPO_ROOT
    / ".deps"
    / "macro-place-challenge-2026"
    / "external"
    / "MacroPlacement"
    / "Testcases"
    / "ICCAD04"
)


def ensure_src_on_path() -> None:
    """So `hrt_place` and top-level `proxy_refine` resolve when the harness loads a file under `src/`."""
    s = str(HRT_SRC)
    if s not in sys.path:
        sys.path.insert(0, s)

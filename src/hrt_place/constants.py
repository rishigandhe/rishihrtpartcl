"""Oracle refinement targets (tail / high-proxy designs)."""

from __future__ import annotations

# Match `proxy_refine.HARD_SUITE`.
PROXY_REFINE_HARD = frozenset(
    {
        "ibm18",
        "ibm17",
        "ibm16",
        "ibm15",
        "ibm12",
        "ibm10",
        "ibm14",
        "ibm08",
        "ibm06",
        "ibm02",
        "ibm04",
    }
)


def proxy_refine_budget_for_tier(base: int, tier: int) -> int:
    mult = 1.0 if tier == 0 else (2.0 if tier == 1 else 3.0)
    return min(max(8, int(round(base * mult))), 140)

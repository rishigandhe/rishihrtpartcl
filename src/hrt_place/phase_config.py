"""
Tunable Phase-1/3 weights — DreamPlace-class tools use different formulations
(Poisson density, Nesterov, HPWL/LSE schedules); these knobs nudge our surrogate
closer to the judge proxy without replacing the whole stack.
"""

from __future__ import annotations

# Fraction of Phase-1 length before congestion ramp begins (0 → full by last iter).
# Lower = routability enters earlier (DreamPlace optimizes density+WL jointly for longer).
CONGESTION_RAMP_START_FRAC = 0.52

# Congestion multiplier in Adam (after ramp) and fixed weight in L-BFGS — tiered by stress.
TIER_CONGESTION_WEIGHT = (0.068, 0.175, 0.28)

# Soft refine (Phase 3): light congestion so moving soft macros does not ignore judge cong.
TIER_CONGESTION_PHASE3 = (0.042, 0.095, 0.145)


def tier_cong_w_phase1(tier: int) -> float:
    return TIER_CONGESTION_WEIGHT[int(tier)]


def tier_cong_w_soft_refine(tier: int) -> float:
    return TIER_CONGESTION_PHASE3[int(tier)]

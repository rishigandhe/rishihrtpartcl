"""
Step 1: post-pass hill climbing on the *true* proxy (PlacementCost via compute_proxy_cost).

Run standalone to A/B a benchmark. Each oracle call can take multiple seconds on
large designs (e.g. ~2–3s per eval on ibm01), so keep --budget modest unless you
are willing to wait.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = ROOT_DIR / ".deps" / "macro-place-challenge-2026"
TESTCASE_ROOT = CHALLENGE_DIR / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
PLACER_PATH = ROOT_DIR / "src" / "analytical_placer.py"

# High-proxy / heavy-runtime cases from a full IBM sweep (order: worst proxy first).
HARD_SUITE = [
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
]


def _ensure_import_path():
    p = str(CHALLENGE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_analytical_module():
    spec = importlib.util.spec_from_file_location("analytical_placer", str(PLACER_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    placer_cls = None
    for attr in vars(mod).values():
        if isinstance(attr, type) and callable(getattr(attr, "place", None)):
            placer_cls = attr
            break
    if placer_cls is None:
        raise RuntimeError("No placer class in analytical_placer.py")
    return mod, placer_cls


def _refine_budget_for_tier(base: int, tier: int) -> int:
    """More oracle trials on stress-tier designs (each eval is expensive)."""
    mult = 1.0 if tier == 0 else (2.0 if tier == 1 else 3.0)
    return min(max(8, int(round(base * mult))), 140)


def _movable_indices(benchmark) -> torch.Tensor:
    """Indices of macros we may move (not fixed)."""
    fix = benchmark.macro_fixed
    return (~fix).nonzero(as_tuple=False).view(-1)


def _clamp_centers(pos: torch.Tensor, benchmark) -> torch.Tensor:
    """Clamp macro centers inside canvas with half-width margins."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hw = benchmark.macro_sizes[:, 0] * 0.5
    hh = benchmark.macro_sizes[:, 1] * 0.5
    out = pos.clone()
    out[:, 0] = out[:, 0].clamp(hw, cw - hw)
    out[:, 1] = out[:, 1].clamp(hh, ch - hh)
    return out


def refine_placement(
    placement: torch.Tensor,
    benchmark,
    plc,
    *,
    max_evals: int = 15,
    step_frac: float = 0.004,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Hill-climb on proxy_cost with random coordinate probes.

    Returns:
        (best_placement_float32_cpu, stats_dict)
    """
    _ensure_import_path()
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    device = placement.device
    pos = placement.detach().float().cpu()
    pos = _clamp_centers(pos, benchmark)

    movable = _movable_indices(benchmark)
    if movable.numel() == 0:
        c0 = compute_proxy_cost(pos, benchmark, plc)
        return pos.to(device), {"evals": 1, "skipped": True, "proxy_before": c0["proxy_cost"]}

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    diag = math.hypot(cw, ch)
    step = max(diag * step_frac, 0.05)

    g = torch.Generator()
    g.manual_seed(seed)

    def score(p):
        return compute_proxy_cost(p, benchmark, plc)

    cur = pos.clone()
    c = score(cur)
    best_p = cur.clone()
    best_proxy = float(c["proxy_cost"])
    best_costs = dict(c)
    evals = 1

    t_ref0 = time.time()
    for it in range(max_evals):
        idx = int(movable[torch.randint(0, movable.numel(), (1,), generator=g).item()].item())
        axis = int(torch.randint(0, 2, (1,), generator=g).item())
        delta = step if torch.rand((), generator=g).item() < 0.5 else -step
        trial = cur.clone()
        trial[idx, axis] += delta
        trial = _clamp_centers(trial, benchmark)
        evals += 1
        ct = score(trial)
        ok, _viol = validate_placement(trial, benchmark, check_overlaps=False)
        if not ok:
            continue
        if int(ct["overlap_count"]) != 0:
            continue
        proxy = float(ct["proxy_cost"])
        if proxy < best_proxy - 1e-9:
            best_proxy = proxy
            best_p = trial.clone()
            best_costs = dict(ct)
            cur = trial

        if verbose and ((it + 1) % max(1, max_evals // 5) == 0 or it + 1 == max_evals):
            elapsed = time.time() - t_ref0
            print(
                f"    [proxy_refine] {it + 1}/{max_evals} evals  "
                f"best_proxy={best_proxy:.4f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    return best_p.to(device), {
        "evals": evals,
        "proxy_before": float(c["proxy_cost"]),
        "proxy_after": best_proxy,
        "best": best_costs,
    }


def main():
    parser = argparse.ArgumentParser(description="Placer + proxy hill climb (oracle refinement)")
    parser.add_argument("-b", "--benchmark", default=None, help="Single benchmark name")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Run several benchmarks in sequence",
    )
    parser.add_argument(
        "--suite",
        choices=("hard",),
        default=None,
        help="Run the pre-defined HARD_SUITE (worst historical proxy / heavy cases)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=12,
        help="Base max proxy evaluations in refine (scaled up on stress-tier designs)",
    )
    parser.add_argument("--step-frac", type=float, default=0.004, help="Step size as fraction of canvas diagonal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placer-verbose", action="store_true")
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable analytical placer stress-tier schedule (for A/B)",
    )
    parser.add_argument(
        "--refine-verbose",
        action="store_true",
        help="Print progress during oracle hill climb",
    )
    args = parser.parse_args()

    if not TESTCASE_ROOT.exists():
        print(f"Missing {TESTCASE_ROOT}. Run: bash scripts/setup_benchmark_env.sh")
        sys.exit(1)

    if args.suite is not None:
        names = list(HARD_SUITE)
    elif args.benchmarks is not None:
        names = list(args.benchmarks)
    elif args.benchmark is not None:
        names = [args.benchmark]
    else:
        names = ["ibm01"]

    _ensure_import_path()
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    mod, Placer = _load_analytical_module()
    stress_tier = mod.benchmark_stress_tier

    rows = []
    for bi, name in enumerate(names):
        bm_dir = str(TESTCASE_ROOT / name)
        print(f"\n=== [{bi + 1}/{len(names)}] {name} ===", flush=True)
        benchmark, plc = load_benchmark_from_dir(bm_dir)
        tier = stress_tier(benchmark)
        budget = _refine_budget_for_tier(args.budget, tier)
        print(f"  stress_tier={tier}  refine_budget={budget}", flush=True)

        placer = Placer(
            seed=args.seed + bi,
            verbose=args.placer_verbose,
            adaptive_hard=not args.no_adaptive,
        )

        t0 = time.time()
        print("Running analytical placer...", flush=True)
        p0 = placer.place(benchmark)
        t1 = time.time()

        c0 = compute_proxy_cost(p0, benchmark, plc)
        print(
            f"  After placer: proxy={c0['proxy_cost']:.4f}  "
            f"wl={c0['wirelength_cost']:.4f}  den={c0['density_cost']:.4f}  "
            f"cong={c0['congestion_cost']:.4f}  overlaps={c0['overlap_count']}  "
            f"time={t1 - t0:.2f}s",
            flush=True,
        )

        print(
            f"Refining (base_budget={args.budget} -> {budget}, step_frac={args.step_frac})...",
            flush=True,
        )
        t2 = time.time()
        _p1, stats = refine_placement(
            p0,
            benchmark,
            plc,
            max_evals=budget,
            step_frac=args.step_frac,
            seed=args.seed + bi + 17,
            verbose=args.refine_verbose,
        )
        t3 = time.time()

        if stats.get("skipped"):
            print("  No movable macros; skip refine.", flush=True)
            rows.append((name, tier, c0["proxy_cost"], c0["proxy_cost"], 0.0, t1 - t0, 0.0))
            continue

        gain = stats["proxy_before"] - stats["proxy_after"]
        c1 = stats["best"]
        print(
            f"  After refine: proxy={stats['proxy_after']:.4f}  "
            f"wl={c1['wirelength_cost']:.4f}  den={c1['density_cost']:.4f}  "
            f"cong={c1['congestion_cost']:.4f}  overlaps={c1['overlap_count']}  "
            f"evals={stats['evals']}  refine_time={t3 - t2:.2f}s",
            flush=True,
        )
        print(
            f"  Delta proxy: {gain:+.4f} ({100.0 * gain / max(stats['proxy_before'], 1e-9):+.2f}%)",
            flush=True,
        )
        rows.append(
            (
                name,
                tier,
                float(stats["proxy_before"]),
                float(stats["proxy_after"]),
                float(gain),
                t1 - t0,
                t3 - t2,
            )
        )

    if len(rows) > 1:
        print("\n--- summary ---", flush=True)
        print(f"{'bench':>8}  tier  {'proxy0':>8}  {'proxy1':>8}  {'gain':>8}  {'place_s':>8}  {'ref_s':>8}", flush=True)
        for name, tier, p0v, p1v, gain, ts, rs in rows:
            print(
                f"{name:>8}  {tier:^4}  {p0v:8.4f}  {p1v:8.4f}  {gain:8.4f}  {ts:8.1f}  {rs:8.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

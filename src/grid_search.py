"""
Hyperparameter grid search for AnalyticalPlacer.
"""

import argparse
import csv
import importlib.util
import itertools
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = ROOT_DIR / ".deps" / "macro-place-challenge-2026"
TESTCASE_ROOT = CHALLENGE_DIR / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

GRID = {
    "dw_end": [1.5, 3.0, 5.0, 8.0],
    "dw_start": [0.005, 0.02, 0.05],
}

DEFAULT_BENCHMARKS = ["ibm01", "ibm09", "ibm17"]


def _load_placer_class():
    path = ROOT_DIR / "src" / "analytical_placer.py"
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in vars(mod).values():
        if isinstance(attr, type) and callable(getattr(attr, "place", None)):
            return attr
    raise RuntimeError("No placer class found in src/analytical_placer.py")


def _ensure_challenge_import_path():
    challenge_path = str(CHALLENGE_DIR)
    if challenge_path not in sys.path:
        sys.path.insert(0, challenge_path)


def _load_benchmark(name):
    from macro_place.loader import load_benchmark_from_dir
    return load_benchmark_from_dir(str(TESTCASE_ROOT / name))


def _evaluate(placer, benchmark, plc):
    from macro_place.objective import compute_proxy_cost
    placement = placer.place(benchmark)
    return compute_proxy_cost(placement, benchmark, plc)


def run_grid_search(benchmarks, output_path):
    _ensure_challenge_import_path()
    placer_class = _load_placer_class()

    print("Loading benchmarks...")
    bench_data = {}
    for name in benchmarks:
        print(f"  {name}...", end=" ", flush=True)
        bm, plc = _load_benchmark(name)
        bench_data[name] = (bm, plc)
        print("ok")

    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\nGrid: {' x '.join(f'{k}({len(v)})' for k, v in GRID.items())} = {total} configs")
    print(f"Benchmarks: {benchmarks}")
    print(f"Total runs: {total * len(benchmarks)}\n")

    results = []
    best_avg = float("inf")
    best_cfg = None

    for cfg_idx, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        if cfg["dw_start"] >= cfg["dw_end"]:
            continue

        row = {"config": cfg_idx, **cfg}
        proxy_scores = []

        t_cfg = time.time()
        for name in benchmarks:
            bm, plc = bench_data[name]
            placer = placer_class(
                dw_start=cfg["dw_start"],
                dw_end=cfg["dw_end"],
                verbose=False,
            )
            t0 = time.time()
            costs = _evaluate(placer, bm, plc)
            elapsed = time.time() - t0

            proxy = costs["proxy_cost"]
            overlaps = costs["overlap_count"]
            proxy_scores.append(proxy if overlaps == 0 else float("inf"))
            row[f"{name}_proxy"] = round(proxy, 4)
            row[f"{name}_overlaps"] = overlaps
            row[f"{name}_time"] = round(elapsed, 1)

        avg = sum(proxy_scores) / len(proxy_scores)
        row["avg_proxy"] = round(avg, 4)
        results.append(row)

        elapsed_cfg = time.time() - t_cfg
        marker = " <- BEST" if avg < best_avg else ""
        if avg < best_avg:
            best_avg = avg
            best_cfg = cfg

        print(
            f"[{cfg_idx + 1:3d}/{total}] "
            f"dw_end={cfg['dw_end']:.3f}  dw_start={cfg['dw_start']:.4f}  "
            f"->  avg={avg:.4f}  ({elapsed_cfg:.1f}s){marker}"
        )

    if not results:
        print("\nNo valid hyperparameter combinations (e.g. all had dw_start >= dw_end).")
        return None, float("inf")

    results.sort(key=lambda r: r["avg_proxy"])

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGS")
    print("=" * 70)
    print(f"{'Rank':>4}  {'avg':>7}  {'dw_end':>7}  {'dw_start':>8}")
    print("-" * 50)
    for i, r in enumerate(results[:10]):
        print(f"{i + 1:>4}  {r['avg_proxy']:>7.4f}  {r['dw_end']:>7.3f}  {r['dw_start']:>8.4f}")

    print(f"\nBest config: {best_cfg}  ->  avg proxy = {best_avg:.4f}")

    if output_path:
        output_file = ROOT_DIR / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")

    return best_cfg, best_avg


def main():
    parser = argparse.ArgumentParser(description="Grid search for AnalyticalPlacer hyperparameters")
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        help="Benchmarks to evaluate on (default: ibm01 ibm09 ibm17)"
    )
    parser.add_argument(
        "--output", default="results/grid_search.csv",
        help="CSV output path relative to repo root"
    )
    args = parser.parse_args()

    if not TESTCASE_ROOT.exists():
        print(f"Error: {TESTCASE_ROOT} not found. Run: bash scripts/setup_benchmark_env.sh")
        sys.exit(1)

    run_grid_search(args.benchmarks, args.output)


if __name__ == "__main__":
    main()

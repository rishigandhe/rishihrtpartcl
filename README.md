# rishihrtpartcl

Minimal private workspace for your Partcl/HRT macro placement work.

This repo intentionally keeps only your code and lightweight runner scripts.
It does **not** commit the challenge repo, testcases, or large benchmark assets.

## Included

- `src/hrt_place/` — modular placer (losses, legalize, `AnalyticalPlacer`, optional `backends/dreamplace.py` hook)
- `src/placer_vnext.py` — challenge entry (VNext: Adam + L-BFGS); **`evaluate` loads this by default**
- `src/analytical_placer.py` — thin subclass for `run_eval_analytical.sh` / grid search
- `src/proxy_refine.py`, `src/grid_search.py`, `scripts/setup_benchmark_env.sh`, `scripts/run_eval.sh`

## Quick start

1. Create a virtual env and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Pull benchmark dependency locally (not committed):

```bash
bash scripts/setup_benchmark_env.sh
```

3. Run a benchmark on your placer:

```bash
bash scripts/run_eval.sh -b ibm01
```

4. Run local grid search:

```bash
python src/grid_search.py --benchmarks ibm01
```

## Notes

- Benchmarks and evaluator are loaded from `.deps/macro-place-challenge-2026`.
- Nothing under `.deps/` is tracked by git to keep this repo lean.

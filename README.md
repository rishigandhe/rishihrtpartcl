# rishihrtpartcl

Minimal private workspace for your Partcl/HRT macro placement work.

This repo intentionally keeps only your code and lightweight runner scripts.
It does **not** commit the challenge repo, testcases, or large benchmark assets.

## Included

- `src/placer_v6_pilot.py` — **main placer** (FP32 GPU, pilot top-k vs eDensity, halos, L-BFGS, congestion); best ICCAD04 proxy in this repo
- `src/analytical_placer.py` — default harness entry (thin subclass of v6 so `evaluate`’s `__module__` check passes); used by `run_eval.sh`, `run_eval_analytical.sh`, and `grid_search.py`
- `src/hrt_place/` + `src/placer_vnext.py` — legacy modular VNext pipeline (Adam + L-BFGS) if you want to compare or tune that path
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

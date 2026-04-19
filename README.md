# rishihrtpartcl

GPU **V6 analytical macro placer** (pilot race: top-k vs eDensity density, macro halos, Adam + L-BFGS, congestion ramp) with the **IBM ICCAD04** benchmarks and **`macro_place`** evaluator **vendored in this repo** so you can clone and run anywhere (no separate `.deps` clone).

Reference full-suite proxy (uniform spread init, typical GPU run): **average ≈ 1.36** (machine-dependent).

## Size

About **~800 MB** of testcase + parser data under `external/`.

## Setup

```bash
git clone https://github.com/rishigandhe/rishihrtpartcl.git
cd rishihrtpartcl
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the repo root:

```bash
source .venv/bin/activate
bash scripts/run_eval.sh -b ibm01
bash scripts/run_eval.sh --all
```

Or:

```bash
export PYTHONPATH="$(pwd)"
python3 -m macro_place.evaluate "$(pwd)/src/analytical_placer.py" --all
```

CUDA + a GPU-enabled PyTorch build are recommended; CPU works but is slower.

## Layout

| Path | Role |
|------|------|
| `src/placer_v6_pilot.py` | Placer implementation |
| `src/analytical_placer.py` | Harness entry (`evaluate` loads this path) |
| `macro_place/` | Challenge evaluator (`evaluate`, proxy, loader) |
| `external/MacroPlacement/CodeElements/Plc_client/` | `PlacementCost` / netlist parser |
| `external/MacroPlacement/Testcases/ICCAD04/` | `ibm01` … `ibm18` |

## Attribution

- Testcases and `Plc_client` from [MacroPlacement](https://github.com/partcleda/MacroPlacement) (Partcl).
- Evaluator layout from [macro-place-challenge-2026](https://github.com/partcleda/macro-place-challenge-2026).

Placer implementation: Rishi Gandhe.

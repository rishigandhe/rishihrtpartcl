#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE_DIR="${ROOT_DIR}/.deps/macro-place-challenge-2026"

if [ ! -d "${CHALLENGE_DIR}" ]; then
  echo "Missing benchmark dependency at ${CHALLENGE_DIR}"
  echo "Run: bash scripts/setup_benchmark_env.sh"
  exit 1
fi

cd "${CHALLENGE_DIR}"
PLACER="${ROOT_DIR}/src/analytical_placer.py"
if [ ! -f "${PLACER}" ]; then
  echo "Missing placer at ${PLACER}"
  exit 1
fi
echo "repo: ${ROOT_DIR}"
echo "placer: ${PLACER}  (default: v6 pilot via placer_v6_pilot.py)"
# Alternate / legacy VNext: src/placer_vnext.py
PYTHONPATH="${CHALLENGE_DIR}" python3 -m macro_place.evaluate "${PLACER}" "$@"

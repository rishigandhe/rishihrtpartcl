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
PYTHONPATH="${CHALLENGE_DIR}" python3 -m macro_place.evaluate "${ROOT_DIR}/src/analytical_placer.py" "$@"

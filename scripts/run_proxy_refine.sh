#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE_DIR="${ROOT_DIR}/.deps/macro-place-challenge-2026"

if [ ! -d "${CHALLENGE_DIR}" ]; then
  echo "Missing benchmark dependency at ${CHALLENGE_DIR}"
  echo "Run: bash scripts/setup_benchmark_env.sh"
  exit 1
fi

export PYTHONPATH="${CHALLENGE_DIR}"
exec python3 "${ROOT_DIR}/src/proxy_refine.py" "$@"

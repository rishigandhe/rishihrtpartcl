#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${ROOT_DIR}/.deps"
CHALLENGE_DIR="${DEPS_DIR}/macro-place-challenge-2026"

mkdir -p "${DEPS_DIR}"

if [ ! -d "${CHALLENGE_DIR}" ]; then
  git clone --depth 1 https://github.com/partcleda/macro-place-challenge-2026.git "${CHALLENGE_DIR}"
fi

mkdir -p "${CHALLENGE_DIR}/external"
if [ ! -d "${CHALLENGE_DIR}/external/MacroPlacement" ]; then
  git clone --depth 1 --branch fix-scientific-notation-parsing \
    https://github.com/partcleda/MacroPlacement.git \
    "${CHALLENGE_DIR}/external/MacroPlacement"
fi

echo "Benchmark dependency ready at: ${CHALLENGE_DIR}"

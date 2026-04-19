#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}"
PLACER="${ROOT_DIR}/src/analytical_placer.py"
if [ ! -f "${PLACER}" ]; then
  echo "Missing placer at ${PLACER}"
  exit 1
fi
if [ ! -d "${ROOT_DIR}/external/MacroPlacement/Testcases/ICCAD04" ]; then
  echo "Missing testcases at ${ROOT_DIR}/external/MacroPlacement/Testcases/ICCAD04"
  exit 1
fi
echo "repo: ${ROOT_DIR}"
echo "placer: ${PLACER}"
python3 -m macro_place.evaluate "${PLACER}" "$@"

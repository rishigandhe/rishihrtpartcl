#!/usr/bin/env bash
# Backward-compatible alias: main eval script now defaults to VNext.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_eval.sh" "$@"

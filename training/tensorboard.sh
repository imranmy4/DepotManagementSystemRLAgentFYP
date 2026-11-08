#!/usr/bin/env bash

# Usage: training/tensorboard.sh [experiment]   (default: full) — shows that experiment's latest run.
# Works from any directory (paths are relative to the repo root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${ROOT}/logs/${1:-full}"

LATEST=$(ls -d ${BASE_DIR}/run_*/MaskablePPO_1 2>/dev/null | sort -t_ -k2 | tail -1)

if [ -z "${LATEST}" ]; then
  echo "Error: No run directories found in ${BASE_DIR}/run_*/MaskablePPO_1"
  exit 1
fi

echo "Latest run: ${LATEST}"
"${ROOT}/.venv/bin/tensorboard" --logdir "${LATEST}" --port 6006

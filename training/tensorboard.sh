#!/usr/bin/env bash

# Usage: training/tensorboard.sh [run_id]   (default: the latest run).
# Works from any directory (paths are relative to the repo root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "$1" ]; then
  LATEST="${ROOT}/logs/$1/MaskablePPO_1"
else
  LATEST=$(ls -d "${ROOT}"/logs/run_*/MaskablePPO_1 2>/dev/null | sort | tail -1)
fi

if [ -z "${LATEST}" ] || [ ! -d "${LATEST}" ]; then
  echo "Error: no run log found (looked for ${LATEST:-${ROOT}/logs/run_*/MaskablePPO_1})"
  exit 1
fi

echo "Run: ${LATEST}"
"${ROOT}/.venv/bin/tensorboard" --logdir "${LATEST}" --port 6006

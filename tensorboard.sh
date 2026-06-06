#!/usr/bin/env bash

BASE_DIR="logs/full"

LATEST=$(ls -d ${BASE_DIR}/run_*/MaskablePPO_1 2>/dev/null | sort -t_ -k2 | tail -1)

if [ -z "${LATEST}" ]; then
  echo "Error: No run directories found in ${BASE_DIR}/run_*/MaskablePPO_1"
  exit 1
fi

echo "Latest run: ${LATEST}"
.venv/bin/tensorboard --logdir "${LATEST}" --port 6006
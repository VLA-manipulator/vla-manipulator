#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

available_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
available_gb=$((available_kb / 1024 / 1024))
printf 'data_root=%s free=%sGiB required=%sGiB\n' "${DATA_ROOT}" "${available_gb}" "${MIN_FREE_GB}"

du -sh \
  "${DATA_ROOT}/openpi-data" \
  "${DATA_ROOT}/datasets" \
  "${DATA_ROOT}/checkpoints" \
  "${DATA_ROOT}/huggingface" \
  2>/dev/null || true

if (( available_gb < MIN_FREE_GB )); then
  echo "Refusing to train: insufficient free space for an atomic checkpoint replacement." >&2
  exit 1
fi

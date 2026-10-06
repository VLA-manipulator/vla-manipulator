#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/root/autodl-tmp/workspace/openpi}"
EXP_NAME="${EXP_NAME:-so101_smoke}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/datasets/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/openpi-data}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

MIN_FREE_GB="${MIN_FREE_GB:-20}" bash "${PROJECT_ROOT}/scripts/check_server_storage.sh"
cd "${OPENPI_ROOT}"
uv run scripts/train.py pi05_so101_lora \
  --exp-name="${EXP_NAME}" \
  --num-train-steps="${TRAIN_STEPS}" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/root/autodl-tmp/workspace/openpi}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/datasets/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/openpi-data}"

cd "${OPENPI_ROOT}"
uv run scripts/compute_norm_stats.py --config-name pi05_so101_lora "$@"

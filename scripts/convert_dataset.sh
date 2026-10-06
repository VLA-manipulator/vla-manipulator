#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/root/autodl-tmp/workspace/openpi}"
INPUT_ROOT="${INPUT_ROOT:-/root/autodl-tmp/datasets/so101_staging/cube_red_smoke_5}"
HF_DATASET_REPO="${HF_DATASET_REPO:-VLA-manipulator/so101-mujoco-pick}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/datasets/lerobot/${HF_DATASET_REPO}}"

cd "${OPENPI_ROOT}"
uv run python "${PROJECT_ROOT}/openpi_ext/convert_so101_staging_to_lerobot.py" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --repo-id "${HF_DATASET_REPO}" \
  "$@"

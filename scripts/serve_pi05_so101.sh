#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/root/autodl-tmp/workspace/openpi}"
CHECKPOINT_DIR="${1:?usage: serve_pi05_so101.sh CHECKPOINT_DIR [extra serve_policy args]}"
shift

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/openpi-data}"
cd "${OPENPI_ROOT}"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora \
  --policy.dir="${CHECKPOINT_DIR}" \
  "$@"

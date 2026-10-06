#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/root/autodl-tmp/workspace/openpi}"
HF_DATASET_REPO="${HF_DATASET_REPO:-VLA-manipulator/so101-mujoco-pick}"

cd "${OPENPI_ROOT}"
uv run python "${PROJECT_ROOT}/openpi_ext/install_into_openpi.py" \
  --openpi-root "${OPENPI_ROOT}" --repo-id "${HF_DATASET_REPO}" --check
uv run python "${PROJECT_ROOT}/openpi_ext/install_into_openpi.py" \
  --openpi-root "${OPENPI_ROOT}" --repo-id "${HF_DATASET_REPO}"
uv run python -c '
from openpi.training import config
c = config.get_config("pi05_so101_lora")
assert c.model.pi05
assert "lora" in c.model.paligemma_variant
assert "lora" in c.model.action_expert_variant
assert c.ema_decay is None and c.keep_period is None
print(c)
'

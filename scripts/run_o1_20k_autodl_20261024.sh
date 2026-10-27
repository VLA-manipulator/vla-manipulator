#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp
repo=$base/workspace/openpi
experiment=o1_40_unique_5epoch_20261023
checkpoint_root=/root/o1_checkpoints/pi05_so101_lora_o1_40/$experiment
status=$base/o1_train_20k_20261024_v3.status
log=$base/o1_train_20k_20261024_v3.log

test -f "$checkpoint_root/11999/_CHECKPOINT_METADATA"
test ! -e "$checkpoint_root/19999"
test -d "$base/datasets/so101_staging/o1_40_unique_20261023"
test -d "$base/datasets/so101_staging/multitask_480_fixed_20261022"
test -f "$base/datasets/lerobot/HF2147552431/so101-mujoco-o1-40-unique-v1/meta/info.json"
if pgrep -af 'scripts/train.py.*pi05_so101_lora_o1_40' | grep -v -e pgrep -e grep; then
  echo TRAIN_ALREADY_RUNNING > "$status"
  exit 1
fi
free_kib=$(df -Pk /root/o1_checkpoints | awk 'NR==2 {print $4}')
if (( free_kib < 11 * 1024 * 1024 )); then
  echo WAITING_FOR_SPACE > "$status"
  exit 1
fi

echo RUNNING > "$status"
export HF_LEROBOT_HOME=$base/datasets/lerobot
cd "$repo"
if .venv/bin/python scripts/train.py pi05_so101_lora_o1_40 \
  --exp-name "$experiment" \
  --resume \
  --num-train-steps 20000 \
  --save-interval 20000 \
  --no-wandb-enabled \
  --keep-period 11999 > "$log" 2>&1; then
  if test -f "$checkpoint_root/19999/_CHECKPOINT_METADATA"; then
    echo COMPLETE > "$status"
  else
    echo MISSING_FINAL_CHECKPOINT > "$status"
    exit 1
  fi
else
  echo FAILED > "$status"
  exit 1
fi

#!/usr/bin/env bash
# Wait for the remainder run, validate/upload 480 demos, then train from pi05_base.
set -Eeuo pipefail

project=/root/autodl-tmp/workspace/vla-manipulator-20261022
openpi=/root/autodl-tmp/workspace/openpi
python="$openpi/.venv/bin/python"
legacy=/root/autodl-tmp/datasets/so101_staging/multitask_v1_480_20261022
partial=/root/autodl-tmp/datasets/so101_staging/multitask_fixed_320_20261022
remainder=/root/autodl-tmp/datasets/so101_staging/multitask_fixed_remainder_138_20261022
collection_status=/root/autodl-tmp/multitask_fixed_remainder_138_20261022.status
merged=/root/autodl-tmp/datasets/so101_staging/multitask_480_fixed_20261022
repo=HF2147552431/so101-mujoco-multitask-v1
converted="/root/autodl-tmp/datasets/lerobot/$repo"
experiment=so101_multitask_480_30k_v1
checkpoint="/root/autodl-tmp/checkpoints/pi05_so101_lora_multitask/$experiment"
status=/root/autodl-tmp/multitask_480_to_30k_20261022.status

if [[ -e "$status" ]]; then
  echo "Refusing to overwrite existing handoff status: $status" >&2
  exit 1
fi
set_status() { printf '%s %s\n' "$1" "$(date -Is)" > "$status"; }
trap 'echo "handoff failed at line $LINENO" >&2' ERR
trap 'code=$?; if (( code != 0 )); then set_status HANDOFF_FAILED; fi' EXIT

export HF_HOME=/root/autodl-tmp/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets/lerobot
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_NO_SYNC=1
export WANDB_MODE=offline
export PATH="/root/.local/bin:$PATH"
source /etc/network_turbo

set_status WAITING_COLLECTION
while true; do
  [[ -f "$collection_status" ]] || { echo "missing collection status" >&2; exit 1; }
  read -r state _ < "$collection_status"
  case "$state" in
    COLLECT_COMPLETE) break ;;
    COLLECT_RUNNING) sleep 60 ;;
    *) echo "collection ended with $state" >&2; exit 1 ;;
  esac
done

set_status VALIDATING
"$python" "$project/openpi_ext/validate_multitask_480.py" \
  --legacy-root "$legacy" --new-root "$partial" --new-root "$remainder"
[[ ! -e "$merged" && ! -e "$converted" && ! -e "$checkpoint" ]] || {
  echo "Refusing to overwrite merged data, converted data, or experiment" >&2
  exit 1
}

"$python" "$project/openpi_ext/merge_so101_staging.py" \
  --source "$legacy" --source "$partial" --source "$remainder" --output "$merged"
"$python" -c 'import sys; sys.path.insert(0, sys.argv[1]); from convert_so101_staging_to_lerobot import inspect_staging; from pathlib import Path; _, rows, _, _ = inspect_staging(Path(sys.argv[2])); assert len(rows) == 480, len(rows); print("validated merged payloads",len(rows))' \
  "$project/openpi_ext" "$merged"

set_status CONVERTING_UPLOADING
cd "$openpi"
"$python" "$project/openpi_ext/convert_so101_staging_to_lerobot.py" \
  --input-root "$merged" --output-root "$converted" --repo-id "$repo" --push-to-hub
"$python" -c 'from huggingface_hub import HfApi; import sys; api=HfApi(); r=api.repo_info(sys.argv[1],repo_type="dataset"); files=api.list_repo_files(sys.argv[1],repo_type="dataset"); assert r.private and files,(r.private,len(files)); print("verified private Hugging Face dataset",sys.argv[1],len(files))' "$repo"

set_status PREPARING_TRAINING
"$python" "$project/openpi_ext/install_multitask_config.py" --openpi-root "$openpi" --check
"$python" "$project/openpi_ext/install_multitask_config.py" --openpi-root "$openpi"
"$python" -c 'from openpi.training import config; c=config.get_config("pi05_so101_lora_multitask"); assert c.data.repo_id=="HF2147552431/so101-mujoco-multitask-v1" and c.model.pi05 and not c.model.discrete_state_input and c.save_interval==5000; print("verified multitask pi05_base training config")'
uv run scripts/compute_norm_stats.py --config-name pi05_so101_lora_multitask

while true; do
  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  if (( available_kb >= 20 * 1024 * 1024 )); then
    break
  fi
  set_status WAITING_FOR_SPACE
  echo "Waiting for >=20 GiB free before training; available KiB=$available_kb"
  sleep 300
done

MIN_FREE_GB=20 bash "$project/scripts/check_server_storage.sh"
[[ ! -e "$checkpoint" ]] || { echo "checkpoint appeared while waiting: $checkpoint" >&2; exit 1; }
set_status TRAINING
uv run scripts/train.py pi05_so101_lora_multitask \
  --exp-name="$experiment" --num-train-steps=30000
set_status TRAIN_COMPLETE

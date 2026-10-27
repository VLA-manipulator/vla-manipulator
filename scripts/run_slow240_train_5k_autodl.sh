#!/usr/bin/env bash
# Convert and train only after an explicit quality-review marker is present.
set -Eeuo pipefail

project=/root/autodl-tmp/workspace/vla-manipulator-20261022
openpi=/root/autodl-tmp/workspace/openpi
python="$openpi/.venv/bin/python"
staging=/root/autodl-tmp/datasets/so101_staging/pick8_o1_slow_240_20261024
repo=HF2147552431/so101-pick8-o1-slow240-v1
output="/root/autodl-tmp/datasets/lerobot/$repo"
config=pi05_so101_lora_pick8_o1_slow240_5k
experiment=pick8_o1_slow240_5k_20261024
checkpoint="/root/autodl-tmp/checkpoints/$config/$experiment"
status=/root/autodl-tmp/so101_slow240_train_5k_20261024.status

set_status() { printf '%s %s\n' "$1" "$(date -Is)" > "$status"; }
trap 'code=$?; if (( code != 0 )); then set_status TRAIN_PIPELINE_FAILED; fi' EXIT
[[ ! -e "$status" && ! -e "$output" && ! -e "$checkpoint" ]] || {
  echo 'Refusing to overwrite a prior slow240 conversion/training run' >&2
  exit 1
}
export HF_HOME=/root/autodl-tmp/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets/lerobot
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_NO_SYNC=1
export WANDB_MODE=offline
export PATH="/root/.local/bin:$PATH"
source /etc/network_turbo

set_status WAITING_FOR_QUALITY_REVIEW
while [[ ! -s "$staging/quality.approved" ]]; do
  sleep 30
done
[[ -s "$staging/quality_report.json" ]] || {
  echo 'Quality report is absent' >&2
  exit 1
}

set_status CONVERTING_AND_UPLOADING
cd "$openpi"
"$python" -u "$project/openpi_ext/convert_so101_staging_to_lerobot.py" \
  --input-root "$staging/pick8" --input-root "$staging/o1" \
  --output-root "$output" --repo-id "$repo" --push-to-hub
"$python" -c 'from huggingface_hub import HfApi; import sys; a=HfApi(); i=a.repo_info(sys.argv[1],repo_type="dataset"); f=a.list_repo_files(sys.argv[1],repo_type="dataset"); assert i.private and len(f)>20; print("PRIVATE_DATASET_VERIFIED",sys.argv[1],len(f))' "$repo"

set_status COMPUTING_NORM_STATS
"$python" "$project/openpi_ext/install_slow240_config.py" --openpi-root "$openpi" --check
"$python" "$project/openpi_ext/install_slow240_config.py" --openpi-root "$openpi"
"$python" -c 'from openpi.training import config; c=config.get_config("pi05_so101_lora_pick8_o1_slow240_5k"); assert c.data.repo_id=="HF2147552431/so101-pick8-o1-slow240-v1" and c.model.pi05 and not c.model.discrete_state_input and c.num_train_steps==5000 and c.save_interval==5000; print("TRAIN_CONFIG_VERIFIED")'
uv run scripts/compute_norm_stats.py --config-name "$config"

set_status WAITING_FOR_SPACE
while true; do
  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  if (( available_kb >= 14 * 1024 * 1024 )); then
    break
  fi
  echo "Waiting for >=14 GiB free before checkpoint creation; available KiB=$available_kb"
  sleep 300
done
[[ ! -e "$checkpoint" ]] || {
  echo "Checkpoint appeared while waiting: $checkpoint" >&2
  exit 1
}
set_status TRAINING
uv run scripts/train.py "$config" --exp-name="$experiment" --num-train-steps=5000
[[ -f "$checkpoint/4999/_CHECKPOINT_METADATA" ]] || {
  echo 'Training exited without its final checkpoint metadata' >&2
  exit 1
}
set_status TRAIN_COMPLETE
echo "TRAIN_COMPLETE $checkpoint/4999"

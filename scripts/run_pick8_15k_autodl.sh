#!/usr/bin/env bash
# Isolated eight-pick baseline: 160 successful demos, pi05_base, 15k steps.
set -Eeuo pipefail

project=/root/autodl-tmp/workspace/vla-manipulator-20261022
openpi=/root/autodl-tmp/workspace/openpi
python="$openpi/.venv/bin/python"
input=/root/autodl-tmp/datasets/so101_staging/multitask_v1_480_20261022
repo=HF2147552431/so101-mujoco-pick8-v2
output="/root/autodl-tmp/datasets/lerobot/$repo"
experiment=so101_pick8_160_15k_v2
checkpoint="/root/autodl-tmp/checkpoints/pi05_so101_lora_pick8/$experiment"
status=/root/autodl-tmp/so101_pick8_15k_20261022.status

[[ ! -e "$status" && ! -e "$output" && ! -e "$checkpoint" ]] || {
  echo "Refusing to overwrite an existing pick8 run" >&2
  exit 1
}
set_status() { printf '%s %s\n' "$1" "$(date -Is)" > "$status"; }
trap 'echo "pick8 run failed at line $LINENO" >&2' ERR
trap 'code=$?; if (( code != 0 )); then set_status PICK8_FAILED; fi' EXIT

export HF_HOME=/root/autodl-tmp/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets/lerobot
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_NO_SYNC=1
export WANDB_MODE=offline
export PATH="/root/.local/bin:$PATH"
source /etc/network_turbo

set_status VALIDATING
"$python" -c 'import collections,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=[json.loads(x) for x in (p/"manifest.jsonl").read_text().splitlines()]; expected={k:20 for k in ("rod_red","rod_yellow","sphere_blue","sphere_green","cube_red","cube_blue","dumbbell_purple","dumbbell_orange")}; actual=collections.Counter(r["task_id"] for r in rows); assert len(rows)==160 and actual==expected,(len(rows),actual); assert all(r["success"] and r["saved"] and (p/r["path"]).is_file() for r in rows); print("verified 160 successful pick8 episodes",sum(r["frames"] for r in rows),"frames")' "$input"

set_status CONVERTING_UPLOADING
cd "$openpi"
"$python" "$project/openpi_ext/convert_so101_staging_to_lerobot.py" \
  --input-root "$input" --output-root "$output" --repo-id "$repo" --push-to-hub
"$python" -c 'from huggingface_hub import HfApi; import sys; api=HfApi(); info=api.repo_info(sys.argv[1],repo_type="dataset"); files=api.list_repo_files(sys.argv[1],repo_type="dataset"); assert info.private and files,(info.private,len(files)); print("verified private Hugging Face dataset",sys.argv[1],len(files))' "$repo"

set_status PREPARING_TRAINING
"$python" "$project/openpi_ext/install_pick8_config.py" --openpi-root "$openpi" --check
"$python" "$project/openpi_ext/install_pick8_config.py" --openpi-root "$openpi"
"$python" -c 'from openpi.training import config; c=config.get_config("pi05_so101_lora_pick8"); assert c.data.repo_id=="HF2147552431/so101-mujoco-pick8-v2" and c.model.pi05 and not c.model.discrete_state_input and c.save_interval==5000 and c.num_train_steps==15000; print("verified pick8 pi05_base training config")'
uv run scripts/compute_norm_stats.py --config-name pi05_so101_lora_pick8

while true; do
  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  if (( available_kb >= 20 * 1024 * 1024 )); then
    break
  fi
  set_status WAITING_FOR_SPACE
  echo "Waiting for >=20 GiB free before pick8 training; available KiB=$available_kb"
  sleep 300
done

MIN_FREE_GB=20 bash "$project/scripts/check_server_storage.sh"
[[ ! -e "$checkpoint" ]] || { echo "checkpoint appeared while waiting: $checkpoint" >&2; exit 1; }
set_status TRAINING
uv run scripts/train.py pi05_so101_lora_pick8 \
  --exp-name="$experiment" --num-train-steps=15000
set_status TRAIN_COMPLETE

#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp
openpi=$base/workspace/openpi
simulation=$base/workspace/vla-manipulator-20261022/simulation
checkpoint=/root/o1_checkpoints/pi05_so101_lora_o1_40/o1_40_unique_5epoch_20261023/11999
staging=$base/datasets/so101_staging/o1_40_unique_20261023
tag=${O1_DIAG_TAG:-20261024}
read -r -a train_seeds <<< "${O1_TRAIN_SEEDS:-10000 10001}"
read -r -a heldout_seeds <<< "${O1_HELDOUT_SEEDS:-20001 20012}"
status=$base/o1_teacher_forced_${tag}.status
serve_log=$base/o1_teacher_forced_${tag}_serve.log
run_log=$base/o1_teacher_forced_${tag}.log
result=$base/o1_teacher_forced_${tag}.jsonl

if pgrep -f '[s]cripts/serve_policy.py' >/dev/null; then
  printf 'OTHER_POLICY_SERVER_RUNNING\n' > "$status"
  exit 1
fi
test -f "$checkpoint/assets/HF2147552431/so101-mujoco-o1-40-unique-v1/norm_stats.json"
test -f "$staging/manifest.jsonl"
nvidia-smi -L
printf 'STARTING_SERVICE\n' > "$status"

cd "$openpi"
export HF_HOME=$base/huggingface
export HF_LEROBOT_HOME=$base/datasets/lerobot
export OPENPI_DATA_HOME=$base/openpi-data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
.venv/bin/python -u scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi05_so101_lora_o1_40 \
  --policy.dir "$checkpoint" > "$serve_log" 2>&1 &
service_pid=$!
cleanup() {
  kill "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if grep -q 'server listening on' "$serve_log"; then
    ready=1
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    printf 'SERVICE_FAILED\n' > "$status"
    exit 1
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  printf 'SERVICE_TIMEOUT\n' > "$status"
  exit 1
fi

printf 'RUNNING\n' > "$status"
cd "$simulation"
export MUJOCO_GL=egl
export PYTHONPATH=$simulation
if /root/autodl-tmp/workspace/vla-manipulator/simulation/.venv/bin/python -u \
  "$base/diagnose_o1_teacher_forced_20261024.py" \
  --training-root "$staging" --train-seeds "${train_seeds[@]}" \
  --heldout-seeds "${heldout_seeds[@]}" --repeats 3 --output "$result" \
  > "$run_log" 2>&1; then
  printf 'COMPLETE\n' > "$status"
else
  printf 'DIAGNOSTIC_FAILED\n' > "$status"
  exit 1
fi

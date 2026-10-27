#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp
checkpoint=$base/checkpoint_backups/pi05_so101_lora_so101_all_v1_step24999
status=$base/so101_25k_contact_seams.status
service_log=$base/so101_25k_contact_seams_serve.log
run_log=$base/so101_25k_contact_seams.log
result=$base/so101_25k_contact_seams.jsonl

test -f "$checkpoint/_CHECKPOINT_METADATA"
test -d "$checkpoint/params"
test -d "$checkpoint/assets"
test -f "$base/diagnose_contact_seams_25k.py"
test ! -e "$status"

export OPENPI_DATA_HOME=$base/openpi-data
export UV_NO_SYNC=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$base/workspace/vla-manipulator/simulation

echo "DIAGNOSE_START $(date -Is)" > "$status"
cd "$base/workspace/openpi"
/root/.local/bin/uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora --policy.dir="$checkpoint" > "$service_log" 2>&1 &
service_pid=$!
cleanup() {
  kill "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for attempt in $(seq 1 180); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "SERVICE_EXITED_EARLY $(date -Is)" | tee -a "$status"
    exit 1
  fi
  if (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "SERVICE_NOT_READY $(date -Is)" | tee -a "$status"
  exit 1
fi
echo "SERVICE_READY $(date -Is)" | tee -a "$status"

cd "$base/workspace/vla-manipulator/simulation"
if /root/.local/bin/uv run python "$base/diagnose_contact_seams_25k.py" \
  --cases \
  rod_red:1000 rod_red:1001 rod_yellow:1005 rod_yellow:1003 \
  sphere_blue:1001 sphere_blue:1000 sphere_green:1000 sphere_green:1002 \
  cube_red:1004 cube_red:1003 cube_blue:1001 cube_blue:1004 \
  dumbbell_purple:1000 dumbbell_purple:1001 dumbbell_orange:1001 dumbbell_orange:1000 \
  --replan-steps 5 --max-steps 300 --output "$result" > "$run_log" 2>&1; then
  echo "DIAGNOSE_COMPLETE $(date -Is)" | tee -a "$status"
else
  code=$?
  echo "DIAGNOSE_FAILED exit=$code $(date -Is)" | tee -a "$status"
  exit "$code"
fi

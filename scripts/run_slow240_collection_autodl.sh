#!/usr/bin/env bash
# Collect fresh slower demonstrations, then stop for a quality-review gate.
set -Eeuo pipefail

project=/root/autodl-tmp/workspace/vla-manipulator-20261022
simulation="$project/simulation"
sim_python=/root/autodl-tmp/workspace/vla-manipulator/simulation/.venv/bin/python
openpi_python=/root/autodl-tmp/workspace/openpi/.venv/bin/python
root=/root/autodl-tmp/datasets/so101_staging/pick8_o1_slow_240_20261024
plan="$root/seed_plan.json"
pick="$root/pick8"
o1="$root/o1"
report="$root/quality_report.json"
status=/root/autodl-tmp/so101_slow240_collect_20261024.status

set_status() { printf '%s %s\n' "$1" "$(date -Is)" > "$status"; }
trap 'code=$?; if (( code != 0 )); then set_status COLLECT_OR_QUALITY_FAILED; fi' EXIT

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$simulation"
set_status WAITING_FOR_SEED_PLAN
while [[ ! -s "$plan" ]]; do
  if ! pgrep -f '[p]lan_diverse_expert_seeds' >/dev/null; then
    echo 'Seed planner stopped without an output plan' >&2
    exit 1
  fi
  sleep 15
done
"$sim_python" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["target_motor_step"]==1.5 and p["pick_count"]==20 and p["o1_count"]==80 and len(p["tasks"])==9; print("verified diverse seed plan")' "$plan"

[[ ! -e "$pick" && ! -e "$o1" && ! -e "$report" ]] || {
  echo 'Refusing to overwrite existing slow240 collection output' >&2
  exit 1
}
cd "$simulation"
set_status COLLECTING_PICK8
"$sim_python" -u -m mj_env.scripts.collect_demonstrations \
  --all --episodes 50 --successes-per-object 20 --max-attempts-per-object 50 \
  --seed-plan "$plan" --root "$pick" --headless
set_status COLLECTING_O1
"$sim_python" -u -m mj_env.scripts.collect_o1_demonstrations \
  --seed-plan "$plan" --root "$o1" --successes 80

set_status AUDITING
export PYTHONPATH="$project"
"$openpi_python" -u "$project/openpi_ext/audit_slow240_staging.py" \
  --pick-root "$pick" --o1-root "$o1" --report "$report"
set_status QUALITY_REVIEW
echo "COLLECTION_AND_AUDIT_COMPLETE $report"

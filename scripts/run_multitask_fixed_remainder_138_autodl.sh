#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/workspace/vla-manipulator-20261022
python=/root/autodl-tmp/workspace/vla-manipulator/simulation/.venv/bin/python
root=/root/autodl-tmp/datasets/so101_staging/multitask_fixed_remainder_138_20261022
status=/root/autodl-tmp/multitask_fixed_remainder_138_20261022.status
log=/root/autodl-tmp/multitask_fixed_remainder_138_20261022.log

if [[ -e "$root" || -e "$status" || -e "$log" ]]; then
  echo "Refusing to overwrite an existing collection or log" >&2
  exit 1
fi

printf 'COLLECT_RUNNING %s\n' "$(date -Is)" > "$status"
cd "$project/simulation"
if PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$python" \
  -m mj_env.scripts.collect_multitask \
  --quota o5_covered_stack=18 \
  --quota p2_language_ref=40 \
  --quota p3_in_container=40 \
  --quota p4_stack=40 \
  --max-attempts-factor 3 --seed 10035 --fixed-instructions \
  --root "$root" --height 224 --width 224 --headless > "$log" 2>&1; then
  if "$python" -c 'import collections,json,pathlib; p=pathlib.Path("/root/autodl-tmp/datasets/so101_staging/multitask_fixed_remainder_138_20261022"); rows=[json.loads(x) for x in (p/"manifest.jsonl").read_text().splitlines()]; counts=collections.Counter(r["task_id"] for r in rows); assert counts=={"o5_covered_stack":18,"p2_language_ref":40,"p3_in_container":40,"p4_stack":40},counts; assert all(r["success"] and r["saved"] for r in rows); labels=collections.Counter(r["instruction"] for r in rows if r["task_id"]=="p2_language_ref"); assert labels=={"put the red cube on the blue disc":20,"put the yellow cube on the blue disc":20},labels; print("VERIFIED",len(rows),dict(counts))' >> "$log" 2>&1; then
    printf 'COLLECT_COMPLETE %s\n' "$(date -Is)" > "$status"
  else
    printf 'COLLECT_VERIFY_FAILED %s\n' "$(date -Is)" > "$status"
    exit 2
  fi
else
  printf 'COLLECT_FAILED %s\n' "$(date -Is)" > "$status"
  exit 2
fi

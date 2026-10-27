"""Fail-closed quality audit for the slower eight-pick plus O1 staging data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from openpi_ext.convert_so101_staging_to_lerobot import inspect_staging


PICK_TASKS = (
    "rod_red", "rod_yellow", "sphere_blue", "sphere_green",
    "cube_red", "cube_blue", "dumbbell_purple", "dumbbell_orange",
)
PICK_LABELS = {
    "rod_red": "pick up the red rod",
    "rod_yellow": "pick up the yellow rod",
    "sphere_blue": "pick up the blue ball",
    "sphere_green": "pick up the green ball",
    "cube_red": "pick up the red cube",
    "cube_blue": "pick up the blue cube",
    "dumbbell_purple": "pick up the purple dumbbell",
    "dumbbell_orange": "pick up the orange dumbbell",
}
O1_LABEL = "put the red cube on the blue disc"


def audit_root(root: Path, *, o1: bool) -> dict:
    metadata, records, height, width = inspect_staging(root)
    if metadata["fps"] != 20 or (height, width) != (224, 224):
        raise ValueError(f"unexpected FPS/resolution for {root}: {metadata['fps']}, {height}x{width}")
    expected_count = 80 if o1 else 160
    if len(records) != expected_count:
        raise ValueError(f"{root}: expected {expected_count} episodes, got {len(records)}")

    per_task: dict[str, list[dict]] = defaultdict(list)
    seeds: dict[str, set[int]] = defaultdict(set)
    for record in records:
        task = "o1_open_baseline" if o1 else str(record["target"])
        expected_label = O1_LABEL if o1 else PICK_LABELS.get(task)
        if expected_label is None or record["task"] != expected_label:
            raise ValueError(f"bad task/label: {task}, {record['task']!r}")
        seed = int(record["seed"])
        if seed in seeds[task] or 20000 <= seed <= 20020:
            raise ValueError(f"duplicate or held-out seed: {task} {seed}")
        seeds[task].add(seed)
        if not record.get("success") or not record.get("saved"):
            raise ValueError(f"unsuccessful episode saved: {task} {seed}")
        if float(record.get("expert_target_motor_step", -1)) != 1.5:
            raise ValueError(f"unexpected speed target: {task} {seed}")
        if float(record.get("wrist_roll_excess_degrees", 1e9)) > 90.0:
            raise ValueError(f"excess wrist motion: {task} {seed}")
        frames = int(record["frames"])
        if not 80 <= frames <= 600:
            raise ValueError(f"implausible episode length: {task} {seed} {frames}")

        with np.load(root / record["path"], allow_pickle=False) as episode:
            actions = episode["actions"]
            state = episode["state"]
            if np.any(actions[:, :5] < -100.01) or np.any(actions[:, :5] > 100.01):
                raise ValueError(f"arm action out of motor range: {task} {seed}")
            if np.any(actions[:, 5] < -0.01) or np.any(actions[:, 5] > 100.01):
                raise ValueError(f"jaw action out of motor range: {task} {seed}")
            # Measured joint position can overshoot a saturated motor target
            # slightly under MuJoCo's actuator dynamics (observed max 100.825).
            # Commands still must stay within the strict [-100, 100] bound.
            if np.any(state[:, :5] < -101.0) or np.any(state[:, :5] > 101.0):
                raise ValueError(f"arm state out of motor range: {task} {seed}")
            if np.std(episode["image"][0]) < 2 or np.std(episode["wrist_image"][0]) < 2:
                raise ValueError(f"blank camera image: {task} {seed}")
            arm_speed = np.max(np.abs(np.diff(actions[:, :5], axis=0)), axis=1)
            moving = arm_speed[arm_speed > 0.02]
            if len(moving) < 30:
                raise ValueError(f"too little arm movement: {task} {seed}")
            per_task[task].append({
                "seed": seed,
                "frames": frames,
                "arm_speed_median": float(np.median(moving)),
                "arm_speed_p95": float(np.percentile(moving, 95)),
                "arm_speed_max": float(np.max(moving)),
            })

    expected_tasks = {"o1_open_baseline": 80} if o1 else {task: 20 for task in PICK_TASKS}
    if {task: len(rows) for task, rows in per_task.items()} != expected_tasks:
        raise ValueError(f"task counts do not match: { {k: len(v) for k, v in per_task.items()} }")
    return {
        "root": str(root), "episodes": len(records), "fps": metadata["fps"],
        "resolution": [height, width],
        "tasks": {
            task: {
                "episodes": len(rows),
                "frames_min": min(row["frames"] for row in rows),
                "frames_median": float(np.median([row["frames"] for row in rows])),
                "frames_max": max(row["frames"] for row in rows),
                "arm_speed_median": float(np.median([row["arm_speed_median"] for row in rows])),
                "arm_speed_p95_median": float(np.median([row["arm_speed_p95"] for row in rows])),
                "initial_xy_range": [
                    float(min(record["initial_target_position"][axis] for record in records if
                              ("o1_open_baseline" if o1 else record["target"]) == task))
                    for axis in (0, 1)
                ] + [
                    float(max(record["initial_target_position"][axis] for record in records if
                              ("o1_open_baseline" if o1 else record["target"]) == task))
                    for axis in (0, 1)
                ],
            }
            for task, rows in per_task.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-root", type=Path, required=True)
    parser.add_argument("--o1-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "pick": audit_root(args.pick_root, o1=False),
        "o1": audit_root(args.o1_root, o1=True),
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("QUALITY_PASS 240 episodes, 20 Hz, aligned timestamps/actions, labels and seed counts", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

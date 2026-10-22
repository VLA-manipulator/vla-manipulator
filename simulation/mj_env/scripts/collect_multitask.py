"""Collect legacy picks and graded tasks with the existing staging recorder.

Only successful episodes with acceptable wrist travel enter the dataset.
Failures are written to collection_results.jsonl and their frame buffers are
discarded.  The original collect_demonstrations.py is deliberately unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mj_env import objects
from mj_env.adapters import LeRobotSO101Adapter
from mj_env.demonstrations import StagingDataset, SynchronizedEpisodeRecorder
from mj_env.env import SO101TabletopEnv
from mj_env.scripts.collect_demonstrations import MAX_WRIST_ROLL_EXCESS_DEGREES
from mj_env.scripts.scripted_pick import GraspSolver, PickTuning, run_episode
from mj_env.scripts.scripted_pick_place import expert_for
from mj_env.specs import LEVEL_ORDER, get
from mj_env.task_env import TaskEnv

MAX_NEW_TASK_WRIST_STEP_DEGREES = 15.0


def _quota(value: str) -> tuple[str, int]:
    try:
        task, amount_text = value.split("=", 1)
        amount = int(amount_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("quota must be TASK=POSITIVE_INTEGER") from exc
    if task not in objects.NAMES and task not in LEVEL_ORDER:
        raise argparse.ArgumentTypeError(f"unknown task {task!r}")
    if amount <= 0:
        raise argparse.ArgumentTypeError("quota must be positive")
    return task, amount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota", type=_quota, action="append", required=True)
    parser.add_argument("--max-attempts-factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.max_attempts_factor < 1 or args.height < 1 or args.width < 1:
        parser.error("attempt factor and image dimensions must be positive")
    names = [task for task, _ in args.quota]
    if len(names) != len(set(names)):
        parser.error("each task may appear only once")
    if args.root.exists():
        parser.error(f"dataset root already exists: {args.root}")

    dataset: StagingDataset | None = None
    attempted = saved = 0
    shortfalls: list[str] = []
    for task_id, quota in args.quota:
        legacy = task_id in objects.NAMES
        if legacy:
            env = SO101TabletopEnv(
                render_mode=None if args.headless else "human",
                image_size=(args.height, args.width),
                max_episode_steps=10_000,
            )
        else:
            spec = get(task_id)
            if not spec.instructions:
                raise ValueError(f"{task_id} has no train instructions")
            env = TaskEnv(
                task_id,
                instruction_pool="train",
                render_mode=None if args.headless else "human",
                image_size=(args.height, args.width),
                max_episode_steps=10_000,
            )
        try:
            fps = round(env.control_hz)
            if abs(env.control_hz - fps) > 1e-9:
                raise ValueError(f"non-integer fps: {env.control_hz}")
            if dataset is None:
                dataset = StagingDataset(root=args.root, fps=fps)
            elif dataset.fps != fps:
                raise ValueError("task fps differs from dataset fps")
            solver = GraspSolver(env.model)
            adapter = LeRobotSO101Adapter(env)
            target_saved = 0
            for offset in range(quota * args.max_attempts_factor):
                if target_saved >= quota:
                    break
                seed = args.seed + offset
                task_text = objects.BY_NAME[task_id].prompt if legacy else ""
                recorder = SynchronizedEpisodeRecorder(
                    dataset=dataset, adapter=adapter, fps=fps, task=task_text
                )

                def on_frame(action, observation, info):
                    if not legacy:
                        instruction = info.get("instruction")
                        if instruction not in spec.instructions:
                            raise ValueError(
                                f"{task_id}: non-train instruction {instruction!r}"
                            )
                        recorder.task = instruction
                    recorder.on_frame(action, observation, info)

                try:
                    if legacy:
                        result = run_episode(
                            env, solver, seed=seed, target=task_id,
                            tuning=PickTuning(), on_frame=on_frame,
                        )
                    else:
                        result = expert_for(env)(
                            env, solver, seed=seed, on_frame=on_frame,
                        )
                    wrist_excess = recorder.wrist_roll_excess_degrees
                    wrist = np.asarray(recorder._wrist_roll_radians, dtype=float)
                    max_wrist_step = (
                        float(np.rad2deg(np.abs(np.diff(wrist)).max()))
                        if len(wrist) > 1 else 0.0
                    )
                    keep = bool(
                        result["success"]
                        and recorder.frame_index > 0
                        and (
                            wrist_excess <= MAX_WRIST_ROLL_EXCESS_DEGREES
                            if legacy else max_wrist_step <= MAX_NEW_TASK_WRIST_STEP_DEGREES
                        )
                    )
                    record = {
                        **result,
                        "task_id": task_id,
                        "seed": seed,
                        "frames": recorder.frame_index,
                        "fps": fps,
                        "instruction": recorder.task,
                        "instruction_kind": "train",
                        "wrist_roll_excess_degrees": wrist_excess,
                        "max_wrist_roll_step_degrees": max_wrist_step,
                        "saved": keep,
                    }
                    if keep:
                        dataset.save_episode(metadata=record)
                        saved += 1
                        target_saved += 1
                    else:
                        dataset.clear_episode_buffer()
                    with (args.root / "collection_results.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    attempted += 1
                    print(
                        f"{task_id} seed={seed} outcome={result['outcome']} "
                        f"saved={keep} frames={recorder.frame_index} "
                        f"progress={target_saved}/{quota}",
                        flush=True,
                    )
                except Exception:
                    dataset.clear_episode_buffer()
                    raise
            if target_saved < quota:
                shortfalls.append(f"{task_id}={target_saved}/{quota}")
        finally:
            env.close()
    print(f"saved {saved}/{attempted} successful episodes to {args.root}")
    if shortfalls:
        print("shortfalls: " + ", ".join(shortfalls))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record only successful O1 pick-and-place episodes from a coverage seed plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mj_env.adapters import LeRobotSO101Adapter
from mj_env.demonstrations import StagingDataset, SynchronizedEpisodeRecorder
from mj_env.scripts.collect_demonstrations import MAX_WRIST_ROLL_EXCESS_DEGREES
from mj_env.scripts.scripted_pick import GraspSolver, PickTuning
from mj_env.scripts.scripted_pick_place import run_pick_place
from mj_env.task_env import TaskEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=80)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()
    if args.successes <= 0 or min(args.height, args.width) <= 0:
        parser.error("success and image dimensions must be positive")
    if args.root.exists():
        parser.error(f"dataset root already exists: {args.root}")

    plan = json.loads(args.seed_plan.read_text(encoding="utf-8"))
    seeds = [int(item["seed"]) for item in plan["tasks"]["o1_open_baseline"]]
    if len(seeds) != len(set(seeds)) or len(seeds) < args.successes:
        parser.error("seed plan needs enough distinct O1 candidates")

    instruction = "put the red cube on the blue disc"
    env = TaskEnv(
        "o1_open_baseline", instruction=instruction, render_mode=None,
        image_size=(args.height, args.width), max_episode_steps=10000,
    )
    fps = round(env.control_hz)
    if abs(env.control_hz - fps) > 1e-9:
        raise ValueError(f"LeRobot requires integer fps, got {env.control_hz}")
    dataset = StagingDataset(root=args.root, fps=fps)
    recorder_adapter = LeRobotSO101Adapter(env)
    solver = GraspSolver(env.model)
    tuning = PickTuning()
    results_path = args.root / "collection_results.jsonl"
    saved = 0
    try:
        for seed in seeds:
            if saved >= args.successes:
                break
            recorder = SynchronizedEpisodeRecorder(
                dataset=dataset, adapter=recorder_adapter, fps=fps, task=instruction,
            )
            result = run_pick_place(
                env, solver, seed=seed, tuning=tuning,
                instruction=instruction, on_frame=recorder.on_frame,
            )
            roll_excess = recorder.wrist_roll_excess_degrees
            accept = bool(
                result["success"] and roll_excess <= MAX_WRIST_ROLL_EXCESS_DEGREES
            )
            record = {
                **result, "frames": recorder.frame_index, "fps": fps,
                "wrist_roll_excess_degrees": roll_excess, "saved": accept,
            }
            if accept:
                dataset.save_episode(metadata=record)
                saved += 1
            else:
                dataset.clear_episode_buffer()
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"O1 seed={seed} outcome={result['outcome']} "
                f"saved={accept} progress={saved}/{args.successes}", flush=True,
            )
    finally:
        env.close()
    print(f"O1 saved {saved}/{args.successes} in {args.root}")
    return 0 if saved == args.successes else 2


if __name__ == "__main__":
    raise SystemExit(main())

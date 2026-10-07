"""Collect successful IK-expert demonstrations into a staging dataset.

Example (from the repository root)::

    python -m mj_env.scripts.collect_demonstrations \
        --object cube_red --episodes 5 \
        --root .tmp/staging/so101_sim_pick_smoke

The staging format deliberately has no LeRobot dependency.  Convert it in the
separate OpenPI environment, whose pinned Gymnasium version conflicts with the
simulator's version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mj_env import objects
from mj_env.adapters import LeRobotSO101Adapter
from mj_env.demonstrations import StagingDataset, SynchronizedEpisodeRecorder
from mj_env.env import SO101TabletopEnv
from mj_env.scripts.scripted_pick import GraspSolver, PickTuning, run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--object", choices=objects.NAMES)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=2, help="attempts per object")
    parser.add_argument(
        "--successes-per-object", type=int,
        help="save this many successful episodes per object; use with --max-attempts-per-object",
    )
    parser.add_argument(
        "--max-attempts-per-object", type=int,
        help="stop retrying each object after this many attempts",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.successes_per_object is not None:
        if args.successes_per_object <= 0:
            parser.error("--successes-per-object must be positive")
        if args.max_attempts_per_object is None or args.max_attempts_per_object < args.successes_per_object:
            parser.error("--max-attempts-per-object must be at least --successes-per-object")
    elif args.max_attempts_per_object is not None:
        parser.error("--max-attempts-per-object requires --successes-per-object")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.root.exists():
        parser.error(
            f"dataset root already exists: {args.root}. Choose a new directory "
            "so an existing dataset is never overwritten."
        )

    env = SO101TabletopEnv(
        render_mode=None if args.headless else "human",
        image_size=(args.height, args.width),
        max_episode_steps=10_000,
    )
    fps = round(env.control_hz)
    if abs(env.control_hz - fps) > 1e-9:
        raise ValueError(f"LeRobot requires integer fps, got {env.control_hz}")

    dataset = StagingDataset(root=args.root, fps=fps)

    solver = GraspSolver(env.model)
    adapter = LeRobotSO101Adapter(env)
    tuning = PickTuning()
    targets = objects.NAMES if args.all else (args.object,)
    results_path = args.root / "collection_results.jsonl"
    saved = 0
    attempted = 0
    shortfalls = []
    try:
        for target in targets:
            target_saved = 0
            target_attempted = 0
            attempt_limit = args.max_attempts_per_object or args.episodes
            for offset in range(attempt_limit):
                if args.successes_per_object is not None and target_saved >= args.successes_per_object:
                    break
                seed = args.seed + offset
                task = objects.BY_NAME[target].prompt
                recorder = SynchronizedEpisodeRecorder(
                    dataset=dataset,
                    adapter=adapter,
                    fps=fps,
                    task=task,
                )
                result = run_episode(
                    env,
                    solver,
                    seed=seed,
                    target=target,
                    tuning=tuning,
                    on_frame=recorder.on_frame,
                )
                attempted += 1
                target_attempted += 1
                record = {
                    **result,
                    "seed": seed,
                    "frames": recorder.frame_index,
                    "fps": fps,
                    "saved": bool(result["success"]),
                }
                if result["success"]:
                    dataset.save_episode(metadata=record)
                    saved += 1
                    target_saved += 1
                else:
                    dataset.clear_episode_buffer()
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"{target} seed={seed} outcome={result['outcome']} "
                    f"frames={recorder.frame_index} saved={record['saved']} "
                    f"target_saved={target_saved}"
                )
            print(f"{target}: saved {target_saved}/{target_attempted} attempts", flush=True)
            if args.successes_per_object is not None and target_saved < args.successes_per_object:
                shortfalls.append(f"{target}={target_saved}/{args.successes_per_object}")
    finally:
        env.close()

    print(f"saved {saved}/{attempted} successful episodes to {args.root}")
    if shortfalls:
        print("success target not reached: " + ", ".join(shortfalls))
        return 2
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

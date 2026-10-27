"""Preflight successful, spatially spread seeds before recording demonstrations.

The planner runs the *complete* expert without recording video.  A candidate
enters the seed plan only if it really succeeds, not merely because its initial
pose has an IK solution.  The recorder must still rerun and verify each seed.
Training seeds should not overlap held-out evaluation seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mj_env import objects
from mj_env.env import SO101TabletopEnv
from mj_env.scripts.scripted_pick import GraspSolver, PickTuning, run_episode
from mj_env.scripts.scripted_pick_place import run_pick_place
from mj_env.task_env import TaskEnv


def features(result: dict, *, place: bool) -> np.ndarray:
    """Normalize geometry so object/goal coverage, not seed order, drives choice."""
    position = np.asarray(result["initial_target_position"], dtype=float)
    yaw = float(result["initial_grasp_yaw"])
    if place:
        goal = np.asarray(result["goal_xy"], dtype=float)
        return np.array([
            position[0] / 0.27, position[1] / 0.27,
            goal[0] / 0.27, goal[1] / 0.27,
            np.linalg.norm(goal - position[:2]) / 0.20,
        ])
    return np.array([
        position[0] / 0.30, position[1] / 0.30,
        0.25 * np.cos(yaw), 0.25 * np.sin(yaw),
    ])


def farthest_first(results: list[dict], count: int, *, place: bool) -> list[dict]:
    """Choose a reproducible maximin ordering over successful layouts."""
    if len(results) < count:
        raise ValueError(f"only {len(results)} successful candidates for {count} seeds")
    vectors = np.stack([features(result, place=place) for result in results])
    distances = np.full(len(results), np.inf)
    available = np.ones(len(results), dtype=bool)
    # Start from the candidate nearest the centre, then fill gaps.  Ties are
    # deterministic because candidates arrive in increasing seed order.
    next_index = int(np.argmin(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))
    chosen: list[dict] = []
    for _ in range(count):
        chosen.append(results[next_index])
        available[next_index] = False
        distances = np.minimum(
            distances, np.linalg.norm(vectors - vectors[next_index], axis=1)
        )
        distances[~available] = -np.inf
        next_index = int(np.argmax(distances))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--pick-candidates", type=int, default=50)
    parser.add_argument("--o1-candidates", type=int, default=180)
    parser.add_argument("--pick-count", type=int, default=20)
    parser.add_argument("--o1-count", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.pick_candidates, args.o1_candidates, args.pick_count, args.o1_count) <= 0:
        parser.error("candidate and selection counts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    tuning = PickTuning()
    plan: dict[str, list[dict]] = {}
    pick_env = SO101TabletopEnv(render_mode=None, max_episode_steps=10000)
    pick_solver = GraspSolver(pick_env.model)
    try:
        for target in objects.NAMES:
            successful = []
            for seed in range(args.seed_start, args.seed_start + args.pick_candidates):
                result = run_episode(
                    pick_env, pick_solver, seed=seed, target=target, tuning=tuning
                )
                if result["success"]:
                    successful.append({"seed": seed, **result})
            if len(successful) < args.pick_count:
                raise RuntimeError(f"{target}: only {len(successful)} successes for {args.pick_count} required")
            plan[target] = farthest_first(successful, len(successful), place=False)
            print(f"{target}: {len(successful)} reachable successes, first {args.pick_count} preferred", flush=True)
    finally:
        pick_env.close()

    o1_env = TaskEnv("o1_open_baseline", render_mode=None, max_episode_steps=10000)
    o1_solver = GraspSolver(o1_env.model)
    try:
        successful = []
        for seed in range(args.seed_start, args.seed_start + args.o1_candidates):
            result = run_pick_place(
                o1_env, o1_solver, seed=seed, tuning=tuning,
                instruction="put the red cube on the blue disc",
            )
            if result["success"]:
                successful.append(result)
        if len(successful) < args.o1_count:
            raise RuntimeError(f"O1: only {len(successful)} successes for {args.o1_count} required")
        plan["o1_open_baseline"] = farthest_first(successful, len(successful), place=True)
        print(f"O1: {len(successful)} reachable successes, first {args.o1_count} preferred", flush=True)
    finally:
        o1_env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"seed_start": args.seed_start, "pick_count": args.pick_count, "o1_count": args.o1_count, "target_motor_step": tuning.target_motor_step, "tasks": plan}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exploratory zero-shot pick-and-return probe; never reports a drop as a place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from mj_env import objects
from mj_env.adapters import LeRobotSO101Adapter
from mj_env.env import SO101TabletopEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target", choices=objects.NAMES, default="cube_red")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--prompt")
    parser.add_argument("--video-dir", type=Path, help="record scene and wrist views as MP4")
    args = parser.parse_args()
    if min(args.episodes, args.replan_steps, args.max_steps) <= 0:
        parser.error("episodes, replan-steps and max-steps must be positive")

    from openpi_client import websocket_client_policy

    prompt = args.prompt or f"{objects.BY_NAME[args.target].prompt} and put it back on the table"
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    env = SO101TabletopEnv(target_object=args.target, render_mode=None, max_episode_steps=args.max_steps)
    adapter = LeRobotSO101Adapter(env)
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, _ = env.reset(seed=seed, options={"target_object": args.target, "task": prompt})
            first_lift_step = None
            first_return_step = None
            first_high_release_step = None
            table_streak = 0
            max_lift_m = 0.0
            steps = 0
            last_info = env._info()
            video = None
            video_path = None
            if args.video_dir is not None:
                scene = observation["observation.image"]
                wrist = observation["observation.wrist_image"]
                if scene.shape != wrist.shape or scene.ndim != 3 or scene.shape[-1] != 3:
                    raise ValueError(f"incompatible video frames: {scene.shape}, {wrist.shape}")
                height, width = scene.shape[:2]
                video_path = args.video_dir / f"{args.target}_seed{seed}.mp4"
                video = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{2 * width}x{height}", "-r", "20", "-i", "-", "-an",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(video_path)],
                    stdin=subprocess.PIPE,
                )
            try:
                while steps < args.max_steps:
                    request = {
                        "observation/image": observation["observation.image"],
                        "observation/wrist_image": observation["observation.wrist_image"],
                        "observation/state": adapter.to_lerobot(observation["observation.state"]),
                        "prompt": prompt,
                    }
                    actions = np.asarray(client.infer(request)["actions"], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
                        raise ValueError(f"invalid policy action chunk: {actions.shape}")
                    for action in actions[: args.replan_steps]:
                        # Continue after pick-and-lift termination: it is not this probe's end.
                        observation, _, _, _, last_info = env.step(adapter.from_lerobot(action))
                        steps += 1
                        if video is not None:
                            frame = np.concatenate(
                                [observation["observation.image"], observation["observation.wrist_image"]],
                                axis=1,
                            )
                            video.stdin.write(np.ascontiguousarray(frame).tobytes())
                        height = float(last_info["lift_height"])
                        grasped = bool(last_info["grasped"])
                        max_lift_m = max(max_lift_m, height)
                        if first_lift_step is None and grasped and height >= 0.06:
                            first_lift_step = steps
                        if first_lift_step is not None:
                            if first_high_release_step is None and not grasped and height > 0.03:
                                first_high_release_step = steps
                            table_streak = table_streak + 1 if not grasped and height <= 0.015 else 0
                            if first_return_step is None and table_streak >= 10:
                                first_return_step = steps - 9
                        if steps >= args.max_steps:
                            break
            finally:
                if video is not None:
                    video.stdin.close()
                    if video.wait() != 0:
                        raise RuntimeError(f"ffmpeg failed to write {video_path}")
            print(json.dumps({
                "target": args.target,
                "seed": seed,
                "prompt": prompt,
                "steps": steps,
                "first_lift_step": first_lift_step,
                "first_high_release_step": first_high_release_step,
                "first_return_to_table_step": first_return_step,
                "max_lift_height_m": round(max_lift_m, 4),
                "final_lift_height_m": round(float(last_info["lift_height"]), 4),
                "final_grasped": bool(last_info["grasped"]),
                "video": str(video_path) if video_path is not None else None,
                "note": "return_to_table may be a drop; visual review is required to claim deliberate placement",
            }), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

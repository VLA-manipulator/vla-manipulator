"""Record one O1 policy rollout and its grasp/placement diagnostics."""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
from pathlib import Path

import numpy as np
import websockets.sync.client

from mj_env.adapters import LeRobotSO101Adapter
from mj_env.task_env import TaskEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    websockets.sync.client.connect = functools.partial(
        websockets.sync.client.connect, ping_interval=None
    )
    client = websocket_client_policy.WebsocketClientPolicy(port=args.port)
    env = TaskEnv("o1_open_baseline", max_episode_steps=args.max_steps)
    adapter = LeRobotSO101Adapter(env)
    observation, info = env.reset(
        seed=args.seed, options={"instruction": "put the red cube on the blue disc"}
    )
    scene = observation["observation.image"]
    height, width = scene.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{2 * width}x{height}",
            "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "24", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    min_tcp_distance = float("inf")
    max_height = float("-inf")
    grasped_steps = 0
    criterion_steps = 0
    first_actions = []
    steps = 0
    try:
        while steps < args.max_steps and not info["success"]:
            request = {
                "observation/image": observation["observation.image"],
                "observation/wrist_image": observation["observation.wrist_image"],
                "observation/state": adapter.to_lerobot(observation["observation.state"]),
                "prompt": env.task,
            }
            chunk = np.asarray(client.infer(request)["actions"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 6 or not np.isfinite(chunk).all():
                raise ValueError(f"invalid action chunk {chunk.shape}")
            if not first_actions:
                first_actions = chunk[: args.replan_steps].round(2).tolist()
            for action in chunk[: args.replan_steps]:
                observation, _, terminated, truncated, info = env.step(
                    adapter.from_lerobot(action)
                )
                steps += 1
                frame = np.concatenate(
                    [observation["observation.image"], observation["observation.wrist_image"]],
                    axis=1,
                )
                assert encoder.stdin is not None
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                min_tcp_distance = min(min_tcp_distance, float(info["tcp_target_distance"]))
                max_height = max(max_height, float(info["height_above_rest"]))
                grasped_steps += int(bool(info["grasped"]))
                criterion_steps += int(bool(info["criterion_ok"]))
                if steps % 50 == 0 or info["success"]:
                    print(json.dumps({
                        "step": steps,
                        "tcp_distance_m": round(float(info["tcp_target_distance"]), 4),
                        "height_m": round(float(info["height_above_rest"]), 4),
                        "grasped": bool(info["grasped"]),
                        "on_goal": bool(info["object_on_goal"]),
                        "criterion_ok": bool(info["criterion_ok"]),
                        "state_motor": adapter.to_lerobot(observation["observation.state"]).round(1).tolist(),
                    }), flush=True)
                if terminated or truncated or steps >= args.max_steps:
                    break
            if terminated or truncated:
                break
    finally:
        env.close()
        assert encoder.stdin is not None
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg recording failed")
    print(json.dumps({
        "seed": args.seed,
        "prompt": env.task,
        "success": bool(info["success"]),
        "steps": steps,
        "min_tcp_distance_m": round(min_tcp_distance, 4),
        "max_height_m": round(max_height, 4),
        "grasped_steps": grasped_steps,
        "criterion_steps": criterion_steps,
        "first_actions_motor": first_actions,
        "video": str(args.output),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

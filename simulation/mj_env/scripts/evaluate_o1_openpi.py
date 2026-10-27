"""Evaluate an O1-only OpenPI checkpoint on unseen pick-and-place seeds."""

from __future__ import annotations

import argparse
import functools

import numpy as np
import websockets.sync.client

from mj_env.adapters import LeRobotSO101Adapter
from mj_env.task_env import TaskEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    if min(args.episodes, args.replan_steps, args.max_steps) < 1:
        parser.error("episodes and step counts must be positive")

    from openpi_client import websocket_client_policy

    websockets.sync.client.connect = functools.partial(
        websockets.sync.client.connect, ping_interval=None
    )
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    env = TaskEnv("o1_open_baseline", max_episode_steps=args.max_steps)
    adapter = LeRobotSO101Adapter(env)
    successes = 0
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, info = env.reset(
                seed=seed, options={"instruction": "put the red cube on the blue disc"}
            )
            steps = 0
            while steps < args.max_steps and not info["success"]:
                request = {
                    "observation/image": observation["observation.image"],
                    "observation/wrist_image": observation["observation.wrist_image"],
                    "observation/state": adapter.to_lerobot(observation["observation.state"]),
                    "prompt": env.task,
                }
                action_chunk = np.asarray(client.infer(request)["actions"], dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 6 or not np.isfinite(action_chunk).all():
                    raise ValueError(f"invalid action chunk: {action_chunk.shape}")
                for action in action_chunk[: args.replan_steps]:
                    observation, _, terminated, truncated, info = env.step(adapter.from_lerobot(action))
                    steps += 1
                    if info["success"] or terminated or truncated or steps >= args.max_steps:
                        break
                if terminated or truncated:
                    break
            success = bool(info["success"])
            successes += int(success)
            print(f"seed={seed} success={success} steps={steps} goal_error_mm={info.get('goal_error_mm')}", flush=True)
    finally:
        env.close()
    print(f"success_rate={successes}/{args.episodes}={successes / args.episodes:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

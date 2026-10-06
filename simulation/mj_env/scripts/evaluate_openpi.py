"""Evaluate a remote OpenPI checkpoint in the SO101 MuJoCo environment."""

from __future__ import annotations

import argparse
import math

import numpy as np

from mj_env import objects
from mj_env.adapters import LeRobotSO101Adapter
from mj_env.env import SO101TabletopEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target", choices=objects.NAMES, default="cube_red")
    parser.add_argument("--seed", type=int, default=1000, help="use held-out seeds for evaluation")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.replan_steps, args.max_steps) <= 0:
        parser.error("episode, replanning and step counts must be positive")

    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "Install OpenPI's packages/openpi-client in the simulation environment first"
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    successes = 0
    env = SO101TabletopEnv(
        target_object=args.target,
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
    )
    adapter = LeRobotSO101Adapter(env)
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, info = env.reset(seed=seed, options={"target_object": args.target})
            steps = 0
            outcome = False
            while steps < args.max_steps:
                request = {
                    "observation/image": observation["observation.image"],
                    "observation/wrist_image": observation["observation.wrist_image"],
                    "observation/state": adapter.to_lerobot(observation["observation.state"]),
                    "prompt": env.task,
                }
                action_chunk = np.asarray(client.infer(request)["actions"], dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 6:
                    raise ValueError(f"policy returned invalid action chunk {action_chunk.shape}")
                if not np.isfinite(action_chunk).all():
                    raise ValueError("policy returned non-finite actions")

                for normalized_action in action_chunk[: args.replan_steps]:
                    observation, _, terminated, truncated, info = env.step(
                        adapter.from_lerobot(normalized_action)
                    )
                    steps += 1
                    if info["success"] or terminated or truncated or steps >= args.max_steps:
                        break
                if info["success"] or terminated or truncated:
                    outcome = bool(info["success"])
                    break

            successes += int(outcome)
            print(
                f"episode={episode} seed={seed} success={outcome} "
                f"steps={steps} lift_height={float(info['lift_height']):.4f}",
                flush=True,
            )
    finally:
        env.close()

    rate = successes / args.episodes
    wilson_half_width = 1.96 * math.sqrt(max(rate * (1 - rate), 0.25 / args.episodes) / args.episodes)
    print(
        f"success_rate={successes}/{args.episodes}={rate:.3f} "
        f"rough_95pct_half_width={wilson_half_width:.3f}"
    )
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())

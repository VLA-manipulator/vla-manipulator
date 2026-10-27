"""Evaluate the opt-in asynchronous RTC server on SO101 MuJoCo scenes."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from mj_env import objects
from mj_env.adapters import LeRobotSO101Adapter
from mj_env.env import SO101TabletopEnv
from mj_env.rtc import RTCController, RTCDeadlineMissed, RealTimePacer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target", choices=objects.NAMES, default="cube_red")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--min-horizon", type=int, default=3)
    parser.add_argument("--initial-delay", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    env = SO101TabletopEnv(
        target_object=args.target,
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
    )
    adapter = LeRobotSO101Adapter(env)

    def request(observation: dict) -> dict:
        return {
            "observation/image": observation["observation.image"],
            "observation/wrist_image": observation["observation.wrist_image"],
            "observation/state": adapter.to_lerobot(observation["observation.state"]),
            "prompt": args.prompt if args.prompt is not None else env.task,
        }

    successes = 0
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, info = env.reset(seed=seed, options={"target_object": args.target})
            controller = RTCController(
                client.infer,
                horizon=args.horizon,
                min_horizon=args.min_horizon,
                initial_delay=args.initial_delay,
            )
            steps = 0
            outcome = False
            deadline_missed = False
            executed_actions: list[np.ndarray] = []
            started = time.monotonic()
            try:
                controller.start(request(observation))
                pacer = RealTimePacer(env.control_hz)
                while steps < args.max_steps:
                    pacer.wait()
                    try:
                        action = controller.get_action(request(observation))
                    except RTCDeadlineMissed:
                        deadline_missed = True
                        break
                    observation, _, terminated, truncated, info = env.step(adapter.from_lerobot(action))
                    executed_actions.append(action)
                    steps += 1
                    if info["success"] or terminated or truncated:
                        outcome = bool(info["success"])
                        break
            finally:
                controller.close()
            successes += int(outcome)
            action_array = np.asarray(executed_actions, dtype=np.float32)
            step_change = np.linalg.norm(np.diff(action_array, axis=0), axis=-1) if len(action_array) > 1 else []
            accel_change = np.linalg.norm(np.diff(action_array, n=2, axis=0), axis=-1) if len(action_array) > 2 else []
            infer_p95_ms = float(np.percentile(controller.infer_times_ms, 95)) if controller.infer_times_ms else float("nan")
            print(
                f"episode={episode} seed={seed} success={outcome} steps={steps} "
                f"deadline_missed={deadline_missed} wall_s={time.monotonic()-started:.2f} "
                f"infer_p95_ms={infer_p95_ms:.1f} "
                f"step_change_p95={float(np.percentile(step_change, 95)) if len(step_change) else float('nan'):.3f} "
                f"second_difference_p95={float(np.percentile(accel_change, 95)) if len(accel_change) else float('nan'):.3f} "
                f"lift_height={float(info['lift_height']):.4f}",
                flush=True,
            )
    finally:
        env.close()
    rate = successes / args.episodes
    half_width = 1.96 * math.sqrt(max(rate * (1 - rate), 0.25 / args.episodes) / args.episodes)
    print(f"success_rate={successes}/{args.episodes}={rate:.3f} rough_95pct_half_width={half_width:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

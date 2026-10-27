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
    parser.add_argument("--latency-margin-steps", type=int, default=1)
    parser.add_argument("--calibration-requests", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.calibration_requests <= 0:
        parser.error("episodes, max-steps and calibration-requests must be positive")
    if not (0 < args.min_horizon < args.horizon) or args.latency_margin_steps < 0:
        parser.error("require 0 < min-horizon < horizon and nonnegative latency margin")

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
        # The first calls compile the ordinary and guided JAX paths.  Exclude
        # compilation from steady-state latency measurement, but do not score
        # or advance the simulator during warm-up.
        warm_observation, _ = env.reset(seed=args.seed, options={"target_object": args.target})
        warm_request = request(warm_observation)
        first = client.infer(warm_request)
        warm_previous = np.zeros((args.horizon, 6), dtype=np.float32)
        warm_previous[: args.horizon - args.min_horizon] = np.asarray(first["actions"], dtype=np.float32)[
            args.min_horizon : args.horizon
        ]
        guided_request = {
            **warm_request,
            "rtc_previous_actions": warm_previous,
            "rtc_delay_steps": min(args.initial_delay, args.horizon - args.min_horizon),
            "rtc_overlap_steps": args.horizon - args.min_horizon,
        }
        client.infer(guided_request)
        latency_ms: list[float] = []
        for _ in range(args.calibration_requests):
            for probe in (warm_request, guided_request):
                began = time.monotonic()
                client.infer(probe)
                latency_ms.append((time.monotonic() - began) * 1000)
        measured_delay = math.ceil(max(latency_ms) * env.control_hz / 1000)
        initial_delay = max(args.initial_delay, measured_delay)
        if initial_delay + args.latency_margin_steps >= args.horizon:
            raise RTCDeadlineMissed(
                f"measured round-trip latency {max(latency_ms):.1f} ms needs "
                f"{initial_delay} steps plus {args.latency_margin_steps} margin; "
                f"the {args.horizon}-step chunk cannot cover it"
            )
        print(
            f"rtc_calibration requests={len(latency_ms)} rtt_p50_ms={np.median(latency_ms):.1f} "
            f"rtt_max_ms={max(latency_ms):.1f} delay_steps={initial_delay} "
            f"margin_steps={args.latency_margin_steps}",
            flush=True,
        )
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, info = env.reset(seed=seed, options={"target_object": args.target})
            controller = RTCController(
                client.infer,
                horizon=args.horizon,
                min_horizon=args.min_horizon,
                initial_delay=initial_delay,
                latency_margin_steps=args.latency_margin_steps,
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

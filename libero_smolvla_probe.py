"""Run the LIBERO-finetuned SmolVLA through a LIBERO control loop."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="lerobot/smolvla_libero")
    parser.add_argument("--steps", type=int, default=5, help="number of VLA action chunks")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--gui", action="store_true", help="open a WSLg MuJoCo viewer")
    args = parser.parse_args()

    if args.gui:
        # robosuite still creates its camera context during import; keep that
        # context on OSMesa and let mujoco.viewer create the separate GLFW UI.
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.setdefault("DISPLAY", ":0")
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir")
    else:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"loading={args.model_id}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.config.device = str(device)
    policy.config.num_steps = args.num_steps
    policy.to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("model=loaded", flush=True)

    config = LiberoEnv(
        task="libero_spatial",
        task_ids=[0],
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_height=128,
        observation_width=128,
        init_states=True,
    )
    envs = config.create_envs(n_envs=1, use_async_envs=False)
    env = envs["libero_spatial"][0]
    viewer = None

    try:
        observation, _ = env.reset()
        task_description = env.envs[0].unwrapped.task_description
        print(f"task={task_description}", flush=True)
        if args.gui:
            import mujoco.viewer

            base_env = env.envs[0].unwrapped
            sim = base_env._env.env.sim
            viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)
            print("GUI=opened; close the MuJoCo window to stop", flush=True)

        for step in range(args.steps):
            pixels = observation["pixels"]
            state = observation["robot_state"]
            image1_np = np.flip(pixels["image"][0], axis=(0, 1)).copy()
            image2_np = np.flip(pixels["image2"][0], axis=(0, 1)).copy()
            image1 = torch.from_numpy(image1_np).permute(2, 0, 1).float() / 255.0
            image2 = torch.from_numpy(image2_np).permute(2, 0, 1).float() / 255.0
            quat_xyzw = state["eef"]["quat"][0].astype(np.float64)
            axis_angle = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
            state8 = np.concatenate(
                [state["eef"]["pos"][0], axis_angle, state["gripper"]["qpos"][0]]
            ).astype(np.float32)
            frame = {
                "observation.images.camera1": image1,
                "observation.images.camera2": image2,
                "observation.images.camera3": image1,
                "observation.state": torch.from_numpy(state8),
                "task": task_description,
            }
            with torch.inference_mode():
                batch = preprocess(frame)
                action_chunk = postprocess(policy.predict_action_chunk(batch))
            action7 = action_chunk.detach().cpu().numpy().reshape(-1, 7)
            action7 = np.clip(action7, -1.0, 1.0)
            print(f"step={step} action_chunk={action7.shape}", flush=True)
            print(np.array2string(action7[:3], precision=4, suppress_small=True), flush=True)
            actions_to_run = action7 if args.gui else action7[:1]
            for action in actions_to_run:
                if viewer is not None and not viewer.is_running():
                    return
                observation, reward, terminated, truncated, info = env.step(action[None, :])
                if viewer is not None:
                    viewer.sync()
                    time.sleep(1.0 / 20.0)
            print(
                f"env_step reward={reward} terminated={terminated} truncated={truncated}",
                flush=True,
            )
        if viewer is not None:
            print("动作块执行完成，窗口保持打开；关闭窗口后程序退出", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
        print("LIBERO_SMOLVLA_OK", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()

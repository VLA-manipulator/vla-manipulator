"""Gymnasium MuJoCo environment for SO101 tabletop grasping."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101 import ARM_JOINT_NAMES, SO101

from . import layout, objects
from .glfw_viewer import GLFWViewer
from .scene import (
    FIXED_FINGER_GEOM,
    GRASP_ZONE_HALF_ANGLE,
    GRASP_ZONE_RADIUS,
    MOVING_FINGER_GEOM,
    SCENE_CAMERA,
    TABLE_HEIGHT,
    TCP_SITE,
    WRIST_CAMERA,
    build_model,
)

#: The six controlled joints, in actuator order.  The gripper is appended to
#: the five arm joints that ``so101`` defines.
JOINT_NAMES: tuple[str, ...] = (*ARM_JOINT_NAMES, "gripper")

#: Arm retracted and raised clear of the object zone, jaw half open.  The TCP
#: sits at roughly (0.15, 0, 0.26), above and behind everything on the table,
#: so objects can settle undisturbed and the scene camera has a clear view.
HOME_QPOS: tuple[float, ...] = (0.0, -1.63, 0.37, 1.27, 0.0, 0.9)

#: An object counts as picked once it is this far above the tabletop while
#: both finger pads are still pressing on it.
LIFT_THRESHOLD = 0.06

#: Free-body drop height above the resting pose, and the settling budget that
#: turns the sampled layout into a physically valid one.
_DROP_MARGIN = 0.004
_SETTLE_STEPS = 250

#: Layout constraints.  The base keep-out is its measured 0.10 m footprint
#: plus clearance; the table bounds are the tabletop inset by 30 mm.
_LAYOUT_MARGIN = 0.012
_BASE_KEEPOUT = 0.12
_TABLE_BOUNDS_X = (-0.02, 0.82)
_TABLE_BOUNDS_Y = (-0.39, 0.39)

_CONTACT_FORCE_EPS = 1e-3


class SO101TabletopEnv(gym.Env):
    """SO101 picking one of eight tabletop objects, with LeRobot-shaped observations.

    Frame convention: the robot base is at the origin, +x points forward across
    the table, +z is up and the tabletop surface is z = 0.

    Actions and ``observation.state`` are the six joint positions in native
    MuJoCo units -- radians for the five arm joints and for the calibrated
    gripper joint.  Actions are absolute position targets for the MJCF's
    position actuators.  Use :class:`mj_env.adapters.LeRobotSO101Adapter` to
    move between these and LeRobot's normalized motor space.
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        image_size: tuple[int, int] = (224, 224),
        control_hz: float = 20.0,
        max_episode_steps: int = 300,
        target_object: str | None = None,
        objects_subset: Sequence[str] | None = None,
        task: str | None = None,
        show_viewer_ui: bool = False,
        viewer_observation_overlays: bool = True,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode={render_mode!r}")
        selected = tuple(objects.NAMES if objects_subset is None else objects_subset)
        if not selected:
            raise ValueError("objects_subset must contain at least one object")
        if len(set(selected)) != len(selected) or any(name not in objects.BY_NAME for name in selected):
            raise ValueError(f"unknown or duplicate object in objects_subset: {selected!r}")
        if target_object is not None and target_object not in objects.BY_NAME:
            raise ValueError(f"unknown target_object={target_object!r}")
        if target_object is not None and target_object not in selected:
            raise ValueError(f"target_object={target_object!r} is not in objects_subset={selected!r}")
        self.render_mode = render_mode
        self.image_height, self.image_width = image_size
        self.max_episode_steps = int(max_episode_steps)
        self.show_viewer_ui = bool(show_viewer_ui)
        self.viewer_observation_overlays = bool(viewer_observation_overlays)
        self.objects_subset = selected

        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.control_hz = 1.0 / (self.frame_skip * self.model.opt.timestep)
        self.metadata = {**self.metadata, "render_fps": round(self.control_hz)}

        self.robot = SO101()
        self._resolve_indices()

        self._objects = objects.measure_all(self.model)
        self._footprints = [self._objects[name].footprint for name in self.objects_subset]

        self._default_target = target_object
        self._default_task = task
        self.target_object = target_object or self.objects_subset[0]
        self.task = task or objects.BY_NAME[self.target_object].prompt
        self._step_count = 0
        self._layout_fallback = False
        #: Grasp state accumulated over the last control step; ``None`` outside
        #: of :meth:`step`, where the instantaneous test is used instead.
        self._grasp_latch: bool | None = None

        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None
        self._latest_observation_images: tuple[np.ndarray, np.ndarray] | None = None

        low = self.model.actuator_ctrlrange[self._actuator_ids, 0].astype(np.float32)
        high = self.model.actuator_ctrlrange[self._actuator_ids, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        image_space = spaces.Box(
            0, 255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8
        )
        # Measured positions can sit a little outside the commandable range
        # while a soft joint limit is being pushed, so the state box is padded.
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(low - 0.1, high + 0.1, dtype=np.float32),
                "observation.image": image_space,
                "observation.wrist_image": image_space,
            }
        )

    # -- model wiring --------------------------------------------------------

    def _resolve_indices(self) -> None:
        arm_qpos, _ = self.robot.resolve_joint_state_indices(self.model)
        gripper_qpos = self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, "gripper")]
        self._qpos_ids = np.append(arm_qpos, gripper_qpos).astype(int)

        arm_actuators = self.robot.resolve_actuator_indices(self.model)
        gripper_actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self._actuator_ids = np.append(arm_actuators, gripper_actuator).astype(int)

        self._tcp_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self._finger_geom_ids = (
            self._id(mujoco.mjtObj.mjOBJ_GEOM, FIXED_FINGER_GEOM),
            self._id(mujoco.mjtObj.mjOBJ_GEOM, MOVING_FINGER_GEOM),
        )

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, obj_type, name)
        if value < 0:
            raise ValueError(f"scene is missing MuJoCo {obj_type.name} named {name!r}")
        return int(value)

    # -- reset ---------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        requested = options.get("target_object", self._default_target)
        if requested is None:
            requested = self.objects_subset[int(self.np_random.integers(len(self.objects_subset)))]
        elif requested not in objects.BY_NAME:
            raise ValueError(f"unknown target_object={requested!r}")
        self.target_object = requested
        self.task = options.get(
            "task", self._default_task or objects.BY_NAME[self.target_object].prompt
        )

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_ids] = HOME_QPOS
        self.data.ctrl[self._actuator_ids] = HOME_QPOS
        self._place_objects()
        self._settle()

        self._grasp_latch = None
        self._step_count = 0
        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _place_objects(self) -> None:
        positions, yaws, self._layout_fallback = layout.sample_layout(
            self.np_random,
            self._footprints,
            radius_range=GRASP_ZONE_RADIUS,
            half_angle=GRASP_ZONE_HALF_ANGLE,
            base_keepout=_BASE_KEEPOUT,
            margin=_LAYOUT_MARGIN,
            bounds_x=_TABLE_BOUNDS_X,
            bounds_y=_TABLE_BOUNDS_Y,
        )
        placed_names = set(self.objects_subset)
        for name in objects.NAMES:
            metrics = self._objects[name]
            if name not in placed_names:
                # Keep catalogue bodies in the compiled model for stable
                # feature IDs, but move unselected objects below the table.
                self.data.qpos[metrics.qpos_adr : metrics.qpos_adr + 3] = (0.0, 0.0, -1.0)
                self.data.qpos[metrics.qpos_adr + 3 : metrics.qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
                continue
            index = self.objects_subset.index(name)
            x, y = positions[index]
            yaw = yaws[index]
            adr = metrics.qpos_adr
            height = TABLE_HEIGHT + metrics.footprint.rest_clearance + _DROP_MARGIN
            self.data.qpos[adr : adr + 3] = (x, y, height)
            self.data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))

    def _settle(self) -> None:
        """Let the dropped objects come to rest before the episode starts."""
        self.data.qvel[:] = 0.0
        for _ in range(_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # -- step ----------------------------------------------------------------

    def step(self, action: np.ndarray, *, capture_observation: bool = True):
        target = np.asarray(action, dtype=np.float32).reshape(-1)
        if target.shape != self.action_space.shape:
            raise ValueError(
                f"expected action shape {self.action_space.shape}, got {target.shape}"
            )
        self.data.ctrl[self._actuator_ids] = np.clip(
            target, self.action_space.low, self.action_space.high
        )
        # A held object breaks contact with one pad for a step here and there
        # while the wrist accelerates, so the grasp is sampled across the whole
        # control step rather than only at its final instant.  Sampling only
        # the end reports "not grasped" for objects that are plainly 13 cm off
        # the table and still in the fingers.
        target_body = self._objects[self.target_object].body_id
        self._grasp_latch = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._grasp_latch = self._grasp_latch or self._pads_engaged(target_body)
        self._step_count += 1

        observation = self._observation() if capture_observation else None
        info = self._info()
        reward = self._reward(info)
        terminated = bool(info["success"])
        truncated = self._step_count >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    @staticmethod
    def _reward(info: dict[str, Any]) -> float:
        """Reach, then hold, then lift.  Only a real grasp earns lift credit."""
        reward = -float(info["tcp_target_distance"])
        if info["grasped"]:
            reward += 1.0
            reward += 5.0 * min(float(info["lift_height"]), LIFT_THRESHOLD) / LIFT_THRESHOLD
        if info["success"]:
            reward += 10.0
        return reward

    # -- observation and diagnostics ----------------------------------------

    def _observation(self) -> dict[str, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        scene_image = self.render_camera(SCENE_CAMERA)
        wrist_image = self.render_camera(WRIST_CAMERA)
        self._latest_observation_images = (scene_image, wrist_image)
        return {
            "observation.state": np.asarray(
                self.data.qpos[self._qpos_ids], dtype=np.float32
            ).copy(),
            "observation.image": scene_image,
            "observation.wrist_image": wrist_image,
        }

    def policy_observation(self, task: str | None = None) -> dict[str, Any]:
        """The observation plus the text prompt, as PI05's processor expects it."""
        observation = dict(self._observation())
        observation["task"] = task or self.task
        return observation

    @property
    def viewer_running(self) -> bool:
        """Whether the interactive viewer exists and has not been closed."""
        return self._viewer is not None and self._viewer.is_running()

    def _info(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        metrics = self._objects[self.target_object]
        target_position = np.asarray(self.data.xpos[metrics.body_id], dtype=np.float32)
        target_com_position = np.asarray(self.data.xipos[metrics.body_id], dtype=np.float32)
        tcp_position = np.asarray(self.data.site_xpos[self._tcp_site_id], dtype=np.float32)
        grasped = (
            self._pads_engaged(metrics.body_id)
            if self._grasp_latch is None
            else self._grasp_latch
        )
        lift_height = float(target_com_position[2]) - (
            TABLE_HEIGHT + metrics.footprint.rest_clearance
        )
        return {
            # MuJoCo's clock is the authoritative timestamp for synchronized
            # demonstrations.  Unlike wall-clock time it advances by exactly
            # one control period per env.step(), regardless of rendering or
            # dataset-writing latency.
            "simulation_time_s": float(self.data.time),
            "task": self.task,
            "target_object": self.target_object,
            "active_objects": self.objects_subset,
            "target_position": target_position.copy(),
            "target_com_position": target_com_position.copy(),
            "tcp_position": tcp_position.copy(),
            "tcp_target_distance": float(np.linalg.norm(tcp_position - target_position)),
            "grasped": grasped,
            "lift_height": lift_height,
            "success": bool(grasped and lift_height >= LIFT_THRESHOLD),
            "object_positions": {
                name: np.asarray(self.data.xpos[m.body_id], dtype=np.float32).copy()
                for name, m in self._objects.items()
                if name in self.objects_subset
            },
            "layout_fallback": self._layout_fallback,
            "steps": self._step_count,
        }

    def _pads_engaged(self, body_id: int) -> bool:
        """True when both finger pads carry load against the target body.

        Contact-based rather than distance-based: an object knocked across the
        table or flung into the air must not read as a successful pick.
        """
        touching = [False, False]
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            for slot, finger in enumerate(self._finger_geom_ids):
                if touching[slot]:
                    continue
                if geom1 == finger:
                    other = geom2
                elif geom2 == finger:
                    other = geom1
                else:
                    continue
                if self.model.geom_bodyid[other] != body_id:
                    continue
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                touching[slot] = abs(wrench[0]) > _CONTACT_FORCE_EPS
        return all(touching)

    # -- rendering -----------------------------------------------------------

    def render_camera(
        self, camera: str, *, height: int | None = None, width: int | None = None
    ) -> np.ndarray:
        """Render one named camera to an RGB array."""
        height = self.image_height if height is None else height
        width = self.image_width if width is None else width
        if (
            self._renderer is None
            or self._renderer.height != height
            or self._renderer.width != width
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_camera(SCENE_CAMERA)
        if self.render_mode == "human":
            if self._viewer is None:
                if self.show_viewer_ui:
                    import mujoco.viewer  # imported lazily: needs a windowing system

                    self._viewer = mujoco.viewer.launch_passive(
                        self.model,
                        self.data,
                        show_left_ui=True,
                        show_right_ui=True,
                    )
                else:
                    self._viewer = GLFWViewer(self.model, self.data)
            if not self._viewer.is_running():
                self._viewer.close()
                self._viewer = None
                return None
            if isinstance(self._viewer, GLFWViewer):
                images = self._latest_observation_images or ()
                self._viewer.render(
                    images if self.viewer_observation_overlays else (),
                    prompt=self.task,
                )
            else:
                self._viewer.sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = ["HOME_QPOS", "JOINT_NAMES", "LIFT_THRESHOLD", "OVERHEAD_CAMERA", "SO101TabletopEnv"]

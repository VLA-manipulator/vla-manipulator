"""High-level SO-101 facade shared by MuJoCo and LeRobot control paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from spatialmath import SE3

from .control.cartesian import CartesianCommand, CartesianPositionController
from .control.motor_mapper import SO101MotorMapper
from .kinematics import ARM_JOINT_NAMES, SO101Arm


@dataclass(frozen=True)
class LeRobotCartesianCommand:
    """Bounded Cartesian command ready for ``SO101Follower.send_action``."""

    cartesian: CartesianCommand
    action: dict[str, float]
    gripper_position: float


class SO101:
    """SO-101 model facade with Cartesian-to-joint and execution adapters.

    ``SO101Arm`` is always the source of FK/IK and uses radians.  MuJoCo
    receives those radians directly.  A real ``SO101Follower`` instead gets a
    complete six-key normalized action through a supplied
    :class:`SO101MotorMapper`.
    """

    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    MUJOCO_TCP_SITE = "gripperframe"
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    # The SO101 MJCF/mesh bundle is kept with this local package.  Keeping
    # the path relative to the package makes the editable uv dependency and
    # the standalone package use the same assets.
    _ASSET_DIR = Path(__file__).resolve().parent / "assets"

    def __init__(
        self,
        *,
        motor_mapper: SO101MotorMapper | None = None,
        max_joint_delta: float | Sequence[float] = np.deg2rad(3.0),
        max_position_error: float = 0.005,
        max_relative_position: float = 0.02,
        position_tolerance: float = 1e-5,
    ) -> None:
        self.kinematics = SO101Arm()
        self._max_joint_delta = max_joint_delta
        self._max_position_error = max_position_error
        self._max_relative_position = max_relative_position
        self._position_tolerance = position_tolerance
        self.motor_mapper: SO101MotorMapper | None = None
        self.cartesian_controller = self._make_controller()
        if motor_mapper is not None:
            self.set_motor_mapper(motor_mapper)

    @classmethod
    def simulated(cls, **kwargs: Any) -> "SO101":
        """Create a MuJoCo-only instance with model-limit motor anchors.

        The all-positive direction assumption is valid only for the included
        simulation bridge.  Do not use this factory for an unverified physical
        arm.
        """
        robot = cls(**kwargs)
        robot.set_motor_mapper(
            SO101MotorMapper.from_model_joint_limits(
                robot.kinematics,
                {name: 1 for name in ARM_JOINT_NAMES},
            )
        )
        return robot

    @classmethod
    def from_joint_mapping(
        cls,
        mapping_path: str | Path,
        *,
        lerobot_calibration_path: str | Path,
        **kwargs: Any,
    ) -> "SO101":
        """Create a real-arm facade from a verified endpoint mapping file.

        The LeRobot calibration file is mandatory so its SHA-256 digest can be
        checked against the file used during endpoint capture.  No serial
        connection is opened by this factory.
        """

        robot = cls(**kwargs)
        mapper = SO101MotorMapper.from_joint_mapping_file(
            robot.kinematics,
            mapping_path,
            lerobot_calibration_path=lerobot_calibration_path,
        )
        robot.set_motor_mapper(mapper)
        return robot

    @property
    def scene_xml_path(self) -> str:
        """Absolute path to the default new-calibration MuJoCo scene."""
        return str(self._ASSET_DIR / "scene.xml")

    @property
    def robot_xml_path(self) -> str:
        """Absolute path to the bare new-calibration SO-101 MuJoCo model."""
        return str(self._ASSET_DIR / "so101_new_calib.xml")

    @property
    def urdf_path(self) -> str:
        """Absolute path to the new-calibration URDF used as model evidence."""
        return str(self._ASSET_DIR / "so101_new_calib.urdf")

    @property
    def joint_limits(self) -> np.ndarray:
        return self.cartesian_controller.joint_limits.copy()

    def set_motor_mapper(self, mapper: SO101MotorMapper) -> None:
        """Attach a verified radians-to-LeRobot normalized-space mapper."""
        if mapper.arm.n != self.kinematics.n:
            raise ValueError("motor mapper does not describe a five-axis SO-101 arm")
        self.motor_mapper = mapper
        self.cartesian_controller = self._make_controller(joint_limits=mapper.joint_limits)

    def load_mjmodel(self, *, scene: bool = True):
        """Load the bundled MuJoCo scene or bare robot model."""
        import mujoco

        return mujoco.MjModel.from_xml_path(self.scene_xml_path if scene else self.robot_xml_path)

    def resolve_joint_state_indices(self, model, *, name_prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
        """Resolve qpos/qvel indexes by arm-joint name in any MuJoCo model."""
        import mujoco

        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        for name in ARM_JOINT_NAMES:
            full_name = f"{name_prefix}{name}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing arm joint {full_name!r}")
            qpos_indices.append(int(model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        return np.asarray(qpos_indices, dtype=int), np.asarray(qvel_indices, dtype=int)

    def resolve_actuator_indices(self, model, *, name_prefix: str = "") -> np.ndarray:
        """Resolve the five arm position actuators without touching other controls."""
        import mujoco

        actuator_indices: list[int] = []
        for name in ARM_JOINT_NAMES:
            full_name = f"{name_prefix}{name}"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            if actuator_id < 0:
                raise ValueError(f"MuJoCo model is missing arm actuator {full_name!r}")
            actuator_indices.append(actuator_id)
        return np.asarray(actuator_indices, dtype=int)

    def mujoco_joint_positions(self, model, data, *, name_prefix: str = "") -> np.ndarray:
        """Read the five simulated arm positions in radians."""
        qpos_indices, _ = self.resolve_joint_state_indices(model, name_prefix=name_prefix)
        return np.asarray(data.qpos[qpos_indices], dtype=float).copy()

    def set_mujoco_arm_target(
        self,
        model,
        data,
        q_target: Sequence[float],
        *,
        name_prefix: str = "",
    ) -> None:
        """Write only the five arm position targets to MuJoCo control slots."""
        q = self._coerce_joint_target(q_target)
        actuator_indices = self.resolve_actuator_indices(model, name_prefix=name_prefix)
        data.ctrl[actuator_indices] = q

    def mujoco_tcp_pose(self, model, data, *, site_name: str | None = None) -> SE3:
        """Read the MuJoCo TCP in the same frame convention as ``fkine``."""
        import mujoco

        name = self.MUJOCO_TCP_SITE if site_name is None else site_name
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"MuJoCo model is missing TCP site {name!r}")
        site_pose = SE3.Rt(
            np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3),
            np.asarray(data.site_xpos[site_id], dtype=float),
            # MuJoCo rotations are orthonormal to floating-point precision,
            # while SpatialMath's default SO(3) check is intentionally exact.
            check=False,
        )
        return site_pose * self.kinematics.mujoco_site_to_tcp

    def cartesian_target(
        self,
        q_current: Sequence[float],
        target_position: Sequence[float],
    ) -> CartesianCommand | None:
        """Convert an absolute base-frame TCP XYZ target into a joint target."""
        return self.cartesian_controller.absolute_target(
            q_current,
            target_position,
        )

    def relative_cartesian_target(
        self,
        q_current: Sequence[float],
        delta_position: Sequence[float],
        *,
        frame: str = "base",
    ) -> CartesianCommand | None:
        """Convert a bounded base/tool-frame XYZ delta into a joint target."""
        return self.cartesian_controller.relative_target(
            q_current,
            delta_position,
            frame=frame,
        )

    def lerobot_cartesian_command(
        self,
        observation: Mapping[str, float],
        target_position: Sequence[float],
        *,
        gripper_position: float | None = None,
    ) -> LeRobotCartesianCommand | None:
        """Create a full normalized LeRobot action from current observation and XYZ.

        The current joint state is decoded from the supplied observation, so
        the first command never starts from a hard-coded joint pose.  Omitting
        ``gripper_position`` preserves the currently observed gripper value.
        """
        mapper = self._require_motor_mapper()
        q_current, observed_gripper = mapper.from_lerobot_observation(observation)
        command = self.cartesian_target(q_current, target_position)
        if command is None:
            return None
        gripper = observed_gripper if gripper_position is None else float(gripper_position)
        action = mapper.to_lerobot_action(command.q_target, gripper_position=gripper)
        return LeRobotCartesianCommand(command, action, gripper)

    def send_cartesian_target(
        self,
        follower: Any,
        target_position: Sequence[float],
        *,
        gripper_position: float | None = None,
    ) -> tuple[LeRobotCartesianCommand, dict[str, float]] | None:
        """Read a real follower, build one safe action, and send it.

        ``follower`` is intentionally duck-typed to avoid binding the core
        kinematics package to a live serial device.  It must provide LeRobot's
        ``get_observation`` and ``send_action`` methods.
        """
        config = getattr(follower, "config", None)
        follower_type = getattr(follower, "robot_type", None)
        config_type = getattr(config, "type", None) if config is not None else None
        if follower_type != "so101_follower" or config_type != "so101_follower":
            raise ValueError(
                "send_cartesian_target requires an explicitly identified SO101Follower"
            )
        if getattr(config, "use_degrees", None) is not False:
            raise ValueError(
                "SO101MotorMapper requires SO101FollowerConfig(use_degrees=False)"
            )
        action_features = getattr(follower, "action_features", None)
        expected = set(SO101MotorMapper.ACTION_NAMES)
        if action_features is None or set(action_features) != expected:
            actual = set() if action_features is None else set(action_features)
            raise ValueError(
                "follower action features do not match SO-101; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

        observation = follower.get_observation()
        command = self.lerobot_cartesian_command(
            observation,
            target_position,
            gripper_position=gripper_position,
        )
        if command is None:
            return None
        sent = follower.send_action(command.action)
        return command, dict(sent)

    def _make_controller(self, *, joint_limits: np.ndarray | None = None) -> CartesianPositionController:
        return CartesianPositionController(
            self.kinematics,
            joint_limits=joint_limits,
            max_joint_delta=self._max_joint_delta,
            max_position_error=self._max_position_error,
            max_relative_position=self._max_relative_position,
            position_tolerance=self._position_tolerance,
        )

    def _require_motor_mapper(self) -> SO101MotorMapper:
        if self.motor_mapper is None:
            raise RuntimeError(
                "a verified SO101MotorMapper is required before producing a real LeRobot action"
            )
        return self.motor_mapper

    def _coerce_joint_target(self, q_target: Sequence[float]) -> np.ndarray:
        q = np.asarray(q_target, dtype=float)
        if q.shape != (self.kinematics.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q_target must contain {self.kinematics.n} finite radians")
        limits = self.kinematics.joint_limits
        if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
            raise ValueError("q_target is outside the SO-101 kinematic limits")
        return q

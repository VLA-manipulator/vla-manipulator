"""Explicit conversion between SO-101 radians and LeRobot motor space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from so101.kinematics import ALL_JOINT_NAMES, ARM_JOINT_NAMES, SO101Arm


@dataclass(frozen=True)
class JointMotorAnchors:
    """Physical joint angles represented by LeRobot -100 and +100 values.

    LeRobot calibration JSON stores raw encoder extrema but not the mapping to
    the URDF zero, direction, or radians.  These two anchors intentionally
    make that missing physical relationship explicit.
    """

    q_at_minus_100: float
    q_at_plus_100: float


@dataclass(frozen=True)
class JointEndpointCalibration:
    """LeRobot readings measured at the two model joint limits.

    The values are allowed to appear in either order.  Their ordering captures
    the physical motor direction relative to the URDF joint axis, while their
    separation captures scale and offset without requiring ``-100`` or
    ``+100`` to coincide with a model limit.
    """

    normalized_at_lower_limit: float
    normalized_at_upper_limit: float


@dataclass(frozen=True)
class GripperEndpointCalibration:
    """Gripper radians and LeRobot observations at closed/open endpoints."""

    q_closed: float
    q_open: float
    normalized_closed: float
    normalized_open: float


class SO101MotorMapper:
    """Map five arm radians to default LeRobot normalized action values.

    This adapter targets ``SO101FollowerConfig(use_degrees=False)``.  It does
    not repeat LeRobot's raw encoder calibration; ``SO101Follower`` performs
    normalized-value-to-tick conversion itself.  The caller must determine the
    two physical anchors for every arm joint after real-arm calibration.
    """

    ARM_ACTION_NAMES = tuple(f"{name}.pos" for name in ARM_JOINT_NAMES)
    GRIPPER_ACTION_NAME = "gripper.pos"
    ACTION_NAMES = (*ARM_ACTION_NAMES, GRIPPER_ACTION_NAME)

    def __init__(
        self,
        arm: SO101Arm,
        anchors: Mapping[str, JointMotorAnchors | Sequence[float]],
    ) -> None:
        self.arm = arm
        missing = set(ARM_JOINT_NAMES) - set(anchors)
        extra = set(anchors) - set(ARM_JOINT_NAMES)
        if missing or extra:
            raise ValueError(f"anchors must match arm joints; missing={missing}, extra={extra}")

        normalized: dict[str, JointMotorAnchors] = {}
        endpoint_calibrations: dict[str, JointEndpointCalibration] = {}
        joint_limits = np.empty((arm.n, 2), dtype=float)
        for index, name in enumerate(ARM_JOINT_NAMES):
            anchor = anchors[name]
            if not isinstance(anchor, JointMotorAnchors):
                anchor = JointMotorAnchors(*anchor)
            low = float(anchor.q_at_minus_100)
            high = float(anchor.q_at_plus_100)
            if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
                raise ValueError(f"{name} anchors must be distinct finite radians")
            model_low, model_high = arm.joint_limits[index]
            if min(low, high) < model_low - 1e-9 or max(low, high) > model_high + 1e-9:
                raise ValueError(f"{name} anchors must stay within the kinematic joint limits")
            normalized[name] = JointMotorAnchors(low, high)
            if low < high:
                endpoint_calibrations[name] = JointEndpointCalibration(-100.0, 100.0)
                joint_limits[index] = (low, high)
            else:
                endpoint_calibrations[name] = JointEndpointCalibration(100.0, -100.0)
                joint_limits[index] = (high, low)
        self._anchors = normalized
        self._endpoint_calibrations = endpoint_calibrations
        self._joint_limits = joint_limits
        self._gripper_calibration: GripperEndpointCalibration | None = None

    @classmethod
    def from_model_joint_limits(
        cls,
        arm: SO101Arm,
        joint_directions: Mapping[str, int | float],
    ) -> "SO101MotorMapper":
        """Create a mapper from model limits for MuJoCo or verified hardware.

        ``+1`` means increasing URDF radians maps from LeRobot ``-100`` to
        ``+100``.  ``-1`` reverses that relationship.  This is appropriate for
        the included MuJoCo model once all directions are set to ``+1``.  It is
        not a substitute for checking the physical arm one joint at a time.
        """
        if set(joint_directions) != set(ARM_JOINT_NAMES):
            raise ValueError("joint_directions must specify every arm joint exactly once")
        anchors: dict[str, JointMotorAnchors] = {}
        for name, (low, high) in zip(ARM_JOINT_NAMES, arm.joint_limits, strict=True):
            direction = float(joint_directions[name])
            if direction not in (-1.0, 1.0):
                raise ValueError(f"{name} direction must be +1 or -1")
            anchors[name] = (
                JointMotorAnchors(low, high)
                if direction > 0
                else JointMotorAnchors(high, low)
            )
        return cls(arm, anchors)

    @classmethod
    def from_joint_endpoint_samples(
        cls,
        arm: SO101Arm,
        samples: Mapping[str, JointEndpointCalibration | Sequence[float]],
        *,
        gripper_calibration: GripperEndpointCalibration | Sequence[float] | None = None,
    ) -> "SO101MotorMapper":
        """Create a mapper from read-only measurements at model limits.

        ``samples[name]`` contains the LeRobot normalized observation measured
        while the physical joint matches the model's lower and upper limits,
        respectively.  Unlike :class:`JointMotorAnchors`, the two measured
        values do not need to be ``-100`` and ``+100``.  This lets a physical
        encoder range be wider than the URDF's software-safe range.
        """

        missing = set(ARM_JOINT_NAMES) - set(samples)
        extra = set(samples) - set(ARM_JOINT_NAMES)
        if missing or extra:
            raise ValueError(
                f"endpoint samples must match arm joints; missing={missing}, extra={extra}"
            )

        endpoint_calibrations: dict[str, JointEndpointCalibration] = {}
        for name in ARM_JOINT_NAMES:
            sample = samples[name]
            if not isinstance(sample, JointEndpointCalibration):
                sample = JointEndpointCalibration(*sample)
            lower = float(sample.normalized_at_lower_limit)
            upper = float(sample.normalized_at_upper_limit)
            if not np.isfinite(lower) or not np.isfinite(upper) or np.isclose(lower, upper):
                raise ValueError(f"{name} endpoint samples must be distinct finite values")
            if lower < -100.0 - 1e-9 or lower > 100.0 + 1e-9:
                raise ValueError(f"{name} lower-limit observation must be in [-100, 100]")
            if upper < -100.0 - 1e-9 or upper > 100.0 + 1e-9:
                raise ValueError(f"{name} upper-limit observation must be in [-100, 100]")
            endpoint_calibrations[name] = JointEndpointCalibration(lower, upper)

        mapper = cls.__new__(cls)
        mapper.arm = arm
        mapper._endpoint_calibrations = endpoint_calibrations
        mapper._joint_limits = arm.joint_limits
        mapper._anchors = mapper._extrapolated_anchors()
        mapper._gripper_calibration = mapper._coerce_gripper_calibration(
            gripper_calibration
        )
        return mapper

    @classmethod
    def from_joint_mapping_file(
        cls,
        arm: SO101Arm,
        path: str | Path,
        *,
        lerobot_calibration_path: str | Path | None = None,
    ) -> "SO101MotorMapper":
        """Load the JSON produced by the read-only endpoint calibrator.

        Supplying ``lerobot_calibration_path`` also verifies that the motor
        mapper was captured against the same LeRobot calibration JSON.
        """

        mapping_path = Path(path)
        with mapping_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported SO-101 joint mapping schema")
        if data.get("robot_type") != "so101_follower":
            raise ValueError("joint mapping does not describe an SO-101 follower")
        if data.get("coordinate_mode") != "range_m100_100":
            raise ValueError("joint mapping is not in LeRobot [-100, 100] mode")
        if lerobot_calibration_path is not None:
            calibration_bytes = Path(lerobot_calibration_path).read_bytes()
            actual_digest = hashlib.sha256(calibration_bytes).hexdigest()
            source = data.get("source")
            expected_digest = (
                source.get("lerobot_calibration_sha256")
                if isinstance(source, dict)
                else None
            )
            if actual_digest != expected_digest:
                raise ValueError("joint mapping was captured with a different LeRobot calibration file")
        joints = data.get("arm_joints")
        if not isinstance(joints, dict) or set(joints) != set(ARM_JOINT_NAMES):
            raise ValueError("joint mapping must contain exactly the five SO-101 arm joints")

        samples: dict[str, JointEndpointCalibration] = {}
        for index, name in enumerate(ARM_JOINT_NAMES):
            entry = joints[name]
            if not isinstance(entry, dict):
                raise ValueError(f"invalid joint mapping entry for {name}")
            model_lower = float(entry.get("model_lower_rad"))
            model_upper = float(entry.get("model_upper_rad"))
            expected_lower, expected_upper = arm.joint_limits[index]
            if not np.allclose(
                (model_lower, model_upper),
                (expected_lower, expected_upper),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(f"{name} mapping was produced for different model limits")
            samples[name] = JointEndpointCalibration(
                float(entry.get("normalized_at_lower_limit")),
                float(entry.get("normalized_at_upper_limit")),
            )
        gripper = data.get("gripper")
        if not isinstance(gripper, dict):
            raise ValueError("joint mapping is missing the gripper endpoints")
        gripper_calibration = GripperEndpointCalibration(
            float(gripper.get("model_closed_rad")),
            float(gripper.get("model_open_rad")),
            float(gripper.get("normalized_closed")),
            float(gripper.get("normalized_open")),
        )
        return cls.from_joint_endpoint_samples(
            arm,
            samples,
            gripper_calibration=gripper_calibration,
        )

    @property
    def joint_limits(self) -> np.ndarray:
        """Physical-anchor limits in kinematic radians, independent of sign."""
        return self._joint_limits.copy()

    @property
    def anchors(self) -> dict[str, JointMotorAnchors]:
        return dict(self._anchors)

    @property
    def endpoint_calibrations(self) -> dict[str, JointEndpointCalibration]:
        """Measured normalized observations at each model-safe limit."""
        return dict(self._endpoint_calibrations)

    @property
    def gripper_calibration(self) -> GripperEndpointCalibration | None:
        """Closed/open gripper mapping, when loaded from an endpoint file."""
        return self._gripper_calibration

    @property
    def gripper_joint_limits(self) -> np.ndarray | None:
        calibration = self._gripper_calibration
        if calibration is None:
            return None
        return np.asarray((calibration.q_closed, calibration.q_open), dtype=float)

    def radians_to_normalized(self, q_arm: Sequence[float]) -> np.ndarray:
        q = self._coerce_q(q_arm)
        values = np.empty(self.arm.n, dtype=float)
        for index, name in enumerate(ARM_JOINT_NAMES):
            low, high = self._joint_limits[index]
            if not low - 1e-9 <= q[index] <= high + 1e-9:
                raise ValueError(f"{name} target is outside its validated physical anchors")
            endpoint = self._endpoint_calibrations[name]
            values[index] = endpoint.normalized_at_lower_limit + (
                (q[index] - low) / (high - low)
            ) * (
                endpoint.normalized_at_upper_limit
                - endpoint.normalized_at_lower_limit
            )
        return np.clip(values, -100.0, 100.0)

    def normalized_to_radians(self, normalized: Sequence[float]) -> np.ndarray:
        values = np.asarray(normalized, dtype=float)
        if values.shape != (self.arm.n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"normalized must contain {self.arm.n} finite values")
        if np.any(values < -100.0 - 1e-9) or np.any(values > 100.0 + 1e-9):
            raise ValueError("arm normalized values must be in [-100, 100]")
        q = np.empty(self.arm.n, dtype=float)
        for index, name in enumerate(ARM_JOINT_NAMES):
            low, high = self._joint_limits[index]
            endpoint = self._endpoint_calibrations[name]
            q[index] = low + (
                (values[index] - endpoint.normalized_at_lower_limit)
                / (
                    endpoint.normalized_at_upper_limit
                    - endpoint.normalized_at_lower_limit
                )
            ) * (high - low)
        return q

    def gripper_radians_to_normalized(self, q_gripper: float) -> float:
        """Convert the model gripper angle to LeRobot's ``[0, 100]`` value."""

        calibration = self._require_gripper_calibration()
        q = float(q_gripper)
        if not np.isfinite(q):
            raise ValueError("q_gripper must be finite")
        low = min(calibration.q_closed, calibration.q_open)
        high = max(calibration.q_closed, calibration.q_open)
        if q < low - 1e-9 or q > high + 1e-9:
            raise ValueError("q_gripper is outside the calibrated gripper range")
        normalized = calibration.normalized_closed + (
            (q - calibration.q_closed)
            / (calibration.q_open - calibration.q_closed)
        ) * (calibration.normalized_open - calibration.normalized_closed)
        return float(np.clip(normalized, 0.0, 100.0))

    def gripper_normalized_to_radians(self, normalized: float) -> float:
        """Convert a LeRobot gripper observation to the model hinge angle."""

        calibration = self._require_gripper_calibration()
        value = float(normalized)
        if not np.isfinite(value) or value < -1e-9 or value > 100.0 + 1e-9:
            raise ValueError("gripper normalized value must be in [0, 100]")
        return float(
            calibration.q_closed
            + (value - calibration.normalized_closed)
            / (calibration.normalized_open - calibration.normalized_closed)
            * (calibration.q_open - calibration.q_closed)
        )

    def to_lerobot_action_radians(
        self,
        q_arm: Sequence[float],
        *,
        q_gripper: float,
    ) -> dict[str, float]:
        """Build a complete LeRobot action from six model-space radians.

        This is an additive interface.  :meth:`to_lerobot_action` retains its
        original normalized-gripper contract for existing control code.
        """

        return self.to_lerobot_action(
            q_arm,
            gripper_position=self.gripper_radians_to_normalized(q_gripper),
        )

    def from_lerobot_observation_radians(
        self,
        observation: Mapping[str, float],
    ) -> tuple[np.ndarray, float]:
        """Decode all six LeRobot position channels into model-space radians."""

        q_arm, gripper_position = self.from_lerobot_observation(observation)
        return q_arm, self.gripper_normalized_to_radians(gripper_position)

    def to_lerobot_action(
        self,
        q_arm: Sequence[float],
        *,
        gripper_position: float,
    ) -> dict[str, float]:
        """Build the complete six-key action expected by ``SO101Follower``."""
        gripper = float(gripper_position)
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 100.0:
            raise ValueError("gripper_position must be a finite value in [0, 100]")
        normalized = self.radians_to_normalized(q_arm)
        action = {
            name: float(value)
            for name, value in zip(self.ARM_ACTION_NAMES, normalized, strict=True)
        }
        action[self.GRIPPER_ACTION_NAME] = gripper
        return action

    def from_lerobot_observation(
        self,
        observation: Mapping[str, float],
    ) -> tuple[np.ndarray, float]:
        """Convert a complete default-mode observation/action to radians."""
        missing = set(self.ACTION_NAMES) - set(observation)
        if missing:
            raise ValueError(f"observation is missing required SO-101 keys: {sorted(missing)}")
        normalized = np.asarray([observation[name] for name in self.ARM_ACTION_NAMES], dtype=float)
        gripper = float(observation[self.GRIPPER_ACTION_NAME])
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 100.0:
            raise ValueError("gripper observation must be in [0, 100]")
        return self.normalized_to_radians(normalized), gripper

    @staticmethod
    def validate_lerobot_calibration_file(path: str | Path) -> dict[str, dict[str, int]]:
        """Validate a LeRobot calibration JSON without inferring URDF anchors.

        The returned raw calibration data is useful for auditing, but it is not
        used in the normalized mapping because ``SO101Follower`` applies it
        internally when commands are sent to the Feetech bus.
        """
        calibration_path = Path(path)
        with calibration_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if set(data) != set(ALL_JOINT_NAMES):
            raise ValueError("calibration file must contain exactly the six SO-101 motor names")
        validated: dict[str, dict[str, int]] = {}
        for expected_id, name in enumerate(ALL_JOINT_NAMES, start=1):
            entry = data[name]
            required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
            if not isinstance(entry, dict) or set(entry) != required:
                raise ValueError(f"invalid LeRobot calibration entry for {name}")
            if entry["id"] != expected_id or entry["range_min"] == entry["range_max"]:
                raise ValueError(f"invalid motor id or range for {name}")
            validated[name] = {key: int(value) for key, value in entry.items()}
        return validated

    def _coerce_q(self, q_arm: Sequence[float]) -> np.ndarray:
        q = np.asarray(q_arm, dtype=float)
        if q.shape != (self.arm.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q_arm must contain {self.arm.n} finite radians")
        return q

    def _extrapolated_anchors(self) -> dict[str, JointMotorAnchors]:
        anchors: dict[str, JointMotorAnchors] = {}
        for index, name in enumerate(ARM_JOINT_NAMES):
            low, high = self._joint_limits[index]
            endpoint = self._endpoint_calibrations[name]
            span = endpoint.normalized_at_upper_limit - endpoint.normalized_at_lower_limit

            def q_at(normalized: float) -> float:
                return float(
                    low
                    + (normalized - endpoint.normalized_at_lower_limit)
                    / span
                    * (high - low)
                )

            anchors[name] = JointMotorAnchors(q_at(-100.0), q_at(100.0))
        return anchors

    @staticmethod
    def _coerce_gripper_calibration(
        calibration: GripperEndpointCalibration | Sequence[float] | None,
    ) -> GripperEndpointCalibration | None:
        if calibration is None:
            return None
        if not isinstance(calibration, GripperEndpointCalibration):
            calibration = GripperEndpointCalibration(*calibration)
        values = np.asarray(
            (
                calibration.q_closed,
                calibration.q_open,
                calibration.normalized_closed,
                calibration.normalized_open,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("gripper endpoint calibration must contain finite values")
        if np.isclose(calibration.q_closed, calibration.q_open):
            raise ValueError("gripper model endpoints must be distinct")
        if np.isclose(calibration.normalized_closed, calibration.normalized_open):
            raise ValueError("gripper normalized endpoints must be distinct")
        if (
            calibration.normalized_closed < -1e-9
            or calibration.normalized_closed > 100.0 + 1e-9
            or calibration.normalized_open < -1e-9
            or calibration.normalized_open > 100.0 + 1e-9
        ):
            raise ValueError("gripper normalized endpoints must be in [0, 100]")
        return calibration

    def _require_gripper_calibration(self) -> GripperEndpointCalibration:
        if self._gripper_calibration is None:
            raise RuntimeError("this motor mapper has no calibrated gripper endpoints")
        return self._gripper_calibration

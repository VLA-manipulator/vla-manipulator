"""Bridge between the environment's radians and LeRobot's motor space.

:class:`~mj_env.env.SO101TabletopEnv` speaks native MuJoCo units so the physics
stays honest.  A LeRobot SO101 follower instead reports and accepts *normalized*
motor positions: ``RANGE_M100_100`` (-100..100) for the five arm joints and
``RANGE_0_100`` (0..100) for the gripper.  A PI05 checkpoint trained on SO101
LeRobot data therefore expects the normalized space, and this adapter is the
only place that conversion lives.

Caveat worth repeating before trusting an evaluation: on real hardware the
normalization anchors come from the follower's calibration file (per-motor
``range_min`` / ``range_max`` encoder counts).  Simulation has no encoders, so
the anchors here are the MJCF ``jnt_range`` limits.  The two agree only if the
arm was calibrated to its full mechanical travel.  When evaluating a checkpoint
trained on a specific physical arm, check that arm's calibration JSON (see
``so101/so101_follower_01.json``) against these limits first.
"""

from __future__ import annotations

import numpy as np

from .env import JOINT_NAMES, SO101TabletopEnv

#: LeRobot's normalized output range per joint, in ``JOINT_NAMES`` order.
NORMALIZED_RANGES: tuple[tuple[float, float], ...] = (
    (-100.0, 100.0),  # shoulder_pan
    (-100.0, 100.0),  # shoulder_lift
    (-100.0, 100.0),  # elbow_flex
    (-100.0, 100.0),  # wrist_flex
    (-100.0, 100.0),  # wrist_roll
    (0.0, 100.0),  # gripper
)


class LeRobotSO101Adapter:
    """Convert between environment radians and LeRobot normalized units."""

    def __init__(self, env: SO101TabletopEnv) -> None:
        self.joint_names = JOINT_NAMES
        self._radian_low = np.asarray(env.action_space.low, dtype=np.float64)
        self._radian_high = np.asarray(env.action_space.high, dtype=np.float64)
        self._norm_low = np.asarray([lo for lo, _ in NORMALIZED_RANGES], dtype=np.float64)
        self._norm_high = np.asarray([hi for _, hi in NORMALIZED_RANGES], dtype=np.float64)
        self._radian_span = self._radian_high - self._radian_low
        self._norm_span = self._norm_high - self._norm_low

    @property
    def radian_limits(self) -> np.ndarray:
        """The ``(6, 2)`` radian limits used as normalization anchors."""
        return np.stack([self._radian_low, self._radian_high], axis=1)

    def to_lerobot(self, radians: np.ndarray) -> np.ndarray:
        """Radians -> normalized motor units."""
        values = self._coerce(radians, "radians")
        fraction = (values - self._radian_low) / self._radian_span
        return (self._norm_low + fraction * self._norm_span).astype(np.float32)

    def from_lerobot(self, normalized: np.ndarray) -> np.ndarray:
        """Normalized motor units -> radians, clipped to the actuator range."""
        values = self._coerce(normalized, "normalized action")
        fraction = (values - self._norm_low) / self._norm_span
        radians = self._radian_low + fraction * self._radian_span
        return np.clip(radians, self._radian_low, self._radian_high).astype(np.float32)

    def to_so100_degrees(self, radians: np.ndarray) -> np.ndarray:
        """Current SO101 MuJoCo radians -> legacy SO100 policy coordinates.

        The official ``lerobot/smolvla_base`` checkpoint was pretrained with
        ``so100_follower`` using ``use_degrees=True``.  The old SO100 and new
        SO101 calibration frames differ by 90 degrees at shoulder lift and
        elbow flex.  The gripper remains LeRobot's 0..100 coordinate.
        """
        values = self._coerce(radians, "radians")
        degrees = np.rad2deg(values[:5])
        gripper = self.to_lerobot(values)[5]
        return np.asarray(
            (
                degrees[0],
                90.0 - degrees[1],
                degrees[2] + 90.0,
                degrees[3],
                degrees[4],
                gripper,
            ),
            dtype=np.float32,
        )

    def from_so100_degrees(self, action: np.ndarray) -> np.ndarray:
        """Legacy SO100 degree/gripper action -> current MuJoCo radians."""
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape != self._radian_low.shape or not np.all(np.isfinite(values)):
            raise ValueError("expected six finite SO100 action values")
        radians = np.asarray(
            (
                np.deg2rad(values[0]),
                np.deg2rad(90.0 - values[1]),
                np.deg2rad(values[2] - 90.0),
                np.deg2rad(values[3]),
                np.deg2rad(values[4]),
                0.0,
            ),
            dtype=np.float64,
        )
        gripper_normalized = np.clip(values[5], self._norm_low[5], self._norm_high[5])
        gripper_fraction = (gripper_normalized - self._norm_low[5]) / self._norm_span[5]
        radians[5] = self._radian_low[5] + gripper_fraction * self._radian_span[5]
        return np.clip(radians, self._radian_low, self._radian_high).astype(np.float32)

    def observation_to_lerobot(self, observation: dict) -> dict:
        """Copy an observation with ``observation.state`` in normalized units."""
        converted = dict(observation)
        converted["observation.state"] = self.to_lerobot(observation["observation.state"])
        return converted

    def _coerce(self, values: np.ndarray, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != self._radian_low.shape:
            raise ValueError(
                f"expected {len(self.joint_names)} {label} values, got {array.shape[0]}"
            )
        return array


def to_degrees(radians: np.ndarray) -> np.ndarray:
    """Radians -> degrees, for logging and for comparing against a real arm."""
    return np.rad2deg(np.asarray(radians, dtype=np.float64)).astype(np.float32)


def from_degrees(degrees: np.ndarray) -> np.ndarray:
    """Degrees -> radians."""
    return np.deg2rad(np.asarray(degrees, dtype=np.float64)).astype(np.float32)


__all__ = ["NORMALIZED_RANGES", "LeRobotSO101Adapter", "from_degrees", "to_degrees"]

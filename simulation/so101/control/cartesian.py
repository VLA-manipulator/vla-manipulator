"""Bounded Cartesian-position commands for the five-axis SO-101 arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from so101.kinematics import SO101Arm


@dataclass(frozen=True)
class CartesianCommand:
    """One bounded Cartesian command represented in joint-radian space."""

    target_position: np.ndarray
    q_current: np.ndarray
    q_goal: np.ndarray
    q_target: np.ndarray
    goal_position_error: float
    target_position_error: float
    limited: bool


class CartesianPositionController:
    """Convert safe absolute or relative XYZ targets to joint targets.

    The controller is deliberately stateless.  A real control loop must call
    it with the newest measured joint state every cycle, then send its absolute
    target through the LeRobot adapter.  This avoids integrating stale command
    targets when the physical arm lags or gets clipped by LeRobot.
    """

    def __init__(
        self,
        arm: SO101Arm,
        *,
        joint_limits: Sequence[Sequence[float]] | None = None,
        max_joint_delta: float | Sequence[float] = np.deg2rad(3.0),
        max_position_error: float = 0.005,
        max_relative_position: float = 0.02,
        position_tolerance: float = 1e-5,
    ) -> None:
        self.arm = arm
        self.joint_limits = self._coerce_limits(
            arm.joint_limits if joint_limits is None else joint_limits
        )
        self.max_joint_delta = self._coerce_positive_vector(
            max_joint_delta,
            "max_joint_delta",
        )
        self.max_position_error = float(max_position_error)
        self.max_relative_position = float(max_relative_position)
        self.position_tolerance = float(position_tolerance)
        position_limits = np.array(
            [
                self.max_position_error,
                self.max_relative_position,
                self.position_tolerance,
            ]
        )
        if np.any(~np.isfinite(position_limits)) or np.any(position_limits <= 0):
            raise ValueError("position limits must be positive finite values")
        if self.position_tolerance > self.max_position_error:
            raise ValueError("position_tolerance must not exceed max_position_error")

    def absolute_target(
        self,
        q_current: Sequence[float],
        target_position: Sequence[float],
    ) -> CartesianCommand | None:
        """Plan one bounded joint target for an absolute base-frame XYZ goal."""
        current = self._coerce_current(q_current)
        target = np.asarray(target_position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain three finite base-frame metres")

        current_error = float(np.linalg.norm(self.arm.fkine(current).t - target))
        if current_error <= self.position_tolerance:
            return CartesianCommand(
                target_position=target.copy(),
                q_current=current,
                q_goal=current.copy(),
                q_target=current.copy(),
                goal_position_error=current_error,
                target_position_error=current_error,
                limited=False,
            )

        q_goal = self.arm.inverse_position(target, q0=current)
        if q_goal is None:
            return None
        q_goal = np.clip(q_goal, self.joint_limits[:, 0], self.joint_limits[:, 1])
        goal_error = float(np.linalg.norm(self.arm.fkine(q_goal).t - target))
        if goal_error > self.max_position_error:
            raise RuntimeError(
                "requested Cartesian target is unreachable within the configured joint anchors "
                f"(position residual {goal_error:.4f} m)"
            )

        delta = q_goal - current
        ratios = np.divide(
            self.max_joint_delta,
            np.abs(delta),
            out=np.full(self.arm.n, np.inf),
            where=np.abs(delta) > 0,
        )
        step_scale = min(1.0, float(np.min(ratios)))
        q_target = current + step_scale * delta
        target_error = float(np.linalg.norm(self.arm.fkine(q_target).t - target))
        minimum_progress = max(1e-10, self.position_tolerance * 1e-3)
        if target_error >= current_error - minimum_progress:
            raise RuntimeError(
                "bounded joint target does not make Cartesian progress "
                f"({current_error:.4e} m -> {target_error:.4e} m)"
            )
        return CartesianCommand(
            target_position=target.copy(),
            q_current=current,
            q_goal=q_goal,
            q_target=q_target,
            goal_position_error=goal_error,
            target_position_error=target_error,
            limited=step_scale < 1.0 - 1e-12,
        )

    def relative_target(
        self,
        q_current: Sequence[float],
        delta_position: Sequence[float],
        *,
        frame: Literal["base", "tool"] = "base",
    ) -> CartesianCommand | None:
        """Plan one target for a small XYZ delta in base or TCP coordinates."""
        current = self._coerce_current(q_current)
        delta = np.asarray(delta_position, dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_position must contain three finite metres")
        if np.linalg.norm(delta) > self.max_relative_position:
            raise ValueError(
                f"relative Cartesian command exceeds {self.max_relative_position:.3f} m limit"
            )
        pose = self.arm.fkine(current)
        if frame == "tool":
            delta = pose.R @ delta
        elif frame != "base":
            raise ValueError("frame must be 'base' or 'tool'")
        return self.absolute_target(current, pose.t + delta)

    def _coerce_current(self, q_current: Sequence[float]) -> np.ndarray:
        current = np.asarray(q_current, dtype=float)
        if current.shape != (self.arm.n,) or not np.all(np.isfinite(current)):
            raise ValueError(f"q_current must contain {self.arm.n} finite radians")
        if np.any(current < self.joint_limits[:, 0] - 1e-9) or np.any(
            current > self.joint_limits[:, 1] + 1e-9
        ):
            raise ValueError("q_current is outside the configured physical joint anchors")
        return current.copy()

    def _coerce_limits(self, limits: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(limits, dtype=float)
        if array.shape != (self.arm.n, 2) or not np.all(np.isfinite(array)):
            raise ValueError(f"joint_limits must have shape ({self.arm.n}, 2)")
        if np.any(array[:, 0] >= array[:, 1]):
            raise ValueError("each joint limit lower bound must be smaller than upper bound")
        model_limits = self.arm.joint_limits
        if np.any(array[:, 0] < model_limits[:, 0] - 1e-9) or np.any(
            array[:, 1] > model_limits[:, 1] + 1e-9
        ):
            raise ValueError("controller joint limits must stay inside the kinematic model limits")
        return array

    def _coerce_positive_vector(
        self,
        value: float | Sequence[float],
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(self.arm.n, float(array))
        if array.shape != (self.arm.n,) or np.any(~np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{name} must be a positive scalar or shape ({self.arm.n},)")
        return array

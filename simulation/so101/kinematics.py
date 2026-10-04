"""Five-axis SO-101 kinematics in radians.

The chain is an explicit Robotics Toolbox ETS model derived from
``asserts/SO101/so101_new_calib.urdf`` and its matching MuJoCo model.  The
published gripper-frame link is the TCP.  The moving jaw is deliberately not
part of this chain: it is the sixth motor, but not a sixth arm degree of
freedom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.Robot import Robot
from scipy.optimize import NonlinearConstraint, least_squares, minimize
from scipy.spatial.transform import Rotation
from spatialmath import SE3, Twist3, UnitQuaternion


from so101.joints import ARM_JOINT_NAMES, ALL_JOINT_NAMES

# New-calibration ranges from asserts/SO101/so101_new_calib.xml.  These are
# kinematic soft limits, not a substitute for a physical arm's calibration.
JOINT_LIMITS = np.array(
    [
        [-1.9198621771937616, 1.9198621771937634],
        [-1.7453292519943224, 1.7453292519943366],
        [-1.69, 1.69],
        [-1.6580628494556928, 1.6580627293335335],
        [-2.7438472969992493, 2.841206309382605],
    ],
    dtype=float,
)
POSITION_MASK = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
ORIENTATION_AXIS_NAMES = ("rx", "ry", "rz")


@dataclass(frozen=True)
class IKResult:
    """Diagnostic result for :meth:`SO101Arm.inverse_pose`.

    For a successful solve, ``position_error`` is target minus achieved TCP
    position in metres and ``orientation_error`` is the base-frame rotation
    vector from the achieved orientation to the target, in radians.  A failed
    solve has ``success=False``, ``q=None``, error fields set to ``None``, and
    a human-readable ``message``.
    """

    success: bool
    mode: str
    q: np.ndarray | None = None
    position_error: np.ndarray | None = None
    position_error_norm: float | None = None
    orientation_error: np.ndarray | None = None
    orientation_error_norm: float | None = None
    selected_soft_axis: str | None = None
    message: str | None = None


def _fixed_transform(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> SE3:
    """Build an SE3 transform from MuJoCo's position and WXYZ quaternion."""
    return SE3.Rt(
        UnitQuaternion(np.asarray(quaternion_wxyz, dtype=float)).R,
        np.asarray(position, dtype=float),
    )


class SO101Arm(Robot):
    """Explicit five-axis SO-101 arm model with a gripper-frame TCP.

    All joint positions and returned poses use SI units: radians and metres.
    ``fkine`` and ``jacob0`` are inherited from :class:`Robot`.  Five joints
    cannot realize arbitrary six-dimensional poses, so :meth:`inverse_pose`
    uses ``rz`` as its default soft orientation component while keeping XYZ,
    ``rx``, and ``ry`` hard-constrained.
    """

    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    JOINT_LIMITS = JOINT_LIMITS
    POSITION_MASK = POSITION_MASK

    # Parent-body transforms in the new-calibration MJCF.  Each active joint
    # rotates around its body's local +Z after its fixed parent transform.
    _JOINT_TRANSFORMS = (
        _fixed_transform(
            (0.0388353, -8.97657e-09, 0.0624),
            (3.56167e-16, 1.22818e-15, -1.0, -4.14635e-16),
        ),
        _fixed_transform(
            (-0.0303992, -0.0182778, -0.0542),
            (0.5, -0.5, -0.5, -0.5),
        ),
        _fixed_transform(
            (-0.11257, -0.028, 1.73763e-16),
            (0.707107, -5.98613e-17, -2.58051e-17, 0.707107),
        ),
        _fixed_transform(
            (-0.1349, 0.0052, 3.62355e-17),
            (0.707107, 9.58722e-16, -7.51313e-16, -0.707107),
        ),
        _fixed_transform(
            (5.55112e-17, -0.0611, 0.0181),
            (0.0172091, -0.0172091, 0.706897, 0.706897),
        ),
    )

    # gripper_frame_joint in the new-calibration URDF.  This is the public
    # TCP frame used by FK/IK, not the rotating jaw frame.
    TCP_TRANSFORM = SE3.Rt(
        SE3.Ry(np.pi).R,
        np.array((-0.0079, -0.000218121, -0.0981274), dtype=float),
    )

    # The MJCF site named ``gripperframe`` shares the TCP position but has a
    # fixed frame convention difference.  Right-multiply this transform to
    # report the URDF TCP convention from MuJoCo site data.
    MUJOCO_SITE_TO_TCP = SE3.Ry(np.pi / 2)

    def __init__(self) -> None:
        links: list[Link] = []
        parent: Link | None = None
        for name, qlim, fixed in zip(
            self.ARM_JOINT_NAMES,
            self.JOINT_LIMITS,
            self._JOINT_TRANSFORMS,
            strict=True,
        ):
            link = Link(
                ET.SE3(fixed) * ET.Rz(qlim=qlim),
                name=name,
                parent=parent,
            )
            links.append(link)
            parent = link

        tcp = Link(ET.SE3(self.TCP_TRANSFORM), name="gripper_frame_link", parent=parent)
        super().__init__(
            [*links, tcp],
            name="SO101Arm",
            manufacturer="TheRobotStudio / Hugging Face",
            comment="Explicit five-axis ETS model from SO-101 new calibration",
        )

        self.qz = np.zeros(self.n, dtype=float)
        self.q_home = self.qz.copy()
        self.addconfiguration("qz", self.qz)
        self.addconfiguration("q_home", self.q_home)

    @property
    def joint_limits(self) -> np.ndarray:
        """Copy of the five kinematic joint limits in radians."""
        return self.JOINT_LIMITS.copy()

    @property
    def tcp_transform(self) -> SE3:
        """TCP transform expressed in the wrist-roll/gripper-link frame."""
        return SE3(self.TCP_TRANSFORM.A.copy(), check=False)

    @property
    def mujoco_site_to_tcp(self) -> SE3:
        """Fixed transform from MJCF ``gripperframe`` site to this TCP."""
        return SE3(self.MUJOCO_SITE_TO_TCP.A.copy(), check=False)

    def inverse_position(
        self,
        target_position: Sequence[float],
        q0: Sequence[float] | None = None,
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 80,
    ) -> np.ndarray | None:
        """Solve an XYZ-only inverse-kinematics task.

        Args:
            target_position: TCP position in the base frame, in metres.
            q0: Initial five-joint configuration in radians.  Supplying the
                latest measured configuration selects the locally continuous
                branch of this redundant position task.
        """
        position = np.asarray(target_position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("target_position must be three finite base-frame metres")
        return self.inverse_pose(
            SE3.Trans(position),
            q0=q0,
            mask=self.POSITION_MASK,
            mode="masked",
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def inverse_pose(
        self,
        target_pose: SE3 | Sequence[float] | np.ndarray,
        q0: Sequence[float] | None = None,
        *,
        mask: Sequence[float] | None = None,
        mode: str | None = None,
        soft_axis: str | None = None,
        orientation_weights: Sequence[float] | None = None,
        return_result: bool = False,
        orientation: str | None = None,
        order: str = "zyx",
        unit: str = "rad",
        tolerance: float = 1e-6,
        max_iterations: int = 80,
    ) -> np.ndarray | IKResult | None:
        """Solve a deterministic, locally continuous TCP task.

        With neither ``mode`` nor ``mask`` supplied, the default is
        ``mode="soft_axis", soft_axis="rz"``.  Supplying ``mask`` without a
        mode preserves the established Robotics Toolbox-style masked path.
        A mask may activate at most five task dimensions.  The modes are:

        - ``"soft_axis"``: two rotation-vector components are hard and the
          named ``soft_axis`` (``"rx"``, ``"ry"``, or ``"rz"``) is soft.
        - ``"auto_soft_axis"``: evaluate all three soft-axis choices and
          select the feasible one with the smallest full orientation error.
        - ``"soft_orientation"``: make all orientation components soft and
          minimize their weighted rotation-vector norm at the requested TCP
          position.  ``orientation_weights`` has three positive entries in
          ``rx, ry, rz`` order and defaults to equal weights.

        Rotation-vector components are base-frame angular error components,
        not RPY/Euler-angle differences.  All modes begin from ``q0`` and use
        no random restarts.  ``return_result=True`` always returns
        :class:`IKResult`; check its ``success`` field before using ``q``.
        The compatibility return is a five-value joint array on success and
        ``None`` when no solution is found.  Invalid inputs still raise
        :class:`ValueError`.
        """
        pose = self._coerce_pose(
            target_pose,
            orientation=orientation,
            order=order,
            unit=unit,
        )
        q_initial = self.q_home if q0 is None else self._coerce_joint_vector(q0, "q0")
        limits = self.joint_limits
        if np.any(q_initial < limits[:, 0]) or np.any(q_initial > limits[:, 1]):
            raise ValueError("q0 is outside the SO-101 joint limits")

        tolerance = float(tolerance)
        max_iterations = int(max_iterations)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        resolved_mode = mode
        if resolved_mode is None:
            resolved_mode = "masked" if mask is not None else "soft_axis"

        target_matrix = pose.A
        selected_soft_axis: str | None = None
        if resolved_mode != "soft_orientation" and orientation_weights is not None:
            raise ValueError("orientation_weights is only valid in mode='soft_orientation'")
        try:
            if resolved_mode == "masked":
                task_mask = self._coerce_mask(mask)
                q_solution = self._solve_hard_mask(
                    target_matrix,
                    q_initial,
                    task_mask,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            elif resolved_mode == "soft_axis":
                self._reject_mask_for_mode(mask, resolved_mode)
                selected_soft_axis = self._coerce_soft_axis(
                    "rz" if soft_axis is None else soft_axis,
                    required=True,
                )
                q_solution = self._solve_soft_axis(
                    target_matrix,
                    q_initial,
                    selected_soft_axis,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            elif resolved_mode == "auto_soft_axis":
                self._reject_mask_for_mode(mask, resolved_mode)
                if soft_axis is not None:
                    raise ValueError(
                        "soft_axis is selected automatically in mode='auto_soft_axis'"
                    )
                q_solution, selected_soft_axis = self._solve_auto_soft_axis(
                    target_matrix,
                    q_initial,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            elif resolved_mode == "soft_orientation":
                self._reject_mask_for_mode(mask, resolved_mode)
                if soft_axis is not None:
                    raise ValueError("soft_axis is only valid in mode='soft_axis'")
                weights = self._coerce_orientation_weights(orientation_weights)
                q_solution = self._solve_soft_orientation(
                    target_matrix,
                    q_initial,
                    weights,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            else:
                raise ValueError(
                    "mode must be 'masked', 'soft_axis', 'auto_soft_axis', or "
                    "'soft_orientation'"
                )
        except RuntimeError as error:
            failure = IKResult(
                success=False,
                mode=resolved_mode,
                selected_soft_axis=selected_soft_axis,
                message=str(error),
            )
            return failure if return_result else None

        result = self._make_ik_result(
            q_solution,
            target_matrix,
            mode=resolved_mode,
            selected_soft_axis=selected_soft_axis,
        )
        if return_result:
            return result
        if result.q is None:
            return None
        return result.q.copy()

    def inverse_twist(
        self,
        target_twist: Twist3 | Sequence[float],
        q: Sequence[float],
        *,
        mask: Sequence[float] | None = None,
        damping: float = 0.02,
        max_joint_speed: float | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Map a constrained base-frame twist to bounded joint velocities.

        A damped least-squares solve is used instead of an unregularized
        pseudoinverse so a near-singular wrist does not create unbounded joint
        velocities.  A three-value input is interpreted as XYZ velocity.
        """
        q_array = self._coerce_joint_vector(q, "q")
        task_mask = self._coerce_mask(mask)
        damping = float(damping)
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be positive")

        if isinstance(target_twist, Twist3):
            twist = np.asarray(target_twist.S, dtype=float)
        else:
            twist = np.asarray(target_twist, dtype=float)
            if twist.shape == (3,):
                twist = np.r_[twist, np.zeros(3, dtype=float)]
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            raise ValueError("target_twist must contain three or six finite values")

        active = task_mask > 0
        jacobian = self.jacob0(q_array)[active] * task_mask[active, None]
        task_velocity = twist[active] * task_mask[active]
        normal = jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0])
        qd = jacobian.T @ np.linalg.solve(normal, task_velocity)

        if max_joint_speed is not None:
            limit = np.asarray(max_joint_speed, dtype=float)
            if limit.ndim == 0:
                limit = np.full(self.n, float(limit))
            if (
                limit.shape != (self.n,)
                or np.any(~np.isfinite(limit))
                or np.any(limit <= 0)
            ):
                raise ValueError(f"max_joint_speed must be positive scalar or shape ({self.n},)")
            qd = np.clip(qd, -limit, limit)
        return qd

    def _coerce_joint_vector(self, q: Sequence[float], name: str) -> np.ndarray:
        q_array = np.asarray(q, dtype=float)
        if q_array.shape != (self.n,) or not np.all(np.isfinite(q_array)):
            raise ValueError(f"{name} must contain {self.n} finite joint values")
        return q_array.copy()

    def _coerce_mask(self, mask: Sequence[float] | None) -> np.ndarray:
        task_mask = self.POSITION_MASK if mask is None else np.asarray(mask, dtype=float)
        if task_mask.shape != (6,) or not np.all(np.isfinite(task_mask)):
            raise ValueError("mask must contain six finite task weights")
        if np.any(task_mask < 0) or not np.any(task_mask > 0):
            raise ValueError("mask must contain at least one positive task weight")
        if np.count_nonzero(task_mask) > self.n:
            raise ValueError(
                "SO-101 has five arm DOF; a mask can activate at most five task dimensions"
            )
        return task_mask.copy()

    @staticmethod
    def _reject_mask_for_mode(mask: Sequence[float] | None, mode: str) -> None:
        if mask is not None:
            raise ValueError(f"mask cannot be combined with mode={mode!r}")

    @staticmethod
    def _coerce_soft_axis(axis: str | None, *, required: bool) -> str | None:
        if axis is None:
            if required:
                raise ValueError("soft_axis must be one of 'rx', 'ry', or 'rz'")
            return None
        if axis not in ORIENTATION_AXIS_NAMES:
            raise ValueError("soft_axis must be one of 'rx', 'ry', or 'rz'")
        return axis

    @staticmethod
    def _coerce_orientation_weights(weights: Sequence[float] | None) -> np.ndarray:
        if weights is None:
            return np.ones(3, dtype=float)
        values = np.asarray(weights, dtype=float)
        if (
            values.shape != (3,)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError("orientation_weights must contain three positive finite values")
        return values.copy()

    def _solve_hard_mask(
        self,
        target: np.ndarray,
        q_initial: np.ndarray,
        task_mask: np.ndarray,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> np.ndarray:
        def residual(q: np.ndarray) -> np.ndarray:
            return self._task_residual(q, target, task_mask)

        initial_residual = residual(q_initial)
        if np.linalg.norm(initial_residual, ord=np.inf) <= tolerance:
            return q_initial.copy()

        limits = self.joint_limits
        solver_tolerance = max(1e-12, min(tolerance, 1e-8))
        solution = least_squares(
            residual,
            q_initial,
            bounds=(limits[:, 0], limits[:, 1]),
            method="trf",
            max_nfev=max_iterations,
            ftol=solver_tolerance,
            xtol=solver_tolerance,
            gtol=solver_tolerance,
            x_scale="jac",
        )
        final_residual = residual(solution.x)
        maximum_error = float(np.linalg.norm(final_residual, ord=np.inf))
        if not solution.success or maximum_error > tolerance:
            raise RuntimeError(
                "SO-101 local IK did not converge within the active task tolerance "
                f"({maximum_error:.3e}): {solution.message}"
            )
        return self._coerce_joint_vector(solution.x, "IK solution")

    def _solve_soft_axis(
        self,
        target: np.ndarray,
        q_initial: np.ndarray,
        soft_axis: str,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> np.ndarray:
        task_mask = np.ones(6, dtype=float)
        soft_index = 3 + ORIENTATION_AXIS_NAMES.index(soft_axis)
        task_mask[soft_index] = 0.0
        q_hard = self._solve_hard_mask(
            target,
            q_initial,
            task_mask,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        def full_error(q: np.ndarray) -> np.ndarray:
            position_error, orientation_error = self._pose_errors(q, target)
            return np.r_[position_error, orientation_error]

        hard_dimensions = task_mask > 0

        def hard_error(q: np.ndarray) -> np.ndarray:
            return full_error(q)[hard_dimensions]

        def objective(q: np.ndarray) -> float:
            error = full_error(q)[soft_index]
            return float(error * error)

        initial_objective = objective(q_hard)
        if initial_objective <= tolerance**2:
            return q_hard

        hard_constraint = NonlinearConstraint(
            hard_error,
            -np.full(self.n, tolerance, dtype=float),
            np.full(self.n, tolerance, dtype=float),
        )
        solution = minimize(
            objective,
            q_hard,
            method="SLSQP",
            bounds=[tuple(limit) for limit in self.joint_limits],
            constraints=(hard_constraint,),
            options={
                "maxiter": max_iterations,
                "ftol": max(1e-12, min(tolerance**2, 1e-10)),
            },
        )
        q_candidate = self._coerce_joint_vector(solution.x, "IK solution")
        maximum_hard_error = float(np.linalg.norm(hard_error(q_candidate), ord=np.inf))
        hard_limit = tolerance + max(1e-12, tolerance * 1e-8)
        if not solution.success or maximum_hard_error > hard_limit:
            raise RuntimeError(
                "SO-101 soft-axis IK did not converge within the hard task "
                f"tolerance ({maximum_hard_error:.3e}): {solution.message}"
            )
        if objective(q_candidate) > initial_objective + 1e-12:
            return q_hard
        return q_candidate

    def _solve_auto_soft_axis(
        self,
        target: np.ndarray,
        q_initial: np.ndarray,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, str]:
        candidates: list[tuple[tuple[float, float, int], np.ndarray, str]] = []
        normalized_ranges = self.JOINT_LIMITS[:, 1] - self.JOINT_LIMITS[:, 0]
        failures: list[str] = []
        for axis_index, axis in enumerate(ORIENTATION_AXIS_NAMES):
            try:
                q = self._solve_soft_axis(
                    target,
                    q_initial,
                    axis,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            except RuntimeError as error:
                failures.append(f"{axis}: {error}")
                continue
            result = self._make_ik_result(q, target, mode="auto_soft_axis", selected_soft_axis=axis)
            continuity = float(np.linalg.norm((q - q_initial) / normalized_ranges))
            candidates.append(((result.orientation_error_norm, continuity, axis_index), q, axis))
        if not candidates:
            detail = "; ".join(failures)
            raise RuntimeError(f"SO-101 auto soft-axis IK found no feasible task: {detail}")
        _, q_solution, selected_axis = min(candidates, key=lambda candidate: candidate[0])
        return q_solution, selected_axis

    def _solve_soft_orientation(
        self,
        target: np.ndarray,
        q_initial: np.ndarray,
        orientation_weights: np.ndarray,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> np.ndarray:
        q_position = self._solve_hard_mask(
            target,
            q_initial,
            self.POSITION_MASK,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        def position_error(q: np.ndarray) -> np.ndarray:
            return self._pose_errors(q, target)[0]

        def objective(q: np.ndarray) -> float:
            orientation_error = self._pose_errors(q, target)[1]
            weighted_error = orientation_weights * orientation_error
            return float(weighted_error @ weighted_error)

        initial_objective = objective(q_position)
        if initial_objective <= tolerance**2:
            return q_position
        position_constraint = NonlinearConstraint(
            position_error,
            -np.full(3, tolerance, dtype=float),
            np.full(3, tolerance, dtype=float),
        )
        solution = minimize(
            objective,
            q_position,
            method="SLSQP",
            bounds=[tuple(limit) for limit in self.joint_limits],
            constraints=(position_constraint,),
            options={
                "maxiter": max_iterations,
                "ftol": max(1e-12, min(tolerance**2, 1e-10)),
            },
        )
        q_candidate = self._coerce_joint_vector(solution.x, "IK solution")
        maximum_position_error = float(np.linalg.norm(position_error(q_candidate), ord=np.inf))
        position_limit = tolerance + max(1e-12, tolerance * 1e-8)
        if not solution.success or maximum_position_error > position_limit:
            raise RuntimeError(
                "SO-101 position-constrained orientation IK did not converge "
                f"within the position tolerance ({maximum_position_error:.3e}): {solution.message}"
            )
        if objective(q_candidate) > initial_objective + 1e-12:
            return q_position
        return q_candidate

    def _make_ik_result(
        self,
        q: np.ndarray,
        target: np.ndarray,
        *,
        mode: str,
        selected_soft_axis: str | None,
    ) -> IKResult:
        position_error, orientation_error = self._pose_errors(q, target)
        return IKResult(
            success=True,
            mode=mode,
            q=self._coerce_joint_vector(q, "IK solution"),
            position_error=position_error,
            position_error_norm=float(np.linalg.norm(position_error)),
            orientation_error=orientation_error,
            orientation_error_norm=float(np.linalg.norm(orientation_error)),
            selected_soft_axis=selected_soft_axis,
        )

    def _pose_errors(self, q: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current = self._forward_matrix(q)
        position_error = target[:3, 3] - current[:3, 3]
        rotation_error = target[:3, :3] @ current[:3, :3].T
        orientation_error = Rotation.from_matrix(rotation_error).as_rotvec()
        return position_error, orientation_error

    def _task_residual(
        self,
        q: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        position_error, orientation_error = self._pose_errors(q, target)
        error = np.zeros(6, dtype=float)
        error[:3] = position_error
        if np.any(mask[3:] > 0):
            error[3:] = orientation_error
        active = mask > 0
        return error[active] * mask[active]

    def _forward_matrix(self, q: Sequence[float]) -> np.ndarray:
        """Evaluate this fixed ETS with NumPy for the local IK residual."""
        q_array = np.asarray(q, dtype=float)
        transform = np.eye(4, dtype=float)
        for fixed, angle in zip(self._JOINT_TRANSFORMS, q_array, strict=True):
            cosine = np.cos(angle)
            sine = np.sin(angle)
            rotation_z = np.array(
                [
                    [cosine, -sine, 0.0, 0.0],
                    [sine, cosine, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            transform = transform @ fixed.A @ rotation_z
        return transform @ self.TCP_TRANSFORM.A

    @staticmethod
    def _coerce_pose(
        target_pose: SE3 | Sequence[float] | np.ndarray,
        *,
        orientation: str | None,
        order: str,
        unit: str,
    ) -> SE3:
        if isinstance(target_pose, SE3):
            if orientation not in (None, "se3"):
                raise ValueError("orientation must be omitted when target_pose is SE3")
            return target_pose

        pose_array = np.asarray(target_pose, dtype=float)
        if pose_array.shape == (4, 4):
            if orientation not in (None, "matrix", "se3"):
                raise ValueError("orientation must be omitted for a 4x4 pose matrix")
            return SE3(pose_array, check=True)
        if pose_array.shape == (6,):
            if orientation != "rpy":
                raise ValueError("a six-value pose requires orientation='rpy'")
            return SE3.Trans(pose_array[:3]) * SE3.RPY(
                pose_array[3:],
                order=order,
                unit=unit,
            )
        raise TypeError("target_pose must be SE3, a 4x4 matrix, or XYZ+RPY")

    def __repr__(self) -> str:
        return "SO101Arm(n=5, calibration='new')"

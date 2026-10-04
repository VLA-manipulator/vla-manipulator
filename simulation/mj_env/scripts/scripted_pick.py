"""IK-driven expert pick, used as the environment's solvability baseline.

Run from the repository root::

    python -m mj_env.scripts.scripted_pick --all
    python -m mj_env.scripts.scripted_pick --object dumbbell_purple --episodes 5

This is deliberately *not* a general grasp planner.  It is a fixed
top-down-approach script that knows each object's grasp axis, so its success
rate answers one question: is the task physically solvable in this scene?  When
PI05 fails, run this first -- if the baseline also fails, the problem is the
environment, not the policy.

Grasp geometry.  With the TCP commanded to ``Trans(p) * Ry(pi) * Rz(psi)``,
``psi`` rotates the jaw hinge in the horizontal plane as
``hinge_yaw = -pi/2 - psi``.  Since the fingers close across the hinge,
aligning the hinge with an object's long axis is what makes the jaw close
across a rod or a dumbbell handle rather than along it.

The IK target is *not* the object.  ``gripperframe`` is the fingertip, while
the pads meet centimetres behind and to one side of it, so the target is the
object's position minus the aperture offset that :mod:`mj_env.gripper`
measures off the finger meshes.  Aiming the fingertip at the object instead
drives the pads straight past it -- the jaw then closes on empty air, which
looks exactly like a policy failure.

Five joints cannot hold an arbitrary pose, so two accommodations matter.  The
jaw is symmetric, so ``psi`` is folded into [-pi/2, pi/2]; without that the
required ``wrist_roll`` runs past its limit for roughly a fifth of object
headings.  And when even the folded roll has no exact solution, the script
falls back to ``auto_soft_axis``, which keeps the TCP position and gives up
the least-costly orientation component instead of refusing to grasp.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Callable

import mujoco
import numpy as np
from spatialmath import SE3

from so101 import SO101Arm

from mj_env import gripper, objects
from mj_env.env import HOME_QPOS, SO101TabletopEnv

#: Extra jaw opening over the object's width while approaching, and how far
#: past the object's width the jaw is commanded so the pads keep pressing.
JAW_CLEARANCE = 0.022
JAW_SQUEEZE = 0.006

#: Half-gap left between each pad and the object while descending.  The fixed
#: finger does not move when the jaw opens, so aiming the closed aperture at
#: the object puts that pad flush against its side and the descent scrapes
#: down it -- enough to tip a cube over before the jaw ever closes.  Aiming a
#: slightly wider aperture instead centres the object between the pads with
#: clearance on both sides.
PAD_CLEARANCE = 0.004

#: Elbow-up seed for the planar joints.  ``inverse_pose`` does no random
#: restarts, so seeding shoulder_pan and wrist_roll near their analytic values
#: is what keeps the solver off local minima.
_PLANAR_SEED = (-0.6, 1.1, 0.7)
#: wrist_roll is very nearly ``_ROLL_OFFSET - psi`` for a vertical approach.
_ROLL_OFFSET = -0.212
#: Position error above which an IK solution is treated as a miss.
_POSITION_TOLERANCE = 0.003

# Small, reproducible perturbations around the retracted HOME_QPOS.  The arm
# perturbation is deliberately bounded so it starts above the table and the
# transit-to-hover motion cannot sweep through the object layout.  The gripper
# itself is sampled across its full commandable range.
_ARM_RANDOM_RANGES = np.array((0.35, 0.20, 0.20, 0.20, 0.40), dtype=float)
_ARM_RANDOM_ATTEMPTS = 64
_ARM_MIN_TCP_HEIGHT = 0.18


class ViewerInterrupted(RuntimeError):
    """Raised when the interactive viewer is closed (ESC or window close)."""


def randomize_initial_arm(
    env: SO101TabletopEnv, seed: int | None
) -> np.ndarray:
    """Set a safe, deterministic random arm pose and return its six commands.

    Object placement is randomized by :meth:`SO101TabletopEnv.reset`; this
    function adds independent arm-pose and gripper variation so demonstrations
    do not all begin from the same configuration.  The gripper is sampled from
    its commandable range; the first hover waypoint later opens it as needed
    for the selected object.
    """
    rng = np.random.default_rng(None if seed is None else int(seed) ^ 0x5EED)
    home = np.asarray(HOME_QPOS, dtype=float)
    low = env.action_space.low.astype(float)
    high = env.action_space.high.astype(float)
    for _ in range(_ARM_RANDOM_ATTEMPTS):
        q = home.copy()
        q[:5] += rng.uniform(-_ARM_RANDOM_RANGES, _ARM_RANDOM_RANGES)
        q[:5] = np.clip(q[:5], low[:5], high[:5])
        q[5] = rng.uniform(low[5], high[5])
        env.data.qpos[env._qpos_ids] = q
        env.data.qvel[:] = 0.0
        env.data.ctrl[env._actuator_ids] = q
        mujoco.mj_forward(env.model, env.data)
        tcp_z = float(env.data.site_xpos[env._tcp_site_id][2])
        if tcp_z >= _ARM_MIN_TCP_HEIGHT:
            return q

    # HOME itself is guaranteed to be safe by the environment definition.
    env.data.qpos[env._qpos_ids] = home
    env.data.qvel[:] = 0.0
    env.data.ctrl[env._actuator_ids] = home
    mujoco.mj_forward(env.model, env.data)
    return home


def ik_seed(point: np.ndarray, psi: float) -> np.ndarray:
    """Analytic starting guess for a vertical approach at ``point``."""
    return np.array(
        [-np.arctan2(point[1], point[0]), *_PLANAR_SEED, _wrap(_ROLL_OFFSET - psi)]
    )


#: Fixed-point iterations used to place the jaw aperture on the object.
_APERTURE_ITERATIONS = 6
_APERTURE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PickTuning:
    """Waypoint offsets, in metres and control steps."""

    #: Absolute height of the transit waypoint, clear of every object, so the
    #: trip from the home pose cannot sweep the arm through the layout.
    hover_height: float = 0.20
    approach_height: float = 0.10
    #: How far above the object's centre to aim the jaw aperture.  Zero grips
    #: through the middle; raise it to grip nearer the top of a tall object.
    grasp_rise: float = 0.0
    lift_height: float = 0.14
    steps_hover: int = 30
    steps_reach: int = 20
    steps_descend: int = 25
    steps_close: int = 20
    steps_lift: int = 25
    steps_hold: int = 10


def grasp_axis_yaw(env: SO101TabletopEnv, name: str) -> float:
    """Heading of the object's long axis, which the jaw hinge must match."""
    body_id = env._objects[name].body_id
    rotation = env.data.xmat[body_id].reshape(3, 3)
    local_x = rotation[:, 0]
    if abs(local_x[2]) > 0.9:  # long axis stood on end: any heading will do
        return 0.0
    return float(np.arctan2(local_x[1], local_x[0]))


def solve_top_down(
    arm: SO101Arm, point: np.ndarray, psi: float, q0: np.ndarray
) -> np.ndarray | None:
    """Joint solution putting the TCP at ``point`` with jaw roll ``psi``.

    ``soft_orientation`` is tried first because it holds the vertical approach
    exactly wherever that is feasible; ``auto_soft_axis`` is the fallback for
    the poses five joints cannot fully realise.
    """
    pose = SE3(*point) * SE3.Ry(np.pi) * SE3.Rz(psi)
    # Numerical IK updates can overshoot a hard limit by a few ulps.  The
    # kinematics implementation intentionally rejects out-of-range seeds, so
    # sanitize the seed here before retrying a waypoint.
    limits = arm.joint_limits
    epsilon = 1e-8
    q0 = np.clip(np.asarray(q0, dtype=float), limits[:, 0] + epsilon, limits[:, 1] - epsilon)
    for mode in ("soft_orientation", "auto_soft_axis"):
        result = arm.inverse_pose(pose, q0=q0, mode=mode, return_result=True)
        if result.success and result.position_error_norm < _POSITION_TOLERANCE:
            return np.asarray(result.q, dtype=float)
    return None


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class GraspSolver:
    """Turns "hold this object" into joint targets the arm can track."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.arm = SO101Arm()
        self.calibration = gripper.calibrate(model)
        self._model = model
        self._scratch = mujoco.MjData(model)
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        arm_joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in SO101Arm.ARM_JOINT_NAMES
        ]
        self._arm_qpos = np.asarray([model.jnt_qposadr[j] for j in arm_joints], dtype=int)

    def site_rotation(self, q: np.ndarray) -> np.ndarray:
        self._scratch.qpos[:] = self._model.qpos0
        self._scratch.qpos[self._arm_qpos] = q
        mujoco.mj_kinematics(self._model, self._scratch)
        return self._scratch.site_xmat[self._site_id].reshape(3, 3).copy()

    def solve_aperture(
        self, aim: np.ndarray, psi: float, offset: np.ndarray, q0: np.ndarray
    ) -> np.ndarray | None:
        """Joints that put the *jaw aperture* -- not the fingertip -- on ``aim``.

        The offset is expressed in the TCP site frame, whose orientation
        depends on the solution, so the correction is applied as a fixed point.
        It converges in two or three passes because the wrist barely rotates
        between one candidate target and the next.
        """
        target = np.asarray(aim, dtype=float)
        solution = None
        for _ in range(_APERTURE_ITERATIONS):
            solution = solve_top_down(self.arm, target, psi, q0 if solution is None else solution)
            if solution is None:
                return None
            corrected = aim - self.site_rotation(solution) @ offset
            shift = float(np.linalg.norm(corrected - target))
            target = corrected
            if shift < _APERTURE_TOLERANCE:
                return solution
        return solution

    def solve_waypoints(
        self,
        aim: np.ndarray,
        hinge_yaw: float,
        offset: np.ndarray,
        tuning: PickTuning,
    ) -> dict[str, np.ndarray] | None:
        """Solve the whole trajectory under one jaw roll.

        Solving each waypoint independently is what wrecks the grasp: the jaw
        is symmetric, so two rolls a half-turn apart are equally valid, and
        letting different waypoints pick different ones makes the wrist spin
        the object out of the fingers between descending and lifting.  One roll
        is chosen for the whole trajectory, and the grasp pose -- the most
        constrained one -- is solved first so the rest inherit its branch.
        """
        base = -np.pi / 2.0 - hinge_yaw
        # Smaller |psi| first: the half-turn-equivalent roll is the one that
        # keeps wrist_roll away from its limit.
        for psi in sorted((_wrap(base), _wrap(base + np.pi)), key=abs):
            grasp = self.solve_aperture(aim, psi, offset, ik_seed(aim, psi))
            if grasp is None:
                continue
            # Everything else is the grasp pose translated straight up, which
            # keeps the aperture over the object throughout the descent.
            grasp_tcp = aim - self.site_rotation(grasp) @ offset
            above = {
                "reach": grasp_tcp + (0.0, 0.0, tuning.approach_height),
                "hover": np.array([grasp_tcp[0], grasp_tcp[1], tuning.hover_height]),
                "lift": grasp_tcp + (0.0, 0.0, tuning.lift_height),
            }
            solutions = {"grasp": grasp}
            seed = grasp
            for label in ("reach", "hover", "lift"):
                solution = solve_top_down(self.arm, above[label], psi, seed)
                if solution is None:
                    break
                solutions[label] = seed = solution
            else:
                return solutions
        return None


FrameCallback = Callable[[np.ndarray, dict[str, np.ndarray], dict], None]


def _glide(
    env: SO101TabletopEnv,
    target: np.ndarray,
    steps: int,
    on_frame: FrameCallback | None = None,
) -> dict:
    """Ramp to ``target``; callbacks pair the pre-action observation with its action."""
    start = env.data.ctrl[env._actuator_ids].copy()
    info: dict = env._info() if on_frame is not None else {}
    observation = env._observation() if on_frame is not None else None
    for index in range(steps):
        alpha = (index + 1) / steps
        action = start + alpha * (target - start)
        if on_frame is not None:
            on_frame(np.asarray(action, dtype=np.float32).copy(), observation, info)
        observation, _, _, _, info = env.step(action)
        if env.render_mode == "human" and not env.viewer_running:
            raise ViewerInterrupted
    return info


def run_episode(
    env: SO101TabletopEnv,
    solver: GraspSolver,
    *,
    seed: int | None,
    target: str,
    tuning: PickTuning,
    randomize_arm: bool = True,
    on_frame: FrameCallback | None = None,
) -> dict:
    """Reset onto ``target`` and run the scripted pick.  Never raises on failure."""
    _, info = env.reset(seed=seed, options={"target_object": target})
    initial_target_position = np.asarray(info["target_position"], dtype=float).tolist()
    if randomize_arm:
        initial_q = randomize_initial_arm(env, seed)
        if env.render_mode == "human":
            env.render()
            if not env.viewer_running:
                raise ViewerInterrupted
    else:
        initial_q = np.asarray(HOME_QPOS, dtype=float)

    width = env._objects[target].grip_width
    aim = np.asarray(info["target_position"], dtype=float) + (0.0, 0.0, tuning.grasp_rise)
    offset = solver.calibration.offset_for(width + 2.0 * PAD_CLEARANCE)
    try:
        solutions = solver.solve_waypoints(aim, grasp_axis_yaw(env, target), offset, tuning)
    except (ValueError, FloatingPointError):
        solutions = None
    if solutions is None:
        return {
            "target": target,
            "initial_target_position": initial_target_position,
            "initial_qpos": initial_q.tolist(),
            "outcome": "ik_fail",
            "success": False,
        }

    jaw_open = solver.calibration.angle_for(width + JAW_CLEARANCE)
    jaw_shut = solver.calibration.angle_for(max(width - JAW_SQUEEZE, 0.0))

    _glide(env, np.array([*solutions["hover"], jaw_open]), tuning.steps_hover, on_frame)
    _glide(env, np.array([*solutions["reach"], jaw_open]), tuning.steps_reach, on_frame)
    _glide(env, np.array([*solutions["grasp"], jaw_open]), tuning.steps_descend, on_frame)
    _glide(env, np.array([*solutions["grasp"], jaw_shut]), tuning.steps_close, on_frame)
    _glide(env, np.array([*solutions["lift"], jaw_shut]), tuning.steps_lift, on_frame)
    info = _glide(
        env,
        np.array([*solutions["lift"], jaw_shut]),
        tuning.steps_hold,
        on_frame,
    )

    return {
        "target": target,
        "initial_target_position": initial_target_position,
        "initial_qpos": initial_q.tolist(),
        "outcome": "success" if info["success"] else ("dropped" if info["grasped"] else "missed"),
        "success": bool(info["success"]),
        "grasped": bool(info["grasped"]),
        "lift_height": float(info["lift_height"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--object", choices=objects.NAMES, help="run one object")
    group.add_argument("--all", action="store_true", help="run every catalogue object")
    parser.add_argument(
        "--episodes",
        "--episode",
        dest="episodes",
        type=int,
        default=5,
        help="layouts per object (both --episodes and --episode are accepted)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first layout seed")
    parser.add_argument("--grasp-rise", type=float, default=PickTuning.grasp_rise)
    parser.add_argument("--approach-height", type=float, default=PickTuning.approach_height)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the interactive MuJoCo window (default: show it)",
    )
    parser.add_argument(
        "--fixed-arm",
        action="store_true",
        help="keep the arm at HOME_QPOS instead of randomizing its initial pose",
    )
    args = parser.parse_args()

    tuning = PickTuning(grasp_rise=args.grasp_rise, approach_height=args.approach_height)
    targets = objects.NAMES if args.all else (args.object,)

    env = SO101TabletopEnv(
        max_episode_steps=10_000,
        render_mode=None if args.headless else "human",
        show_viewer_ui=False,
        viewer_observation_overlays=True,
    )
    solver = GraspSolver(env.model)
    print(
        f"jaw calibration: opening {solver.calibration.min_opening * 1000:.0f}"
        f"-{solver.calibration.max_opening * 1000:.0f} mm"
    )
    totals = {"success": 0, "grasped": 0, "count": 0}
    try:
        try:
            for target in targets:
                outcomes = []
                for episode in range(args.episodes):
                    result = run_episode(
                        env,
                        solver,
                        seed=args.seed + episode,
                        target=target,
                        tuning=tuning,
                        randomize_arm=not args.fixed_arm,
                    )
                    outcomes.append(result)
                    totals["count"] += 1
                    totals["success"] += int(result["success"])
                    totals["grasped"] += int(result.get("grasped", False))
                hits = sum(r["success"] for r in outcomes)
                detail = " ".join(r["outcome"][:7] for r in outcomes)
                print(f"{target:18s} {hits}/{len(outcomes)}  {detail}")
        except ViewerInterrupted:
            print("\nviewer closed; run interrupted")
            return 0
    finally:
        env.close()

    print(
        f"\nbaseline: lifted {totals['success']}/{totals['count']}, "
        f"gripper closed on the object in {totals['grasped']}/{totals['count']}"
    )
    return 0 if totals["success"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

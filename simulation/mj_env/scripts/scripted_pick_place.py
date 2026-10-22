"""IK-driven pick-and-place expert, and the acceptance gate for a level.

Run from the ``simulation/`` directory::

    python -m mj_env.scripts.scripted_pick_place --task p1_place_disc --episodes 20

This is the P-family baseline: grasp, carry, put down on the goal, let go.  It
exists for one reason -- to answer "is this level solvable at all, and is the
layout band inside the solvable region?" -- before any episode is collected.
A level whose expert is not 100% over its own randomisation produces a dataset
of failures, and a policy trained on that learns to give up.

Two lessons from the WYSD project are baked in:

* **The expert is a milestone controller, not a timed trajectory.**  Every
  waypoint is solved from the scene as it is now, so the same state always maps
  to the same action.  A timed script makes the task unobservable to the policy.
* **Release is a slow ramp, not a jump.**  Opening the jaw in one step flings a
  25 g cube off the disc (measured at 17 cm in the Uranus project).  The jaw is
  opened over ``steps_release`` control steps while the arm holds position.

One roll (``psi``) is chosen for the *whole* trajectory, grasp and place
together: the jaw is symmetric, so two rolls half a turn apart are equally valid
grasps, and letting the grasp and the placement pick different ones spins the
object out of the fingers in transit.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from spatialmath import SE3

from so101 import SO101Arm

from mj_env.env import HOME_QPOS
from mj_env.specs import LEVEL_ORDER, get
from mj_env.specs.criteria import RELEASE_MARGIN
from mj_env.task_env import TaskEnv

from .scripted_pick import (
    JAW_CLEARANCE,
    JAW_SQUEEZE,
    PAD_CLEARANCE,
    _MAX_GRASP_ROLL_CHANGE,
    _POSITION_TOLERANCE,
    GraspSolver,
    PickTuning,
    ViewerInterrupted,
    _glide,
    _wrap,
    grasp_axis_yaw,
    ik_seed,
    randomize_initial_arm,
    solve_top_down,
)

__all__ = ["PlaceTuning", "run_pick_place", "solve_pick_place"]


@dataclass(frozen=True)
class PlaceTuning:
    """The place step's own knobs (the carry is distance-driven, not timed)."""

    #: Open the jaw over this many control steps -- deliberately slow, see the
    #: module docstring.  40 steps = 2 s.
    steps_release: int = 40
    #: Hold still after retreating so the criterion's hold time can elapse.
    #:
    #: 40 steps = 2 s.  A thin bar dropped onto a disc can rock for well over a
    #: second, and "rested" is part of the criterion, so a short hold turns a
    #: good placement into a spurious failure (measured: 6 of 8 episodes were
    #: placed within 3 mm and still failed on this).
    steps_hold: int = 40
    #: Carry speed, as distance per control step.  8 mm at 20 Hz is 16 cm/s.
    #:
    #: Counter-intuitive but measured: a *slower* carry sheds the object more
    #: often, not less.  The cube is held by friction under a sustained squeeze,
    #: and the pad contact is soft, so the object creeps out over time; halving
    #: the speed doubled the loss rate.
    carry_speed_mm_per_step: float = 8.0
    #: Control steps per carry sub-target.  One is enough while the target only
    #: moves 8 mm; more just lengthens the exposure to creep.
    carry_steps_per_point: int = 1
    #: Jaw interference over the object's width during the carry, in metres.
    #:
    #: Deliberately *smaller* than the pick step's ``JAW_SQUEEZE``.  Measured on
    #: P1's 12-layout sweep, with the shipped pad friction (1.5):
    #:
    #:     2 mm -> 12/12, worst placement error 10 mm
    #:     4 mm -> 11/12, worst 39 mm
    #:     8 mm ->  7/12, worst 65 mm
    #:
    #: A hard squeeze is what *loses* the cube: the pad contact is soft, so
    #: pressing the pads deep into a rigid cube puts it in a state where it
    #: creeps out over the seconds a carry lasts.  Grip lightly and let friction
    #: do the work.  (Stiffening the contact instead -- ``solref`` 0.002, the
    #: fix that worked for the Uranus hand -- is much worse here: 0/12, with the
    #: cube flung ~190 mm because a stiff contact plus a position servo ejects
    #: it at the close step.)
    carry_squeeze: float = 0.002
    #: Per-side jaw clearance while descending, in metres.
    #:
    #: The waypoint solver centres the *aperture* for ``width + 2*PAD_CLEARANCE``,
    #: so descending with a wider opening leaves the aperture off-centre by the
    #: difference -- and makes the hand far wider than the object needs.
    #: Measured on the M0 outline probe: at the shipped ``JAW_CLEARANCE`` the hand
    #: is **130 mm** across while descending; at this clearance it is **83 mm**.
    #: A narrow slot is impossible until the jaws are closed before entering, and
    #: O4 measured exactly that as a 4.6 mm grind along the wall.
    approach_clearance: float = PAD_CLEARANCE


def solve_pick_place(
    solver: GraspSolver,
    arm: SO101Arm,
    grasp_aim: np.ndarray,
    hinge_yaw: float,
    place_aim: np.ndarray,
    offset: np.ndarray,
    tuning: PickTuning,
) -> dict[str, np.ndarray] | None:
    """Solve grasp *and* placement under one jaw roll, or return ``None``.

    The grasp pose is solved first because it is the most constrained one, and
    every other waypoint inherits its branch through the IK seed.  The place
    pose is solved too, even though the transport itself is Cartesian: if the
    placement pose has no IK solution the level is unreachable, and finding
    that out before the episode starts is the whole point of this function.
    """
    base = -np.pi / 2.0 - hinge_yaw
    for psi in sorted((_wrap(base), _wrap(base + np.pi)), key=abs):
        grasp = solver.solve_aperture(grasp_aim, psi, offset, ik_seed(grasp_aim, psi))
        if grasp is None:
            continue
        grasp_tcp = grasp_aim - solver.site_rotation(grasp) @ offset
        above = {
            "reach": grasp_tcp + (0.0, 0.0, tuning.approach_height),
            "hover": np.array([grasp_tcp[0], grasp_tcp[1], tuning.hover_height]),
            "lift": grasp_tcp + (0.0, 0.0, tuning.lift_height),
        }
        solutions: dict[str, np.ndarray] = {"grasp": grasp, "psi": np.array([psi])}
        for label in ("reach", "hover", "lift"):
            # Seed every waypoint from the constrained grasp pose.  Chaining
            # them (reach -> hover -> lift) lets the wrist land on a different
            # IK branch at each step, which is what made one level sweep the
            # joint 843 degrees -- two and a half turns -- while every
            # waypoint still solved.
            solution = solve_top_down(arm, above[label], psi, grasp)
            if solution is None or abs(solution[4] - grasp[4]) > _MAX_GRASP_ROLL_CHANGE:
                solution = solver.solve_continuous_height(
                    grasp_tcp, above[label], psi, grasp
                )
                if solution is None:
                    break
            solutions[label] = solution
        else:
            if not all(
                abs(solutions[label][4] - grasp[4]) <= _MAX_GRASP_ROLL_CHANGE
                for label in ("hover", "reach", "lift")
            ):
                continue
            place_descend = solver.solve_aperture(place_aim, psi, offset, solutions["lift"])
            if place_descend is None:
                continue
            solutions["place_descend"] = place_descend
            return solutions
    return None


def aperture_midpoint(env: TaskEnv, offset: np.ndarray) -> np.ndarray:
    """World position of the jaw aperture's centre for the current pose."""
    rotation = env.data.site_xmat[env._tcp_site_id].reshape(3, 3)
    return np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=float) + rotation @ offset


def _steps_between(start: float, end: float, mm_per_step: float) -> list[float]:
    """Values from ``start`` to ``end`` inclusive, spaced by ~``mm_per_step``."""
    span = abs(end - start) * 1000.0
    count = max(1, int(round(span / mm_per_step)))
    return list(np.linspace(start, end, count + 1)[1:])


def carry_path(
    start_mid: np.ndarray,
    end_mid: np.ndarray,
    hover_z: float,
    *,
    mm_per_step: float = 8.0,
) -> list[np.ndarray]:
    """Up, across and down again -- in cylindrical coordinates, not straight lines.

    A straight line between "above the object" and "above the goal" can pass
    inside ``r = 0.17``, right by the robot's own base, where a 5-DoF arm with
    this approach orientation has no IK solution.  Measured: the failures were
    all at ``r = 0.15``, on the centre line.  Interpolating radius and azimuth
    instead keeps the whole carry inside the band M0 verified as 100% solvable
    (``r in [0.185, 0.27]``), at the cost of a slightly longer arc.

    Both ends are *aperture* positions, so ``end_mid`` may sit well above the
    tabletop -- see how :func:`run_pick_place` compensates the held object's
    droop.
    """
    start_mid = np.asarray(start_mid, dtype=float)
    end_mid = np.asarray(end_mid, dtype=float)
    r_start = float(np.hypot(start_mid[0], start_mid[1]))
    t_start = float(np.arctan2(start_mid[1], start_mid[0]))
    r_goal = float(np.hypot(end_mid[0], end_mid[1]))
    t_goal = float(np.arctan2(end_mid[1], end_mid[0]))
    delta_theta = _wrap(t_goal - t_start)

    points = [start_mid.copy()]
    for z in _steps_between(float(start_mid[2]), hover_z, mm_per_step):
        points.append(np.array([r_start * np.cos(t_start), r_start * np.sin(t_start), z]))

    arc_length = abs(delta_theta) * 0.5 * (r_start + r_goal)
    count = max(2, int(np.ceil(arc_length * 1000.0 / mm_per_step)))
    for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
        radius = r_start + fraction * (r_goal - r_start)
        theta = t_start + fraction * delta_theta
        points.append(np.array([radius * np.cos(theta), radius * np.sin(theta), hover_z]))

    for z in _steps_between(hover_z, float(end_mid[2]), mm_per_step):
        points.append(np.array([r_goal * np.cos(t_goal), r_goal * np.sin(t_goal), z]))
    return points


def solve_exact_top_down(
    arm: SO101Arm, point: np.ndarray, psi: float, q0: np.ndarray | None
) -> np.ndarray | None:
    """Vertical-approach IK without the ``auto_soft_axis`` fallback."""
    pose = SE3(*np.asarray(point, dtype=float)) * SE3.Ry(np.pi) * SE3.Rz(psi)
    limits = arm.joint_limits
    epsilon = 1e-8
    seed = np.clip(
        np.asarray(
            q0 if q0 is not None else ik_seed(np.asarray(point, dtype=float), psi), dtype=float
        ),
        limits[:, 0] + epsilon,
        limits[:, 1] - epsilon,
    )
    result = arm.inverse_pose(pose, q0=seed, mode="soft_orientation", return_result=True)
    if result.success and float(result.position_error_norm) < _POSITION_TOLERANCE:
        return np.asarray(result.q, dtype=float)
    return None


#: Largest wrist-roll change one carry step may ask for.  The jaw is symmetric,
#: so every pose has two equally valid IK solutions, and ``solve_top_down``
#: falls back to ``auto_soft_axis`` -- which drops the orientation term -- when
#: ``soft_orientation`` cannot hold the pose.  Consecutive steps then land on
#: opposite branches.  A straight carry swings the site by 20-50 degrees on this
#: arm, so 90 leaves the path room while refusing the ~180 degree flip that
#: produces whole turns: one level swept the wrist 843 degrees -- two and a half
#: turns -- and repeatedly pinned it against the joint limit.
ROLL_STEP_LIMIT = math.radians(120.0)


def _mode_candidates(
    arm: SO101Arm, target: np.ndarray, psi: float, q0: np.ndarray
) -> list[np.ndarray]:
    """Every IK solution for the pose, not just the first mode that works."""
    pose = SE3(*target) * SE3.Ry(np.pi) * SE3.Rz(psi)
    limits = arm.joint_limits
    epsilon = 1e-8
    seed = np.clip(
        np.asarray(q0, dtype=float), limits[:, 0] + epsilon, limits[:, 1] - epsilon
    )
    candidates = []
    for mode in ("soft_orientation", "auto_soft_axis"):
        result = arm.inverse_pose(pose, q0=seed, mode=mode, return_result=True)
        if result.success and result.position_error_norm < _POSITION_TOLERANCE:
            candidates.append(np.asarray(result.q, dtype=float))
    return candidates


def solve_aperture_point(
    solver: GraspSolver,
    point: np.ndarray,
    psi: float,
    offset: np.ndarray,
    q0: np.ndarray | None,
    *,
    tol: float = 0.002,
    roll_ref: float | None = None,
    roll_limit: float = ROLL_STEP_LIMIT,
) -> np.ndarray | None:
    """Joints that put the *aperture centre* on ``point``, or ``None``.

    The one thing verified is the quantity that matters: the object sits at the
    aperture, not at the fingertip, so the aperture -- not the TCP -- is what
    has to land on the requested point.  An unnoticed centimetre of aperture
    offset *is* "the cube lands 5 cm off the disc".

    ``roll_ref`` turns on wrist continuity: the solve takes the IK branch whose
    wrist roll stays closest to that value and refuses anything more than
    ``roll_limit`` away, so a carry cannot flip branches between steps.  Without
    it the same call is the plain position-only solve, which is what the
    approach uses before there is a wrist pose worth staying near.
    """
    requested = np.asarray(point, dtype=float)
    target = requested.copy()
    solution = q0
    for _ in range(8):
        candidates = _mode_candidates(solver.arm, target, psi, solution)
        if not candidates:
            return None
        if roll_ref is None:
            candidate = candidates[0]
        else:
            # Prefer the branch that keeps the wrist near where the carry
            # started.  Anything past the limit is still taken if it is the
            # only option -- a failed carry loses the whole episode -- but the
            # preference is strong enough that the joint stops wandering.
            within = [
                c for c in candidates if abs(float(c[4]) - roll_ref) <= roll_limit
            ]
            if not within:
                return None
            candidate = min(within, key=lambda c: abs(float(c[4]) - roll_ref))
        solution = candidate
        achieved = solver.aperture_centre(solution, offset)
        # Move the TCP command by however far the aperture missed; adding the
        # residual to the desired point instead never converges.
        target = target + (requested - achieved)
        if float(np.linalg.norm(requested - achieved)) <= tol:
            return solution
    return None


def cartesian_transport(
    env: TaskEnv,
    solver: GraspSolver,
    waypoints: list[np.ndarray],
    offset: np.ndarray,
    psi: float,
    jaw: float,
    *,
    on_frame=None,
    q0: np.ndarray | None = None,
    speed_mm_per_step: float = 8.0,
    steps_per_point: int = 1,
) -> tuple[bool, np.ndarray | None]:
    """Carry the held object along a dense *aperture* path.

    Why not just ramp the joints, which is what the pick step does: a
    joint-space ramp between two distant poses sweeps an arc nobody asked for.
    On this arm that arc throws a 25 g cube out of the jaws during the carry --
    measured, and visible as ``grasped=False`` halfway to the goal.  Following
    the path in small steps keeps the object on it the whole way.
    (``WYSD/Lerobot-Uranus-VLA-Demo`` learned the same lesson with its
    ``cart_move``.)

    ``waypoints`` are aperture positions measured from the *object's* point of
    view, and :func:`solve_aperture_point` verifies the aperture after every
    solve -- checking the TCP instead would accept a wrist rotation that slides
    the object sideways by centimetres.  Returns ``(ok, last_joint_target)``.
    """
    points: list[np.ndarray] = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(round(length * 1000.0 / speed_mm_per_step)))
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            points.append(start + fraction * (end - start))

    seed = q0
    last: np.ndarray | None = None
    # The wrist reference is the pose the carry starts from and does not move:
    # updating it every step would let a slow drift walk the joint all the way
    # round while every individual step still looked continuous.
    roll_ref = None if seed is None else float(seed[4])
    for point in points:
        solution = solve_aperture_point(
            solver,
            point,
            psi,
            offset,
            seed,
            roll_ref=roll_ref,
        )
        if solution is None:
            return False, last
        seed = last = solution
        _glide(env, np.array([*solution, jaw]), max(1, steps_per_point), on_frame)
    return True, last


def _static_penetration(env: TaskEnv) -> float:
    """Deepest overlap between the robot and any static prop, in metres.

    A level's obstacle geometry is only verified if something checks that the
    hand actually fits where the level claims it does.  This reads the contact
    list for robot-vs-static pairs; the goal disc carries no collision, so the
    only thing it can report is a wall.
    """
    static_bodies = set(env._static_ids.values())
    worst = 0.0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        first = int(env.model.geom_bodyid[contact.geom1])
        second = int(env.model.geom_bodyid[contact.geom2])
        if (first in static_bodies) != (second in static_bodies):
            worst = min(worst, float(contact.dist))
    return worst


def run_pick_place(
    env: TaskEnv,
    solver: GraspSolver,
    *,
    seed: int | None,
    tuning: PickTuning | None = None,
    place_tuning: PlaceTuning | None = None,
    randomize_arm: bool = True,
    instruction: str | None = None,
    on_frame=None,
    target: str | None = None,
    place_xy: np.ndarray | None = None,
    reset: bool = True,
) -> dict:
    """Reset onto this level's layout and run the whole pick-and-place.

    Never raises on failure: an unreachable layout, a dropped object and a
    misplaced object all come back as a result dict, because a failed episode
    is data about the level, not an error in the harness.

    ``target``, ``place_xy`` and ``reset`` exist for **multi-stage** levels
    (O5: move the cover off, *then* fetch what was under it).  A second stage
    must not reset the scene -- that would put the cover back -- so it passes
    ``reset=False`` and names what it wants moved and where.  Everything else
    about the motion is deliberately unchanged, so a two-stage level still runs
    the same expert as every other level.
    """
    tuning = tuning or PickTuning()
    place_tuning = place_tuning or PlaceTuning()

    #: Deepest static prop the arm ever touched, tracked across every frame.
    penetration = {"min": 0.0}
    caller_on_frame = on_frame

    def _hook(action, observation, info) -> None:
        penetration["min"] = min(penetration["min"], _static_penetration(env))
        if caller_on_frame is not None:
            caller_on_frame(action, observation, info)

    on_frame = _hook

    if reset:
        options: dict = {}
        if target is None and not env.spec.instruction_targets:
            # Levels with a fixed target say so; language-referenced levels (P2)
            # let the instruction decide, which the environment resolves on reset.
            options["target_object"] = env.spec.target
        if instruction is not None:
            options["instruction"] = instruction
        _, info = env.reset(seed=seed, options=options)

    target = target or env.target_object
    initial_target_position = np.asarray(env.prop_pos(target), dtype=float).tolist()
    # A stage may aim somewhere that is not the level's goal -- O5 parks the
    # cover away from the disc so it is out of the carry path.
    goal_xy = (
        np.asarray(place_xy, dtype=float)[:2]
        if place_xy is not None
        else np.asarray(env.goal_xy(), dtype=float)
    )
    if randomize_arm:
        initial_q = randomize_initial_arm(env, seed)
        if env.render_mode == "human":
            env.render()
            if not env.viewer_running:
                raise ViewerInterrupted
    else:
        initial_q = np.asarray(HOME_QPOS, dtype=float)

    result: dict = {
        "task_id": env.spec.task_id,
        "seed": seed,
        "target": target,
        "instruction": env.instruction,
        "instruction_kind": env.instruction_kind,
        "initial_target_position": initial_target_position,
        "goal_xy": goal_xy.tolist(),
        "initial_qpos": initial_q.tolist(),
    }

    width = env._objects[target].grip_width
    offset = solver.calibration.offset_for(width + 2.0 * PAD_CLEARANCE)
    grasp_aim = np.asarray(env.prop_pos(target), dtype=float)
    place_aim = np.array([goal_xy[0], goal_xy[1], env.place_height(target)])

    try:
        solutions = solve_pick_place(
            solver, arm_for(env), grasp_aim, grasp_axis_yaw(env, target), place_aim, offset, tuning
        )
    except (ValueError, FloatingPointError):
        solutions = None
    if solutions is None:
        return {**result, "outcome": "ik_fail", "success": False}

    jaw_open = solver.calibration.angle_for(width + JAW_CLEARANCE)
    # Close the jaws before descending: the aperture the IK aimed at is the one
    # at `approach_clearance`, so matching it keeps the object centred between
    # the pads *and* keeps the hand narrow enough to enter a slot.
    jaw_approach = solver.calibration.angle_for(
        width + 2.0 * place_tuning.approach_clearance
    )
    # One squeeze for the whole pick-and-carry, sized for the carry: the object
    # has to stay gripped for seconds, not for the milliseconds a bare lift
    # takes, and the soft pad contact lets it creep out.
    jaw_shut = solver.calibration.angle_for(
        max(width - env.grip_squeeze(target, place_tuning.carry_squeeze), 0.0)
    )
    psi = float(solutions["psi"][0])

    _glide(env, np.array([*solutions["hover"], jaw_open]), tuning.steps_hover, on_frame)
    _glide(env, np.array([*solutions["reach"], jaw_approach]), tuning.steps_reach, on_frame)
    _glide(env, np.array([*solutions["grasp"], jaw_approach]), tuning.steps_descend, on_frame)
    _glide(env, np.array([*solutions["grasp"], jaw_shut]), tuning.steps_close, on_frame)
    _glide(env, np.array([*solutions["lift"], jaw_shut]), tuning.steps_lift, on_frame)

    # Carry along the cylindrical path: up, around at a safe radius, then down.
    #
    # Closed-loop place target.  The held cube does not sit at the aperture
    # centre: the pad contact is soft, so it hangs **12-21 mm below** and a few
    # millimetres to one side of it (measured).  Commanding the aperture down to
    # the goal's resting height therefore drives the *object* into the tabletop
    # about 17 mm early, and the pads keep descending and squeeze it out
    # sideways -- which is exactly the bimodal placement error that was showing
    # up (a few mm, or 30-80 mm).  Aim the aperture at the goal minus wherever
    # the object actually is relative to it.
    held_offset = env.prop_pos(target) - aperture_midpoint(env, offset)
    place_aim = place_aim - held_offset
    start_mid = aperture_midpoint(env, offset)
    ok, carried = cartesian_transport(
        env,
        solver,
        carry_path(
            start_mid,
            place_aim,
            tuning.hover_height,
            mm_per_step=place_tuning.carry_speed_mm_per_step,
        ),
        offset,
        psi,
        jaw_shut,
        on_frame=on_frame,
        q0=solutions["lift"],
        speed_mm_per_step=place_tuning.carry_speed_mm_per_step,
        steps_per_point=place_tuning.carry_steps_per_point,
    )
    if not ok or carried is None:
        return {**result, "outcome": "transport_fail", "success": False}

    # Slow release while holding the place pose.
    _glide(env, np.array([*carried, jaw_open]), place_tuning.steps_release, on_frame)
    # Retreat straight up, clear of the placed object.
    ok, _ = cartesian_transport(
        env,
        solver,
        [place_aim, np.array([place_aim[0], place_aim[1], tuning.hover_height])],
        offset,
        psi,
        jaw_open,
        on_frame=on_frame,
        q0=carried,
        speed_mm_per_step=place_tuning.carry_speed_mm_per_step,
        steps_per_point=place_tuning.carry_steps_per_point,
    )
    if not ok:
        return {**result, "outcome": "retreat_fail", "success": False}

    info = env._info()
    for _ in range(place_tuning.steps_hold):
        action = np.asarray(env.data.ctrl[env._actuator_ids], dtype=np.float32)
        # Record the settling phase like every other phase (`_glide`'s
        # convention: hand the pre-action observation to the callback, then
        # step).  Two reasons.  The criterion is evaluated *here*, so an episode
        # that stops recording before it never shows a policy the moment success
        # is decided.  And a recorder cannot tolerate a gap in simulation time:
        # leaving this loop unrecorded makes the next recorded frame jump by the
        # whole hold, which a two-stage level (O5) hits mid-episode and
        # single-stage levels only ever hit after their last frame.
        if on_frame is not None:
            on_frame(action.copy(), env._observation(), info)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    placed = bool(info.get("object_on_goal"))
    lifted = float(info.get("lift_height", 0.0))
    if info["success"]:
        outcome = "success"
    elif placed:
        outcome = "not_settled"
    else:
        outcome = "misplaced"

    return {
        **result,
        "outcome": outcome,
        "success": bool(info["success"]),
        "grasped": bool(info.get("grasped")),
        "lift_height": lifted,
        "object_on_goal": placed,
        "goal_error_mm": float(np.linalg.norm(env.prop_pos(target)[:2] - goal_xy) * 1000.0),
        "height_above_rest_mm": float(info.get("height_above_rest", 0.0) * 1000.0),
        "grasp_opening_mm": float(info.get("grasp_opening", 0.0) * 1000.0),
        "spec_hash": env.spec_hash(),
        "static_penetration_mm": penetration["min"] * 1000.0,
    }


def arm_for(env: TaskEnv) -> SO101Arm:
    """One shared :class:`SO101Arm` per environment; IK is stateless."""
    arm = getattr(env, "_expert_arm", None)
    if arm is None:
        arm = SO101Arm()
        env._expert_arm = arm
    return arm


def expert_for(env: TaskEnv):
    """The expert a level needs, chosen from the spec rather than a task list.

    Adding a multi-stage level therefore needs no edit here: the level declares
    ``multi_stage=True`` and the harness picks the two-stage expert.
    """
    return run_cover_then_place if env.spec.multi_stage else run_pick_place


def run_cover_then_place(
    env: TaskEnv,
    solver: GraspSolver,
    *,
    seed: int | None,
    cover: str = "cover",
    on_frame=None,
    **kwargs,
) -> dict:
    """Two stages: move the cover off the target, then do the ordinary task.

    This is the only level that needs more than one pick-and-place, and the
    second stage deliberately reuses :func:`run_pick_place` unchanged: if the
    second stage needed its own motion, the level would be measuring two things
    at once and the success rate could not be attributed to sequencing.

    The cover is parked at the level's ``park_spot``, a collision-free marker
    body the layout places away from both the target and the goal -- park it on
    the carry path and the second stage collides with its own debris.
    """
    # Resolve the parking spot from this episode's sampled layout, not from
    # the XML/default pose or the previous episode's layout.
    options = {}
    if kwargs.get("instruction") is not None:
        options["instruction"] = kwargs["instruction"]
    env.reset(seed=seed, options=options)
    park_xy = env.static_xy("park_spot")
    first = run_pick_place(
        env,
        solver,
        seed=seed,
        target=cover,
        place_xy=park_xy,
        reset=False,
        on_frame=on_frame,
        **kwargs,
    )
    # Judge the first stage on *its own* geometry, not on the level's criterion:
    # the level's criterion also demands the target be on the disc, which the
    # cover stage cannot satisfy by construction.  Using it here failed every
    # episode even when the cover landed 4.6 mm from its parking spot.
    cover_ok = (
        "goal_error_mm" in first
        and float(first["goal_error_mm"]) <= 30.0
        and abs(float(first["height_above_rest_mm"])) <= 3.0
        and float(first["grasp_opening_mm"])
        >= (env.grip_width(cover) + RELEASE_MARGIN) * 1000.0
    )
    if not cover_ok:
        # A failed first stage is not a failed episode of the real task: report
        # it as its own outcome so the two are never confused in a data table.
        return {**first, "outcome": "cover_failed", "stage": "cover"}
    # The second stage continues from where the first one left the arm, so it
    # must not re-randomise; drop the caller's flag rather than pass both.
    kwargs.pop("randomize_arm", None)
    second = run_pick_place(
        env,
        solver,
        seed=seed,
        reset=False,
        randomize_arm=False,
        on_frame=on_frame,
        **kwargs,
    )
    return {
        **second,
        "stage": "target",
        "cover_error_mm": float(
            np.linalg.norm(np.asarray(env.prop_pos(cover), dtype=float)[:2] - park_xy) * 1000.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=LEVEL_ORDER, required=True)
    parser.add_argument("--episodes", type=int, default=20, help="layouts to attempt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require",
        type=float,
        default=1.0,
        help="exit non-zero below this success rate (default: the expert must be perfect)",
    )
    parser.add_argument("--instruction", help="pin one instruction instead of sampling a pool")
    parser.add_argument(
        "--instruction-pool", default="train", choices=("train", "holdout", "all")
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-arm",
        action="store_true",
        help="keep the arm at HOME_QPOS instead of randomizing its initial pose",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    spec = get(args.task)
    print(f"task {spec.task_id} ({spec.title}) scene={spec.scene}")
    print(f"  difficulty : {spec.difficulty_axis}")
    print(f"  criterion  : {spec.criterion_text}  hold={spec.hold_s}s")

    env = TaskEnv(
        args.task,
        render_mode=None if args.headless else "human",
        instruction=args.instruction,
        instruction_pool=args.instruction_pool,
        max_episode_steps=10_000,
    )
    solver = GraspSolver(env.model)
    print(
        f"  jaw        : {solver.calibration.min_opening * 1000:.1f}"
        f"-{solver.calibration.max_opening * 1000:.1f} mm"
    )

    outcomes: list[str] = []
    successes = 0
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            try:
                result = expert_for(env)(
                    env,
                    solver,
                    seed=seed,
                    randomize_arm=not args.fixed_arm,
                )
            except ViewerInterrupted:
                print("\nviewer closed; run interrupted")
                return 0
            outcomes.append(result["outcome"])
            successes += int(result["success"])
            print(
                f"  seed={seed:<6} {result['outcome']:<10}"
                f" goal_err={result.get('goal_error_mm', float('nan')):>6.1f}mm"
                f" lift={result.get('lift_height', 0.0) * 1000:>6.1f}mm"
                f" jaw={result.get('grasp_opening_mm', 0.0):>5.1f}mm"
                f" pen={result.get('static_penetration_mm', 0.0):>+6.1f}mm"
                f" target={result.get('target', '?')}"
                f"  {result.get('instruction', '')!r}"
            )
    finally:
        env.close()

    rate = successes / args.episodes
    print(f"\n{spec.task_id}: expert {successes}/{args.episodes} = {rate * 100:.1f}%")
    from collections import Counter

    print("  outcomes:", dict(Counter(outcomes)))
    if rate + 1e-9 < args.require:
        print(f"  FAIL: below the required {args.require * 100:.0f}%")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

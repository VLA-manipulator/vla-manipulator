"""Compile the SO101 tabletop MuJoCo model.

``assets/scene.xml`` is a plain, viewer-openable MJCF that includes the
calibrated robot from the ``so101`` package.  Three things this environment
needs cannot be expressed there, because they belong to bodies owned by the
included robot file and MJCF has no way to amend an included body:

* the wrist camera, which must hang off the ``gripper`` body;
* flat gripping pads on the two jaws, measured from the meshes, because the
  curved prongs collide only along a line at their tips and let an object sink
  into the jaw;
* high friction and stiff contacts on those pads, so a friction-driven grasp
  does not slip and does not let the object sink either.

All three are applied here with ``mujoco.MjSpec`` after parsing, which keeps
``so101/assets/so101_new_calib.xml`` a pure hardware description.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS_DIR / "scene.xml"
#: Scene-composed task levels.  Each one includes ``../base_table.xml`` (the
#: shared table, lights and cameras) plus the robot, and declares only the
#: props that level adds.  Kept out of ``assets/`` so the legacy single-file
#: scene stays exactly as it was.
SCENES_DIR = ASSETS_DIR / "scenes"

# --- Frame convention -------------------------------------------------------
# Robot base at the origin, +x forward across the table, +z up, tabletop at z=0.
TABLE_HEIGHT = 0.0

# Object placement band, in polar coordinates about the base.
#
# The SO101 has five joints, so it cannot hold an arbitrary six-DoF pose: a
# vertical approach plus an arbitrary jaw roll is seven constraints on five
# axes.  What saves the top-down grasp is that the jaw is symmetric, so a roll
# and that roll plus half a turn are the same physical grasp; folding the
# required roll into [-pi/2, pi/2] keeps wrist_roll inside its -2.74..2.84
# limit.  With that reduction, a swept IK check (5 azimuths x 8 jaw rolls x
# 5 heights per radius) solves an exact vertical grasp for 100% of poses out
# to r = 0.26, 90-100% at r = 0.28, 25-90% at r = 0.30 and almost none at
# r = 0.32.  Objects are therefore kept inside r = 0.28.
GRASP_ZONE_RADIUS = (0.17, 0.28)
GRASP_ZONE_HALF_ANGLE = math.radians(60.0)

TCP_SITE = "gripperframe"
SCENE_CAMERA = "global_camera"
WRIST_CAMERA = "wrist_camera"
# Backward-compatible alias: the former overhead view is now the one global
# tabletop camera.  Keeping the alias avoids breaking older annotation tools.
OVERHEAD_CAMERA = SCENE_CAMERA

FIXED_FINGER_GEOM = "fixed_finger"
MOVING_FINGER_GEOM = "moving_finger"
#: The flat plates that actually grip.  The finger meshes above keep their
#: names -- callers still use them to find the jaw -- but they no longer collide.
FIXED_PAD_GEOM = "fixed_pad"
MOVING_PAD_GEOM = "moving_pad"

# Global tabletop camera, expressed in the world frame.  This uses the former
# overhead camera placement, but is exposed under the single name
# ``global_camera`` for policy input, debugging, and GUI overlays.
# Focus on the reachable object band rather than the far half of the table.
GLOBAL_CAMERA_POS = (0.17, 0.0, 0.66)
GLOBAL_CAMERA_TARGET = (0.17, 0.0, TABLE_HEIGHT)
GLOBAL_CAMERA_UP = (1.0, 0.0, 0.0)
GLOBAL_CAMERA_FOVY = 46.0

# Wrist camera, expressed in the gripper body frame.  The moving-jaw hinge
# anchor is (0.0202, 0.0188, -0.0234) and its axis is local -y.  Place the
# camera on that axis, 12 cm to the -y side, then look forward into the local
# x-z opening plane.  Image-up follows local -z: closed fingers meet in the
# lower centre, while opening toward local +x moves the active finger left and
# down in the image.
# WRIST_CAMERA_POS = (0.0202, -0.1012, -0.0234)
WRIST_CAMERA_POS = (-0.0, -0.06, -0.015)
# WRIST_CAMERA_TARGET = (0.0202, 0.0188, -0.0734)
WRIST_CAMERA_TARGET = (-0.0, 0.0, -0.15)
WRIST_CAMERA_UP = (0.0, 0.0, -1.0)
WRIST_CAMERA_FOVY = 80.0

# Finger pad contact parameters.  ``priority`` makes these win outright over
# the object's settings rather than being combined with them.  The solver
# parameters are deliberately left alone: stiffening the pads was tried and
# made grasping worse, not better -- a hard pad bounces a cube out of the jaw
# faster than a soft one lets it sink in.
_FINGER_FRICTION = (1.5, 0.05, 0.005)
_FINGER_CONDIM = 4
_FINGER_PRIORITY = 1

# Flat gripping pads.  The MJCF fingers are curved prongs, and their convex
# hull -- which is what MuJoCo collides -- is far wider than the visible mesh.
# The hulls meet on a line at the tips, so a box held between them rests on that
# line and the soft contact lets it sink 5-10 mm into the jaw, which renders as
# the finger cutting through the object.  Two explicit plates give the grasp a
# flat, measured surface: they carry every finger contact, and
# :mod:`mj_env.gripper` calibrates the opening from them, so "how wide is the
# jaw" means the same thing to the controller, the collision engine and the
# renderer.
PAD_THICKNESS = 0.0015
PAD_DEPTH = 0.018
PAD_WIDTH = 0.016
#: Contact stiffness of the plates.  Softer than this and the object sinks in;
#: stiffer and the plate kicks the object and the arm instead of holding them.
PAD_SOLREF = 0.016
#: Jaw angle the plates are designed at: the middle of the widths the catalogue
#: actually uses (16-40 mm), which keeps them near-parallel across that range.
_PAD_DESIGN_ANGLE = 0.17
#: Vertices closer to the hinge axis than this are structure, not gripping
#: surface; including them would measure the hinge gap instead of the jaw.
PAD_MIN_HINGE_RADIUS = 0.05


def look_at_quat(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float]
) -> np.ndarray:
    """Camera orientation quaternion (wxyz) for a look-at, in the parent frame.

    MuJoCo cameras look down their own -Z with +Y up and +X right, so the
    rotation columns are (right, up, -forward).
    """
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    z_axis = -forward / np.linalg.norm(forward)
    right = np.cross(np.asarray(up, dtype=float), z_axis)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError("look-at 'up' vector is parallel to the view direction")
    x_axis = right / norm
    y_axis = np.cross(z_axis, x_axis)

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.stack([x_axis, y_axis, z_axis], axis=1).reshape(9))
    return quat


def _collision_geom(spec: mujoco.MjSpec, body: str, mesh: str) -> mujoco.MjsGeom:
    for geom in spec.body(body).geoms:
        if geom.classname.name == "collision" and geom.meshname == mesh:
            return geom
    raise ValueError(f"body {body!r} has no collision geom for mesh {mesh!r}")


def finger_vertices(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_name: str,
    joint_id: int | None = None,
) -> np.ndarray:
    """World-frame mesh vertices of a finger, hinge structure excluded.

    With ``joint_id`` the vertices within :data:`PAD_MIN_HINGE_RADIUS` of that
    joint's axis are dropped, which is what turns "closest vertices of the two
    fingers" into "jaw opening" instead of "gap between the motor shells".
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"scene has no geom named {geom_name!r}")
    mesh_id = model.geom_dataid[geom_id]
    start, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    local = model.mesh_vert[start : start + num].astype(float)
    world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    if joint_id is None:
        return world
    anchor, axis = data.xanchor[joint_id], data.xaxis[joint_id]
    radius = np.linalg.norm(np.cross(world - anchor, axis), axis=1)
    return world[radius > PAD_MIN_HINGE_RADIUS]


def _pad_poses(model: mujoco.MjModel) -> tuple[tuple, tuple]:
    """Measure where the two jaws meet and turn that into two pad boxes.

    Returns ``((body, geom, pos, quat), (...))`` with poses expressed in each
    finger's own body frame, so the plates follow the jaw exactly.  Each plate's
    local +z points into the gap and its face sits on the measured contact
    point, which is what makes the calibrated opening the physical gap.
    """
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    qpos_adr = int(model.jnt_qposadr[joint_id])

    data.qpos[:] = model.qpos0
    data.qpos[qpos_adr] = _PAD_DESIGN_ANGLE
    mujoco.mj_forward(model, data)

    fixed = finger_vertices(model, data, FIXED_FINGER_GEOM, joint_id)
    moving = finger_vertices(model, data, MOVING_FINGER_GEOM, joint_id)
    distances, indices = cKDTree(fixed).query(moving)
    closest = int(np.argmin(distances))
    p_fixed = fixed[indices[closest]]
    p_moving = moving[closest]

    normal = p_moving - p_fixed
    normal /= np.linalg.norm(normal)
    # Keep the plates square to the closing direction: x runs along the
    # approach so the plates are as deep as the fingertip they replace.
    site_x = data.site_xmat[site_id].reshape(3, 3)[:, 0]
    approach = site_x - site_x.dot(normal) * normal
    approach /= np.linalg.norm(approach)

    poses = []
    for body_name, geom_name, point, inward in (
        ("gripper", FIXED_PAD_GEOM, p_fixed, normal),
        ("moving_jaw_so101_v1", MOVING_PAD_GEOM, p_moving, -normal),
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        # The third column is the plate's normal, so the lateral axis must come
        # from it: the two plates face each other, and taking a shared lateral
        # axis instead would make the moving plate a *reflection* of the fixed
        # one.  mju_mat2Quat then returns a rotation about 90 degrees away, and
        # the jaw grips on the plate's edge instead of its face.
        lateral = np.cross(inward, approach)
        rotation = np.stack([approach, lateral, inward], axis=1)
        centre = point - inward * (PAD_THICKNESS / 2.0)
        body_rotation = data.xmat[body_id].reshape(3, 3)
        local_centre = body_rotation.T @ (centre - data.xpos[body_id])
        local_rotation = body_rotation.T @ rotation
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, local_rotation.reshape(9))
        poses.append((body_name, geom_name, local_centre, quat))
    return tuple(poses)


def _add_gripper_pads(spec: mujoco.MjSpec) -> None:
    """Attach the measured plates to the two jaw bodies."""
    for body_name, geom_name, pos, quat in _pad_poses(spec.compile()):
        geom = spec.body(body_name).add_geom()
        geom.name = geom_name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [PAD_DEPTH / 2.0, PAD_WIDTH / 2.0, PAD_THICKNESS / 2.0]
        geom.pos = np.asarray(pos).tolist()
        geom.quat = np.asarray(quat).tolist()
        geom.group = 3
        geom.contype = 1
        geom.conaffinity = 1
        geom.friction = list(_FINGER_FRICTION)
        geom.condim = _FINGER_CONDIM
        geom.priority = _FINGER_PRIORITY
        # Plate-on-object contact: a flat pad has no line-contact give, so it
        # needs to be stiffer than MuJoCo's default to stop the object sinking
        # into it.  Too stiff and the plate stops being a pad at all: at
        # solref 0.008 blocks the shoulder before the moving pad reaches the
        # blue cube in the legacy scene.  0.012 fixes that layout but still
        # misses a composed-scene grasp (O1, seed 1).  0.016 lets the servo
        # reach both sides without giving up the flat contact surface.  Keep
        # this calibrated against measured contact penetration.
        geom.solref = [PAD_SOLREF, 1.0]
        geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def build_spec(scene_xml: str | Path | None = None) -> mujoco.MjSpec:
    """Parse a scene and apply the amendments the MJCF cannot express.

    ``scene_xml`` defaults to the legacy eight-object tabletop.  Passing a file
    from :data:`SCENES_DIR` composes the shared table with that level's props.
    """
    source = SCENE_XML if scene_xml is None else Path(scene_xml)
    spec = mujoco.MjSpec.from_file(str(source))

    camera = spec.body("gripper").add_camera(name=WRIST_CAMERA)
    camera.pos = list(WRIST_CAMERA_POS)
    camera.quat = look_at_quat(
        WRIST_CAMERA_POS, WRIST_CAMERA_TARGET, WRIST_CAMERA_UP
    ).tolist()
    camera.fovy = WRIST_CAMERA_FOVY

    fingers = (
        (_collision_geom(spec, "gripper", "wrist_roll_follower_so101_v1"), FIXED_FINGER_GEOM),
        (_collision_geom(spec, "moving_jaw_so101_v1", "moving_jaw_so101_v1"), MOVING_FINGER_GEOM),
    )
    for geom, name in fingers:
        geom.name = name
        # The flat pads below carry every finger contact.  Colliding the curved
        # prong itself grips on a line and sinks the object into the jaw.
        geom.contype = 0
        geom.conaffinity = 0

    _add_gripper_pads(spec)

    return spec


def build_model(scene_xml: str | Path | None = None) -> mujoco.MjModel:
    """Compile a tabletop scene into an ``MjModel``."""
    model = build_spec(scene_xml).compile()
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, SCENE_CAMERA)
    model.cam_pos[camera_id] = GLOBAL_CAMERA_POS
    model.cam_quat[camera_id] = look_at_quat(
        GLOBAL_CAMERA_POS, GLOBAL_CAMERA_TARGET, GLOBAL_CAMERA_UP
    )
    model.cam_fovy[camera_id] = GLOBAL_CAMERA_FOVY
    return model

"""Compile the SO101 tabletop MuJoCo model.

``assets/scene.xml`` is a plain, viewer-openable MJCF that includes the
calibrated robot from the ``so101`` package.  Two things this environment needs
cannot be expressed there, because they belong to bodies owned by the included
robot file and MJCF has no way to amend an included body:

* the wrist camera, which must hang off the ``gripper`` body;
* higher friction on the two finger collision geoms, so a friction-driven
  grasp does not slip.

Both are applied here with ``mujoco.MjSpec`` after parsing, which keeps
``so101/assets/so101_new_calib.xml`` a pure hardware description.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS_DIR / "scene.xml"

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


def build_spec() -> mujoco.MjSpec:
    """Parse the scene and apply the amendments the MJCF cannot express."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))

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
        geom.friction = list(_FINGER_FRICTION)
        geom.condim = _FINGER_CONDIM
        geom.priority = _FINGER_PRIORITY

    return spec


def build_model() -> mujoco.MjModel:
    """Compile the tabletop scene into an ``MjModel``."""
    model = build_spec().compile()
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, SCENE_CAMERA)
    model.cam_pos[camera_id] = GLOBAL_CAMERA_POS
    model.cam_quat[camera_id] = look_at_quat(
        GLOBAL_CAMERA_POS, GLOBAL_CAMERA_TARGET, GLOBAL_CAMERA_UP
    )
    model.cam_fovy[camera_id] = GLOBAL_CAMERA_FOVY
    return model

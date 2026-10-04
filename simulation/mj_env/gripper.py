"""Where the SO101 jaw actually grips, measured from the model.

The MJCF's ``gripperframe`` site sits at the *fingertip*, but the pads that
hold an object meet well behind and to one side of it: the fixed finger's
inner face lies roughly on the site axis while the moving jaw swings away from
it, so the aperture centre is offset by 1-4 cm depending on how wide the jaw
is open.  Aiming inverse kinematics at the site therefore drives the fingertips
past the object and the jaw closes on nothing.

This module sweeps the jaw joint once against the finger meshes and records,
per jaw angle, the pad separation and the aperture centre expressed in the TCP
site frame.  Callers ask "how do I grip something ``w`` wide?" and get back the
jaw angle and the offset to add to their IK target.  Nothing is hard-coded, so
the numbers follow the meshes if the model is ever changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .scene import FIXED_FINGER_GEOM, MOVING_FINGER_GEOM, TCP_SITE

#: Vertices closer to the hinge than this are structure, not pad; including
#: them would report the hinge gap instead of the jaw opening.
_PAD_MIN_HINGE_RADIUS = 0.05
#: Number of closest pad vertex pairs averaged into the aperture centre.
_PAD_SAMPLE = 150


@dataclass(frozen=True)
class JawCalibration:
    """Jaw angle -> pad separation and aperture centre, sampled from the model."""

    angles: np.ndarray
    openings: np.ndarray
    #: ``(n, 3)`` aperture centres in the TCP site frame; +x is the approach
    #: direction out through the fingers.
    offsets: np.ndarray

    @property
    def max_opening(self) -> float:
        return float(self.openings.max())

    @property
    def min_opening(self) -> float:
        return float(self.openings.min())

    def angle_for(self, width: float) -> float:
        """Jaw angle whose opening matches ``width`` metres."""
        return float(np.interp(width, self.openings, self.angles))

    def offset_for(self, width: float) -> np.ndarray:
        """TCP-site-frame vector from the fingertip to the aperture centre.

        Add ``site_rotation @ offset_for(w)`` to an object's position to get
        the TCP position that puts the object between the pads.
        """
        return np.array(
            [np.interp(width, self.openings, self.offsets[:, axis]) for axis in range(3)]
        )


def calibrate(model: mujoco.MjModel, *, samples: int = 25) -> JawCalibration:
    """Sweep the jaw and measure its pad geometry against the finger meshes."""
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    qpos_adr = int(model.jnt_qposadr[joint_id])
    low, high = model.jnt_range[joint_id]

    angles, openings, offsets = [], [], []
    for angle in np.linspace(low, high, samples):
        data.qpos[:] = model.qpos0
        data.qpos[qpos_adr] = angle
        mujoco.mj_forward(model, data)

        fixed = _pad_vertices(model, data, FIXED_FINGER_GEOM, joint_id)
        moving = _pad_vertices(model, data, MOVING_FINGER_GEOM, joint_id)
        distances, indices = cKDTree(fixed).query(moving)
        closest = np.argsort(distances)[:_PAD_SAMPLE]
        centre = ((moving[closest] + fixed[indices[closest]]) / 2.0).mean(axis=0)

        origin = data.site_xpos[site_id]
        rotation = data.site_xmat[site_id].reshape(3, 3)
        angles.append(float(angle))
        openings.append(float(distances.min()))
        offsets.append(rotation.T @ (centre - origin))

    # np.interp needs an increasing x; opening grows monotonically with angle.
    return JawCalibration(
        angles=np.asarray(angles),
        openings=np.asarray(openings),
        offsets=np.asarray(offsets),
    )


def _pad_vertices(
    model: mujoco.MjModel, data: mujoco.MjData, geom_name: str, hinge_joint: int
) -> np.ndarray:
    """World-frame mesh vertices of a finger, restricted to its pad region."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    local = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]].astype(float)
    world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]

    anchor, axis = data.xanchor[hinge_joint], data.xaxis[hinge_joint]
    hinge_radius = np.linalg.norm(np.cross(world - anchor, axis), axis=1)
    return world[hinge_radius > _PAD_MIN_HINGE_RADIUS]


__all__ = ["JawCalibration", "calibrate"]

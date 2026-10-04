"""Tabletop object catalogue.

Only what cannot be derived from the compiled model lives here: the body name,
the natural-language prompt handed to the policy, and a note on why the object
is interesting.  Everything dimensional -- footprint radius, drop height -- is
measured from ``MjModel`` by :func:`measure`, so the catalogue can never drift
out of sync with ``assets/scene.xml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class TableObject:
    """A graspable object present in ``assets/scene.xml``."""

    name: str
    shape: str
    prompt: str
    grasp_note: str
    #: Width the jaw has to close to, in metres.  ``None`` means "the object's
    #: narrow horizontal dimension", which is right for a rod, cube or ball.
    #: A dumbbell needs it stated: its widest part is the discs, but the only
    #: thing the jaw can actually grip is the much thinner handle.
    grip_width: float | None = None

    @property
    def freejoint(self) -> str:
        return f"{self.name}_free"


CATALOGUE: tuple[TableObject, ...] = (
    TableObject(
        "rod_red",
        "rod",
        "pick up the red rod",
        "168 mm shaft; graspable anywhere along it, but the jaw must close across the shaft",
    ),
    TableObject(
        "rod_yellow",
        "rod",
        "pick up the yellow rod",
        "168 mm shaft; graspable anywhere along it, but the jaw must close across the shaft",
    ),
    TableObject(
        "sphere_blue",
        "sphere",
        "pick up the blue ball",
        "40 mm ball; no flat faces, needs a wide approach and a firm close",
    ),
    TableObject(
        "sphere_green",
        "sphere",
        "pick up the green ball",
        "40 mm ball; no flat faces, needs a wide approach and a firm close",
    ),
    TableObject(
        "cube_red",
        "cube",
        "pick up the red cube",
        "35 mm cube; the easy parallel-face baseline",
    ),
    TableObject(
        "cube_blue",
        "cube",
        "pick up the blue cube",
        "35 mm cube; the easy parallel-face baseline",
    ),
    TableObject(
        "dumbbell_purple",
        "dumbbell",
        "pick up the purple dumbbell",
        "44 mm end discs are wider than the jaw; only the 16 mm centre handle can be grasped",
        grip_width=0.016,
    ),
    TableObject(
        "dumbbell_orange",
        "dumbbell",
        "pick up the orange dumbbell",
        "44 mm end discs are wider than the jaw; only the 16 mm centre handle can be grasped",
        grip_width=0.016,
    ),
)

BY_NAME: dict[str, TableObject] = {obj.name: obj for obj in CATALOGUE}
NAMES: tuple[str, ...] = tuple(obj.name for obj in CATALOGUE)


@dataclass(frozen=True)
class Footprint:
    """A body's horizontal extent as a 2-D capsule in its own frame.

    A single enclosing circle is far too pessimistic for a 168 mm rod -- it
    would claim a 170 mm exclusion disc for a 18 mm-thick stick and make any
    reasonable table layout unpackable.  A capsule along the body's long axis
    is both tight and cheap to test pairwise.
    """

    #: Half-length of the capsule's spine, along the body's local +x.
    half_length: float
    #: Capsule radius, which also covers the corners of the horizontal AABB.
    radius: float
    #: Distance from the body origin down to the lowest point, i.e. the height
    #: at which the body origin rests on a flat table.
    rest_clearance: float

    @property
    def max_radius(self) -> float:
        """Largest horizontal distance from the body origin to any point."""
        return self.half_length + self.radius

    @property
    def narrow_width(self) -> float:
        """Full extent across the short horizontal axis, undoing the corner term."""
        return self.radius * np.sqrt(2.0)


@dataclass(frozen=True)
class ObjectMetrics:
    """A catalogue object resolved and measured against a compiled model."""

    body_id: int
    qpos_adr: int
    footprint: Footprint
    #: Width the jaw must close to in order to hold this object, in metres.
    grip_width: float


def _body_geom_corners(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return the geom's local bounding-box corners in the parent body frame."""
    centre = model.geom_aabb[geom_id, :3]
    half = model.geom_aabb[geom_id, 3:]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    corners = centre + signs * half

    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    return corners @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]


def measure_footprint(model: mujoco.MjModel, body_id: int) -> Footprint:
    """Measure a body's horizontal capsule footprint and resting height.

    Objects are only ever yaw-rotated on reset, so a body-frame measurement is
    exact for every pose the environment produces.  For a horizontal AABB of
    half-extents ``a >= b`` the tightest enclosing capsule along the long axis
    has spine half-length ``a - b`` and radius ``b * sqrt(2)``, which is what
    reaches the AABB corners.
    """
    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    if not geom_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        raise ValueError(f"body {name!r} has no geoms")
    corners = np.vstack([_body_geom_corners(model, g) for g in geom_ids])

    x_half, y_half = np.abs(corners[:, :2]).max(axis=0)
    long_half, short_half = max(x_half, y_half), min(x_half, y_half)
    return Footprint(
        half_length=float(long_half - short_half),
        radius=float(short_half * np.sqrt(2.0)),
        rest_clearance=float(-corners[:, 2].min()),
    )


def measure(model: mujoco.MjModel, name: str) -> ObjectMetrics:
    """Resolve and measure one catalogue object against a compiled model."""
    obj = BY_NAME[name]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.name)
    if body_id < 0:
        raise ValueError(f"scene is missing body {obj.name!r}")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj.freejoint)
    if joint_id < 0:
        raise ValueError(f"scene is missing free joint {obj.freejoint!r}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"joint {obj.freejoint!r} is not a free joint")

    footprint = measure_footprint(model, body_id)
    return ObjectMetrics(
        body_id=body_id,
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        footprint=footprint,
        grip_width=footprint.narrow_width if obj.grip_width is None else obj.grip_width,
    )


def measure_all(model: mujoco.MjModel) -> dict[str, ObjectMetrics]:
    """Resolve and measure every catalogue object."""
    return {name: measure(model, name) for name in NAMES}

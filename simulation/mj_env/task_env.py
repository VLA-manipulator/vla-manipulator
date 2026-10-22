"""The spec-driven environment: a scene, a layout rule and a criterion, wired up.

``SO101TabletopEnv`` is the legacy eight-object tabletop and stays exactly as it
was -- it is what the 320 collected episodes were made with.  ``TaskEnv`` is the
generalisation: it takes a :class:`~mj_env.specs.registry.TaskSpec`, loads that
level's composed scene, samples the layout the spec prescribes, and reports
success using the spec's own criterion.

Three things this class is deliberately strict about:

* **Every free prop must be placed by the layout.**  A prop left at its XML
  pose is a prop whose position does not change with the seed, so the episode
  is not reproducible from the seed alone.
* **``success`` is the criterion plus a hold time**, never the criterion alone.
  Carrying the object *over* the goal must not read as placing it.
* **A reset must not already satisfy the criterion** ("no free success").  The
  scene self-test asserts this for every level; :attr:`TaskEnv.criterion_ok` is
  what it looks at.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import mujoco
import numpy as np

from . import gripper, objects
from .env import HOME_QPOS, SO101TabletopEnv
from .scene import SCENES_DIR, TABLE_HEIGHT
from .specs import TaskSpec, get
from .vfx import CLEAN_RUNG, Imaging, get_rung

__all__ = ["TaskEnv"]

#: Below this many metres of jaw opening the pads are still holding something.
_CONTAINER_CAVITY_GEOM = "container_cavity"


class TaskEnv(SO101TabletopEnv):
    """A graded scene level, driven entirely by its :class:`TaskSpec`."""

    def __init__(
        self,
        task_id: str,
        *,
        instruction: str | None = None,
        instruction_pool: str = "train",
        vfx: str = CLEAN_RUNG,
        vfx_seed: int | None = None,
        render_mode: str | None = None,
        image_size: tuple[int, int] = (224, 224),
        max_episode_steps: int = 600,
        **kwargs: Any,
    ) -> None:
        spec = get(task_id)
        self.spec: TaskSpec = spec
        self.instruction_pool = instruction_pool
        #: Imaging rung.  ``v0_clean`` keeps the parent's renderer untouched;
        #: anything else routes every camera through the water model.  Every
        #: level is collected in both conditions with the same seed, so this is
        #: the switch that makes a clean/underwater pair a pair.
        self.vfx_rung = vfx
        self._imaging: Imaging | None = None
        self._vfx_seed = vfx_seed
        super().__init__(
            scene_xml=Path(SCENES_DIR) / spec.scene,
            prop_bodies=spec.prop_bodies,
            prop_grip_widths=spec.grip_widths,
            target_object=spec.target,
            task=instruction,
            render_mode=render_mode,
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        self._static_ids = {name: self._body_id(name) for name in spec.statics}
        self._goal_body_id = self._body_id(spec.goal_body) if spec.goal_body else None
        self._free_dofs = {name: self._free_dof(name) for name in spec.prop_bodies}
        self._hold_frames = max(1, round(spec.hold_s * self.control_hz))
        self._jaw: gripper.JawCalibration | None = None

        self.layout: dict[str, Any] | None = None
        self.instruction: str | None = instruction
        self.instruction_kind: str = "none"
        self.init_props: dict[str, np.ndarray] = {}
        self.init_goal: np.ndarray | None = None
        self._hold_count = 0
        self._criterion_ok = False

    # -- wiring --------------------------------------------------------------

    def _body_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if value < 0:
            raise ValueError(f"scene {self.spec.scene!r} has no body named {name!r}")
        return int(value)

    def _free_dof(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        if joint < 0:
            raise ValueError(f"scene {self.spec.scene!r} has no joint named {name + '_free'!r}")
        return int(self.model.jnt_dofadr[joint])

    @property
    def jaw(self) -> gripper.JawCalibration:
        """Jaw geometry, calibrated off the meshes on first use."""
        if self._jaw is None:
            self._jaw = gripper.calibrate(self.model)
        return self._jaw

    # -- reset ---------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})
        # Seed the RNG with gymnasium's own bookkeeping; the parent's reset is
        # not called because the layout comes from the spec, not from the
        # eight-object catalogue sampler.
        gym.Env.reset(self, seed=seed)

        self.instruction, self.instruction_kind = self._pick_instruction(options)
        self.task = self.instruction or self._prompt_for(self.spec.target)
        # Levels whose language *chooses* the object (P2) resolve the target from
        # the instruction.  Everything downstream -- the expert, the criterion,
        # ``info["target_object"]`` -- then agrees on which object the episode is
        # about, which is the entire point of such a level.
        if self.spec.instruction_targets and self.instruction:
            self.target_object = self.spec.target_for_instruction(self.instruction)
        else:
            requested = options.get("target_object", self.spec.target)
            if requested not in self.objects_subset:
                raise ValueError(
                    f"target_object={requested!r} is not in {self.objects_subset!r}"
                )
            self.target_object = requested

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_ids] = HOME_QPOS
        self.data.ctrl[self._actuator_ids] = HOME_QPOS
        self._apply_layout()
        self._settle()

        self.init_props = {name: self.prop_pos(name) for name in self._objects}
        self.init_goal = self.goal_xy() if self._goal_body_id is not None else None
        self._grasp_latch = None
        self._step_count = 0
        self._criterion_ok = False

        observation = self._observation()
        info = self._info()
        # After _info(), so an already-satisfied layout is *reported* by
        # criterion_ok but can never be reported as success on frame zero.
        self._hold_count = 0
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _apply_layout(self) -> None:
        layout = self.spec.sample_layout(self.np_random)
        missing = set(self.spec.prop_bodies).difference(layout.get("props", {}))
        if missing:
            raise ValueError(
                f"{self.spec.task_id}: layout did not place {sorted(missing)}; "
                "every free prop must be sampled or the episode is not reproducible"
            )
        self.layout = layout
        self._layout_fallback = False

        for name, pose in layout["props"].items():
            metrics = self._objects[name]
            address = metrics.qpos_adr
            yaw = float(pose[3]) if len(pose) > 3 else 0.0
            self.data.qpos[address : address + 3] = (pose[0], pose[1], pose[2])
            self.data.qpos[address + 3 : address + 7] = (
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            )

        for name, pose in layout.get("statics", {}).items():
            if name not in self._static_ids:
                raise ValueError(f"{self.spec.task_id}: {name!r} is not a declared static prop")
            body = self._static_ids[name]
            self.model.body_pos[body] = pose[:3]
            # A static prop may carry a yaw: obstacle levels rotate their walls
            # with the target, so "the wall is this far from the object" stays
            # true whatever the object's heading is.
            yaw = float(pose[3]) if len(pose) > 3 else 0.0
            self.model.body_quat[body] = (
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            )

    def _pick_instruction(self, options: dict[str, Any]) -> tuple[str, str]:
        if "instruction" in options and options["instruction"] is not None:
            text = str(options["instruction"])
            kind = options.get("instruction_kind") or (
                "holdout" if text in self.spec.holdout else "train"
            )
            return text, str(kind)
        pool_name = str(options.get("instruction_pool", self.instruction_pool))
        pool = self.spec.instruction_pool(pool_name)
        if not pool:
            return self._prompt_for(self.spec.target), "none"
        text = pool[int(self.np_random.integers(len(pool)))]
        return text, ("holdout" if text in self.spec.holdout else "train")

    # -- criterion surface (the only thing criteria may use) -----------------

    def prop_pos(self, name: str = "object") -> np.ndarray:
        """World position of a prop's body origin."""
        return np.asarray(self.data.xpos[self._objects[name].body_id], dtype=float).copy()

    def rest_height(self, name: str = "object") -> float:
        """Body-origin height at which the body rests on the tabletop."""
        return TABLE_HEIGHT + self._objects[name].footprint.rest_clearance

    def place_height(self, name: str = "object") -> float:
        """Body-origin height the object has to end up at.

        The tabletop for every level except stacking, where the spec supplies the
        top of the support instead.  The expert asks *this*, so a stack level
        needs no expert of its own.
        """
        if self.spec.place_height is not None:
            return float(self.spec.place_height(self, name))
        return self.rest_height(name)

    def grip_squeeze(self, name: str = "object", default: float = 0.002) -> float:
        """Jaw interference per side while holding ``name``, in metres."""
        return float(self.spec.grip_squeezes.get(name, default))

    def grip_width(self, name: str = "object") -> float:
        """Width the jaw has to close to in order to hold ``name``, in metres."""
        return float(self._objects[name].grip_width)

    def height_above_rest(self, name: str = "object") -> float:
        return float(self.prop_pos(name)[2] - self.rest_height(name))

    def static_xy(self, name: str) -> np.ndarray:
        """World xy of a declared static prop.

        Used by multi-stage levels (O5) to find the marker body they park one
        object on before fetching another.  Statics are positioned by the layout
        on every reset, so this is the only correct way to read it.
        """
        if name not in self.spec.statics:
            raise ValueError(f"{self.spec.task_id}: {name!r} is not a declared static")
        return np.asarray(self.model.body_pos[self._static_ids[name]], dtype=float)[:2]

    def moved(self, name: str) -> float:
        """How far a prop has travelled since the episode started."""
        if name not in self.init_props:
            raise KeyError(f"no recorded initial position for {name!r}")
        return float(np.linalg.norm(self.prop_pos(name) - self.init_props[name]))

    def body_speed(self, name: str) -> float:
        """Linear speed of a free prop, in m/s."""
        dof = self._free_dofs[name]
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def grasp_opening(self) -> float:
        """Current jaw opening in metres, interpolated off the calibration."""
        angle = float(self.data.qpos[self._qpos_ids[-1]])
        return float(np.interp(angle, self.jaw.angles, self.jaw.openings))

    def grasp_open(self, width: float = 0.05) -> bool:
        return bool(self.grasp_opening() >= width)

    def goal_xy(self) -> np.ndarray:
        if self._goal_body_id is None:
            raise ValueError(f"{self.spec.task_id} has no goal body")
        return np.asarray(self.model.body_pos[self._goal_body_id], dtype=float)[:2].copy()

    def on_goal(self, name: str = "object", tol: float = 0.025) -> bool:
        return bool(np.linalg.norm(self.prop_pos(name)[:2] - self.goal_xy()) <= tol)

    def inside_container(self, name: str = "object", *, inset: float = 0.008) -> bool:
        """Whether a prop is inside the cavity marked by a ``container_cavity`` geom.

        Container levels mark their cavity explicitly with a zero-collision geom
        rather than having this function guess at wall thickness from an outer
        bounding box.  That keeps the criterion honest: the cavity is whatever
        the level declares it to be.
        """
        cavity = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, _CONTAINER_CAVITY_GEOM)
        if cavity < 0:
            raise ValueError(
                f"{self.spec.task_id} has no {_CONTAINER_CAVITY_GEOM!r} geom; "
                "container levels must declare their cavity explicitly"
            )
        body = int(self.model.geom_bodyid[cavity])
        # The cavity geom's *world* pose, not the parent body's: the cavity is
        # usually offset upward from the container's base, and using the body
        # origin would put the cavity half a wall-height too low and make the
        # criterion impossible to satisfy.
        centre = np.asarray(self.data.geom_xpos[cavity], dtype=float)
        half = np.asarray(self.model.geom_size[cavity], dtype=float)
        # Express the offset in the cavity's own frame, so a rotated container
        # still measures correctly.
        rotation = self.data.geom_xmat[cavity].reshape(3, 3)
        offset = rotation.T @ (self.prop_pos(name) - centre)
        inner = np.maximum(half - inset, 0.0)
        return bool(np.all(np.abs(offset) <= inner))

    def criterion_ok(self) -> bool:
        """The spec's criterion on *this* instant, with no hold time."""
        return bool(self.spec.success(self))

    def spec_hash(self) -> str:
        return self.spec.hash()

    @property
    def vfx_hash(self) -> str:
        """Imaging fingerprint: empty for the clean rung, the water hash otherwise.

        Written into every episode record next to ``layout_seed``.  A valid
        clean/underwater pair has the same layout seed and *different* hashes;
        two episodes with the same hash are two copies, not a comparison.
        """
        if self.vfx_rung == CLEAN_RUNG:
            return ""
        if self._imaging is None:
            return get_rung(self.vfx_rung).hash
        return self._imaging.params_hash

    # -- rendering -----------------------------------------------------------

    def render_camera(
        self, camera: str, *, height: int | None = None, width: int | None = None
    ) -> np.ndarray:
        """Render one camera, through the water model when a water rung is set.

        The clean rung delegates straight to the parent, so the legacy path and
        every clean episode are pixel-identical to what the 320-episode dataset
        was collected with.
        """
        if self.vfx_rung == CLEAN_RUNG:
            return super().render_camera(camera, height=height, width=width)
        height = self.image_height if height is None else height
        width = self.image_width if width is None else width
        if (
            self._imaging is None
            or self._imaging.height != height
            or self._imaging.width != width
        ):
            if self._imaging is not None:
                self._imaging.close()
            self._imaging = Imaging(
                self.model, width, height, self.vfx_rung, seed=self._vfx_seed
            )
        return self._imaging.render(self.data, camera)

    def close(self) -> None:
        if self._imaging is not None:
            self._imaging.close()
            self._imaging = None
        super().close()

    def static_footprints(self) -> dict[str, objects.Footprint]:
        """Measured footprints of the static props, for tests against the sampler."""
        return {
            name: objects.measure_footprint(self.model, body_id)
            for name, body_id in self._static_ids.items()
        }

    # -- observation ---------------------------------------------------------

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        ok = self.criterion_ok()
        self._criterion_ok = ok
        self._hold_count = self._hold_count + 1 if ok else 0
        held_long_enough = self._hold_count >= self._hold_frames

        info.update(
            {
                # The legacy grasp-and-lift verdict is kept *and* relabelled, so
                # an analysis can always separate "never picked it up" from
                # "picked it up but put it in the wrong place".
                "grasp_and_lift_success": bool(info["success"]),
                "criterion_ok": ok,
                "criterion_hold_frames": self._hold_count,
                "success": bool(ok and held_long_enough),
                "task_id": self.spec.task_id,
                "family": self.spec.family,
                "level": self.spec.level,
                "spec_hash": self.spec_hash(),
                "vfx_rung": self.vfx_rung,
                "vfx_hash": self.vfx_hash,
                "instruction": self.instruction,
                "instruction_kind": self.instruction_kind,
                "layout": self.layout,
                "init_props": {name: value.tolist() for name, value in self.init_props.items()},
                "goal_xy": self.init_goal.tolist() if self.init_goal is not None else None,
                "object_on_goal": self.on_goal() if self._goal_body_id is not None else None,
                "height_above_rest": self.height_above_rest(),
                "grasp_opening": self.grasp_opening(),
            }
        )
        return info
"""Regression checks for the experimental mesh-collision rollback."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from mj_env import gripper
from mj_env.scene import (
    FIXED_FINGER_GEOM,
    MOVING_FINGER_GEOM,
    build_model,
)


def _geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


class GripperMeshCollisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_model()
        cls.calibration = gripper.calibrate(cls.model)

    def test_finger_meshes_collide(self) -> None:
        for name in (FIXED_FINGER_GEOM, MOVING_FINGER_GEOM):
            geom = _geom_id(self.model, name)
            self.assertEqual(self.model.geom_contype[geom], 1, name)
            self.assertEqual(self.model.geom_conaffinity[geom], 1, name)

    def test_flat_pads_are_not_installed(self) -> None:
        for name in ("fixed_pad", "moving_pad"):
            self.assertEqual(_geom_id(self.model, name), -1, name)

    def test_opening_is_monotonic_and_spans_the_catalogue(self) -> None:
        openings = self.calibration.openings
        self.assertTrue(np.all(np.diff(openings) > 0), "opening must grow")
        # Narrowest catalogue grasp is the 16 mm dumbbell handle; widest is the
        # 40 mm ball.  Both have to land strictly inside the measured span.
        self.assertLess(self.calibration.min_opening, 0.016)
        self.assertGreater(self.calibration.max_opening, 0.040)

    def test_calibrated_opening_is_the_mesh_gap(self) -> None:
        """Sweep back from an opening to its angle and measure the meshes."""
        for width in (0.020, 0.035):
            angle = self.calibration.angle_for(width)
            data = mujoco.MjData(self.model)
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            data.qpos[:] = self.model.qpos0
            data.qpos[self.model.jnt_qposadr[joint]] = angle
            mujoco.mj_forward(self.model, data)

            vertices = [
                gripper._pad_vertices(self.model, data, name, joint)
                for name in (FIXED_FINGER_GEOM, MOVING_FINGER_GEOM)
            ]
            from scipy.spatial import cKDTree

            gap = cKDTree(vertices[0]).query(vertices[1])[0].min()
            self.assertAlmostEqual(gap, width, delta=0.003, msg=f"{width}")


class LegacyPickContactRegressionTest(unittest.TestCase):
    """Mesh contacts must still support the legacy pick tasks."""

    def test_edge_layouts_remain_graspable(self) -> None:
        from mj_env.env import SO101TabletopEnv
        from mj_env.scripts.scripted_pick import GraspSolver, PickTuning, run_episode

        env = SO101TabletopEnv(max_episode_steps=10_000)
        try:
            solver = GraspSolver(env.model)
            for target, seed in (("cube_blue", 0), ("cube_red", 3)):
                with self.subTest(target=target, seed=seed):
                    body_id = env._objects[target].body_id
                    finger_ids = set(env._finger_geom_ids)
                    contact_depths: list[float] = []

                    def record_contact(_action, _observation, _info) -> None:
                        for contact in env.data.contact[: env.data.ncon]:
                            geom1, geom2 = int(contact.geom1), int(contact.geom2)
                            if (geom1 in finger_ids or geom2 in finger_ids) and body_id in (
                                int(env.model.geom_bodyid[geom1]),
                                int(env.model.geom_bodyid[geom2]),
                            ):
                                contact_depths.append(float(contact.dist))

                    result = run_episode(
                        env,
                        solver,
                        seed=seed,
                        target=target,
                        tuning=PickTuning(),
                        on_frame=record_contact,
                    )
                    self.assertTrue(result["success"], result)
                    self.assertTrue(contact_depths, "no finger-to-object contact")
        finally:
            env.close()

    def test_composed_scene_edge_layout_remains_graspable(self) -> None:
        from mj_env.scripts.scripted_pick import GraspSolver
        from mj_env.scripts.scripted_pick_place import run_pick_place
        from mj_env.task_env import TaskEnv

        env = TaskEnv("o1_open_baseline", max_episode_steps=10_000)
        try:
            result = run_pick_place(env, GraspSolver(env.model), seed=1)
            self.assertTrue(result["success"], result)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

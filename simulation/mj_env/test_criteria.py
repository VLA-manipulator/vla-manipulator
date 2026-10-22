"""Criterion tests: the negative controls a success predicate has to pass.

Every criterion needs a case where "the expert did not do the step" returns
``False``.  Without it a criterion that the *starting layout* already satisfies
("free success") quietly inflates every success rate in the ladder table, and
nothing downstream would notice.

These run in a second or two and need no expert and no rendering.

    uv run python -m unittest mj_env.test_criteria
"""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from mj_env.specs import get
from mj_env.specs import criteria
from mj_env.specs.family_p import (
    MAX_SEPARATION,
    MIN_SEPARATION,
    layout_p1,
    layout_p2,
)
from mj_env.task_env import TaskEnv


def _place(env: TaskEnv, name: str, xyz) -> None:
    """Teleport a free prop, as if the expert had put it there."""
    metrics = env._objects[name]
    address = metrics.qpos_adr
    env.data.qpos[address : address + 3] = xyz
    env.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


class PlaceDiscCriterionTest(unittest.TestCase):
    """``p1_place_disc``: the criterion must be true only for a real placement."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = TaskEnv("p1_place_disc")
        cls.spec = get("p1_place_disc")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.env.reset(seed=7)
        self.goal = self.env.goal_xy()
        self.rest = self.env.rest_height("object")

    # -- negative controls ---------------------------------------------------
    def test_starting_layout_is_not_a_success(self) -> None:
        """No free success: resetting must never satisfy the criterion."""
        self.assertFalse(self.env.criterion_ok())

    def test_object_carried_over_the_goal_is_not_a_success(self) -> None:
        """Above the goal but still in the air: not placed."""
        _place(self.env, "object", (self.goal[0], self.goal[1], self.rest + 0.12))
        self.assertFalse(criteria.on_place_target(self.env, "object"))
        self.assertFalse(self.spec.success(self.env))

    def test_object_moved_but_off_the_goal_is_not_a_success(self) -> None:
        """Somewhere else on the table: not placed."""
        _place(self.env, "object", (self.goal[0] + 0.08, self.goal[1] + 0.06, self.rest))
        self.assertFalse(criteria.on_place_target(self.env, "object"))

    def test_object_on_the_goal_with_the_jaw_still_shut_is_not_a_success(self) -> None:
        """Placed but never let go: the release is part of the task."""
        _place(self.env, "object", (self.goal[0], self.goal[1], self.rest))
        shut = self.env.jaw.angle_for(self.env._objects["object"].grip_width - 0.004)
        self.env.data.qpos[self.env._qpos_ids[-1]] = shut
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertTrue(criteria.on_place_target(self.env, "object"))
        self.assertFalse(criteria.released(self.env))
        self.assertFalse(self.spec.success(self.env))

    # -- positive control ----------------------------------------------------
    def test_object_resting_on_the_goal_with_an_open_jaw_is_a_success(self) -> None:
        _place(self.env, "object", (self.goal[0] + 0.004, self.goal[1] - 0.003, self.rest))
        self.env.data.qpos[self.env._qpos_ids[-1]] = self.env.jaw.angle_for(0.08)
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertTrue(criteria.on_place_target(self.env, "object"))
        self.assertTrue(criteria.released(self.env))
        self.assertTrue(self.spec.success(self.env))

    # -- criterion parts -----------------------------------------------------
    def test_rested_needs_the_body_to_have_stopped(self) -> None:
        _place(self.env, "object", (self.goal[0], self.goal[1], self.rest))
        dof = self.env._free_dofs["object"]
        self.env.data.qvel[dof : dof + 3] = (0.5, 0.0, 0.0)
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertFalse(criteria.rested(self.env, "object", max_speed=0.02))
        self.assertFalse(criteria.on_place_target(self.env, "object"))

    def test_undisturbed_measures_travel_since_reset(self) -> None:
        before = self.env.prop_pos("object")
        self.assertAlmostEqual(self.env.moved("object"), 0.0, places=6)
        _place(self.env, "object", before + np.array([0.02, 0.0, 0.0]))
        self.assertGreater(self.env.moved("object"), 0.01)


class LanguageReferenceCriterionTest(unittest.TestCase):
    """``p2_language_ref``: the sentence picks the object, and the wrong one fails.

    The "other cube untouched" clause is what makes this level meaningful.  A
    policy that ignores the instruction and always grabs the same cube would
    otherwise still succeed whenever the sentence happened to name that cube --
    50% on a two-cube scene, which is exactly the number that hides the problem.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = TaskEnv("p2_language_ref")
        cls.spec = get("p2_language_ref")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.env.reset(seed=5, options={"instruction": "put the red cube on the blue disc"})
        self.goal = self.env.goal_xy()
        self.rest = self.env.rest_height("object")

    def _open_jaw(self) -> None:
        self.env.data.qpos[self.env._qpos_ids[-1]] = self.env.jaw.angle_for(0.08)
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_instruction_selects_the_target(self) -> None:
        self.assertEqual(self.env.target_object, "object")
        self.env.reset(seed=5, options={"instruction": "put the yellow cube on the blue disc"})
        self.assertEqual(self.env.target_object, "object2")

    def test_unmapped_instruction_is_refused(self) -> None:
        """A sentence that names nothing must fail loudly, not silently default."""
        with self.assertRaises(KeyError):
            self.env.reset(seed=5, options={"instruction": "put the green cube on the disc"})

    def test_named_cube_on_the_goal_is_a_success(self) -> None:
        _place(self.env, "object", (self.goal[0], self.goal[1], self.rest))
        self._open_jaw()
        self.assertTrue(self.spec.success(self.env))

    def test_the_other_cube_on_the_goal_is_not_a_success(self) -> None:
        """The whole point: moving the wrong object must not count."""
        _place(self.env, "object2", (self.goal[0], self.goal[1], self.rest))
        self._open_jaw()
        self.assertFalse(criteria.on_place_target(self.env, "object"))
        self.assertFalse(self.spec.success(self.env))

    def test_named_cube_on_the_goal_but_other_cube_shoved_is_not_a_success(self) -> None:
        _place(self.env, "object", (self.goal[0], self.goal[1], self.rest))
        self._open_jaw()
        before = self.env.prop_pos("object2").copy()
        _place(self.env, "object2", before + np.array([0.03, 0.0, 0.0]))
        self.assertTrue(criteria.on_place_target(self.env, "object"))
        self.assertFalse(criteria.undisturbed(self.env, "object2", tol=0.010))
        self.assertFalse(self.spec.success(self.env))

    def test_starting_layout_is_not_a_success(self) -> None:
        self.assertFalse(self.env.criterion_ok())


class LayoutTest(unittest.TestCase):
    """The sampler's contract: every prop placed, inside the documented band."""

    def test_layout_places_the_object_and_the_goal(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            layout = layout_p1(rng)
            self.assertIn("object", layout["props"])
            self.assertIn("place_target", layout["statics"])
            self.assertEqual(len(layout["goal_xy"]), 2)

    def test_separation_stays_inside_the_configured_window(self) -> None:
        """A carry longer than MAX_SEPARATION sheds the object (measured)."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            layout = layout_p1(rng)
            obj = np.asarray(layout["props"]["object"][:2])
            goal = np.asarray(layout["goal_xy"])
            separation = float(np.linalg.norm(obj - goal))
            self.assertGreaterEqual(separation, MIN_SEPARATION - 1e-9)
            self.assertLessEqual(separation, MAX_SEPARATION + 1e-9)

    def test_object_and_goal_stay_inside_the_measured_grasp_band(self) -> None:
        """Both must remain inside the band M0 measured as 100% solvable."""
        rng = np.random.default_rng(2)
        for _ in range(50):
            layout = layout_p1(rng)
            for point in (layout["props"]["object"][:2], layout["goal_xy"]):
                radius = math.hypot(*point)
                self.assertGreater(radius, 0.16)
                self.assertLess(radius, 0.30)

    def test_spec_hash_is_stable_and_identifies_the_level(self) -> None:
        spec = get("p1_place_disc")
        self.assertEqual(spec.hash(), get("p1_place_disc").hash())
        self.assertNotEqual(spec.hash(), get("o1_open_baseline").hash())

    def test_every_level_declares_instructions_and_a_holdout(self) -> None:
        """Language ablation needs >= 3 training templates and >= 1 hold-out."""
        for task_id in ("p1_place_disc", "p2_language_ref", "o1_open_baseline"):
            spec = get(task_id)
            self.assertGreaterEqual(len(spec.instructions), 3, task_id)
            self.assertGreaterEqual(len(spec.holdout), 1, task_id)
            self.assertTrue(set(spec.instructions).isdisjoint(spec.holdout), task_id)

    def test_p2_layout_keeps_both_cubes_in_the_carry_window(self) -> None:
        """A cube further than the carry window sheds the object mid-carry."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            layout = layout_p2(rng)
            goal = np.asarray(layout["goal_xy"])
            first = np.asarray(layout["props"]["object"][:2])
            second = np.asarray(layout["props"]["object2"][:2])
            self.assertGreaterEqual(float(np.linalg.norm(first - goal)), MIN_SEPARATION - 1e-9)
            self.assertLessEqual(float(np.linalg.norm(first - goal)), MAX_SEPARATION + 1e-9)
            self.assertLessEqual(float(np.linalg.norm(second - goal)), MAX_SEPARATION + 1e-9)
            self.assertGreater(float(np.linalg.norm(first - second)), 0.05)

    def test_every_language_instruction_maps_to_a_declared_body(self) -> None:
        spec = get("p2_language_ref")
        for text in (*spec.instructions, *spec.holdout):
            body = spec.target_for_instruction(text)
            self.assertIn(body, spec.prop_bodies, text)


if __name__ == "__main__":
    unittest.main()
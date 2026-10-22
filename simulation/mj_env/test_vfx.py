"""Imaging-layer tests: the pairing invariant, and that clean is still clean.

Two things have to hold or the whole clean/underwater comparison is worthless:

1. **A pair must differ in nothing but the imaging.**  Same seed -> same layout,
   same prop positions, same goal, same instruction; only ``vfx_hash`` and the
   pixels differ.  The scene self-test will assert this for every collected
   episode; this is the same assertion at unit-test speed.
2. **The clean rung must not change the picture.**  It is the control side, so
   it has to be pixel-identical to what the parent environment renders -- the
   same renderer the existing 320-episode dataset was collected with.

Water tests are skipped when ``underwater_vision`` is not importable, so the
suite still runs on a machine without the WYSD checkout.

    PYTHONPATH=<WYSD> uv run python -m unittest mj_env.test_vfx
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from mj_env.env import SO101TabletopEnv
from mj_env.task_env import TaskEnv
from mj_env.vfx import CLEAN_RUNG, LADDER, load_rungs

try:
    import underwater_vision  # noqa: F401

    HAVE_WATER = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_WATER = False

TASK = "p1_place_disc"
SEED = 11


def _observation(env: TaskEnv):
    observation, _ = env.reset(seed=SEED)
    return observation


class CleanRungTest(unittest.TestCase):
    """The control side must be untouched."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = TaskEnv(TASK)
        cls.env.reset(seed=SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_clean_rung_is_the_default(self) -> None:
        self.assertEqual(self.env.vfx_rung, CLEAN_RUNG)
        self.assertEqual(self.env.vfx_hash, "")

    def test_clean_rung_delegates_to_the_parent_renderer(self) -> None:
        """Delegation, verified rather than assumed.

        A sentinel makes this a pure statement about the call path: the clean
        rung must return whatever the parent environment returned, without
        touching the array.  Comparing two real renders cannot prove that, and
        it is not even reproducible -- see the test below.
        """
        sentinel = np.zeros((4, 4, 3), dtype=np.uint8)
        with mock.patch.object(
            SO101TabletopEnv, "render_camera", return_value=sentinel
        ):
            self.assertIs(self.env.render_camera("global_camera"), sentinel)

    def test_clean_render_matches_the_parent(self) -> None:
        """The clean rung must not change the picture.

        Two renders of the *same* scene are not bit-identical on every GPU: the
        wrist view resolves 18 of 150528 pixels (always the same ones) one
        8-bit step apart on roughly 12% of consecutive frames.  That is
        rasterisation jitter, not a different picture, so a one-step delta on
        under 0.1% of pixels is tolerated -- a real change of imaging would
        move whole regions and fail both bounds.
        """
        for camera in ("global_camera", "wrist_camera"):
            delegated = self.env.render_camera(camera)
            parent = super(TaskEnv, self.env).render_camera(camera)
            self.assertEqual(delegated.shape, parent.shape, camera)
            delta = np.abs(delegated.astype(np.int16) - parent.astype(np.int16))
            self.assertLessEqual(int(delta.max()), 1, camera)
            changed = float((delta.max(axis=-1) > 0).mean())
            self.assertLessEqual(changed, 1e-3, camera)

    def test_observation_reports_the_imaging_identity(self) -> None:
        _, info = self.env.reset(seed=SEED)
        self.assertEqual(info["vfx_rung"], CLEAN_RUNG)
        self.assertEqual(info["vfx_hash"], "")


@unittest.skipUnless(HAVE_WATER, "underwater_vision is not importable")
class PairingTest(unittest.TestCase):
    """A clean/underwater pair, on one seed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rungs = load_rungs()
        if not cls.rungs:
            raise unittest.SkipTest(
                "no calibrated rungs; run mj_env.scripts.water_probe first"
            )
        cls.clean = TaskEnv(TASK, vfx=CLEAN_RUNG)
        cls.water = TaskEnv(TASK, vfx="v2_medium", vfx_seed=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.clean.close()
        cls.water.close()

    def test_ladder_is_complete_and_hashes_are_distinct(self) -> None:
        for name in LADDER:
            self.assertIn(name, self.rungs, name)
        hashes = [rung.hash for name, rung in self.rungs.items() if not rung.clean]
        self.assertEqual(len(hashes), len(set(hashes)), "water rungs share a hash")
        self.assertTrue(all(hashes), "a water rung has an empty hash")

    def test_same_seed_gives_the_same_layout_on_both_sides(self) -> None:
        _, clean_info = self.clean.reset(seed=SEED)
        _, water_info = self.water.reset(seed=SEED)
        self.assertEqual(clean_info["layout"], water_info["layout"])
        self.assertEqual(clean_info["goal_xy"], water_info["goal_xy"])
        self.assertEqual(clean_info["task_id"], water_info["task_id"])
        for name, position in clean_info["init_props"].items():
            np.testing.assert_allclose(
                position, water_info["init_props"][name], atol=1e-12, err_msg=name
            )

    def test_the_pair_differs_in_the_imaging_identity_and_in_pixels(self) -> None:
        clean_observation, clean_info = self.clean.reset(seed=SEED)
        water_observation, water_info = self.water.reset(seed=SEED)
        self.assertNotEqual(clean_info["vfx_hash"], water_info["vfx_hash"])
        self.assertFalse(water_info["vfx_hash"] == "")

        for key in ("observation.image", "observation.wrist_image"):
            difference = np.abs(
                clean_observation[key].astype(int) - water_observation[key].astype(int)
            )
            self.assertGreater(
                float(difference.mean()),
                2.0,
                f"{key} barely changed; the water rung is not doing anything",
            )

    def test_state_and_action_space_are_unchanged_by_the_imaging(self) -> None:
        """The schema must not depend on which side of the pair you are on."""
        clean_observation, _ = self.clean.reset(seed=SEED)
        water_observation, _ = self.water.reset(seed=SEED)
        self.assertEqual(clean_observation["observation.state"].shape, (6,))
        self.assertEqual(water_observation["observation.state"].shape, (6,))
        self.assertEqual(clean_observation["observation.image"].shape, (224, 224, 3))
        self.assertEqual(water_observation["observation.image"].shape, (224, 224, 3))
        self.assertEqual(self.clean.action_space.shape, self.water.action_space.shape)


if __name__ == "__main__":
    unittest.main()
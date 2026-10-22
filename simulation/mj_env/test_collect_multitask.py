"""Regression checks for the separate graded-task collection path."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from mj_env.scripts.collect_multitask import _quota
from mj_env.scripts.scripted_pick_place import run_cover_then_place


class MultitaskCollectionTests(unittest.TestCase):
    def test_quota_parses_legacy_and_graded_tasks(self):
        self.assertEqual(_quota("cube_red=20"), ("cube_red", 20))
        self.assertEqual(_quota("o5_covered_stack=40"), ("o5_covered_stack", 40))

    def test_cover_failure_does_not_crash_and_uses_seeded_layout(self):
        events: list[str] = []

        class Env:
            def reset(self, *, seed, options):
                self.seed = seed
                events.append("reset")

            def static_xy(self, name):
                self_name = name
                events.append("static")
                assert self_name == "park_spot"
                return np.array([0.1, 0.2])

        with patch(
            "mj_env.scripts.scripted_pick_place.run_pick_place",
            return_value={"outcome": "ik_fail", "success": False},
        ) as pick:
            result = run_cover_then_place(Env(), object(), seed=10000)
        self.assertEqual(events, ["reset", "static"])
        self.assertEqual(result["outcome"], "cover_failed")
        self.assertFalse(result["success"])
        self.assertFalse(pick.call_args.kwargs["reset"])


if __name__ == "__main__":
    unittest.main()

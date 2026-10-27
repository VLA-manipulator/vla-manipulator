"""Check that RTC guidance follows SO101's current-state action coordinates."""

from __future__ import annotations

import unittest

import numpy as np

from openpi_ext.rtc_target import prepare_guided_inputs


class RTCTargetTest(unittest.TestCase):
    def test_reanchors_joint_deltas_and_preserves_absolute_gripper(self):
        def transform(request):
            state = np.asarray(request["observation/state"])
            actions = np.asarray(request["actions"]).copy()
            actions[:, :5] -= state[:5]
            actions = (actions - mean) / std
            return {"state": state, "actions": np.pad(actions, ((0, 0), (0, 2)))}

        mean = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        std = np.array([2, 2, 2, 2, 2, 3], dtype=np.float32)
        old_absolute = np.array([[11, 12, 13, 14, 15, 7]], dtype=np.float32)
        observation = {"observation/state": np.array([2, 3, 4, 5, 6, 99], dtype=np.float32)}
        inputs, target = prepare_guided_inputs(
            transform, observation, old_absolute, horizon=1, action_dim=8
        )
        np.testing.assert_allclose(target[0, :5], [4, 3.5, 3, 2.5, 2])
        np.testing.assert_allclose(target[0, 5], (7 - 6) / 3)
        self.assertNotIn("actions", inputs)
        np.testing.assert_array_equal(old_absolute, [[11, 12, 13, 14, 15, 7]])

    def test_rejects_mismatched_physical_action_width(self):
        with self.assertRaises(ValueError):
            prepare_guided_inputs(
                lambda request: request,
                {"observation/state": np.zeros(6)},
                np.zeros((10, 32)),
                horizon=10,
                action_dim=32,
            )


if __name__ == "__main__":
    unittest.main()

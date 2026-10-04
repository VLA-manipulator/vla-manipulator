"""Tests for synchronized demonstration recording without requiring LeRobot."""

from __future__ import annotations

import unittest

import numpy as np

from mj_env.demonstrations import SynchronizedEpisodeRecorder, lerobot_features


class _Dataset:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


class _Adapter:
    def to_lerobot(self, value):
        return np.asarray(value, dtype=np.float32) * 10.0


def _observation():
    return {
        "observation.state": np.arange(6, dtype=np.float32),
        "observation.image": np.zeros((8, 10, 3), dtype=np.uint8),
        "observation.wrist_image": np.ones((8, 10, 3), dtype=np.uint8),
    }


class DemonstrationRecorderTest(unittest.TestCase):
    def test_pairs_pre_action_frame_and_fixed_rate_timestamp(self):
        dataset = _Dataset()
        recorder = SynchronizedEpisodeRecorder(dataset, _Adapter(), 20, "pick")
        recorder.on_frame(np.ones(6), _observation(), {"simulation_time_s": 3.0})
        recorder.on_frame(np.ones(6) * 2, _observation(), {"simulation_time_s": 3.05})

        self.assertEqual(len(dataset.frames), 2)
        self.assertEqual(dataset.frames[0]["timestamp"], 0.0)
        self.assertEqual(dataset.frames[1]["timestamp"], 0.05)
        np.testing.assert_array_equal(dataset.frames[0]["actions"], np.ones(6) * 10)
        self.assertEqual(dataset.frames[0]["task"], "pick")

    def test_rejects_timing_gap(self):
        recorder = SynchronizedEpisodeRecorder(_Dataset(), _Adapter(), 20, "pick")
        recorder.on_frame(np.ones(6), _observation(), {"simulation_time_s": 1.0})
        with self.assertRaisesRegex(ValueError, "non-uniform demonstration timing"):
            recorder.on_frame(np.ones(6), _observation(), {"simulation_time_s": 1.08})

    def test_schema_has_two_cameras_and_six_coordinates(self):
        features = lerobot_features(480, 640, use_videos=True)
        self.assertEqual(features["image"]["shape"], (480, 640, 3))
        self.assertEqual(features["image"]["dtype"], "video")
        self.assertEqual(features["state"]["shape"], (6,))
        self.assertEqual(features["actions"]["shape"], (6,))


if __name__ == "__main__":
    unittest.main()

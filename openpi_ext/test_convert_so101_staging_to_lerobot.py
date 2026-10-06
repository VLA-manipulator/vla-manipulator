from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openpi_ext.convert_so101_staging_to_lerobot import convert_dataset


class _FakeDataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved_episodes = 0
        self.stopped = False

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1

    def stop_image_writer(self) -> None:
        self.stopped = True


class ConverterTest(unittest.TestCase):
    def test_converts_valid_episode_without_lerobot_installed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "staging"
            episodes_dir = source / "episodes"
            episodes_dir.mkdir(parents=True)
            (source / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": "so101_staging_v1",
                        "fps": 20,
                        "joint_names": [
                            "shoulder_pan",
                            "shoulder_lift",
                            "elbow_flex",
                            "wrist_flex",
                            "wrist_roll",
                            "gripper",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (source / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "path": "episodes/episode_000000.npz",
                        "frames": 2,
                        "task": "pick up the red cube",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            np.savez_compressed(
                episodes_dir / "episode_000000.npz",
                image=np.zeros((2, 8, 10, 3), dtype=np.uint8),
                wrist_image=np.ones((2, 8, 10, 3), dtype=np.uint8),
                state=np.zeros((2, 6), dtype=np.float32),
                actions=np.ones((2, 6), dtype=np.float32),
                timestamp=np.array([0.0, 0.05], dtype=np.float64),
                task=np.asarray("pick up the red cube"),
            )

            fake = _FakeDataset()
            captured: dict = {}

            def factory(**kwargs):
                captured.update(kwargs)
                return fake

            count, frames = convert_dataset(
                source,
                root / "lerobot",
                "local/test",
                use_videos=False,
                dataset_factory=factory,
            )

            self.assertEqual((count, frames), (1, 2))
            self.assertEqual(captured["fps"], 20)
            self.assertEqual(captured["features"]["state"]["shape"], (6,))
            self.assertEqual(captured["features"]["image"]["shape"], (8, 10, 3))
            self.assertEqual(len(fake.frames), 2)
            self.assertNotIn("timestamp", fake.frames[1])
            self.assertEqual(fake.saved_episodes, 1)
            self.assertTrue(fake.stopped)


if __name__ == "__main__":
    unittest.main()

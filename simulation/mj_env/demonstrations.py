"""Synchronized SO101 demonstration frames for LeRobot datasets.

The scripted expert calls the recorder immediately before ``env.step``.  Each
saved frame therefore means ``observation[t] -> action[t]``.  MuJoCo simulation
time, rather than wall-clock time, is used to verify the fixed-rate sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .adapters import LeRobotSO101Adapter


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class FrameDataset(Protocol):
    """Small portion of the LeRobotDataset API used by the recorder."""

    def add_frame(self, frame: dict) -> None: ...


@dataclass
class StagingDataset:
    """Small dependency-free episode store for later LeRobot conversion."""

    root: Path
    fps: int
    _frames: list[dict] = field(default_factory=list, init=False)
    _episode_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.root.exists():
            raise FileExistsError(f"staging root already exists: {self.root}")
        (self.root / "episodes").mkdir(parents=True)
        metadata = {
            "format": "so101_staging_v1",
            "fps": self.fps,
            "joint_names": list(JOINT_NAMES),
            "state_units": "normalized_motor_coordinates",
            "action_units": "normalized_motor_coordinates",
            "pairing": "observation[t] -> action[t]",
        }
        (self.root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_frame(self, frame: dict) -> None:
        self._frames.append(frame)

    def save_episode(self, *, metadata: dict | None = None) -> Path:
        if not self._frames:
            raise ValueError("cannot save an empty episode")
        tasks = {str(frame["task"]) for frame in self._frames}
        if len(tasks) != 1:
            raise ValueError(f"episode must have exactly one task, got {tasks}")
        path = self.root / "episodes" / f"episode_{self._episode_index:06d}.npz"
        np.savez_compressed(
            path,
            image=np.stack([frame["image"] for frame in self._frames]),
            wrist_image=np.stack([frame["wrist_image"] for frame in self._frames]),
            state=np.stack([frame["state"] for frame in self._frames]).astype(np.float32),
            actions=np.stack([frame["actions"] for frame in self._frames]).astype(np.float32),
            timestamp=np.asarray([frame["timestamp"] for frame in self._frames], dtype=np.float64),
            task=np.asarray(next(iter(tasks))),
        )
        manifest_record = {
            "episode_index": self._episode_index,
            "path": path.relative_to(self.root).as_posix(),
            "frames": len(self._frames),
            "task": next(iter(tasks)),
            **(metadata or {}),
        }
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
        self._episode_index += 1
        self._frames.clear()
        return path

    def clear_episode_buffer(self) -> None:
        self._frames.clear()


@dataclass
class SynchronizedEpisodeRecorder:
    """Validate and append one fixed-rate expert episode."""

    dataset: FrameDataset
    adapter: LeRobotSO101Adapter
    fps: int
    task: str
    timestamp_tolerance_s: float = 1e-5

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.frame_index = 0
        self.start_simulation_time_s: float | None = None

    def on_frame(self, action: np.ndarray, observation: dict, info: dict) -> None:
        """Append the pre-action observation and its corresponding action."""
        required = {
            "observation.state",
            "observation.image",
            "observation.wrist_image",
        }
        missing = required.difference(observation)
        if missing:
            raise KeyError(f"observation is missing fields: {sorted(missing)}")

        simulation_time_s = float(info["simulation_time_s"])
        if self.start_simulation_time_s is None:
            self.start_simulation_time_s = simulation_time_s
        relative_time_s = simulation_time_s - self.start_simulation_time_s
        expected_time_s = self.frame_index / self.fps
        if not np.isclose(
            relative_time_s,
            expected_time_s,
            rtol=0.0,
            atol=self.timestamp_tolerance_s,
        ):
            raise ValueError(
                "non-uniform demonstration timing: "
                f"frame={self.frame_index}, simulation_time={relative_time_s:.9f}, "
                f"expected={expected_time_s:.9f}"
            )

        state = self._vector(observation["observation.state"], "state")
        action = self._vector(action, "action")
        image = self._image(observation["observation.image"], "image")
        wrist_image = self._image(
            observation["observation.wrist_image"], "wrist_image"
        )

        self.dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": self.adapter.to_lerobot(state),
                "actions": self.adapter.to_lerobot(action),
                "task": self.task,
                # LeRobot would generate the same value automatically.  We
                # pass it explicitly so its own synchronization validator also
                # checks the exact convention used here.
                "timestamp": expected_time_s,
            }
        )
        self.frame_index += 1

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (len(JOINT_NAMES),):
            raise ValueError(f"{name} must have shape (6,), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array.copy()

    @staticmethod
    def _image(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (H, W, 3), got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {array.dtype}")
        return np.ascontiguousarray(array)


def lerobot_features(height: int, width: int, *, use_videos: bool) -> dict:
    """Return the fixed SO101 feature schema used by training and inference."""
    camera_dtype = "video" if use_videos else "image"
    vector_names = list(JOINT_NAMES)
    return {
        "image": {
            "dtype": camera_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": camera_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (6,),
            "names": vector_names,
        },
        "actions": {
            "dtype": "float32",
            "shape": (6,),
            "names": vector_names,
        },
    }


__all__ = [
    "JOINT_NAMES",
    "StagingDataset",
    "SynchronizedEpisodeRecorder",
    "lerobot_features",
]

"""Convert ``so101_staging_v1`` episodes into a LeRobot dataset.

Run this script from the virtual environment of the official OpenPI checkout,
not from the MuJoCo simulation environment.  The import of LeRobot is delayed
until conversion starts so the staging validator can be tested without it.

Example::

    uv run python /path/to/vla-manipulator/openpi_ext/convert_so101_staging_to_lerobot.py \
        --input-root /root/autodl-tmp/datasets/so101_staging/cube_red_smoke_5 \
        --output-root /root/autodl-tmp/datasets/so101_lerobot/cube_red_smoke_5 \
        --repo-id local/so101_cube_red_smoke_5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


FORMAT_NAME = "so101_staging_v1"
EXPECTED_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def load_staging_metadata(input_root: Path) -> dict[str, Any]:
    """Load and validate the dataset-level staging metadata."""
    metadata_path = input_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing staging metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(
            f"unsupported staging format {metadata.get('format')!r}; expected {FORMAT_NAME!r}"
        )
    fps = metadata.get("fps")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"metadata fps must be a positive integer, got {fps!r}")
    if tuple(metadata.get("joint_names", ())) != EXPECTED_JOINT_NAMES:
        raise ValueError("staging joint order does not match the SO101 adapter")
    return metadata


def load_manifest(input_root: Path) -> list[dict[str, Any]]:
    """Load the ordered episode manifest and reject unsafe paths."""
    manifest_path = input_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing staging manifest: {manifest_path}")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("staging manifest contains no episodes")
    expected_indices = list(range(len(records)))
    indices = [record.get("episode_index") for record in records]
    if indices != expected_indices:
        raise ValueError(f"episode indices must be contiguous, got {indices}")
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe episode path in manifest: {relative}")
    return records


def lerobot_features(height: int, width: int, *, use_videos: bool) -> dict[str, dict]:
    """Return the feature schema expected by the project OpenPI adapter."""
    camera_dtype = "video" if use_videos else "image"
    vector_names = list(EXPECTED_JOINT_NAMES)
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
        "state": {"dtype": "float32", "shape": (6,), "names": vector_names},
        "actions": {"dtype": "float32", "shape": (6,), "names": vector_names},
    }


def _validate_episode(
    episode: Any, record: dict[str, Any], *, fps: int
) -> tuple[int, int, int, str]:
    required = {"image", "wrist_image", "state", "actions", "timestamp", "task"}
    missing = required.difference(episode.files)
    if missing:
        raise ValueError(f"episode {record['episode_index']} is missing {sorted(missing)}")

    image = episode["image"]
    wrist_image = episode["wrist_image"]
    state = episode["state"]
    actions = episode["actions"]
    timestamp = episode["timestamp"]
    task = str(episode["task"].item())
    frame_count = int(record["frames"])

    if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"invalid image array: shape={image.shape}, dtype={image.dtype}")
    if wrist_image.shape != image.shape or wrist_image.dtype != np.uint8:
        raise ValueError("wrist_image must match the uint8 scene image shape")
    if image.shape[0] != frame_count:
        raise ValueError("manifest frame count does not match image array")
    if state.shape != (frame_count, 6) or state.dtype != np.float32:
        raise ValueError(f"invalid state array: shape={state.shape}, dtype={state.dtype}")
    if actions.shape != (frame_count, 6) or actions.dtype != np.float32:
        raise ValueError(f"invalid actions array: shape={actions.shape}, dtype={actions.dtype}")
    expected_timestamps = np.arange(frame_count, dtype=np.float64) / fps
    if timestamp.shape != (frame_count,) or not np.allclose(
        timestamp, expected_timestamps, rtol=0.0, atol=1e-6
    ):
        raise ValueError("episode timestamps are not a uniform zero-based sequence")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("episode contains non-finite state or action values")
    if not task or task != record.get("task"):
        raise ValueError("episode task does not match its manifest record")
    return frame_count, int(image.shape[1]), int(image.shape[2]), task


def inspect_staging(input_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    """Validate every episode before creating any output directory."""
    metadata = load_staging_metadata(input_root)
    records = load_manifest(input_root)
    image_shape: tuple[int, int] | None = None
    for record in records:
        episode_path = input_root / record["path"]
        if not episode_path.is_file():
            raise FileNotFoundError(f"missing episode file: {episode_path}")
        with np.load(episode_path, allow_pickle=False) as episode:
            _, height, width, _ = _validate_episode(episode, record, fps=metadata["fps"])
        if image_shape is None:
            image_shape = (height, width)
        elif image_shape != (height, width):
            raise ValueError("all episodes must use the same image resolution")
    assert image_shape is not None
    return metadata, records, image_shape[0], image_shape[1]


def convert_dataset(
    input_root: Path,
    output_root: Path,
    repo_id: str,
    *,
    use_videos: bool = True,
    image_writer_threads: int = 4,
    dataset_factory: Callable[..., Any] | None = None,
    push_to_hub: bool = False,
    private: bool = True,
) -> tuple[int, int]:
    """Validate staging data, create LeRobot output, and return episode/frame totals."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists: {output_root}; choose a new path to avoid overwriting data"
        )
    metadata, records, height, width = inspect_staging(input_root)

    if dataset_factory is None:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is unavailable. Run this converter from the official OpenPI virtual environment."
            ) from exc
        dataset_factory = LeRobotDataset.create

    dataset = dataset_factory(
        repo_id=repo_id,
        root=output_root,
        robot_type="so101_sim",
        fps=metadata["fps"],
        features=lerobot_features(height, width, use_videos=use_videos),
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
    )
    total_frames = 0
    try:
        for record in records:
            with np.load(input_root / record["path"], allow_pickle=False) as episode:
                frame_count, _, _, task = _validate_episode(
                    episode, record, fps=metadata["fps"]
                )
                for index in range(frame_count):
                    dataset.add_frame(
                        {
                            "image": episode["image"][index],
                            "wrist_image": episode["wrist_image"][index],
                            "state": episode["state"][index],
                            "actions": episode["actions"][index],
                            "task": task,
                            # Staging timestamps were validated above. LeRobot
                            # generates its version-specific timestamp value
                            # from frame_index / fps in add_frame().
                        }
                    )
                dataset.save_episode()
                total_frames += frame_count
                print(
                    f"converted episode={record['episode_index']} frames={frame_count} task={task!r}",
                    flush=True,
                )
        if push_to_hub:
            dataset.push_to_hub(
                tags=["so101", "mujoco", "openpi", "pi05"],
                private=private,
                push_videos=use_videos,
            )
    finally:
        stop_writer = getattr(dataset, "stop_image_writer", None)
        if callable(stop_writer):
            stop_writer()
    return len(records), total_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="LeRobot/Hugging Face dataset ID")
    parser.add_argument(
        "--images",
        action="store_true",
        help="store individual images instead of MP4 videos (larger; useful for debugging)",
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--public",
        action="store_true",
        help="make a pushed dataset public (default is private)",
    )
    args = parser.parse_args()
    if args.image_writer_threads < 0:
        parser.error("--image-writer-threads must be non-negative")

    episodes, frames = convert_dataset(
        args.input_root,
        args.output_root,
        args.repo_id,
        use_videos=not args.images,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
        private=not args.public,
    )
    print(f"conversion complete: episodes={episodes} frames={frames} output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

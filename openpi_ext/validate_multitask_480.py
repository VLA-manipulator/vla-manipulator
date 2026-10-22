"""Fail closed unless the two staging runs contain the requested 480 demos."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from convert_so101_staging_to_lerobot import load_manifest, load_staging_metadata


LEGACY = (
    "rod_red", "rod_yellow", "sphere_blue", "sphere_green",
    "cube_red", "cube_blue", "dumbbell_purple", "dumbbell_orange",
)
SIMPLE = ("o1_open_baseline", "p1_place_disc")
HARD = (
    "o2_thin_bar", "o3_wrist_occlusion", "o4_narrow_slot",
    "o5_covered_stack", "p2_language_ref", "p3_in_container", "p4_stack",
)
P2_LABELS = {
    "put the red cube on the blue disc": 20,
    "put the yellow cube on the blue disc": 20,
}


def validate(legacy_root: Path, new_roots: list[Path]) -> None:
    if not new_roots:
        raise ValueError("at least one new-task root is required")
    runs = [(root, load_manifest(root)) for root in (legacy_root, *new_roots)]
    if any(load_staging_metadata(root) != load_staging_metadata(legacy_root)
           for root in new_roots):
        raise ValueError("staging metadata differs")
    for root, records in runs:
        if not records:
            raise ValueError(f"{root}: no episodes")
        if root == legacy_root and Counter(r.get("task_id") for r in records) != {
            task: 20 for task in LEGACY
        }:
            raise ValueError(f"{root}: legacy quotas differ")
        if root != legacy_root and any(r.get("task_id") in LEGACY for r in records):
            raise ValueError(f"{root}: unexpected legacy task")
        for record in records:
            if record.get("success") is not True or record.get("saved") is not True:
                raise ValueError(f"{root}: non-successful episode {record['episode_index']}")
            if record.get("instruction_kind") != "train" or not record.get("instruction"):
                raise ValueError(f"{root}: invalid instruction {record['episode_index']}")
            if not isinstance(record.get("frames"), int) or record["frames"] <= 0:
                raise ValueError(f"{root}: empty episode {record['episode_index']}")
            if not (root / record["path"]).is_file():
                raise FileNotFoundError(root / record["path"])
    new_records = [record for _, records in runs[1:] for record in records]
    expected = {**{task: 20 for task in SIMPLE}, **{task: 40 for task in HARD}}
    counts = Counter(record.get("task_id") for record in new_records)
    if counts != expected:
        raise ValueError(f"new-task quotas {counts} != {expected}")
    labels: dict[str, set[str]] = defaultdict(set)
    for record in new_records:
        labels[record["task_id"]].add(record["instruction"])
    p2 = Counter(record["instruction"] for record in new_records
                 if record["task_id"] == "p2_language_ref")
    if p2 != P2_LABELS:
        raise ValueError(f"P2 red/yellow balance differs: {p2}")
    for task in (*SIMPLE, *HARD):
        if task != "p2_language_ref" and len(labels[task]) != 1:
            raise ValueError(f"{task}: expected one fixed instruction, got {labels[task]}")
    print("VERIFIED 480 successful episodes: 160 legacy + 320 fixed-label new")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    validate(args.legacy_root, args.new_root)


if __name__ == "__main__":
    main()

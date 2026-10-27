"""Combine completed SO101 staging runs without duplicating episode payloads.

The output is a new staging dataset. Source datasets remain untouched; episode
files are hard-linked on the same filesystem so conversion needs no extra copy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from convert_so101_staging_to_lerobot import load_manifest, load_staging_metadata


def merge(sources: list[Path], output: Path) -> int:
    if len(sources) < 2:
        raise ValueError("at least two source datasets are required")
    if output.exists():
        raise FileExistsError(output)
    metadata = [load_staging_metadata(source) for source in sources]
    if any(item != metadata[0] for item in metadata[1:]):
        raise ValueError("staging metadata differs between sources")
    manifests = [load_manifest(source) for source in sources]
    entries: list[tuple[Path, dict]] = []
    for source, records in zip(sources, manifests, strict=True):
        for record in records:
            if not record.get("success") or not record.get("saved"):
                raise ValueError(f"unsuccessful episode in {source}: {record['episode_index']}")
            episode = source / record["path"]
            if not episode.is_file():
                raise FileNotFoundError(episode)
            entries.append((episode, record))

    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    (output / "metadata.json").write_text(
        json.dumps(metadata[0], indent=2) + "\n", encoding="utf-8"
    )
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for index, (episode, record) in enumerate(entries):
            relative = f"episodes/episode_{index:06d}.npz"
            os.link(episode, output / relative)
            merged_record = {**record, "episode_index": index, "path": relative}
            handle.write(json.dumps(merged_record, ensure_ascii=False) + "\n")
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"merged {merge(args.source, args.output)} successful episodes into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Back up a checkpoint to a private Hugging Face model repo and verify it."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from huggingface_hub import HfApi


def retry(label: str, operation):
    for attempt in range(8):
        try:
            return operation()
        except Exception as exc:
            if attempt == 7:
                raise
            delay = min(15 * (attempt + 1), 90)
            print(f"{label} attempt {attempt + 1} failed: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    root = args.checkpoint.resolve(strict=True)
    local = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}
    if not local or "_CHECKPOINT_METADATA" not in local:
        raise ValueError("checkpoint is empty or missing metadata")

    api = HfApi()
    retry("create_repo", lambda: api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True))
    info = retry("repo_info", lambda: api.repo_info(args.repo_id, repo_type="model"))
    if not info.private:
        raise RuntimeError("refusing to upload checkpoint to a public repo")
    print(f"UPLOAD_BEGIN {args.repo_id} files={len(local)}", flush=True)
    retry("upload_large_folder", lambda: api.upload_large_folder(
        repo_id=args.repo_id, repo_type="model", folder_path=root
    ))

    remote = retry("list_repo_tree", lambda: {
        item.path: item
        for item in api.list_repo_tree(args.repo_id, repo_type="model", recursive=True, expand=True)
        if hasattr(item, "size")
    })
    missing = set(local) - set(remote)
    if missing:
        raise RuntimeError(f"remote checkpoint is missing {len(missing)} files: {sorted(missing)[:5]}")
    for name, path in local.items():
        item = remote[name]
        if item.size != path.stat().st_size:
            raise RuntimeError(f"remote file size mismatch: {name}")
        lfs = getattr(item, "lfs", None)
        if lfs and lfs.get("sha256") != sha256(path):
            raise RuntimeError(f"remote LFS checksum mismatch: {name}")
    print(f"BACKUP_VERIFIED {args.repo_id} files={len(local)}", flush=True)


if __name__ == "__main__":
    main()

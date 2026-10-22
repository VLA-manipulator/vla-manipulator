"""Install a separate 30k SO101 multitask config without changing the old run."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 TRAIN CONFIG"
OLD_END = "    # END VLA-MANIPULATOR SO101 TRAIN CONFIG"
NEW_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 MULTITASK TRAIN CONFIG"
NEW_END = "    # END VLA-MANIPULATOR SO101 MULTITASK TRAIN CONFIG"
OLD_NAME = 'name="pi05_so101_lora"'
NEW_NAME = 'name="pi05_so101_lora_multitask"'
OLD_REPO = 'repo_id="HF2147552431/so101-mujoco-pick"'
NEW_REPO = 'repo_id="HF2147552431/so101-mujoco-multitask-v1"'


def render_config(source: str) -> str:
    if NEW_BEGIN in source or NEW_END in source:
        if source.count(NEW_BEGIN) == source.count(NEW_END) == 1:
            block = source.split(NEW_BEGIN, 1)[1].split(NEW_END, 1)[0]
            if NEW_NAME in block and NEW_REPO in block and block.count("discrete_state_input=False") == 2:
                return source
        raise RuntimeError("existing multitask config is incomplete or different")
    if source.count(OLD_BEGIN) != 1 or source.count(OLD_END) != 1:
        raise RuntimeError("expected exactly one original SO101 train config")
    block = source.split(OLD_BEGIN, 1)[1].split(OLD_END, 1)[0]
    if block.count(OLD_NAME) != 1 or block.count(OLD_REPO) != 1:
        raise RuntimeError("original SO101 name or dataset differs from expected")
    if block.count("discrete_state_input=False") != 2:
        raise RuntimeError("original SO101 config is not state-token-free")
    if '"gs://openpi-assets/checkpoints/pi05_base/params"' not in block:
        raise RuntimeError("original SO101 config does not load pi05_base")
    if "save_interval=5_000" not in block or "num_train_steps=30_000" not in block:
        raise RuntimeError("unexpected original checkpoint schedule")
    addition = NEW_BEGIN + block.replace(OLD_NAME, NEW_NAME).replace(OLD_REPO, NEW_REPO) + NEW_END
    return source.replace(OLD_END, OLD_END + "\n" + addition, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = args.openpi_root / "src/openpi/training/config.py"
    original = path.read_text(encoding="utf-8")
    updated = render_config(original)
    if args.check:
        print("multitask config validated")
    elif updated != original:
        path.write_text(updated, encoding="utf-8")
        print("installed pi05_so101_lora_multitask")
    else:
        print("pi05_so101_lora_multitask already installed")


if __name__ == "__main__":
    main()

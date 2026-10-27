"""Install an isolated state-token-free SO101 eight-pick training config."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 TRAIN CONFIG"
OLD_END = "    # END VLA-MANIPULATOR SO101 TRAIN CONFIG"
NEW_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 PICK8 TRAIN CONFIG"
NEW_END = "    # END VLA-MANIPULATOR SO101 PICK8 TRAIN CONFIG"


def render_config(source: str) -> str:
    if NEW_BEGIN in source or NEW_END in source:
        if source.count(NEW_BEGIN) == source.count(NEW_END) == 1:
            block = source.split(NEW_BEGIN, 1)[1].split(NEW_END, 1)[0]
            if ('name="pi05_so101_lora_pick8"' in block
                    and 'repo_id="HF2147552431/so101-mujoco-pick8-v2"' in block
                    and block.count("discrete_state_input=False") == 2):
                return source
        raise RuntimeError("existing pick8 config is incomplete or different")
    if source.count(OLD_BEGIN) != 1 or source.count(OLD_END) != 1:
        raise RuntimeError("expected one original SO101 training config")
    old = source.split(OLD_BEGIN, 1)[1].split(OLD_END, 1)[0]
    for anchor in (
        'name="pi05_so101_lora"',
        'repo_id="HF2147552431/so101-mujoco-pick"',
        '"gs://openpi-assets/checkpoints/pi05_base/params"',
        "save_interval=5_000",
    ):
        if old.count(anchor) != 1:
            raise RuntimeError(f"unexpected original SO101 config: {anchor}")
    if old.count("discrete_state_input=False") != 2:
        raise RuntimeError("original config is not state-token-free")
    new = old.replace('name="pi05_so101_lora"', 'name="pi05_so101_lora_pick8"')
    new = new.replace('repo_id="HF2147552431/so101-mujoco-pick"',
                      'repo_id="HF2147552431/so101-mujoco-pick8-v2"')
    new = new.replace("num_train_steps=30_000", "num_train_steps=15_000")
    if new == old or "num_train_steps=15_000" not in new:
        raise RuntimeError("could not construct pick8 training config")
    return source.replace(OLD_END, OLD_END + "\n" + NEW_BEGIN + new + NEW_END, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = args.openpi_root / "src/openpi/training/config.py"
    original = path.read_text(encoding="utf-8")
    updated = render_config(original)
    if args.check:
        print("pick8 config validated")
    elif updated != original:
        path.write_text(updated, encoding="utf-8")
        print("installed pi05_so101_lora_pick8")
    else:
        print("pi05_so101_lora_pick8 already installed")


if __name__ == "__main__":
    main()

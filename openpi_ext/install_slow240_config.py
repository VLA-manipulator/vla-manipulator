"""Install an isolated pi0.5 LoRA config for 240 slower pick/O1 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path


SOURCE_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 O1-40 TRAIN CONFIG"
SOURCE_END = "    # END VLA-MANIPULATOR SO101 O1-40 TRAIN CONFIG"
NEW_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 SLOW240 TRAIN CONFIG"
NEW_END = "    # END VLA-MANIPULATOR SO101 SLOW240 TRAIN CONFIG"
NAME = "pi05_so101_lora_pick8_o1_slow240_5k"
REPO = "HF2147552431/so101-pick8-o1-slow240-v1"


def render_config(source: str) -> str:
    if NEW_BEGIN in source or NEW_END in source:
        if source.count(NEW_BEGIN) == source.count(NEW_END) == 1:
            block = source.split(NEW_BEGIN, 1)[1].split(NEW_END, 1)[0]
            if f'name="{NAME}"' in block and f'repo_id="{REPO}"' in block:
                return source
        raise RuntimeError("existing slow240 config is incomplete or different")
    if source.count(SOURCE_BEGIN) != 1 or source.count(SOURCE_END) != 1:
        raise RuntimeError("expected one tested O1-40 config")
    old = source.split(SOURCE_BEGIN, 1)[1].split(SOURCE_END, 1)[0]
    for anchor in (
        'name="pi05_so101_lora_o1_40"',
        'repo_id="HF2147552431/so101-mujoco-o1-40-unique-v1"',
        '"gs://openpi-assets/checkpoints/pi05_base/params"',
        "num_train_steps=6000",
        "save_interval=6000",
        "discrete_state_input=False",
    ):
        if old.count(anchor) != (2 if anchor == "discrete_state_input=False" else 1):
            raise RuntimeError(f"unexpected O1 training config: {anchor}")
    new = old.replace('name="pi05_so101_lora_o1_40"', f'name="{NAME}"')
    new = new.replace(
        'repo_id="HF2147552431/so101-mujoco-o1-40-unique-v1"', f'repo_id="{REPO}"'
    )
    new = new.replace("num_train_steps=6000", "num_train_steps=5000")
    new = new.replace("save_interval=6000", "save_interval=5000")
    new = new.replace("decay_steps=6000", "decay_steps=5000")
    new = new.replace('checkpoint_base_dir="/root/o1_checkpoints"',
                      'checkpoint_base_dir="/root/autodl-tmp/checkpoints"')
    return source.replace(SOURCE_END, SOURCE_END + "\n" + NEW_BEGIN + new + NEW_END, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = args.openpi_root / "src/openpi/training/config.py"
    original = path.read_text(encoding="utf-8")
    updated = render_config(original)
    if args.check:
        print("slow240 config validated")
    elif updated != original:
        path.write_text(updated, encoding="utf-8")
        print(f"installed {NAME}")
    else:
        print(f"{NAME} already installed")


if __name__ == "__main__":
    main()

"""Install the SO101 adapter and training config into an official OpenPI checkout.

The edit is idempotent and guarded by exact official-source anchors.  It exits
without modifying ``config.py`` when those anchors are absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


IMPORT_ANCHOR = "import openpi.policies.libero_policy as libero_policy\n"
IMPORT_LINE = "import openpi.policies.so101_policy as so101_policy\n"
DATA_ANCHOR = "\n\n@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig(DataConfigFactory):"
CONFIG_ANCHOR = "_CONFIGS = [\n"
DATA_BEGIN = "# BEGIN VLA-MANIPULATOR SO101 DATA CONFIG"
DATA_END = "# END VLA-MANIPULATOR SO101 DATA CONFIG"
TRAIN_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 TRAIN CONFIG"
TRAIN_END = "    # END VLA-MANIPULATOR SO101 TRAIN CONFIG"


DATA_BLOCK = r'''

# BEGIN VLA-MANIPULATOR SO101 DATA CONFIG
@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """SO101 absolute motor targets: five delta joints plus absolute gripper."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[so101_policy.SO101Inputs(model_type=model_config.model_type)],
            outputs=[so101_policy.SO101Outputs()],
        )
        # The simulator records absolute joint targets.  Pi models learn joint
        # deltas while the gripper remains an absolute target.
        delta_action_mask = _transforms.make_bool_mask(5, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )
# END VLA-MANIPULATOR SO101 DATA CONFIG
'''


TRAIN_BLOCK_TEMPLATE = r'''    # BEGIN VLA-MANIPULATOR SO101 TRAIN CONFIG
    # Combines the official pi05_libero architecture with the official
    # pi0_libero_low_mem_finetune LoRA variants and freeze filter.
    TrainConfig(
        name="pi05_so101_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotSO101DataConfig(
            repo_id="__SO101_REPO_ID__",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=None,
        assets_base_dir="/root/autodl-tmp/assets",
        checkpoint_base_dir="/root/autodl-tmp/checkpoints",
    ),
    # END VLA-MANIPULATOR SO101 TRAIN CONFIG
'''


def render_config(original: str, *, repo_id: str = "VLA-manipulator/so101-mujoco-pick") -> str:
    if repo_id.count("/") != 1 or any(part.strip() != part or not part for part in repo_id.split("/")):
        raise ValueError("repo_id must have the form namespace/dataset")
    if DATA_BEGIN in original or TRAIN_BEGIN in original:
        if DATA_BEGIN in original and DATA_END in original and TRAIN_BEGIN in original and TRAIN_END in original:
            installed_train_block = original.split(TRAIN_BEGIN, 1)[1].split(TRAIN_END, 1)[0]
            if f'repo_id="{repo_id}"' not in installed_train_block:
                raise RuntimeError(
                    "SO101 is already installed with a different dataset repo_id; "
                    "restore config.py with git and reinstall"
                )
            return original
        raise RuntimeError("config.py contains a partial SO101 installation; restore it with git before retrying")
    for anchor in (IMPORT_ANCHOR, DATA_ANCHOR, CONFIG_ANCHOR):
        if anchor not in original:
            raise RuntimeError(f"unsupported OpenPI config.py: missing anchor {anchor!r}")
    updated = original.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
    updated = updated.replace(DATA_ANCHOR, DATA_BLOCK + DATA_ANCHOR, 1)
    train_block = TRAIN_BLOCK_TEMPLATE.replace("__SO101_REPO_ID__", repo_id)
    updated = updated.replace(CONFIG_ANCHOR, CONFIG_ANCHOR + train_block, 1)
    return updated


def install(
    openpi_root: Path,
    *,
    check_only: bool = False,
    repo_id: str = "VLA-manipulator/so101-mujoco-pick",
) -> bool:
    root = openpi_root.resolve()
    config_path = root / "src/openpi/training/config.py"
    policy_dir = root / "src/openpi/policies"
    if not config_path.is_file() or not policy_dir.is_dir():
        raise FileNotFoundError(f"not an OpenPI checkout: {root}")
    original = config_path.read_text(encoding="utf-8")
    updated = render_config(original, repo_id=repo_id)
    changed = updated != original
    if check_only:
        return changed

    source_policy = Path(__file__).with_name("so101_policy.py")
    shutil.copy2(source_policy, policy_dir / "so101_policy.py")
    if changed:
        config_path.write_text(updated, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate compatibility without writing")
    parser.add_argument("--repo-id", default="VLA-manipulator/so101-mujoco-pick")
    args = parser.parse_args()
    changed = install(args.openpi_root, check_only=args.check, repo_id=args.repo_id)
    print("compatible; installation required" if args.check and changed else "SO101 extension ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# OpenPI extensions

This directory contains project-owned integration code for the official
Physical Intelligence OpenPI repository. OpenPI itself is not vendored here.

Implemented:

- `convert_so101_staging_to_lerobot.py`: validates every staging episode before
  creating output and converts it with OpenPI's pinned LeRobot API.
- `so101_policy.py`: official-style two-camera, six-axis input/output adapter.
- `install_into_openpi.py`: guarded, idempotent installer for
  `LeRobotSO101DataConfig` and `pi05_so101_lora`.

The LoRA config combines OpenPI's official `pi05_libero` architecture with the
official `pi0_libero_low_mem_finetune` LoRA variants and freeze filter. OpenPI
does not currently publish this exact PI0.5 + SO101 combination, so run the
10-step smoke workflow before any long training run.

Install into an OpenPI checkout:

```bash
bash scripts/prepare_openpi_so101.sh
```

Run these tools from the separate OpenPI virtual environment so that the
LeRobot version exactly matches OpenPI's lock file.

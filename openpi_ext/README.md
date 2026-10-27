# OpenPI extensions

This directory contains project-owned integration code for the official
Physical Intelligence OpenPI repository. OpenPI itself is not vendored here.

Implemented:

- `convert_so101_staging_to_lerobot.py`: validates every staging episode before
  creating output and converts it with OpenPI's pinned LeRobot API.
- `so101_policy.py`: official-style two-camera, six-axis input/output adapter.
- `install_into_openpi.py`: guarded, idempotent installer for
  `LeRobotSO101DataConfig` and `pi05_so101_lora`.
- `install_state_conditioned_config.py`: adds a separate
  `pi05_so101_lora_state` config (`discrete_state_input=True`) without changing
  the original state-free config. This was used for the 2026-10-18 25k
  comparison; it reuses the same dataset and normalization assets.
- `check_state_experiment.py`: validates that the two configs have separate
  checkpoint directories and the expected state-input setting.
- `check_so101_roundtrip.py`: reads a real LeRobot frame and checks the
  six-dimensional field mapping, delta/absolute action conversion, and
  checkpoint normalization roundtrip. It does not evaluate model quality.
- `serve_rtc.py` / `rtc_sampler.py`: opt-in real-time chunking inference with
  the paper's soft overlap guidance. Setup and limitations: [RTC_SO101.md](RTC_SO101.md).

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

The completed state-input comparison and its measured outcome are documented
in `docs/so101_state_input_experiment_2026-10-18.md`. The old 25k action-chain
test and failure diagnosis are in `docs/so101_old25k_diagnosis_2026-10-19.md`.

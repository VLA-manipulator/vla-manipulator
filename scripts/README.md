# Operational scripts

Run these on the AutoDL Linux server from the repository root.

```bash
# Set this to your real Hugging Face namespace. Use the same value every time.
export HF_DATASET_REPO=your-hf-username/so101-mujoco-pick

# 1. Install the guarded SO101 extension into the official OpenPI checkout.
bash scripts/prepare_openpi_so101.sh

# 2. Convert five local smoke episodes to LeRobot format.
bash scripts/convert_dataset.sh

# Optional: authenticate with `hf auth login`, then convert and upload privately.
bash scripts/convert_dataset.sh --push-to-hub

# 3. Compute q01/q99 normalization statistics using the official script.
bash scripts/compute_norm_stats.sh

# 4. Run a 10-step smoke train (default).
bash scripts/train_pi05_so101_lora.sh

# Longer run only after smoke validation.
EXP_NAME=so101_v1 TRAIN_STEPS=30000 bash scripts/train_pi05_so101_lora.sh

# 5. Serve a selected checkpoint.
bash scripts/serve_pi05_so101.sh \
  /root/autodl-tmp/checkpoints/pi05_so101_lora/so101_smoke/9
```

`check_server_storage.sh` refuses training below 20 GiB free space by default.
`archive_checkpoint.sh` creates a `.tar.zst` and uploads it to a Hugging Face
model repository; it never deletes the local checkpoint automatically.

Large datasets, model weights, logs and checkpoints are intentionally excluded
from Git.

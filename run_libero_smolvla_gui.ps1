Write-Host "Starting LIBERO SmolVLA GUI..."
$wslCommand = "export DISPLAY=:0; export WAYLAND_DISPLAY=wayland-0; export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir; export LIBERO_CONFIG_PATH=/mnt/d/wsl/libero/config; export PATH=/mnt/d/wsl/conda_envs/libero/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; export PYTHONPATH=/mnt/d/wsl/src/lerobot/src; python -u /mnt/c/Users/21475/Desktop/lerobot/libero_smolvla_probe.py --gui --steps 5"
wsl.exe -d libero -- bash -lc $wslCommand
Write-Host "LIBERO process exited with code $LASTEXITCODE"

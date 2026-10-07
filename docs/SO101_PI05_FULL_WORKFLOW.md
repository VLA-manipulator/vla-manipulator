# SO101 MuJoCo → OpenPI π0.5 LoRA 全流程（AutoDL）

目标：在同一桌面场景中，根据英文指令抓取 8 种物体之一。每种物体采集 40 条**成功**示范，共 320 条；先训练本地模型并做留出种子闭环评估，再决定是否上传 Hugging Face。

适用环境：AutoDL Ubuntu 22.04、单卡 vGPU 48 GB、OpenPI 官方 JAX 环境，以及本仓库的独立 MuJoCo 环境。以下命令均在**服务器 Bash** 中执行。两套 `.venv` 分开，数据都放在 `/root/autodl-tmp`。已有的 5 条冒烟数据和 `so101_smoke/9` checkpoint 保留。

## 0. 先确认服务器状态

```bash
cd /root/autodl-tmp/workspace/vla-manipulator
git pull --ff-only
git log -1 --oneline
nvidia-smi
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/openpi-data /root/autodl-tmp/checkpoints /root/autodl-tmp/datasets 2>/dev/null
```

此前服务器访问 GitHub 偶尔很慢；需要时先 `source /etc/network_turbo`。若拉取中断，先不要执行下面的新 `--successes-per-object` 参数，确认服务器已获得包含该参数的代码。

空间不足时先在 AutoDL 扩充数据盘。不要降低 `scripts/check_server_storage.sh` 的 20 GiB 门槛来绕过 checkpoint 空间检查。若服务器归一化时报 TorchCodec 找不到 `libavutil`，安装系统 FFmpeg：`apt-get update && apt-get install -y ffmpeg && ldconfig`。

## 1. 设置本次实验的路径

```bash
export HF_DATASET_REPO=HF2147552431/so101-mujoco-pick
export HF_HOME=/root/autodl-tmp/huggingface
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-data
export OPENPI_ROOT=/root/autodl-tmp/workspace/openpi
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache

# 与 5 条冒烟数据的 LeRobot 根目录分开；训练脚本也会读取此变量。
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets/lerobot_v1
export INPUT_ROOT=/root/autodl-tmp/datasets/so101_staging/all_objects_v1_40
export OUTPUT_ROOT="$HF_LEROBOT_HOME/$HF_DATASET_REPO"
export EXP_NAME=so101_all_v1
```

这些 `export` 只在当前 SSH 会话生效。断线重连后重新执行本节，或把它们放进自己管理的环境脚本。`HF_DATASET_REPO` 是 LeRobot 本地数据集的标识符；下面没有 `--push-to-hub`，因此不会上传。`HF_LEROBOT_HOME` 一定要在**转换、归一化、训练**三个步骤保持一致。现有 OpenPI 配置中 `repo_id` 应为 `HF2147552431/so101-mujoco-pick`；检查：

```bash
rg -n 'pi05_so101_lora|HF2147552431/so101-mujoco-pick' "$OPENPI_ROOT/src/openpi/training/config.py"
```

如果 `rg` 未安装，用 `grep -nE`。若显示其他数据集 ID，先运行 `bash scripts/prepare_openpi_so101.sh`；如果提示已经安装了不同 ID，不要覆盖 OpenPI 配置，应先检查具体差异。

## 2. 采集每种物体 40 条成功示范

先确认新的 staging 目录不存在：

```bash
ls -ld "$INPUT_ROOT" 2>/dev/null || echo "采集目录尚不存在，可以开始"
mkdir -p /root/autodl-tmp/datasets/so101_staging
```

开始无界面采集。`--max-attempts-per-object 80` 表示每种物体最多尝试 80 次；达到 40 次成功就切换到下一种。若某种物体不足 40 次，脚本会报告缺口并以状态码 2 退出；已保存的数据仍在目录里，不会丢失。长时间采集建议在 `tmux` 中运行；未安装可先 `apt-get install -y tmux`。

```bash
cd /root/autodl-tmp/workspace/vla-manipulator/simulation
UV_NO_SYNC=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run python -m mj_env.scripts.collect_demonstrations \
  --all \
  --successes-per-object 40 \
  --max-attempts-per-object 80 \
  --seed 10000 \
  --root "$INPUT_ROOT" \
  --height 224 --width 224 --headless
```

8 种目标为红/黄杆、蓝/绿球、红/蓝方块、紫/橙哑铃。环境默认会把这 8 个物体放在同一场景中；专家使用目标 ID 规划动作，但存给模型的是两路 RGB 图、6 维机械臂状态、对应的任务文本和动作。语言仍以文本保存，OpenPI 在训练时做 token 化。`--seed 10000` 开始的训练种子与下面 `1000` 开始的评估种子不重叠。

采集完成后，先核验实际成功数。脚本汇总应为 `saved 320/...`。再查看每类目标的 manifest 数量：

```bash
INPUT_ROOT="$INPUT_ROOT" python -c 'import json,os,collections,pathlib; p=pathlib.Path(os.environ["INPUT_ROOT"]); rows=[json.loads(s) for s in (p/"manifest.jsonl").read_text().splitlines()]; counts=collections.Counter(r["target"] for r in rows); print("total:",len(rows)); print("by target:",dict(sorted(counts.items()))); assert len(rows)==320 and len(counts)==8 and all(v==40 for v in counts.values())'
du -sh "$INPUT_ROOT"
df -h /root/autodl-tmp
```

如果数量不足，**不要重新运行到同一目录，也不要直接删除已采集的数据**。记录各类成功数和失败结果，再补采并合并数据；当前采集器故意拒绝覆盖已有目录。

## 3. 转换为本地 LeRobot 数据

原始示范是 `.npz` staging 格式，转换时把 RGB 帧编码为 LeRobot 视频，保留任务文本、状态、动作以及 20 Hz 对齐的时间戳。转换脚本在 OpenPI 的 `.venv` 中运行。

```bash
cd /root/autodl-tmp/workspace/vla-manipulator
ls -ld "$OUTPUT_ROOT" 2>/dev/null || echo "LeRobot 目录尚不存在，可以开始"
UV_NO_SYNC=1 bash scripts/convert_dataset.sh
du -sh "$OUTPUT_ROOT"
```

预期输出 `conversion complete: episodes=320 frames=41600`（当前专家策略成功轨迹每条约 130 帧；以实际输出为准）。这里**不加** `--push-to-hub`。若转换中断，先检查输出目录；转换器拒绝覆盖半成品，切勿删除 staging 原始数据。

## 4. 计算新数据的归一化统计量

```bash
cd /root/autodl-tmp/workspace/vla-manipulator
UV_NO_SYNC=1 bash scripts/compute_norm_stats.sh
```

必须使用新的 `HF_LEROBOT_HOME` 重新算统计量，不能沿用 5 条红方块冒烟数据的 q01/q99。OpenPI 数据流会把前 5 个关节的绝对目标改成相对**当前观测状态**的 delta，夹爪维度保持绝对值，然后按本次数据统计量归一化；推理时反向处理。

## 5. 开始 π0.5 Base 的 LoRA 微调

当前 `pi05_so101_lora` 配置以官方 `pi05_base` 参数初始化，batch size 为 8，默认计划 30,000 步，每 5,000 步定期保存，结束时也会保存最后一步。这个配置已用 5 条数据完成 10 步训练与 checkpoint 加载验证。320 条数据的最佳训练步数未知；建议先把 **5,000 步作为第一个可评估节点**，观察训练 loss、时间与磁盘，再决定是否继续。不要把 loss 下降当作抓取成功率。

```bash
cd /root/autodl-tmp/workspace/vla-manipulator
df -h /root/autodl-tmp
UV_NO_SYNC=1 WANDB_MODE=offline \
EXP_NAME=so101_all_v1 TRAIN_STEPS=5000 \
bash scripts/train_pi05_so101_lora.sh
```

不要使用 `--overwrite` 覆盖冒烟实验。该命令会在 `/root/autodl-tmp/checkpoints/pi05_so101_lora/so101_all_v1/` 写入 checkpoint；具体 step 目录以 `find` 输出为准，**不要假定一定是 4999 或 5000**。

```bash
find /root/autodl-tmp/checkpoints/pi05_so101_lora/so101_all_v1 -maxdepth 1 -type d -printf '%f\n'
du -sh /root/autodl-tmp/checkpoints/pi05_so101_lora/so101_all_v1
df -h /root/autodl-tmp
```

关注终端中有限的 loss、显存使用、是否出现 OOM/NaN，以及 Orbax `Save Finalize` 完成。W&B 离线日志在 OpenPI 的 `wandb/` 下，暂不需要联网同步。中断后先检查 checkpoint，再按 OpenPI 当前版本的恢复参数续训；**不要仅换同一个实验名重跑，避免覆盖或从头训练**。

## 6. 启动服务与逐物体闭环评估

选择上一步实际保存的数字 step 目录（下面用占位符 `<STEP>`），在第一个 SSH 窗口启动服务：

```bash
cd /root/autodl-tmp/workspace/vla-manipulator
UV_NO_SYNC=1 bash scripts/serve_pi05_so101.sh \
  /root/autodl-tmp/checkpoints/pi05_so101_lora/so101_all_v1/<STEP>
```

在第二个 SSH 窗口确认仿真 `.venv` 里已有 `openpi-client`。在留出的 seed 1000～1019 上**分别测试 8 种指令**；每个目标 20 次，共 160 次。成功率为零也是有效评估结果；连接或推理报错会停止循环：

```bash
cd /root/autodl-tmp/workspace/vla-manipulator/simulation
for target in rod_red rod_yellow sphere_blue sphere_green cube_red cube_blue dumbbell_purple dumbbell_orange; do
  echo "===== $target ====="
  UV_NO_SYNC=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run python -m mj_env.scripts.evaluate_openpi \
    --host localhost --port 8000 --target "$target" \
    --seed 1000 --episodes 20 --replan-steps 5 --max-steps 300 || break
done
```

每类记录 `success_rate=x/20`；还要观察有没有抓错目标。环境的 `success` 只在**指定目标**被夹住并抬起时为真。若 5,000 步表现差，结合每类成功率、专家示范质量和动作轨迹，再决定增加数据、调整训练步数或修正相机分布。不要仅用 1 次评估判断模型好坏。

## 7. 保存结果，之后再考虑 Hugging Face 上传

当前流程只在 AutoDL 数据盘保存数据和 checkpoint。关机前检查 AutoDL 的数据盘保留策略、剩余空间、checkpoint 完整性，并保留本地 `staging`、LeRobot 数据及训练统计量。`HF2147552431/so101-mujoco-pick` 是**Dataset** 仓库，`HF2147552431/pi05-so101-mujoco-lora` 是**Model** 仓库；现在都不要求上传。

需要备份 checkpoint 时，再登录 Hugging Face 并运行 `scripts/archive_checkpoint.sh`；该脚本会先在数据盘生成压缩包，因此必须另外预留压缩包空间。确认远端文件可下载后再考虑清理本地副本。数据若要上传，用 `--push-to-hub` 转换全新数据集或另行上传已转换目录；避免对已有输出目录重跑转换。

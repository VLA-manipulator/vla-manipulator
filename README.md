# VLA Manipulator

SO101 机械臂的仿真数据采集、OpenPI π0.5 微调和评估工程。

当前已完成：

- MuJoCo SO101 桌面环境；
- 全局相机和腕部相机观测；
- 随机物体布局与随机机械臂初始姿态；
- IK 专家抓取策略；
- 20 Hz 同步示范采集；
- 不依赖 LeRobot 的 staging 数据格式。
- staging → LeRobot 转换器；
- SO101 OpenPI 输入/输出适配器；
- 基于官方配置组合的 `pi05_so101_lora`；
- 训练、推理、磁盘检查和 checkpoint 归档脚本。

## 仓库结构

```text
vla-manipulator/
├── simulation/                 # 独立的 MuJoCo/采集 Python 项目
│   ├── mj_env/                 # 环境、专家策略和数据记录器
│   ├── so101/                  # 运动学、MJCF/URDF 和网格资源
│   ├── pyproject.toml
│   └── uv.lock
├── openpi_ext/                 # SO101→LeRobot、OpenPI policy/config（逐步补齐）
└── scripts/                    # 服务器训练和评估入口（逐步补齐）
```

仿真和 OpenPI 使用两个独立虚拟环境。原因是仿真固定
`gymnasium==1.3.0`，而 OpenPI 固定的 LeRobot 版本使用
`gymnasium==0.29.1`。缓存、模型权重和数据目录可以共享，
但不能共享同一个 `.venv`。

## 本地仿真环境

```powershell
cd simulation
uv sync
uv run python -m mj_env.scripts.scripted_pick --object cube_red --episodes 5
```

采集五条 staging 轨迹：

```powershell
uv run python -m mj_env.scripts.collect_demonstrations `
  --object cube_red `
  --episodes 5 `
  --seed 0 `
  --root .tmp/staging/cube_red_smoke_5 `
  --headless
```

每帧保存两个 `224×224 RGB` 视角、六维状态、六维动作、任务文本和
严格的 20 Hz 时间戳。只有成功 episode 会被保存。`.tmp/`、正式数据集、
模型权重和 checkpoint 均不提交到 Git。

## 服务器规划

```text
/root/autodl-tmp/
├── workspace/vla-manipulator/     # 本仓库
├── workspace/openpi/              # 官方 OpenPI 仓库及独立 .venv
├── datasets/so101_staging/        # 上传的临时轨迹
├── datasets/so101_lerobot/        # 转换后的正式数据
├── huggingface/                   # Hugging Face 共享缓存
├── openpi-data/                   # OpenPI 权重缓存
├── uv-cache/                      # uv 共享下载缓存
└── checkpoints/                   # 训练结果
```

五条数据只用于验证“采集→转换→归一化→训练→推理”链路，不足以训练出
可靠策略。流程跑通后再采集每个任务至少 100～200 条具有不同布局的成功轨迹。

## 服务器执行顺序

1. `bash scripts/prepare_openpi_so101.sh`
2. 上传五条 staging 数据；
3. `bash scripts/convert_dataset.sh`
4. `bash scripts/compute_norm_stats.sh`
5. `bash scripts/train_pi05_so101_lora.sh`（默认仅10步）；
6. 启动 policy server，并使用 `mj_env.scripts.evaluate_openpi` 闭环评估。

OpenPI 官方没有发布现成的 PI0.5 + SO101 LoRA 配置。本项目只组合官方
`pi05_libero` 的 PI0.5 设置、官方低内存 LoRA 模型变体/冻结规则以及官方
UR5/LIBERO 机器人适配方式。长训练前必须先验证10步训练、checkpoint恢复和
闭环推理。

## 测试

```powershell
cd simulation
uv run python -m unittest \
  mj_env.test_demonstrations
```

更详细的仿真说明见 [`simulation/README.md`](simulation/README.md)。

8 种物体各采集 40 条成功示范的服务器全流程，见
[`docs/SO101_PI05_FULL_WORKFLOW.md`](docs/SO101_PI05_FULL_WORKFLOW.md)。

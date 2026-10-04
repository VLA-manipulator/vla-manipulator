# VLA Manipulator

SO101 机械臂的仿真数据采集、OpenPI π0.5 微调和评估工程。

当前已完成：

- MuJoCo SO101 桌面环境；
- 全局相机和腕部相机观测；
- 随机物体布局与随机机械臂初始姿态；
- IK 专家抓取策略；
- 20 Hz 同步示范采集；
- 不依赖 LeRobot 的 staging 数据格式。

## 仓库结构

```text
vla-manipulator/
├── simulation/                 # 独立的 MuJoCo/采集 Python 项目
│   ├── mj_env/                 # 环境、专家策略和数据记录器
│   ├── so101/                  # 运动学、MJCF/URDF 和网格资源
│   ├── pyproject.toml
│   └── uv.lock
├── openpi_ext/                 # SO101→LeRobot、OpenPI policy/config（逐步补齐）
├── scripts/                    # 服务器训练和评估入口（逐步补齐）
├── libero_smolvla_probe.py     # 早期 LIBERO/SmolVLA 探索实验
└── run_libero_smolvla_gui.ps1
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

## 当前下一步

1. 在 `openpi_ext/` 添加 staging→LeRobot 转换器；
2. 添加 SO101 的 OpenPI 输入/输出映射；
3. 基于 `pi05_base` 定义 LoRA 训练配置；
4. 转换五条 smoke 数据并计算 normalization statistics；
5. 启动短训练，验证 checkpoint 和闭环推理链路。

## 测试

```powershell
cd simulation
uv run python -m unittest \
  mj_env.test_demonstrations \
  mj_env.test_closed_loop \
  mj_env.test_task_runtime
```

更详细的仿真说明见 [`simulation/README.md`](simulation/README.md)。

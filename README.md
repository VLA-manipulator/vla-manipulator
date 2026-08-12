# LIBERO SmolVLA Probe

在 [LIBERO](https://libero-project.github.io/) 基准环境中运行微调后的 [SmolVLA](https://huggingface.co/lerobot/smolvla) 视觉-语言-动作 (VLA) 模型的探针脚本。

通过 [LeRobot](https://github.com/huggingface/lerobot) 加载 `lerobot/smolvla_libero` 预训练权重，在 LIBERO 仿真环境中执行一个开环控制循环，输出预测的动作块 (action chunk)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `libero_smolvla_probe.py` | 主程序:加载模型 → 创建 LIBERO 环境 → 循环预测并执行动作 |
| `run_libero_smolvla_gui.ps1` | WSL 启动器:通过 WSLg 打开 MuJoCo 可视化窗口并运行主程序 |

## 环境要求

- Python 3.10+ (开发环境为 3.12)
- `lerobot` 源码 (来自 `huggingface/lerobot`)
- `libero` 包及其仿真依赖 (mujoco, robosuite 等)
- PyTorch + CUDA
- GUI 模式需要 WSLg (Windows 11 自带)

## 运行方式

### 无界面 (headless)

```bash
python libero_smolvla_probe.py --model-id lerobot/smolvla_libero --steps 5
```

### 带 MuJoCo 可视化窗口

```bash
python libero_smolvla_probe.py --gui --steps 5
```

### 通过 WSL PowerShell 启动器

```powershell
# 在 Windows 上直接运行
.\run_libero_smolvla_gui.ps1
```

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-id` | `lerobot/smolvla_libero` | HuggingFace 上的模型 ID |
| `--steps` | `5` | VLA 动作块数量 |
| `--num-steps` | `5` | 模型内部推理步数 |
| `--gui` | 关 | 打开 MuJoCo 可视化窗口 |

## 说明

- 模型权重首次运行时会从 HuggingFace 下载。国内网络可通过 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。
- 当前探针使用 LIBERO Spatial 任务的第 0 个任务 (task_id=0)，128×128 像素观测。

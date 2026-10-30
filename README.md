# VLA Manipulator：SO101 静态基座仿真与 π0.5 微调

本仓库包含 SO101 MuJoCo 桌面仿真、双相机观测、专家示范采集、LeRobot 数据转换、OpenPI π0.5 LoRA 微调与闭环评测。当前验证范围是**静态基座**。数据集、模型权重、缓存和视频不提交到 Git。

## 目录与运行环境

| 路径 | 内容 |
| --- | --- |
| `simulation/mj_env/` | 环境、组合场景、任务、专家、采集与评测 |
| `simulation/so101/` | 机器人 MJCF、URDF、网格 |
| `openpi_ext/` | SO101 policy/config、LeRobot 转换、备份 |
| `scripts/` | AutoDL 采集、训练、诊断入口 |
| `docs/` | 历史实验记录 |

仿真与官方 OpenPI 使用**独立虚拟环境**。仿真固定 `gymnasium==1.3.0`，OpenPI 使用其锁定的 LeRobot/Gymnasium 版本。

```bash
cd simulation
uv sync
uv run python -m mj_env.scripts.view_scene --seed 3 --target cube_red
uv run python -m mj_env.specs
```

窗口自由视角只供人工观察。策略输入是世界坐标中固定的全局相机和装在夹爪上的腕部相机。机械臂基座在世界原点；环境种子改变物体布局与朝向，重置时机械臂回到 `HOME_QPOS`。代码见 `simulation/mj_env/scene.py`、`env.py` 和 `assets/base_table.xml`。

## 构建仿真任务

1. 在 `simulation/mj_env/assets/scenes/` 创建场景 XML，包含共用的 `base_table.xml`，加入目标物体、目标区域或障碍。原八类抓取使用 `assets/scene.xml`。
2. 在 `simulation/mj_env/specs/` 注册 `TaskSpec`：任务 ID、布局范围、语言指令和可检查的成功条件。现有 P1–P4、O1–O5 可用 `python -m mj_env.specs` 列出。
3. 用 `TaskEnv(task_id)` 创建环境，以 `reset(seed=...)` 在可达区域采样布局。观测含全局/腕部 RGB、六维关节状态和指令文本。SO101 五个臂关节及夹爪的绝对位置目标按 20 Hz 更新；`LeRobotSO101Adapter` 转换为五关节 `[-100,100]`、夹爪 `[0,100]`。
4. 先运行 IK 专家，检查成功条件、物体可达性、关节步幅，再采集成功 episode。仿真成功条件只用于验收，不作为模型的控制输入。

```bash
cd simulation
uv run python -m mj_env.scripts.scripted_pick_place \
  --task p1_place_disc --episodes 5 --seed 0 --headless --require 0
```

`--require 0` 用于诊断失败。采集脚本只保存成功且满足步幅门槛的轨迹，并记录失败原因。任务与坐标细节见 [仿真说明](simulation/README.md)。

## 完整微调流程：八类抓取 + O1 慢速 240 条

该实验使用八类抓取各 20 条、O1 80 条，共 240 条**成功示范**。每帧包含两路 224×224 RGB、六维状态/动作、任务文本和 20 Hz 时间戳。以下为已采用的 AutoDL 流程；脚本内含具体服务器路径、HF 命名空间和防覆盖检查，运行前须核对。正式数据目录不可重跑覆盖。

### 1. 环境、种子规划和采集

在 AutoDL 安装独立的 `simulation/.venv` 与官方 OpenPI `.venv`；从仓库根目录运行 `bash scripts/prepare_openpi_so101.sh` 安装 SO101 集成。先运行 `mj_env.scripts.plan_diverse_expert_seeds`：在训练种子范围预演完整专家轨迹，从成功布局中选空间覆盖较广的种子，生成 `seed_plan.json`。采集 runner 随后读取计划，依次采集八类 160 条与 O1 80 条，最后运行 `openpi_ext/audit_slow240_staging.py` 生成质量报告。runner 有防覆盖检查，已存在正式输出时不要重跑。

```bash
# 在 AutoDL 的项目根目录；先生成计划，再启动采集 runner。
cd simulation
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run python -m mj_env.scripts.plan_diverse_expert_seeds \
  --seed-start 10000 --pick-candidates 50 --o1-candidates 180 \
  --pick-count 20 --o1-count 80 \
  --output /root/autodl-tmp/datasets/so101_staging/pick8_o1_slow_240_20261024/seed_plan.json
cd ..
bash scripts/run_slow240_collection_autodl.sh
```

小规模采集示例：

```bash
cd simulation
uv run python -m mj_env.scripts.collect_demonstrations \
  --object cube_red --episodes 5 --seed 10000 \
  --root .tmp/staging/cube_red_smoke_5 --headless
```

检查 manifest、轨迹与 `quality_report.json` 后，由操作者创建 `quality.approved` 标记。保留 staging 原件，以便审计和重新转换。

### 2. LeRobot 转换、归一化、LoRA

审查 `quality_report.json`、manifest 和轨迹后，由操作者写入非空的 `quality.approved`。从仓库根目录运行 `bash scripts/run_slow240_train_5k_autodl.sh`；它等待质量标记，然后顺序执行：

1. `openpi_ext/convert_so101_staging_to_lerobot.py` 合并两个 staging 输入并转换为 LeRobot；该实验 runner 还上传私有数据集，核验仓库为 private。
2. `openpi_ext/install_slow240_config.py` 检查并安装独立配置 `pi05_so101_lora_pick8_o1_slow240_5k`。核对数据集 ID、官方 `pi05_base` 初始化、LoRA 变体及当前 OpenPI checkout。
3. 在 OpenPI 环境执行 `scripts/compute_norm_stats.py --config-name pi05_so101_lora_pick8_o1_slow240_5k`。训练和推理使用同一统计量；前五关节按相对当前观测的 delta 处理，夹爪保持绝对值。
4. 检查 GPU 与磁盘，运行 `scripts/train.py` 至 5,000 步，确认最终 checkpoint 含参数、训练状态和 `norm_stats`。续训从明确的 checkpoint 恢复并保留旧版本；后续实验采用预热后恒定 `5e-5` 的学习率。训练 loss 不等同于闭环成功率。

旧版通用入口为 `scripts/convert_dataset.sh`、`scripts/compute_norm_stats.sh`、`scripts/train_pi05_so101_lora.sh`。[八类各 40 条全流程](docs/SO101_PI05_FULL_WORKFLOW.md) 是此前 320 条数据流程，其配置与慢速 240 条实验不同。

### 3. 服务、闭环评测和备份

用 `scripts/serve_pi05_so101.sh <CHECKPOINT_DIR>` 启动 policy server。八类抓取使用 `mj_env.scripts.evaluate_openpi`，O1 使用 `mj_env.scripts.evaluate_o1_openpi`。历史对照的原始留出种子为八类各 `1000–1019`、O1 `20000–20017、20019、20020`，`replan_steps=5`、`max_steps=600`。O1 跳过 20018；当前评测 CLI 只接受连续种子，因此逐个运行。服务器启动后，在仿真环境中执行：

```bash
cd simulation
for object in rod_red rod_yellow sphere_blue sphere_green cube_red cube_blue dumbbell_purple dumbbell_orange; do
  uv run python -m mj_env.scripts.evaluate_openpi \
    --target "$object" --seed 1000 --episodes 20 --replan-steps 5 --max-steps 600
done
for seed in $(seq 20000 20017) 20019 20020; do
  uv run python -m mj_env.scripts.evaluate_o1_openpi \
    --seed "$seed" --episodes 1 --replan-steps 5 --max-steps 600
done
```

保持场景、提示词、种子与评测设置一致，保存逐种子日志；模型采样随机性还需单独控制或重复试验。

检查 checkpoint 完整性及评测结果。私有备份逐文件验证前保留本地模型。`openpi_ext/backup_checkpoint_to_hf.py` 用于私有 HF 备份；`scripts/archive_checkpoint.sh` 是压缩包归档入口。

## 测试

```bash
cd simulation
uv run python -m unittest mj_env.test_gripper mj_env.test_criteria mj_env.test_demonstrations mj_env.test_collect_multitask
```

静态基座结果只代表当前仿真场景。真机仍需校准关节顺序、方向、零点、范围和相机相对基座的位置。

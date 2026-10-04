# SO101 MuJoCo simulation

用于 OpenPI π0.5 数据采集和闭环评估的 SO101 桌面抓取仿真。

该子项目只包含仿真、专家策略、示范采集和坐标适配，不调用任何在线语言模型。

## 安装

```powershell
uv sync
```

需要 Python 3.12。仿真固定使用 `gymnasium==1.3.0`，因此与 OpenPI 的
LeRobot 环境分开安装。

## 查看环境

```powershell
uv run python -m mj_env.scripts.view_scene --seed 3 --target cube_red
```

窗口中的自由视角只用于人工检查；策略输入是全局相机和腕部相机两路图像。

## 专家策略

```powershell
uv run python -m mj_env.scripts.scripted_pick --object cube_red --episodes 5
```

每次 reset 会依据 seed 随机采样物体 XY 位置和绕 Z 轴朝向。专家策略默认还会
轻微随机化机械臂初始关节姿态，并根据新物体位置重新求解 IK。

## 采集 staging 数据

```powershell
uv run python -m mj_env.scripts.collect_demonstrations `
  --object cube_red `
  --episodes 5 `
  --seed 0 `
  --root .tmp/staging/cube_red_smoke_5 `
  --headless
```

每帧包含：

- `image`：全局 RGB 图像；
- `wrist_image`：腕部 RGB 图像；
- `state`：六维 SO101 归一化电机坐标；
- `actions`：六维 SO101 归一化电机坐标；
- `timestamp`：基于 MuJoCo 仿真时间的 20 Hz 时间戳；
- `task`：自然语言任务。

只有成功 episode 会保存。输出目录被 `.gitignore` 忽略，使用仓库根目录
`openpi_ext/convert_so101_staging_to_lerobot.py` 在服务器 OpenPI 环境中转换。

## 坐标与控制

环境原生状态和动作均为六个关节的绝对位置目标：五个机械臂关节加夹爪，
单位为 MuJoCo 弧度，控制频率为 20 Hz。`LeRobotSO101Adapter` 将它们映射为
五关节 `[-100, 100]`、夹爪 `[0, 100]` 的统一电机坐标。

真机部署前必须使用真实 SO101 校准文件核对关节顺序、方向、零点和范围。

## 主要文件

| 路径 | 作用 |
| --- | --- |
| `mj_env/env.py` | Gymnasium/MuJoCo 环境与双相机观测 |
| `mj_env/layout.py` | 随机物体布局 |
| `mj_env/adapters.py` | MuJoCo 弧度与 SO101 电机坐标转换 |
| `mj_env/demonstrations.py` | 同步帧记录和 staging 存储 |
| `mj_env/scripts/scripted_pick.py` | IK 专家策略 |
| `mj_env/scripts/collect_demonstrations.py` | 示范采集入口 |
| `mj_env/scripts/view_scene.py` | 本地可视化入口 |
| `so101/` | 运动学、MJCF/URDF 和网格资源 |

## 测试

```powershell
uv run python -m unittest mj_env.test_demonstrations
```

无显示器的 Linux 服务器如需离屏渲染，可设置 `MUJOCO_GL=egl`。

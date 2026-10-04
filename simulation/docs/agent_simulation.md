# Codex / Claude Code 控制 MuJoCo 仿真

本接口让编码 agent 通过本地 HTTP 或 Python CLI 操作常驻的八物体 SO101
仿真。无需 VLA 权重、API Key 或 MCP 插件。agent 自己查看图片、决定动作、
根据结果继续调整。此服务仅控制 MuJoCo。

## 启动

在项目根目录的 PowerShell 中运行并保持进程开启：

```powershell
uv run python -u -m mj_env.agent_server --seed 3 --target cube_red
```

看到 `Simulation tools ready` 后，窗口与 `127.0.0.1:8765` 接口就绪。
`--headless` 可取消窗口；`--port` 可更换端口。同一端口只能运行一个服务。
关闭窗口或调用 `shutdown` 结束服务。

最小化窗口不会停止物理或双相机离屏渲染；只跳过桌面窗口绘制。
agent 仍可通过 `observe` 获取持续更新的相机图像和对应关节状态。

**物理按实际时间连续运行，模型思考和等待请求期间也不会暂停。** 机械臂保持
最后的执行器目标；物体会因重力、接触或夹持不稳滑落。控制频率为 20 Hz，
每个控制步包含 25 个 500 Hz 物理子步。`wait` 是等待后返回状态，并非启动物理。
控制命令异步执行，提交后返回 `action_id` 和 `status: running`。
执行期间可并发调用 `observe` / `state`，通过返回的 `action` 检查进度。
已有动作执行期间会拒绝新运动；先 `stop` 取消当前轨迹，再发送新动作。
`stop` 保持最近执行的关节目标，不会关闭物理或模拟瞬间制动。

物理线程发布状态副本，相机线程独立渲染双相机，HTTP 线程编码和传输图片；
慢客户端不会持有物理状态锁。相机处理不过来时丢弃待渲染旧副本。
IK 在独立请求线程计算，也不会占住物理循环。
图像和关节值来自同一快照；请检查 `frame_id`、`captured_at_unix_s`、
`simulation_time_s`、`snapshot_age_s` 判断新鲜度。
帧龄是服务生成响应时的值，不包含后续网络或模型读图时间。
这不是硬实时系统，仍受 CPU/GPU 调度影响，不模拟电机通信和相机传输延迟。

## 给新会话的提示词

将下面文字交给在本项目根目录打开的 Codex 或 Claude Code：

> 阅读 docs/agent_simulation.md。使用 mj_env.agent_cli 连接已运行的
> http://127.0.0.1:8765 仿真。先调用 info 和 observe，用你的图像查看工具
> 打开返回的 global.png 和 wrist.png。通过 move / joints / gripper / wait
> 逐步控制 SO101 抓起红色方块。每次动作后重新观察，根据反馈调整，并在抬起后
> wait 2 秒确认 success。不要调用 scripted_pick 或 VLA 模型替你完成任务。
> 默认只根据相机和机械臂状态判断物体位置；若需要真值辅助调试，先明确说明。

这里的图像读取依赖该会话具备本地图片查看能力；CLI 输出图片绝对路径，
不会直接将图片内容注入模型上下文。HTTP 客户端也可读取返回的 PNG base64。

## 调用示例

```powershell
uv run python -m mj_env.agent_cli info
uv run python -m mj_env.agent_cli observe
uv run python -m mj_env.agent_cli state
```

`observe` 将同一仿真时刻的图片和状态保存至 `.tmp/agent_observation/`，
输出 `global.png`、`wrist.png` 的绝对路径。文件按服务 session ID 和帧号存入
子目录，不同帧不复用路径，避免读图缓存。相同帧号表示同一快照。
旧帧会保留，可用 `--output` 指定其他根目录。

有参数时建议先写 JSON 文件，避免 PowerShell 的嵌套引号问题：

```powershell
Set-Content -Encoding utf8 .tmp/command.json '{"width_m":0.07,"duration_s":1}'
uv run python -m mj_env.agent_cli gripper --args-file .tmp/command.json

Set-Content -Encoding utf8 .tmp/command.json '{"position_m":[0.20,0,0.16],"hinge_yaw_rad":0,"duration_s":2}'
uv run python -m mj_env.agent_cli move --args-file .tmp/command.json
uv run python -m mj_env.agent_cli observe

Set-Content -Encoding utf8 .tmp/command.json '{"duration_s":2}'
uv run python -m mj_env.agent_cli wait --args-file .tmp/command.json
```

以上是打开夹爪并移动到空中一个位置的示例，目标坐标需要 agent 自己决定。

## 工具约定

| 命令 | JSON 参数 | 行为 |
| --- | --- | --- |
| `info` | `{}` | 对象名称、关节顺序、限位、夹爪范围、坐标系 |
| `observe` | 可选 `ground_truth: true` | 状态及全局、腕部两张 PNG |
| `state` | 可选 `ground_truth: true` | 关节实测值和目标值、TCP/夹持中心、接触/抬升/成功状态 |
| `reset` | `seed`、`target`，可选 `task` | 重置八物体布局与机械臂；默认 seed=3，target 保持当前值 |
| `joints` | `positions_rad: [六个值]`，可选 `duration_s` | 六关节绝对目标，平滑插值执行 |
| `gripper` | `width_m`，可选 `duration_s` | 按标定的夹爪间隙设置开合，保持五个机械臂关节目标 |
| `move` | `position_m: [x,y,z]`，可选 `hinge_yaw_rad`、`aperture_width_m`、`duration_s` | 用 IK 将夹持中心移动到指定世界坐标，保持原夹爪命令 |
| `wait` | 可选 `duration_s` | 保持关节目标等待指定时长，返回新状态 |
| `stop` | `{}` | 取消当前运动轨迹，保持最近执行的目标 |
| `shutdown` | `{}` | 关闭仿真与服务 |

长度为米，角度为弧度；机械臂基座为原点，+x 朝桌面前方，+z 朝上，桌面 z=0。
关节顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。
`duration_s` 默认 1，范围 0.05–10，取整到 20 Hz 的控制步。
动作提交即返回 ID；通过 `state` / `observe` 的 `action.status` 检查
`running` / `completed` / `cancelled`。插值结束不代表完全到位。
`wait` 只延迟当前 HTTP 请求并返回最新状态，不会打断动作或阻塞其他请求。

`move` 是俯视夹持姿态的低层几何工具，不负责识别物体、规划抓取序列或避障。
`hinge_yaw_rad` 表示夹爪铰链在世界 XY 平面的朝向，默认 0；抓长杆/哑铃时
通常让铰链方向沿长轴，手指横跨长轴闭合。IK 优先选择接近当前关节的对称解，
但五轴机械臂无法保证任意姿态。不可达请求返回错误，且不会执行动作。
关节插值路径并非直线末端路径，也没有碰撞规划；先抬高、再平移、后下降。

`aperture_width_m` 指定用于计算夹持中心偏移的参考间隙，默认使用实测开度。
它不会改变夹爪开合。TCP 是指尖，夹持中心位于后方，二者不能混用。
夹爪最小几何间隙约 2 mm、最大约 94 mm；`width_m: 0` 表示闭到下限。

`success` 要求目标抬升至少 6 cm 且两侧指垫接触承载。它反映当前控制步，
不是永久锁存；应额外 `wait` 检查保持结果。成功后服务仍接受控制。

默认状态不提供物体真实位姿；`ground_truth: true` 显式返回八个物体的世界
位置和 WXYZ 四元数。这是仿真真值，不是视觉识别结果。奖励/成功反馈也来自
仿真接触与位姿计算。八个可选 target 见 `info`。

## 原生 HTTP

`POST http://127.0.0.1:8765/call`，`Content-Type: application/json`：

```json
{"command":"move","args":{"position_m":[0.2,0,0.16],"duration_s":2}}
```

成功返回 JSON，参数错误返回 HTTP 400 与 `error`。`observe` 的原生返回含
`images_png_base64.global` 和 `.wrist`。服务只绑定本机回环地址，无身份认证；
适用于同一台机器上的可信会话。客户端等待超时或断开不会撤销已经提交的动作，
不要自动重试动作；先读取 state 确认。窗口关闭可在控制步之间中止当前动作。

验证接口：`uv run python -m unittest mj_env.test_agent_tools`。

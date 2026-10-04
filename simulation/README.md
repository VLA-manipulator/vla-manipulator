# llm-plan：用 DeepSeek (deepseek-flash) 控制 SO101 机械臂（MuJoCo 仿真）

大模型通过 OpenAI 兼容的 Chat Completions + function calling 接口，看双相机图像（全局 + 腕部），
逐段调用受限的机器人工具，控制 MuJoCo 中的 SO101 机械臂完成抓取任务。
不依赖任何 agent 框架；模型没有 shell、任意文件或任意 HTTP 访问能力。

> 现状：交互链路（看图、调用工具、执行、记录）已跑通，**自主抓取尚未通过验收**。
> 详见 [docs/deepseek_runner.md](docs/deepseek_runner.md)。

## 架构

```text
example/deepseek_robot.py  ──HTTPS──▶  DeepSeek API (deepseek-flash)
        │  tool calls
        ▼
mj_env/task_journal.py (任务日志 / 安全校验 / 观测包)
        │  HTTP 127.0.0.1:8765
        ▼
mj_env/agent_server.py (MuJoCo 仿真)
```

| 目录 | 内容 |
| --- | --- |
| `example/` | DeepSeek 机器人运行器、纯文本/图像客户端 |
| `mj_env/` | SO101 桌面仿真、仿真 HTTP 服务、任务日志与运行时、色块与光流视觉工具 |
| `so101/` | SO101 运动学与 MJCF/URDF/网格模型 |
| `docs/` | 运行说明、仿真接口说明、模型运行时系统提示词 `robot_runtime_prompt.md` |
| `configs/` | DeepSeek 配置模板 |

## 环境

- Python 3.12，[uv](https://docs.astral.sh/uv/)
- 不需要 GPU。
- 支持平台：Windows、Linux（x86_64 / aarch64）。实际运行验证过 Windows 11；Linux 依赖均有预编译 wheel，尚未实际运行验证。
- 没有显示器的 Linux 服务器上，仿真服务需加 `--headless`，并设置 `MUJOCO_GL=egl`（有 GPU）或 `MUJOCO_GL=osmesa`（纯 CPU，需安装 OSMesa）用于相机离屏渲染：

  ```bash
  MUJOCO_GL=egl uv run python -u -m mj_env.agent_server --headless --seed 3 --target cube_red
  ```

```powershell
uv sync
```

依赖版本已固定为验证过的版本（见 `pyproject.toml` / `uv.lock`）。

## 配置 API Key

```powershell
copy configs\deepseek.example.json configs\deepseek.local.json
```

Linux：`cp configs/deepseek.example.json configs/deepseek.local.json`

在 `configs/deepseek.local.json` 中填写 `api_key`，或设置环境变量 `DEEPSEEK_API_KEY`。
该文件已被 `.gitignore` 忽略，不要把密钥写进模板。

## 运行

以下命令均在项目根目录运行，每个服务各开一个终端。

1. 启动仿真服务（保持运行）：

   ```powershell
   uv run python -u -m mj_env.agent_server --seed 3 --target cube_red
   ```

2. 运行 DeepSeek 控制器：

   ```powershell
   uv run python example/deepseek_robot.py --task-id task_red_001 --instruction "抓起红色方块"
   ```

   不带参数启动时可以在命令行交互输入任务；运行期间输入新的一行即可修订任务，`/quit` 退出。
   任务日志写入 `.tmp/tasks/<task-id>.jsonl`，用同一 ID 再次启动会续跑。

仅测试 API 连通性和图像输入（不控制机械臂）：

```powershell
uv run python example/deepseek_client.py --prompt "用一句话回答：你能接收图片吗？"
```

## 离线测试

```powershell
uv run python -m unittest example.test_deepseek_robot mj_env.test_task_runtime mj_env.test_task_journal mj_env.test_observation_bundle mj_env.test_closed_loop mj_env.test_visual_servo mj_env.test_image_regions mj_env.test_gripper_identification mj_env.test_agent_tools
```

## 更多文档

- [docs/deepseek_runner.md](docs/deepseek_runner.md)：运行器参数、token 预算、工具说明与已知限制
- [docs/agent_simulation.md](docs/agent_simulation.md)：仿真 HTTP 接口
- [docs/robot_runtime_prompt.md](docs/robot_runtime_prompt.md)：发送给模型的系统提示词

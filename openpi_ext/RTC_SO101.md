# SO101 的 RTC 推理原型

依据 Physical Intelligence 的 [RTC 论文](https://arxiv.org/abs/2506.07339)及其[官方示例代码](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)。RTC 不改模型权重：执行当前动作块时，后台推理下一块；对已经承诺执行的动作施加强引导，对剩余重叠动作使用指数递减软掩码。采样器实现了论文的伪逆引导和最大权重 5。

本实现是**独立、默认不启用**的实验入口，原 `serve_policy.py` 和 `evaluate_openpi.py` 均未修改。控制端在 20 Hz 下按真实时间推进模拟器；如果模型推理超过当前块可用的剩余动作，就明确报 `deadline_missed` 并结束该次测试，不会悄悄复用越界动作。

## 文件

- `rtc_sampler.py`：π0/π0.5 的 JAX 引导流采样。OpenPI 用 `t=1` 表示噪声、`t=0` 表示动作，与论文时间方向相反，代码已转换符号。
- `rtc_policy.py` 和 `rtc_target.py`：独立策略包装。控制端回传上一块的实际电机目标；服务端用当前观测和原训练配置的 DeltaActions/Normalize 变换重新编码到模型动作空间，再作为引导目标。前五个关节是相对当前状态的增量，不能直接沿用上一块的归一化增量。
- `serve_rtc.py`：独立 WebSocket 策略服务。
- `simulation/mj_env/rtc.py`：后台请求与旧动作块并行执行的控制器。
- `simulation/mj_env/scripts/evaluate_openpi_rtc.py`：单物体小规模闭环评测入口。

## 在 AutoDL 上试运行

先检查 OpenPI 当前版本与 `sample_actions`、`Policy.infer` 接口兼容，并确认没有已有的训练或评测占用 GPU。建议先用已验证的 30k 检查点、一个原始测试种子做烟测，再决定是否扩大评测。

```bash
export REPO=/root/autodl-tmp/workspace/vla-manipulator-20261022
export OPENPI_ROOT=/root/autodl-tmp/workspace/openpi
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-data
export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$OPENPI_ROOT"
uv run "$REPO/openpi_ext/serve_rtc.py" \
  --config pi05_so101_lora_pick8_o1_slow240_5k \
  --checkpoint /root/autodl-tmp/checkpoints/pi05_so101_lora_pick8_o1_slow240_5k/pick8_o1_slow240_5k_20261024/29999 \
  --num-steps 5 --port 8000
```

在具有 `openpi-client` 与模拟器依赖的另一进程执行：

```bash
cd "$REPO/simulation"
python -m mj_env.scripts.evaluate_openpi_rtc \
  --host 127.0.0.1 --port 8000 --target cube_red \
  --seed 1000 --episodes 1 --horizon 10 \
  --min-horizon 3 --initial-delay 2 --max-steps 600
```

这项测试使用真实 20 Hz 节拍。启动时分别预热普通与 RTC 引导采样，再测量包含客户端、网络和服务端的往返时延；以测得的最大延迟折算控制步数，并加一个控制步作为余量，自动提前发起下一块推理。若时延连 10 步动作块都容不下，会明确失败。录屏或大量图像渲染会改变延迟，应单独计时。重点记录 `deadline_missed`、往返时延、每次 `policy_timing.infer_ms`、相邻动作块关节位置/速度/加速度跳变，以及和同步基线同种子的成功率。RTC 可能减少停顿，但不能仅凭动作变平滑推断抓取成功率提高。

目前已完成本地掩码、当前状态下的相对动作重锚定和控制器测试；AutoDL 实例在核查时仅暴露 CPU，服务端 JAX 编译、推理时延和闭环效果仍未完成 GPU 验证。不要直接接入真机控制链。

与论文真实机械臂实验的差异：该实验使用 50 步动作块、50 Hz 控制；这里的既有 SO101 模型是 10 步动作块、20 Hz 控制，推理必须在剩余动作的时间预算内完成。`initial-delay` 是预计延迟的步数，示例值 2 尚未按实际服务器时延校准；使用前要测量包含传输时间的延迟，并核对是否小于剩余动作块时间。若推理超时，本评测会报告 `deadline_missed` 而不是继续执行越界动作。

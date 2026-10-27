# SO101 的 RTC 推理原型

依据 Physical Intelligence 的 [RTC 论文](https://arxiv.org/abs/2506.07339)及其[官方示例代码](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)。RTC 不改模型权重：执行当前动作块时，后台推理下一块；对已经承诺执行的动作施加强引导，对剩余重叠动作使用指数递减软掩码。采样器实现了论文的伪逆引导和最大权重 5。

本实现是**独立、默认不启用**的实验入口，原 `serve_policy.py` 和 `evaluate_openpi.py` 均未修改。控制端在 20 Hz 下按真实时间推进模拟器；如果模型推理超过当前块可用的剩余动作，就明确报 `deadline_missed` 并结束该次测试，不会悄悄复用越界动作。

## 文件

- `rtc_sampler.py`：π0/π0.5 的 JAX 引导流采样。OpenPI 用 `t=1` 表示噪声、`t=0` 表示动作，与论文时间方向相反，代码已转换符号。
- `rtc_policy.py`：独立策略包装。将模型归一化空间内的原始动作返回给控制端，下一次作为引导目标；这样避免把电机单位值误当作模型归一化值。
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

这项测试使用真实 20 Hz 节拍，以便发现推理延迟是否超过 10 步动作块的预算。录屏或大量图像渲染会改变延迟，应单独计时。重点记录 `deadline_missed`、每次 `policy_timing.infer_ms`、相邻动作块关节位置/速度/加速度跳变，以及和同步基线同种子的成功率。RTC 可能减少停顿，但不能仅凭动作变平滑推断抓取成功率提高。

目前仅完成本地控制器和语法测试；AutoDL 连接在验证时被远端关闭，因此服务端 JAX 编译和闭环效果仍需在实例可用时验证。不要直接接入真机控制链。

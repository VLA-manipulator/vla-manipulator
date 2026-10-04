# 直接调用 DeepSeek 执行机器人工作流

入口 `example/deepseek_robot.py` 使用 OpenAI Chat Completions 请求格式，经现有 `httpx` 依赖访问 `https://api.deepseek.com/chat/completions`，模型 ID 为用户指定的 `deepseek-flash`。不依赖 Claude Code，也不使用其 effort 设置。

## 配置与启动

在 `configs/deepseek.local.json` 填入 `api_key`，也可用环境变量 `DEEPSEEK_API_KEY` 覆盖。该本地文件已加入 gitignore；可共享模板是 `configs/deepseek.example.json`。不要把密钥写进模板。

服务启动后，在项目根目录运行（会实际控制机械臂）：

```powershell
uv run python example/deepseek_robot.py --task-id task_blue_002 --instruction "抓起蓝色方块"
```

新任务只初始化一个日志；已有任务先 `resume`，固定使用同一 ID。不要与其他机器人控制窗口同时运行。程序加载精简入口 `docs/robot_runtime_prompt.md`，执行器仍使用通用工作流和适配器的检查。只暴露 `robot_step`、`select_target`、`task_journal`、过滤后的 `robot_info` 和当前观测清单的 `view_timeline`。模型没有 shell、任意文件访问或任意 HTTP 能力。请求参数通过 stdin 传给日志工具，不生成每轮参数文件。

`robot_step` 用实测夹持中心计算相对位移，自动填入事件引用；不使用 TCP 代替夹持中心。一次只允许位移、关节增量、夹爪开度之一。`select_target` 将模型看图选出的色块索引固定到同一日志，分别跟踪全局和腕部像素；没有有效目标关联时拒绝闭合、接触阶段下降和成功提交，张开和上退仍可恢复。当前色块检测仅覆盖蓝、红、绿、紫、橙；不支持的外观应报告限制，不能把颜色当成物体身份。跟踪丢失不会自动跳到远处的同色物体，需重新看图选择。此检查只能减少串目标，不能证明抓住。

每次新观测后只保留最近两组工具交互和最新指令，完整记录仍保留在原 JSONL。模型每轮实际输入以打印的 usage 为准，图片也计入输入消耗。

也可以不带参数启动，随后在命令行输入任务；不提供 `--task-id` 时程序生成一个固定 ID 并显示：

```powershell
uv run python example/deepseek_robot.py
```

运行过程中直接输入一行新指令（或 `/update 新指令`）即可修订当前任务。输入线程只排队，不控制机械臂；新指令在当前工具执行段结束后生效。如果输入发生在 API 请求期间，收到的旧响应会在工具执行前被丢弃。修订追加为同一日志的 `task_update`，原历史保留；模型须先更新验收条件，然后重新观察规划。`/quit` 在执行段边界退出，不是急停；API 请求期间可能需要等响应或超时。`--non-interactive` 禁止读取控制台，适合脚本启动，必须同时给出 `--instruction`。任务结束或预算耗尽后进程退出；继续同一任务使用显示过的 ID。

每次观察自动读取工具清单中的必读拼图，将真实图像按 OpenAI `image_url` 数据 URL 格式附到下一次请求。模型需要同时支持图片输入与 function calling；**本程序不把模型名称当作能力证明**。若接口不支持图片或工具、返回错误，程序停止，不退化成仅凭文字控制。新图加入后删除上下文中的旧图片字节，保留事件及数值；需要旧图时经日志工具回看。

## 限制思考与消耗

- `max_tokens`：每次请求默认2048。具体模型是否将思考计入这一上限取决于接口；截断响应中的工具调用一律不执行。
- `max_rounds`：默认40次模型请求，每次最多允许一个工具调用；返回多个时全部不执行。
- `max_total_tokens`：默认100000，按响应 `usage.total_tokens` 累计，包含重复发送的上下文。达到预算后不再执行返回的工具调用。它是**响应后的停止阈值，不是服务端费用硬上限**，最后一次请求可能超额；缺少 usage 时停止。
- `max_images_per_round`：默认32张，超过时报告问题，不静默省略必读证据。
- `timeout_s`：API 网络读写超时，默认120秒，不是思考 token 上限。不自动重试请求。
- `reasoning_effort` 默认 `null`（不发送），`extra_body` 现在默认 `{"thinking":{"type":"disabled"}}`，已在线确认该接口接受。不能假定 `low` 会关闭思考。

`thinking` 是供应商扩展，不是所有 OpenAI 兼容服务的通用字段。2026-09-17 实测该接口接受关闭思考，图片请求和工具调用能够往返，模型可描述场景。但本次蓝色方块抓取未成功：一次探测移动完成，后续运动返回 HTTP 400，最后累计预算耗尽。默认思考模式曾在一轮中耗尽2048个输出 token，全部为 reasoning_tokens；关闭后无此现象。完整交互链路可运行不代表可靠抓取能力已验证。

机器人模式默认不输出思考内容，但隐藏思考**不等于减少思考 token**。为兼容需要回传 `reasoning_content` 的接口，返回的该字段保留在当前进程上下文内。每轮打印 API 的完整 usage 供核对；任务观测、决策、计划及动作仍由现有工具追加到同一个任务 JSONL，不另建每轮日志。

补充调试结果：精简入口和图像去重已实现，但尚未通过自主抓取验收。蓝色方块续跑曾被模型报为成功，人工复核发现腕部目标尺度随上提缩小，已在该任务原 JSONL 追加 `validation_audit`，不计成功。红色方块从初始状态测试仍因夹指定位和运动方向判断反复失误而耗尽预算。另一次启用思考、单轮上限8192的对照在第6轮耗尽全部8192个 reasoning tokens，没有执行截断动作；默认仍关闭思考。官方接口已接受 `strict:true` 工具 schema 的无动作测试，程序同时保留本地参数校验，不能把接口接受当作永远符合 schema 的保证。

预算耗尽、API 错误或模型结束时不自动松爪或移动；停在执行段边界不代表抓取状态可以无限维持。现有执行器负责其自身动作异常处置。恢复时使用同一任务 ID；不得重放结果未知的命令。

## 局部视觉修正



`set_robot_reference` 和 `visual_align_width_m` 是待实机闭环验证的局部修正功能。必须视觉确认固定夹指 TCP 点，通过独立小幅探测和额外验证位置学习像素与实测机器人位移的关系。至少五个锚点，拟合及留出验证误差不超过 2 像素；姿态变化超过 0.10 rad、离开初始点 0.12 m 或跟踪不一致会失效。该工具没有相机外参或深度输入，投影对齐不能替代高度判断，也不是抓取成功证据。

自主抓取仍未通过验收。不要把服务可用、单元测试通过或模型自行报成功当作抓取成功。

最近两次有界续跑分别执行 7 / 5 轮请求，接口统计累计 90191 / 61919 tokens，均未完成抓取。第二次验证了参考点丢失后的横向 probe 被工具拒绝。观测包的 `visual_relation` 明确区分目标像素和夹指相对位置；该约束目前只覆盖有全局目标跟踪的横向 probe，不是通用碰撞检查，也不保证其他动作正确。自动选择固定夹指参考点仍是待解决问题。

## 独立接口测试命令

### 主动夹指辨识试验（尚不能用于自动对准）

`identify_gripper` 将小幅闭合和返回分别作为独立执行段；每段仍需图像复核。`mj_env.gripper_identification` 仅使用两路图像与实测姿态，提取光流往返一致的运动候选，并检查臂关节漂移和实际开度变化。输出 `reference_valid=false`、`alignment_motion_allowed=false`，不会自动生成 TCP、接触点或深度。影子也会随夹指运动；不能将候选框中心当作夹指参考点。

标注图采用 G0/W0 等固定编号，并附候选区域放大图。原图、标注图、测量和接口测试结果仍追加到任务原 JSONL；缓存覆盖写入。返回动作必须有当前匹配的 probe 证据，缺失时拒绝执行。

2026-09-17 在现有任务中进行了一次 0.015 m 闭合与反向张开：腕部实体夹指候选的往返误差最大约 0.76 px，臂关节漂移小于 0.000002 rad，但夹指投影也产生了往返一致的影子候选。没有将这些候选授权为控制参考。

官方 deepseek-flash、关闭思考的只读图像测试：初始排版出现编号错误；增加局部放大与明确编号后，一次实际观测用例正确识别腕部实体夹指和影子，四个状态用例的下一步操作均符合限制。但同图不同状态的身份描述不一致，后续独立视觉分类三次重复仅一次完整符合腕部人工标签。因此尚未证明模型能够可靠完成身份复核，更未验证完整标定或自主抓取。不能把状态规则通过率当成视觉识别通过率。

（本独立项目未包含该评估脚本，它依赖原仓库中人工复核过的任务日志。）原复现实验：`uv run python -m example.evaluate_gripper_understanding --task-id red_autonomous_validation_03`，加 `--visual-only` 做三次重复视觉分类。此评估使用这次已人工复核的场景标签，**不是通用数据集**。仅调用模型并将响应写回任务日志，不执行模型返回的动作；故意错误的标注或状态明确记录为测试 fixture，不作为真实机器人观测使用。

不控制机器人时可使用文本/图像客户端：

```powershell
uv run python example/deepseek_client.py --prompt "用一句话回答：你能接收图片吗？"
uv run python example/deepseek_client.py --prompt "描述这张图" --image PATH_TO_IMAGE.png
```

模型自称支持图片不构成验证；应以实际图片请求是否被正确处理为准。客户端支持 `--system-file`、`--prompt-file`、`--no-stream`、`--show-reasoning` 和可选 `--output`（不覆盖已有文件）。默认显示回答、usage、耗时及接收到的思考字符数；字符数不是 token 数。API Key 不输出、不写入结果。

离线验证：

```powershell
uv run python -m unittest example.test_deepseek_robot mj_env.test_task_runtime mj_env.test_task_journal mj_env.test_observation_bundle mj_env.test_closed_loop
```

# EMAS Agents Runtime

当前主线是“Task Execution Service（内含 EmbodiedGPT relay agent）+ 原生 AI2-THOR 多 robot receiver”。

```text
client / coordinator -> task_execution_server.py -> ai2thor_receiver_server.py -> AI2-THOR
```

## 当前稳定入口

你日常测试 agents 模块时，优先使用 EmbodiedGPT runtime 的 CLI：

```bash
cd /225010231/mwl/EMAS/agents/EmbodiedGPT_Pytorch

SEND_ACTIONS_URL="http://10.20.18.3:19000/execute_actions" \
./auto_scene_actions.sh \
  --task "go to fridge." \
  --primary-robot-id 0 \
  --relay-mode \
  --closed-loop-replan \
  --print-raw-output
```

行为约定：

- `go to ...` / `navigate to ...` / `move to ...` / `walk to ...` 这类 navigation-only 任务直接调用 receiver 的 `/goto`，不加载 Qwen，不进入 relay。
- `PickupObject` / `PutObject` / `OpenObject` / `CloseObject` 等交互任务走 EmbodiedGPT semantic planning + object grounding + closed-loop relay。
- primary robot 通过硬验证即可执行时不进入 relay。
- Coordinator 每次任务都从 receiver 的权威状态发现全部 robot，并刷新其 pose、视野、
  inventory 和最近动作状态；调用方不提供候选或已知 robot 列表。
- primary robot 不能执行时，relay 会打印 primary 失败原因、候选摘要、协调成功或失败原因。
- 所有普通 `/execute_actions` payload 都包含顶层 `"stop_on_failure": false`。
- `/goto` 失败、relay 失败和执行失败都应返回结构化 JSON，不能只抛截断异常。

## 动作能力分层

`ai2thor_receiver_server.py` 的 `/execute_actions` 会把动作字典交给原生
`controller.step`，因此底层 build 支持并可直接透传导航、交互、物体操作以及
`Pass`/`Done` 等原生动作。但这不表示 planning Qwen 可以请求所有动作：

| 层级 | 动作 |
|---|---|
| planner 可请求 | `pick/place/open/close/toggle_on/toggle_off/clean/slice/drop/push/pull/move_held/break/cook/fill`，以及逻辑 `navigate/find/inspect` |
| runtime 内部 | `MoveAhead/MoveBack/MoveLeft/MoveRight/RotateLeft/RotateRight/LookUp/LookDown/Pass/Done` |
| 特权或查询，仅内部 | `Teleport/TeleportFull/GetReachablePositions/SetObjectStates` |

参数约定：

- `MoveHeldObject` 使用 `right/up/ahead`，不是 `x/y/z`。
- `FillObjectWithLiquid` 除 `objectId` 外还必须提供 `fillLiquid`，当前允许 `water/coffee/wine`。
- `SetObjectStates` 使用嵌套的批量状态结构（`objectType/stateChange` 及相应布尔状态），不是 `objectId + stateChange`。
- 原生开关动作是 `ToggleObjectOn` 和 `ToggleObjectOff`；不存在通用 `ToggleObject`。
- `GotoObject`/`NavigateTo` 是 EMAS 逻辑动作或 benchmark 记分名称，通过 `/goto` 和低层移动实现，不直接发送给原生 controller。

## 启动

先启动共享场景 receiver：

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 --robots 2 --port 19000 --no-show
```

receiver 对外提供：

```text
GET  /health
GET  /robots
GET  /state
GET  /reachable_positions?robot_id=0
POST /observe
POST /execute_actions
POST /goto
POST /reset
```

再启动 task execution service：

```bash
bash run_task_execution_server.sh \
  --model-path models/Qwen3.5-4B \
  --receiver-url http://127.0.0.1:19000 \
  --port 18080 --device cuda --dtype float16
```

任务服务对外提供：

```text
POST /execute_task
```

这个 HTTP 入口用于 planning/memory 联调。推荐由 EMAS 根目录下的
`scripts/hybrid_decision_loop.py --execution-mode task_service` 调用它；planning
模块不需要知道 `auto_scene_actions.py` 的内部细节。

这个接口的上游边界是当前 subtask 及其任务局部约束和 Allocation 选出的
`primary_robot_id`。Hybrid/Allocation 不传 `known_robot_ids`、`eligible_agent_ids`、
Agent 状态快照、完整 Task Graph 或 Execution Plan。Coordinator 会在每次请求中直接
查询 receiver，自主得到并刷新全部 Agent；内部诊断中的 known/discovered 集合只是该次
查询结果。响应中的 `completion_agent_id` 表示真正完成任务的 executor。



### Planning/Memory 联调入口

不再使用已删除的独立 `emas_relay_bridge.py`。当前联调路径是：

```bash
cd /225010231/mwl/EMAS
python scripts/hybrid_decision_loop.py \
  --scene_name train_3 \
  --task "open the fridge" \
  --scenegraph-info /path/to/scenegraph_info.json \
  --execution-mode task_service \
  --task-service-url http://127.0.0.1:18080/execute_task \
  --task-service-dry-run \
  --max-task-loops 1
```

`task_service` 模式会逐个读取 planning allocation 中的 `{agent_id, subtask}`，把
`agent_id` 映射为 `primary_robot_id` 后调用 `/execute_task`，再把结果写回原有
`communication_inbox.json`、`task_statuses` 和 `execution.traces`。Coordinator 自己从
receiver 发现全部 robot；若它选择了非 primary executor，Hybrid 以返回的
`completion_agent_id` 记录真实执行者。成功的 subtask 写入
`execution.completed_task_ids`；失败、HTTP error、timeout 或非 JSON 响应都会映射为
`WAIT_RETRY`，不会让 hybrid loop 直接崩溃。

## Smoke Test

没有 `curl` 时可直接用 Python 进行健康检查：

```bash
cd /225010231/mwl/EMAS/agents

python scripts/agents_smoke_test.py \
  --execute-actions-url http://10.20.18.3:19000/execute_actions \
  --robot-id 0 \
  --goto-target Fridge \
  --skip-cli
```

如果还想检查 CLI wrapper 是否仍指向当前 EMAS 目录，并验证 navigation-only dry-run 路径：

```bash
python scripts/agents_smoke_test.py \
  --execute-actions-url http://10.20.18.3:19000/execute_actions \
  --robot-id 0 \
  --goto-target Fridge
```

关键验收点：

- `wrapper_path` 必须显示 OK，表示没有跳回旧的 `/225010231/mwl/Linhao/EmbodiedGPT_Pytorch`。
- `/health`、`/state`、`/reachable_positions`、`/execute_actions` 都应 OK。
- `/goto` dry-run 可以成功，也可以返回结构化 failed，例如目标不存在；但不能是连接错误或非 JSON 响应。

## 回归测试

```bash
cd /225010231/mwl/EMAS/agents
python -m py_compile ai2thor_receiver_server.py task_execution_server.py
python -m unittest tests/test_ai2thor_navigation_planner.py tests/test_task_execution_server.py

cd /225010231/mwl/EMAS/agents/EmbodiedGPT_Pytorch
python -m py_compile demo/auto_scene_actions.py demo/relay_agent.py demo/plan_media.py
python -m unittest tests/test_plan_media.py
python -m unittest tests/test_relay_agent.py
python -m unittest tests/test_relay_agent_integration.py tests/test_relay_agent_closed_loop.py
```

任务测试表见 [AGENTS_TASK_TEST_MATRIX.md](AGENTS_TASK_TEST_MATRIX.md)。

任务 API 见 [TASK_EXECUTION_SERVICE.md](TASK_EXECUTION_SERVICE.md)，relay 闭环设计见 [relay_closed_loop_design_cn.md](relay_closed_loop_design_cn.md)。AI2-THOR HTTP 接口和项目架构见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## ProcTHOR 场景烟测

`procthor_benchmark.py` 独立于 HTTP receiver，用于验证 ProcTHOR house JSON 能否加载、传送、查询可达位置、执行一步移动并返回图像。它不会占用 `18080` 或 `19000` 端口：

```bash
python procthor_benchmark.py \
  --scene-json /home/kinova-1/procthor-10k/house_0001.json \
  --local-executable-path /home/kinova-1/.ai2thor/releases/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917 \
  --x-display :1 \
  --output-dir /tmp/procthor_benchmark_run
```

结果会写入输出目录的 `summary.json` 和渲染帧。传入包含多个 `.json` 或 `.json.gz` 文件的目录即可进行批量烟测；这还不是标准 ObjectNav evaluator。

### 生成固定 ProcTHOR 测试集

`generate_procthor_dataset.py` 使用安装的官方 ProcTHOR `HouseGenerator` 和 `PROCTHOR10K_ROOM_SPEC_SAMPLER`。它按固定 seed 生成场景、转换为当前本地 Unity build 可加载的 schema、验证起点和可达位置，并将结果记录到 `manifest.json`：

```bash
python generate_procthor_dataset.py \
  --output-dir /home/kinova-1/procthor-10k/generated \
  --split train --start-seed 0 --count 20 \
  --local-executable-path /home/kinova-1/.ai2thor/releases/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917 \
  --x-display :1
```

重复同一命令会保留已有场景；可用新的 `--start-seed` 和 `--count` 断点续跑。默认跳过当前 build 不兼容的小物体生成阶段，因此该测试集适合导航、场景理解和 relay 验证，而不是完整官方 ObjectNav 对比。

## 文件状态

完整文件分层与保留状态见 [ARCHITECTURE.md](ARCHITECTURE.md)。

当前仓库已移除单步 `qwen_action_service.py`、本地 Qwen loop、早期 `MultiAgentEnv`、custom agent 和 OpenVLA 实验代码，只保留 Qwen3.5-4B 的 relay 闭环路线。

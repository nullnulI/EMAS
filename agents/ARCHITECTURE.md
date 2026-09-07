# Agents Runtime Architecture

> EMAS 系统级 Planning/Allocation/Hybrid runtime 契约以
> [`../ARCHITECTURE.md`](../ARCHITECTURE.md) 为唯一权威来源。本文说明 agents
> 子系统的当前主线、接口和文件归属；历史草稿仅保存在外部 legacy archive 中，
> 不定义当前接口。

## 1. 子系统定位

`agents` 是 EMAS 的仿真执行端和模型接入端，负责：

- 在同一个 Unity 世界中启动 AI2-THOR `Controller(agentCount=N)`；
- 用 `robot_id` 路由 observation 和 action；
- 提供 pose、RGB、object metadata、inventory 与动作错误；
- 接收 Hybrid 运行时调度循环从 Execution Plan 首个 unit 下发的任务；
- 在 task service 内完成 relay executor 选择、语义动作规划、objectId grounding、
  局部恢复与结构化执行反馈；
- 支持 post-trained Qwen3.5 模型的第一视角推理。

它不拥有 semantic Task Graph、全局 `depends_on`、Allocation Execution Plan、长期
progress 或全局 replan budget。这些分别属于 Planning、Allocation 和 Hybrid runtime。

## 2. 当前唯一服务主线

```text
Hybrid runtime loop
  | units[0] task + task-local constraints + primary_robot_id
  | POST /execute_task
  v
task_execution_server.py (:18080)
  | discover/refresh every robot, relay selection, grounding and local recovery
  | POST /observe, /state, /execute_actions, /goto
  v
ai2thor_receiver_server.py (:19000)
  |
  v
AI2-THOR shared Unity scene
  Controller(agentCount=N), agentId == robot_id
```

`ai2thor_receiver_server.py` 是唯一保留的仿真 HTTP receiver。它使用标准
`controller.step(..., agentId=robot_id)`，确保多个 robot 共享世界状态，同时拥有
各自 pose、camera 与 inventory。旧的单 camera teleport、custom agent、OpenVLA、
单步 Qwen service 和 `MultiAgentEnv` 原型不属于当前主线。

## 3. 文件地图

| 路径 | 状态 | 作用 |
| --- | --- | --- |
| `ai2thor_receiver_server.py` | 主线 | 共享 Unity receiver；提供 robot、scene、observation、action、navigation 与 reset 接口。 |
| `task_execution_server.py` | 主线 | 常驻任务服务；执行一个由 Execution Plan assignment 指定的语义任务。 |
| `EmbodiedGPT_Pytorch/demo/auto_scene_actions.py` | 主线引擎 | 任务意图、语义动作、grounding、执行反馈和局部闭环控制。 |
| `EmbodiedGPT_Pytorch/demo/relay_agent.py` | 主线引擎 | 基于硬约束和证据选择 executor，并输出结构化失败。 |
| `EmbodiedGPT_Pytorch/demo/qwen35_backend.py` | 主线引擎 | Qwen3.5 模型加载、视觉输入和工具调用推理。 |
| `models/Qwen3.5-4B/` | 主线资源 | post-trained Qwen 权重、processor 和配置。 |
| `TASK_EXECUTION_SERVICE.md` | 接口文档 | `/execute_task` 的启动方式、请求和响应契约。 |
| `relay_closed_loop_design_cn.md` | 设计文档 | relay、executor 选择、硬验证和局部恢复。 |
| `tests/test_task_execution_server.py` | 测试 | 不加载模型地验证 HTTP 包装和 relay/closed-loop 参数映射。 |
| `README.md` | 入口文档 | agents 服务启动命令和文档索引。 |

`__pycache__/` 与 `output/` 是生成内容，不定义接口，也不属于源代码结构。

## 4. AI2-THOR Receiver 契约

推荐无界面启动：

```bash
python agents/ai2thor_receiver_server.py \
  --scene FloorPlan1 \
  --robots 2 \
  --port 19000 \
  --no-show
```

核心映射为：

```python
Controller(scene="FloorPlan1", agentCount=robot_count, port=0)
controller.step(action="MoveAhead", agentId=robot_id, renderImage=True)
```

### Robot state

每个 robot state 至少描述：

| 字段 | 含义 |
| --- | --- |
| `robot_id` | AI2-THOR `agentId`。 |
| `position`, `rotation`, `horizon` | 当前 pose 与相机俯仰。 |
| `inventory`, `held_object` | 当前手持状态；必须来自最新 metadata。 |
| `last_action`, `last_success`, `last_error` | 最近一次动作及结果。 |

所有涉及 object interaction 的 `objectId` 必须取自最新 `/observe` 或 `/state`；物体
移动、拾取、放置、切片或状态变化后，不能继续信任旧 id 或旧 inventory。

### HTTP endpoints

| Endpoint | 用途 |
| --- | --- |
| `GET /robots` | 获取全部 robot pose、inventory 和最近动作状态。 |
| `GET /state` | 获取场景对象、所选 robot 状态，并可返回图像。 |
| `POST /observe` | 获取一个 robot 的第一视角 observation。 |
| `POST /execute_actions` | 为指定 `robot_id` 执行一批 AI2-THOR actions。 |
| `GET /reachable_positions` | 获取指定 robot 的可达位置。 |
| `POST /goto` | 将目标位置、object id/type 转换为导航动作，可选择执行。 |
| `POST /reset` | 重置 scene 或 agent 数量。 |

`/execute_actions` 返回 `success`、`partial` 或 `failed`，并保留每步 action、robot、
pose、inventory 和原始错误。`stop_on_failure=true` 时，前一步失败会阻止后续动作；
调用方不得把未执行动作记为成功。

## 5. Task Execution Service 契约

`task_execution_server.py` 是任务级唯一入口。它接收 Hybrid 运行时调度循环已选定的语义任务
与 preferred executor，随后执行：

1. 解析任务意图；
2. 从 receiver 的权威状态发现全部 robot，并刷新各自的 observation、pose、metadata、
   inventory、可达性和最近动作状态；
3. 使用硬前置条件与 relay 证据选择 executor；
4. 生成不长期绑定 objectId 的 semantic action plan；
5. 用当前 observation 将 object type grounding 为精确 objectId；
6. 调用 receiver，合并动作反馈并在本地恢复预算内重试；
7. 返回 success、可恢复执行失败，或需要上游 Planning 的结构化失败。

`primary_robot_id` 是偏好而不是伪造可执行性的许可。若任何 robot 都不满足硬约束，
task service 必须返回可解释失败，不能悄悄切换到不合法 executor。

最小请求边界是：

- `task_id` 和当前 `task`；
- Allocation assignment 映射得到的 `primary_robot_id`。

请求可以附带当前 structured `subtask` 的 action/grounding 等任务局部约束、精简的
scene hint、dry-run 与局部闭环预算。它不包含完整 semantic Task Graph、Execution
Plan、全局 progress、`known_robot_ids` 或 `eligible_agent_ids`。Coordinator 在每次请求
中直接查询 receiver；内部日志或返回诊断中的 known/discovered robot set 仅表示此次
发现结果，不是上游提供的候选名单。

典型响应边界包括：

- `closed_loop_result`、成功/失败状态及 reason；
- 实际 `completion_agent_id`（详细 trace 中为 `executor_robot_id`）；
- `closed_loop_trace`、grounded actions 和 executor 证据；
- receiver pose、inventory、object/relation changes；
- `failure_code`、`recoverable` 与 `recommended_recovery`。

`needs_upstream_planning` 表示当前 observation 与硬约束下无法继续，不是 HTTP
transport error。上游应依据结构化 failure code 决定 replan；agents 不改写全局图。

## 6. Qwen3.5 接入边界

`EmbodiedGPT_Pytorch/demo/qwen35_backend.py` 接收第一视角图像或工具调用 messages，
输出任务意图或受约束 semantic plan。模型输出不能作为执行成功事实，也不应长期
携带 objectId。`auto_scene_actions.py` 使用 receiver 的当前对象、affordance、
visibility 与 inventory 做确定性验证和 grounding。

主线通过 chat template 设置 `enable_thinking=False`，要求结构化 JSON/tool call；
不要依赖或记录模型隐藏思维链。图像大小、输出 token 数和并发模型数量会直接影响
显存与延迟，正式运行应固定服务 GPU 并保留足够余量。

## 7. 导航与动作约束

`/goto` 使用 AI2-THOR `GetReachablePositions`、二维可达图和 A*，不是 LLM 或
ObjectNav policy。物体目标会选择满足距离限制的可达站立点，再转换为 rotate/move
动作。常见结构化失败包括 `invalid_target`、`target_not_found`、
`no_reachable_positions`、`no_reachable_goal_near_target`、`no_path`、
`action_limit_exceeded` 和 `execution_failed`。

物体动作还必须满足 AI2-THOR 状态约束。例如 `PutObject` 要求 robot 当前持物、目标
确为 receptacle、目标可达且容器状态允许。动作失败是闭环证据，不是可以忽略的日志。

## 8. 运行与排错

建议顺序：

1. 确认 receiver 与 task service health 均为 ready；
2. 检查 `/robots` 的实际 agent 数、pose 与 inventory；
3. 检查 task response 的 `failure_code`、`closed_loop_trace` 和 executor 证据；
4. 查看 `/execute_actions` 每步 `results[i].error`，确认是否被前一步提前停止；
5. 用最新 `/state?robot_id=N&render_image=1` 验证对象、容器和持有状态；
6. 对 `needs_upstream_planning` 保留完整报告并交回 Hybrid 运行时调度循环，不在 agents 内改图。

服务器环境优先使用 `--no-show`。Qwen 服务应固定到有足够空闲显存的 GPU；正式
排障以结构化 HTTP payload、receiver metadata 和保存的 observation 为准。

## 9. 文档与维护规则

- 系统级职责、Execution Plan state 和 edge kind 只在根
  [`ARCHITECTURE.md`](../ARCHITECTURE.md) 定义。
- agents HTTP 细节在 `TASK_EXECUTION_SERVICE.md` 和本文件维护。
- 本文件是 agents 当前架构入口；外部归档中的历史草稿不作为新增设计或接口判断依据。
- 历史 output、benchmark result、cache 和 archived source 只作为证据，不能反向
  定义当前接口。
- 新增 endpoint、failure code 或 payload 字段时，必须同步契约测试和本文档。

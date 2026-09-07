# EMAS MAP-THOR Benchmark

> 系统当前的权威架构与 Planning/Allocation 契约见
> [`../ARCHITECTURE.md`](../ARCHITECTURE.md)。历史 benchmark 产物只用于复现，
> 不定义当前接口。

这个目录包含 EMAS 与 MAP-THOR 的 benchmark 集成。当前推荐使用
`task_service` full-system 模式评估完整项目，而不是只测试 direct
AI2-THOR adapter。

## 系统连接方式

Full-system benchmark 的调用链如下：

```text
benchmark.batch_runner
  -> MAP-THOR official initializer
  -> AI2-THOR Controller + official checker
  -> benchmark_hybrid_decision_loop
  -> Planning: 完整、不可变的 semantic Task Graph
  -> Allocation: 基于完整 Task Graph、progress、agent state 和 scene context 生成 Execution Plan
  -> Hybrid 运行时调度循环：只执行 Execution Plan 的 units[0]，观测后重新规划下一轮
  -> task_execution_server.py /execute_task
  -> ai2thor_receiver_server.py /execute_actions
  -> 同一个 AI2-THOR Controller
  -> action_log.jsonl
  -> MAP-THOR official checker metrics
```

关键行为：

- 每轮 Allocation 接收完整 active Task Graph；Hybrid 不再只传 ready task，
  也不会在分配前清空 `depends_on`。
- Execution Plan 可以增加仅供本轮执行的 resource/handoff 依赖，但不得修改
  semantic Task Graph；每轮只提交首个非空 unit。
- `blocked` 或模型计划校验失败属于 pre-dispatch Planning replan 输入，不能用
  空 assignments、round-robin assignment 或通用 empty-first-unit 错误伪装。
- 不传 `--execution-mode task_service` 时，仍使用原来的 direct
  `ai2thor` adapter。
- `--execution-mode task_service --task-service-autostart` 会为每个
  episode 自动启动 receiver 和 task execution server。
- receiver、task execution service（内含 relay agent）和 official checker 绑定同一个 controller。
- benchmark 结束后自动清理 receiver 和 task execution service 进程。
- 不需要手动启动 `127.0.0.1:18080` 或 `127.0.0.1:19000`。
- 自动模式默认使用 episode-local 端口 `18090` 和 `19010`；批量运行时会
  根据 port base 管理端口。
- EMAS 的 `all_done` 只作为内部状态记录，不会覆盖 MAP-THOR official
  checker 的成功判定。

## MAP-THOR 和 Unity 资源

`benchmark/mapthor_assets` 保存 MAP-THOR 官方 configs、floorplan
initializers、task checker 和基线工具，来源为 `nsidn98/LLaMAR`，许可证为
MIT。可通过 `--mapthor-root` 切换到其他官方 checkout。

服务器端 Unity 使用 AI2-THOR CloudRendering。当前项目已经复用解压后的
build，不需要重新安装 Unity 环境：

```text
ai2thor/releases/
  thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917/
```

benchmark 默认 commit 为：

```text
f0825767cd50d69f666c7f282e54abfe58f1e917
```

运行时会将项目内 release 映射到 `/tmp/emas_ai2thor/releases`，AI2-THOR
后续从该缓存查找可执行文件。除非显式传入 `--commit-id`、`--branch`、
`--local-build` 或 `--local-executable-path`，否则不需要每次重新查找
`thor-CloudRendering`。

## 环境检查

先检查 GPU 和 Vulkan：

```bash
nvidia-smi

env -u __EGL_VENDOR_LIBRARY_DIRS \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
vulkaninfo --summary
```

`vulkaninfo --summary` 应显示 NVIDIA GPU。若系统没有 Vulkan 工具或运行库：

```bash
sudo apt update
sudo apt install -y \
  vulkan-tools \
  libvulkan1 \
  mesa-vulkan-drivers \
  libglvnd0 \
  libegl1 \
  libglx0 \
  libopengl0
```

安装 Mesa 运行库不能代替 NVIDIA Vulkan ICD。CloudRendering 仍要求
`/etc/vulkan/icd.d/nvidia_icd.json` 及与宿主机驱动匹配的 NVIDIA library。

当前服务器必须同时：

1. 清除 Conda 激活时残留的 `__EGL_VENDOR_LIBRARY_DIRS`。
2. 将 `LD_LIBRARY_PATH` 和 `EMAS_LD_LIBRARY_PATH` 指向系统 GL/Vulkan
   library。
3. 不要把 `conceptgraph/lib` 放到 `LD_LIBRARY_PATH` 最前面。

## 单次完整运行

下面是当前服务器已经能够启动 Unity、完成 ConceptGraphs、自动启动 task execution service
并跑完 episode 的完整命令：

```bash
cd /225010231/mwl/EMAS

env -u __EGL_VENDOR_LIBRARY_DIRS 
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES=0,1 \
EMAS_PLANNING_DEVICE=cuda:1 \
EMAS_PLANNING_GPU_MAX_MEMORY=34GiB \
EMAS_AGENTS_CUDA_VISIBLE_DEVICES=1 \
GSA_PATH=/225010231/mwl/EMAS/memory/Grounded-Segment-Anything \
/225010231/miniconda3/envs/conceptgraph/bin/python \
-m benchmark.batch_runner \
  --rerun \
  --limit 10 \
  --mapthor-task-ids 1 \
  --mapthor-floorplan-indices 0 \
  --benchmark-seeds 0 \
  --agentnum 2 \
  --platform cloud \
  --class-set scene \
  --execution-mode task_service \
  --task-service-autostart \
  --task-service-python /225010231/mwl/Linhao/.homes/userlh/.conda/envs/qwen35/bin/python \
  --task-service-model-path /225010231/mwl/Linhao/models/Qwen3.5-4B \
  --task-service-device cuda \
  --qwen-model-path /225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
  --planning-model-path /225010231/mwl/EMAS/Qwen3.5-9B \
  --qwen-num-gpus 1 \
  --max-task-retries 6\
  --output-dir benchmark/results/emas_full_project_test
```

参数说明：

- `--rerun`：即使 episode 已存在也重新执行。
- `--limit 1`：只执行筛选结果中的第一个 episode。
- `--mapthor-task-ids 1`：选择 MAP-THOR task 1。
- `--mapthor-floorplan-indices 0`：选择该任务配置中的第一个可用 floorplan。
- `--benchmark-seeds 0`：使用 seed 0。
- `--agentnum 2`：使用两个 agent。
- `--task-service-autostart`：自动启动 receiver 和 task execution service，不依赖手动服务。
- `--task-service-python`：启动 task execution service 使用的 Python。
- `--task-service-model-path`：task execution service 内部 relay agent 使用的 Qwen3.5 模型。
- `--task-service-cuda-visible-devices`：只对 agents task execution service 子进程可见的物理
  GPU，默认是 `1`；子进程内部仍将它表示为 `cuda:0`。
- `--qwen-model-path`：scene graph 使用的 Qwen2.5-VL。
- `--planning-model-path`：semantic Task Graph、Execution Plan 和技能规划使用的 Qwen3.5-9B。

`benchmark.batch_runner` 主进程默认可见物理 GPU 0 和 1：memory 使用
`cuda:0`，planning 模型通过 `EMAS_PLANNING_DEVICE=cuda:1` 固定到物理 GPU 1；
自动启动的 agents task execution service 子进程也默认只看物理 GPU 1。可通过
`EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES` 和
`EMAS_PLANNING_DEVICE`、`EMAS_AGENTS_CUDA_VISIBLE_DEVICES` 覆盖。GPU 分离需要使用
`--execution-mode task_service --task-service-autostart`；旧的 direct
`ai2thor` 模式没有独立的 agents 模型进程。

Qwen3.5-9B 可在当前 48 GB GPU 上运行。batch runner 默认通过
`EMAS_PLANNING_GPU_MAX_MEMORY=34GiB` 限制 planning 在 GPU 1 上的占用，并为
agents 模型预留显存。可用
`EMAS_PLANNING_CPU_MAX_MEMORY` 调整 CPU offload 上限。

建议每次正式实验使用新的 `--output-dir`。`--rerun` 会覆盖 episode 内结果；
每次 batch runner 启动时也会重建 `manifest.jsonl`，使它只包含当前批次记录并与
`summary.json` 对应。

## 批量完整运行

先用少量任务做 full-system regression：

```bash
cd /225010231/mwl/EMAS

env -u __EGL_VENDOR_LIBRARY_DIRS 
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES=0,1 \
EMAS_PLANNING_DEVICE=cuda:1 \
EMAS_PLANNING_GPU_MAX_MEMORY=34GiB \
EMAS_AGENTS_CUDA_VISIBLE_DEVICES=1 \
GSA_PATH=/225010231/mwl/EMAS/memory/Grounded-Segment-Anything \
/225010231/miniconda3/envs/conceptgraph/bin/python \
-m benchmark.batch_runner \
  --rerun \
  --mapthor-task-ids 1,10,16 \
  --mapthor-floorplan-indices 0 \
  --benchmark-seeds 0 \
  --agentnum 2 \
  --platform cloud \
  --class-set scene \
  --execution-mode task_service \
  --task-service-autostart \
  --task-service-python /225010231/mwl/Linhao/.homes/userlh/.conda/envs/qwen35/bin/python \
  --task-service-model-path /225010231/mwl/Linhao/models/Qwen3.5-4B \
  --task-service-device cuda \
  --qwen-model-path /225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
  --planning-model-path /225010231/mwl/EMAS/Qwen3.5-9B \
  --qwen-num-gpus 1 \
  --max-task-retries 6\
  --output-dir benchmark/results/emas_full_project_regression
```

执行所有可运行任务在前五个适用 floorplan、seed 0 上的 full-system
benchmark：

```bash
cd /225010231/mwl/EMAS

env -u __EGL_VENDOR_LIBRARY_DIRS 
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES=0,1 \
EMAS_PLANNING_DEVICE=cuda:1 \
EMAS_PLANNING_GPU_MAX_MEMORY=34GiB \
EMAS_AGENTS_CUDA_VISIBLE_DEVICES=1 \
GSA_PATH=/225010231/mwl/EMAS/memory/Grounded-Segment-Anything \
/225010231/miniconda3/envs/conceptgraph/bin/python \
-m benchmark.batch_runner \
  --rerun \
  --mapthor-floorplan-indices 0,1,2,3,4 \
  --benchmark-seeds 0 \
  --agentnum 2 \
  --platform cloud \
  --class-set scene \
  --execution-mode task_service \
  --task-service-autostart \
  --task-service-python /225010231/mwl/Linhao/.homes/userlh/.conda/envs/qwen35/bin/python \
  --task-service-model-path /225010231/mwl/Linhao/models/Qwen3.5-4B \
  --task-service-device cuda \
  --qwen-model-path /225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
  --planning-model-path /225010231/mwl/EMAS/Qwen3.5-9B \
  --qwen-num-gpus 1 \
  --output-dir benchmark/results/emas_full_project_eval
```

还可使用：

- `--limit N`：限制 episode 数量。
- `--mapthor-task-categories 1,2`：按 ambiguity category 筛选。
- `--mapthor-task-ids 1,10,16`：按 official task ID 筛选。
- `--benchmark-seeds 0,1,2`：运行多个随机种子。
- `--task-service-port-base`：修改 task execution service 起始端口，默认 `18090`。
- `--receiver-port-base`：修改 receiver 起始端口，默认 `19010`。

- `--kitchen-only`：仅保留 AI2-THOR 标准厨房场景 `FloorPlan1`–`FloorPlan30`。

批处理默认启用 `--reuse-planning-model` 和 `--reuse-task-service`：planning
模型与 task execution service 模型在整个 batch 中各加载一次；每个 episode 只重启轻量的
receiver。scene graph VLM 仍按 episode 加载并释放，以便下一轮 Grounded-SAM
使用 GPU 0。需要逐 episode 加载时可传入对应的 `--no-reuse-*` 参数。

## Direct 模式

Direct 模式用于初始化检查或对照实验，不经过 task execution service。

只检查 MAP-THOR initializer、Unity 和 controller：

```bash
python -m benchmark.runner \
  --mapthor-task-id 1 \
  --mapthor-floorplan-index 0 \
  --agentnum 2 \
  --initialize-only \
  --output-dir benchmark/results/init_check
```

轻量 integration smoke：

```bash
python -m benchmark.runner \
  --mapthor-task-id 1 \
  --mapthor-floorplan-index 0 \
  --agentnum 2 \
  --platform cloud \
  --disable-qwen \
  --disable-skill-qwen \
  --scenegraph-dry-run \
  --output-dir benchmark/results/smoke
```

## 最新单次运行结果

运行 episode：

```text
type1_task1_1_put_bread_lettuce_tomato_fridge_FloorPlan1_seed0_agents2
```

原始指令：

```text
Put the bread, lettuce, and tomato in the fridge
```

最新 episode 结果：

```json
{
  "runtime_error": null,
  "success": 0,
  "transport_rate": 0.0,
  "coverage": 0.0,
  "balance": 0.0,
  "steps": 30,
  "timeout": 30,
  "low_level_actions": 58,
  "successful_interactions": 0,
  "termination_reason": "timeout",
  "internal_all_done": false
}
```

这表示 full-system 技术链路已经跑通，并非进程崩溃：

- Unity CloudRendering 成功启动。
- ConceptGraphs 完成并生成 7 个节点、0 条边。
- receiver 自动启动在 `127.0.0.1:19010`。
- task execution server 自动启动在 `127.0.0.1:18090`。
- task service 请求进入同一个 official checker controller。
- episode 完整运行 30 个 high-level step 后结束。

但是任务没有取得有效进展：

- 58 条低层 action 全部为 `Pass`。
- relay 返回 29 次 `target_not_visible`。
- relay 返回 1 次 `task_normalization_failed`。
- 没有成功的 pickup、put、open 或 close interaction。
- official checker 的所有 required subtask 均未完成。

结果以 episode 内以下文件为准：

```text
benchmark/results/emas_full_project_test/
  type1_task1_1_put_bread_lettuce_tomato_fridge_FloorPlan1_seed0_agents2/
    benchmark_result.json
    benchmark_evaluation.json
    initial_planning/task_graph.json
    action_log.jsonl
    run_record.json
    services/receiver.log
    services/task_execution_server.log
```

`manifest.jsonl` 只描述当前 batch runner 进程已经处理的 episode；当前 episode
的完整结果仍以其目录中的 `benchmark_result.json` 为准。

## 当前失败原因

当前失败发生在 relay 执行动作之前，主要是 task planning、grounding 和 retry
之间的语义没有对齐。

### 1. 复杂任务退化为 heuristic fallback

成功的 standalone hybrid 样例使用简单指令 `go to cabinet.`，其
`planner_backend` 为 `qwen_local`，生成明确的 `navigate cabinet` 任务，并且
Cabinet 在当前 agent 视野内，所以 relay 的 `goto` 能成功。

benchmark 指令包含 bread、lettuce、tomato 和 fridge，但本地 Qwen task
planner 没有产生可解析计划，随后静默切换为：

```json
{
  "planner_backend": "heuristic_fallback",
  "flat_tasks": [
    {
      "id": "T1",
      "name": "Place the object at destination",
      "action": "place",
      "object_tags": ["loaf of bread"]
    }
  ]
}
```

fallback 虽然在 `task_spec.target_objects` 中识别出四个实体，但最终 executable
task graph 丢失了 lettuce、tomato 和 fridge，只剩一个泛化的 place task。
当前 `planning/task_graph.py` 捕获 planner exception 或 JSON parse 失败后直接
返回 `None`，没有保存异常和原始模型响应，所以还不能从产物区分具体是哪一种
planner 失败。

### 2. ConceptGraph grounded 不等于 AI2-THOR 可执行

ConceptGraphs 从初始 RGB 图像中生成了 `loaf of bread` 节点，Hybrid 因此将
T1 标记为 `grounded`，没有插入 `find bread`。

但 ConceptGraph 节点只有视觉语义和 3D bbox，没有当前 controller 中可用的
AI2-THOR `objectId`。执行时两个 agent 的 observation 都看不到 Bread，所以
relay 正确返回：

```text
target_not_visible: 'Bread' is not visible to any successfully observed robot
```

这说明当前系统把“在视觉场景图中语义匹配成功”和“Unity 中当前可见且可交互”
错误地当成了同一种 grounding。

### 3. Relay 收到的任务信息已经不完整

relay 实际收到的是：

```text
Place the object at destination
Place or insert the object at the destination supported by the subgraph.
place loaf of bread
```

请求中没有 lettuce、tomato、fridge，也没有完整的
`find -> navigate -> pickup -> open -> put -> close` 依赖链。relay normalizer
只能将其解释为对 Bread 的 pickup/place 意图，然后因为 Bread 不可见而停止。

因此本次零分不能单独归因于 relay。relay 的保守可见性检查是合理的，但它当前
也不会自动将这个不完整的 place 请求扩展为全局搜索和完整 transport 计划。

### 4. Retry 计数没有累积

relay 每次返回：

```json
{
  "status": "wait_retry",
  "retry_count_before": 0,
  "retry_count_after": 1
}
```

Hybrid 的 `apply_progress()` 直接采用 relay 报告中的
`retry_count_after: 1`，导致本地 retry count 每轮都被重置为 1。相同 T1
因此重复执行 30 轮，不会达到最大重试次数，也不会触发失败或重规划，最终只能
由 MAP-THOR timeout 终止。

## 已修复问题

### EGL/Vulkan library 污染

历史错误：

```text
vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER
Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'
CloudRendering is selected, but Vulkan did not expose a hardware GPU
```

原因是激活 `conceptgraph` 环境后：

- `__EGL_VENDOR_LIBRARY_DIRS` 指向了不匹配的 EGL vendor。
- `LD_LIBRARY_PATH` 优先加载 Conda 中与宿主 NVIDIA driver 不匹配的库。

修复方式已经固化在推荐运行命令中：

```bash
env -u __EGL_VENDOR_LIBRARY_DIRS \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
EMAS_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
...
```

只执行 `env -u LD_LIBRARY_PATH` 或只设置 `EMAS_LD_LIBRARY_PATH` 不足以修复
当前服务器环境。

### ConceptGraphs relation 文件类型错误

ConceptGraphs 同时输出：

- `relations`：binary pickle graph edges。
- `object_relations`：标准 JSON relation records。

旧代码把 `relations` 当 JSON 读取，会在 scene graph 增量更新时失败。当前
Hybrid 已优先使用 `object_relations`，仅在它不存在时回退到 `relations`。
对应 benchmark 和 relay 测试已经覆盖真实 pickle 内容。

### 本地 CloudRendering 路径复用

CloudRendering build 已移动并解压到 `ai2thor/releases` 下。benchmark 会自动
复用这个项目内 release，不需要每次指定路径，也不需要重新下载或安装 Unity。

## 待修复方案

建议按以下顺序修改和验证：

1. **修复复合 transport fallback。** 对每个源对象生成独立的
   `find/navigate/pickup/navigate/open/put` 任务，并保留 destination fridge
   及依赖关系，最后关闭 fridge。
2. **增加端到端回归。** 至少覆盖一个简单 navigation、一个单物体 transport
   和 task 1 三物体 transport，并断言 action log 中出现非 Pass action。

完成修复后的最低验收条件：

- `planner_backend` 为 `qwen_local`，或 fallback 保留全部任务实体。
- task graph 同时包含 Bread、Lettuce、Tomato 和 Fridge。
- 不可见对象会产生 find/search，而不是直接 place。
- retry count 为 `1, 2, 3, ...`，不会一直停留在 1。
- `action_log.jsonl` 中出现成功的导航和 interaction action。
- `runtime_error` 为 `null`，且 episode 不再因无进展 timeout。

## Downstream runtime recovery 修复

在不修改 Planning 输出的前提下，Hybrid/Relay 后半段已经增加：

- task service health 会同时验证 backend、receiver 和 controller/robots 就绪；managed
  benchmark 只有在返回 `status=ready` 后才开始 episode。
- Hybrid 向 task execution service 发送兼容的原 `task` 字符串，同时附带 `parent_task`、结构化
  `subtask` 和压缩后的 scene context。
- task service 响应的 `failure_code`、`recoverable` 和 `recommended_recovery` 会进入
  Hybrid task status。
- `target_not_visible`、`object_not_actionable` 和
  `missing_required_state` 会覆盖纯语义 scene-graph match，使下一轮插入 runtime
  find task。
- find 成功返回的 goto/discovered object 会带 `objectId` 写回 Memory，并解除
  原 interaction task 的 runtime-unresolved 状态。
- retry count 由 Hybrid 单调维护，达到 `--max-task-retries` 后明确失败，不再被
  relay 的单轮计数重置。
- 每次 benchmark episode 启动前清理该 episode 上一次 attempt 的 loops、
  runtime、service logs 和 final result，避免中断重跑混合产物。

这些修改不恢复初始 task graph 中已经丢失的 Lettuce、Tomato 或 Fridge
任务节点；复合 transport planner/fallback 仍需单独修复。

Planning 另已增加纯诊断字段，不改变 Qwen/fallback 选择、任务节点、依赖或
grounding：

- `planner_diagnostics.qwen` 保存 import/model-call exception、最多 8000 字符的
  raw response preview 和 JSON parse 状态。
- `planner_diagnostics.entity_coverage` 对比原始 `task_spec` 实体与最终 task
  graph，报告 covered/missing entities。
- entity coverage 当前为 audit-only（`enforced=false`），不会拒绝或改写已有
  planner 输出。

## 当前实现与验证

Full-system benchmark 的主要实现文件：

- `scripts/benchmark_hybrid_decision_loop.py`：MAP-THOR controller/checker、
  CloudRendering release 复用，以及 episode-local 服务生命周期。
- `scripts/hybrid_decision_loop.py`：共享 planning/memory loop、task service adapter、
  progress 和 scene graph update。
- `benchmark/action_log.py`：将 receiver 执行的低层 action 写入 official
  checker 可消费的 action log。
- `agents/ai2thor_receiver_server.py`：在 benchmark 创建的同一个 controller
  上执行 relay agent 生成的 actions。
- `benchmark/tests/test_full_system_task_service.py`：full-system managed service
  集成测试。
- `scripts/tests/test_hybrid_task_service.py`：task service adapter、relation path 和
  Hybrid 状态测试。

最新本地验证：

```text
agents + scripts + benchmark: 274 passed, 20 subtests passed
```

验证范围：

```bash
/225010231/miniconda3/envs/conceptgraph/bin/python \
  -m pytest -q \
  benchmark/tests \
  scripts/tests/test_hybrid_task_service.py

/225010231/miniconda3/envs/conceptgraph/bin/python \
  -m py_compile \
  scripts/benchmark_hybrid_decision_loop.py \
  scripts/hybrid_decision_loop.py \
  benchmark/action_log.py \
  agents/ai2thor_receiver_server.py
```

这些测试证明 benchmark/task-service wiring 和已修复的 relation 文件选择逻辑可以
工作，但不代表当前 task 1 已获得成功指标。task 1 仍受前述 planner、
actionable grounding 和 retry 问题影响。

## 输出与指标

每个 episode 包含：

- `episode.json`：任务、floorplan、seed 和 timeout。
- `initial_object_state.json` / `final_object_state.json`：Unity 物体状态。
- `initial_planning/task_graph.json`：初始任务图和 planner backend。
- `loops/loop_NNN/`：每轮分配、通信、状态、scene graph 和 execution trace。
- `loops/loop_NNN/allocation_request.json`：送入 Allocation 的完整 semantic
  Task Graph、graph version/fingerprint、progress、agent states、remaining task IDs
  和合并后的 scene context。该文件用于确认调用方没有裁成 ready-only graph。
- `loops/loop_NNN/execution_plan.json`：经过验证的 Execution Plan，包括 `state`、
  `units`、execution-only dependency edges、blocking 信息和 diagnostics。只有
  `units[0]` 会在该轮下发，其余 unit 是下一次观测前的临时投影。
- `loops/loop_NNN/allocation_blocked.json`：仅在 Allocation 返回 `blocked` 或
  `invalid_model_plan` 时产生，记录 reason code、受影响任务、inventory/resource
  冲突、建议恢复动作以及 pre-dispatch replan 结果；它不能包含伪造 assignment。
- `action_log.jsonl`：进入 official checker 的真实 AI2-THOR action。
- `services/managed_services.json`：receiver/task-service URL、端口、命令和日志路径。
- `benchmark_evaluation.json`：official metrics 和终止原因。
- `benchmark_result.json`：episode、evaluation 和 hybrid record 汇总。

批量目录额外包含：

- `manifest.jsonl`：当前批次每完成一个 episode 追加一行；新批次启动时重建。
- `summary.json`：本次进程选中 episode 的聚合指标。

汇总指标包括 Success Rate、Transport Rate、Coverage、Balance、Steps 和
95% confidence interval。Success Rate 在 SciPy 可用时使用
Clopper-Pearson，否则使用 Wilson interval。

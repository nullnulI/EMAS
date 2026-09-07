本项目由 planning、allocation、memory、agents，以及
`scripts/hybrid_decision_loop.py` 运行时调度循环组成。

> 当前运行架构、模块边界和 JSON 契约以 [`ARCHITECTURE.md`](ARCHITECTURE.md)
> 为唯一权威说明。`benchmark/results/`、恢复快照和历史设计稿仅用于诊断，不能
> 作为当前接口依据。

planning 的功能包括：
    1. Generate query program
    2. Generate immutable semantic Task Graph
    3. Generate a resource-aware, independent Execution Plan
memory的功能包括:
    1. Scene graph generation(conceptgraphs)
    2. Query and retrieve
    3. Modify the graph
    所得场景图存储文件置于database中
agents的功能包括：
    1. Feasibility judgement and negotiation (EmbodiedGPT)
    2. Communication (MCP & A2A)
    3. Negotiation judgement (LLM)
    4. Restriction conclusion (LLM)
    5. Coordinator 从 receiver 自主发现全部 Agent、刷新实时状态并执行局部恢复
    6. Agent node generation and sending

环境安装指引：
mkdir -p ~/envs/conceptgraph
tar -xzf conceptgraph_env.tar.gz -C ~/envs/conceptgraph
conda config --add envs_dirs /225010231/miniconda3/envs
/home/kinova-1/anaconda3/envs/conceptgraph/bin/conda-unpack

尝试激活环境
conda activate conceptgraph
然后
python -m pip install -e /new/path/EMAS/ai2thor
python -m pip install -e /new/path/EMAS/memory/concept-graphs
python -m pip install --no-build-isolation -e /new/path/EMAS/memory/Grounded-Segment-Anything/GroundingDINO
python -m pip install -e recognize-anything
python -m pip install -e segment_anything

环境变量：
export ROOT_DIR=/home/kinova-1/EMAS/225010231/mwl/EMAS
export AI2THOR_ROOT="$ROOT_DIR/memory/database"
export AI2THOR_CONFIG_PATH="$ROOT_DIR/memory/concept-graphs/conceptgraph/dataset/dataconfigs/ai2thor/ai2thor.yaml"
export GSA_PATH="$ROOT_DIR/memory/Grounded-Segment-Anything"
export PYTHONPATH="$ROOT_DIRmemory/Grounded-Segment-Anything/segment_anything:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/memory/Grounded-Segment-Anything:$PYTHONPATH$



## Planning/Memory 调用 Agents 的推荐模式

当前推荐入口是 `scripts/hybrid_decision_loop.py --execution-mode task_service`。
Planning 生成语义 Task Graph，Allocation 根据完整剩余图与实时资源生成 Execution
Plan，Hybrid 运行时调度循环每轮只执行首个 unit 并重新观测；agents 只负责执行当前分配给
某个 robot 的 subtask，并通过 HTTP 返回结构化执行结果。

调用链如下：

```text
memory scene graph
  -> planning extract_subgraph / task_graph / task_allocation
  -> scripts/hybrid_decision_loop.py --execution-mode task_service
  -> agents/task_execution_server.py POST /execute_task
  -> agents/ai2thor_receiver_server.py /observe /state /execute_actions /goto
  -> communication_inbox.json + task_statuses + run_record.json
```

先在 agents 侧启动 receiver 和 task execution service：

```bash
cd /225010231/mwl/EMAS/agents
python ai2thor_receiver_server.py --scene FloorPlan1 --robots 2 --port 19000 --no-show

bash run_task_execution_server.sh \
  --receiver-url http://127.0.0.1:19000 \
  --model-path models/Qwen3.5-4B \
  --port 18080
```

然后从 EMAS 根目录运行 hybrid loop：

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

`task_service` 会把每个 allocation assignment 映射成一次最小
`/execute_task` 请求：`task_id`/`task` 来自当前 subtask，`primary_robot_id` 来自
assignment 的 `agent_id`，还可携带该 subtask 自身的结构化约束。Hybrid/Allocation
不会下传 `known_robot_ids`、候选 Agent 列表、完整 Task Graph 或 Execution Plan。
Coordinator 每次直接从 receiver 发现全部 Agent 并刷新其实时状态；若局部恢复时换了
机器人，则通过 `completion_agent_id` 回报实际 executor。失败时返回 `WAIT_RETRY`
而不是让 hybrid loop 崩溃。旧的 `adapter` 和 `ai2thor` 模式仍然保留。

全流程运行：
用已有场景图跑 adapter 流程示例：
python EMAS/scripts/hybrid_decision_loop.py \
  --scene_name train_3 \
  --task "take the spoon into the bowl and put the bowl into the fridge" \
  --scenegraph-info /path/to/scenegraph_info.json \
  --execution-mode adapter \
  --qwen-model-path /225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
  --planning-model-path /225010231/mwl/EMAS/Qwen3.5-9B \
  --max-task-loops 10
scenegraph_info.json 需要类似这样：
{
  "scene_graph": "/path/to/scene_graph.json",
  "relations": "/path/to/cfslam_object_relations.json",
  "object_relations": "/path/to/cfslam_object_relations.json"
}

真实 AI2-THOR 执行可以这样：
DISPLAY=:1 python EMAS/scripts/hybrid_decision_loop.py \
  --scene_name train_3 \
  --task "take the spoon into the bowl and put the bowl into the fridge" \
  --execution-mode ai2thor \
  --x-display :1 \
  --platform default \
  --class-set scene \
  --disable-qwen

启动SG构建程序：（从ai2thor返回的观测数据到构建完毕的场景图）
python conceptgraph/scripts/run_full_scenegraph_pipeline.py \
    --dataset_root /225010231/mwl/EMAS/memory/database \
    --dataset_config /225010231/mwl/EMAS/memory/concept-graphs/conceptgraph/dataset/dataconfigs/ai2thor/ai2thor.yaml \
    --scene_id train_3 \
    --stride 5 \
    --class_set ram \
    --add_bg_classes \
    --accumu_classes \
    --gsa_exp_suffix withbg_allclasses \
    --sim_threshold 1.2 \
    --mask_conf_threshold 0.25 \
    --cfslam_save_suffix overlap_maskconf0.25_simsum1.2_dbscan.1 \
    --vlm_backend qwen \
    --vlm_model_path /225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
    --skip_existing

子图提取程序：
python /225010231/mwl/EMAS/planning/extract_subgraph.py \
  --cachedir /225010231/mwl/EMAS/memory/database/train_3/sg_cache \
  --mode task \
  --task "find the chair near the shelf"
也可以通过大模型进行匹配：
python /225010231/mwl/EMAS/planning/extract_subgraph.py \
  --cachedir /225010231/mwl/EMAS/memory/database/train_3/sg_cache \
  --mode task \
  --task "find the chair near the shelf" \
  --use-qwen \
  --qwen-model-path /225010231/mwl/EMAS/Qwen3.5-9B

任务分解：
python /225010231/mwl/EMAS/planning/task_graph.py \
  --task "find the chair near the shelf and inspect it" \
  --subgraph /225010231/mwl/EMAS/memory/database/train_3/sg_cache/task_relevant_subgraph.json \
  --qwen-model-path /225010231/mwl/EMAS/Qwen3.5-9B

数据生成：
python planning/datagen/generation.py \
  --scene_name train_3 \
  --task "take the spoon into the bowl and put the bowl into the fridge" \
  --episode 1 \
  --agentnum 2 \
  --platform cloud \
  --save-agent-video \
  --qwen-model-path /home/kinova-1/EMAS/225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
  --planning-model-path /home/kinova-1/EMAS/225010231/mwl/EMAS/Qwen3.5-9B \
  --output-dir /home/kinova-1/EMAS/225010231/mwl/EMAS/dataset
低显存指令:
python planning/datagen/generation.py \
  --scene_name train_3 \
  --task "take the spoon into the bowl and put the bowl into the fridge" \
  --episode 1 \
  --agentnum 2 \
  --platform cloud \
  --class-set scene \
  --global-map-samples 2 \
  --scenegraph-stride 2 \
  --log-memory \
  --save-agent-video \
  --qwen-model-path /home/kinova-1/EMAS/225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct \
  --planning-model-path /home/kinova-1/EMAS/225010231/mwl/EMAS/Qwen3.5-9B \
  --output-dir /home/kinova-1/EMAS/225010231/mwl/EMAS/dataset

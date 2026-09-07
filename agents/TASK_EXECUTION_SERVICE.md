# Relay Task Service

`task_execution_server.py` is the task-level entry point for this repository. It
wraps the existing `EmbodiedGPT_Pytorch` relay closed-loop runtime and sends
all simulator actions to `ai2thor_receiver_server.py`.

```text
client
  | POST /execute_task
  v
task_execution_server.py
  | task intent -> relay tool calls -> executor-view plan -> grounding
  v
ai2thor_receiver_server.py
  | /observe, /state, /execute_actions
  v
AI2-THOR shared multi-robot scene
```

The service always enables `--relay-mode --closed-loop-replan`. It does not
use the removed one-step Qwen action service.

For every request, the Coordinator discovers all robots from the receiver's
authoritative state and refreshes their observations before executor selection.
When the primary robot cannot execute the current step, the relay model receives all
first-person images, relevant metadata, and the complete candidate validation
table in its first decision input; it then selects an executor or reports a
validated failure. Callers do not supply the candidate/known robot set.

## Start

Start the receiver first:

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 --robots 2 --port 19000 --no-show
```

Then start the task execution service. One process owns one Qwen model instance.

```bash
bash run_task_execution_server.sh \
  --receiver-url http://127.0.0.1:19000 \
  --model-path models/Qwen3.5-4B \
  --port 18080 \
  --device cuda \
  --device-map auto \
  --dtype float16 \
  --max-new-tokens 128
```

Use `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` before this command
when the model must be pinned to one physical GPU.

The launcher clears ROS `PYTHONPATH` and gives Conda's `libstdc++` precedence.
This is required on this workstation because the ROS library path otherwise
breaks the Transformers import chain through `sklearn` and `pyarrow`.

## HTTP API

`GET /health` returns service configuration and whether the model has been
loaded. The model is loaded lazily by the first task.

The service saves each semantic-planning model response as
`output/task_execution/<task_id>_qwen_raw.txt` before validation and grounding.

`POST /execute_task` accepts:

```json
{
  "task_id": "bread-001",
  "task": "pick up the bread",
  "primary_robot_id": 0,
  "dry_run": false,
  "max_replan_steps": 10,
  "relay_agent_max_turns": 8,
  "max_actions": 8,
  "relay_strategy": "agent"
}
```

`task` may also be supplied as `instruction` or `prompt`. `relay_strategy`
can be `agent` for Qwen tool-calling coordination or `rules` for the
deterministic candidate selector. `dry_run=true` performs observation,
planning, grounding, and relay selection without changing the scene.

The minimal EMAS request is `task_id`, `task`, and `primary_robot_id`.
`subtask`, compact task-local scene hints, and local execution limits are optional.
The request does not contain `known_robot_ids`, `eligible_agent_ids`, Agent-state
snapshots, the semantic Task Graph, or the complete Execution Plan. The Coordinator
derives its internal discovered/known set from receiver state on every request.

Example:

```bash
curl -X POST http://127.0.0.1:18080/execute_task \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "bread-001",
    "task": "pick up the bread",
    "primary_robot_id": 0,
    "dry_run": false
  }'
```

The response contains `completion_agent_id` for the actual successful executor,
`result.closed_loop_trace` (including `executor_robot_id`), each grounded action
payload, executor-selection evidence, and `closed_loop_result`. Internal
`known_robot_ids`/discovery fields, when present in diagnostics, are receiver-derived
observations rather than echoed routing instructions. A logical inability to
complete the task returns `status: "needs_upstream_planning"` with a structured
reason; HTTP `4xx` and `5xx` are reserved for malformed requests and runtime
failures.

## EMAS integration

The current EMAS integration does not use a standalone bridge script. Instead,
`/225010231/mwl/EMAS/scripts/hybrid_decision_loop.py` has a `task_service`
execution mode. The hybrid loop keeps ownership of task graph progress,
allocation, communication files, and memory update; this service only executes
the allocated subtask and returns a structured task-level result.

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

For each allocation assignment, the hybrid loop sends one `/execute_task`
request. The request uses the allocation `agent_id` as `primary_robot_id`, the
subtask text as `task`, and may include only that subtask's structured constraints.
It does not forward the Allocation/Hybrid Agent list or full graph/plan. The
Coordinator refreshes all robots directly from the receiver and may locally choose a
non-primary executor; its `completion_agent_id` is recorded as the actual executor.
A successful relay closed-loop result is mapped to EMAS `SUCCESS`; structured task
failures, HTTP errors, timeouts, and invalid JSON are mapped to `WAIT_RETRY` and stored
in `communication_inbox.json`.

For the complete relay decision model, see `relay_closed_loop_design_cn.md`.

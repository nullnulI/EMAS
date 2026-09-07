# EMAS Runtime Architecture

This document is the authoritative description of the active EMAS runtime. Historical
benchmark artifacts, recovery snapshots, and archived source are evidence only; they do
not define current interfaces.

## Active runtime

```text
User task
  -> Planning: immutable semantic Task Graph
  -> Allocation: resource-aware Execution Plan
  -> Hybrid runtime: execute the first unit
  -> Coordinator: refresh all Agent state from the receiver, then execute/recover
  -> observation, progress, and scene update
  -> Allocation again, or Planning replan when structurally blocked
```

- `planning/task_graph.py` owns semantic tasks, `depends_on`, dependency edges, and
  semantic chain views.
- `planning/task_allocation.py` owns the independent Execution Plan. It may derive
  execution-only dependencies but must never mutate the semantic Task Graph.
- `scripts/hybrid_decision_loop.py` owns progress, graph versions, dispatch, recovery
  budgets, and the rolling `first_unit_then_replan` loop.
- `scripts/benchmark_hybrid_decision_loop.py` is the MAP-THOR adapter around the same
  Hybrid runtime. It is not a second runtime scheduler.
- `planning/datagen/generation.py` is a trajectory collector. It consumes the same
  Execution Plan contract and must not invent assignments when Allocation is blocked.

## Planning contract

Planning returns a semantic Task Graph containing `flat_tasks`, task-local
`depends_on`, `dependency_edges`, `root_task_ids`, and display-oriented `chains`.
Selector expansion may add provenance under `task.runtime.selector_expansion`; this
metadata records compiler operations but does not change the ownership of semantic
dependencies.

The Task Graph passed to Allocation is the complete active graph. Completion and
failure are supplied separately. Callers must not reduce it to a ready-only graph or
clear `depends_on` before allocation.

## Execution Plan contract

The canonical Allocation entry point is `build_execution_plan(...)`. Its result has:

```json
{
  "state": "dispatchable | blocked",
  "task_graph_version": 1,
  "graph_fingerprint": "sha256:...",
  "execution_policy": "first_unit_then_replan",
  "units": [
    {
      "time_step": 1,
      "assignments": [{"task_id": "T1", "agent_id": "0"}]
    }
  ],
  "dependency_edges": [
    {
      "from_task_id": "T1",
      "to_task_id": "T2",
      "kind": "semantic | object_handoff | agent_sequence | resource_mutex"
    }
  ],
  "blocking": null,
  "diagnostics": {}
}
```

A `dispatchable` plan covers every remaining task exactly once and has a non-empty
first unit. Empty `units` are legal only for `blocked`, which must include a reason
code, affected tasks, current inventories, conflicts, and
`recommended_recovery: replan_task_graph`.

Execution dependencies are ephemeral. In particular, compiler-proven aggregate
Pick/Place barriers may become exact-object `Pick_i -> Place_i` handoff edges, while
the input Task Graph remains byte-for-byte unchanged. Tasks that interact with the
same receptacle are serialized in the Execution Plan.

The `agent_id` selected by Allocation becomes `primary_robot_id` at the task-service
boundary: it is the initial executor assignment, not a list of every robot the
Coordinator may use for local recovery. The Hybrid runtime does not send
`known_robot_ids`, `eligible_agent_ids`, Agent-state snapshots, the semantic Task
Graph, or the complete Execution Plan to the task service. The minimal execution
request contains the task identity/content, its structured task-local constraints,
and `primary_robot_id`.

The Coordinator discovers the complete Agent set from the authoritative receiver and
refreshes pose, visibility, reachability, inventory, capability evidence, and recent
action status for every Agent on each request. Names such as `known_robot_ids` or
`discovered_robot_ids` may appear inside Coordinator diagnostics, but they describe
that receiver-derived internal set and are not an upstream input contract. If local
recovery selects another robot, the service reports the actual executor as
`completion_agent_id` (with `executor_robot_id` in its detailed trace) so runtime
progress and Memory reflect what really happened.

## Runtime state machine

1. Reconcile already-satisfied tasks and obtain current agent states.
2. Build one Allocation request from the full Task Graph, progress, and scene context.
3. Validate the complete model-produced schedule against semantic dependencies,
   skills, inventory transitions, object ownership, and shared-resource exclusion.
4. Send each task in the first unit with its Allocation-selected primary Agent and
   task-local constraints. The Coordinator independently refreshes all Agent state,
   executes or locally recovers the task, and reports the actual executor.
5. Observe again, discard the remaining projection, and rebuild Allocation input from
   authoritative progress and scene state.
6. On a structural Allocation block or exhausted plan-validation retries, call
   Planning with a pre-dispatch failure context. Do not use `WAIT_RETRY`.
7. `WAIT_RETRY` is reserved for execution attempts where an external-state change may
   make the same task executable later.

The default recovery budgets remain three Task Graph replans per episode and one per
graph-version/source fingerprint. Model initialization or transport failure is an
infrastructure error and is not sent back to the same unavailable model.

## Source-of-truth rules

- Do not infer current architecture from `benchmark/results/`, external archives,
  `.orig` files, caches, or generated run directories.
- Do not add compatibility copies of Planning or Allocation modules. Change the
  canonical module and all active callers together.
- Any new Execution Plan state or edge kind must be documented here and covered by a
  contract test before it is consumed by the Hybrid runtime.

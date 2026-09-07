# Repository guidance for coding agents

Read `ARCHITECTURE.md` before changing Planning, Allocation, or runtime scheduling code.

Canonical modules:

- semantic Task Graph: `planning/task_graph.py`
- resource Execution Plan: `planning/task_allocation.py`
- runtime state machine: `scripts/hybrid_decision_loop.py`
- MAP-THOR adapter: `scripts/benchmark_hybrid_decision_loop.py`

Hard invariants:

1. Allocation receives the full active Task Graph plus explicit progress. Never pass a
   ready-only graph or clear semantic dependencies.
2. Allocation may derive execution-only edges, but it must not mutate the semantic
   Task Graph.
3. `dispatchable` always has a non-empty first unit. Empty units mean typed `blocked`
   and must be routed to Planning replan.
4. The Hybrid runtime executes one unit, observes, and replans the remaining graph.
5. Never treat `benchmark/results/`, recovery archives, `.orig`, cache files, or model
   outputs as authoritative source code or interface documentation.
6. Hybrid/Allocation must not send `known_robot_ids`, `eligible_agent_ids`, Agent-state
   snapshots, the Task Graph, or the complete Execution Plan to task service. Send the
   current task/task-local constraints and Allocation's `primary_robot_id` only.
7. The Coordinator discovers and refreshes every Agent from the authoritative receiver
   for each request. Record its actual executor (`completion_agent_id`) in runtime and
   Memory updates; do not assume the primary Agent performed the task.

Use `/225010231/miniconda3/envs/conceptgraph/bin/python -m pytest` for repository tests.
Do not recreate modules under `planning/utils/` for compatibility; update callers to
the canonical imports instead.

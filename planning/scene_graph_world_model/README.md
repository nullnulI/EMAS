# Scene Graph World Model

This package implements a one-step recurrent state-space model for macro-time-step transitions:

```text
G_t --GraphEncoder--\
                       Fusion --> x_t --> q(z_t | h_t, x_t) --\
A_t --SetEncoder----/                                      |
                                                            +--> GRU --> h_(t+1) --> 3 prediction heads
U_t --AssignmentEncoder--> u_t -----------------------------/
                         p(z_t | h_t) ----------------------/
```

The three heads predict:

1. the next scene graph: dynamic node states, node existence, and a dense relation matrix;
2. status for every assignment: completion, non-negative duration, and collision;
3. the next state and existence of every agent.

## Temporal semantics

At training time, use the posterior `q(z_t | h_t, x_t)` and minimize `KL(q || p)` against `p(z_t | h_t)`. At observed online steps, also use the posterior. For open-loop imagination, use the prior by calling the model with `use_posterior=False`.

The recurrent update is exactly:

```text
h_(t+1) = GRU([z_t, u_t], h_t)
```

The prediction heads consume `h_(t+1)`, `z_t`, and `u_t` through a fused transition context.

## Data contract

`SceneGraphBatch` uses padding and boolean masks, so samples may contain different numbers of nodes, edges, agents, and assignments.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `node_states` | `[B, N, F_n]` | Dynamic node features at time `t` |
| `node_mask` | `[B, N]` | Existing input node slots |
| `edge_index` | `[B, 2, E]` | Directed local node indices |
| `edge_type` | `[B, E]` | Relation id; `0` is reserved for no relation |
| `edge_mask` | `[B, E]` | Valid sparse input edges |
| `agent_states` | `[B, A, F_a]` | Current agent features |
| `agent_mask` | `[B, A]` | Valid agents |
| `assignments` | `[B, M, F_u]` | One vector per assigned subtask-agent pair |
| `assignment_mask` | `[B, M]` | Valid assignments |

The target edge relation tensor is dense `[B, N, N]`. This makes edge addition and deletion explicit. Relation id `0` means no edge. The loss downweights class `0` to reduce class imbalance.

Object identity fields such as `original_id`, `object_tag`, and `caption` should be retained outside the learned dynamic state. When decoding, copy those fields by slot and replace only predicted dynamic attributes, existence, and relations. Reserve unused node slots if incremental exploration must discover objects that were absent at `t`.

Recommended initial dynamic node vector:

```text
[bbox_center(3), bbox_extent(3), confidence(1), visible(1),
 pickupable(1), openable(1), is_open(1), is_toggled(1),
 is_dirty(1), is_filled(1), is_cooked(1), learned_object_type_embedding(...)]
```

Recommended agent vector:

```text
[position(3), rotation(3), camera_horizon(1), inventory_summary(...),
 last_action_success(1), collision_last_step(1), availability(1)]
```

An assignment vector should combine a subtask/action embedding, grounding-object embedding, assigned-agent embedding, and optional retry/dependency features.

## Usage

```python
config = WorldModelConfig(
    node_state_dim=32,
    agent_state_dim=24,
    assignment_feature_dim=32,
    num_relations=6,
)
model = SceneGraphWorldModel(config)
h_t = model.initial_state(batch.node_states.shape[0], device=batch.node_states.device)
output = model(batch, h_t)
losses = world_model_loss(output, batch, targets)
losses["total"].backward()
h_t = output.h_next.detach()
```

Run the included shape and backward-pass check from the repository root:

```bash
/225010231/miniconda3/envs/conceptgraph/bin/python3.10 \
  -m planning.scene_graph_world_model.smoke_test
```

## Dataset integration

One dataset transition should be assembled from each `loops/loop_xxx` directory:

- input graph: `pre_task_subgraph.json` or the global graph referenced by `pre_scenegraph_info.json`;
- input assignment: `executed_allocation_unit.json`;
- input agent state: `pre_agent_states.json`;
- target graph: `post_task_subgraph.json` for local prediction, or `merged_scenegraph/scene_graph.json` for global prediction;
- target task status: `task_status.json` plus timing/collision fields from the execution trace;
- target agent state: `post_agent_states.json`.

Use a stable `original_id -> slot` map within an episode. Without stable slots, a node-wise transition loss compares unrelated objects and cannot learn meaningful dynamics.

from __future__ import annotations

import torch

from planning.scene_graph_world_model import (
    SceneGraphBatch,
    SceneGraphWorldModel,
    WorldModelConfig,
    WorldModelTargets,
    world_model_loss,
)


def make_batch(config: WorldModelConfig) -> tuple[SceneGraphBatch, WorldModelTargets]:
    batch_size, num_nodes, num_edges = 2, 6, 7
    num_agents, num_assignments = 3, 2
    node_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    agent_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    assignment_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    edge_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    edge_index = torch.tensor(
        [
            [[0, 1, 2, 3, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0]],
        ]
    )
    batch = SceneGraphBatch(
        node_states=torch.randn(batch_size, num_nodes, config.node_state_dim),
        node_mask=node_mask,
        edge_index=edge_index,
        edge_type=torch.randint(1, config.num_relations, (batch_size, num_edges)),
        edge_mask=edge_mask,
        agent_states=torch.randn(batch_size, num_agents, config.agent_state_dim),
        agent_mask=agent_mask,
        assignments=torch.randn(batch_size, num_assignments, config.assignment_feature_dim),
        assignment_mask=assignment_mask,
    )
    next_edges = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.long)
    next_edges[:, 0, 1] = 1
    targets = WorldModelTargets(
        next_node_states=torch.randn_like(batch.node_states),
        next_node_mask=node_mask,
        next_edge_relations=next_edges,
        task_completed=torch.randint(0, 2, assignment_mask.shape).float(),
        task_duration=torch.rand(assignment_mask.shape) * 10.0,
        task_collision=torch.randint(0, 2, assignment_mask.shape).float(),
        next_agent_states=torch.randn_like(batch.agent_states),
        next_agent_mask=agent_mask,
    )
    return batch, targets


def main() -> None:
    torch.manual_seed(7)
    config = WorldModelConfig(
        node_state_dim=12,
        agent_state_dim=10,
        assignment_feature_dim=8,
        hidden_dim=32,
        latent_dim=8,
        max_nodes=8,
        max_agents=4,
        max_assignments=4,
    )
    model = SceneGraphWorldModel(config)
    batch, targets = make_batch(config)
    output = model(batch, deterministic_latent=True)
    losses = world_model_loss(output, batch, targets)
    losses["total"].backward()

    assert output.scene_graph.node_states.shape == batch.node_states.shape
    assert output.scene_graph.edge_relation_logits.shape == (2, 6, 6, config.num_relations)
    assert output.task_status.completion_logits.shape == batch.assignment_mask.shape
    assert output.agent_state.states.shape == batch.agent_states.shape
    assert torch.isfinite(losses["total"])
    print(f"smoke test passed; loss={losses['total'].item():.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Dimensions and capacity limits for one padded macro-time-step batch."""

    node_state_dim: int = 32
    agent_state_dim: int = 24
    assignment_feature_dim: int = 32
    num_relations: int = 6

    hidden_dim: int = 256
    latent_dim: int = 64
    edge_decoder_rank: int = 32
    gnn_layers: int = 3
    dropout: float = 0.1

    max_nodes: int = 128
    max_agents: int = 8
    max_assignments: int = 8

    min_log_std: float = -5.0
    max_log_std: float = 2.0
    predict_residuals: bool = True


@dataclass
class WorldModelLossConfig:
    node_state_weight: float = 1.0
    node_existence_weight: float = 0.25
    edge_relation_weight: float = 1.0
    completion_weight: float = 1.0
    duration_weight: float = 0.5
    collision_weight: float = 1.0
    agent_state_weight: float = 1.0
    agent_existence_weight: float = 0.25
    kl_weight: float = 1e-3

    # Dense adjacency labels contain many more "no relation" entries than edges.
    no_relation_class_weight: float = 0.1
    free_nats: float = 1.0

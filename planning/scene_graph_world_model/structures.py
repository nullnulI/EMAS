from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SceneGraphBatch:
    """Padded inputs for one macro time step.

    Relation id 0 is reserved for "no relation". Padding edges are excluded by
    ``edge_mask`` and may contain zero indices/types.
    """

    node_states: Tensor
    node_mask: Tensor
    edge_index: Tensor
    edge_type: Tensor
    edge_mask: Tensor
    agent_states: Tensor
    agent_mask: Tensor
    assignments: Tensor
    assignment_mask: Tensor

    def validate(self) -> None:
        if self.node_states.ndim != 3:
            raise ValueError("node_states must have shape [B, N, F_node]")
        batch_size, num_nodes, _ = self.node_states.shape
        if self.node_mask.shape != (batch_size, num_nodes):
            raise ValueError("node_mask must have shape [B, N]")
        if self.edge_index.ndim != 3 or self.edge_index.shape[:2] != (batch_size, 2):
            raise ValueError("edge_index must have shape [B, 2, E]")
        num_edges = self.edge_index.shape[2]
        if self.edge_type.shape != (batch_size, num_edges):
            raise ValueError("edge_type must have shape [B, E]")
        if self.edge_mask.shape != (batch_size, num_edges):
            raise ValueError("edge_mask must have shape [B, E]")
        if self.agent_states.ndim != 3 or self.agent_states.shape[0] != batch_size:
            raise ValueError("agent_states must have shape [B, A, F_agent]")
        if self.agent_mask.shape != self.agent_states.shape[:2]:
            raise ValueError("agent_mask must have shape [B, A]")
        if self.assignments.ndim != 3 or self.assignments.shape[0] != batch_size:
            raise ValueError("assignments must have shape [B, M, F_assignment]")
        if self.assignment_mask.shape != self.assignments.shape[:2]:
            raise ValueError("assignment_mask must have shape [B, M]")

    @property
    def batch_size(self) -> int:
        return int(self.node_states.shape[0])


@dataclass
class DiagonalGaussian:
    mean: Tensor
    log_std: Tensor

    def sample(self, deterministic: bool = False) -> Tensor:
        if deterministic:
            return self.mean
        return self.mean + self.log_std.exp() * torch.randn_like(self.mean)


@dataclass
class SceneGraphPrediction:
    node_states: Tensor
    node_existence_logits: Tensor
    edge_relation_logits: Tensor


@dataclass
class TaskStatusPrediction:
    completion_logits: Tensor
    duration: Tensor
    collision_logits: Tensor


@dataclass
class AgentStatePrediction:
    states: Tensor
    existence_logits: Tensor


@dataclass
class WorldModelOutput:
    h_next: Tensor
    z_t: Tensor
    prior: DiagonalGaussian
    posterior: DiagonalGaussian
    x_t: Tensor
    u_t: Tensor
    scene_graph: SceneGraphPrediction
    task_status: TaskStatusPrediction
    agent_state: AgentStatePrediction

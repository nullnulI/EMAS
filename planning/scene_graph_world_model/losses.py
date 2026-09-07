from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import WorldModelLossConfig
from .structures import DiagonalGaussian, SceneGraphBatch, WorldModelOutput


@dataclass
class WorldModelTargets:
    next_node_states: Tensor
    next_node_mask: Tensor
    next_edge_relations: Tensor
    task_completed: Tensor
    task_duration: Tensor
    task_collision: Tensor
    next_agent_states: Tensor
    next_agent_mask: Tensor


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def diagonal_gaussian_kl(posterior: DiagonalGaussian, prior: DiagonalGaussian) -> Tensor:
    posterior_var = torch.exp(2.0 * posterior.log_std)
    prior_var = torch.exp(2.0 * prior.log_std)
    kl = (
        prior.log_std
        - posterior.log_std
        + (posterior_var + (posterior.mean - prior.mean).square()) / (2.0 * prior_var)
        - 0.5
    )
    return kl.sum(dim=-1)


def world_model_loss(
    output: WorldModelOutput,
    batch: SceneGraphBatch,
    targets: WorldModelTargets,
    config: WorldModelLossConfig | None = None,
) -> dict[str, Tensor]:
    config = config or WorldModelLossConfig()

    node_state = masked_mean(
        (output.scene_graph.node_states - targets.next_node_states).square(),
        targets.next_node_mask,
    )
    node_existence = F.binary_cross_entropy_with_logits(
        output.scene_graph.node_existence_logits,
        targets.next_node_mask.to(output.scene_graph.node_existence_logits.dtype),
    )

    num_relations = output.scene_graph.edge_relation_logits.shape[-1]
    relation_weights = output.scene_graph.edge_relation_logits.new_ones(num_relations)
    relation_weights[0] = config.no_relation_class_weight
    edge_loss_raw = F.cross_entropy(
        output.scene_graph.edge_relation_logits.permute(0, 3, 1, 2),
        targets.next_edge_relations.long(),
        weight=relation_weights,
        reduction="none",
    )
    pair_mask = targets.next_node_mask.unsqueeze(1) & targets.next_node_mask.unsqueeze(2)
    diagonal = torch.eye(pair_mask.shape[-1], dtype=torch.bool, device=pair_mask.device)
    pair_mask = pair_mask & ~diagonal.unsqueeze(0)
    edge_relation = masked_mean(edge_loss_raw, pair_mask)

    task_mask = batch.assignment_mask.bool()
    completion = masked_mean(
        F.binary_cross_entropy_with_logits(
            output.task_status.completion_logits,
            targets.task_completed.to(output.task_status.completion_logits.dtype),
            reduction="none",
        ),
        task_mask,
    )
    duration = masked_mean(
        F.smooth_l1_loss(
            torch.log1p(output.task_status.duration),
            torch.log1p(targets.task_duration.clamp_min(0.0)),
            reduction="none",
        ),
        task_mask,
    )
    collision = masked_mean(
        F.binary_cross_entropy_with_logits(
            output.task_status.collision_logits,
            targets.task_collision.to(output.task_status.collision_logits.dtype),
            reduction="none",
        ),
        task_mask,
    )

    agent_state = masked_mean(
        (output.agent_state.states - targets.next_agent_states).square(),
        targets.next_agent_mask,
    )
    agent_existence = F.binary_cross_entropy_with_logits(
        output.agent_state.existence_logits,
        targets.next_agent_mask.to(output.agent_state.existence_logits.dtype),
    )

    kl_per_sample = diagonal_gaussian_kl(output.posterior, output.prior)
    kl = torch.clamp(kl_per_sample, min=config.free_nats).mean()

    total = (
        config.node_state_weight * node_state
        + config.node_existence_weight * node_existence
        + config.edge_relation_weight * edge_relation
        + config.completion_weight * completion
        + config.duration_weight * duration
        + config.collision_weight * collision
        + config.agent_state_weight * agent_state
        + config.agent_existence_weight * agent_existence
        + config.kl_weight * kl
    )
    return {
        "total": total,
        "node_state": node_state,
        "node_existence": node_existence,
        "edge_relation": edge_relation,
        "completion": completion,
        "duration": duration,
        "collision": collision,
        "agent_state": agent_state,
        "agent_existence": agent_existence,
        "kl": kl,
    }

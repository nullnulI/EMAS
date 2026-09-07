from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import WorldModelConfig
from .layers import GaussianHead, MLP, SceneGraphEncoder, SetEncoder
from .structures import (
    AgentStatePrediction,
    DiagonalGaussian,
    SceneGraphBatch,
    SceneGraphPrediction,
    TaskStatusPrediction,
    WorldModelOutput,
)


class SceneGraphDecoder(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.predict_residuals = config.predict_residuals
        self.num_relations = config.num_relations
        self.edge_rank = config.edge_decoder_rank
        self.node_head = MLP(hidden_dim * 2, hidden_dim, config.node_state_dim + 1)
        edge_projection_dim = config.num_relations * config.edge_decoder_rank
        self.edge_source = nn.Linear(hidden_dim, edge_projection_dim)
        self.edge_target = nn.Linear(hidden_dim, edge_projection_dim)
        self.edge_global = nn.Linear(hidden_dim, config.num_relations)

    def forward(
        self,
        current_node_states: Tensor,
        node_context: Tensor,
        transition_context: Tensor,
    ) -> SceneGraphPrediction:
        num_nodes = node_context.shape[1]
        global_nodes = transition_context.unsqueeze(1).expand(-1, num_nodes, -1)
        node_output = self.node_head(torch.cat([node_context, global_nodes], dim=-1))
        node_update, existence_logits = node_output[..., :-1], node_output[..., -1]
        if self.predict_residuals:
            node_update = current_node_states + node_update

        batch_size = node_context.shape[0]
        source = self.edge_source(node_context).reshape(
            batch_size, num_nodes, self.num_relations, self.edge_rank
        )
        target = self.edge_target(node_context).reshape(
            batch_size, num_nodes, self.num_relations, self.edge_rank
        )
        edge_relation_logits = torch.einsum("bnrk,bmrk->bnmr", source, target)
        edge_relation_logits = edge_relation_logits / math.sqrt(self.edge_rank)
        edge_relation_logits = edge_relation_logits + self.edge_global(transition_context).unsqueeze(1).unsqueeze(1)
        return SceneGraphPrediction(
            node_states=node_update,
            node_existence_logits=existence_logits,
            edge_relation_logits=edge_relation_logits,
        )


class TaskStatusDecoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.head = MLP(hidden_dim * 2, hidden_dim, 3)

    def forward(self, assignment_tokens: Tensor, transition_context: Tensor) -> TaskStatusPrediction:
        context = transition_context.unsqueeze(1).expand(-1, assignment_tokens.shape[1], -1)
        outputs = self.head(torch.cat([assignment_tokens, context], dim=-1))
        return TaskStatusPrediction(
            completion_logits=outputs[..., 0],
            duration=F.softplus(outputs[..., 1]),
            collision_logits=outputs[..., 2],
        )


class AgentStateDecoder(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.predict_residuals = config.predict_residuals
        self.head = MLP(config.hidden_dim * 2, config.hidden_dim, config.agent_state_dim + 1)

    def forward(
        self,
        current_agent_states: Tensor,
        agent_tokens: Tensor,
        transition_context: Tensor,
    ) -> AgentStatePrediction:
        context = transition_context.unsqueeze(1).expand(-1, agent_tokens.shape[1], -1)
        output = self.head(torch.cat([agent_tokens, context], dim=-1))
        state_update, existence_logits = output[..., :-1], output[..., -1]
        if self.predict_residuals:
            state_update = current_agent_states + state_update
        return AgentStatePrediction(states=state_update, existence_logits=existence_logits)


class SceneGraphWorldModel(nn.Module):
    """One-step recurrent state-space model for macro-time-step transitions.

    During training and online filtering, use the posterior latent sampled from
    q(z_t | h_t, x_t). For open-loop imagination, set ``use_posterior=False``
    to sample from p(z_t | h_t).
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.graph_encoder = SceneGraphEncoder(
            node_state_dim=config.node_state_dim,
            hidden_dim=h,
            num_relations=config.num_relations,
            max_nodes=config.max_nodes,
            num_layers=config.gnn_layers,
            dropout=config.dropout,
        )
        self.agent_encoder = SetEncoder(
            config.agent_state_dim,
            h,
            config.max_agents,
            config.dropout,
        )
        self.assignment_encoder = SetEncoder(
            config.assignment_feature_dim,
            h,
            config.max_assignments,
            config.dropout,
        )
        self.observation_fusion = MLP(h * 2, h, h, dropout=config.dropout)

        self.prior_head = GaussianHead(
            h,
            h,
            config.latent_dim,
            config.min_log_std,
            config.max_log_std,
        )
        self.posterior_head = GaussianHead(
            h * 2,
            h,
            config.latent_dim,
            config.min_log_std,
            config.max_log_std,
        )
        self.recurrent = nn.GRUCell(config.latent_dim + h, h)
        self.transition_fusion = MLP(
            h * 2 + config.latent_dim,
            h,
            h,
            dropout=config.dropout,
        )

        self.scene_graph_decoder = SceneGraphDecoder(config)
        self.task_status_decoder = TaskStatusDecoder(h)
        self.agent_state_decoder = AgentStateDecoder(config)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.config.hidden_dim,
            device=device if device is not None else parameter.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def forward(
        self,
        batch: SceneGraphBatch,
        h_t: Tensor | None = None,
        *,
        use_posterior: bool = True,
        deterministic_latent: bool = False,
    ) -> WorldModelOutput:
        batch.validate()
        if batch.node_states.shape[-1] != self.config.node_state_dim:
            raise ValueError("node state feature dimension does not match config")
        if batch.agent_states.shape[-1] != self.config.agent_state_dim:
            raise ValueError("agent state feature dimension does not match config")
        if batch.assignments.shape[-1] != self.config.assignment_feature_dim:
            raise ValueError("assignment feature dimension does not match config")
        if h_t is None:
            h_t = self.initial_state(
                batch.batch_size,
                device=batch.node_states.device,
                dtype=batch.node_states.dtype,
            )

        node_context, graph_context = self.graph_encoder(
            batch.node_states,
            batch.node_mask,
            batch.edge_index,
            batch.edge_type,
            batch.edge_mask,
        )
        agent_tokens, agent_context = self.agent_encoder(batch.agent_states, batch.agent_mask)
        assignment_tokens, u_t = self.assignment_encoder(batch.assignments, batch.assignment_mask)
        x_t = self.observation_fusion(torch.cat([graph_context, agent_context], dim=-1))

        prior_mean, prior_log_std = self.prior_head(h_t)
        posterior_mean, posterior_log_std = self.posterior_head(torch.cat([h_t, x_t], dim=-1))
        prior = DiagonalGaussian(prior_mean, prior_log_std)
        posterior = DiagonalGaussian(posterior_mean, posterior_log_std)
        latent_distribution = posterior if use_posterior else prior
        z_t = latent_distribution.sample(deterministic=deterministic_latent)

        h_next = self.recurrent(torch.cat([z_t, u_t], dim=-1), h_t)
        transition_context = self.transition_fusion(torch.cat([h_next, z_t, u_t], dim=-1))

        return WorldModelOutput(
            h_next=h_next,
            z_t=z_t,
            prior=prior,
            posterior=posterior,
            x_t=x_t,
            u_t=u_t,
            scene_graph=self.scene_graph_decoder(
                batch.node_states,
                node_context,
                transition_context,
            ),
            task_status=self.task_status_decoder(assignment_tokens, transition_context),
            agent_state=self.agent_state_decoder(
                batch.agent_states,
                agent_tokens,
                transition_context,
            ),
        )

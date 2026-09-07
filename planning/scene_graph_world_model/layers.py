from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool()
        logits = self.score(values).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class RelationalMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout)
        self.update = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: Tensor,
        edge_index: Tensor,
        edge_embeddings: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        batch_size, num_nodes, hidden_dim = nodes.shape
        source_index = edge_index[:, 0].clamp(0, num_nodes - 1)
        target_index = edge_index[:, 1].clamp(0, num_nodes - 1)
        source = torch.gather(
            nodes,
            1,
            source_index.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )
        messages = self.message(torch.cat([source, edge_embeddings], dim=-1))
        messages = messages * edge_mask.unsqueeze(-1).to(messages.dtype)

        aggregated = nodes.new_zeros(batch_size, num_nodes, hidden_dim)
        aggregated.scatter_add_(
            1,
            target_index.unsqueeze(-1).expand(-1, -1, hidden_dim),
            messages,
        )
        degree = nodes.new_zeros(batch_size, num_nodes, 1)
        degree.scatter_add_(
            1,
            target_index.unsqueeze(-1),
            edge_mask.unsqueeze(-1).to(nodes.dtype),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        update = self.update(torch.cat([nodes, aggregated], dim=-1))
        return self.norm(nodes + update)


class SceneGraphEncoder(nn.Module):
    def __init__(
        self,
        node_state_dim: int,
        hidden_dim: int,
        num_relations: int,
        max_nodes: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.node_projection = nn.Linear(node_state_dim, hidden_dim)
        self.node_slot_embedding = nn.Embedding(max_nodes, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.layers = nn.ModuleList(
            RelationalMessagePassing(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(
        self,
        node_states: Tensor,
        node_mask: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_nodes = node_states.shape[1]
        if num_nodes > self.max_nodes:
            raise ValueError(f"batch has {num_nodes} nodes, max_nodes={self.max_nodes}")
        slots = torch.arange(num_nodes, device=node_states.device)
        nodes = self.node_projection(node_states) + self.node_slot_embedding(slots).unsqueeze(0)
        edges = self.relation_embedding(edge_type)
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edges, edge_mask.bool())
        return nodes, self.pool(nodes, node_mask.bool())


class SetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_items: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_items = max_items
        self.item_projection = MLP(input_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.slot_embedding = nn.Embedding(max_items, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        num_items = values.shape[1]
        if num_items > self.max_items:
            raise ValueError(f"batch has {num_items} items, max_items={self.max_items}")
        slots = torch.arange(num_items, device=values.device)
        tokens = self.item_projection(values) + self.slot_embedding(slots).unsqueeze(0)
        tokens = self.norm(tokens)
        return tokens, self.pool(tokens, mask.bool())


class GaussianHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        super().__init__()
        self.network = MLP(input_dim, hidden_dim, latent_dim * 2)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_log_std = self.network(inputs).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

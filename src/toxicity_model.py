"""
Edge-aware Graph Isomorphism Network (GINE) backbone and toxicity prediction model.

Architecture:
- Message Passing: GINEConv layers incorporating 30-dim atom features and 6-dim bond features.
- Node update: MLP([node_dim, hidden_dim, hidden_dim]) applied to (h_u + ReLU(W_edge * e_uv)).
- Graph Pooling: Disjoint-union global mean & sum pooling respecting batch vector.
- Latent Embedding: D-dimensional graph embedding for downstream OOD detection.
- Prediction Head: Linear layer producing 12 logits for multi-task Tox21 classification.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import BatchGraphs


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of scatter_add without requiring external C++ extensions.

    Args:
        src: Source tensor to scatter [N, D]
        index: Target index tensor [N]
        dim: Dimension along which to scatter
        dim_size: Number of target elements along scattered dimension

    Returns:
        Tensor of shape [dim_size, D]
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    if index.numel() == 0:
        return out
    index_expanded = index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)
    return out.scatter_add_(dim, index_expanded, src)


class GINELayer(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE) layer.
    
    Formula:
        m_v = sum_{u in N(v)} ReLU(h_u + Linear(e_uv))
        h'_v = MLP((1 + eps) * h_v + m_v)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        train_eps: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_dim, node_dim)
        
        # 2-layer MLP for node update
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        
        self.eps = nn.Parameter(torch.zeros(1)) if train_eps else 0.0

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a single GINE layer.

        Args:
            x: Node features [N, node_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, edge_dim]

        Returns:
            Updated node features [N, out_dim]
        """
        num_nodes = x.size(0)
        num_edges = edge_attr.size(0)

        if num_edges == 0:
            # Handle acyclic or isolated nodes with no edges
            agg = torch.zeros_like(x)
        else:
            src_nodes = edge_index[0]  # u
            dst_nodes = edge_index[1]  # v

            # Encode edge features to node_dim
            e_emb = self.edge_encoder(edge_attr)
            # Message: ReLU(h_u + e_uv)
            messages = F.relu(x[src_nodes] + e_emb)

            # Aggregate incoming messages onto destination nodes v
            agg = scatter_add(messages, dst_nodes, dim=0, dim_size=num_nodes)

        # GIN update: (1 + eps) * h_v + agg
        if isinstance(self.eps, nn.Parameter):
            out = (1.0 + self.eps) * x + agg
        else:
            out = x + agg

        return self.mlp(out)


class GINEToxicityModel(nn.Module):
    """
    Complete Edge-Aware GNN for Multi-Task Toxicity Prediction & OOD Embedding Extraction.

    Args:
        in_node_dim: Dimension of input node features (default: 30)
        in_edge_dim: Dimension of input edge features (default: 6)
        hidden_dim: Hidden representation dimension (default: 128)
        num_layers: Number of GINE message-passing layers (default: 4)
        num_tasks: Number of toxicity prediction targets (default: 12)
        dropout: Dropout rate (default: 0.1)
        pooling: Global pooling strategy: 'mean', 'sum', or 'mean_sum' (default: 'mean_sum')
    """

    def __init__(
        self,
        in_node_dim: int = 30,
        in_edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.1,
        pooling: str = "mean_sum",
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.pooling = pooling

        # Initial node projection
        self.node_encoder = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            GINELayer(
                node_dim=hidden_dim,
                edge_dim=in_edge_dim,
                out_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Graph embedding dimension based on pooling strategy
        if pooling == "mean_sum":
            self.embed_dim = hidden_dim * 2
        else:
            self.embed_dim = hidden_dim

        # Predictor head for 12 multi-task toxicity endpoints
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def _pool_nodes(self, h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        """
        Pool node representations to graph-level representations.
        
        Handles disjoint-union batch vectors of variable-sized graphs.
        """
        # Sum pooling
        h_sum = scatter_add(h, batch, dim=0, dim_size=num_graphs)

        if self.pooling == "sum":
            return h_sum

        # Calculate node counts per graph
        ones = torch.ones((h.size(0), 1), dtype=h.dtype, device=h.device)
        node_counts = scatter_add(ones, batch, dim=0, dim_size=num_graphs)
        node_counts = torch.clamp(node_counts, min=1.0)
        h_mean = h_sum / node_counts

        if self.pooling == "mean":
            return h_mean
        elif self.pooling == "mean_sum":
            return torch.cat([h_mean, h_sum], dim=-1)
        else:
            raise ValueError(f"Unsupported pooling mode: '{self.pooling}'")

    def get_embeddings(self, batch_data: Union[BatchGraphs, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Extract graph-level latent representations [batch_size, embed_dim].

        Guarantees:
        - 1-to-1 correspondence with the graphs in batch_data.uid.
        - Usable post-hoc by novelty estimators and validity detectors.
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch
        num_graphs = getattr(batch_data, "num_graphs", int(batch.max().item()) + 1 if batch.numel() > 0 else 0)

        # Initial projection
        h = self.node_encoder(x)

        # GINE message passing with residual connections
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_attr)

        # Global graph pooling
        embeddings = self._pool_nodes(h, batch, num_graphs)
        return embeddings

    def forward(
        self,
        batch_data: Union[BatchGraphs, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing toxicity prediction logits and latent graph embeddings.

        Args:
            batch_data: BatchGraphs container or dictionary containing x, edge_index, edge_attr, batch.

        Returns:
            logits: Predicted toxicity logits [batch_size, 12]
            embeddings: Graph latent embeddings [batch_size, embed_dim]
        """
        embeddings = self.get_embeddings(batch_data)
        logits = self.classifier(embeddings)
        return logits, embeddings

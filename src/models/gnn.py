"""
Graph Neural Network models for fungal pathogenicity prediction.

Implements a Graph Attention Network (GAT) backbone and a lightweight
Graph Convolutional Network (GCN) baseline, both capable of node-level
or graph-level (pooled) classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool

_POOL_FNS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "sum": global_add_pool,
}


class GATPathogenicityModel(nn.Module):
    """Graph Attention Network for binary pathogenicity classification.

    Parameters
    ----------
    in_channels:
        Dimensionality of input node features.
    hidden_channels:
        Number of hidden units per attention head.
    num_classes:
        Number of output classes (2 for binary classification).
    num_layers:
        Number of GAT message-passing layers.
    heads:
        Number of attention heads per layer.
    dropout:
        Dropout probability applied after each layer.
    pooling:
        Pooling strategy for graph-level tasks (``"mean"``, ``"max"``,
        ``"sum"``).  Set to ``None`` for node-level tasks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.pooling = _POOL_FNS.get(pooling, global_mean_pool)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_channels → hidden_channels * heads
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Final layer: collapse multi-head to single output per node
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
            classifier_in = hidden_channels
        else:
            classifier_in = hidden_channels * heads

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_in // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is not None:
            x = self.pooling(x, batch)

        return self.classifier(x)


class GCNPathogenicityModel(nn.Module):
    """Graph Convolutional Network baseline for pathogenicity classification.

    Parameters
    ----------
    in_channels:
        Dimensionality of input node features.
    hidden_channels:
        Number of hidden units per GCN layer.
    num_classes:
        Number of output classes.
    num_layers:
        Number of GCN message-passing layers.
    dropout:
        Dropout probability applied after each layer.
    pooling:
        Pooling strategy for graph-level tasks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.pooling = _POOL_FNS.get(pooling, global_mean_pool)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for in_dim, out_dim in zip(dims, dims[1:]):
            self.convs.append(GCNConv(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is not None:
            x = self.pooling(x, batch)

        return self.classifier(x)

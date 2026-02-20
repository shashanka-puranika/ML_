"""
Integrated multi-omics GNN model.

Extends the base GNN models by adding a per-layer omics encoder that projects
each omics modality into a shared embedding space before graph message-passing.
This allows the model to learn modality-specific representations that are fused
at the node level prior to propagation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

_POOL_FNS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "sum": global_add_pool,
}


class OmicsEncoder(nn.Module):
    """MLP encoder that projects a single omics feature vector to an embedding.

    Parameters
    ----------
    in_dim:
        Input feature dimensionality for this omics modality.
    out_dim:
        Output embedding dimensionality.
    dropout:
        Dropout probability.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiOmicsGNNModel(nn.Module):
    """Multi-omics aware GAT model with per-modality encoders.

    Each omics modality is independently encoded by a small MLP before
    the embeddings are concatenated and fed into a stack of GAT layers.

    Parameters
    ----------
    omics_dims:
        Dict mapping omics name → feature dimensionality (e.g.
        ``{"genomics": 500, "transcriptomics": 500, "proteomics": 500}``).
    embed_dim:
        Per-modality embedding dimensionality after encoding.
    hidden_channels:
        Hidden channels in GAT layers.
    num_classes:
        Number of output classes.
    num_layers:
        Number of GAT layers.
    heads:
        Attention heads per GAT layer.
    dropout:
        Dropout probability.
    pooling:
        Graph-level pooling strategy.
    """

    def __init__(
        self,
        omics_dims: dict[str, int],
        embed_dim: int = 64,
        hidden_channels: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.omics_names = list(omics_dims.keys())
        self.dropout = dropout
        self.pooling = _POOL_FNS.get(pooling, global_mean_pool)

        # Per-modality encoders
        self.encoders = nn.ModuleDict(
            {name: OmicsEncoder(dim, embed_dim, dropout=dropout) for name, dim in omics_dims.items()}
        )

        fused_dim = embed_dim * len(omics_dims)

        # Attention-based modality fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim, len(omics_dims)),
            nn.Softmax(dim=-1),
        )

        gat_in = embed_dim  # after weighted sum fusion
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATConv(gat_in, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

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
        omics_splits: list[int],
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Concatenated node features of shape ``(N, sum(F_i))``.
        edge_index:
            COO edge index of shape ``(2, E)``.
        omics_splits:
            List of feature dimensions in the order of ``self.omics_names``,
            used to slice ``x`` back into individual modalities.
        batch:
            Batch vector for graph-level pooling.
        """
        # Split concatenated features back into individual modalities
        embeddings = []
        offset = 0
        for name, dim in zip(self.omics_names, omics_splits):
            x_i = x[:, offset : offset + dim]
            embeddings.append(self.encoders[name](x_i))
            offset += dim

        # Attention-based fusion:
        # 1. Stack per-modality embeddings along a new modality dimension.
        # 2. Concatenate them to produce gate weights (one scalar weight per modality).
        # 3. Weighted-sum the stacked embeddings using the gates.
        stacked = torch.stack(embeddings, dim=-1)            # (N, embed_dim, M)
        fused_cat = torch.cat(embeddings, dim=-1)            # (N, embed_dim * M) – used only for gate input
        gates = self.fusion_gate(fused_cat).unsqueeze(1)     # (N, 1, M)
        h = (stacked * gates).sum(dim=-1)                    # (N, embed_dim)

        # GAT layers
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is not None:
            h = self.pooling(h, batch)

        return self.classifier(h)

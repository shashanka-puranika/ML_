"""
Evaluation metrics for fungal pathogenicity prediction.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    graph: Data,
    mask: torch.Tensor,
    device: str = "cpu",
    multiomics_splits: list | None = None,
) -> Dict[str, float]:
    """Compute classification metrics for nodes indicated by *mask*.

    Parameters
    ----------
    model:
        Trained GNN model.
    graph:
        Full graph (will be moved to *device*).
    mask:
        Boolean tensor selecting evaluation nodes.
    device:
        Torch device string.
    multiomics_splits:
        Per-modality feature dimensions for ``MultiOmicsGNNModel``; ``None``
        for standard GNN models.

    Returns
    -------
    metrics : dict
        Dictionary with keys ``accuracy``, ``auroc``, ``auprc``, ``f1``,
        ``mcc``.
    """
    _device = torch.device(device)
    model = model.to(_device)
    graph = graph.to(_device)
    mask = mask.to(_device)

    model.eval()
    with torch.no_grad():
        if multiomics_splits is not None:
            logits = model(graph.x, graph.edge_index, multiomics_splits)
        else:
            logits = model(graph.x, graph.edge_index)

    logits_masked = logits[mask].cpu()
    y_true = graph.y[mask].cpu().numpy()
    y_pred = logits_masked.argmax(dim=-1).numpy()
    y_score = torch.softmax(logits_masked, dim=-1)[:, 1].numpy()

    n_classes = len(np.unique(y_true))

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary" if n_classes == 2 else "macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if n_classes == 2:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
        metrics["auprc"] = average_precision_score(y_true, y_score)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    for k, v in metrics.items():
        logger.info("  %-10s = %.4f", k, v)

    return metrics

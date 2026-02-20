"""
Training loop and utilities for the fungal pathogenicity GNN models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node-level train / val / test masks
# ---------------------------------------------------------------------------


def make_masks(
    n: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random boolean masks for train / val / test splits.

    Parameters
    ----------
    n:
        Total number of nodes.
    test_size:
        Fraction of nodes to use for testing.
    val_size:
        Fraction of nodes to use for validation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train_mask, val_mask, test_mask : torch.BoolTensor
        Boolean tensors of length *n*.
    """
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)

    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    def _mask(idx: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    return _mask(train_idx), _mask(val_idx), _mask(test_idx)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Trains a GNN model on a single ``torch_geometric.data.Data`` graph.

    Parameters
    ----------
    model:
        The GNN model to train.
    graph:
        Full graph with node features, edges, and labels.
    train_mask:
        Boolean mask for training nodes.
    val_mask:
        Boolean mask for validation nodes.
    lr:
        Learning rate.
    weight_decay:
        L2 regularisation weight.
    epochs:
        Maximum number of training epochs.
    patience:
        Early stopping patience (in epochs without validation-loss improvement).
    checkpoint_dir:
        Directory to save the best model checkpoint.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    multiomics_splits:
        If using ``MultiOmicsGNNModel``, pass the list of per-modality feature
        dimensions here so the model can split node features correctly.
    """

    def __init__(
        self,
        model: nn.Module,
        graph: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        patience: int = 20,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        multiomics_splits: Optional[list] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.graph = graph.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.multiomics_splits = multiomics_splits

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=patience // 2
        )
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------

    def _forward(self) -> torch.Tensor:
        """Run a forward pass, handling both standard and multi-omics models."""
        if self.multiomics_splits is not None:
            return self.model(
                self.graph.x,
                self.graph.edge_index,
                self.multiomics_splits,
            )
        return self.model(self.graph.x, self.graph.edge_index)

    def _step(self, mask: torch.Tensor) -> Tuple[float, float]:
        """Compute loss and accuracy for nodes indicated by *mask*."""
        logits = self._forward()
        loss = self.criterion(logits[mask], self.graph.y[mask])
        preds = logits[mask].argmax(dim=-1)
        acc = (preds == self.graph.y[mask]).float().mean().item()
        return loss, acc

    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """Run the full training loop.

        Returns
        -------
        history : dict
            Dictionary with ``"train_loss"``, ``"train_acc"``, ``"val_loss"``,
            ``"val_acc"`` lists over epochs.
        """
        history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_ckpt = self.checkpoint_dir / "best_model.pt"

        for epoch in range(1, self.epochs + 1):
            # --- training ---
            self.model.train()
            self.optimizer.zero_grad()
            train_loss, train_acc = self._step(self.train_mask)
            train_loss.backward()
            self.optimizer.step()

            # --- validation ---
            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self._step(self.val_mask)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss.item())
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss.item())
            history["val_acc"].append(val_acc)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %03d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f",
                    epoch,
                    train_loss.item(),
                    train_acc,
                    val_loss.item(),
                    val_acc,
                )

            # --- early stopping ---
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), best_ckpt)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        # Restore best checkpoint
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device, weights_only=True))
        logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
        return history

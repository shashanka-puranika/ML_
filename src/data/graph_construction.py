"""
Graph construction for multi-omics integration.

Builds a heterogeneous graph where:
- Nodes  represent individual fungal samples.
- Intra-omics edges connect samples that share high feature-space correlation
  within a single omics layer (genomics, transcriptomics, or proteomics).
- Inter-omics edges optionally connect samples across layers based on a
  cross-layer correlation.

The resulting graph is returned as a ``torch_geometric.data.Data`` object
suitable for graph neural network training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

_OMICS_LAYERS = ["genomics", "transcriptomics", "proteomics"]


# ---------------------------------------------------------------------------
# Edge construction helpers
# ---------------------------------------------------------------------------


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of *X*."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    X_norm = X / norms
    return X_norm @ X_norm.T


def _build_edges_from_similarity(
    sim_matrix: np.ndarray,
    threshold: float,
    self_loops: bool = False,
) -> np.ndarray:
    """Return COO edge index (2, E) for entries of *sim_matrix* >= *threshold*.

    Parameters
    ----------
    sim_matrix:
        Square matrix of shape ``(N, N)``.
    threshold:
        Minimum similarity value for an edge to be included.
    self_loops:
        Whether to include self-loop edges.
    """
    rows, cols = np.where(sim_matrix >= threshold)
    if not self_loops:
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
    return np.stack([rows, cols], axis=0)


def build_intra_omics_edges(
    omics_matrices: Dict[str, np.ndarray],
    threshold: float = 0.7,
) -> Dict[str, np.ndarray]:
    """Build intra-omics sample similarity edges for each omics layer.

    Parameters
    ----------
    omics_matrices:
        Dict mapping omics name → array of shape ``(N, F)``.
    threshold:
        Cosine similarity threshold for edge inclusion.

    Returns
    -------
    edges_per_layer : dict
        Keys are omics layer names; values are COO arrays of shape ``(2, E)``.
    """
    edges: Dict[str, np.ndarray] = {}
    for name, X in omics_matrices.items():
        sim = _cosine_similarity_matrix(X)
        edge_index = _build_edges_from_similarity(sim, threshold=threshold)
        edges[name] = edge_index
        logger.info(
            "Intra-omics edges for '%s': %d edges (threshold=%.2f)",
            name,
            edge_index.shape[1],
            threshold,
        )
    return edges


def build_inter_omics_edges(
    omics_matrices: Dict[str, np.ndarray],
    threshold: float = 0.7,
    layer_pairs: Optional[List[tuple]] = None,
) -> Dict[tuple, np.ndarray]:
    """Build cross-layer sample similarity edges between pairs of omics layers.

    The same sample index space is shared across layers (samples are aligned).

    Parameters
    ----------
    omics_matrices:
        Dict mapping omics name → array of shape ``(N, F)``.
    threshold:
        Cosine similarity threshold for edge inclusion.
    layer_pairs:
        List of ``(layer_a, layer_b)`` tuples to connect.  Defaults to all
        consecutive pairs in ``_OMICS_LAYERS``.

    Returns
    -------
    edges_per_pair : dict
        Keys are ``(layer_a, layer_b)`` tuples; values are COO arrays ``(2, E)``.
    """
    if layer_pairs is None:
        available = [k for k in _OMICS_LAYERS if k in omics_matrices]
        layer_pairs = list(zip(available, available[1:]))

    cross_edges: Dict[tuple, np.ndarray] = {}
    for layer_a, layer_b in layer_pairs:
        if layer_a not in omics_matrices or layer_b not in omics_matrices:
            logger.warning("Skipping pair (%s, %s): layer not found.", layer_a, layer_b)
            continue
        X_a = omics_matrices[layer_a]
        X_b = omics_matrices[layer_b]
        # Cross-layer similarity: average of per-layer intra-sample similarity matrices.
        # Both X_a and X_b live in different feature spaces (different dimensionalities),
        # so direct cross-feature cosine similarity is not applicable.  Instead we average
        # the two N×N intra-layer similarity matrices to obtain a joint affinity that
        # rewards sample pairs consistently similar in both omics layers.
        sim_a = _cosine_similarity_matrix(X_a)
        sim_b = _cosine_similarity_matrix(X_b)
        sim = (sim_a + sim_b) / 2.0
        edge_index = _build_edges_from_similarity(sim, threshold=threshold)
        cross_edges[(layer_a, layer_b)] = edge_index
        logger.info(
            "Inter-omics edges (%s ↔ %s): %d edges",
            layer_a,
            layer_b,
            edge_index.shape[1],
        )
    return cross_edges


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(
    omics_matrices: Dict[str, np.ndarray],
    labels: np.ndarray,
    intra_threshold: float = 0.7,
    inter_threshold: float = 0.7,
    include_inter_omics: bool = True,
) -> Data:
    """Assemble a ``torch_geometric`` graph from multi-omics sample data.

    Node features are formed by concatenating all omics feature vectors for
    each sample.  Edges are constructed from intra- and (optionally)
    inter-omics cosine similarity.

    Parameters
    ----------
    omics_matrices:
        Dict mapping omics name → array of shape ``(N, F_i)``.
    labels:
        Integer array of shape ``(N,)`` with pathogenicity labels.
    intra_threshold:
        Cosine similarity threshold for intra-omics edges.
    inter_threshold:
        Cosine similarity threshold for inter-omics edges.
    include_inter_omics:
        Whether to add cross-layer edges.

    Returns
    -------
    graph : torch_geometric.data.Data
        Graph with node features ``x``, edge index ``edge_index``, and
        labels ``y``.
    """
    n_samples = next(iter(omics_matrices.values())).shape[0]

    # Node features: concatenation of all omics vectors
    feature_parts = [
        omics_matrices[name]
        for name in _OMICS_LAYERS
        if name in omics_matrices
    ]
    x = np.concatenate(feature_parts, axis=1).astype(np.float32)

    # Collect all edge indices
    all_edges: List[np.ndarray] = []

    intra_edges = build_intra_omics_edges(omics_matrices, threshold=intra_threshold)
    all_edges.extend(intra_edges.values())

    if include_inter_omics:
        inter_edges = build_inter_omics_edges(omics_matrices, threshold=inter_threshold)
        all_edges.extend(inter_edges.values())

    if all_edges:
        edge_index = np.concatenate(all_edges, axis=1)
        # Deduplicate edges
        edge_index = np.unique(edge_index, axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    graph = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
        num_nodes=n_samples,
    )

    logger.info(
        "Built graph: %d nodes, %d edges, node feature dim=%d",
        graph.num_nodes,
        graph.num_edges,
        graph.x.shape[1],
    )
    return graph

"""
Tests for graph construction and GNN model modules.
"""

import numpy as np
import pytest
import torch

from src.data.graph_construction import (
    _cosine_similarity_matrix,
    build_graph,
    build_intra_omics_edges,
    build_inter_omics_edges,
)
from src.models.gnn import GATPathogenicityModel, GCNPathogenicityModel
from src.models.multiomics_gnn import MultiOmicsGNNModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_omics(n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "genomics": rng.random((n, 30)).astype(np.float32),
        "transcriptomics": rng.random((n, 40)).astype(np.float32),
        "proteomics": rng.random((n, 25)).astype(np.float32),
    }


def _make_labels(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n)


# ---------------------------------------------------------------------------
# Graph construction tests
# ---------------------------------------------------------------------------


class TestCosineSimMatrix:
    def test_shape(self):
        X = np.random.rand(10, 5).astype(np.float32)
        sim = _cosine_similarity_matrix(X)
        assert sim.shape == (10, 10)

    def test_diagonal_ones(self):
        X = np.random.rand(8, 6).astype(np.float32)
        sim = _cosine_similarity_matrix(X)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_range(self):
        X = np.random.rand(15, 10).astype(np.float32)
        sim = _cosine_similarity_matrix(X)
        assert sim.min() >= -1.0 - 1e-5
        assert sim.max() <= 1.0 + 1e-5


class TestIntraOmicsEdges:
    def test_returns_all_layers(self):
        omics = _make_omics(15)
        edges = build_intra_omics_edges(omics, threshold=0.0)
        assert set(edges.keys()) == {"genomics", "transcriptomics", "proteomics"}

    def test_high_threshold_fewer_edges(self):
        omics = _make_omics(15)
        edges_low = build_intra_omics_edges(omics, threshold=0.0)
        edges_high = build_intra_omics_edges(omics, threshold=0.9)
        for name in omics:
            assert edges_high[name].shape[1] <= edges_low[name].shape[1]

    def test_no_self_loops(self):
        omics = _make_omics(15)
        edges = build_intra_omics_edges(omics, threshold=0.0)
        for edge_index in edges.values():
            assert not (edge_index[0] == edge_index[1]).any()


class TestInterOmicsEdges:
    def test_returns_pairs(self):
        omics = _make_omics(15)
        edges = build_inter_omics_edges(omics, threshold=0.0)
        assert ("genomics", "transcriptomics") in edges
        assert ("transcriptomics", "proteomics") in edges

    def test_missing_layer_skipped(self):
        omics = {"genomics": _make_omics(10)["genomics"]}
        edges = build_inter_omics_edges(omics, threshold=0.0)
        assert len(edges) == 0


class TestBuildGraph:
    def test_graph_attributes(self):
        omics = _make_omics(20)
        labels = _make_labels(20)
        graph = build_graph(omics, labels, intra_threshold=0.0, inter_threshold=0.0)

        assert graph.x.shape[0] == 20
        assert graph.x.shape[1] == 30 + 40 + 25  # concatenated features
        assert graph.y.shape[0] == 20
        assert graph.edge_index.shape[0] == 2

    def test_graph_no_inter_edges(self):
        omics = _make_omics(20)
        labels = _make_labels(20)
        g_with = build_graph(omics, labels, include_inter_omics=True, intra_threshold=0.0)
        g_without = build_graph(omics, labels, include_inter_omics=False, intra_threshold=0.0)
        assert g_with.num_edges >= g_without.num_edges


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


IN_CHANNELS = 30 + 40 + 25  # total features from _make_omics


class TestGATPathogenicityModel:
    def test_output_shape(self):
        model = GATPathogenicityModel(
            in_channels=IN_CHANNELS, hidden_channels=16, num_classes=2, num_layers=2, heads=2
        )
        omics = _make_omics(20)
        labels = _make_labels(20)
        graph = build_graph(omics, labels, intra_threshold=0.0)

        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
        assert out.shape == (20, 2)

    def test_single_layer_model(self):
        model = GATPathogenicityModel(
            in_channels=IN_CHANNELS, hidden_channels=8, num_classes=2, num_layers=1, heads=2
        )
        omics = _make_omics(10)
        labels = _make_labels(10)
        graph = build_graph(omics, labels, intra_threshold=0.0)

        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
        assert out.shape == (10, 2)


class TestGCNPathogenicityModel:
    def test_output_shape(self):
        model = GCNPathogenicityModel(
            in_channels=IN_CHANNELS, hidden_channels=16, num_classes=2, num_layers=2
        )
        omics = _make_omics(20)
        labels = _make_labels(20)
        graph = build_graph(omics, labels, intra_threshold=0.0)

        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
        assert out.shape == (20, 2)


class TestMultiOmicsGNNModel:
    def _omics_dims(self):
        return {"genomics": 30, "transcriptomics": 40, "proteomics": 25}

    def _splits(self):
        return [30, 40, 25]

    def test_output_shape(self):
        dims = self._omics_dims()
        model = MultiOmicsGNNModel(
            omics_dims=dims,
            embed_dim=16,
            hidden_channels=16,
            num_classes=2,
            num_layers=2,
            heads=2,
        )
        omics = _make_omics(20)
        labels = _make_labels(20)
        graph = build_graph(omics, labels, intra_threshold=0.0)

        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, self._splits())
        assert out.shape == (20, 2)

    def test_probabilities_sum_to_one(self):
        dims = self._omics_dims()
        model = MultiOmicsGNNModel(
            omics_dims=dims, embed_dim=8, hidden_channels=8, num_classes=2, num_layers=1, heads=1
        )
        omics = _make_omics(10)
        labels = _make_labels(10)
        graph = build_graph(omics, labels, intra_threshold=0.0)

        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index, self._splits())
            probs = torch.softmax(logits, dim=-1)
        np.testing.assert_allclose(
            probs.sum(dim=-1).numpy(), np.ones(10), atol=1e-5
        )

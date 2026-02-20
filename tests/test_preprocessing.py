"""
Tests for the multi-omics preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    MultiOmicsPreprocessor,
    align_samples,
    load_labels,
    log_transform,
    normalise,
    remove_low_variance_features,
    select_top_k_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n_samples: int, n_features: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.random((n_samples, n_features))
    index = [f"sample_{i}" for i in range(n_samples)]
    cols = [f"feat_{j}" for j in range(n_features)]
    return pd.DataFrame(data, index=index, columns=cols)


def _make_labels(n_samples: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2, size=n_samples)
    index = [f"sample_{i}" for i in range(n_samples)]
    return pd.Series(values, index=index, name="label")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestAlignSamples:
    def test_common_subset(self):
        df1 = _make_df(10, 5)
        df2 = _make_df(8, 3)  # has samples 0–7
        df1.index = [f"sample_{i}" for i in range(10)]
        df2.index = [f"sample_{i}" for i in range(8)]
        a1, a2 = align_samples(df1, df2)
        assert len(a1) == 8
        assert len(a2) == 8
        assert list(a1.index) == list(a2.index)

    def test_no_overlap_raises(self):
        df1 = pd.DataFrame({"a": [1]}, index=["s1"])
        df2 = pd.DataFrame({"b": [2]}, index=["s2"])
        with pytest.raises(ValueError, match="No common samples"):
            align_samples(df1, df2)


class TestVarianceFilter:
    def test_constant_columns_removed(self):
        df = _make_df(20, 10)
        # Make one column constant
        df["feat_0"] = 5.0
        result = remove_low_variance_features(df, threshold=1e-6)
        assert "feat_0" not in result.columns
        assert result.shape[1] == 9

    def test_all_variable_columns_kept(self):
        df = _make_df(20, 10)
        result = remove_low_variance_features(df, threshold=0.0)
        assert result.shape[1] == 10


class TestTopKFeatures:
    def test_correct_k(self):
        df = _make_df(20, 50)
        result = select_top_k_features(df, k=10)
        assert result.shape[1] == 10

    def test_k_larger_than_features_returns_all(self):
        df = _make_df(20, 5)
        result = select_top_k_features(df, k=100)
        assert result.shape[1] == 5


class TestNormalise:
    def test_zero_mean(self):
        df = _make_df(100, 10)
        result = normalise(df)
        np.testing.assert_allclose(result.mean(axis=0).abs().values, 0, atol=1e-6)

    def test_unit_std(self):
        df = _make_df(100, 10)
        result = normalise(df)
        # sklearn StandardScaler uses ddof=0; compare using numpy std with ddof=0
        stds = result.to_numpy().std(axis=0, ddof=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-5)


class TestLogTransform:
    def test_values_non_negative(self):
        df = _make_df(20, 10) * 100  # simulate count data
        result = log_transform(df)
        assert (result.values >= 0).all()

    def test_monotone(self):
        df = _make_df(20, 10) * 100
        result = log_transform(df)
        assert (result.values <= np.log1p(df.values.max())).all()


class TestMultiOmicsPreprocessor:
    def test_fit_transform_shapes(self):
        n = 30
        genomics = _make_df(n, 200, seed=1)
        transcriptomics = _make_df(n, 300, seed=2)
        proteomics = _make_df(n, 150, seed=3)
        labels = _make_labels(n, seed=4)

        preprocessor = MultiOmicsPreprocessor(top_k_features=50)
        matrices, y = preprocessor.fit_transform(genomics, transcriptomics, proteomics, labels)

        assert set(matrices.keys()) == {"genomics", "transcriptomics", "proteomics"}
        for name, mat in matrices.items():
            assert mat.ndim == 2
            assert mat.shape[0] == n
            assert mat.shape[1] <= 50

        assert y.shape == (n,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_fit_transform_aligned_samples(self):
        n_full = 30
        n_small = 20  # only 20 samples in proteomics
        genomics = _make_df(n_full, 100, seed=1)
        transcriptomics = _make_df(n_full, 100, seed=2)
        proteomics = _make_df(n_small, 100, seed=3)
        labels = _make_labels(n_full, seed=4)

        # Only first 20 samples are in all three layers
        preprocessor = MultiOmicsPreprocessor(top_k_features=20)
        matrices, y = preprocessor.fit_transform(genomics, transcriptomics, proteomics, labels)

        assert matrices["genomics"].shape[0] == n_small
        assert y.shape[0] == n_small

    def test_save_and_load(self, tmp_path):
        n = 20
        genomics = _make_df(n, 60, seed=1)
        transcriptomics = _make_df(n, 60, seed=2)
        proteomics = _make_df(n, 60, seed=3)
        labels = _make_labels(n, seed=4)

        preprocessor = MultiOmicsPreprocessor(top_k_features=20)
        matrices, y = preprocessor.fit_transform(genomics, transcriptomics, proteomics, labels)
        MultiOmicsPreprocessor.save(matrices, y, tmp_path)

        loaded_matrices, loaded_y = MultiOmicsPreprocessor.load(tmp_path)
        for name in matrices:
            np.testing.assert_array_equal(matrices[name], loaded_matrices[name])
        np.testing.assert_array_equal(y, loaded_y)

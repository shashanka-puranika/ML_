"""
Multi-omics data preprocessing for fungal pathogenicity prediction.

Handles loading and normalising genomics, transcriptomics, and proteomics data
from CSV files or a local SQLite database, and produces aligned sample matrices
ready for graph construction.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual omics loaders (CSV)
# ---------------------------------------------------------------------------


def load_genomics(path: str | Path) -> pd.DataFrame:
    """Load genomics data (gene presence/absence or SNP matrix).

    Expected format: rows = samples, columns = genomic features.
    Values should be numeric (0/1 for presence-absence, or continuous for SNPs).
    """
    df = pd.read_csv(path, index_col=0)
    logger.info("Loaded genomics data: %s samples, %s features", *df.shape)
    return df


def load_transcriptomics(path: str | Path) -> pd.DataFrame:
    """Load transcriptomics data (RNA-seq read counts or TPM values).

    Expected format: rows = samples, columns = gene identifiers.
    """
    df = pd.read_csv(path, index_col=0)
    logger.info("Loaded transcriptomics data: %s samples, %s features", *df.shape)
    return df


def load_proteomics(path: str | Path) -> pd.DataFrame:
    """Load proteomics data (protein abundance profiles).

    Expected format: rows = samples, columns = protein identifiers.
    """
    df = pd.read_csv(path, index_col=0)
    logger.info("Loaded proteomics data: %s samples, %s features", *df.shape)
    return df


def load_labels(path: str | Path) -> pd.Series:
    """Load binary pathogenicity labels.

    Expected format: single-column CSV with sample IDs as index.
    Values: 0 (non-pathogenic) or 1 (pathogenic).
    """
    df = pd.read_csv(path, index_col=0)
    labels = df.iloc[:, 0].astype(int)
    logger.info(
        "Loaded labels: %d samples (%d pathogenic, %d non-pathogenic)",
        len(labels),
        labels.sum(),
        (labels == 0).sum(),
    )
    return labels


# ---------------------------------------------------------------------------
# Database loaders (SQLite)
# ---------------------------------------------------------------------------


def load_from_database(
    db_path: str | Path,
    table_name: str,
    index_col: str = "sample_id",
) -> pd.DataFrame:
    """Load a table from a local SQLite database into a DataFrame.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    table_name:
        Name of the table to read.
    index_col:
        Column to use as the DataFrame index (default ``"sample_id"``).

    Returns
    -------
    pd.DataFrame
        Data with *index_col* set as the index.  All remaining columns are
        cast to numeric where possible; non-numeric columns are dropped with
        a warning.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
    finally:
        conn.close()

    if index_col not in df.columns:
        raise ValueError(
            f"Index column '{index_col}' not found in table '{table_name}'. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.set_index(index_col)

    # Cast columns to numeric; drop any that cannot be converted
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    non_numeric = numeric_df.columns[numeric_df.isna().all()]
    if len(non_numeric) > 0:
        logger.warning(
            "Dropped %d non-numeric columns from table '%s': %s",
            len(non_numeric),
            table_name,
            list(non_numeric),
        )
        numeric_df = numeric_df.drop(columns=non_numeric)

    logger.info(
        "Loaded table '%s' from database: %s samples, %s features",
        table_name,
        *numeric_df.shape,
    )
    return numeric_df


def load_labels_from_database(
    db_path: str | Path,
    table_name: str = "labels",
    index_col: str = "sample_id",
    label_col: str = "label",
) -> pd.Series:
    """Load binary pathogenicity labels from a local SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    table_name:
        Name of the labels table (default ``"labels"``).
    index_col:
        Column to use as the sample index (default ``"sample_id"``).
    label_col:
        Name of the column containing the binary labels (default ``"label"``).

    Returns
    -------
    pd.Series
        Integer label series with sample IDs as the index.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
    finally:
        conn.close()

    if index_col not in df.columns:
        raise ValueError(
            f"Index column '{index_col}' not found in table '{table_name}'."
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in table '{table_name}'."
        )

    df = df.set_index(index_col)
    labels = df[label_col].astype(int)
    logger.info(
        "Loaded labels from database: %d samples (%d pathogenic, %d non-pathogenic)",
        len(labels),
        labels.sum(),
        (labels == 0).sum(),
    )
    return labels


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def align_samples(*dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Return copies of all DataFrames restricted to their common sample index."""
    common = dataframes[0].index
    for df in dataframes[1:]:
        common = common.intersection(df.index)
    if len(common) == 0:
        raise ValueError("No common samples found across the provided omics datasets.")
    logger.info("Aligned to %d common samples.", len(common))
    return tuple(df.loc[common] for df in dataframes)


def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Drop features whose variance falls below *threshold*."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    selected = df.columns[selector.get_support()]
    logger.info(
        "Variance filtering: kept %d / %d features (threshold=%.4f)",
        len(selected),
        df.shape[1],
        threshold,
    )
    return df[selected]


def select_top_k_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the *k* features with the highest variance."""
    if k >= df.shape[1]:
        return df
    variances = df.var(axis=0)
    top_cols = variances.nlargest(k).index
    logger.info("Selected top %d features by variance.", k)
    return df[top_cols]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalise each feature column (zero mean, unit variance)."""
    scaler = StandardScaler()
    normalised = scaler.fit_transform(df)
    return pd.DataFrame(normalised, index=df.index, columns=df.columns)


def log_transform(df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """Apply log(x + pseudocount) transformation (useful for count-based transcriptomics/proteomics)."""
    return np.log(df + pseudocount)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


class MultiOmicsPreprocessor:
    """End-to-end preprocessing pipeline for multi-omics fungal data.

    Parameters
    ----------
    top_k_features:
        Number of top-variance features to retain per omics layer.
    variance_threshold:
        Minimum variance for a feature to be retained before top-k selection.
    log_transform_rna:
        Whether to apply log1p to transcriptomics data before normalisation.
    log_transform_prot:
        Whether to apply log1p to proteomics data before normalisation.
    """

    def __init__(
        self,
        top_k_features: int = 500,
        variance_threshold: float = 0.01,
        log_transform_rna: bool = True,
        log_transform_prot: bool = True,
    ) -> None:
        self.top_k_features = top_k_features
        self.variance_threshold = variance_threshold
        self.log_transform_rna = log_transform_rna
        self.log_transform_prot = log_transform_prot

    def fit_transform(
        self,
        genomics: pd.DataFrame,
        transcriptomics: pd.DataFrame,
        proteomics: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Preprocess all omics layers and return aligned numpy arrays.

        Returns
        -------
        omics_matrices : dict
            Keys are ``"genomics"``, ``"transcriptomics"``, ``"proteomics"``.
            Values are 2-D numpy arrays of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Integer label vector of shape ``(n_samples,)``.
        """
        # 1. Align samples across all modalities and labels
        genomics, transcriptomics, proteomics, labels_df = align_samples(
            genomics, transcriptomics, proteomics, labels.to_frame()
        )
        y = labels_df.iloc[:, 0].to_numpy(dtype=int)

        # 2. Optional log transform for count data
        if self.log_transform_rna:
            transcriptomics = log_transform(transcriptomics)
        if self.log_transform_prot:
            proteomics = log_transform(proteomics)

        # 3. Variance filtering + top-k selection + normalisation
        processed: Dict[str, np.ndarray] = {}
        for name, df in [
            ("genomics", genomics),
            ("transcriptomics", transcriptomics),
            ("proteomics", proteomics),
        ]:
            df = remove_low_variance_features(df, threshold=self.variance_threshold)
            df = select_top_k_features(df, k=self.top_k_features)
            df = normalise(df)
            processed[name] = df.to_numpy(dtype=np.float32)
            logger.info("'%s' final shape: %s", name, processed[name].shape)

        return processed, y

    @staticmethod
    def save(
        omics_matrices: Dict[str, np.ndarray],
        y: np.ndarray,
        output_dir: str | Path,
    ) -> None:
        """Persist processed arrays to *output_dir* as .npy files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, matrix in omics_matrices.items():
            np.save(output_dir / f"{name}.npy", matrix)
        np.save(output_dir / "labels.npy", y)
        logger.info("Saved processed data to '%s'.", output_dir)

    @staticmethod
    def load(input_dir: str | Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load previously saved processed arrays from *input_dir*."""
        input_dir = Path(input_dir)
        omics_names = ["genomics", "transcriptomics", "proteomics"]
        omics_matrices = {
            name: np.load(input_dir / f"{name}.npy") for name in omics_names
        }
        y = np.load(input_dir / "labels.npy")
        return omics_matrices, y

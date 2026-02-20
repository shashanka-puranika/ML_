# AI-Driven Predictive Modelling of Pathogenicity Determinants in Fungi

> Multi-omics integration and graph-based deep learning for fungal pathogenicity prediction.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Data Format](#data-format)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Quick Start](#quick-start)
7. [Module Reference](#module-reference)
8. [Running Tests](#running-tests)
9. [Configuration](#configuration)
10. [Contributing](#contributing)

---

## Project Overview

This project provides an end-to-end machine learning pipeline for predicting
whether a fungal species (or strain) is pathogenic based on multi-omics
evidence.  It integrates three complementary data layers:

| Layer           | Data type                              | Example sources                            |
|-----------------|----------------------------------------|--------------------------------------------|
| **Genomics**    | Gene presence/absence or SNP profiles | PHI-base, NCBI GenBank, FungiDB            |
| **Transcriptomics** | RNA-seq counts / TPM              | GEO, ArrayExpress, FungiDB                 |
| **Proteomics**  | Protein abundance profiles             | UniProt, PRIDE, FungiDB                    |

Samples from all three layers are integrated into a **sample-similarity graph**
where nodes are fungal strains/isolates and edges encode cross-sample similarity.
A **Graph Attention Network (GAT)** is then trained to classify each node as
pathogenic or non-pathogenic.

---

## Repository Structure

```
ML_/
├── configs/
│   └── config.yaml          # All hyperparameters and file paths
├── data/
│   ├── raw/                 # Place your raw CSV files here
│   └── processed/           # Preprocessed .npy arrays written here
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Multi-omics loading and normalisation
│   │   └── graph_construction.py # Build torch-geometric graphs
│   ├── models/
│   │   ├── gnn.py               # GAT and GCN model definitions
│   │   └── multiomics_gnn.py    # Multi-omics GNN with per-modality encoders
│   ├── training/
│   │   └── trainer.py           # Training loop with early stopping
│   └── evaluation/
│       └── metrics.py           # AUROC, AUPRC, F1, MCC, accuracy
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python ≥ 3.9
- pip

### Install dependencies

```bash
# (Recommended) create a virtual environment first
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU support** – If you have a CUDA-capable GPU, install the matching
> `torch` and `torch-geometric` wheels following the official instructions at
> https://pytorch.org and https://pytorch-geometric.readthedocs.io.

---

## Data Format

The pipeline supports two data sources: **CSV files** and a **local SQLite
database**.

### Option A – CSV files

Place CSV files in `data/raw/` with the following format:

| File                     | Rows          | Columns                | Values              |
|--------------------------|---------------|------------------------|---------------------|
| `genomics.csv`           | Sample IDs    | Gene/SNP identifiers   | 0/1 or float        |
| `transcriptomics.csv`    | Sample IDs    | Gene identifiers       | Read counts or TPM  |
| `proteomics.csv`         | Sample IDs    | Protein identifiers    | Abundance float     |
| `labels.csv`             | Sample IDs    | `label` (single col.)  | 0 or 1              |

The first column of every file is used as the sample index.  Only samples
present in **all four** files are used.

### Option B – SQLite database

Place a SQLite database file (e.g. `data/raw/omics.db`) containing the
following tables:

| Table              | Required columns                                     |
|--------------------|------------------------------------------------------|
| `genomics`         | `sample_id` + one column per genomic feature         |
| `transcriptomics`  | `sample_id` + one column per gene expression feature |
| `proteomics`       | `sample_id` + one column per protein feature         |
| `labels`           | `sample_id`, `label` (0 or 1)                        |

All tables use `sample_id` as the sample identifier.  Non-numeric columns
(other than `sample_id`) are automatically dropped during loading.  Enable
database loading by setting `data.database.enabled: true` in
`configs/config.yaml`.

---

## Pipeline Architecture

The pipeline is composed of five stages that transform raw omics data into
pathogenicity predictions:

### 1. Data Ingestion

Raw multi-omics data is loaded from **CSV files** or a **local SQLite
database**.  Each omics modality (genomics, transcriptomics, proteomics)
and the pathogenicity labels are loaded into pandas DataFrames.

### 2. Preprocessing

The `MultiOmicsPreprocessor` class applies the following steps:

1. **Sample alignment** – Restrict all modalities to their common sample set.
2. **Log transformation** – Apply `log(x + 1)` to count-based data
   (transcriptomics and proteomics) to stabilise variance.
3. **Low-variance filtering** – Remove near-constant features that carry
   little discriminative signal.
4. **Top-k feature selection** – Retain the *k* features with the highest
   variance per modality to keep the model tractable.
5. **Z-score normalisation** – Standardise each feature to zero mean and unit
   variance.

### 3. Graph Construction

Processed omics matrices are converted into a `torch_geometric.data.Data`
graph:

- **Node features** – Each node represents a fungal sample; its feature
  vector is the concatenation of all omics modalities.
- **Intra-omics edges** – Pairs of samples with cosine similarity above a
  configurable threshold within each omics layer.
- **Inter-omics edges** – Cross-layer edges connecting samples that are
  similar across different omics modalities.

### 4. Model Training

Three model architectures are available:

| Model                   | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `GCNPathogenicityModel` | Graph Convolutional Network baseline                               |
| `GATPathogenicityModel` | Graph Attention Network with multi-head attention (recommended)    |
| `MultiOmicsGNNModel`    | Per-modality encoders with attention-based fusion + GAT layers     |

Training uses cross-entropy loss with early stopping on a validation set.
The best checkpoint is automatically saved.

### 5. Evaluation

The held-out test set is evaluated using the following metrics:

- **Accuracy**
- **AUROC** (Area Under the Receiver Operating Characteristic)
- **AUPRC** (Area Under the Precision-Recall Curve)
- **F1 Score**
- **MCC** (Matthews Correlation Coefficient)

---

## Quick Start

### Using CSV files

```python
import yaml
from src.data.preprocessing import (
    MultiOmicsPreprocessor, load_genomics, load_transcriptomics,
    load_proteomics, load_labels,
)
from src.data.graph_construction import build_graph
from src.models.multiomics_gnn import MultiOmicsGNNModel
from src.training.trainer import Trainer, make_masks
from src.evaluation.metrics import evaluate

# 1. Load configuration
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Load raw data from CSV files
genomics       = load_genomics("data/raw/genomics.csv")
transcriptomics = load_transcriptomics("data/raw/transcriptomics.csv")
proteomics     = load_proteomics("data/raw/proteomics.csv")
labels         = load_labels("data/raw/labels.csv")

# 3. Preprocess
preprocessor = MultiOmicsPreprocessor(
    top_k_features=cfg["graph"]["top_k_features"]
)
omics_matrices, y = preprocessor.fit_transform(
    genomics, transcriptomics, proteomics, labels
)
preprocessor.save(omics_matrices, y, "data/processed")

# 4. Build graph
graph = build_graph(
    omics_matrices, y,
    intra_threshold=cfg["graph"]["correlation_threshold"],
)

# 5. Create train/val/test splits
n = graph.num_nodes
train_mask, val_mask, test_mask = make_masks(
    n,
    test_size=cfg["data"]["test_size"],
    val_size=cfg["data"]["val_size"],
    seed=cfg["data"]["random_seed"],
)

# 6. Build model
omics_dims = {name: mat.shape[1] for name, mat in omics_matrices.items()}
model = MultiOmicsGNNModel(
    omics_dims=omics_dims,
    hidden_channels=cfg["model"]["hidden_channels"],
    num_layers=cfg["model"]["num_layers"],
    heads=cfg["model"]["heads"],
    dropout=cfg["model"]["dropout"],
)
omics_splits = list(omics_dims.values())

# 7. Train
trainer = Trainer(
    model=model,
    graph=graph,
    train_mask=train_mask,
    val_mask=val_mask,
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
    epochs=cfg["training"]["epochs"],
    patience=cfg["training"]["patience"],
    multiomics_splits=omics_splits,
)
history = trainer.train()

# 8. Evaluate on test set
metrics = evaluate(model, graph, test_mask, multiomics_splits=omics_splits)
print(metrics)
```

### Using a local SQLite database

```python
from src.data.preprocessing import (
    MultiOmicsPreprocessor, load_from_database, load_labels_from_database,
)

# Load data from a local SQLite database
db_path = "data/raw/omics.db"
genomics       = load_from_database(db_path, "genomics")
transcriptomics = load_from_database(db_path, "transcriptomics")
proteomics     = load_from_database(db_path, "proteomics")
labels         = load_labels_from_database(db_path)

# The rest of the pipeline is identical – preprocess, build graph, train, evaluate
preprocessor = MultiOmicsPreprocessor(top_k_features=500)
omics_matrices, y = preprocessor.fit_transform(
    genomics, transcriptomics, proteomics, labels
)
```

---

## Module Reference

### `src.data.preprocessing`

| Symbol | Description |
|--------|-------------|
| `load_genomics(path)` | Load genomics CSV → `pd.DataFrame` |
| `load_transcriptomics(path)` | Load transcriptomics CSV → `pd.DataFrame` |
| `load_proteomics(path)` | Load proteomics CSV → `pd.DataFrame` |
| `load_labels(path)` | Load label CSV → `pd.Series` |
| `load_from_database(db_path, table_name, index_col)` | Load a table from a SQLite database → `pd.DataFrame` |
| `load_labels_from_database(db_path, table_name, index_col, label_col)` | Load labels from a SQLite database → `pd.Series` |
| `align_samples(*dfs)` | Restrict all DataFrames to common sample index |
| `remove_low_variance_features(df, threshold)` | Drop near-constant features |
| `select_top_k_features(df, k)` | Keep top-k by variance |
| `normalise(df)` | Z-score normalise |
| `log_transform(df)` | Apply log1p |
| `MultiOmicsPreprocessor` | End-to-end preprocessing class |

### `src.data.graph_construction`

| Symbol | Description |
|--------|-------------|
| `build_intra_omics_edges(omics, threshold)` | Cosine-similarity edges within each layer |
| `build_inter_omics_edges(omics, threshold)` | Cosine-similarity edges across layers |
| `build_graph(omics, labels, ...)` | Assemble `torch_geometric.data.Data` |

### `src.models.gnn`

| Symbol | Description |
|--------|-------------|
| `GATPathogenicityModel` | Graph Attention Network (recommended) |
| `GCNPathogenicityModel` | Graph Convolutional Network (baseline) |

### `src.models.multiomics_gnn`

| Symbol | Description |
|--------|-------------|
| `OmicsEncoder` | MLP projector for a single omics modality |
| `MultiOmicsGNNModel` | Full multi-omics model with per-modality encoders and attention-based fusion |

### `src.training.trainer`

| Symbol | Description |
|--------|-------------|
| `make_masks(n, test_size, val_size, seed)` | Create random train/val/test boolean masks |
| `Trainer` | Training loop with early stopping and best-checkpoint saving |

### `src.evaluation.metrics`

| Symbol | Description |
|--------|-------------|
| `evaluate(model, graph, mask, ...)` | Compute accuracy, AUROC, AUPRC, F1, MCC |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All tuneable parameters live in `configs/config.yaml`.  The file is
self-documenting; key sections are:

- **data** – file paths, train/val/test split sizes, random seed, and optional
  database configuration
- **data.database** – SQLite database path, table names, and column mappings
  (set `enabled: true` to use database loading)
- **graph** – similarity thresholds, top-k feature count
- **model** – hidden channels, number of layers, attention heads, dropout
- **training** – epochs, learning rate, early-stopping patience
- **evaluation** – list of metrics to report

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install dev dependencies (`pip install -r requirements.txt`).
3. Run `pytest tests/` before opening a pull request.
4. Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions.

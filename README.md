# AI-Driven Predictive Modelling of Pathogenicity Determinants in Fungi

> Multi-omics integration and graph-based deep learning for fungal pathogenicity prediction.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Data Format](#data-format)
5. [Quick Start](#quick-start)
6. [3-Month Project Roadmap](#3-month-project-roadmap)
   - [Month 1 – Data Collection & Preprocessing](#month-1--data-collection--preprocessing)
   - [Month 2 – Model Development](#month-2--model-development)
   - [Month 3 – Evaluation, Analysis & Deployment](#month-3--evaluation-analysis--deployment)
7. [Module Reference](#module-reference)
8. [Running Tests](#running-tests)
9. [Configuration](#configuration)
10. [Contributing](#contributing)

---

## Project Overview

This project builds a machine learning pipeline to predict whether a fungal
species (or strain) is pathogenic based on multi-omics evidence.  The three
omics layers are:

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

Place CSV files in `data/raw/` with the following format:

| File                     | Rows          | Columns                | Values              |
|--------------------------|---------------|------------------------|---------------------|
| `genomics.csv`           | Sample IDs    | Gene/SNP identifiers   | 0/1 or float        |
| `transcriptomics.csv`    | Sample IDs    | Gene identifiers       | Read counts or TPM  |
| `proteomics.csv`         | Sample IDs    | Protein identifiers    | Abundance float     |
| `labels.csv`             | Sample IDs    | `label` (single col.)  | 0 or 1              |

The first column of every file is used as the sample index.  Only samples
present in **all four** files are used.

---

## Quick Start

```python
import yaml
import pandas as pd
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

# 2. Load raw data
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

---

## 3-Month Project Roadmap

The roadmap below breaks the project into weekly milestones spread across
three months.  Each week has concrete deliverables and the code in this
repository that supports them.

---

### Month 1 – Data Collection & Preprocessing

**Goal**: Assemble and clean a high-quality multi-omics dataset of fungal
strains with known pathogenicity phenotypes.

#### Week 1 – Literature & database survey

| Task | Detail |
|------|--------|
| Identify target fungi | Select 5–10 well-studied pathogenic species (e.g. *Candida albicans*, *Aspergillus fumigatus*, *Fusarium oxysporum*) and matched non-pathogenic relatives |
| Survey databases | PHI-base (pathogen–host interactions), FungiDB, NCBI GenBank, UniProt, GEO/ArrayExpress |
| Define phenotype labels | Binary label: 1 = confirmed pathogen, 0 = non-pathogen; record evidence type |
| Create `labels.csv` | One row per sample/strain, column `label` |

**Deliverable**: A curated list of fungal strains with confirmed pathogenicity
labels stored in `data/raw/labels.csv`.

#### Week 2 – Genomics data collection

| Task | Detail |
|------|--------|
| Download genome assemblies | Use NCBI Datasets or FungiDB bulk download |
| Call virulence-factor genes | Align against PHI-base gene set (BLAST or HMMER) |
| Build presence/absence matrix | Rows = strains, columns = virulence-associated genes |
| Save as `data/raw/genomics.csv` | Binary or normalised copy-number values |

**Tip**: Limit to known secretome, effector, and CAZyme gene families to keep
the feature space tractable (< 2 000 features initially).

#### Week 3 – Transcriptomics & Proteomics collection

| Task | Detail |
|------|--------|
| Download RNA-seq datasets | Search GEO with `fungus AND pathogenicity AND RNA-seq` |
| Normalise counts | Use TPM or DESeq2 size-factor normalisation |
| Download proteomics datasets | PRIDE archive; focus on secretome/effector studies |
| Align sample IDs | Ensure the same strain identifiers are used across all three CSV files |

**Tip**: Use a minimum of 50 samples per class to avoid severe class imbalance.

#### Week 4 – Data preprocessing

| Task | Code | Detail |
|------|------|--------|
| Run `MultiOmicsPreprocessor` | `src/data/preprocessing.py` | Remove low-variance features, log-transform count data, z-score normalise |
| Inspect distributions | `notebooks/exploratory_analysis.ipynb` | PCA, UMAP, class-balance checks |
| Handle missing values | Edit `preprocessing.py` if needed | Impute or drop samples |
| Save processed arrays | `data/processed/` | `.npy` files consumed by graph builder |

**Deliverable**: Cleaned, aligned numpy arrays for all three omics layers,
saved in `data/processed/`.

---

### Month 2 – Model Development

**Goal**: Build, train, and iteratively improve a graph-based deep learning
model.

#### Week 5 – Graph construction

| Task | Code | Detail |
|------|------|--------|
| Tune similarity threshold | `src/data/graph_construction.py` | Try `intra_threshold` values in {0.5, 0.6, 0.7, 0.8}; aim for ~5–20 edges per node |
| Visualise graph | `networkx` + `matplotlib` | Check connectivity; isolated nodes indicate threshold too high |
| Add biological priors | Optional: add pathway-based edges from KEGG/GO enrichment | Can replace or supplement correlation-based edges |

**Deliverable**: A `torch_geometric.data.Data` object with sensible
connectivity and node-feature dimensions.

#### Week 6 – Baseline models

| Task | Code | Detail |
|------|------|--------|
| Train GCN baseline | `src/models/gnn.py` → `GCNPathogenicityModel` | 2–3 layers, hidden=64 |
| Train GAT model | `src/models/gnn.py` → `GATPathogenicityModel` | 3 layers, heads=4, hidden=64 |
| Log metrics | `src/evaluation/metrics.py` | Record AUROC, AUPRC, F1, MCC on validation set |
| Compare to non-graph baseline | `sklearn` Random Forest on concatenated features | Provides a sanity-check lower bound |

**Deliverable**: Baseline performance numbers for GCN, GAT, and Random Forest.

#### Week 7 – Multi-omics GNN

| Task | Code | Detail |
|------|------|--------|
| Train `MultiOmicsGNNModel` | `src/models/multiomics_gnn.py` | Per-modality encoders + attention-based fusion |
| Ablation: single-omics | Remove 2 of 3 layers; retrain | Quantifies each layer's contribution |
| Ablation: no inter-omics edges | Set `include_inter_omics=False` | Quantifies cross-layer edges |
| Record all results | Table in notebook | Compare to baselines from Week 6 |

**Deliverable**: Trained `MultiOmicsGNNModel` weights in `checkpoints/`.

#### Week 8 – Hyperparameter tuning

| Task | Detail |
|------|--------|
| Grid search or random search | Vary `hidden_channels` {64,128,256}, `num_layers` {2,3,4}, `heads` {2,4,8}, `dropout` {0.2,0.3,0.5} |
| Update `configs/config.yaml` | Record best hyperparameter set |
| Re-train best configuration | Save final model checkpoint |

**Deliverable**: Best hyperparameter configuration and corresponding validation
AUROC.

---

### Month 3 – Evaluation, Analysis & Deployment

**Goal**: Rigorous test-set evaluation, biological interpretation, and a
reusable inference interface.

#### Week 9 – Test-set evaluation

| Task | Code | Detail |
|------|------|--------|
| Evaluate best model on held-out test set | `src/evaluation/metrics.py` | Report accuracy, AUROC, AUPRC, F1, MCC |
| Confidence intervals | Bootstrap (1 000 resamples) | Assess statistical reliability |
| Confusion matrix | `sklearn.metrics.ConfusionMatrixDisplay` | Understand false positive/negative patterns |

**Deliverable**: Final test-set performance table with confidence intervals.

#### Week 10 – Biological interpretation

| Task | Detail |
|------|--------|
| Extract GAT attention weights | `return_attention_weights=True` in `GATConv` forward call |
| Identify high-attention edges | Pairs of samples with consistently high attention across heads |
| Map top genomic features | Use variance ranking from `select_top_k_features` to map back to gene names |
| Pathway enrichment | Run top features through FungiDB or KEGG pathway tools |

**Deliverable**: List of candidate pathogenicity-associated genes/pathways
ranked by model attention, ready for wet-lab validation.

#### Week 11 – Robustness checks

| Task | Detail |
|------|--------|
| Cross-validation (5-fold) | Re-run training on 5 folds; report mean ± std AUROC |
| Leave-species-out | Hold out all strains of one species; tests generalisation to unseen fungi |
| Noise injection | Add Gaussian noise to features; assess degradation |
| Class imbalance | If imbalanced, try weighted `CrossEntropyLoss` or SMOTE |

**Deliverable**: Cross-validation results table; updated model if robustness
issues found.

#### Week 12 – Documentation & deployment

| Task | Detail |
|------|--------|
| Write inference script | `predict.py` – takes new sample CSVs, returns pathogenicity probability |
| Write `requirements.txt` | Already present; pin exact versions for reproducibility |
| Finalise `configs/config.yaml` | Document every parameter |
| Write unit tests | Already present in `tests/`; add any missing edge cases |
| Update this README | Add results table, citation, acknowledgements |

**Deliverable**: A fully documented, tested, and runnable codebase suitable
for sharing with collaborators or as a supplementary to a publication.

---

## Module Reference

### `src.data.preprocessing`

| Symbol | Description |
|--------|-------------|
| `load_genomics(path)` | Load genomics CSV → `pd.DataFrame` |
| `load_transcriptomics(path)` | Load transcriptomics CSV → `pd.DataFrame` |
| `load_proteomics(path)` | Load proteomics CSV → `pd.DataFrame` |
| `load_labels(path)` | Load label CSV → `pd.Series` |
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

- **data** – file paths, train/val/test split sizes, random seed
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

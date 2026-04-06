# GREED

Graph neural network embedding model that maps graphs to fixed-dimensional vectors such that the L2 distance between vectors approximates the graph edit distance (GED) between the corresponding graphs. Used to generate graph embeddings for downstream similarity search.

**Reference:** R. Ranjan et al., "GREED: A Neural Framework for Learning Graph Distance Functions", NeurIPS 2022.

## Directory Structure

```
GREED/
├── greed/                    # Core code
│   ├── train.py              # Training script
│   ├── models.py             # Neural network model (NormGEDModel)
│   ├── gisma_dataset.py      # Data loading and processing
│   ├── config.py             # Configuration (device, etc.)
│   ├── utils.py              # Utility functions
│   ├── fix_openmp.py         # Windows OpenMP fix
│   └── saved_models/         # Pre-trained models
│       ├── AIDS/
│       ├── PubChem/
│       ├── Chemical1M/
│       └── SYN/
└── README.md
```

## Dependencies

```bash
pip install torch torch-geometric numpy scipy tqdm pandas matplotlib scikit-learn
```

On Windows, import `fix_openmp` at the start of scripts to avoid OpenMP conflicts.

## Training

```bash
cd greed

# Train on AIDS dataset
python train.py --dataset AIDS --ged_file ../../Gisma/datasets/AIDS/ged_results.txt --max_pairs 100000 --epochs 50

# Train on other datasets
python train.py --dataset PubChem --ged_file ../../Gisma/datasets/PubChem/ged_results.txt --max_pairs 100000 --epochs 50
```

**Training data** (from `../Gisma/datasets/{dataset}/`):
- `db.txt` - Graph database file
- `ged_results.txt` - GED labels (format: `graph_id1,graph_id2,ged`); included for AIDS and PubChem

Trained models are saved to `greed/saved_models/{dataset}/`.

## Pre-trained Models

Pre-trained models are included for all four datasets:

| Dataset | Model Path |
|---------|-----------|
| AIDS | `greed/saved_models/AIDS/` |
| PubChem | `greed/saved_models/PubChem/` |
| Chemical1M | `greed/saved_models/Chemical1M/` |
| SYN | `greed/saved_models/SYN/` |

## Usage with Gisma

GREED provides the embedding component for the Gisma similarity search system:
1. Train a GREED model on a dataset
2. Use the trained model to generate embeddings for all database graphs
3. Gisma loads these embeddings for its search index

Pre-computed embeddings for AIDS and PubChem are included in `../Gisma/embeddings/`.

## Datasets

Graph datasets are in `../Gisma/datasets/`:
- `{dataset}/db.txt` - Database graphs
- `{dataset}/ged_results.txt` - GED labels for training (included for AIDS and PubChem)

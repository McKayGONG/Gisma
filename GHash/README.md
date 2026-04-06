# GHash

Neural network-based graph hashing for approximate similarity search. GHash trains a GNN to produce hash codes for graphs, uses an inverted index for fast candidate retrieval, and verifies candidates with exact GED computation.

**Reference:** Z. Qin et al., "GHashing: Semantic Graph Hashing for Approximate Similarity Search in Graph Databases", KDD 2020.

## Directory Structure

```
GHash/
├── model/                        # Core Python code
│   ├── TrainModel.py             # Model training (entry point)
│   ├── train.py                  # Training functions
│   ├── simple_search.py          # Similarity search (main entry point)
│   ├── config.py                 # Dataset configurations
│   ├── DataFetcher.py            # Data loading and preprocessing
│   ├── graphHashFunctions.py     # Neural network model
│   └── SavedModel/               # Pre-trained models and inverted indexes
│       ├── 0211_All_AIDS.ckpt.*          # AIDS model checkpoint
│       ├── 0211_All_PubChem.ckpt.*       # PubChem model checkpoint
│       ├── inverted_index_0211_All_AIDS.pkl
│       └── inverted_index_0211_All_PubChem.pkl
├── data/                         # Graph datasets (BSS format)
│   ├── AIDS/                     # train/graphs.bss, test/graphs.bss
│   └── PubChem/                  # train/graphs.bss, test/graphs.bss
├── main/                         # C++ GED verification code
├── run_tau_experiments.sh         # Batch tau experiments
└── run_margin_experiments.sh      # Batch margin experiments
```

## Dependencies

- Python 3
- TensorFlow 1.x
- numpy, scipy, networkx, tqdm

## Training

```bash
cd model
python TrainModel.py    # Edit config.py to change dataset (default: SYN)
```

Trained models are saved to `SavedModel/`.

## Similarity Search

```bash
cd model
python simple_search.py --dataset AIDS --tau 4 --error_margin 0.5 --query_start 0 --query_end 99 --use_parallel
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | (required) | AIDS, PubChem, Chemical1M, SYN |
| `--tau` | (required) | GED threshold |
| `--error_margin` | 0.5 | ML filtering margin (controls recall vs. speed) |
| `--query_start` | 0 | Start query index |
| `--query_end` | 99 | End query index |
| `--use_parallel` | False | Enable parallel GED verification |
| `--ged_method` | AStar-BMao | GED method: `AStar-BMao` or `BSS-GED` |
| `--no_save` | False | Skip saving results |

Results are saved to `results/similarity_search/{dataset}/tau{X}_margin{Y}/`.

## Batch Experiments

```bash
# On Linux, fix line endings first:
for f in run_*.sh; do sed -i 's/\r$//' "$f" && chmod +x "$f"; done

cd model

# Run tau experiments (margin=0.5)
bash ../run_tau_experiments.sh

# Run margin experiments
bash ../run_margin_experiments.sh
```

| Script | Description |
|--------|-------------|
| `run_tau_experiments.sh` | AIDS/PubChem: tau=2-12; SYN: tau=1-6; Chemical1M: tau=2-10 |
| `run_margin_experiments.sh` | Tests margin=0,1,2,4,8,16,32,64,128 at fixed tau |

## Pre-trained Models

| Dataset | Model Checkpoint | Inverted Index |
|---------|-----------------|----------------|
| AIDS | `SavedModel/0211_All_AIDS.ckpt` | `SavedModel/inverted_index_0211_All_AIDS.pkl` |
| PubChem | `SavedModel/0211_All_PubChem.ckpt` | `SavedModel/inverted_index_0211_All_PubChem.pkl` |

Chemical1M and SYN models are not included; retrain with `train.py`.

## Datasets

GHash uses its own BSS data format in `data/`:
- `data/{dataset}/train/graphs.bss` - Training (database) graphs
- `data/{dataset}/test/graphs.bss` - Test (query) graphs

Ground truth files are read from `../Gisma/datasets/{dataset}/ground_truth.txt`.

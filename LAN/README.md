# LAN

Learning-based approximate graph similarity search using HNSW proximity graphs. LAN navigates a proximity graph built over the database to find graphs within a given GED threshold of the query.

**Reference:** Y. Peng et al., "LAN: Learning-based Approximate k-Nearest Neighbor Search in Graph Databases", ICDE 2022.

## Directory Structure

```
LAN/
├── similarity_search.py         # Main search script (range query)
├── lan_original_search.py       # LAN search engine (HNSW + GED + batch expansion)
├── gisma_ged_calculator.py      # GED calculator (exact + approximate fallback)
├── run_all_full.py              # Batch experiment runner
├── merge_results.py             # Merge per-query results
├── mrk_model.py                 # Neighbor ranking model
├── mrk_train.py                 # Neighbor ranking model training
├── neigh_pruning_model_training.py  # Neighbor pruning model training
├── generate_node2vec_emb.py     # Node2Vec embedding generation
├── generate_all_training_data.py    # Training data generation
├── indexes/                     # Pre-built HNSW indexes
│   ├── AIDS_LAN_index.pkl
│   └── PubChem_LAN_index.pkl
└── *.sh                         # Experiment launch scripts
```

## Dependencies

```bash
pip install numpy networkx scipy lapjv tqdm torch dgl node2vec
```

## Search (Range Query)

Given a query graph and threshold tau, find all database graphs with GED <= tau.

```bash
# Basic usage
python similarity_search.py --dataset AIDS --tau 4 --timeout 10

# Specify query range and save results
python similarity_search.py --dataset AIDS --tau 4 --timeout 10 --query_start 0 --query_end 99 --save

# Merge per-query results into summary
python merge_results.py --dataset AIDS --tau 4 --query_start 0 --query_end 99
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tau` | (required) | GED threshold |
| `--dataset` | AIDS | Dataset: AIDS, PubChem, Chemical1M, SYN |
| `--query_start` | 0 | Start query index |
| `--query_end` | 99 | End query index |
| `--ef` | 50 | Search width (larger = more accurate but slower) |
| `--timeout` | 10 | GED computation timeout in seconds |
| `--pruning_ratio` | 0.2 | Neighbor pruning ratio |
| `--beam_width` | 100 | Beam search width for approximate GED |
| `--save` | False | Save results to file |

Results are saved to `results/{dataset}/tau{tau}/ratio_{ratio}/per_query/LAN/`.

## Pre-built Indexes

| Dataset | Index File | Distance Mode |
|---------|-----------|---------------|
| AIDS | `indexes/AIDS_LAN_index.pkl` | GED |
| PubChem | `indexes/PubChem_LAN_index.pkl` | GED |

Chemical1M and SYN indexes are not included due to size.

## Batch Experiments

```bash
# Run all dataset-tau combinations (24 workers in parallel)
python run_all_full.py
```

Runs 100 queries for each combination:
- AIDS: tau = 2, 4, 6, 8, 10, 12
- PubChem: tau = 2, 4, 6, 8, 10, 12
- SYN: tau = 1, 2, 3, 4, 5, 6
- Chemical1M: tau = 2, 4, 6, 8, 10, 12

## GED Computation

LAN uses a fallback strategy for GED computation:
1. First attempt exact GED (A* search with iteration limit)
2. On timeout, fall back to `min(VJ, Hungarian, Beam)` upper bounds

## Datasets

Graph datasets should be placed in `../Gisma/datasets/`:
- `{dataset}/db.txt` - Database graphs
- `{dataset}/queries.txt` - Query graphs (100 queries)
- `{dataset}/ground_truth.txt` - Ground truth results

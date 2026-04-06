# GED-via-Optimal-Transport

Approximate graph edit distance computation using optimal transport, extended for graph similarity search. Provides three GED estimation methods: GEDIOT (neural network), GEDGW (Gromov-Wasserstein, training-free), and GEDHOT (hybrid of both).

**Reference:** Q. Cheng et al., "Computing Approximate Graph Edit Distance via Optimal Transport", SIGMOD 2025.

## Directory Structure

```
GED-via-Optimal-Transport/
├── src/                          # Python source code
│   └── main.py                   # Main entry point
├── model_save/                   # Pre-trained models
│   └── GEDIOT/                   # GEDIOT models (AIDS, PubChem, Chemical1M, SYN)
├── generate_summary.py           # Generate summary reports
└── calculate_similarity_metrics.py
```

## Dependencies

```
Python 3.9
dgl==1.0.2, pot==0.8.2, networkx==3.1, numpy==1.26.4
scipy==1.12.0, pytorch==2.2.2, pyg==2.5.2
torchvision==0.17.2, texttable==1.6.4, tqdm==4.65.0
```

## Training

Train the GEDIOT neural network model:

```bash
# AIDS dataset
python src/main.py --dataset AIDS --model-name GEDIOT --epochs 20 --batch-size 128 \
    --delta-only --num-delta-graphs 100 --learning-rate 0.0001

# PubChem dataset (use smaller learning rate)
python src/main.py --dataset PubChem --model-name GEDIOT --epochs 20 --batch-size 128 \
    --delta-only --num-delta-graphs 100 --learning-rate 0.00001
```

The `--delta-only` flag is required to avoid O(N^2) memory usage. Models are saved to `model_save/GEDIOT/`.

## GED Estimation (Testing)

```bash
# GEDIOT (neural network, requires trained model)
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --path

# GEDHOT (hybrid method, requires trained model)
python src/main.py --model-name GEDHOT --dataset AIDS --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --GW --path

# GEDGW (training-free, Gromov-Wasserstein only)
python src/main.py --model-name GEDGW --dataset AIDS --GW --path
```

## Similarity Search

Given a set of candidate graphs (pre-filtered by another method), estimate GED for each candidate to determine which are within the threshold.

**Step 1: Generate candidate files using Gisma:**

```bash
cd ../Gisma
./build/GismaProject --mode export-candidates --dataset AIDS --tau_search 4 --alpha 12.0 --tau_index 8.0
```

This creates CSV files in `candidates/` with format: `query_id,tau_search,candidate_ids`.

**Step 2: Run similarity search:**

```bash
# Run GEDHOT on AIDS, tau=2,4,6,8,10,12
python src/main.py --similarity-search --batch-parallel --method GEDHOT \
    --dataset AIDS --candidates-dir candidates/AIDS \
    --tau-thresholds "2,4,6,8,10,12" --query-start 0 --query-end 99 \
    --model-epoch-start 20

# Run all three methods for comparison
python src/main.py --similarity-search --batch-parallel --method all \
    --dataset AIDS --candidates-dir candidates/AIDS \
    --tau-thresholds "2,4,6,8,10,12" --query-start 0 --query-end 99 \
    --model-epoch-start 20
```

**Method options:**

| Method | Description | Requires Training |
|--------|-------------|-------------------|
| `GEDGW` | Gromov-Wasserstein optimization | No |
| `GEDIOT` | Neural network prediction | Yes |
| `GEDHOT` | Hybrid (GEDGW + GEDIOT) | Yes |
| `all` | Run all three for comparison | Yes |

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `--similarity-search` | Enable similarity search mode |
| `--batch-parallel` | Use batch parallel mode (faster) |
| `--method` | GEDGW, GEDIOT, GEDHOT, or all |
| `--dataset` | AIDS, PubChem, Chemical1M, SYN |
| `--candidates-dir` | Directory containing candidate files |
| `--tau-thresholds` | Comma-separated GED thresholds |
| `--query-start/end` | Query range |
| `--model-epoch-start` | Model epoch to load (GEDIOT/GEDHOT) |

Results are saved to `results/similarity_search/{dataset}/tau{value}/`.

## Pre-trained Models

GEDHOT uses GEDIOT's trained model internally, so only GEDIOT checkpoints are needed.

| Dataset | GEDIOT Epochs |
|---------|--------------|
| AIDS | 1-20 |
| PubChem | 1-20 |
| Chemical1M | 1-5 |
| SYN | 1-10 |

## Datasets

For similarity search experiments, graph datasets should be placed in `../Gisma/datasets/`:
- `{dataset}/db.txt` - Database graphs
- `{dataset}/queries.txt` - Query graphs
- `{dataset}/ground_truth.txt` - Ground truth results

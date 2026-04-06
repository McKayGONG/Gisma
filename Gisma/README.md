# Gisma: A Giant-Step-Small-Step Indexing Framework for Approximate Similarity Search in Graph Databases

A high-performance two-layer indexing framework for approximate graph similarity search based on Graph Edit Distance (GED).

## Quick Start

```bash
# View all available options
./build/GismaProject --help
```

## Project Structure

```
Gisma/
├── src/                    # Source code
├── include/                # Header files
├── scripts/                # Build and analysis scripts
├── datasets/               # Graph datasets (AIDS, PubChem)
├── embeddings/             # Graph embedding files
├── EPFs/                   # Edit Path Forest index files
└── NetDags/                # NetDag index files
```

## Requirements

- CMake 3.10+
- C++17 compatible compiler (MSVC, GCC, or Clang)

## Compilation

```bash
cd Gisma
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

The executable will be generated at `build/GismaProject`.

## Usage

### 1. Index Construction

Build the NetDag and EPF index:

```bash
# PubChem
./build/GismaProject --mode construct \
    --dataset PubChem \
    --alpha 12.0 \
    --tau_index 8.0 \
    --use_parallel

# AIDS
./build/GismaProject --mode construct \
    --dataset AIDS \
    --alpha 12.0 \
    --tau_index 8.0 \
    --use_parallel
```

### 2. Search

```bash
# PubChem
./build/GismaProject --mode search \
    --dataset PubChem \
    --tau_search 4 \
    --q_start 0 \
    --q_end 99 \
    --alpha 12.0 \
    --tau_index 8.0

# AIDS
./build/GismaProject --mode search \
    --dataset AIDS \
    --tau_search 4 \
    --q_start 0 \
    --q_end 99 \
    --alpha 12.0 \
    --tau_index 8.0
```

### 3. Experiment Mode

#### Overall Runtime Comparison (App-BMao, AStar-BMao, Gisma)

Compare the three core methods at default tau:

```bash
./build/GismaProject -m experiment -s PubChem \
  --alpha 12.0 --tau_index 8.0 --tau_values "4" \
  --exp overall --q_start 0 --q_end 99

./build/GismaProject -m experiment -s PubChem \
  --alpha 12.0 --tau_index 8.0 --tau_values "8" \
  --exp overall --use_parallel --num_workers 100

./build/GismaProject -m experiment -s AIDS \
  --alpha 12.0 --tau_index 8.0 --tau_values "4" \
  --exp overall --q_start 0 --q_end 99

./build/GismaProject -m experiment -s AIDS \
  --alpha 12.0 --tau_index 8.0 --tau_values "8" \
  --exp overall --use_parallel --num_workers 100
```

#### Ablation Study of Gisma (Giant Step + Small Step)

Evaluate the contribution of the Giant Step (GS) and Small Step (SS) stages by comparing Gisma with variants that remove one or both stages. Method order: `App-BMao` (no GS, no SS), `Base+GS` (GS only), `Base+SS` (SS only), `Gisma` (full).

```bash
./build/GismaProject -m experiment -s PubChem \
  --alpha 12.0 --tau_index 8.0 --tau_values "4" \
  --exp ablation_gisma --q_start 0 --q_end 99

./build/GismaProject -m experiment -s PubChem \
  --alpha 12.0 --tau_index 8.0 --tau_values "8" \
  --exp ablation_gisma --use_parallel --num_workers 100

./build/GismaProject -m experiment -s AIDS \
  --alpha 12.0 --tau_index 8.0 --tau_values "4" \
  --exp ablation_gisma --q_start 0 --q_end 99

./build/GismaProject -m experiment -s AIDS \
  --alpha 12.0 --tau_index 8.0 --tau_values "8" \
  --exp ablation_gisma --use_parallel --num_workers 100
```

#### Ablation Study of EPT Optimizations

Evaluate the contribution of each optimization (search tree reuse, subtree pruning, lower-bound propagation) by comparing Gisma with variants that disable one optimization at a time. Methods are ordered by expected impact: `Gisma-no-reuse` (most impactful removed first), then `Gisma-no-SP`, `Gisma-no-LP`, and `Gisma` (full version).

```bash
./build/GismaProject -m experiment -s PubChem \
  --alpha 12.0 --tau_index 8.0 --tau_values "4" \
  --exp ablation_epf --q_start 0 --q_end 99

./build/GismaProject -m experiment -s PubChem \
  --alpha 12.0 --tau_index 8.0 --tau_values "8" \
  --exp ablation_epf --use_parallel --num_workers 100

./build/GismaProject -m experiment -s AIDS \
  --alpha 12.0 --tau_index 8.0 --tau_values "4" \
  --exp ablation_epf --q_start 0 --q_end 99

./build/GismaProject -m experiment -s AIDS \
  --alpha 12.0 --tau_index 8.0 --tau_values "8" \
  --exp ablation_epf --use_parallel --num_workers 100
```

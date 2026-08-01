# Gisma: A Giant-Step-Small-Step Indexing Framework for Approximate Similarity Search in Graph Databases

A high-performance two-layer indexing framework for approximate graph similarity search based on Graph Edit Distance (GED).

## Quick Start

```bash
# View all available options
./build/Release/GismaProject.exe --help
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

### Windows (MSVC)

```bash
cd Gisma
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

> **Note:** If you encounter linker errors about architecture mismatch (x86 vs x64), add `-A x64` to the cmake command:
> ```bash
> cmake -B build -A x64 -DCMAKE_BUILD_TYPE=Release
> ```

### Linux (GCC)

```bash
cd Gisma
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

The executable will be generated at `build/Release/GismaProject.exe` (Windows) or `build/GismaProject` (Linux).

## Usage

### 1. Index Construction

#### 1.1 Overall Construction (All-in-one)

Build NetDag and EPF index in one command:

```bash
./build/Release/GismaProject.exe --mode construct \
    --dataset PubChem \
    --alpha 12.0 \
    --tau_index 8.0 \
    --use_parallel
```

#### 1.2 Step-by-step Construction

```bash
./build/GismaProject -m construct_ND  -s PubChem --alpha 12 --tau_index 8
./build/GismaProject -m compute_paths -s PubChem --alpha 12 --tau_index 8
./build/GismaProject -m reassign      -s PubChem --alpha 12 --tau_index 8
./build/GismaProject -m construct_EPF -s PubChem --alpha 12 --tau_index 8 --use_parallel
```

### 2. Search

#### Single Query Search

```bash
./build/Release/GismaProject.exe --mode search \
    --dataset PubChem \
    --search_method Gisma \
    --tau_search 4 \
    --q_start 0 \
    --q_end 99 \
    --alpha 12.0 \
    --tau_index 8.0
```

### 3. Experiment Mode

Run batch experiments with multiple methods and tau values:

```bash
./build/GismaProject -m experiment \
    --dataset PubChem \
    --alpha 12.0 \
    --tau_index 8.0 \
    --tau_values "8" \
    --methods "Base+GS,Base+SS,Gisma" \
    --q_start 0 \
    --q_end 99 \
    --use_parallel
```

`--app_max_iter` controls the iteration budget of the approximate A\* search, which trades runtime
against recall. It is set automatically per dataset, so the commands here do not pass it; run
`--help` to see the values. Pass it explicitly to sweep the knob, for example when producing a
QPS-recall curve.

## Default Dataset Parameters

| Dataset | `--alpha` | `--tau_index` | `--tau_search` |
|---------|-----------|---------------|----------------|
| AIDS | 12 | 8 | 8 |
| PubChem | 12 | 8 | 8 |
| Chemical1M | 12 | 8 | 8 |
| SYN | 6 | 4 | 4 |


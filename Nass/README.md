# Nass

Index-based exact graph similarity search. Given a GED threshold, Nass finds all graphs in the database within that distance from the query graph.

**Reference:** Jongik Kim, "Boosting Graph Similarity Search through Pre-computation", SIGMOD 2021. ([arXiv:2004.01124](http://arxiv.org/abs/2004.01124))

## Directory Structure

```
Nass/
├── src/                  # C++ source files
├── include/              # Header files
├── indexes/              # Pre-built index files
│   ├── AIDS/             # AIDS indexes (tau 1-12)
│   ├── PubChem/          # PubChem indexes (tau 1-12)
│   └── SYN/              # SYN indexes (tau 1-6)
├── build_all_indexes.sh  # Script to build all indexes
├── search_all.sh         # Script to run all search experiments
└── Makefile
```

## Build

```bash
make
```

This produces two binaries: `nass` (search) and `nass-index` (index construction).

Tested on Ubuntu and macOS with GCC and `-O3` optimization.

## Search

```bash
nass <threshold> <data_file> [index_file] [-q queries_file]
```

**Examples:**

```bash
# Search without index (label filtering + exact GED verification)
./nass 4 ../Gisma/datasets/AIDS/db.txt -q ../Gisma/datasets/AIDS/queries.txt

# Search with pre-built index (faster)
./nass 4 ../Gisma/datasets/AIDS/db.txt indexes/AIDS/AIDS_tau4.idx -q ../Gisma/datasets/AIDS/queries.txt
```

If `-q` is not specified, Nass randomly selects 100 queries from the database.

## Index Construction

```bash
nass-index <threshold> <data_file> <index_file> [options]
```

**Options:**
- `-M <memory_MB>`: Memory limit in MB (default: 1000)
- `-p <threads>`: Number of threads (default: 8)

**Example:**

```bash
./nass-index 8 ../Gisma/datasets/AIDS/db.txt indexes/AIDS/AIDS_tau8.idx -p 8 -M 2000
```

Distributed index construction is also supported; see the source code for details.

## Batch Experiments

```bash
# Build all indexes
bash build_all_indexes.sh

# Run all search experiments
bash search_all.sh
```

## Data Format

Nass accepts two graph header formats:
- Original: `t # <id>`
- Gisma format: `ID <id>`

Vertices: `v <id> <label>`, Edges: `e <src> <dst> <label>`

## Datasets

Graph datasets should be placed in `../Gisma/datasets/`:
- `AIDS/db.txt`, `AIDS/queries.txt`
- `PubChem/db.txt`, `PubChem/queries.txt`
- `SYN/db.txt`, `SYN/queries.txt`
- `Chemical1M/db.txt`, `Chemical1M/queries.txt`

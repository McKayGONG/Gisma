# Gisma

Source code and datasets for the paper "Gisma: Giant-Step-Small-Step Indexing for Approximate Similarity Search in Graph Databases".

## Structure

| Directory | Description |
|-----------|-------------|
| `Gisma/` | Gisma indexing framework (C++) |
| `Nass/` | Nass baseline (C++) |
| `LAN/` | LAN baseline (Python) |
| `GHash/` | GHash baseline (Python + C++) |
| `GEDHOT/` | GED-via-Optimal-Transport baseline (Python) |
| `GREED/` | GREED graph embedding model (Python) |

## Datasets

Shared datasets are in `Gisma/datasets/`: AIDS (42K), PubChem (22K), Chemical1M (1M), SYN (1M).

## Quick Start

```bash
cd Gisma
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
./build/GismaProject --help
```

See each subdirectory's README for detailed usage.

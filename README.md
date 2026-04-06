# Gisma

Source code and datasets for the paper "Gisma: Giant-Step-Small-Step Indexing for Approximate Similarity Search in Graph Databases".

## Structure

| Directory | Description |
|-----------|-------------|
| `Gisma/` | Gisma, App-BMao, AStar-BMao |
| `Nass/` | Nass baseline |
| `LAN/` | LAN baseline |
| `GHash/` | GHashing baseline |
| `GEDHOT/` | GED-via-Optimal-Transport baseline |
| `GREED/` | GREED graph embedding model |

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

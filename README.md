# Gisma

Source code and datasets for the paper "Gisma: Giant-Step-Small-Step Indexing for Approximate Similarity Search in Graph Databases".

## Structure

| Directory | Description |
|-----------|-------------|
| `Gisma/` | Gisma, and the same competitors reachable through its experiment mode |
| `Graph_Edit_Distance/` | App-BMao and AStar-BMao, the full-scan competitors (see its `MODIFICATIONS.md`) |
| `Nass/` | Nass baseline |
| `LAN/` | LAN baseline |
| `GHash/` | GHashing baseline |
| `GEDHOT/` | GED-via-Optimal-Transport baseline |
| `GREED/` | GREED graph embedding model |

## Datasets

Shared datasets are in `Gisma/datasets/`: AIDS (42K), PubChem (22K), Chemical1M (1M), SYN (1M).

See each subdirectory's README for detailed usage.

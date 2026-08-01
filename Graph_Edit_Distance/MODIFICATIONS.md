# Origin and Modifications

This directory contains the A\* graph edit distance code of

> Lijun Chang, Xing Feng, Xuemin Lin, Lu Qin, Wenjie Zhang, Dian Ouyang.
> *Speeding Up GED Verification for Graph Similarity Search.* ICDE 2020.

taken from <https://github.com/LijunChang/Graph_Edit_Distance> (MIT License, see
`LICENSE.md`; the upstream README is kept as `README.md`). The copy here is at upstream
commit `244d25a` plus the changes listed below.

It is included so that the full-scan competitors reported in our paper can be reproduced.
The upstream code answers a similarity search but does not report the quantities we compare
against, so it cannot be used unmodified for that purpose.

## Changes relative to upstream

| Change | Why it is needed |
|--------|------------------|
| `load_db` also accepts the `ID <id>` graph header | so that the same database and query files can be fed to both systems |
| Parallel search (`-w`) and a query range (`--q_start` / `--q_end`) | the comparison is run with a fixed thread count on the same 100 queries |
| Recall, precision and IoU against a ground truth file (`--ground_truth`) | upstream reports neither; the averaging convention matches ours, including how queries with an empty result set are counted |
| Average per-query latency alongside total wall-clock time | a per-query cost that is not divided by the thread count, so the two systems can be compared at a fixed thread count. It is measured inside the worker threads, so it still reflects contention at that thread count |
| `--app_max_iter`: cap the A\* iterations per graph pair | turns the exact search into the approximate method used in the comparison |
| `--method AStar-BMao \| App-BMao` | selects the exact or the approximate variant explicitly |

One further change is unrelated to the measurements: the makefile now creates the `.obj`
directory before compiling. Upstream only recreates it *after* a successful link, so a fresh
checkout fails with `can't create .obj/main.o`.

No part of the search algorithm or of the lower bounds was changed.

## Building

```bash
cd Graph_Edit_Distance
make
```

This produces `ged`. Run `./ged -h` for the full option list; the upstream `README.md` describes
the original options and the graph file format. Note that the example commands in that file refer
to `datasets/` files which are not included here, since the datasets of the enclosing repository
are used instead.

## Usage

Run from the root of the enclosing repository, so that the dataset paths resolve.

Approximate search, the setting used for the comparison. `--app_max_iter` is the per-dataset
operating point at which recall reaches about 90 percent: 3000 for AIDS, 6000 for PubChem, 3000
for Chemical1M, 250 for SYN.

```bash
./Graph_Edit_Distance/ged \
  -d Gisma/datasets/PubChem/db.txt -q Gisma/datasets/PubChem/queries.txt \
  -m search -p astar -l BMao -t 8 --method App-BMao --app_max_iter 6000 \
  -w 100 --q_start 0 --q_end 99 --ground_truth Gisma/datasets/PubChem/ground_truth.txt
```

Exact search: `--method AStar-BMao` removes the iteration cap, so `--app_max_iter` is ignored.

```bash
./Graph_Edit_Distance/ged \
  -d Gisma/datasets/PubChem/db.txt -q Gisma/datasets/PubChem/queries.txt \
  -m search -p astar -l BMao -t 8 --method AStar-BMao \
  -w 100 --q_start 0 --q_end 99 --ground_truth Gisma/datasets/PubChem/ground_truth.txt
```

Use `-l LSa` for the LSa lower bound instead of BMao.

Options added here:

| Option | Meaning |
|--------|---------|
| `-w <n>` | worker threads; `0` means one per core |
| `--q_start` / `--q_end` | inclusive query range |
| `--ground_truth <file>` | enables recall, precision and IoU reporting |
| `--app_max_iter <n>` | A\* iteration cap per graph pair |
| `--method` | `App-BMao` (approximate) or `AStar-BMao` (exact) |

**Give both sides the same thread count when comparing.** `-w 0` means one thread per core, which
on a many-core machine gives this program far more threads than an indexed run configured with a
fixed worker count; memory-bandwidth contention then biases the comparison. Run comparisons on a
single dedicated machine, one dataset at a time.

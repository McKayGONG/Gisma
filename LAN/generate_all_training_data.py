#!/usr/bin/env python3
"""
Generate training data for PubChem, Chemical1M, SYN (AIDS already done).

For each dataset, runs 3 steps:
  Step 1: determine_nq  → gamma* + meta.json
  Step 2: mrk_data_gen  → mrk_train_data.json
  Step 3: gen_ori_data   → aids_train.perc{20,40,60,80}.data

Usage:
  python generate_all_training_data.py --dataset PubChem
  python generate_all_training_data.py --dataset all
  python generate_all_training_data.py --dataset PubChem --workers 8
  python generate_all_training_data.py --dataset PubChem --step 2  # only step 2
"""

import argparse
import json
import math
import logging
import os
import pickle
import random
import struct
import sys
import time

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GISMA_BASE = os.path.join(SCRIPT_DIR, '..', 'Gisma')

DATASET_CONFIGS = {
    'AIDS': {
        'dataset_path': os.path.join(GISMA_BASE, 'datasets', 'AIDS'),
        'index_file': os.path.join(SCRIPT_DIR, 'indexes', 'AIDS_LAN_index.pkl'),
        'embedding_file': os.path.join(GISMA_BASE, 'embeddings', 'AIDS', 'AIDS_embeddings.bin'),
    },
    'PubChem': {
        'dataset_path': os.path.join(GISMA_BASE, 'datasets', 'PubChem'),
        'index_file': os.path.join(SCRIPT_DIR, 'indexes', 'PubChem_LAN_index.pkl'),
        'embedding_file': os.path.join(GISMA_BASE, 'embeddings', 'PubChem', 'PubChem_embeddings.bin'),
    },
    'Chemical1M': {
        'dataset_path': os.path.join(GISMA_BASE, 'datasets', 'Chemical1M'),
        'index_file': os.path.join(SCRIPT_DIR, 'indexes', 'Chemical1M_LAN_index.pkl'),
        'embedding_file': os.path.join(GISMA_BASE, 'embeddings', 'Chemical1M', 'Chemical1M_embeddings.bin'),
    },
    'SYN': {
        'dataset_path': os.path.join(GISMA_BASE, 'datasets', 'SYN'),
        'index_file': os.path.join(SCRIPT_DIR, 'indexes', 'SYN_LAN_index.pkl'),
        'embedding_file': os.path.join(GISMA_BASE, 'embeddings', 'SYN', 'SYN_embeddings.bin'),
    },
}

D_OF_PG = 80
PERCS = [20, 40, 60, 80]


def load_embeddings(embedding_file):
    """Load GREED embeddings from binary file for distance computation."""
    print(f"  Loading embeddings from {embedding_file}...")
    with open(embedding_file, 'rb') as f:
        num_graphs, emb_dim = struct.unpack('ii', f.read(8))
        embeddings = np.zeros((num_graphs, emb_dim), dtype=np.float32)
        for i in range(num_graphs):
            gid = struct.unpack('i', f.read(4))[0]
            emb = np.frombuffer(f.read(emb_dim * 4), dtype=np.float32).copy()
            embeddings[i] = emb
    print(f"  {num_graphs} embeddings, dim={emb_dim}")
    return embeddings, emb_dim


def load_index(index_file):
    print(f"  Loading index from {index_file}...")
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    pg = index_data['proximity_graph']
    graph_map = index_data['graph_id_to_original']
    print(f"  PG: {pg.number_of_nodes()} nodes, {pg.number_of_edges()} edges")
    return pg, graph_map


def load_test_query_ids(dataset_path):
    gt_file = os.path.join(dataset_path, 'ground_truth.txt')
    test_ids = set()
    if not os.path.exists(gt_file):
        return test_ids
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if parts:
                test_ids.add(parts[0])
    print(f"  {len(test_ids)} test queries excluded")
    return test_ids


# ============================================================
# Step 1: Determine gamma*
# ============================================================
def step1_determine_nq(dataset_name, config, output_dir, num_queries=2400, seed=42):
    print(f"\n{'='*70}")
    print(f"  Step 1: Determine gamma* for {dataset_name}")
    print(f"{'='*70}")

    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    embeddings, emb_dim = load_embeddings(config['embedding_file'])

    # Load graph IDs from index
    with open(config['index_file'], 'rb') as f:
        index_data = pickle.load(f)
    graph_map = index_data['graph_id_to_original']
    all_ids = sorted(graph_map.keys(), key=int)
    print(f"  {len(all_ids)} graphs in index")

    test_ids = load_test_query_ids(config['dataset_path'])
    id_to_idx = {gid: i for i, gid in enumerate(all_ids)}

    # Sample training queries
    candidates = [gid for gid in all_ids if gid not in test_ids]
    train_query_ids = random.sample(candidates, min(num_queries, len(candidates)))
    print(f"  Sampled {len(train_query_ids)} training queries")

    # For each query, find 200-NN embedding distance
    target_k = 200
    nn_200_distances = []
    for qid in tqdm(train_query_ids, desc="  Computing 200-NN distances"):
        q_idx = id_to_idx.get(qid)
        if q_idx is None:
            continue
        q_emb = embeddings[q_idx]
        dists = np.linalg.norm(embeddings - q_emb, axis=1)
        dists_sorted = np.sort(dists)
        if len(dists_sorted) > target_k:
            nn_200_distances.append(float(dists_sorted[target_k]))

    nn_200_distances.sort()
    idx_90 = min(int(len(nn_200_distances) * 0.9), len(nn_200_distances) - 1)
    gamma_star = nn_200_distances[idx_90]

    print(f"  200-NN dist: min={nn_200_distances[0]:.2f}, "
          f"median={nn_200_distances[len(nn_200_distances)//2]:.2f}, "
          f"90th={gamma_star:.2f}, max={nn_200_distances[-1]:.2f}")
    print(f"  gamma* = {gamma_star:.2f}")

    # Compute avg |N_Q|
    nq_sizes = []
    for qid in tqdm(train_query_ids, desc="  Computing |N_Q|"):
        q_idx = id_to_idx.get(qid)
        if q_idx is None:
            continue
        q_emb = embeddings[q_idx]
        dists = np.linalg.norm(embeddings - q_emb, axis=1)
        nq_size = int(np.sum(dists < gamma_star)) - 1
        nq_sizes.append(nq_size)
    avg_nq = float(np.mean(nq_sizes))
    print(f"  avg |N_Q| = {avg_nq:.1f}")

    # Save meta
    meta = {
        'dataset': dataset_name,
        'gamma_star': float(gamma_star),
        'distance_type': 'greed_embedding',
        'avg_nq_size': avg_nq,
        'train_query_ids': train_query_ids,
        'seed': seed,
        'num_db_graphs': len(all_ids),
    }
    meta_file = os.path.join(output_dir, 'meta.json')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved to {meta_file}")
    return gamma_star


# ============================================================
# Step 2: Generate mrk_train_data.json
# ============================================================

# Global for worker processes (loaded once in main, shared via fork on Linux)
_worker_ged_calc = None


def _worker_init_noop():
    """No-op init: workers inherit _worker_ged_calc from main process via fork."""
    import logging as _logging
    _logging.getLogger('gisma_ged_calculator').setLevel(_logging.WARNING)
    import os as _os
    _os.environ['TQDM_DISABLE'] = '1'


def _worker_process_query(args):
    global _worker_ged_calc
    qid, nq_ids, pg_neighbors_map = args
    ged_calc = _worker_ged_calc

    q_graph = ged_calc.all_graphs.get(qid)
    if q_graph is None:
        return [], {'qid': qid, 'nq': len(nq_ids), 'neigh': 0, 'ged': 0, 'samples': 0}

    samples = []
    ged_count = 0
    total_neigh = 0

    for gid in nq_ids:
        pg_neighbors = pg_neighbors_map.get(gid, [])
        if len(pg_neighbors) < 2:
            continue
        total_neigh += len(pg_neighbors)

        neighbor_data = []
        for nid in pg_neighbors:
            nid_str = str(nid)
            n_graph = ged_calc.all_graphs.get(nid_str)
            if n_graph is None:
                continue
            ged = ged_calc.compute_ged(q_graph, n_graph)
            ged_count += 1
            neighbor_data.append({'id': nid_str, 'ged': float(ged)})

        if len(neighbor_data) >= 2:
            samples.append({
                'query_id': qid,
                'pg_node_id': gid,
                'neighbors': neighbor_data
            })

    stats = {'qid': qid, 'nq': len(nq_ids), 'neigh': total_neigh,
             'ged': ged_count, 'samples': len(samples)}
    return samples, stats


def step2_generate_json(dataset_name, config, output_dir, workers=1, nq_sample=1.0):
    print(f"\n{'='*70}")
    print(f"  Step 2: Generate mrk_train_data.json for {dataset_name}")
    print(f"{'='*70}")

    meta_file = os.path.join(output_dir, 'meta.json')
    if not os.path.exists(meta_file):
        print(f"  ERROR: {meta_file} not found. Run step 1 first.")
        return

    with open(meta_file) as f:
        meta = json.load(f)
    gamma_star = meta['gamma_star']
    train_query_ids = meta['train_query_ids']
    print(f"  gamma* = {gamma_star:.2f}, {len(train_query_ids)} training queries")

    pg, graph_map = load_index(config['index_file'])
    embeddings, _ = load_embeddings(config['embedding_file'])
    all_ids = sorted(graph_map.keys(), key=int)
    id_to_idx = {gid: i for i, gid in enumerate(all_ids)}

    # Pre-compute tasks
    print(f"  Pre-computing N_Q for each query (nq_sample={nq_sample})...")
    tasks = []
    skipped = 0
    total_nq_before = 0
    total_nq_after = 0

    for qid in tqdm(train_query_ids, desc="  Pre-computing N_Q"):
        q_idx = id_to_idx.get(qid)
        if q_idx is None:
            skipped += 1
            continue

        q_emb = embeddings[q_idx]
        dists = np.linalg.norm(embeddings - q_emb, axis=1)
        nq_indices = np.where(dists < gamma_star)[0]
        nq_ids = [all_ids[i] for i in nq_indices if all_ids[i] != qid]
        total_nq_before += len(nq_ids)

        if nq_sample < 1.0 and len(nq_ids) > 1:
            sample_size = max(1, int(len(nq_ids) * nq_sample))
            nq_ids = random.sample(nq_ids, sample_size)
        total_nq_after += len(nq_ids)

        if len(nq_ids) == 0:
            skipped += 1
            continue

        pg_neighbors_map = {}
        for gid in nq_ids:
            if gid in pg:
                pg_neighbors_map[gid] = list(pg.neighbors(gid))

        tasks.append((qid, nq_ids, pg_neighbors_map))

    print(f"  {len(tasks)} queries to process, {skipped} skipped")
    print(f"  N_Q total: {total_nq_before} -> {total_nq_after} (sample={nq_sample})")

    # Process
    all_samples = []
    total_ged = 0
    dataset_path = config['dataset_path']

    # Load GED calculator ONCE in main process (shared via fork to workers)
    import logging as _logging
    _logging.getLogger('gisma_ged_calculator').setLevel(_logging.WARNING)
    from gisma_ged_calculator import GismaGEDCalculator
    global _worker_ged_calc
    print(f"  Loading GED calculator (once)...")
    _worker_ged_calc = GismaGEDCalculator(dataset_path=dataset_path, timeout=0.1, beam_width=1)
    print(f"  GED calculator ready. {len(_worker_ged_calc.all_graphs)} graphs loaded.")

    if workers <= 1:
        ged_calc = _worker_ged_calc

        pbar = tqdm(tasks, desc="  Queries")
        for qid, nq_ids, pg_neighbors_map in pbar:
            q_graph = ged_calc.all_graphs.get(qid)
            if q_graph is None:
                continue
            q_ged = 0
            q_neigh = 0
            q_samples = 0
            for gid in nq_ids:
                pg_neighbors = pg_neighbors_map.get(gid, [])
                if len(pg_neighbors) < 2:
                    continue
                q_neigh += len(pg_neighbors)
                neighbor_data = []
                for nid in pg_neighbors:
                    nid_str = str(nid)
                    n_graph = ged_calc.all_graphs.get(nid_str)
                    if n_graph is None:
                        continue
                    ged = ged_calc.compute_ged(q_graph, n_graph)
                    total_ged += 1
                    q_ged += 1
                    neighbor_data.append({'id': nid_str, 'ged': float(ged)})
                if len(neighbor_data) >= 2:
                    all_samples.append({
                        'query_id': qid,
                        'pg_node_id': gid,
                        'neighbors': neighbor_data
                    })
                    q_samples += 1
            pbar.set_postfix_str(
                f"GED={total_ged}, samples={len(all_samples)}, avg={total_ged//max(1,pbar.n)}/q | "
                f"last: q={qid} |N_Q|={len(nq_ids)} neigh={q_neigh} ged={q_ged}"
            )
    else:
        print(f"  Starting {workers} workers (shared memory via fork)...")
        pool = Pool(processes=workers, initializer=_worker_init_noop)
        try:
            pbar = tqdm(pool.imap(_worker_process_query, tasks),
                        total=len(tasks), desc=f"  Queries ({workers}w)")
            for samples, stats in pbar:
                all_samples.extend(samples)
                total_ged += stats['ged']
                pbar.set_postfix_str(
                    f"GED={total_ged}, samples={len(all_samples)}, avg={total_ged//max(1,pbar.n)}/q | "
                    f"last: q={stats['qid']} |N_Q|={stats['nq']} neigh={stats['neigh']} ged={stats['ged']}"
                )
        except KeyboardInterrupt:
            print("\n  Ctrl+C, terminating workers...")
            pool.terminate()
            pool.join()
            return
        else:
            pool.terminate()
            pool.join()

    # Save JSON
    output_file = os.path.join(output_dir, 'mrk_train_data.json')
    data = {
        'dataset': dataset_name,
        'gamma_star': gamma_star,
        'num_samples': len(all_samples),
        'total_ged_computed': total_ged,
        'samples': all_samples
    }
    print(f"  Saving {len(all_samples)} samples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"  Done. GED computed: {total_ged}")


# ============================================================
# Step 3: Convert JSON to .data files (perc 20/40/60/80)
# ============================================================
def step3_generate_data_files(dataset_name, config, output_dir):
    print(f"\n{'='*70}")
    print(f"  Step 3: Generate .data files for {dataset_name}")
    print(f"{'='*70}")

    json_file = os.path.join(output_dir, 'mrk_train_data.json')
    if not os.path.exists(json_file):
        print(f"  ERROR: {json_file} not found. Run step 2 first.")
        return

    print(f"  Loading {json_file}...")
    with open(json_file) as f:
        data = json.load(f)
    print(f"  {data['num_samples']} samples")

    # Get padding ID = num_db_graphs (from meta or index)
    meta_file = os.path.join(output_dir, 'meta.json')
    with open(meta_file) as f:
        meta = json.load(f)
    padding_id = str(meta.get('num_db_graphs', 99999))
    padding_ged = 999999.0

    # dataset-specific output prefix
    ds_lower = dataset_name.lower()

    for perc in PERCS:
        output_file = os.path.join(SCRIPT_DIR, 'training_data', f'{ds_lower}_train.perc{perc}.data')
        print(f"  Generating perc={perc}% → {os.path.basename(output_file)}...")

        with open(output_file, 'w') as f:
            for sample in data['samples']:
                qid = sample['query_id']
                gid = sample['pg_node_id']
                neighbors = sample['neighbors']

                neighbors_sorted = sorted(neighbors, key=lambda x: x['ged'])
                num_positive = max(1, math.ceil(len(neighbors_sorted) * perc / 100))

                labels = []
                neigh_ids = []
                neigh_geds = []

                for i in range(D_OF_PG):
                    if i < len(neighbors_sorted):
                        labels.append(0 if i < num_positive else 1)
                        neigh_ids.append(neighbors_sorted[i]['id'])
                        neigh_geds.append(neighbors_sorted[i]['ged'])
                    else:
                        labels.append(1)
                        neigh_ids.append(padding_id)
                        neigh_geds.append(padding_ged)

                parts = [qid, gid, '0.0']
                parts.extend(str(l) for l in labels)
                parts.extend(neigh_ids)
                parts.extend(str(g) for g in neigh_geds)
                f.write(' '.join(parts) + '\n')

        # Verify
        with open(output_file) as f:
            first_line = f.readline().strip().split()
        n_pos = sum(1 for x in first_line[3:3+D_OF_PG] if x == '0')
        n_neg = sum(1 for x in first_line[3:3+D_OF_PG] if x == '1')
        print(f"    {data['num_samples']} lines, first sample: {n_pos} pos / {n_neg} neg")

    print(f"  Done! 4 files generated.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data for specified datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['AIDS', 'PubChem', 'Chemical1M', 'SYN', 'all'],
                        help='Dataset to process')
    parser.add_argument('--step', type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help='Run only this step (0=all three steps)')
    default_workers = max(1, int(os.cpu_count() * 0.7))
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'Parallel workers for step 2 (default: 70%% CPU = {default_workers})')
    parser.add_argument('--num_queries', type=int, default=2400,
                        help='Number of training queries for step 1 (default: 2400)')
    parser.add_argument('--nq_sample', type=float, default=1.0,
                        help='Sample ratio of N_Q per query in step 2 (default: 1.0, e.g. 0.001)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix for output dir (e.g. "_test" → mrk_training_data_SYN_test)')
    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    for ds_name in datasets:
        config = DATASET_CONFIGS[ds_name]
        output_dir = os.path.join(SCRIPT_DIR, 'training_data', f'mrk_training_data_{ds_name}{args.output_suffix}')

        print(f"\n{'#'*70}")
        print(f"# {ds_name}")
        print(f"# Output: {output_dir}")
        print(f"{'#'*70}")

        # Check prerequisites
        if not os.path.exists(config['index_file']):
            print(f"  ERROR: Index file not found: {config['index_file']}")
            print(f"  Run build_all_indexes.py first.")
            continue
        if not os.path.exists(config['embedding_file']):
            print(f"  ERROR: Embedding file not found: {config['embedding_file']}")
            continue

        start = time.time()

        if args.step in [0, 1]:
            step1_determine_nq(ds_name, config, output_dir,
                               num_queries=args.num_queries, seed=args.seed)

        if args.step in [0, 2]:
            step2_generate_json(ds_name, config, output_dir, workers=args.workers,
                                nq_sample=args.nq_sample)

        if args.step in [0, 3]:
            step3_generate_data_files(ds_name, config, output_dir)

        elapsed = time.time() - start
        print(f"\n  [{ds_name}] Total time: {elapsed:.1f}s ({elapsed/3600:.1f}h)")

    print(f"\nAll done!")

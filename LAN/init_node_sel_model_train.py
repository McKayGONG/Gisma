#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Init Node Selection Model Training

Train a model to predict the best entry point in the Proximity Graph (PG)
for a given query graph.

Training data generation:
  1. Use ~2400 DB graphs as pseudo-queries (from mrk meta.json, or sampled)
  2. For each pseudo-query, find top-N nearest by Node2Vec embedding distance
  3. Compute actual GED for those N candidates
  4. Top-K by GED -> positive (label=1)
  5. Sample neg_ratio of remaining DB graphs -> negative (label=0)

Architecture:
  Q (query graph): degree one-hot (20) -> GIN 2-layer -> mean pooling -> q_emb [emb_dim]
  G (DB graph): Node2Vec pre-computed embedding [emb_dim]
  concat(q_emb, g_emb) [emb_dim*2] -> MLP(128->1) -> sigmoid -> score in [0,1]

Usage:
  # Generate GED data + train (auto-caches GED computation)
  python init_node_sel_model_train.py --dataset AIDS

  # Only generate GED data (for running on server)
  python init_node_sel_model_train.py --dataset PubChem --step generate

  # Only train (load cached GED data)
  python init_node_sel_model_train.py --dataset PubChem --step train
"""

from __future__ import annotations

import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GINConv
import numpy as np
import os
import time
import json
import pickle
import random
import argparse
import copy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from neigh_pruning_model_training import (
    read_and_split_to_individual_graph,
    make_a_dglgraph,
    read_initial_gemb,
)


# ============================================================================
# Training data generation
# ============================================================================

def _get_db_graph_ids(db_file):
    """Extract graph IDs from db.txt (lightweight, no torch dependency)."""
    ids = []
    with open(db_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('ID '):
                gid = line.split()[1]
                ids.append(gid)
    return ids


def get_training_query_ids(dataset, gisma_base, num_queries, seed):
    """Sample pseudo-query IDs from DB graphs."""
    dataset_path = os.path.join(gisma_base, 'datasets', dataset)
    db_file = os.path.join(dataset_path, 'db.txt')
    all_db_ids = _get_db_graph_ids(db_file)

    rng = random.Random(seed)
    sampled = rng.sample(all_db_ids, min(num_queries, len(all_db_ids)))
    print(f"  Sampled {len(sampled)} training queries from {len(all_db_ids)} DB graphs")
    return sampled


def _worker_compute_query(args):
    """Worker: compute GED for one query's candidates.

    Each task carries its own graph subset (only the graphs this query needs).
    No global state or initializer required — works with spawn on any OS.

    Args:
        args: (qid, candidate_ids, graph_subset, gisma_exe, dataset_path, timeout, beam_width)

    Returns:
        (qid, [(gid, ged), ...]) sorted by GED
    """
    qid, candidate_ids, graph_subset, gisma_exe, dataset_path, timeout, beam_width = args

    # Create a lightweight GED calculator with only the needed graphs
    import logging as _logging
    _logging.getLogger('gisma_ged_calculator').setLevel(_logging.WARNING)
    from gisma_ged_calculator import GismaGEDCalculator

    ged_calc = object.__new__(GismaGEDCalculator)
    ged_calc.gisma_exe = gisma_exe
    ged_calc.dataset_path = dataset_path
    ged_calc.db_file = os.path.join(dataset_path, 'db.txt')
    ged_calc.embedding_file = ''
    ged_calc.use_app_for_computation = True
    ged_calc.timeout = timeout
    ged_calc.beam_width = beam_width
    ged_calc.ged_source_cache = {}
    ged_calc.query_embeddings = None
    ged_calc.current_query_emb = None
    ged_calc.embeddings = None
    ged_calc.all_graphs = graph_subset
    ged_calc.approx_methods_available = ['beam']
    try:
        import scipy
        ged_calc.approx_methods_available.append('hung')
    except ImportError:
        pass
    try:
        import lapjv
        ged_calc.approx_methods_available.append('vj')
    except ImportError:
        pass
    ged_calc.stats = {'total_calls': 0, 'ged_success': 0, 'ged_timeout': 0,
                      'ged_exception': 0, 'approx_fallback': 0}

    q_graph = graph_subset.get(qid)
    if q_graph is None:
        return qid, []

    import time as _time
    query_deadline = _time.time() + 300  # 5 min per query

    ged_list = []
    for cid in candidate_ids:
        if _time.time() > query_deadline:
            break
        c_graph = graph_subset.get(cid)
        if c_graph is None:
            continue
        ged = ged_calc.compute_ged(q_graph, c_graph)
        ged_list.append((cid, float(ged)))
    ged_list.sort(key=lambda x: x[1])
    return qid, ged_list


def _load_greed_embeddings(embedding_file):
    """Load GREED embeddings from binary file. Returns (embeddings, emb_dim, id_list)."""
    import struct
    print(f"  Loading GREED embeddings from {embedding_file}...")
    with open(embedding_file, 'rb') as f:
        num_graphs, emb_dim = struct.unpack('ii', f.read(8))
        ids = []
        embeddings = np.zeros((num_graphs, emb_dim), dtype=np.float32)
        for i in range(num_graphs):
            gid = struct.unpack('i', f.read(4))[0]
            ids.append(str(gid))
            emb = np.frombuffer(f.read(emb_dim * 4), dtype=np.float32).copy()
            embeddings[i] = emb
    print(f"  {num_graphs} embeddings, dim={emb_dim}")
    return embeddings, emb_dim, ids


def compute_init_ged_data(dataset, dataset_path, train_query_ids,
                          top_candidates, greed_emb_file,
                          cache_file, workers=1):
    """Compute GED for top-N GREED-embedding-nearest neighbors per training query.

    Phase 1: For each query, find top_candidates by GREED embedding distance (fast).
              GREED embedding distance is an unbiased estimate of GED.
    Phase 2: Compute actual GED for all collected pairs (parallel).

    Results are cached to avoid recomputation.

    Returns: {query_id: [(graph_id, ged), ...]} sorted by GED
    """
    print(f"  Computing GED for Init training data...")
    print(f"  {len(train_query_ids)} queries x {top_candidates} candidates")

    # ---- Phase 1: Collect all (query, candidates) pairs ----
    print(f"\n  Phase 1: Finding top-{top_candidates} by GREED embedding distance...")
    emb_matrix, greed_dim, all_ids = _load_greed_embeddings(greed_emb_file)
    id_to_idx = {gid: i for i, gid in enumerate(all_ids)}

    tasks = []  # [(qid_str, [candidate_ids])]
    total_pairs = 0

    for qid in tqdm(train_query_ids, desc="  Embedding search"):
        qid_str = str(qid)
        q_idx = id_to_idx.get(qid_str)
        if q_idx is None:
            continue

        q_emb = emb_matrix[q_idx]
        dists = np.linalg.norm(emb_matrix - q_emb, axis=1)
        sorted_indices = np.argsort(dists)

        candidates = []
        for idx in sorted_indices:
            cid = all_ids[idx]
            if cid != qid_str:
                candidates.append(cid)
                if len(candidates) >= top_candidates:
                    break

        tasks.append((qid_str, candidates))
        total_pairs += len(candidates)

    del emb_matrix  # free memory before loading GED calculator
    print(f"  {len(tasks)} queries, {total_pairs} total GED pairs to compute")

    # ---- Phase 2: Compute GED for all pairs ----
    print(f"\n  Phase 2: Computing GED ({workers} workers)...")

    # Load GED calculator ONCE in main process
    import logging as _logging
    _logging.getLogger('gisma_ged_calculator').setLevel(_logging.WARNING)
    from gisma_ged_calculator import GismaGEDCalculator
    print(f"  Loading GED calculator...")
    ged_calc = GismaGEDCalculator(
        dataset_path=dataset_path, timeout=0.1, beam_width=1)
    print(f"  GED calculator ready. {len(ged_calc.all_graphs)} graphs.")

    q2g_dist = {}
    total_ged = 0
    start_time = time.time()

    if workers <= 1:
        pbar = tqdm(total=total_pairs, desc="  GED computation", unit="pair")
        for qid_str, candidate_ids in tasks:
            q_graph = ged_calc.all_graphs.get(qid_str)
            if q_graph is None:
                continue
            ged_list = []
            for cid in candidate_ids:
                c_graph = ged_calc.all_graphs.get(cid)
                if c_graph is None:
                    continue
                ged = ged_calc.compute_ged(q_graph, c_graph)
                ged_list.append((cid, float(ged)))
            ged_list.sort(key=lambda x: x[1])
            q2g_dist[qid_str] = ged_list
            total_ged += len(ged_list)
            pbar.update(len(ged_list))
            elapsed = time.time() - start_time
            rate = total_ged / elapsed if elapsed > 0 else 0
            pbar.set_postfix(queries=len(q2g_dist), rate=f"{rate:.0f}/s")
        pbar.close()
    else:
        # Queue-based: each task = 1 query + its graph subset (~500 graphs)
        # Workers grab tasks from queue, no pre-assignment
        import multiprocessing
        worker_tasks = []
        for qid_str, candidate_ids in tasks:
            needed_ids = set(candidate_ids)
            needed_ids.add(qid_str)
            graph_subset = {gid: ged_calc.all_graphs[gid]
                            for gid in needed_ids if gid in ged_calc.all_graphs}
            worker_tasks.append((qid_str, candidate_ids, graph_subset,
                                 ged_calc.gisma_exe, dataset_path,
                                 ged_calc.timeout, ged_calc.beam_width))

        del ged_calc  # free main process memory

        ctx = multiprocessing.get_context('spawn')
        print(f"  Starting {workers} workers (spawn, queue-based)...")
        pool = ctx.Pool(processes=workers)
        try:
            pbar = tqdm(total=total_pairs, desc=f"  GED ({workers}w)", unit="pair")
            for qid_str, ged_list in pool.imap_unordered(
                    _worker_compute_query, worker_tasks):
                q2g_dist[qid_str] = ged_list
                total_ged += len(ged_list)
                pbar.update(len(ged_list))
                elapsed = time.time() - start_time
                rate = total_ged / elapsed if elapsed > 0 else 0
                pbar.set_postfix(queries=len(q2g_dist), rate=f"{rate:.0f}/s")
            pbar.close()
            pool.close()
            pool.join()
        except (KeyboardInterrupt, Exception) as e:
            print(f"\n  Interrupted: {e}. Saving {len(q2g_dist)} queries...")
            try:
                pool.terminate()
                pool.join(timeout=5)
            except Exception:
                pass

    elapsed = time.time() - start_time
    print(f"  Done: {len(q2g_dist)} queries, {total_ged} GED computed "
          f"in {elapsed:.0f}s ({total_ged/max(elapsed,1):.0f} GED/s)")

    # Save cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache_data = {
        'dataset': dataset,
        'top_candidates': top_candidates,
        'num_queries': len(q2g_dist),
        'total_ged_computed': total_ged,
        'computation_time_s': round(elapsed, 1),
        'q2g_dist': q2g_dist,
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    print(f"  Cached to {cache_file}")

    return q2g_dist


def build_query_data(query_ids, q2g_dist, db_id_set, top_k=200,
                     neg_ratio=0.1, max_neg=4000):
    """Build training/test data for Init model.

    For each query:
      - top_k nearest DB graphs (by GED) -> positive (label=1), including ties
      - Train: min(neg_ratio * remaining, max_neg) negatives -> negative (label=0)
      - Test: ALL remaining DB graphs -> negative (label=0)

    Returns:
      list of (qid, [graph_ids], [labels])
    """
    q_data = []

    for qid in tqdm(query_ids, desc="  Building query data", unit="q"):
        if qid not in q2g_dist:
            continue

        dists = q2g_dist[qid]  # sorted by GED ascending

        # Collect top-k nearest graphs, including ties at the k-th position
        top_k_ids = set()
        kth_ged = None
        count = 0
        for gid, ged in dists:
            if gid not in db_id_set:
                continue
            if count < top_k:
                top_k_ids.add(gid)
                kth_ged = ged
                count += 1
            elif ged == kth_ged:
                # tie with k-th position, also include
                top_k_ids.add(gid)
            else:
                break

        # Positives
        gid_list = list(top_k_ids)
        labels = [1.0] * len(gid_list)

        # Negatives (both train and test sample equally)
        neg_pool = list(db_id_set - top_k_ids)
        n_neg = min(int(len(neg_pool) * neg_ratio), max_neg)
        n_neg = max(1, n_neg)
        if n_neg < len(neg_pool):
            neg_indices = np.random.choice(len(neg_pool), size=n_neg, replace=False)
            neg_sample = [neg_pool[i] for i in neg_indices]
        else:
            neg_sample = neg_pool
        gid_list.extend(neg_sample)
        labels.extend([0.0] * len(neg_sample))

        q_data.append((qid, gid_list, labels))

    return q_data


# ============================================================================
# Model
# ============================================================================

class InitModel(nn.Module):
    """Init node selection model.

    Architecture:
      Q: degree one-hot (20) -> GIN 2-layer -> mean pooling -> q_emb [emb_dim]
      G: Node2Vec embedding [emb_dim]
      concat(q_emb, g_emb) -> MLP(128->1) -> sigmoid
    """

    def __init__(self, emb_dim=512):
        super().__init__()
        self.emb_dim = emb_dim

        # GIN encoder for query
        self.fc_init = nn.Linear(20, emb_dim)
        self.conv1 = GINConv(None, 'mean')
        self.conv2 = GINConv(None, 'mean')
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)

        # MLP classifier
        self.fc1 = nn.Linear(emb_dim * 2, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

    def encode_query(self, dg):
        """Encode a query DGL graph -> embedding vector [emb_dim]."""
        dg.ndata['h2'] = self.fc_init(dg.ndata['h'])
        dg.ndata['h2'] = self.relu(self.bn1(self.conv1(dg, dg.ndata['h2'])))
        dg.ndata['h2'] = self.relu(self.bn2(self.conv2(dg, dg.ndata['h2'])))
        q_emb = dgl.mean_nodes(dg, 'h2').squeeze()
        return q_emb

    def encode_query_cg(self, cg_data):
        """Encode a query using CG compressed graph -> embedding vector [emb_dim].
        Init model uses self-loops in graphs during training/inference."""
        from cg_utils import cg_gin_forward
        return cg_gin_forward(cg_data, self.fc_init, self.bn1, self.bn2, self.relu,
                              model_has_selfloop=True).squeeze()

    def forward(self, q_dg, g_embs):
        """
        Args:
            q_dg: DGL graph for query (with self-loop, degree one-hot features)
            g_embs: [N, emb_dim] tensor of selected DB graph embeddings

        Returns:
            scores: [N] tensor of sigmoid scores
        """
        q_emb = self.encode_query(q_dg)  # [emb_dim]
        q_repeated = q_emb.unsqueeze(0).expand(g_embs.shape[0], -1)  # [N, emb_dim]

        H = torch.cat([q_repeated, g_embs], dim=1)  # [N, emb_dim*2]
        H = self.relu(self.bn_fc(self.fc1(H)))
        scores = torch.sigmoid(self.fc2(H)).squeeze(-1)  # [N]
        return scores


# ============================================================================
# Loss and evaluation
# ============================================================================

def weighted_bce(output, target, weights=None):
    """Weighted binary cross entropy loss."""
    output = torch.clamp(output, min=1e-6, max=1 - 1e-6)
    if weights is not None:
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))


def eval_model(model, q_data, qid2dg, gid2emb, emb_dim, top_k=200):
    """Evaluate model on test data. Returns (avg_auc, avg_precision@top_k)."""
    model.eval()
    total_auc = 0.0
    total_prec = 0.0
    count = 0

    with torch.no_grad():
        for qid, gid_list, label_list in q_data:
            if qid not in qid2dg:
                continue

            q_dg = qid2dg[qid]
            device = next(model.parameters()).device
            g_embs = []
            for gid in gid_list:
                if gid in gid2emb:
                    g_embs.append(gid2emb[gid])
                else:
                    g_embs.append(torch.zeros(emb_dim, device=device))
            g_embs = torch.stack(g_embs)

            scores = model(q_dg, g_embs)
            scores_np = scores.cpu().numpy()
            labels_np = np.array(label_list)

            # AUC
            if len(set(labels_np)) > 1:
                auc = roc_auc_score(labels_np, scores_np)
                total_auc += auc

            # Precision@top_k
            ranked = sorted(zip(scores_np, labels_np), key=lambda x: -x[0])
            k = min(top_k, len(ranked))
            top = ranked[:k]
            prec = sum(1 for _, lb in top if lb == 1.0) / k
            total_prec += prec

            count += 1

    if count == 0:
        return 0.0, 0.0
    return total_auc / count, total_prec / count


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Init node selection model')
    parser.add_argument('--dataset', type=str, default='AIDS',
                        choices=['AIDS', 'PubChem', 'SYN', 'Chemical1M'])
    parser.add_argument('--step', type=str, default='all',
                        choices=['generate', 'train', 'all'],
                        help='generate=compute GED data only, train=load cache and train, '
                             'all=both (default: all)')
    parser.add_argument('--epochs', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--top_candidates', type=int, default=500,
                        help='Number of embedding-nearest candidates to compute GED for (default: 500)')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Number of GED-nearest graphs to label as positive (default: 200)')
    parser.add_argument('--neg_ratio', type=float, default=0.1,
                        help='Fraction of remaining DB to sample as negatives (default: 0.1 = 10%%)')
    parser.add_argument('--max_neg', type=int, default=4000,
                        help='Max negatives per query (default: 4000)')
    parser.add_argument('--num_queries', type=int, default=2400,
                        help='Number of pseudo-queries if no mrk meta.json (default: 2400)')
    default_workers = max(1, int(os.cpu_count() * 0.7))
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'Parallel workers for GED computation (default: 70%% CPU = {default_workers})')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of queries per gradient update (default: 200)')
    parser.add_argument('--lr_step', type=int, default=5,
                        help='LR decay step size in epochs (default: 5)')
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                        help='LR decay factor (default: 0.9)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of queries for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DGL on Windows has no CUDA backend; force CPU if DGL can't use CUDA
    try:
        import dgl
        _test_g = dgl.graph(([0], [1]))
        _test_g.to(device)
        del _test_g
    except Exception:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = args.dataset
    ds_lower = dataset.lower()
    emb_dim = args.emb_dim

    # Paths
    gisma_base = os.path.join(SCRIPT_DIR, '..', 'Gisma')
    dataset_path = os.path.join(gisma_base, 'datasets', dataset)
    db_file = os.path.join(dataset_path, 'db.txt')
    greed_emb_file = os.path.join(gisma_base, 'embeddings', dataset, f'{dataset}_embeddings.bin')
    emb_dir = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{emb_dim}')
    pkl_path = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{emb_dim}_graph_emb.pkl')

    cache_dir = os.path.join(SCRIPT_DIR, 'training_data', f'init_training_data_{dataset}')
    cache_file = os.path.join(cache_dir, 'init_ged_cache.json')
    model_save_dir = os.path.join(SCRIPT_DIR, 'models', 'init', ds_lower)

    print("=" * 60)
    print(f"  Init Model: {dataset}  (step={args.step})")
    print(f"  top_candidates={args.top_candidates}, top_k={args.top_k}, "
          f"neg_ratio={args.neg_ratio}")
    print("=" * 60)

    # ==================================================================
    # Step: Generate GED data
    # ==================================================================
    if args.step in ('generate', 'all'):
        print("\n--- Generate Init training data ---")

        # Get training query IDs (pseudo-queries from DB)
        print("Getting training query IDs...")
        train_query_ids = get_training_query_ids(
            dataset, gisma_base, args.num_queries, args.seed)

        # Compute GED for top_candidates embedding-nearest neighbors
        print("Computing GED data...")
        q2g_dist = compute_init_ged_data(
            dataset, dataset_path, train_query_ids,
            args.top_candidates, greed_emb_file, cache_file,
            workers=args.workers)

        # Show sample
        for qid in list(q2g_dist.keys())[:3]:
            dists = q2g_dist[qid]
            print(f"    query {qid}: {len(dists)} candidates, "
                  f"min_ged={dists[0][1]}, max_ged={dists[-1][1]}")

        if args.step == 'generate':
            print("\nGeneration done. Run with --step train to train the model.")
            exit(0)

    # ==================================================================
    # Step: Train
    # ==================================================================
    print("\n--- Train Init model ---")

    # Load cached GED data if not already loaded
    if 'q2g_dist' not in dir():
        if not os.path.exists(cache_file):
            print(f"ERROR: Cache not found: {cache_file}")
            print(f"Run with --step generate first.")
            exit(1)
        print("Loading cached GED data...")
        with open(cache_file) as f:
            cached = json.load(f)
        q2g_dist = {}
        for qid, pairs in cached['q2g_dist'].items():
            q2g_dist[qid] = [(str(p[0]), float(p[1])) for p in pairs]
        print(f"  {len(q2g_dist)} queries loaded")

    # Load DB graph IDs
    print("Loading DB graphs...")
    db_graphs = read_and_split_to_individual_graph(db_file, 0, 10000000)
    db_graph_ids = [g.graph.get('id') for g in db_graphs]
    db_id_set = set(db_graph_ids)
    print(f"  {len(db_graph_ids)} DB graphs")

    # Build DGL graphs for pseudo-queries (they are DB graphs)
    print("Building DGL graphs for pseudo-queries...")
    qid2dg = {}
    query_ids_in_data = set(q2g_dist.keys())
    count = 0
    for g in tqdm(db_graphs, desc="  DGL conversion", unit="g"):
        gid = g.graph.get('id')
        if gid in query_ids_in_data:
            dg = make_a_dglgraph(g)
            dg = dgl.add_self_loop(dg)
            qid2dg[gid] = dg.to(device)
            count += 1
    print(f"  {count} pseudo-query DGL graphs built")
    del db_graphs

    # Load Node2Vec embeddings
    print("Loading Node2Vec embeddings...")
    gid2emb = read_initial_gemb(emb_dir, pkl_path=pkl_path)
    for gid in gid2emb:
        gid2emb[gid] = gid2emb[gid].to(device)
    print(f"  {len(gid2emb)} graph embeddings loaded (on {device})")

    # Train/test split
    all_query_ids = sorted(q2g_dist.keys(), key=lambda x: int(x))
    n_train = int(len(all_query_ids) * args.train_ratio)

    rng = np.random.RandomState(args.seed)
    shuffled_ids = list(all_query_ids)
    rng.shuffle(shuffled_ids)
    train_qids = shuffled_ids[:n_train]
    test_qids = shuffled_ids[n_train:]
    print(f"\nQuery split: {len(train_qids)} train, {len(test_qids)} test")

    # Build datasets
    print("\nBuilding training data...")
    train_data = build_query_data(train_qids, q2g_dist, db_id_set,
                                  top_k=args.top_k, neg_ratio=args.neg_ratio,
                                  max_neg=args.max_neg)
    total_pos = sum(sum(1 for lb in labels if lb == 1.0) for _, _, labels in train_data)
    total_neg = sum(sum(1 for lb in labels if lb == 0.0) for _, _, labels in train_data)
    print(f"  Train: {len(train_data)} queries, "
          f"{total_pos} positives, {total_neg} negatives "
          f"(ratio 1:{total_neg // max(total_pos, 1)})")

    print("Building test data...")
    test_data = build_query_data(test_qids, q2g_dist, db_id_set,
                                 top_k=args.top_k, neg_ratio=args.neg_ratio,
                                 max_neg=args.max_neg)
    total_pos_t = sum(sum(1 for lb in labels if lb == 1.0) for _, _, labels in test_data)
    total_neg_t = sum(sum(1 for lb in labels if lb == 0.0) for _, _, labels in test_data)
    print(f"  Test: {len(test_data)} queries, "
          f"{total_pos_t} positives, {total_neg_t} negatives")

    # Train
    os.makedirs(model_save_dir, exist_ok=True)
    model = InitModel(emb_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    batch_size = min(args.batch_size, len(train_data))

    print(f"\nTraining config:")
    print(f"  epochs={args.epochs}, lr={args.lr}, emb_dim={emb_dim}")
    print(f"  top_candidates={args.top_candidates}, top_k={args.top_k}, neg_ratio={args.neg_ratio}")
    print(f"  batch_size={batch_size}, lr_step={args.lr_step}, lr_gamma={args.lr_gamma}")

    best_auc = 0.0
    best_epoch = -1
    best_model_state = None
    early_stop_patience = 200
    epochs_without_improvement = 0

    total_start = time.time()
    epoch_bar = tqdm(range(args.epochs), desc="Epochs", unit="ep")

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        query_count = 0

        # Shuffle training order
        indices = list(range(len(train_data)))
        np.random.shuffle(indices)

        # Process queries with gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for i in indices:
            qid, gid_list, label_list = train_data[i]
            if qid not in qid2dg:
                continue

            q_dg = qid2dg[qid]

            # Build embedding tensor for selected graphs
            g_embs = []
            for gid in gid_list:
                if gid in gid2emb:
                    g_embs.append(gid2emb[gid])
                else:
                    g_embs.append(torch.zeros(emb_dim, device=device))
            g_embs = torch.stack(g_embs)

            # Forward + loss
            scores = model(q_dg, g_embs)
            gt = torch.tensor(label_list, device=device)
            loss = weighted_bce(scores, gt, weights=[1.0, 10.0])

            # Accumulate gradient (divide by batch_size for mean)
            (loss / batch_size).backward()
            accum_loss += loss.item()
            accum_count += 1
            query_count += 1

            # Step optimizer every batch_size queries
            if accum_count >= batch_size:
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0

        # Handle remaining queries
        if accum_count > 0:
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += accum_loss / accum_count

        scheduler.step()

        avg_loss = epoch_loss / max(query_count // batch_size + (1 if query_count % batch_size else 0), 1)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_auc, test_prec = eval_model(model, test_data, qid2dg,
                                             gid2emb, emb_dim, top_k=args.top_k)

            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                auc=f"{test_auc:.4f}",
                prec=f"{test_prec:.4f}",
                best=f"e{best_epoch}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            if test_auc > best_auc:
                best_auc = test_auc
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 10

            if epochs_without_improvement >= early_stop_patience:
                epoch_bar.close()
                print(f"\nEarly stopping at epoch {epoch}, "
                      f"best AUC={best_auc:.4f} at epoch {best_epoch}")
                break
        else:
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                auc=f"{best_auc:.4f}",
                best=f"e{best_epoch}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}")

    total_time = time.time() - total_start
    print(f"\nTraining done. Total time: {total_time:.1f}s")

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_path = os.path.join(model_save_dir, 'best.pkl')
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model (epoch {best_epoch}, AUC={best_auc:.4f}) to {save_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation on test set:")
    test_auc, test_prec = eval_model(model, test_data, qid2dg, gid2emb, emb_dim, top_k=args.top_k)
    print(f"  AUC:            {test_auc:.4f}")
    print(f"  Precision@{args.top_k}:  {test_prec:.4f}")
    print("=" * 60)

#!/usr/bin/env python3
"""
Generate Node2Vec embeddings for all graphs in AIDS dataset.

For each graph, runs Node2Vec to get node-level embeddings (512-dim),
then averages to get graph-level embedding.

Output:
  - emb/{dataset}512/g<id>.txt  (per-graph text files, LAN_ori format)
  - emb/{dataset}512_graph_emb.pkl  (pickle: dict of graph_id -> np.ndarray)

Usage:
  python generate_node2vec_emb.py --dim 512
  python generate_node2vec_emb.py --dim 512 --parallel 8
"""

import os
import sys
import pickle
import argparse
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Suppress gensim verbose logging
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('node2vec').setLevel(logging.WARNING)

import numpy as np
import networkx as nx
from node2vec import Node2Vec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_graph_map(dataset='AIDS'):
    index_file = os.path.join(SCRIPT_DIR, 'indexes', f'{dataset}_LAN_index.pkl')
    logger.info(f"Loading graphs from {index_file}...")
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    graph_map = index_data['graph_id_to_original']
    logger.info(f"  {len(graph_map)} graphs loaded")
    return graph_map


def _suppress_stderr():
    """Redirect OS-level stderr (fd 2) to devnull. Catches C extension output."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr_fd


def _restore_stderr(old_stderr_fd):
    """Restore OS-level stderr from saved fd."""
    os.dup2(old_stderr_fd, 2)
    os.close(old_stderr_fd)


def run_node2vec_on_graph(g, dimensions=512, walk_length=80, num_walks=10,
                          p=1.0, q=1.0, workers=1):
    """
    Run Node2Vec on a single graph and return node embeddings.

    Returns:
        dict: node_id -> numpy array of shape (dimensions,)
    """
    if g.number_of_nodes() == 0:
        return {}

    if g.number_of_nodes() == 1:
        node = list(g.nodes())[0]
        return {node: np.zeros(dimensions, dtype=np.float32)}

    if g.number_of_edges() == 0:
        return {n: np.zeros(dimensions, dtype=np.float32) for n in g.nodes()}

    node2vec = Node2Vec(
        g,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        quiet=True
    )

    # Suppress C-level "Exception ignored in" messages from gensim
    old_fd = _suppress_stderr()
    try:
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
    finally:
        _restore_stderr(old_fd)

    embeddings = {}
    for node in g.nodes():
        node_str = str(node)
        if node_str in model.wv:
            embeddings[node] = model.wv[node_str].astype(np.float32)
        else:
            embeddings[node] = np.zeros(dimensions, dtype=np.float32)

    return embeddings


def save_text_format(node_embeddings, graph_id, output_dir):
    """Save node embeddings in LAN_ori text format."""
    filepath = os.path.join(output_dir, f'g{graph_id}.txt')
    num_nodes = len(node_embeddings)
    dim = next(iter(node_embeddings.values())).shape[0] if num_nodes > 0 else 0

    with open(filepath, 'w') as f:
        f.write(f'{num_nodes} {dim}\n')
        for node_id, emb in sorted(node_embeddings.items(), key=lambda x: str(x[0])):
            emb_str = ' '.join(f'{v:.6f}' for v in emb)
            f.write(f'{node_id} {emb_str}\n')


def _worker_init():
    """Initializer for worker processes: suppress stderr globally."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    logging.getLogger('gensim').setLevel(logging.WARNING)
    logging.getLogger('node2vec').setLevel(logging.WARNING)


def _process_one_graph(args):
    """Worker function for parallel processing."""
    gid, g, dimensions, walk_length, num_walks, p, q, emb_dir = args
    try:
        node_embs = run_node2vec_on_graph(
            g, dimensions=dimensions,
            walk_length=walk_length, num_walks=num_walks,
            p=p, q=q, workers=1
        )
        save_text_format(node_embs, gid, emb_dir)

        if len(node_embs) > 0:
            all_embs = np.stack(list(node_embs.values()))
            graph_emb_np = np.mean(all_embs, axis=0)
        else:
            graph_emb_np = np.zeros(dimensions, dtype=np.float32)

        return gid, graph_emb_np, None
    except Exception as e:
        return gid, np.zeros(dimensions, dtype=np.float32), str(e)


def load_txt_embedding(filepath, dimensions):
    """Read a txt file and return graph-level embedding (mean of node embeddings)."""
    node_embs = []
    with open(filepath, 'r') as f:
        header = f.readline().strip().split()
        num_nodes = int(header[0])
        for _ in range(num_nodes):
            parts = f.readline().strip().split()
            emb = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            node_embs.append(emb)
    if node_embs:
        return np.mean(np.stack(node_embs), axis=0)
    return np.zeros(dimensions, dtype=np.float32)


def generate_embeddings(dataset='AIDS', dimensions=512, walk_length=80, num_walks=10,
                        p=1.0, q=1.0, workers=1, parallel=1, resume=False):
    graph_map = load_graph_map(dataset)

    # Output directories
    ds_lower = dataset.lower()
    emb_dir = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{dimensions}')
    os.makedirs(emb_dir, exist_ok=True)

    graph_emb_map = {}  # graph_id -> np.ndarray (graph-level embedding)
    failed = []

    logger.info(f"Generating Node2Vec embeddings (dim={dimensions})...")
    logger.info(f"  walk_length={walk_length}, num_walks={num_walks}, p={p}, q={q}")

    sorted_ids = sorted(graph_map.keys(), key=int)

    # Resume mode: skip graphs that already have txt files, load their embeddings
    if resume:
        # Try loading from pkl first (much faster than reading individual txt files)
        pkl_path_check = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{dimensions}_graph_emb.pkl')
        if os.path.exists(pkl_path_check):
            logger.info(f"  Loading existing embeddings from pkl: {pkl_path_check}")
            with open(pkl_path_check, 'rb') as f:
                graph_emb_map = pickle.load(f)
            logger.info(f"  Loaded {len(graph_emb_map)} embeddings from pkl")

        existing_ids = []
        missing_ids = []
        for gid in sorted_ids:
            if gid in graph_emb_map:
                existing_ids.append(gid)
            else:
                txt_path = os.path.join(emb_dir, f'g{gid}.txt')
                if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                    existing_ids.append(gid)
                else:
                    missing_ids.append(gid)
        # Load remaining txt files not covered by pkl (parallel)
        txt_to_load = [gid for gid in existing_ids if gid not in graph_emb_map]
        if txt_to_load:
            logger.info(f"  Loading {len(txt_to_load)} embeddings from txt files (parallel)...")
            def _load_one(gid):
                txt_path = os.path.join(emb_dir, f'g{gid}.txt')
                try:
                    return gid, load_txt_embedding(txt_path, dimensions), None
                except Exception as e:
                    return gid, None, str(e)
            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
                for gid, emb, err in tqdm(executor.map(_load_one, txt_to_load),
                                          total=len(txt_to_load), desc="Loading txt"):
                    if err:
                        logger.warning(f"  Failed to load txt for graph {gid}: {err}")
                        missing_ids.append(gid)
                    else:
                        graph_emb_map[gid] = emb

        logger.info(f"  Resume mode: {len(graph_emb_map)} existing, {len(missing_ids)} to generate")
        sorted_ids = sorted(missing_ids, key=int)

    if parallel > 1:
        logger.info(f"  Using {parallel} parallel processes")
        task_args = [
            (gid, graph_map[gid], dimensions, walk_length, num_walks, p, q, emb_dir)
            for gid in sorted_ids
        ]

        # Use Pool instead of ProcessPoolExecutor for better Ctrl+C handling
        pool = multiprocessing.Pool(
            processes=parallel,
            initializer=_worker_init
        )
        try:
            results = pool.imap_unordered(_process_one_graph, task_args)
            for gid, emb_np, err in tqdm(results, total=len(task_args), desc="Node2Vec"):
                graph_emb_map[gid] = emb_np
                if err:
                    logger.warning(f"  Failed for graph {gid}: {err}")
                    failed.append(gid)
        except KeyboardInterrupt:
            logger.info("\nInterrupted! Terminating workers...")
            pool.terminate()
            pool.join()
            logger.info("Workers terminated. Saving partial results...")
        else:
            pool.close()
            pool.join()
    else:
        # Sequential mode
        for gid in tqdm(sorted_ids, desc="Node2Vec"):
            g = graph_map[gid]
            try:
                node_embs = run_node2vec_on_graph(
                    g, dimensions=dimensions,
                    walk_length=walk_length, num_walks=num_walks,
                    p=p, q=q, workers=workers
                )
                save_text_format(node_embs, gid, emb_dir)

                if len(node_embs) > 0:
                    all_embs = np.stack(list(node_embs.values()))
                    graph_emb = np.mean(all_embs, axis=0).astype(np.float32)
                else:
                    graph_emb = np.zeros(dimensions, dtype=np.float32)
                graph_emb_map[gid] = graph_emb
            except Exception as e:
                logger.warning(f"  Failed for graph {gid} ({g.number_of_nodes()} nodes): {e}")
                graph_emb_map[gid] = np.zeros(dimensions, dtype=np.float32)
                failed.append(gid)

    # Save graph-level embeddings as pickle
    pkl_path = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{dimensions}_graph_emb.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(graph_emb_map, f)

    logger.info(f"\nDone! {len(graph_emb_map)} graphs processed, {len(failed)} failed")
    logger.info(f"Text files: {emb_dir}/")
    logger.info(f"Pickle: {pkl_path}")

    if failed:
        logger.info(f"Failed graph IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows
    parser = argparse.ArgumentParser(description='Generate Node2Vec embeddings')
    parser.add_argument('--dataset', type=str, default='AIDS',
                        choices=['AIDS', 'PubChem', 'SYN', 'Chemical1M'],
                        help='Dataset name (default: AIDS)')
    parser.add_argument('--dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=1, help='Node2Vec internal workers (per graph)')
    default_parallel = max(1, int(multiprocessing.cpu_count() * 0.7))
    parser.add_argument('--parallel', type=int, default=default_parallel,
                        help=f'Number of parallel processes (default: 70%% CPU = {default_parallel})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip existing txt files, only generate missing ones')
    args = parser.parse_args()

    generate_embeddings(
        dataset=args.dataset,
        dimensions=args.dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        workers=args.workers,
        parallel=args.parallel,
        resume=args.resume,
    )

"""
Simple Similarity Search using trained Graph Hashing model
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage to avoid GPU errors

import sys
import multiprocessing

# Parse custom flags before TensorFlow parses sys.argv
def _extract_arg(name, default=None, is_flag=False):
    """Extract and remove custom argument from sys.argv"""
    if is_flag:
        if name in sys.argv:
            sys.argv.remove(name)
            return True
        return False
    if name in sys.argv:
        idx = sys.argv.index(name)
        value = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        return value
    return default

_use_parallel_flag = _extract_arg('--use_parallel', is_flag=True)
_num_workers = _extract_arg('--num_workers')
if _num_workers is not None:
    _num_workers = int(_num_workers)
_tau = _extract_arg('--tau')
if _tau is not None:
    _tau = float(_tau)
_query_start = _extract_arg('--query_start')
if _query_start is not None:
    _query_start = int(_query_start)
_query_end = _extract_arg('--query_end')
if _query_end is not None:
    _query_end = int(_query_end)
_error_margin = _extract_arg('--error_margin')
if _error_margin is not None:
    _error_margin = float(_error_margin)
_beam_width = _extract_arg('--beam_width')
if _beam_width is not None:
    _beam_width = int(_beam_width)
_ged_method = _extract_arg('--ged_method')
if _ged_method is None:
    _ged_method = 'AStar-BMao'  # Default to AStar-BMao (Gisma AppForComputation), can be 'BSS-GED'
_no_save = _extract_arg('--no_save', is_flag=True)

import tensorflow as tf
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
from graphHashFunctions import GraphHash_Emb_Code
from utils import *
import pickle
import time
import subprocess
import os
import tempfile
import platform
import sys
# ProcessPoolExecutor was removed due to Windows TensorFlow issues
# Using subprocess.Popen directly for parallel BSS-GED

# Configuration
model_name = "0211_All_" + FLAGS.dataset
model_path = "SavedModel/" + model_name + ".ckpt"

# GED verification method configuration
GISMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Gisma'))
if platform.system() == 'Windows':
    GISMA_EXE = os.path.join(GISMA_DIR, 'build', 'Release', 'GismaProject.exe')
else:
    GISMA_EXE = os.path.join(GISMA_DIR, 'build', 'GismaProject')  # Linux

print("=" * 60)
print("Graph Hashing Similarity Search")
print("=" * 60)

# 1. Create data fetcher (skip MyGraph conversion for train graphs)
print("\n[1/5] Loading dataset...")
data_fetcher = DataFetcher(dataset=FLAGS.dataset, exact_ged=True, wrp_train_graph=False)
print(f"  - Test graphs: {data_fetcher.get_test_graphs_num()}")

# Use raw train graphs stored by DataFetcher (no re-reading needed)
from tqdm import tqdm
print("  - Building train graph index...")
train_graphs_raw = data_fetcher.train_graphs_raw
# Build gid to raw graph mapping
train_gid_to_raw = {}
for g in tqdm(train_graphs_raw, desc='Building index', unit='graph'):
    train_gid_to_raw[g['gid']] = g
print(f"  - Training graphs: {len(train_graphs_raw)}")

# 2. Define placeholders
print("\n[2/5] Setting up model placeholders...")
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.ecd_batchsize)),
    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}

# 3. Create model
print("\n[3/5] Creating model architecture...")
model = GraphHash_Emb_Code(placeholders,
                           input_dim=data_fetcher.get_node_feature_dim(),
                           next_ele=None,  # No data pipeline for inference
                           logging=False)

# 4. Restore model
print("\n[4/5] Loading trained model...")
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, model_path)
print(f"  - Model loaded: {model_path}")

# 5. Load inverted index
print("\n[5/5] Loading inverted index...")
with open('SavedModel/inverted_index_'+model_name+'.pkl', 'rb') as f:
    inverted_index = pickle.load(f)
print(f"  - Indexed graphs: {len(inverted_index)}")

print("\n" + "=" * 60)
print("Setup complete! Ready for queries.")
print("=" * 60)


def encode_graph(graph_idx, tvt='train'):
    """Encode a single graph to embedding and hash code"""
    idx_list = [graph_idx]
    while len(idx_list) < FLAGS.ecd_batchsize:
        idx_list.append(0)

    feed_dict = construct_feed_dict_for_encode(data_fetcher,
                                               placeholders,
                                               idx_list,
                                               tvt)
    thres = np.zeros(FLAGS.hash_code_len)
    feed_dict.update({placeholders['thres']: thres})

    code, emb = sess.run([model.codes, model.ecd_embeddings], feed_dict=feed_dict)
    return code[0], emb[0]


def search_similar(query_idx, top_k=10, use_train=False):
    """
    Search for top-k most similar graphs

    Args:
        query_idx: Index of query graph
        top_k: Number of results to return
        use_train: If True, query from training set; else from test set

    Returns:
        List of (graph_idx, distance) tuples
    """
    print(f"\n{'='*60}")
    print(f"Query: {'Train' if use_train else 'Test'} Graph #{query_idx}")
    print(f"{'='*60}")

    # Encode query graph
    t0 = time.time()
    tvt = 'train' if use_train else 'test'
    code_q, emb_q = encode_graph(query_idx, tvt)
    encode_time = time.time() - t0

    print(f"\n[Encoding] Time: {encode_time:.4f}s")
    print(f"  - Embedding: {emb_q.shape}")
    print(f"  - Hash code: {code_q.shape}")

    # Search through training graphs
    print(f"\n[Searching] Computing distances to {len(train_graphs_raw)} training graphs...")
    t0 = time.time()

    distances = []
    train_num = len(train_graphs_raw)

    # Batch encoding for efficiency
    batch_size = 100
    for i in range(0, train_num, batch_size):
        batch_end = min(i + batch_size, train_num)
        batch_dists = []

        for j in range(i, batch_end):
            code_t, emb_t = encode_graph(j, 'train')
            # Use squared Euclidean distance in embedding space
            # This approximates GED directly (no transformation needed)
            dist = np.sum((emb_q - emb_t)**2)
            batch_dists.append((j, dist))

        distances.extend(batch_dists)

        if (i + batch_size) % 500 == 0:
            print(f"  Progress: {i + batch_size}/{train_num}")

    search_time = time.time() - t0
    print(f"\n[Results] Search time: {search_time:.4f}s")

    # Sort and return top-k
    distances.sort(key=lambda x: x[1])
    results = distances[:top_k]

    print(f"\nTop-{top_k} most similar training graphs:")
    print("-" * 60)
    for rank, (train_idx, dist) in enumerate(results, 1):
        train_graph = train_graphs_raw[train_idx]
        print(f"  Rank {rank:2d}: Train #{train_idx:4d}  Distance: {dist:.4f}  "
              f"Nodes: {len(train_graph['nodes']):3d}  "
              f"Edges: {len(train_graph['edges']):3d}")

    return results


def range_query_ml_only(query_idx, tau, error_margin=0.5, use_train=False):
    """
    ML-only range query: Find candidates using ML model, no BSS-GED verification.

    Args:
        query_idx: Index of query graph
        tau: GED threshold
        error_margin: Error tolerance (default 0.5 to handle ML prediction errors)
        use_train: If True, query is from training set, else from test set

    Returns:
        query_data: dict with 'gid', 'nodes', 'edges' for query graph
        candidates_data: list of dicts with 'train_idx', 'pred_ged', 'gid', 'nodes', 'edges'
        ml_time: time for ML filtering
    """
    # Encode query graph
    tvt = 'train' if use_train else 'test'

    t0 = time.time()
    code_q, emb_q = encode_graph(query_idx, tvt)

    # Get query graph data (test graphs have MyGraph, use ori_graph)
    query_graph = data_fetcher.test_graphs[query_idx]
    query_gid = query_graph.ori_graph.get('gid', query_idx)
    query_data = {
        'gid': query_gid,
        'nodes': query_graph.ori_graph['nodes'],
        'edges': query_graph.ori_graph['edges']
    }

    # Phase 1: ML filtering using precomputed embeddings
    threshold = tau + error_margin
    candidates_data = []

    for hash_code, entries in inverted_index.items():
        for gid, emb_t in entries:
            # Skip the query graph itself (avoid self-match)
            if gid == query_gid:
                continue

            # Use squared Euclidean distance in embedding space
            predicted_ged = np.sum((emb_q - emb_t)**2)

            # Only keep if predicted_ged <= tau + error_margin
            if predicted_ged <= threshold:
                # Get raw graph data from lightweight train graphs
                if gid not in train_gid_to_raw:
                    continue
                raw_graph = train_gid_to_raw[gid]
                candidates_data.append({
                    'train_idx': gid,  # Use gid as train_idx
                    'pred_ged': predicted_ged,
                    'gid': gid,
                    'nodes': raw_graph['nodes'],
                    'edges': raw_graph['edges']
                })

    ml_time = time.time() - t0
    return query_data, candidates_data, ml_time


def range_query(query_idx, tau, error_margin=0.5, use_train=False, beam_width=15):
    """
    Range query: Find all graphs with predicted GED <= tau + error_margin

    Args:
        query_idx: Index of query graph
        tau: GED threshold
        error_margin: Error tolerance (default 0.5 to handle ML prediction errors)
        use_train: If True, query is from training set, else from test set
        beam_width: Beam width for BSS-GED (default 15)

    Returns:
        List of (graph_idx, predicted_ged, true_ged) where predicted_ged <= tau + error_margin
    """
    # Encode query graph
    tvt = 'train' if use_train else 'test'
    print(f"\n[Query] Graph #{query_idx} from {'training' if use_train else 'test'} set")
    print(f"[Query] Tau = {tau}, Error margin = {error_margin}")
    print(f"[Query] Accepting candidates with predicted_ged <= {tau + error_margin}")

    t0 = time.time()
    code_q, emb_q = encode_graph(query_idx, tvt)
    encode_time = time.time() - t0

    print(f"\n[Encoding] Time: {encode_time:.4f}s")

    # Get query graph for GED computation
    if tvt == 'train':
        query_graph = data_fetcher.train_graphs[query_idx]
    else:
        query_graph = data_fetcher.test_graphs[query_idx]

    # Search through training graphs using precomputed embeddings from inverted_index
    print(f"\n[Searching] Filtering training graphs using inverted index...")
    t0 = time.time()

    candidates = []
    threshold = tau + error_margin
    bssged_time = 0.0

    # Use BSS-GED executable (platform-specific path)
    if platform.system() == 'Windows':
        ged_exe = os.path.join(os.path.dirname(__file__), 'BSS-GED', 'ged.exe')
    else:
        ged_exe = os.path.join(os.path.dirname(__file__), 'BSS-GED', 'ged')

    # Build gid to train_idx mapping
    gid_to_idx = {}
    for idx in range(data_fetcher.get_train_graphs_num()):
        gid = data_fetcher.get_train_graph_gid(idx)
        gid_to_idx[gid] = idx

    # Iterate through all entries in inverted_index
    total_graphs = sum(len(entries) for entries in inverted_index.values())
    print(f"  - Total indexed graphs: {total_graphs}")

    # Phase 1: ML filtering using precomputed embeddings
    ml_candidates = []  # (train_idx, predicted_ged)
    for hash_code, entries in inverted_index.items():
        for gid, emb_t in entries:
            # Use squared Euclidean distance in embedding space
            predicted_ged = np.sum((emb_q - emb_t)**2)

            # Only keep if predicted_ged <= tau + error_margin
            if predicted_ged <= threshold:
                # Get train index from gid
                if gid not in gid_to_idx:
                    continue
                train_idx = gid_to_idx[gid]
                ml_candidates.append((train_idx, predicted_ged))

    ml_time = time.time() - t0
    print(f"\n[Results] ML filtering time: {ml_time:.4f}s")
    print(f"[Results] ML candidates: {len(ml_candidates)}")

    # Phase 2: Batch BSS-GED verification
    candidates = []
    bssged_time = 0.0

    if len(ml_candidates) > 0:
        t_ged = time.time()
        try:
            # Write query graph to temp file
            query_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bss')
            query_file.close()
            write_graph_to_bss(query_graph, query_file.name)

            # Write all candidates to one temp file
            candidates_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bss')
            candidates_file.close()
            with open(candidates_file.name, 'w') as f:
                for train_idx, pred_ged in ml_candidates:
                    candidate_graph = data_fetcher.train_graphs[train_idx]
                    ori_graph = candidate_graph.ori_graph
                    gid = ori_graph.get('gid', train_idx)
                    f.write(f"{gid}\n")
                    f.write(f"{len(ori_graph['nodes'])} {len(ori_graph['edges'])}\n")
                    for node_label in ori_graph['nodes']:
                        f.write(f"{node_label}\n")
                    for edge in ori_graph['edges']:
                        f.write(f"{edge[0]} {edge[1]} 0\n")

            # Call BSS-GED once for all candidates
            ret = subprocess.check_output(
                [ged_exe, query_file.name, '1', candidates_file.name, str(len(ml_candidates)),
                 str(int(tau)), str(beam_width)]
            )

            ged_values = [float(x) for x in ret.split()][:-1]

            # Match GED values with candidates
            for i, (train_idx, pred_ged) in enumerate(ml_candidates):
                true_ged = ged_values[i] if i < len(ged_values) else None
                candidates.append((train_idx, pred_ged, true_ged))

            os.remove(query_file.name)
            os.remove(candidates_file.name)
        except Exception as e:
            print(f"Error in batch GED computation: {e}")
            # Fallback: add candidates without true_ged
            for train_idx, pred_ged in ml_candidates:
                candidates.append((train_idx, pred_ged, None))

        bssged_time = time.time() - t_ged

    total_time = time.time() - t0
    print(f"[Results] BSS-GED time: {bssged_time:.4f}s")
    print(f"[Results] Total time: {total_time:.4f}s")
    print(f"[Results] Found {len(candidates)} candidates (out of {total_graphs} total)")

    # Sort by predicted GED
    candidates.sort(key=lambda x: x[1])

    method = f"BSS-GED (beam_width={beam_width})"
    print(f"\nCandidate graphs (predicted GED <= {threshold:.1f}): {len(candidates)} total")
    print(f"Method: {method}")
    print("-" * 90)
    for rank, (train_idx, pred_ged, true_ged) in enumerate(candidates[:20], 1):  # Show top 20
        train_graph = data_fetcher.train_graphs[train_idx]
        if true_ged is not None:
            error = abs(true_ged - pred_ged)
            status = "OK" if true_ged <= tau else "X"
            ged_str = f">{tau:.0f}" if true_ged > tau else f"{true_ged:6.2f}"
            print(f"  #{rank:3d}: Train #{train_idx:4d}  Pred: {pred_ged:6.2f} -> True: {ged_str:>6s} (Err={error:5.2f}) {status}  "
                  f"Nodes: {len(train_graph.ori_graph['nodes']):3d}  Edges: {len(train_graph.ori_graph['edges']):3d}")
        else:
            print(f"  #{rank:3d}: Train #{train_idx:4d}  Pred_GED: {pred_ged:.4f}  "
                  f"Nodes: {len(train_graph.ori_graph['nodes']):3d}  "
                  f"Edges: {len(train_graph.ori_graph['edges']):3d}")

    if len(candidates) > 20:
        print(f"  ... and {len(candidates) - 20} more candidates")

    return candidates, ml_time, bssged_time


def write_graph_to_bss(graph, filename):
    """
    Write a MyGraph object to BSS-GED format file
    Format:
        graph_id
        num_nodes num_edges
        node_label_1
        node_label_2
        ...
        edge_from edge_to
        edge_from edge_to
        ...
    """
    with open(filename, 'w') as f:
        # Write graph ID
        gid = graph.ori_graph.get('gid', 0)
        f.write(f"{gid}\n")

        # Write node and edge counts
        num_nodes = len(graph.ori_graph['nodes'])
        num_edges = len(graph.ori_graph['edges'])
        f.write(f"{num_nodes} {num_edges}\n")

        # Write node labels
        for node_label in graph.ori_graph['nodes']:
            f.write(f"{node_label}\n")

        # Write edges (src dst edge_label)
        for edge in graph.ori_graph['edges']:
            f.write(f"{edge[0]} {edge[1]} 0\n")


def prepare_bssged_files(query_idx, query_data, candidates_data):
    """
    Prepare temp files for BSS-GED execution.

    Returns:
        query_file_path, candidates_file_path
    """
    if len(candidates_data) == 0:
        return None, None

    # Write query graph to temp file
    query_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_q{query_idx}.bss')
    query_file.write(f"{query_data['gid']}\n")
    query_file.write(f"{len(query_data['nodes'])} {len(query_data['edges'])}\n")
    for node_label in query_data['nodes']:
        query_file.write(f"{node_label}\n")
    for edge in query_data['edges']:
        query_file.write(f"{edge[0]} {edge[1]} 0\n")
    query_file.close()

    # Write all candidates to one temp file
    candidates_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_c{query_idx}.bss')
    for cand in candidates_data:
        candidates_file.write(f"{cand['gid']}\n")
        candidates_file.write(f"{len(cand['nodes'])} {len(cand['edges'])}\n")
        for node_label in cand['nodes']:
            candidates_file.write(f"{node_label}\n")
        for edge in cand['edges']:
            candidates_file.write(f"{edge[0]} {edge[1]} 0\n")
    candidates_file.close()

    return query_file.name, candidates_file.name


def parse_bssged_output(output, candidates_data):
    """
    Parse BSS-GED output and match with candidates.

    Returns:
        tuple of (results, cpu_time) where:
        - results: list of (train_idx, pred_ged, true_ged)
        - cpu_time: CPU time reported by BSS-GED process (float, in seconds)
    """
    results = []
    cpu_time = 0.0
    try:
        all_values = [float(x) for x in output.split()]
        # Last value is the CPU time reported by BSS-GED
        if len(all_values) > 0:
            cpu_time = all_values[-1]
            ged_values = all_values[:-1]
        else:
            ged_values = []
        for i, cand in enumerate(candidates_data):
            true_ged = ged_values[i] if i < len(ged_values) else None
            results.append((cand['train_idx'], cand['pred_ged'], true_ged))
    except Exception:
        for cand in candidates_data:
            results.append((cand['train_idx'], cand['pred_ged'], None))
    return results, cpu_time


def prepare_gisma_files(query_idx, candidates_data):
    """
    Prepare temp file containing candidate IDs for Gisma batch_ged mode.

    Returns:
        candidate_ids_file_path (str) or None if no candidates
    """
    if len(candidates_data) == 0:
        return None

    # Write candidate IDs (gid) to temp file, one per line
    candidates_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_gisma_c{query_idx}.txt')
    for cand in candidates_data:
        # Use gid (original graph ID) for Gisma
        gid = cand.get('gid', cand['train_idx'])
        candidates_file.write(f"{gid}\n")
    candidates_file.close()

    return candidates_file.name


def parse_gisma_output(output, candidates_data):
    """
    Parse Gisma batch_ged output and match with candidates.
    Output format: one GED value per line, last line is CPU time.

    Returns:
        tuple of (results, cpu_time) where:
        - results: list of (train_idx, pred_ged, true_ged)
        - cpu_time: CPU time reported by Gisma (float, in seconds)
    """
    results = []
    cpu_time = 0.0
    try:
        all_values = [float(x) for x in output.split()]
        # Last value is the CPU time
        if len(all_values) > 0:
            cpu_time = all_values[-1]
            ged_values = all_values[:-1]
        else:
            ged_values = []
        for i, cand in enumerate(candidates_data):
            true_ged = ged_values[i] if i < len(ged_values) else None
            results.append((cand['train_idx'], cand['pred_ged'], true_ged))
    except Exception:
        for cand in candidates_data:
            results.append((cand['train_idx'], cand['pred_ged'], None))
    return results, cpu_time


def simple_ged_estimate(g1, g2):
    """
    Simple Python-based GED estimation (for Windows compatibility)
    Uses graph structure differences as approximation
    """
    # Node and edge counts from ori_graph dictionary
    n1, n2 = len(g1.ori_graph['nodes']), len(g2.ori_graph['nodes'])
    e1, e2 = len(g1.ori_graph['edges']), len(g2.ori_graph['edges'])

    # Basic structural distance
    node_diff = abs(n1 - n2)
    edge_diff = abs(e1 - e2)

    # Simple estimate: at least need to insert/delete different nodes and edges
    # This is a lower bound, real GED is usually higher
    base_cost = node_diff + edge_diff

    # Add cost for node/edge label differences (rough estimate)
    # Assume some nodes/edges need relabeling
    common_nodes = min(n1, n2)
    common_edges = min(e1, e2)

    # Rough heuristic: ~30% of common elements might need relabeling
    estimated_relabel = int(0.3 * (common_nodes + common_edges))

    return base_cost + estimated_relabel


def verify_candidates(query_idx, candidates, tau, use_train=False, beam_width=15, use_python_ged=None):
    """
    Verification phase: Filter candidates based on true GED

    Args:
        query_idx: Index of query graph
        candidates: List of (graph_idx, predicted_ged, true_ged) from candidate generation phase
        tau: GED threshold
        use_train: If True, query is from training set
        beam_width: Beam width for BSS-GED (default 15, unused if GED already computed)
        use_python_ged: Unused, kept for compatibility

    Returns:
        verified_results: List of (graph_idx, predicted_ged, true_ged) where true_ged <= tau
    """
    if len(candidates) == 0:
        print("\n[Verification] No candidates to verify")
        return []

    print(f"\n[Verification] Filtering {len(candidates)} candidates based on true GED...")
    t0 = time.time()

    verified_results = []

    # Filter candidates based on true GED
    for candidate_idx, pred_ged, true_ged in candidates:
        if true_ged is not None and true_ged != -1 and true_ged <= tau:
            verified_results.append((candidate_idx, pred_ged, true_ged))

    verification_time = time.time() - t0

    print(f"\n[Verification] Time: {verification_time:.4f}s")
    print(f"[Verification] Verified results: {len(verified_results)}/{len(candidates)} candidates")

    # Sort by true GED
    verified_results.sort(key=lambda x: x[2])

    print(f"\nVerified results (true GED <= {tau}): {len(verified_results)} total")
    print("-" * 80)
    for rank, (train_idx, pred_ged, true_ged) in enumerate(verified_results[:20], 1):
        train_graph = data_fetcher.train_graphs[train_idx]
        error = abs(true_ged - pred_ged)
        print(f"  #{rank:3d}: Train #{train_idx:4d}  Pred: {pred_ged:6.2f} -> True: {true_ged:6.2f} (Err={error:5.2f})  "
              f"Nodes: {len(train_graph.ori_graph['nodes']):3d}  Edges: {len(train_graph.ori_graph['edges']):3d}")

    if len(verified_results) > 20:
        print(f"  ... and {len(verified_results) - 20} more results")

    return verified_results


def load_ground_truth(ground_truth_path, tau):
    """
    Load ground truth and return dict: {query_id: set of train_graph_ids with GED <= tau}
    """
    ground_truth = {}
    with open(ground_truth_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: query_id ged_value [list_of_train_ids]
            parts = line.split(' ', 2)
            query_id = int(parts[0])
            ged_value = float(parts[1])
            # Parse list of train ids
            ids_str = parts[2].strip('[]')
            if ids_str:
                train_ids = [int(x.strip()) for x in ids_str.split(',')]
            else:
                train_ids = []

            if ged_value <= tau:
                if query_id not in ground_truth:
                    ground_truth[query_id] = set()
                ground_truth[query_id].update(train_ids)

    return ground_truth


# Example usage
if __name__ == "__main__":
    # Use named arguments (parsed before TensorFlow import)
    tau = _tau if _tau is not None else 20.0
    query_start = _query_start if _query_start is not None else 0
    query_end = _query_end if _query_end is not None else 99
    error_margin = _error_margin if _error_margin is not None else 0.5
    beam_width = _beam_width if _beam_width is not None else 15
    use_parallel = _use_parallel_flag
    num_workers = _num_workers if _num_workers is not None else max(1, int(multiprocessing.cpu_count() * 0.8))

    print("\n\n" + "="*60)
    print("Batch Two-Phase Range Query")
    print("="*60)

    # Show command line with parameter names
    import sys
    cmd = f"python simple_search.py --tau {tau} --query_start {query_start} --query_end {query_end} --error_margin {error_margin}"
    if use_parallel:
        cmd += " --use_parallel"
    print(f"Command: {cmd}")

    print("\nParameters:")
    print(f"  - tau (GED threshold): {tau}")
    print(f"  - query_start: {query_start}")
    print(f"  - query_end: {query_end}")
    print(f"  - error_margin: {error_margin}")
    print(f"  - beam_width: {beam_width}")
    print(f"  - use_parallel: {use_parallel}")
    if use_parallel:
        print(f"  - num_workers: {num_workers}")

    # Validate query range
    test_num = data_fetcher.get_test_graphs_num()
    query_end = min(query_end, test_num - 1)

    print(f"\nProcessing queries {query_start} to {query_end} (total: {query_end - query_start + 1})")

    # Load ground truth
    ground_truth_path = f"../../Gisma/datasets/{FLAGS.dataset}/ground_truth.txt"
    if os.path.exists(ground_truth_path):
        print(f"\nLoading ground truth from {ground_truth_path}...")
        ground_truth = load_ground_truth(ground_truth_path, tau)
        print(f"  - Loaded ground truth for {len(ground_truth)} queries")
    else:
        print(f"\nWarning: Ground truth file not found: {ground_truth_path}")
        ground_truth = {}

    # Statistics
    total_candidates = 0
    total_verified = 0
    total_ml_time = 0.0
    total_bssged_time = 0.0
    query_results = []  # (query_idx, recall, precision, f1, gt_count, found_count, correct_count)

    import time as time_module
    total_start_time = time_module.time()

    # =====================================================
    # Phase 1: Serial ML filtering for all queries
    # =====================================================
    print("\n" + "="*60)
    print("Phase 1: ML Filtering (Serial)")
    print("="*60)

    ml_results = {}  # {query_idx: (query_data, candidates_data, ml_time)}
    ml_pbar = tqdm(range(query_start, query_end + 1), desc='ML Filtering', unit='query')
    for query_idx in ml_pbar:
        try:
            query_data, candidates_data, ml_time = range_query_ml_only(
                query_idx=query_idx, tau=tau, error_margin=error_margin, use_train=False
            )
            ml_results[query_idx] = (query_data, candidates_data, ml_time)
            total_ml_time += ml_time
            total_candidates += len(candidates_data)
            ml_pbar.set_postfix({'q': query_idx, 'cands': len(candidates_data), 'time': f'{ml_time:.2f}s'})
        except Exception as e:
            ml_pbar.set_postfix({'q': query_idx, 'error': str(e)[:20]})
            ml_results[query_idx] = (None, [], 0.0)

    print(f"ML Filtering complete: {total_candidates} total candidates")

    # =====================================================
    # Phase 2: GED verification (AStar-BMao or BSS-GED)
    # =====================================================
    ged_method = _ged_method  # 'AStar-BMao' or 'BSS-GED'
    print("\n" + "="*60)
    print(f"Phase 2: {ged_method} Verification ({'Parallel' if use_parallel else 'Serial'})")
    print("="*60)

    # Get GED executable path
    if ged_method == 'AStar-BMao':
        ged_exe = GISMA_EXE
        print(f"Using AStar-BMao (Gisma AppForComputation): {ged_exe}")
    else:
        if platform.system() == 'Windows':
            ged_exe = os.path.join(os.path.dirname(__file__), 'BSS-GED', 'ged.exe')
        else:
            ged_exe = os.path.join(os.path.dirname(__file__), 'BSS-GED', 'ged')

    num_queries_with_candidates = sum(1 for q in range(query_start, query_end + 1)
                                      if ml_results[q][1])
    print(f"Processing {num_queries_with_candidates} queries with candidates")

    bssged_results = {}  # {query_idx: [(train_idx, pred_ged, true_ged), ...]}
    bssged_times = {}  # {query_idx: time_in_seconds}
    bssged_start = time_module.time()

    if use_parallel:
        # Parallel mode: process in batches of num_workers
        # Collect queries with candidates
        queries_to_process = []
        for query_idx in range(query_start, query_end + 1):
            query_data, candidates_data, _ = ml_results[query_idx]
            if query_data is not None and len(candidates_data) > 0:
                queries_to_process.append(query_idx)
            else:
                bssged_results[query_idx] = []
                bssged_times[query_idx] = 0.0

        # Process in batches
        for batch_start in range(0, len(queries_to_process), num_workers):
            batch = queries_to_process[batch_start:batch_start + num_workers]
            processes = {}  # {query_idx: (process, query_file, candidates_file, candidates_data)}
            batch_start_time = time_module.time()

            # Launch batch of processes
            for query_idx in batch:
                query_data, candidates_data, _ = ml_results[query_idx]

                if ged_method == 'AStar-BMao':
                    # AStar-BMao mode: only need candidate IDs file
                    candidates_file = prepare_gisma_files(query_idx, candidates_data)
                    if candidates_file:
                        proc = subprocess.Popen(
                            [ged_exe, '--mode', 'batch_ged',
                             '--dataset', FLAGS.dataset,
                             '--query_id', str(query_data['gid']),
                             '--candidate_ids_file', candidates_file,
                             '--tau_search', str(int(tau))],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=GISMA_DIR
                        )
                        processes[query_idx] = (proc, None, candidates_file, candidates_data)
                    else:
                        bssged_results[query_idx] = []
                        bssged_times[query_idx] = 0.0
                else:
                    # BSS-GED mode
                    query_file, candidates_file = prepare_bssged_files(query_idx, query_data, candidates_data)
                    if query_file and candidates_file:
                        proc = subprocess.Popen(
                            [ged_exe, query_file, '1', candidates_file, str(len(candidates_data)),
                             str(int(tau)), str(beam_width)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        processes[query_idx] = (proc, query_file, candidates_file, candidates_data)
                    else:
                        bssged_results[query_idx] = []
                        bssged_times[query_idx] = 0.0

            # Wait for batch to complete and collect results
            batch_results = {}
            batch_cpu_times = {}
            for query_idx, (proc, query_file, candidates_file, candidates_data) in processes.items():
                try:
                    stdout, stderr = proc.communicate()
                    if ged_method == 'AStar-BMao':
                        results, cpu_time = parse_gisma_output(stdout, candidates_data)
                    else:
                        results, cpu_time = parse_bssged_output(stdout, candidates_data)
                    batch_results[query_idx] = results
                    batch_cpu_times[query_idx] = cpu_time
                except Exception as e:
                    print(f"  Query {query_idx}: {ged_method} error - {e}")
                    batch_results[query_idx] = []
                    batch_cpu_times[query_idx] = 0.0
                finally:
                    try:
                        if query_file:
                            os.remove(query_file)
                        if candidates_file:
                            os.remove(candidates_file)
                    except:
                        pass

            batch_end_time = time_module.time()
            batch_wall_time = batch_end_time - batch_start_time

            # Use CPU time reported by each GED process
            for query_idx in batch:
                if query_idx in batch_results:
                    bssged_results[query_idx] = batch_results[query_idx]
                    bssged_times[query_idx] = batch_cpu_times.get(query_idx, 0.0)
                    print(f"  Query {query_idx}: {ged_method} complete, {len(batch_results[query_idx])} results, {bssged_times[query_idx]:.4f}s")
                else:
                    bssged_results[query_idx] = []
                    bssged_times[query_idx] = 0.0
    else:
        # Serial mode: process one query at a time
        ged_pbar = tqdm(range(query_start, query_end + 1), desc=ged_method, unit='query')
        for query_idx in ged_pbar:
            query_data, candidates_data, _ = ml_results[query_idx]
            if query_data is not None and len(candidates_data) > 0:
                if ged_method == 'AStar-BMao':
                    # AStar-BMao mode
                    candidates_file = prepare_gisma_files(query_idx, candidates_data)
                    if candidates_file:
                        try:
                            ret = subprocess.check_output(
                                [ged_exe, '--mode', 'batch_ged',
                                 '--dataset', FLAGS.dataset,
                                 '--query_id', str(query_data['gid']),
                                 '--candidate_ids_file', candidates_file,
                                 '--tau_search', str(int(tau))],
                                cwd=GISMA_DIR
                            )
                            results, cpu_time = parse_gisma_output(ret, candidates_data)
                            bssged_times[query_idx] = cpu_time
                            bssged_results[query_idx] = results
                            ged_pbar.set_postfix({'q': query_idx, 'results': len(results), 'time': f'{cpu_time:.2f}s'})
                        except Exception as e:
                            ged_pbar.set_postfix({'q': query_idx, 'error': str(e)[:20]})
                            bssged_results[query_idx] = []
                            bssged_times[query_idx] = 0.0
                        finally:
                            try:
                                os.remove(candidates_file)
                            except:
                                pass
                    else:
                        bssged_results[query_idx] = []
                        bssged_times[query_idx] = 0.0
                else:
                    # BSS-GED mode
                    query_file, candidates_file = prepare_bssged_files(query_idx, query_data, candidates_data)
                    if query_file and candidates_file:
                        try:
                            ret = subprocess.check_output(
                                [ged_exe, query_file, '1', candidates_file, str(len(candidates_data)),
                                 str(int(tau)), str(beam_width)]
                            )
                            results, cpu_time = parse_bssged_output(ret, candidates_data)
                            bssged_times[query_idx] = cpu_time
                            bssged_results[query_idx] = results
                            ged_pbar.set_postfix({'q': query_idx, 'results': len(results), 'time': f'{cpu_time:.2f}s'})
                        except Exception as e:
                            ged_pbar.set_postfix({'q': query_idx, 'error': str(e)[:20]})
                            bssged_results[query_idx] = []
                            bssged_times[query_idx] = 0.0
                        finally:
                            try:
                                os.remove(query_file)
                                os.remove(candidates_file)
                            except:
                                pass
                    else:
                        bssged_results[query_idx] = []
                        bssged_times[query_idx] = 0.0
            else:
                bssged_results[query_idx] = []
                bssged_times[query_idx] = 0.0

    total_bssged_wall_time = time_module.time() - bssged_start
    total_bssged_cpu_time = sum(bssged_times.values())
    print(f"\n{ged_method} complete: {total_bssged_wall_time:.4f}s wall time, {total_bssged_cpu_time:.4f}s total CPU time")

    # =====================================================
    # Phase 3: Evaluation
    # =====================================================
    print("\n" + "="*60)
    print("Phase 3: Evaluation")
    print("="*60)

    for query_idx in range(query_start, query_end + 1):
        candidates = bssged_results.get(query_idx, [])

        # Filter verified results (true_ged <= tau)
        verified_results = [(train_idx, pred_ged, true_ged)
                           for train_idx, pred_ged, true_ged in candidates
                           if true_ged is not None and true_ged != -1 and true_ged <= tau]
        total_verified += len(verified_results)

        # Calculate recall and precision
        gt_set = ground_truth.get(query_idx, set())
        # train_idx is already gid in our lightweight approach
        found_set = set(r[0] for r in verified_results)

        correct = len(gt_set & found_set)
        recall = correct / len(gt_set) if len(gt_set) > 0 else None
        precision = correct / len(found_set) if len(found_set) > 0 else None
        if recall is not None and precision is not None and (recall + precision) > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = None

        # Get time info for this query
        q_ml_time = ml_results[query_idx][2]
        q_bssged_time = bssged_times.get(query_idx, 0.0)
        q_total_time = q_ml_time + q_bssged_time

        query_results.append((query_idx, recall, precision, f1, len(gt_set), len(found_set), correct,
                              q_ml_time, q_bssged_time, q_total_time))

        recall_str = f"{recall:.4f}" if recall is not None else "N/A"
        precision_str = f"{precision:.4f}" if precision is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        print(f"  Query {query_idx}: GT={len(gt_set)}, Found={len(found_set)}, Correct={correct}, "
              f"Recall={recall_str}, Precision={precision_str}, F1={f1_str}")

    # Summary
    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    print(f"{'Query':<8} {'GT':<8} {'Found':<8} {'Correct':<8} {'Recall':<10} {'Precision':<10} {'F1':<10} {'ML(s)':<10} {'GED(s)':<10} {'Total(s)':<10}")
    print("-" * 120)
    for query_idx, recall, precision, f1, gt_count, found_count, correct, q_ml, q_ged, q_total in query_results:
        recall_str = f"{recall:.4f}" if recall is not None else "N/A"
        precision_str = f"{precision:.4f}" if precision is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        print(f"{query_idx:<8} {gt_count:<8} {found_count:<8} {correct:<8} {recall_str:<10} {precision_str:<10} {f1_str:<10} {q_ml:<10.4f} {q_ged:<10.4f} {q_total:<10.4f}")

    # Average metrics (only count non-None values)
    recalls = [r[1] for r in query_results if r[1] is not None]
    precisions = [r[2] for r in query_results if r[2] is not None]
    f1s = [r[3] for r in query_results if r[3] is not None]

    avg_recall = sum(recalls) / len(recalls) if recalls else None
    avg_precision = sum(precisions) / len(precisions) if precisions else None
    avg_f1 = sum(f1s) / len(f1s) if f1s else None

    # Average for GT, Found, Correct
    avg_gt = sum(r[4] for r in query_results) / len(query_results) if query_results else 0
    avg_found = sum(r[5] for r in query_results) / len(query_results) if query_results else 0
    avg_correct = sum(r[6] for r in query_results) / len(query_results) if query_results else 0

    # Average times
    avg_ml = sum(r[7] for r in query_results) / len(query_results) if query_results else 0
    avg_ged = sum(r[8] for r in query_results) / len(query_results) if query_results else 0
    avg_total = sum(r[9] for r in query_results) / len(query_results) if query_results else 0

    avg_recall_str = f"{avg_recall:.4f}" if avg_recall is not None else "N/A"
    avg_precision_str = f"{avg_precision:.4f}" if avg_precision is not None else "N/A"
    avg_f1_str = f"{avg_f1:.4f}" if avg_f1 is not None else "N/A"

    print("-" * 120)
    print(f"{'AVG':<8} {avg_gt:<8.2f} {avg_found:<8.2f} {avg_correct:<8.2f} {avg_recall_str:<10} {avg_precision_str:<10} {avg_f1_str:<10} {avg_ml:<10.4f} {avg_ged:<10.4f} {avg_total:<10.4f}")
    print(f"(Recall/Precision/F1 based on {len(recalls)} queries with GT>0, {len(precisions)} with Found>0, {len(f1s)} with both)")

    # Time statistics
    total_time = time_module.time() - total_start_time
    num_queries = query_end - query_start + 1

    # Calculate average times
    ml_times = [ml_results[q][2] for q in range(query_start, query_end + 1)]
    avg_ml_time = sum(ml_times) / len(ml_times) if ml_times else 0.0

    bssged_time_list = [bssged_times.get(q, 0.0) for q in range(query_start, query_end + 1)]
    avg_bssged_time = sum(bssged_time_list) / len(bssged_time_list) if bssged_time_list else 0.0

    print("\n" + "="*80)
    print("TIME STATISTICS")
    print("="*80)
    print(f"ML Filtering (serial):")
    print(f"  Total:   {total_ml_time:.4f}s")
    print(f"  Average: {avg_ml_time:.4f}s per query")
    print(f"{ged_method} Verification ({'parallel' if use_parallel else 'serial'}):")
    print(f"  Wall time:     {total_bssged_wall_time:.4f}s")
    print(f"  Total CPU:     {total_bssged_cpu_time:.4f}s")
    print(f"  Avg per query: {avg_bssged_time:.4f}s")
    print(f"Overall:")
    print(f"  Total time:    {total_time:.4f}s")
    print(f"  Avg per query: {(avg_ml_time + avg_bssged_time):.4f}s (ML + {ged_method})")

    # =====================================================
    # Save results to files (skip if --no_save)
    # =====================================================
    if not _no_save:
        import json
        from datetime import datetime

        # Create results directory structure (under GHash_final)
        # Include error_margin in directory name to avoid overwriting results with different margins
        results_base = f"../results/similarity_search/{FLAGS.dataset}/tau{tau}_margin{error_margin}"
        per_query_dir = os.path.join(results_base, "per_query", "GHash")
        os.makedirs(per_query_dir, exist_ok=True)

        # Save per-query results as JSON
        for query_idx, recall, precision, f1, gt_count, found_count, correct, q_ml, q_ged, q_total in query_results:
            query_result = {
                "metadata": {
                    "dataset": FLAGS.dataset,
                    "model": "GHash",
                    "tau_threshold": tau,
                    "error_margin": error_margin,
                    "use_parallel": use_parallel,
                    "query_id": query_idx
                },
                "result": {
                    "query_id": query_idx,
                    "found_count": found_count,
                    "candidates_count": len(ml_results[query_idx][1]),
                    "ml_time": q_ml,
                    "verification_time": q_ged,
                    "query_time": q_total,
                    "gt_count": gt_count,
                    "correct_count": correct,
                    "recall": recall if recall is not None else 0.0,
                    "precision": precision if precision is not None else 0.0,
                    "f1": f1 if f1 is not None else 0.0
                }
            }
            query_file = os.path.join(per_query_dir, f"query_{query_idx}.json")
            with open(query_file, 'w') as f:
                json.dump(query_result, f, indent=2)

        # Save summary report
        summary_file = os.path.join(results_base, f"summary_report_q{query_start}-{query_end}.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("SIMILARITY SEARCH SUMMARY REPORT\n")
            f.write("=" * 120 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {FLAGS.dataset}\n")
            f.write(f"Query Range: {query_start} to {query_end}\n")
            f.write(f"Tau Threshold: {tau}\n")
            f.write(f"Error Margin: {error_margin}\n")
            f.write(f"Parallel: {use_parallel}\n")
            if use_parallel:
                f.write(f"Num Workers: {num_workers}\n")
            f.write("=" * 120 + "\n\n")

            # Summary table header
            f.write(f"{'Query':<8} {'GT':<8} {'Found':<8} {'Correct':<8} {'Recall':<10} {'Precision':<10} {'F1':<10} {'ML(s)':<10} {'GED(s)':<10} {'Total(s)':<10}\n")
            f.write("-" * 120 + "\n")

            # Per-query rows
            for query_idx, recall, precision, f1, gt_count, found_count, correct, q_ml, q_ged, q_total in query_results:
                recall_str = f"{recall:.4f}" if recall is not None else "N/A"
                precision_str = f"{precision:.4f}" if precision is not None else "N/A"
                f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                f.write(f"{query_idx:<8} {gt_count:<8} {found_count:<8} {correct:<8} {recall_str:<10} {precision_str:<10} {f1_str:<10} {q_ml:<10.4f} {q_ged:<10.4f} {q_total:<10.4f}\n")

            f.write("-" * 120 + "\n")
            f.write(f"{'AVG':<8} {avg_gt:<8.2f} {avg_found:<8.2f} {avg_correct:<8.2f} {avg_recall_str:<10} {avg_precision_str:<10} {avg_f1_str:<10} {avg_ml:<10.4f} {avg_ged:<10.4f} {avg_total:<10.4f}\n")
            f.write(f"(Recall/Precision/F1 based on {len(recalls)} queries with GT>0, {len(precisions)} with Found>0, {len(f1s)} with both)\n\n")

            # Time analysis section
            f.write("--- Time Analysis ---\n")
            f.write(f"GHash: Total={total_time:.2f}s, Avg/query={avg_total:.4f}s\n")
            f.write(f"  ML Filtering:  Total={total_ml_time:.2f}s, Avg={avg_ml:.4f}s\n")
            f.write(f"  {ged_method}:       Wall={total_bssged_wall_time:.2f}s, CPU={total_bssged_cpu_time:.2f}s, Avg={avg_ged:.4f}s\n")
            f.write("\n" + "=" * 120 + "\n")

        print(f"\n[Results saved]")
        print(f"  Per-query JSON: {per_query_dir}/")
        print(f"  Summary report: {summary_file}")
    else:
        print("\n[Results not saved (--no_save flag)]")


#!/usr/bin/env python3
"""
Similarity Search using Incremental k-NN
Uses k-NN as a black box: start with k=1, check if dist <= tau, increment k until dist > tau
With ground truth comparison support
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
import numpy as np
import struct
import networkx as nx
import ast
from tqdm import tqdm

# Default: suppress detailed logs (set to WARNING)
logging.basicConfig(level=logging.WARNING)

from gisma_ged_calculator import GismaGEDCalculator
from lan_original_search import LANOriginalSearchEngine, GEDCalculator


def load_ground_truth(ground_truth_file, tau):
    """
    Load ground truth for similarity search.

    Ground truth format: query_id exact_distance [list of graph_ids]
    e.g., "1 2.0 [10528, 14121, 17738]" means graphs 10528, 14121, 17738
          have exactly distance 2.0 from query 1.

    For similarity search (distance <= tau), we need to accumulate all graphs
    with distance <= tau for each query.

    Args:
        ground_truth_file: Path to ground_truth.txt
        tau: Distance threshold

    Returns:
        dict: query_id -> set of graph_ids with distance <= tau
    """
    ground_truth = {}  # query_id -> set of graph_ids

    with open(ground_truth_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: query_id distance [graph_ids]
            # Find the position of '['
            bracket_pos = line.find('[')
            if bracket_pos == -1:
                continue

            # Parse prefix: "query_id distance"
            prefix = line[:bracket_pos].strip()
            parts = prefix.split()
            if len(parts) < 2:
                continue

            query_id = int(parts[0])
            distance = float(parts[1])

            # Parse graph list
            graph_list_str = line[bracket_pos:]
            try:
                graph_ids = ast.literal_eval(graph_list_str)
            except:
                continue

            # Only include if distance <= tau
            if distance <= tau:
                if query_id not in ground_truth:
                    ground_truth[query_id] = set()
                ground_truth[query_id].update(graph_ids)

    return ground_truth


def compute_metrics(found_ids, ground_truth_ids):
    """
    Compute precision, recall, F1 for similarity search results.

    Args:
        found_ids: Set of graph IDs found by search
        ground_truth_ids: Set of graph IDs in ground truth

    Returns:
        dict with precision, recall, f1, true_positives, false_positives, false_negatives
        Note: precision/recall/f1 can be None (N/A) when:
        - GT=0 → Recall=N/A
        - Found=0 → Precision=N/A
        - Either is N/A → F1=N/A
    """
    true_positives = len(found_ids & ground_truth_ids)
    false_positives = len(found_ids - ground_truth_ids)
    false_negatives = len(ground_truth_ids - found_ids)

    # Precision: N/A if Found=0
    if len(found_ids) == 0:
        precision = None
    else:
        precision = true_positives / len(found_ids)

    # Recall: N/A if GT=0
    if len(ground_truth_ids) == 0:
        recall = None
    else:
        recall = true_positives / len(ground_truth_ids)

    # F1: N/A if either precision or recall is N/A
    if precision is None or recall is None:
        f1 = None
    elif (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'ground_truth_size': len(ground_truth_ids)
    }


def load_query_embeddings(embedding_file):
    """Load query embeddings (GREED format: ii header + [id + emb] × N)"""
    with open(embedding_file, 'rb') as f:
        # GREED format: 2 ints header
        num_graphs, embedding_dim = struct.unpack('ii', f.read(8))

        # Read embeddings (each entry: graph_id + embedding)
        embeddings_list = []
        for i in range(num_graphs):
            graph_id = struct.unpack('i', f.read(4))[0]
            emb = np.frombuffer(f.read(embedding_dim * 4), dtype=np.float32).copy()
            embeddings_list.append(emb)

        embeddings = np.array(embeddings_list, dtype=np.float32)

    return embeddings


def load_queries(query_file):
    """Load query graphs from queries.txt (Gisma format: ID, v, e)"""
    queries = []
    current_graph = None

    with open(query_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'ID':
                if current_graph is not None:
                    queries.append(current_graph)
                graph_id = parts[1]
                current_graph = nx.Graph(id=graph_id)

            elif parts[0] == 'v' and current_graph is not None:
                node_id = int(parts[1])
                label = parts[2] if len(parts) > 2 else "0"
                current_graph.add_node(node_id, label=label)

            elif parts[0] == 'e' and current_graph is not None:
                src = int(parts[1])
                dst = int(parts[2])
                current_graph.add_edge(src, dst)

    if current_graph is not None:
        queries.append(current_graph)

    return queries


def find_entry_point_by_embedding(query_emb, db_embeddings, graph_ids, k=1, num_candidates=4, seed=42):
    """
    Find entry point using embedding similarity from random candidates.

    Args:
        query_emb: Query embedding vector
        db_embeddings: Database embeddings matrix
        graph_ids: List of graph IDs
        k: Number of entry points to return
        num_candidates: Number of random candidates to sample
        seed: Random seed for reproducibility

    Returns:
        List of graph IDs (best entry points from random candidates)
    """
    import random
    random.seed(seed)

    # Randomly select num_candidates candidates
    num_graphs = len(graph_ids)
    candidate_indices = random.sample(range(num_graphs), min(num_candidates, num_graphs))

    # Compute embedding distances between these candidates and the query
    candidate_distances = []
    for idx in candidate_indices:
        dist = np.linalg.norm(db_embeddings[idx] - query_emb)
        candidate_distances.append((idx, dist))

    # Sort by distance, take the nearest k
    candidate_distances.sort(key=lambda x: x[1])
    nearest_indices = [idx for idx, dist in candidate_distances[:k]]

    return [graph_ids[i] for i in nearest_indices]


def similarity_search_incremental(engine, query_graph, tau, ef=50, entry_point_id=None, max_ef=10000, query_emb=None):
    """
    Similarity search using incremental ef approach.

    Optimized approach:
    1. Search with ef candidates (k=ef to get all candidates)
    2. Check if the furthest candidate (ef-th) distance <= tau
    3. If yes, expand ef += ef_init and search again
    4. Stop when furthest candidate distance > tau
    5. Return all results with distance <= tau

    Args:
        engine: LANOriginalSearchEngine instance
        query_graph: Query graph
        tau: Distance threshold
        ef: Initial search ef parameter (also the increment step)
        entry_point_id: Entry point for HNSW search
        max_ef: Maximum ef to try (safety limit)
        query_emb: Query embedding vector (important for correct distance computation)

    Returns:
        results: List of (distance, graph) tuples where distance <= tau
        stats: Search statistics
    """
    total_search_time = 0
    total_ged_computations = 0
    total_nodes_visited = 0

    ef_init = ef
    current_ef = ef
    knn_results = []
    iterations = 0

    while current_ef <= max_ef:
        iterations += 1

        # Search with k=current_ef to get all candidates
        knn_results, stats = engine.search_k_nearest(
            query_graph,
            k=current_ef,
            ef=current_ef,
            entry_point_id=entry_point_id,
            query_emb=query_emb
        )

        total_search_time += stats['search_time']
        total_ged_computations += stats['distance_computations']
        total_nodes_visited += stats['nodes_visited']

        # Check if we got enough results
        if len(knn_results) < current_ef:
            # No more results available in the graph
            break

        # Check the furthest candidate (last one) distance
        furthest_dist = knn_results[-1][0]

        if furthest_dist > tau:
            # Furthest candidate exceeds threshold, we have all results
            break

        # All candidates within tau, need to expand search
        current_ef += ef_init

    # Filter results to only include those within tau
    similar_results = [(dist, g) for dist, g in knn_results if dist <= tau]

    final_stats = {
        'search_time': total_search_time,
        'distance_computations': total_ged_computations,
        'nodes_visited': total_nodes_visited,
        'final_ef': current_ef,
        'iterations': iterations,
        'num_results': len(similar_results)
    }

    return similar_results, final_stats


def main():
    parser = argparse.ArgumentParser(description='Similarity Search using Incremental k-NN')
    parser.add_argument('--dataset', type=str, default='AIDS',
                        help='Dataset name: AIDS, PubChem, Chemical1M, SYN (default: AIDS)')
    parser.add_argument('--query_start', '-qs', type=int, default=0,
                        help='Start query index (default: 0)')
    parser.add_argument('--query_end', '-qe', type=int, default=99,
                        help='End query index (default: 99)')
    parser.add_argument('--tau', '-t', type=float, required=True,
                        help='Distance threshold (required)')
    parser.add_argument('--ef', type=int, default=50,
                        help='Search ef parameter (default: 50)')
    parser.add_argument('--timeout', type=float, default=10,
                        help='GED computation timeout in seconds (default: 10)')
    parser.add_argument('--max_ef', type=int, default=10000,
                        help='Maximum ef to try (default: 10000)')
    parser.add_argument('--no_pruning', action='store_true',
                        help='Disable neighbor pruning')
    parser.add_argument('--pruning_ratio', type=float, default=0.2,
                        help='Neighbor pruning ratio (default: 0.2)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to files')
    parser.add_argument('--beam_width', type=int, default=100,
                        help='Beam search width (default: 100)')
    parser.add_argument('--pruning_method', type=str, default='mrk',
                        choices=['embedding', 'mrk'],
                        help='Neighbor pruning method (default: mrk)')
    parser.add_argument('--mrk_model_epoch', type=int, default=999,
                        help='Mrk model epoch number (default: 999)')
    parser.add_argument('--ds', type=float, default=1.0,
                        help='Gamma step size for batch expansion (default: 1.0)')
    parser.add_argument('--init_method', type=str, default='embedding',
                        choices=['random', 'embedding', 'model'],
                        help='Entry point selection method: random, embedding (default), model (Init model)')
    parser.add_argument('--use_cg', action='store_true',
                        help='Use CG (Compressed GNN-Graph) for faster query GIN embedding')
    parser.add_argument('--expand_ef', action='store_true',
                        help='Enable incremental ef expansion (default: off, single-shot search)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output (default: quiet with progress bar)')
    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('gisma_ged_calculator').setLevel(logging.INFO)
        logging.getLogger('lan_original_search').setLevel(logging.INFO)

    dataset = args.dataset
    tau = args.tau
    verbose = args.verbose

    if verbose:
        print("=" * 70)
        print(f"Similarity Search (Incremental k-NN) - Dataset: {dataset}")
        print(f"Threshold tau = {tau}")
        print("=" * 70)

    # Paths (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gisma_base = os.path.join(script_dir, '..', 'Gisma')
    dataset_path = os.path.join(gisma_base, 'datasets', dataset)
    query_file = os.path.join(dataset_path, 'queries.txt')
    query_emb_file = os.path.join(gisma_base, 'embeddings', dataset, f'{dataset}_queries_embeddings.bin')
    ground_truth_file = os.path.join(dataset_path, 'ground_truth.txt')

    # Load queries
    if verbose:
        print("\nLoading queries...")
    queries = load_queries(query_file)
    if verbose:
        print(f"Loaded {len(queries)} queries")

    # Load ground truth
    if verbose:
        print("Loading ground truth...")
    ground_truth = load_ground_truth(ground_truth_file, tau)
    queries_with_gt = len([q for q in ground_truth if len(ground_truth[q]) > 0])
    if verbose:
        print(f"Ground truth loaded for {queries_with_gt} queries with tau<={tau}")

    # Load query embeddings
    if verbose:
        print("Loading query embeddings...")
    query_embeddings = load_query_embeddings(query_emb_file)
    if verbose:
        print(f"Query embeddings shape: {query_embeddings.shape}")

    # Initialize GED calculator
    if verbose:
        print("\nInitializing GED calculator...")
    ged_calc = GEDCalculator(ged_method='gisma', dataset_path=dataset_path)
    if ged_calc.gisma_calculator is None:
        print("ERROR: Gisma GED calculator failed to initialize!")
        print("Please check that GismaProject executable exists in ../Gisma/build/")
        sys.exit(1)
    ged_calc.gisma_calculator.use_app_for_computation = True  # Use AppForComputation (exact_max_iter=1000000)
    ged_calc.gisma_calculator.timeout = args.timeout
    ged_calc.gisma_calculator.beam_width = args.beam_width

    # Load index
    if verbose:
        print("Loading index...")
    index_file = os.path.join('indexes', f'{dataset}_LAN_index.pkl')

    # Prepare mrk model parameters if needed
    mrk_kwargs = {}
    if args.pruning_method == 'mrk':
        mrk_kwargs = {
            'mrk_model_dir': os.path.join(script_dir, 'models', 'mrk', dataset.lower()),
            'mrk_model_epoch': args.mrk_model_epoch,
            'mrk_emb_path': os.path.join(script_dir, 'emb', dataset.lower() + '512'),
            'mrk_graph_file': os.path.join(gisma_base, 'datasets', dataset, 'db.txt'),
        }

    # Prepare init model parameters if needed
    init_kwargs = {}
    if args.init_method == 'model':
        init_kwargs['init_model_dir'] = os.path.join(script_dir, 'models', 'init', dataset.lower())

    engine = LANOriginalSearchEngine(
        index_file,
        ged_calc,
        use_neighbor_pruning=not args.no_pruning,
        pruning_ratio=args.pruning_ratio,
        pruning_method=args.pruning_method,
        ds=args.ds,
        use_cg=args.use_cg,
        **mrk_kwargs,
        **init_kwargs
    )

    # Load database embeddings for entry point selection
    if verbose:
        print("Loading database embeddings...")
    db_embeddings = ged_calc.gisma_calculator.embeddings
    db_graph_ids = sorted(ged_calc.gisma_calculator.all_graphs.keys(), key=int)
    if verbose:
        print(f"Database: {len(db_graph_ids)} graphs")

    # Query range
    query_start = args.query_start
    query_end = min(args.query_end, len(queries) - 1)
    ef_search = args.ef
    effective_max_ef = args.max_ef if args.expand_ef else ef_search

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Testing queries {query_start} to {query_end}")
        print(f"tau={tau}, ef={ef_search}, timeout={args.timeout}s, max_ef={effective_max_ef}, expand_ef={args.expand_ef}")
        pruning_desc = 'OFF' if args.no_pruning else f'ON (method={args.pruning_method}, ratio={args.pruning_ratio})'
        batch_desc = f', batch_expansion=ON, ds={args.ds}' if args.pruning_method == 'mrk' else ''
        print(f"Neighbor pruning: {pruning_desc}{batch_desc}")
        print(f"Entry point: {args.init_method}")
        print(f"{'=' * 70}\n")

    total_search_time = 0
    total_ged_computations = 0
    total_results = 0
    num_queries = 0

    # Metrics accumulation (separate counters for N/A handling)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    queries_with_precision = 0
    queries_with_recall = 0
    queries_with_f1 = 0

    # Per-query results for final table
    query_results_table = []

    # Set up per-query save directory (save immediately after each query)
    per_query_dir = None
    if getattr(args, 'save', False):
        results_base = os.path.join(script_dir, 'results', dataset,
                                    f'tau{int(tau)}', f'ef{ef_search}')
        per_query_dir = os.path.join(results_base, 'per_query')
        os.makedirs(per_query_dir, exist_ok=True)

    # Create progress bar (only in quiet mode)
    query_range = range(query_start, query_end + 1)
    if not verbose:
        query_range = tqdm(query_range, desc=f"{dataset} tau={tau}", unit="query")

    for i in query_range:
        query_graph = queries[i]
        query_emb = query_embeddings[i]

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Query {i}: ID={query_graph.graph.get('id')}, "
                  f"Nodes={query_graph.number_of_nodes()}, "
                  f"Edges={query_graph.number_of_edges()}")

        # Get ground truth for this query
        gt_ids = ground_truth.get(i, set())
        if verbose:
            print(f"Ground truth: {len(gt_ids)} graphs within tau={tau}")

        # Find entry point
        if args.init_method == 'model' and engine.init_model is not None:
            # Use learned Init model
            entry_point_id = engine.predict_entry_point(query_graph)
            if verbose:
                print(f"Entry point (by Init model): ID={entry_point_id}")
        elif args.init_method == 'random':
            import random
            random.seed(42 + i)
            entry_point_id = random.choice(db_graph_ids)
            if verbose:
                print(f"Entry point (random): ID={entry_point_id}")
        else:
            # Default: embedding-based (random 4 candidates, pick best)
            entry_point_ids = find_entry_point_by_embedding(
                query_emb, db_embeddings, db_graph_ids, k=1,
                num_candidates=4, seed=42 + i
            )
            entry_point_id = entry_point_ids[0]
            if verbose:
                print(f"Entry point (by embedding from 4 random): ID={entry_point_id}")

        # Similarity search
        results, stats = similarity_search_incremental(
            engine,
            query_graph,
            tau=tau,
            ef=ef_search,
            entry_point_id=entry_point_id,
            max_ef=effective_max_ef,
            query_emb=query_emb
        )

        total_search_time += stats['search_time']
        total_ged_computations += stats['distance_computations']
        total_results += stats['num_results']
        num_queries += 1

        # Extract found IDs
        found_ids = set()
        for dist, g in results:
            gid = g.graph.get('id')
            found_ids.add(int(gid))

        # Compute metrics against GED ground truth
        metrics = compute_metrics(found_ids, gt_ids)

        # Accumulate metrics (only if not N/A)
        if metrics['precision'] is not None:
            total_precision += metrics['precision']
            queries_with_precision += 1
        if metrics['recall'] is not None:
            total_recall += metrics['recall']
            queries_with_recall += 1
        if metrics['f1'] is not None:
            total_f1 += metrics['f1']
            queries_with_f1 += 1

        # Display results
        if verbose:
            print(f"\nFound {len(results)} graphs (Ground truth: {len(gt_ids)}):")
            for rank, (dist, g) in enumerate(results[:20], 1):  # Show max 20
                gid = g.graph.get('id')
                nodes = g.number_of_nodes()
                edges = g.number_of_edges()
                in_gt = int(gid) in gt_ids
                gt_label = "[GT]" if in_gt else "[  ]"
                source = ged_calc.gisma_calculator.get_ged_source(query_graph, g)
                source_label = {
                    'exact': '[Exact]',
                    'approx': '[Approx]'
                }.get(source, '[?]')
                print(f"  {rank:3d}. ID={gid:>6}, Dist={dist:6.2f} {source_label:8s} {gt_label}, "
                      f"Nodes={nodes:2d}, Edges={edges:2d}")

            if len(results) > 20:
                print(f"  ... and {len(results) - 20} more")

            # Show metrics
            prec_str = f"{metrics['precision']:.3f}" if metrics['precision'] is not None else "N/A"
            recall_str = f"{metrics['recall']:.3f}" if metrics['recall'] is not None else "N/A"
            f1_str = f"{metrics['f1']:.3f}" if metrics['f1'] is not None else "N/A"
            print(f"\nGED Metrics: Precision={prec_str}, Recall={recall_str}, F1={f1_str}")
            print(f"  TP={metrics['true_positives']}, FP={metrics['false_positives']}, "
                  f"FN={metrics['false_negatives']}")

            print(f"\nStats: Time={stats['search_time']:.2f}s, "
                  f"GED_computations={stats['distance_computations']}, "
                  f"Final_ef={stats['final_ef']}, Iterations={stats['iterations']}, "
                  f"Results={stats['num_results']}")

        # Store for final table
        query_results_table.append({
            'query_id': i,
            'gt_size': len(gt_ids),
            'found': len(results),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['true_positives'],
            'fp': metrics['false_positives'],
            'fn': metrics['false_negatives'],
            'time': stats['search_time'],
            'ged_comps': stats['distance_computations'],
            'final_ef': stats['final_ef'],
            'iterations': stats['iterations']
        })

        # Save immediately after each query
        if per_query_dir is not None:
            r = query_results_table[-1]
            query_result = {
                "metadata": {
                    "dataset": dataset, "model": "LAN", "tau_threshold": tau,
                    "ef": ef_search, "timeout": args.timeout,
                    "beam_width": args.beam_width,
                    "pruning_method": args.pruning_method,
                    "pruning_ratio": args.pruning_ratio if not args.no_pruning else None,
                    "no_pruning": args.no_pruning,
                    "init_method": args.init_method,
                    "use_cg": args.use_cg,
                    "ds": args.ds,
                    "query_id": r['query_id']
                },
                "result": {
                    "query_id": r['query_id'], "gt_count": r['gt_size'],
                    "found_count": r['found'], "correct_count": r['tp'],
                    "precision": r['precision'], "recall": r['recall'], "f1": r['f1'],
                    "query_time": r['time'], "ged_computations": r['ged_comps'],
                    "final_ef": r['final_ef'], "iterations": r['iterations']
                }
            }
            query_file = os.path.join(per_query_dir, f"query_{r['query_id']}.json")
            with open(query_file, 'w') as f:
                json.dump(query_result, f, indent=2)

    # Summary (always show)
    print(f"\n{'=' * 70}")
    print(f"Summary - {dataset} tau={tau}")
    print(f"{'=' * 70}")
    print(f"Total queries: {num_queries}, Avg time: {total_search_time/num_queries:.2f}s, Avg GED comps: {total_ged_computations/num_queries:.1f}")

    # Show key metrics
    prec_avg = total_precision/queries_with_precision if queries_with_precision > 0 else None
    recall_avg = total_recall/queries_with_recall if queries_with_recall > 0 else None
    f1_avg = total_f1/queries_with_f1 if queries_with_f1 > 0 else None
    prec_str = f"{prec_avg:.3f}" if prec_avg is not None else "N/A"
    recall_str = f"{recall_avg:.3f}" if recall_avg is not None else "N/A"
    f1_str = f"{f1_avg:.3f}" if f1_avg is not None else "N/A"
    print(f"Precision: {prec_str}, Recall: {recall_str}, F1: {f1_str}")

    if verbose:
        print(f"\nGED Calculator Stats:")
        ged_calc.gisma_calculator.print_stats()

    # Final summary table - GED metrics (always show)
    print(f"\n{'=' * 120}")
    print("Results Table (GED Ground Truth)")
    print(f"{'=' * 120}")
    print(f"{'Query':>6} | {'GT':>4} | {'Found':>5} | {'TP':>4} | {'FP':>4} | {'FN':>4} | "
          f"{'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Time':>7} | {'GED#':>6} | {'ef':>5} | {'Iter':>4}")
    print("-" * 120)

    for r in query_results_table:
        prec_str = f"{r['precision']:>6.3f}" if r['precision'] is not None else "   N/A"
        recall_str = f"{r['recall']:>6.3f}" if r['recall'] is not None else "   N/A"
        f1_str = f"{r['f1']:>6.3f}" if r['f1'] is not None else "   N/A"
        print(f"{r['query_id']:>6} | {r['gt_size']:>4} | {r['found']:>5} | {r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | "
              f"{prec_str} | {recall_str} | {f1_str} | {r['time']:>6.2f}s | {r['ged_comps']:>6} | {r['final_ef']:>5} | {r['iterations']:>4}")

    print("-" * 120)

    # Avg row
    n = len(query_results_table)
    avg_gt = sum(r['gt_size'] for r in query_results_table) / n
    avg_found = sum(r['found'] for r in query_results_table) / n
    avg_tp = sum(r['tp'] for r in query_results_table) / n
    avg_fp = sum(r['fp'] for r in query_results_table) / n
    avg_fn = sum(r['fn'] for r in query_results_table) / n
    avg_time = total_search_time / n
    avg_ged = total_ged_computations / n

    avg_prec_str = f"{total_precision / queries_with_precision:>6.3f}" if queries_with_precision > 0 else "   N/A"
    avg_recall_str = f"{total_recall / queries_with_recall:>6.3f}" if queries_with_recall > 0 else "   N/A"
    avg_f1_str = f"{total_f1 / queries_with_f1:>6.3f}" if queries_with_f1 > 0 else "   N/A"

    print(f"{'Avg':>6} | {avg_gt:>4.1f} | {avg_found:>5.1f} | {avg_tp:>4.1f} | {avg_fp:>4.1f} | {avg_fn:>4.1f} | "
          f"{avg_prec_str} | {avg_recall_str} | {avg_f1_str} | {avg_time:>6.2f}s | {avg_ged:>6.1f} | {'':>5} | {'':>4}")

    print(f"{'=' * 120}")

    # ========== Save Summary Report ==========
    if getattr(args, 'save', False):
        if args.no_pruning:
            results_base = os.path.join(script_dir, 'results', dataset, f'tau{int(tau)}', 'no_pruning')
        else:
            results_base = os.path.join(script_dir, 'results', dataset, f'tau{int(tau)}', f'ratio_{args.pruning_ratio}')

        os.makedirs(results_base, exist_ok=True)
        # Save summary report
        summary_file = os.path.join(results_base, f"summary_report_q{query_start}-{query_end}.txt")
        print(f"Saving summary report to: {summary_file}")

        with open(summary_file, 'w') as f:
            f.write(f"LAN Similarity Search Results\n")
            f.write(f"{'=' * 120}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Query Range: {query_start} - {query_end}\n")
            f.write(f"Tau: {tau}\n")
            f.write(f"ef: {ef_search}, timeout: {args.timeout}s\n")
            f.write(f"Neighbor Pruning: {'OFF' if args.no_pruning else f'ON (ratio={args.pruning_ratio})'}\n")
            f.write(f"{'=' * 120}\n\n")

            # Results table
            f.write(f"{'Query':>6} | {'GT':>4} | {'Found':>5} | {'TP':>4} | {'FP':>4} | {'FN':>4} | "
                    f"{'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Time':>7} | {'GED#':>6} | {'ef':>5} | {'Iter':>4}\n")
            f.write("-" * 120 + "\n")

            for r in query_results_table:
                prec_str = f"{r['precision']:>6.3f}" if r['precision'] is not None else "   N/A"
                recall_str = f"{r['recall']:>6.3f}" if r['recall'] is not None else "   N/A"
                f1_str = f"{r['f1']:>6.3f}" if r['f1'] is not None else "   N/A"
                f.write(f"{r['query_id']:>6} | {r['gt_size']:>4} | {r['found']:>5} | {r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | "
                        f"{prec_str} | {recall_str} | {f1_str} | {r['time']:>6.2f}s | {r['ged_comps']:>6} | {r['final_ef']:>5} | {r['iterations']:>4}\n")

            f.write("-" * 120 + "\n")
            f.write(f"{'Avg':>6} | {avg_gt:>4.1f} | {avg_found:>5.1f} | {avg_tp:>4.1f} | {avg_fp:>4.1f} | {avg_fn:>4.1f} | "
                    f"{avg_prec_str} | {avg_recall_str} | {avg_f1_str} | {avg_time:>6.2f}s | {avg_ged:>6.1f} | {'':>5} | {'':>4}\n")
            f.write(f"{'=' * 120}\n\n")

            # Summary stats
            f.write(f"Summary Statistics\n")
            f.write(f"-" * 40 + "\n")
            f.write(f"Total queries: {num_queries}\n")
            f.write(f"Total search time: {total_search_time:.2f}s\n")
            f.write(f"Total GED computations: {total_ged_computations}\n")
            f.write(f"Average search time: {avg_time:.2f}s\n")
            f.write(f"Average GED computations: {avg_ged:.1f}\n")
            if queries_with_precision > 0:
                f.write(f"Average Precision: {total_precision/queries_with_precision:.3f}\n")
            if queries_with_recall > 0:
                f.write(f"Average Recall: {total_recall/queries_with_recall:.3f}\n")
            if queries_with_f1 > 0:
                f.write(f"Average F1: {total_f1/queries_with_f1:.3f}\n")

        print(f"\nResults saved successfully!")


if __name__ == '__main__':
    main()

"""
Worker functions for batch-parallel GED computation (GEDGW, GEDIOT, GEDHOT)
Separated into its own module for multiprocessing compatibility
"""

# Global variables to store model and data (loaded independently per worker)
_worker_model = None
_worker_gn = None
_worker_edge_index = None
_worker_features = None

def initialize_worker(model_args):
    """
    Initialize worker: load data independently (bypasses /dev/shm).
    Uses the same load_all_graphs and load_labels as main process.
    """
    import multiprocessing
    import torch
    from utils import load_all_graphs, load_labels

    global _worker_model, _worker_gn, _worker_edge_index, _worker_features

    worker_name = multiprocessing.current_process().name
    print(f"[{worker_name}] Loading data independently (bypasses /dev/shm)...")

    # Load graphs from graphs_cache.pt
    dataset = model_args['dataset']
    abs_path = model_args.get('abs_path', '')
    db_num, query_num, test_num, graphs = load_all_graphs(abs_path, dataset, parallel=False, max_workers=1)
    print(f"[{worker_name}] Loaded {len(graphs)} graphs")

    # Load labels and features
    global_labels, features = load_labels(abs_path, dataset, parallel=False, max_workers=1, graphs=graphs)
    number_of_labels = len(global_labels)

    # Convert to torch tensors (same as Trainer.transfer_data_to_torch)
    edge_index = []
    for g in graphs:
        edge = g['graph']
        edge = edge + [[y, x] for x, y in edge]
        edge = edge + [[x, x] for x in range(g['n'])]  # Add self-loops (critical for GEDGW!)
        edge_index.append(torch.LongTensor(edge).t())

    _worker_gn = [g['n'] for g in graphs]
    _worker_edge_index = edge_index
    _worker_features = [torch.FloatTensor(f) for f in features]

    print(f"[{worker_name}] Data loaded and converted to tensors")

    # Load model if needed
    method = model_args.get('method', 'GEDGW')
    if method == 'GEDIOT':
        from models import GEDIOT
        args = model_args['args']
        _worker_model = GEDIOT(args, number_of_labels)

        model_path = model_args['model_path']
        epoch = model_args['model_epoch_start']
        model_file = f"{model_path}/GEDIOT/{dataset}_{epoch}"
        _worker_model.load_state_dict(torch.load(model_file, map_location='cpu'))
        _worker_model.eval()
        print(f"[{worker_name}] GEDIOT model loaded")

    print(f"[{worker_name}] Initialization complete")

def compute_batch_distances_worker(args):
    """
    Worker function to compute GED for a batch of graph pairs
    Uses worker-local data loaded by initialize_worker

    Args:
        args: tuple of (pairs, model_args) where
            - pairs: list of (query_id, candidate_id) tuples
            - model_args: dict containing parameters (NOT data)

    Returns:
        list of (query_id, candidate_id, predicted_ged, computation_time) tuples
    """
    import multiprocessing
    import time
    global _worker_model, _worker_gn, _worker_edge_index, _worker_features

    pairs, model_args = args
    results = []
    method = model_args.get('method', 'GEDGW')

    worker_name = multiprocessing.current_process().name
    print(f"[{worker_name}] Starting computation of {len(pairs)} pairs...")

    for orig_query_id, orig_candidate_id in pairs:
        try:
            # Map query_id to actual graph index (query graphs are stored after db graphs)
            query_graph_idx = model_args['db_num'] + orig_query_id
            candidate_id = orig_candidate_id

            # Get node counts from worker-local data
            n1 = _worker_gn[query_graph_idx]
            n2 = _worker_gn[candidate_id]

            # All methods require n1 <= n2, swap if needed
            if n1 > n2:
                # Swap graphs
                graph_idx_1 = candidate_id
                graph_idx_2 = query_graph_idx
                n1, n2 = n2, n1
            else:
                graph_idx_1 = query_graph_idx
                graph_idx_2 = candidate_id

            # Prepare data from worker-local data
            edge_idx_1 = _worker_edge_index[graph_idx_1]
            edge_idx_2 = _worker_edge_index[graph_idx_2]
            num_edges_1 = edge_idx_1.shape[1]
            num_edges_2 = edge_idx_2.shape[1]

            data = {
                "id_1": graph_idx_1,
                "id_2": graph_idx_2,
                "n1": n1,
                "n2": n2,
                "edge_index_1": edge_idx_1,
                "edge_index_2": edge_idx_2,
                "features_1": _worker_features[graph_idx_1],
                "features_2": _worker_features[graph_idx_2],
                "avg_v": (n1 + n2) / 2.0,
                "hb": max(n1, n2) + max(num_edges_1, num_edges_2)
            }

            # Start timing GED computation
            start_time = time.time()

            # Compute GED based on method
            if method == 'GEDGW':
                from models import GEDGW
                gedgw_model = GEDGW(data, model_args['args'])
                _, predicted_ged = gedgw_model.process()

            elif method == 'GEDIOT':
                # Use pre-loaded model
                import torch
                with torch.no_grad():
                    _, predicted_ged, _ = _worker_model(data)
                    predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged

            # Note: GEDHOT is not computed here, it's computed in main process
            # as min(GEDGW, GEDIOT)

            # End timing
            computation_time = time.time() - start_time

            # Store with original IDs and computation time
            results.append((orig_query_id, orig_candidate_id, predicted_ged, computation_time))

        except Exception as e:
            # Print first few errors for debugging
            if len(results) == 0:
                print(f"Worker error on query {orig_query_id}, candidate {orig_candidate_id}: {e}")
                import traceback
                traceback.print_exc()
            continue

    return results

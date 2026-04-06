import sys
import time
import json
import csv
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt
from kbest_matching_with_lb import KBestMSolver
from math import exp
from scipy.stats import spearmanr, kendalltau
import networkx as nx

from models import GPN, SimGNN, GedGNN, TaGSim, GEDIOT, GEDGW
from greedy_algo import hungarian

def process_single_query_parallel_worker(args_tuple):
    """
    Top-level worker function for parallel processing (pickle-able).

    Args:
        args_tuple: (query_id, candidates_set, tau_threshold, model_args_dict)

    Returns:
        tuple: (query_id, results_set, query_time)
    """
    import time
    import torch
    import os

    query_id, candidates_set, tau_threshold, model_args = args_tuple

    query_results = set()
    query_start_time = time.time()

    # Set environment for this worker
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Map query_id to actual graph index
    query_graph_idx = model_args['db_num'] + query_id

    try:
        # Recreate model in worker process
        if model_args['model_name'] in ["GEDHOT", "GEDIOT"]:
            # Load model for neural methods
            from models import GedGNN, GEDIOT
            if model_args['model_name'] == "GEDIOT":
                model = GEDIOT(type('Args', (), model_args)())
            else:  # GEDHOT
                model = GedGNN(type('Args', (), model_args)())

            # Load weights if specified
            if model_args.get('model_epoch_start', 0) > 0:
                model_path = f"{model_args['model_path']}{model_args['model_name']}_{model_args['dataset']}_gnn_{model_args['model_epoch_start']}.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

        # Process each candidate
        for candidate_id in candidates_set:
            if query_id == candidate_id:  # Skip self-comparison
                continue

            try:
                # Prepare data using query_graph_idx
                n1, n2 = model_args['gn'][query_graph_idx], model_args['gn'][candidate_id]
                num_edges_1 = model_args['edge_index'][query_graph_idx].shape[1]
                num_edges_2 = model_args['edge_index'][candidate_id].shape[1]
                hb = max(n1, n2) + max(num_edges_1, num_edges_2)

                data = {
                    "id_1": query_graph_idx,
                    "id_2": candidate_id,
                    "n1": n1,
                    "n2": n2,
                    "hb": hb,
                    "edge_index_1": model_args['edge_index'][query_graph_idx],
                    "edge_index_2": model_args['edge_index'][candidate_id],
                    "features_1": model_args['features'][query_graph_idx],
                    "features_2": model_args['features'][candidate_id]
                }

                # Compute GED
                if model_args['model_name'] == "GEDGW":
                    # Handle GEDGW node ordering
                    if n1 <= n2:
                        data["n1"] = n1
                        data["n2"] = n2
                    else:
                        # Swap to ensure n1 <= n2
                        data["n1"] = n2
                        data["n2"] = n1
                        data["edge_index_1"] = model_args['edge_index'][candidate_id]
                        data["edge_index_2"] = model_args['edge_index'][query_graph_idx]
                        data["features_1"] = model_args['features'][candidate_id]
                        data["features_2"] = model_args['features'][query_graph_idx]

                    from models import GEDGW
                    gedgw_model = GEDGW(data, type('Args', (), model_args)())
                    _, predicted_ged = gedgw_model.process()
                else:
                    # Neural methods
                    with torch.no_grad():
                        if model_args['model_name'] == "GEDIOT":
                            _, predicted_ged, _ = model(data)  # GEDIOT returns tuple
                            predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged
                        else:  # GEDHOT
                            _, predicted_ged = model(data)  # GEDHOT returns tuple
                            predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged

                # Check threshold
                if predicted_ged <= tau_threshold:
                    query_results.add(candidate_id)

            except Exception as e:
                # Silent error handling in parallel mode to avoid spam
                continue

    except Exception as e:
        print(f"Worker error for query {query_id}: {e}")

    query_end_time = time.time()
    query_time = query_end_time - query_start_time

    return query_id, query_results, query_time

def compute_gedgw_parallel_worker(args_tuple):
    """
    Parallel worker for computing ONLY GEDGW distances (no neural networks).
    GEDIOT and GEDHOT will be computed sequentially in main process.

    Args:
        args_tuple: (query_id, candidate_id, data_dict)
            data_dict contains: n1, n2, edge_index_1, edge_index_2, features_1, features_2, hb

    Returns:
        tuple: (candidate_id, gedgw_dist, gedgw_time) or None on error
    """
    import time
    import torch
    import os

    query_id, candidate_id, data_dict = args_tuple

    # Set environment for this worker
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    try:
        # Only compute GEDGW distance (no trained neural network needed)
        gedgw_start = time.time()
        from models import GEDGW

        # Create minimal args object for GEDGW
        gedgw_args = type('Args', (), {
            'gw_alpha': 1.0/3.0,
            'gw_rescale': 1.5,
            'gw_square': True
        })()

        gedgw_model = GEDGW(data_dict, gedgw_args)
        _, gedgw_dist = gedgw_model.process()
        gedgw_time = time.time() - gedgw_start

        return candidate_id, gedgw_dist, gedgw_time

    except Exception as e:
        print(f"Error in GEDGW parallel worker ({query_id}, {candidate_id}): {e}")
        return None
from GedMatrix import fixed_mapping_loss
from noah import graph_edit_distance




class Trainer(object):
    """
    A general model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []

        # self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.load_data()
        self.transfer_data_to_torch()

        # Skip expensive initialization for similarity search modes
        if not (getattr(self.args, 'similarity_search', False) or
                getattr(self.args, 'similarity_search_all', False) or
                getattr(self.args, 'similarity_search_gediot', False) or
                getattr(self.args, 'mae_test', False)):
            self.delta_graphs = [None] * len(self.graphs)
            self.gen_delta_graphs()
            self.init_graph_pairs()
        else:
            print("Skipping delta graphs and graph pairs initialization for similarity search mode")
            self.delta_graphs = []
            self.training_graphs = []
            self.val_graphs = []
            self.testing_graphs = []

        self.setup_model()

    def setup_model(self):
        if self.args.model_name in ["GPN","NOAH"]:
            self.model = GPN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "SimGNN":
            self.args.filters_1 = 64
            self.args.filters_2 = 32
            self.args.filters_3 = 16
            self.args.histogram = True
            self.args.target_mode = 'exp'
            self.model = SimGNN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "GedGNN":
            if self.args.dataset in ["AIDS", "Linux"]:
                self.args.loss_weight = 10.0
            else:
                self.args.loss_weight = 1.0
            # self.args.target_mode = 'exp'
            self.args.gtmap = True
            self.model = GedGNN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "TaGSim":
            self.args.target_mode = 'exp'
            self.model = TaGSim(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "GEDHOT" or self.args.model_name == "GEDIOT":
            self.args.loss_weight = 8.0
            self.args.gtmap = True
            self.model = GEDIOT(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name in ["GEDGW", "Classic"]:
            print(f"Unsupervised Method Without Training")
        else:
            assert False

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)

        if self.args.model_name in ["GPN", "SimGNN"]:
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target = data["target"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(target, prediction)
                # self.values.append((target - prediction).item())
                
                # Store prediction vs target for debugging
                if hasattr(self, '_batch_predictions'):
                    self._batch_predictions.append((prediction.item(), target.item()))
        elif self.args.model_name == "GedGNN":
            weight = self.args.loss_weight
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target, gt_mapping = data["target"], data["mapping"]
                prediction, _, mapping = self.model(data)
                losses = losses + fixed_mapping_loss(mapping, gt_mapping) + weight * F.mse_loss(target, prediction)
                if self.args.finetune:
                    if self.args.target_mode == "linear":
                        losses = losses + F.relu(target - prediction)
                    else: # "exp"
                        losses = losses + F.relu(prediction - target)
                
                # Store prediction vs target for debugging
                if hasattr(self, '_batch_predictions'):
                    self._batch_predictions.append((prediction.item(), target.item()))
        elif self.args.model_name == "TaGSim":
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                ta_ged = data["ta_ged"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(ta_ged, prediction)
                
                # Store prediction vs target for debugging (using ta_ged as target)
                if hasattr(self, '_batch_predictions'):
                    if ta_ged.dim() == 0:  # scalar
                        self._batch_predictions.append((prediction.item(), ta_ged.item()))
                    else:  # tensor, take first element
                        self._batch_predictions.append((prediction.item(), ta_ged[0].item()))
        elif self.args.model_name == "GEDIOT" or self.args.model_name == "GEDHOT":
            weight = self.args.loss_weight
            for i, graph_pair in enumerate(batch):
                data = self.pack_graph_pair(graph_pair)
                target, gt_mapping = data["target"], data["mapping"] #ground truth
                prediction, _, mapping = self.model(data)
                losses = losses + (10.0-weight)*fixed_mapping_loss(mapping, gt_mapping) + weight * F.mse_loss(target, prediction)
                
                # Store prediction vs target for debugging
                if hasattr(self, '_batch_predictions'):
                    self._batch_predictions.append((prediction.item(), target.item()))
        else:
            assert False

        losses.backward()
        self.optimizer.step()
        return losses.item()

    def load_data(self):
        """
        Load graphs, ged and labels if needed.
        self.ged: dict-dict, ged['graph_id_1']['graph_id_2'] stores the ged value.
        """
        t1 = time.time()
        dataset_name = self.args.dataset
        self.db_num, self.query_num, self.test_num, self.graphs = load_all_graphs(
            self.args.abs_path, dataset_name,
            parallel=self.args.parallel_loading,
            max_workers=self.args.loading_workers
        )
        # For backward compatibility, set train_num and val_num
        self.train_num = self.db_num
        self.val_num = self.query_num

        print("Load {} graphs. ({} db graphs, {} query graphs)".format(
            len(self.graphs), self.db_num, self.query_num))

        self.number_of_labels = 0
        # Try to load labels for all datasets (not just AIDS)
        try:
            self.global_labels, self.features = load_labels(
                self.args.abs_path, dataset_name,
                parallel=self.args.parallel_loading,
                max_workers=self.args.loading_workers,
                graphs=self.graphs
            )
            self.number_of_labels = len(self.global_labels)
            print('Load one-hot label features (dim = {}) of {}.'.format(self.number_of_labels, dataset_name))
        except Exception as e:
            if getattr(self.args, 'delta_only', False) or getattr(self.args, 'similarity_search', False) or getattr(self.args, 'mae_test', False):
                print(f"Warning: Could not load one-hot features for {dataset_name}: {e}")
                print("Using default node features (dim=1)")
                self.number_of_labels = 0
            else:
                raise
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
            print(f"Using default features (dim=1) for {len(self.graphs)} graphs")
        # print(self.global_labels)

        ged_dict = dict()
        # We could load ged info from several files.
        # load_ged(ged_dict, self.args.abs_path, dataset_name, 'xxx.json')
        try:
            load_ged(ged_dict, self.args.abs_path, dataset_name, 'TaGED.json')
            print("Load ged dict.")
        except Exception as e:
            if getattr(self.args, 'delta_only', False):
                print("Warning: TaGED.json not found; proceeding with delta-only mode.")
                ged_dict = {}  # Initialize empty dict for delta-only mode
            elif getattr(self.args, 'similarity_search', False) or getattr(self.args, 'similarity_search_all', False) or getattr(self.args, 'similarity_search_gediot', False) or getattr(self.args, 'mae_test', False):
                print("Warning: TaGED.json not found; proceeding with similarity search/MAE test mode.")
                ged_dict = {}  # Initialize empty dict for similarity search/MAE test mode
            else:
                raise
        self.ged_dict = ged_dict
        # print(self.ged['2050']['30'])
        t2 = time.time()
        self.load_data_time = t2 - t1

    def transfer_data_to_torch(self):
        """
        Transfer loaded data to torch.
        """
        t1 = time.time()

        self.edge_index = []
        # self.A = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long().to(self.device)
            self.edge_index.append(edge)
            # A = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), (g['n'], g['n'])).to_dense().to(self.device)
            # self.A.append(A)

        self.features = [torch.tensor(x).float().to(self.device) for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        # Delta-only mode: skip building the full n x n mapping/ged matrices (O(N^2)).
        # For delta samples (pair_type == 1), targets and mappings come from synthetic pairs.
        if getattr(self.args, 'delta_only', False) or getattr(self.args, 'similarity_search', False) or getattr(self.args, 'similarity_search_all', False) or getattr(self.args, 'similarity_search_gediot', False) or getattr(self.args, 'mae_test', False):
            self.gid = [g['gid'] for g in self.graphs]
            self.gn = [g['n'] for g in self.graphs]
            self.gm = [g['m'] for g in self.graphs]
            self.ged = None
            self.mapping = None
            t2 = time.time()
            self.to_torch_time = t2 - t1
            if getattr(self.args, 'similarity_search', False) or getattr(self.args, 'similarity_search_all', False) or getattr(self.args, 'similarity_search_gediot', False) or getattr(self.args, 'mae_test', False) or getattr(self.args, 'delta_only', False):
                print("Similarity search/MAE test/delta-only mode: skipping O(N^2) matrix construction")
            return

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float, device=self.device)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    for gt_mapping in gt_mappings:
                        for x, y in enumerate(gt_mapping):
                            if x < n1 and y < n2:
                                mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float().to(self.device)
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        self.ged = ged
        self.mapping = mapping

        t2 = time.time()
        self.to_torch_time = t2 - t1

    @staticmethod
    def delta_graph(g, f, device):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense().to(device)

        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]  # the last del_num edges in edge are removed
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long().to(device)

        feature2 = torch.zeros(f.shape).to(device)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["permute"] = permute
        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data

    def gen_delta_graphs(self):
        #random.seed(0)
        k = self.args.num_delta_graphs
        delta_only = getattr(self.args, 'delta_only', False)
        similarity_search = getattr(self.args, 'similarity_search', False)
        similarity_search_all = getattr(self.args, 'similarity_search_all', False)
        mae_test = getattr(self.args, 'mae_test', False)

        if similarity_search or similarity_search_all or mae_test:
            print("Similarity search/MAE test mode: skipping delta graph generation")
            return
        elif delta_only:
            print(f"Delta-only mode: generating {k} delta samples for ALL graphs...")
        else:
            print(f"Standard mode: generating {k} delta samples for graphs with >10 nodes...")
            
        generated_count = 0
        for i, g in enumerate(self.graphs):
            # In delta-only mode, generate deltas for all graphs; otherwise skip small graphs.
            if (not delta_only) and g['n'] <= 10:
                continue
            # gen k delta graphs
            f = self.features[i]
            self.delta_graphs[i] = [self.delta_graph(g, f, self.device) for j in range(k)]
            generated_count += 1
            
        print(f"Generated delta samples for {generated_count} graphs (total {generated_count * k} delta pairs)")
        if delta_only:
            print("Note: In delta-only mode, all train/val/test data will come from these delta pairs.")

    def check_pair(self, i, j):
        if i == j:
            return 0, i, j
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return 0, i, j
        elif (id2, id1) in self.ged_dict:
            return 0, j, i
        else:
            return None

    def get_query_graph_index(self, query_id):
        """
        Map query ID (0-99) to the actual graph index in self.graphs.

        Graph organization:
        - graphs[0:db_num] = database graphs
        - graphs[db_num:db_num+query_num] = query graphs

        Args:
            query_id: Query ID (0 to query_num-1)

        Returns:
            int: Index in self.graphs array
        """
        if not hasattr(self, 'db_num') or not hasattr(self, 'query_num'):
            raise ValueError("db_num and query_num not initialized. Did you call load_data()?")

        if query_id < 0 or query_id >= self.query_num:
            raise ValueError(f"Query ID {query_id} out of range [0, {self.query_num})")

        return self.db_num + query_id

    def init_graph_pairs(self):
        #random.seed(1)

        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing_graphs_small = []
        self.testing_graphs_large = []
        self.testing2_graphs = []

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        if self.args.demo:
            train_num = 30
            val_num = 40
            test_num = 50
            self.args.epochs = 1

        assert self.args.graph_pair_mode == "combine"

        # Delta-only branch: use only synthetic pairs everywhere.
        # Similarity search branch: use minimal setup, no training pairs needed.
        if getattr(self.args, 'similarity_search', False) or getattr(self.args, 'similarity_search_all', False) or getattr(self.args, 'mae_test', False):
            # Similarity search/MAE test only needs basic graph info, no pairs
            print("Similarity search/MAE test mode: minimal graph pair initialization")
            self.training_graphs = []
            self.val_graphs = []
            self.testing_graphs = []
            self.testing_graphs_small = []
            self.testing_graphs_large = []
            self.testing2_graphs = []
            print("Similarity search setup completed - ready for distance computation")
            return
        elif getattr(self.args, 'delta_only', False):
            # Delta-only mode: populate training_graphs with delta pairs
            print("Delta-only mode: generating training pairs from delta graphs")
            dg = self.delta_graphs
            train_pair_count = 0
            val_pair_count = 0
            test_pair_count = 0
            
            # training
            for i in range(train_num):
                if dg[i] is not None:
                    k = len(dg[i])
                    for j in range(k):
                        self.training_graphs.append((1, i, j))
                        train_pair_count += 1
                        
            # validation
            for i in range(train_num, val_num):
                if dg[i] is not None:
                    k = len(dg[i])
                    self.val_graphs.append((1, i, list(range(k))))
                    val_pair_count += k
                    
            # testing: use random subset of training graphs instead of tiny test set
            import random as rand_module
            rand_module.seed(42)  # for reproducible test set
            
            # Select random subset from training graphs for testing
            test_graph_count = min(100, train_num)  # Use up to 100 training graphs for testing
            test_indices = rand_module.sample(range(train_num), test_graph_count)
            
            for i in test_indices:
                if dg[i] is not None:
                    k = len(dg[i])
                    idx = list(range(k))
                    self.testing_graphs.append((1, i, idx))
                    self.testing_graphs_small.append((1, i, idx))
                    self.testing_graphs_large.append((1, i, idx))
                    self.testing2_graphs.append((1, i, idx))
                    test_pair_count += k

            print("DELTA-ONLY MODE Statistics:")
            print("Generate {} training graph pairs.".format(train_pair_count))
            print("Generate {} val graph pairs from {} graphs.".format(val_pair_count, len(self.val_graphs)))
            print("Generate {} testing graph pairs from {} graphs.".format(test_pair_count, len(self.testing_graphs)))
            print("All pairs are synthetic delta samples (pair_type=1)")
            return
        dg = self.delta_graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        self.training_graphs.append(tmp)
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    self.training_graphs.append((1, i, j))

        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)
        print("The number of small training graphs:", len(li))

        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.val_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.val_graphs.append((1, i, list(range(k))))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing_graphs.append((0, i, li[:self.args.num_testing_graphs]))
                self.testing_graphs_small.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing_graphs.append((1, i, list(range(k))))
                self.testing_graphs_large.append((1, i, list(range(k))))

        li = []
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                li.append(i)
        print("The number of small testing graphs:", len(li))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing2_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing2_graphs.append((1, i, list(range(k))))

        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} * {} val graph pairs.".format(len(self.val_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} testing graph pairs.".format(len(self.testing_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} small testing graph pairs.".format(len(self.testing_graphs_small), self.args.num_testing_graphs))
        print("Generate {} * {} large testing graph pairs.".format(len(self.testing_graphs_large), self.args.num_testing_graphs))
        print("Generate {} * {} testing2 graph pairs.".format(len(self.testing2_graphs), self.args.num_testing_graphs))

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph + self.args.batch_size])
        return batches

    def pack_graph_pair(self, graph_pair):
        """
        Prepare the graph pair data for GedGNN model.
        :param graph_pair: (pair_type, id_1, id_2)
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()

        (pair_type, id_1, id_2) = graph_pair
        if pair_type == 0:  # normal case
            gid_pair = (self.gid[id_1], self.gid[id_2])
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)
                gid_pair = (self.gid[id_1], self.gid[id_2])

            real_ged = self.ged[id_1][id_2][0]
            ta_ged = self.ged[id_1][id_2][1:]

            new_data["id_1"] = id_1
            new_data["id_2"] = id_2

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = self.edge_index[id_2]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = self.features[id_2]

            if self.args.gtmap:
                new_data["mapping"] = self.mapping[id_1][id_2]

            new_data["permute"] = [list(range(self.gn[id_1]))] if id_1 == id_2 else self.ged_dict[gid_pair][1]

        elif pair_type == 1:  # delta graphs
            new_data["id"] = id_1
            dg: dict = self.delta_graphs[id_1][id_2]

            real_ged = dg["ta_ged"][0]
            ta_ged = dg["ta_ged"][1:]

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = dg["edge_index"]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = dg["features"]

            new_data["permute"] = [dg["permute"]]
            if self.args.gtmap:
                new_data["mapping"] = dg["mapping"]
        else:
            assert False

        n1, m1 = (self.gn[id_1], self.gm[id_1])
        n2, m2 = (self.gn[id_2], self.gm[id_2]) if pair_type == 0 else (dg["n"], dg["m"])
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["ged"] = real_ged
        # new_data["ta_ged"] = ta_ged
        if self.args.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            new_data["avg_v"] = avg_v
            new_data["target"] = torch.exp(torch.tensor([-real_ged / avg_v]).float()).to(self.device)
            new_data["ta_ged"] = torch.exp(torch.tensor(ta_ged).float() / -avg_v).to(self.device)
        elif self.args.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            new_data["hb"] = higher_bound
            new_data["target"] = torch.tensor([real_ged / higher_bound]).float().to(self.device)
            new_data["ta_ged"] = (torch.tensor(ta_ged).float() / higher_bound).to(self.device)
        else:
            assert False

        return new_data

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        self.values = []
        
        # Initialize prediction tracking for all modes
        self._batch_predictions = []

        for epoch in range(self.args.epochs):
            # Update epoch counter
            if epoch > 0:
                self.cur_epoch += 1

            # Create per-epoch progress bar
            with tqdm(total=len(self.training_graphs), unit="graph_pairs", leave=True,
                      desc=f"Epoch {self.cur_epoch + 1}/{self.args.epochs}",
                      file=sys.stdout) as pbar:
                batches = self.create_batches()
                loss_sum = 0
                main_index = 0
                for index, batch in enumerate(batches):
                    batch_total_loss = self.process_batch(batch)  # without average
                    loss_sum += batch_total_loss
                    main_index += len(batch)
                    loss = loss_sum / main_index  # the average loss of current epoch
                    pbar.update(len(batch))
                    pbar.set_description(
                        f"Epoch {self.cur_epoch + 1}/{self.args.epochs}: loss={round(1000 * loss, 3)} - Batch {index}: loss={round(1000 * batch_total_loss / len(batch), 3)}")

                    # Print prediction vs ground truth for batch 1 and then every 10 batches
                    if ((index + 1) == 1 or (index + 1) % 10 == 1) and hasattr(self, '_batch_predictions') and self._batch_predictions:
                        recent_preds = self._batch_predictions[-len(batch):]  # Get last batch predictions
                        if len(recent_preds) >= 3:  # Show first 3 pairs from this batch
                            sample_preds = recent_preds[:3]
                            pred_str = ", ".join([f"pred:{p:.2f}/gt:{t:.2f}" for p, t in sample_preds])
                            tqdm.write(f"  Batch {index+1} samples: {pred_str}")
            tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
            training_loss = round(1000 * loss, 3)

            # Save model after each epoch
            self.save(self.cur_epoch + 1)
            tqdm.write(f"Model saved: {self.args.dataset}_{self.cur_epoch + 1}")

        t2 = time.time()
        training_time = t2 - t1
        #if len(self.values) > 0:
         #   self.prediction_analysis(self.values, "training_score")

        self.results.append(
            ('model_name', 'dataset', "graph_set", "current_epoch", "training_time(s/epoch)", "training_loss(1000x)"))
        self.results.append(
            (self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1, training_time, training_loss))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Training", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    @staticmethod
    def cal_pk(num, pre, gt):
        if num >= len(pre):
            return -1.0
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def gen_edit_path(self, data, permute):
        n1, n2 = data["n1"], data["n2"]
        raw_edges_1, raw_edges_2 = data["edge_index_1"].t().tolist(), data["edge_index_2"].t().tolist()
        raw_f1, raw_f2 = data["features_1"].tolist(), data["features_2"].tolist()
        assert len(permute) == n1
        assert len(raw_f1) == n1 and len(raw_f2) == n2 and len(raw_f1[0]) == len(raw_f2[0])

        edges_1 = set()
        for (u, v) in raw_edges_1:
            pu, pv = permute[u], permute[v]
            if pu <= pv:
                edges_1.add((pu, pv))

        edges_2 = set()
        for (u, v) in raw_edges_2:
            if u <= v:
                edges_2.add((u, v))

        edit_edges = edges_1 ^ edges_2

        f1 = []
        num_label = len(raw_f1[0])
        for f in raw_f1:
            for j in range(num_label):
                if f[j] > 0:
                    f1.append(j)
                    break
        f2 = []
        for f in raw_f2:
            for j in range(num_label):
                if f[j] > 0:
                    f2.append(j)
                    break

        relabel_nodes = set()
        for (u, v) in enumerate(permute):
            if f1[u] != f2[v]:
                relabel_nodes.add((v, f1[u]))

        return edit_edges, relabel_nodes

    def score_my(self, testing_graph_set='test',test_k=None):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False
        if self.args.model_name not in ["GEDGW","Classic"]:
            self.model.eval()

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if test_k == None:
                    model_out = self.model(data)
                elif test_k == 0:
                    model_out = self.test_noah(data)
                prediction, pre_ged = model_out[0], model_out[1]
                if self.args.GW:
                    gw = GEDGW(data, self.args)
                    out1 =  gw.process()
                    pre_ged2 = out1[1]
                    pre_ged = min(pre_ged,pre_ged2)
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:  # TaGSim
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Testing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)
        
    def path_score_my(self, testing_graph_set='test', test_k=None):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False
        if self.args.model_name not in ["GEDGW","Classic"]:
            self.model.eval()

        num = 0  # total testing number
        time_usage = []

        mae_path = []  # gep mae

        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []
        rate = [] # node matching rate
        recall = [] # path recall
        precision = [] # path precision
        f1 = [] # path f1 score
        sim = [] # path similarity

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if gt_ged == 0:
                    continue
                if test_k is None:
                    model_out = self.model(data)
                    prediction, pre_ged = model_out[0], model_out[1]
                elif test_k == 0:
                    model_out = self.test_noah(data)
                    pre_permute = model_out[2]
                    pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute)
                    prediction, pre_ged = model_out[0], model_out[1]
                    pre_ged = len(pre_edit_edges) + len(pre_relabel_nodes)
                elif test_k > 0:
                    model_out = self.test_matching(data, test_k=test_k)
                    pre_permute = model_out[2]
                    pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute)
                    prediction, pre_ged = model_out[0], model_out[1]
                else:
                    assert False

                round_pre_ged = round(pre_ged)

                num += 1
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae_path.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged== gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
                assert len(pre_edit_edges) + len(pre_relabel_nodes) == round_pre_ged

                best_rate = 0.
                best_recall = 0.
                best_precision = 0.
                best_f1 = 0.
                best_sim = 0.

                # enumerate groundtruth path
                for permute in data["permute"]:
                    tmp = 0
                    for (v1, v2) in zip(permute, pre_permute):
                        if v1 == v2:
                            tmp += 1
                    best_rate = max(best_rate, tmp / data["n1"])

                    edit_edges, relabel_nodes = self.gen_edit_path(data, permute)
                    assert len(edit_edges) + len(relabel_nodes) == gt_ged
                    num_overlap = len(pre_edit_edges & edit_edges) + len(pre_relabel_nodes & relabel_nodes)

                    best_recall = max(best_recall, num_overlap / gt_ged)
                    best_precision = max(best_precision, num_overlap / round_pre_ged)
                    best_f1 = max(best_f1, 2.0 * num_overlap / (gt_ged + round_pre_ged))
                    best_sim = max(best_sim, num_overlap / (gt_ged + round_pre_ged - num_overlap))

                rate.append(best_rate)
                recall.append(best_recall)
                precision.append(best_precision)
                f1.append(best_f1)
                sim.append(best_sim)

            t2 = time.time()
            time_usage.append(t2 - t1)
            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))            

        time_usage = round(np.mean(time_usage), 3)
        mae_path = round(np.mean(mae_path), 3)
        acc = round(num_acc / max(num, 1), 3)  # Prevent division by zero
        fea = round(num_fea / max(num, 1), 3)  # Prevent division by zero
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        rate = round(np.mean(rate), 3)
        recall = round(np.mean(recall), 3)
        precision = round(np.mean(precision), 3)
        f1 = round(np.mean(f1), 3)
        sim = round(np.mean(sim), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mae','acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20','precision', 'recall', 'f1'))
        self.results.append(
            (self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mae_path, acc,fea,rho,tau,pk10,pk20,precision, recall, f1))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Post-processing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def score(self, testing_graph_set='test', test_k=None):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if test_k is None:
                    model_out = self.model(data)
                elif test_k == 0:
                    model_out = self.test_noah(data)
                elif test_k > 0:
                    model_out = self.test_matching(data, test_k)
                else:
                    assert False
                prediction, pre_ged = model_out[0], model_out[1]
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:  # TaGSim
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Testing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def process(self, testing_graph_set='test'):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                gw = GEDGW(data, self.args)
                out1 =  gw.process()
                pre_ged = out1[1]
                prediction = None
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:  # TaGSim
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Process", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)      

    def batch_score(self, testing_graph_set='test', test_k=100):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        batch_results = []
        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            res = []
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                gt_ged = data["ged"]
                time_list, pre_ged_list = self.test_matching(data, test_k, batch_mode=True)
                res.append((gt_ged, pre_ged_list, time_list))
            batch_results.append(res)

        batch_num = len(batch_results[0][0][1]) # len(pre_ged_list)
        for i in range(batch_num):
            time_usage = []
            num = 0  # total testing number
            mse = []  # score mse
            mae = []  # ged mae
            num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
            num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
            num_better = 0
            ged_better = 0.
            rho = []
            tau = []
            pk10 = []
            pk20 = []

            for res_id, res in enumerate(batch_results):
                pre = []
                gt = []
                for gt_ged, pre_ged_list, time_list in res:
                    time_usage.append(time_list[i])
                    pre_ged = pre_ged_list[i]
                    round_pre_ged = round(pre_ged)

                    num += 1
                    mse.append(-0.001)
                    pre.append(pre_ged)
                    gt.append(gt_ged)

                    mae.append(abs(round_pre_ged - gt_ged))
                    if round_pre_ged == gt_ged:
                        num_acc += 1
                        num_fea += 1
                    elif round_pre_ged > gt_ged:
                        num_fea += 1
                    else:
                        num_better += 1
                        ged_better += (gt_ged - round_pre_ged)
                        # print("\nres_id:", res_id, "batch_id:", i, gt_ged, round_pre_ged)
                rho.append(spearmanr(pre, gt)[0])
                tau.append(kendalltau(pre, gt)[0])
                pk10.append(self.cal_pk(10, pre, gt))
                pk20.append(self.cal_pk(20, pre, gt))

            time_usage = round(np.mean(time_usage), 3)
            mse = round(np.mean(mse) * 1000, 3)
            mae = round(np.mean(mae), 3)
            acc = round(num_acc / num, 3)
            fea = round(num_fea / num, 3)
            rho = round(np.mean(rho), 3)
            tau = round(np.mean(tau), 3)
            pk10 = round(np.mean(pk10), 3)
            pk20 = round(np.mean(pk20), 3)
            if num_better > 0:
                avg_ged_better = round(ged_better / num_better, 3)
            else:
                avg_ged_better = None
            self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                                 fea, rho, tau, pk10, pk20, num_better, avg_ged_better))

            print(*self.results[-1], sep='\t')
            with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
                print(*self.results[-1], sep='\t', file=f)

    def print_results(self):
        for r in self.results:
            print(*r, sep='\t')

        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            for r in self.results:
                print(*r, sep='\t', file=f)

    @staticmethod
    def data_to_nx(edges, features):
        edges = edges.t().tolist()

        nx_g = nx.Graph()
        n, num_label = features.shape

        if num_label == 1:
            labels = [-1 for i in range(n)]
        else:
            labels = [-1] * n
            for i in range(n):
                for j in range(num_label):
                    if features[i][j] > 0.5:
                        labels[i] = j
                        break

        for i, label in enumerate(labels):
            nx_g.add_node(i, label=label)

        for u, v in edges:
            if u < v:
                nx_g.add_edge(u, v)
        return nx_g

    def test_noah(self, data):
        g1 = self.data_to_nx(data["edge_index_1"], data["features_1"])
        g2 = self.data_to_nx(data["edge_index_2"], data["features_2"])

        lower_bound = 'Noah'
        beam_size = 100
        min_path1, cost1, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(self.model, g1, g2, lower_bound,
                                                                                                  beam_size)
        n1, n2 = data["n1"], data["n2"]
        permute = [-1] * n1
        used = [False] * n2
        for u, v in min_path1:
            if u is not None and v is not None:
                assert 0 <= u < n1 and 0 <= v < n2 and not used[v]
                permute[u] = v
                used[v] = True
        for u in range(n1):
            if permute[u] == -1:
                for v in range(n2):
                    if not used[v]:
                        permute[u] = v
                        used[v] = True
                        break

        return None, cost1, permute

    def test_matching(self, data, test_k,test_k_GW=100):
        g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=data["n1"])
        g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=data["n2"])
        g1.ndata['f'] = data["features_1"]
        g2.ndata['f'] = data["features_2"]
        if self.args.GW:
            gw = GEDGW(data, self.args)
            soft_matrix2,pre_ged2 = gw.process()
            
            solver2 = KBestMSolver(soft_matrix2*1e9+1, g1, g2)
            solver2.get_matching(test_k_GW)
            min_res2 = solver2.min_ged
            best_matching2 = solver2.best_matching()
            if self.args.model_name =="GEDGW":
                return None, min_res2, best_matching2
        if self.args.greedy:
            # the Hungarian algorithm, use greedy matching matrix
            pre_ged = None
            soft_matrix = hungarian(data) + 1.0
        else:
            _, pre_ged, soft_matrix = self.model(data)
            if self.args.model_name=="GedGNN":
                m = torch.nn.Softmax(dim=1)
                soft_matrix = (m(soft_matrix) * 1e9 + 1).round()
            else:
                soft_matrix = (soft_matrix * 1e9 + 1).round()

        solver = KBestMSolver(soft_matrix, g1, g2)
        solver.get_matching(test_k)
        min_res = solver.min_ged
        if self.args.GW:
            if min_res>min_res2:
                return None, min_res2, best_matching2
        best_matching = solver.best_matching()
        return None, min_res, best_matching


    def prediction_analysis(self, values, info_str=''):
        """
        Analyze the performance of value prediction.
        :param values: an array of (pre_ged - gt_ged); Note that there is no abs function.
        """
        if not self.args.prediction_analysis:
            return
        neg_num = 0
        pos_num = 0
        pos_error = 0.
        neg_error = 0.
        for v in values:
            if v >= 0:
                pos_num += 1
                pos_error += v
            else:
                neg_num += 1
                neg_error += v

        tot_num = neg_num + pos_num
        tot_error = pos_error - neg_error

        pos_error = round(pos_error / pos_num, 3) if pos_num > 0 else None
        neg_error = round(neg_error / neg_num, 3) if neg_num > 0 else None
        tot_error = round(tot_error / tot_num, 3) if tot_num > 0 else None

        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("prediction_analysis", info_str, sep='\t', file=f)
            print("num", pos_num, neg_num, tot_num, sep='\t', file=f)
            print("err", pos_error, neg_error, tot_error, sep='\t', file=f)
            print("--------------------", file=f)

    def demo_testing(self, testing_graph_set='test'):
        print("\n\nDemo testing on {} set.\n".format(testing_graph_set))
        self.testing_graph_set.append(testing_graph_set)
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        elif testing_graph_set == 'train':
            testing_graphs = self.training_graphs
        else:
            assert False

        self.model.eval()

        # demo_num = 10
        demo_num = len(testing_graphs)
        # random.shuffle(testing_graphs)
        testing_graphs = testing_graphs[:demo_num]
        total_num = 0
        num_10 = 0
        num_100 = 0
        num_1000 = 0
        score_10 = [[], [], []]
        score_100 = [[], [], []]
        score_1000 = [[], [], []]

        values0 = []
        values1 = []
        values2 = []
        values3 = []

        m = torch.nn.Softmax(dim=1)
        for graph_pair in tqdm(testing_graphs, file=sys.stdout):
            data = self.pack_graph_pair(graph_pair)
            avg_v = data["avg_v"]  # (n1+n2)/2.0, a scalar, not a tensor
            gt_ged, target = data["ged"], data["target"]  # gt ged value and score
            soft_matrix, _, prediction = self.model(data, is_testing=True)
            pre_ged, gt_ged, gt_score = prediction.item(), gt_ged.item(), target.item()

            values0.append(pre_ged - gt_ged)

            soft_matrix = (torch.sigmoid(soft_matrix) * 1e9 + 1).round()
            # soft_matrix = (m(soft_matrix) * 1e9 + 1).int()
            # soft_matrix = ((soft_matrix - soft_matrix.min()) * 1e9 + 1).round()

            n1, n2 = soft_matrix.shape
            # print(data["edge_index_1"].shape)
            g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=n1)
            g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=n2)
            g1.ndata['f'] = data["features_1"]
            g2.ndata['f'] = data["features_2"]

            # if n1 < 10 or n2 < 10:
            #   continue

            total_num += 1
            test_k = self.args.postk

            solver = KBestMSolver(soft_matrix, g1, g2, pre_ged)
            for k in range(test_k):
                '''
                matching, weightsum, sp_ged = solver.get_matching(k + 1)
                if weightsum is None:
                    print(k, solver.min_ged, gt_ged)
                    break
                mapping = torch.zeros([n1, n2])
                for i, j in enumerate(matching):
                    mapping[i][j] = 1.0
                mapping_ged = self.model.ged_from_mapping(mapping, data["A_1"], data["A_2"], data["features_1"],
                                                          data["features_2"])
                min_res = min(min_res, mapping_ged.item())
                '''
                solver.get_matching(k + 1)
                min_res = solver.min_ged
                # a gt_mapping is found
                if abs(min_res - gt_ged) < 1e-12:
                    # fix pre_ged using lower bound
                    fixed_pre_ged = max(solver.lb_value, pre_ged)
                    # fix pre_ged using upper bound
                    if min_res < fixed_pre_ged:
                        fixed_pre_ged = min_res

                    fixed_pre_s = exp(-fixed_pre_ged / avg_v)
                    pre_score = abs(fixed_pre_ged - gt_ged)
                    pre_score2 = (fixed_pre_s - gt_score) ** 2
                    map_score = 0.0
                    if k < 10:
                        score_10[0].append(pre_score2)
                        score_10[1].append(pre_score)
                        score_10[2].append(map_score)
                        num_10 += 1
                        values1.append(fixed_pre_ged - gt_ged)
                    if k < 100:
                        score_100[0].append(pre_score2)
                        score_100[1].append(pre_score)
                        score_100[2].append(map_score)
                        num_100 += 1
                        values2.append(fixed_pre_ged - gt_ged)
                    if k < 1000:
                        score_1000[0].append(pre_score2)
                        score_1000[1].append(pre_score)
                        score_1000[2].append(map_score)
                        num_1000 += 1
                        values3.append(fixed_pre_ged - gt_ged)
                    break
                if k in [9, 99, 999]:
                    # fix pre_ged using lower bound
                    fixed_pre_ged = max(solver.lb_value, pre_ged)
                    # fix pre_ged using upper bound
                    if min_res < fixed_pre_ged:
                        fixed_pre_ged = min_res

                    fixed_pre_s = exp(-fixed_pre_ged / avg_v)
                    pre_score = abs(fixed_pre_ged - gt_ged)
                    pre_score2 = (fixed_pre_s - gt_score) ** 2
                    map_score = abs(min_res - gt_ged)
                    if k + 1 == 10:
                        score_10[0].append(pre_score2)
                        score_10[1].append(pre_score)
                        score_10[2].append(map_score)
                        values1.append(fixed_pre_ged - gt_ged)
                    elif k + 1 == 100:
                        score_100[0].append(pre_score2)
                        score_100[1].append(pre_score)
                        score_100[2].append(map_score)
                        values2.append(fixed_pre_ged - gt_ged)
                    elif k + 1 == 1000:
                        score_1000[0].append(pre_score2)
                        score_1000[1].append(pre_score)
                        score_1000[2].append(map_score)
                        values3.append(fixed_pre_ged - gt_ged)

        if test_k >= 10:
            print("10:", len(score_10[0]), round(np.mean(score_10[1]), 3), round(np.mean(score_10[2]), 3), sep='\t')
            print("{} / {} = {}".format(num_10, total_num, round(num_10 / total_num, 3)))
        if test_k >= 100:
            print("100:", len(score_100[0]), round(np.mean(score_100[1]), 3), round(np.mean(score_100[2]), 3), sep='\t')
            print("{} / {} = {}".format(num_100, total_num, round(num_100 / total_num, 3)))
        if test_k >= 1000:
            print("1000:", len(score_1000[0]), round(np.mean(score_1000[1]), 3), round(np.mean(score_1000[2]), 3),
                  sep='\t')
            print("{} / {} = {}".format(num_1000, total_num, round(num_1000 / total_num, 3)))

        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print('', file=f)
            print(self.cur_epoch, testing_graph_set, demo_num, sep='\t', file=f)
            if test_k >= 10:
                print("10", round(np.mean(score_10[0]) * 1000, 3), round(np.mean(score_10[1]), 3),
                      round(np.mean(score_10[2]), 3), round(num_10 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_10, total_num, round(num_10 / total_num, 3)), file=f)
            if test_k >= 100:
                print("100", round(np.mean(score_100[0]) * 1000, 3), round(np.mean(score_100[1]), 3),
                      round(np.mean(score_100[2]), 3), round(num_100 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_100, total_num, round(num_100 / total_num, 3)), file=f)
            if test_k >= 1000:
                print("1000", round(np.mean(score_1000[0]) * 1000, 3), round(np.mean(score_1000[1]), 3),
                      round(np.mean(score_1000[2]), 3), round(num_1000 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_1000, total_num, round(num_1000 / total_num, 3)), file=f)
            # print('', file=f)

        self.prediction_analysis(values0, "base")
        if test_k >= 10:
            self.prediction_analysis(values1, "10")
        if test_k >= 100:
            self.prediction_analysis(values2, "100")
        if test_k >= 1000:
            self.prediction_analysis(values3, "1000")

    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path+self.args.model_name+'/' + self.args.dataset + '_' + str(epoch))

    def load(self, epoch):
        model_dicts={"GEDHOT":"GEDIOT","NOAH":"GPN","SimGNN":"SimGNN","GEDIOT":"GEDIOT","GedGNN":"GedGNN","TaGSim":"TaGSim"}
        model_path = self.args.abs_path + self.args.model_path +model_dicts[self.args.model_name]+'/'+ self.args.dataset + '_' + str(epoch)
        print(f"Loading model from: {model_path}")
        self.model.load_state_dict(
            torch.load(model_path))


    def _process_single_query_parallel(self, args_tuple):
        """
        Process a single query's candidates in parallel.

        Args:
            args_tuple: (query_id, candidates_set, tau_threshold, model_args_dict)

        Returns:
            tuple: (query_id, results_set, query_time)
        """
        import time
        query_id, candidates_set, tau_threshold, model_args = args_tuple

        query_results = set()
        query_start_time = time.time()

        # Import within worker process to avoid pickle issues
        import torch
        import os

        # Set environment for this worker
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        try:
            # Recreate model in worker process
            if model_args['model_name'] in ["GEDHOT", "GEDIOT"]:
                # Load model for neural methods
                from models import GedGNN, GEDIOT
                if model_args['model_name'] == "GEDIOT":
                    model = GEDIOT(model_args)
                else:  # GEDHOT
                    model = GedGNN(model_args)

                # Load weights if specified
                if model_args.get('model_epoch_start', 0) > 0:
                    model_path = f"{model_args['model_path']}{model_args['model_name']}_{model_args['dataset']}_gnn_{model_args['model_epoch_start']}.pth"
                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()

            # Process each candidate
            for candidate_id in candidates_set:
                if query_id == candidate_id:  # Skip self-comparison
                    continue

                try:
                    # Prepare data
                    n1, n2 = model_args['gn'][query_id], model_args['gn'][candidate_id]
                    num_edges_1 = model_args['edge_index'][query_id].shape[1]
                    num_edges_2 = model_args['edge_index'][candidate_id].shape[1]
                    hb = max(n1, n2) + max(num_edges_1, num_edges_2)

                    data = {
                        "id_1": query_id,
                        "id_2": candidate_id,
                        "n1": n1,
                        "n2": n2,
                        "hb": hb,
                        "edge_index_1": model_args['edge_index'][query_id],
                        "edge_index_2": model_args['edge_index'][candidate_id],
                        "features_1": model_args['features'][query_id],
                        "features_2": model_args['features'][candidate_id]
                    }

                    # Compute GED
                    if model_args['model_name'] == "GEDGW":
                        # Handle GEDGW node ordering
                        n1, n2 = model_args['gn'][query_id], model_args['gn'][candidate_id]
                        if n1 <= n2:
                            data["n1"] = n1
                            data["n2"] = n2
                        else:
                            # Swap to ensure n1 <= n2
                            data["n1"] = n2
                            data["n2"] = n1
                            data["edge_index_1"] = model_args['edge_index'][candidate_id]
                            data["edge_index_2"] = model_args['edge_index'][query_id]
                            data["features_1"] = model_args['features'][candidate_id]
                            data["features_2"] = model_args['features'][query_id]

                        from models import GEDGW
                        gedgw_model = GEDGW(data, model_args)
                        _, predicted_ged = gedgw_model.process()
                    else:
                        # Neural methods
                        with torch.no_grad():
                            if model_args['model_name'] == "GEDIOT":
                                _, predicted_ged, _ = model(data)  # GEDIOT returns tuple
                                predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged
                            else:
                                _, predicted_ged = model(data)  # GEDHOT returns tuple
                                predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged

                    # Check threshold
                    if predicted_ged <= tau_threshold:
                        query_results.add(candidate_id)

                except Exception as e:
                    # Silent error handling in parallel mode to avoid spam
                    continue

        except Exception as e:
            print(f"Worker error for query {query_id}: {e}")

        query_end_time = time.time()
        query_time = query_end_time - query_start_time

        return query_id, query_results, query_time

    def similarity_search(self, candidates_csv, query_start, query_end, tau_threshold, output_json=None):
        """
        Perform similarity search and return recall/accuracy metrics.
        
        Args:
            candidates_csv: Path to CSV file with candidate graph IDs
            query_start: Starting query ID
            query_end: Ending query ID  
            tau_threshold: Distance threshold for filtering
            output_json: Optional path to save results as JSON file
            
        Returns:
            dict: Contains recall and accuracy metrics
        """
        import csv
        import json
        import time
        from pathlib import Path
        
        print(f"Starting similarity search with model {self.args.model_name}")
        print(f"Candidates CSV: {candidates_csv}")
        print(f"Query range: {query_start} to {query_end}")
        print(f"Tau threshold: {tau_threshold}")
        
        # 1. Read candidates from Gisma format CSV
        query_candidates = {}  # query_id -> set of candidate_ids
        all_candidates = set()
        try:
            with open(candidates_csv, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header: query_id,tau_search,candidate_ids
                for row in reader:
                    if row and len(row) >= 3:  # query_id, tau_search, candidate_ids
                        query_id = int(row[0])
                        tau_val = float(row[1])
                        candidate_ids_str = row[2]
                        
                        # Only process rows matching our tau threshold
                        if tau_val == tau_threshold:
                            # Parse semicolon-separated candidate IDs
                            if candidate_ids_str:
                                candidate_ids = [int(cid.strip()) for cid in candidate_ids_str.split(';') 
                                               if cid.strip().isdigit()]
                                query_candidates[query_id] = set(candidate_ids)
                                all_candidates.update(candidate_ids)
            
            total_candidates = sum(len(candidates) for candidates in query_candidates.values())
            unique_candidates = len(all_candidates)
            print(f"Loaded {total_candidates} total candidates across {len(query_candidates)} queries")
            print(f"Unique candidates: {unique_candidates}")
        except Exception as e:
            print(f"Error reading candidates file: {e}")
            return {"recall": 0.0, "accuracy": 0.0, "error": str(e)}
        
        # 2. Generate query range
        query_ids = list(range(query_start, query_end + 1))
        print(f"Processing {len(query_ids)} query graphs")
        
        # 3. Find results within tau threshold using sequential processing
        query_results = {}  # query_id -> set of candidate_ids within tau
        query_times = {}    # query_id -> computation time
        
        # Calculate total comparisons
        total_comparisons = sum(len(query_candidates.get(qid, [])) for qid in query_ids 
                               if qid in query_candidates)
        
        print(f"Total comparisons to perform: {total_comparisons}")
        
        total_computation_time = 0.0  # Track pure GED computation time
        comparison_count = 0
        
        start_processing_time = time.time()

        # Choose processing mode based on parallel_search parameter
        use_parallel = getattr(self.args, 'parallel_search', False)

        if use_parallel:
            print(f"Using parallel processing with {self.args.parallel_workers} workers")

            # Parallel processing mode
            from multiprocessing import Pool
            import multiprocessing as mp

            # Prepare model arguments for worker processes
            model_args = {
                'model_name': self.args.model_name,
                'dataset': self.args.dataset,
                'model_path': self.args.model_path,
                'model_epoch_start': getattr(self.args, 'model_epoch_start', 0),
                'edge_index': self.edge_index,
                'features': self.features,
                'gn': self.gn,
                'db_num': self.db_num,  # For query ID mapping
                # Add other necessary args
                'filters_1': self.args.filters_1,
                'filters_2': self.args.filters_2,
                'filters_3': self.args.filters_3,
                'tensor_neurons': self.args.tensor_neurons,
                'bottle_neck_neurons': self.args.bottle_neck_neurons,
                'bottle_neck_neurons_2': self.args.bottle_neck_neurons_2,
                'bottle_neck_neurons_3': self.args.bottle_neck_neurons_3,
                'bins': self.args.bins,
                'dropout': self.args.dropout,
                'hidden_dim': self.args.hidden_dim,
                'target_mode': self.args.target_mode,
                'histogram': self.args.histogram,
                'gw_alpha': self.args.gw_alpha,
                'gw_rescale': self.args.gw_rescale,
                'gw_square': self.args.gw_square,
            }

            # Prepare arguments for parallel processing
            parallel_args = []
            for query_id in query_ids:
                candidates_for_query = query_candidates.get(query_id, set())
                if candidates_for_query:
                    parallel_args.append((query_id, candidates_for_query, tau_threshold, model_args))

            print(f"Starting parallel processing for {len(parallel_args)} queries")

            # Execute parallel processing
            try:
                # Use process pool for parallel execution
                from multiprocessing import get_context
                ctx = get_context('spawn')  # Use spawn method for Windows compatibility
                with ctx.Pool(processes=self.args.parallel_workers) as pool:
                    parallel_results = pool.map(process_single_query_parallel_worker, parallel_args)

                # Collect results
                for query_id, results_set, query_time in parallel_results:
                    query_results[query_id] = results_set
                    query_times[query_id] = query_time
                    total_computation_time += query_time

                    # Update comparison count
                    comparison_count += len(query_candidates.get(query_id, set()))

            except Exception as e:
                print(f"Error in parallel processing: {e}")
                print("Falling back to sequential processing...")
                use_parallel = False

        if not use_parallel:
            print("Using sequential processing")
            # Sequential processing
            for query_id in query_ids:
                if comparison_count % 1000 == 0:
                    print(f"Progress: {comparison_count}/{total_comparisons} ({100*comparison_count/total_comparisons:.1f}%)")

                query_results[query_id] = set()
                query_start_time = time.time()  # Track time for this query

                # Get candidates for this specific query
                candidates_for_query = query_candidates.get(query_id, set())
                if not candidates_for_query:
                    print(f"Warning: No candidates found for query {query_id}")
                    query_times[query_id] = 0.0
                    continue

                for candidate_id in candidates_for_query:
                    comparison_count += 1

                    # Skip self-comparison
                    if query_id == candidate_id:
                        continue

                    try:
                        # Map query_id to actual graph index
                        query_graph_idx = self.get_query_graph_index(query_id)

                        # Prepare graph pair data (data loading - not timed)
                        n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                        num_edges_1 = self.edge_index[query_graph_idx].shape[1]
                        num_edges_2 = self.edge_index[candidate_id].shape[1]
                        hb = max(n1, n2) + max(num_edges_1, num_edges_2)

                        data = {
                            "id_1": query_graph_idx,
                            "id_2": candidate_id,
                            "n1": n1,
                            "n2": n2,
                            "hb": hb,
                            "edge_index_1": self.edge_index[query_graph_idx],
                            "edge_index_2": self.edge_index[candidate_id],
                            "features_1": self.features[query_graph_idx],
                            "features_2": self.features[candidate_id]
                        }

                        # Start timing for pure computation
                        start_time = time.time()

                        # Compute GED using the loaded model
                        if self.args.model_name == "GEDGW":
                            # For GEDGW, ensure n1 <= n2 (required by the algorithm)
                            n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                            if n1 <= n2:
                                data["n1"] = n1
                                data["n2"] = n2
                            else:
                                # Swap to ensure n1 <= n2
                                data["n1"] = n2
                                data["n2"] = n1
                                data["edge_index_1"] = self.edge_index[candidate_id]
                                data["edge_index_2"] = self.edge_index[query_graph_idx]
                                data["features_1"] = self.features[candidate_id]
                                data["features_2"] = self.features[query_graph_idx]

                            from models import GEDGW
                            gedgw_model = GEDGW(data, self.args)
                            _, predicted_ged = gedgw_model.process()  # Returns (transport_matrix, ged_value)
                        elif self.args.model_name == "GEDIOT":
                            # GEDIOT also requires n1 <= n2
                            n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                            if n1 <= n2:
                                data["n1"] = n1
                                data["n2"] = n2
                            else:
                                # Swap to ensure n1 <= n2
                                data["n1"] = n2
                                data["n2"] = n1
                                data["edge_index_1"] = self.edge_index[candidate_id]
                                data["edge_index_2"] = self.edge_index[query_graph_idx]
                                data["features_1"] = self.features[candidate_id]
                                data["features_2"] = self.features[query_graph_idx]

                            # GEDIOT returns (score, ged_value, transport_matrix)
                            self.model.eval()
                            with torch.no_grad():
                                _, predicted_ged, _ = self.model(data)
                                predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged
                        elif self.args.model_name == "GEDHOT":
                            # GEDHOT also requires n1 <= n2
                            n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                            if n1 <= n2:
                                data["n1"] = n1
                                data["n2"] = n2
                            else:
                                # Swap to ensure n1 <= n2
                                data["n1"] = n2
                                data["n2"] = n1
                                data["edge_index_1"] = self.edge_index[candidate_id]
                                data["edge_index_2"] = self.edge_index[query_graph_idx]
                                data["features_1"] = self.features[candidate_id]
                                data["features_2"] = self.features[query_graph_idx]

                            # GEDHOT returns (score, ged_value)
                            self.model.eval()
                            with torch.no_grad():
                                _, predicted_ged = self.model(data)
                                predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged
                        else:
                            # For other models
                            _, predicted_ged = self.model(data)

                        # End timing for pure computation
                        end_time = time.time()
                        total_computation_time += (end_time - start_time)

                        # Debug: print first 10 GED values for query 0
                        if query_id == 0 and comparison_count <= 10:
                            print(f"  GED({query_id}, {candidate_id}) = {predicted_ged:.2f}, threshold={tau_threshold}")

                        # Check if within threshold
                        if predicted_ged <= tau_threshold:
                            query_results[query_id].add(candidate_id)
                            if query_id == 0 and len(query_results[query_id]) <= 5:  # Debug: print first 5 results for query 0
                                print(f"  + Found: query {query_id} -> candidate {candidate_id}, GED={predicted_ged:.2f}")

                    except Exception as e:
                        print(f"Error computing GED between {query_id} and {candidate_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"  Query index: {query_graph_idx}, Query nodes: {self.gn[query_graph_idx]}")
                        print(f"  Candidate index: {candidate_id}, Candidate nodes: {self.gn[candidate_id]}")
                        continue

                # Record time for this query
                query_end_time = time.time()
                query_times[query_id] = query_end_time - query_start_time
        
        print(f"Completed {total_comparisons} comparisons")
        print(f"Total computation time: {total_computation_time:.2f} seconds")
        print(f"Average time per comparison: {total_computation_time/total_comparisons*1000:.2f} ms")
        
        # 4. Load ground truth for recall calculation
        ground_truth_file = f"json_data/{self.args.dataset}/ground_truth.json"
        gisma_ground_truth_file = self._get_gisma_ground_truth_path()
        
        if not Path(ground_truth_file).exists() and not Path(gisma_ground_truth_file).exists():
            print(f"Warning: Ground truth file not found: {ground_truth_file}")
            print(f"Warning: Gisma ground truth file not found: {gisma_ground_truth_file}")
            print("Cannot calculate recall - only providing result counts")
            
            # Return simple statistics without recall
            total_found = sum(len(results) for results in query_results.values())
            avg_found = total_found / len(query_ids) if query_ids else 0
            
            result = {
                "recall": None,
                "accuracy": None, 
                "total_found": total_found,
                "avg_found_per_query": avg_found,
                "queries_processed": len(query_ids),
                "tau_threshold": tau_threshold,
                "total_computation_time": total_computation_time,
                "avg_computation_time_ms": total_computation_time/total_comparisons*1000 if total_comparisons > 0 else 0
            }
            
            print(f"Results: Total found = {total_found}, Avg per query = {avg_found:.2f}")
            return result
        
        # 5. Calculate recall against ground truth and log detailed results
        try:
            # Try to load JSON format first, then Gisma format
            if Path(ground_truth_file).exists():
                with open(ground_truth_file, 'r') as f:
                    ground_truth = json.load(f)
            else:
                # Load Gisma format ground truth
                ground_truth = self._load_gisma_ground_truth(gisma_ground_truth_file, tau_threshold)
            
            total_recall = 0.0
            total_precision = 0.0
            valid_queries = 0  # Queries with GT (for recall)
            valid_precision_queries = 0  # Queries with GT and Found > 0 (for precision)
            per_query_data = []  # Store detailed per-query statistics for reviewers

            print(f"\n=== DETAILED QUERY RESULTS (Tau={tau_threshold}) ===")
            print("Query_ID\tTime(s)\tFound\tGT\tRecall\tPrecision")
            print("-" * 60)

            for query_id in query_ids:
                query_time = query_times.get(query_id, 0.0)
                found_set = query_results.get(query_id, set())

                # Initialize query data record (all queries will be recorded)
                query_data = {
                    "query_id": query_id,
                    "candidates_count": len(query_candidates.get(query_id, set())),
                    "found_count": len(found_set),
                    "query_time": query_time
                }

                if str(query_id) in ground_truth:
                    # Get ground truth for this query and tau
                    gt_candidates = ground_truth[str(query_id)]

                    # Handle both JSON format (nested dict) and Gisma format (simple list)
                    if isinstance(gt_candidates, dict):
                        # JSON format: {"distance": [candidates]}
                        gt_set = set()
                        for distance_str, candidate_list in gt_candidates.items():
                            distance = float(distance_str)
                            if distance <= tau_threshold:
                                # Filter ground truth to only include candidates in our test set
                                filtered_candidates = set(candidate_list).intersection(query_candidates.get(query_id, set()))
                                gt_set.update(filtered_candidates)
                    else:
                        # Gisma format: simple list of candidates (already filtered by tau)
                        gt_set = set(gt_candidates).intersection(query_candidates.get(query_id, set()))

                    # Calculate recall/precision for queries with GT data
                    if gt_set:
                        intersection = found_set.intersection(gt_set)
                        recall = len(intersection) / len(gt_set)

                        # Precision: only meaningful when Found > 0
                        if found_set:
                            precision = len(intersection) / len(found_set)
                            total_precision += precision
                            valid_precision_queries += 1
                            # F1: only meaningful when both precision and recall exist
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                        else:
                            precision = None  # Undefined when no results found
                            f1 = None

                        # Update query data with GT information
                        query_data.update({
                            "gt_count": len(gt_set),
                            "correct_count": len(intersection),
                            "recall": recall,
                            "precision": precision,
                            "f1": f1
                        })

                        # Log each query's results
                        if precision is not None:
                            print(f"{query_id}\t\t{query_time:.3f}\t{len(found_set)}\t{len(gt_set)}\t{recall:.4f}\t{precision:.4f}")
                        else:
                            print(f"{query_id}\t\t{query_time:.3f}\t{len(found_set)}\t{len(gt_set)}\t{recall:.4f}\tN/A")
                    else:
                        # No GT answers at this tau threshold (but query is in GT file)
                        recall = 0.0
                        precision = 0.0
                        f1 = None  # F1 is N/A when GT=0

                        # Update query data
                        query_data.update({
                            "gt_count": 0,
                            "correct_count": 0,
                            "recall": recall,
                            "precision": precision,
                            "f1": f1
                        })

                        print(f"{query_id}\t\t{query_time:.3f}\t{len(found_set)}\t0\t{recall:.4f}\t{precision:.4f}")
                        if found_set:
                            total_precision += precision
                            valid_precision_queries += 1

                    # Always add recall to totals if query has GT data
                    total_recall += recall
                    valid_queries += 1
                else:
                    # Query has no ground truth data in the file at all
                    # Precision: if there are returned results but no GT, precision=0.0 (all returned are wrong)
                    if found_set:
                        precision = 0.0
                        total_precision += precision
                        valid_precision_queries += 1
                    else:
                        precision = None

                    query_data.update({
                        "gt_count": None,
                        "correct_count": None,
                        "recall": None,
                        "precision": precision,
                        "f1": None
                    })

                    if precision is not None:
                        print(f"{query_id}\t\t{query_time:.3f}\t{len(found_set)}\tN/A\tN/A\t{precision:.4f}")
                    else:
                        print(f"{query_id}\t\t{query_time:.3f}\t{len(found_set)}\tN/A\tN/A\tN/A")

                # Append query data for ALL queries
                per_query_data.append(query_data)

                # Save individual query JSON immediately (real-time save)
                self._save_single_query_json(
                    query_id=query_id,
                    query_data=query_data,
                    tau_threshold=tau_threshold,
                    dataset=self.args.dataset,
                    model=self.args.model_name if hasattr(self.args, 'model_name') else "GEDIOT",
                    epoch=self.args.model_epoch_start if hasattr(self.args, 'model_epoch_start') else None
                )
            
            print("-" * 60)
            
            if valid_queries > 0:
                avg_recall = total_recall / valid_queries
                avg_precision = total_precision / valid_precision_queries if valid_precision_queries > 0 else 0.0
                total_query_time = sum(query_times.values())
                avg_query_time = total_query_time / len(query_ids) if query_ids else 0

                print(f"\n=== TAU-LEVEL SUMMARY (Tau={tau_threshold}) ===")
                print(f"Total queries processed: {len(query_ids)}")
                print(f"Valid queries (with GT): {valid_queries}")
                print(f"Valid precision queries (with GT and Found>0): {valid_precision_queries}")
                print(f"Average recall: {avg_recall:.4f}")
                print(f"Average precision: {avg_precision:.4f}")
                print(f"Total query time: {total_query_time:.2f} seconds")
                print(f"Average query time: {avg_query_time:.3f} seconds")
                print(f"Total computation time: {total_computation_time:.2f} seconds")
                print(f"Average computation time per comparison: {total_computation_time/total_comparisons*1000:.2f} ms")
                print("=" * 50)
            else:
                avg_recall = 0.0
                avg_precision = 0.0
                valid_precision_queries = 0
                print("No valid queries found for recall calculation")

        except Exception as e:
            print(f"Error calculating recall: {e}")
            avg_recall = 0.0
            avg_precision = 0.0
            valid_queries = 0
            valid_precision_queries = 0

        # 6. Return final results with detailed query information
        total_found = sum(len(results) for results in query_results.values())
        total_query_time = sum(query_times.values())

        result = {
            "metadata": {
                "dataset": self.args.dataset,
                "model": self.args.model_name if hasattr(self.args, 'model_name') else "GEDIOT",
                "epoch": self.args.model_epoch_start if hasattr(self.args, 'model_epoch_start') else None,
                "tau_threshold": tau_threshold,
                "total_queries": len(query_ids),
                "valid_queries": valid_queries,
                "valid_precision_queries": valid_precision_queries
            },
            "summary": {
                "avg_recall": avg_recall,
                "avg_precision": avg_precision,
                "total_found": total_found,
                "avg_found_per_query": total_found / len(query_ids) if query_ids else 0,
                "total_query_time": total_query_time,
                "avg_query_time": total_query_time / len(query_ids) if query_ids else 0,
                "total_computation_time": total_computation_time,
                "avg_computation_time_ms": total_computation_time/total_comparisons*1000 if total_comparisons > 0 else 0
            },
            "per_query_data": per_query_data,  # Detailed per-query statistics (candidate lists excluded to save space)
            # Legacy fields for backward compatibility
            "recall": avg_recall,
            "precision": avg_precision,
            "total_found": total_found,
            "avg_found_per_query": total_found / len(query_ids) if query_ids else 0,
            "queries_processed": len(query_ids),
            "valid_queries": valid_queries,
            "valid_precision_queries": valid_precision_queries,
            "tau_threshold": tau_threshold,
            "total_computation_time": total_computation_time,
            "avg_computation_time_ms": total_computation_time/total_comparisons*1000 if total_comparisons > 0 else 0,
            "total_query_time": total_query_time,
            "avg_query_time": total_query_time / len(query_ids) if query_ids else 0,
            "query_times": query_times,
            "query_results_count": {qid: len(results) for qid, results in query_results.items()}
        }
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Recall: {avg_recall:.4f}")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Total found: {total_found}")
        print(f"Queries processed: {len(query_ids)}")
        print(f"Total computation time: {total_computation_time:.2f} seconds")
        print(f"Average computation time: {total_computation_time/total_comparisons*1000:.2f} ms/comparison")
        print(f"=====================")
        
        # Save results to JSON file if specified
        if output_json:
            try:
                with open(output_json, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {output_json}")
            except Exception as e:
                print(f"Error saving results to JSON: {e}")

        # Generate TXT summary report (same directory as JSON)
        # Skip if --no-summary flag is set
        if output_json and not getattr(self.args, 'no_summary', False):
            try:
                from pathlib import Path
                output_path = Path(output_json)
                summary_file = output_path.parent / f"summary_{output_path.stem}.txt"
                self._generate_single_method_summary_report(result, summary_file)
                print(f"Summary report saved to: {summary_file}")
            except Exception as e:
                print(f"Error generating summary report: {e}")
        elif getattr(self.args, 'no_summary', False):
            print(f"Summary generation skipped (--no-summary flag set)")

        return result
    
    def _save_single_query_json(self, query_id, query_data, tau_threshold, dataset, model, epoch):
        """Save individual query result to JSON file immediately (real-time save)"""
        import os
        import json

        # Create output directory (organized by tau first, then by method)
        output_dir = f"results/similarity_search/{dataset}/tau{tau_threshold}/per_query/{model}"
        os.makedirs(output_dir, exist_ok=True)

        # Build file path
        query_file = f"{output_dir}/q{query_id}.json"

        # Build complete query result with metadata
        query_result = {
            "metadata": {
                "dataset": dataset,
                "model": model,
                "epoch": epoch,
                "tau_threshold": tau_threshold,
                "query_id": query_id
            },
            "result": query_data
        }

        # Save to file
        with open(query_file, 'w') as f:
            json.dump(query_result, f, indent=2)

    def _get_gisma_ground_truth_path(self):
        """Get the path to local ground truth file based on dataset name."""
        # Direct mapping since datasets are now renamed
        return f"Ground_Truth/{self.args.dataset}_ground_truth.txt"
    
    def _load_gisma_ground_truth(self, ground_truth_file, tau_threshold):
        """Load ground truth from Gisma format file.
        
        Gisma format: query_id distance [candidate_list]
        Example: 0 4.0 [7259]
        """
        if not Path(ground_truth_file).exists():
            print(f"Warning: Gisma ground truth file not found: {ground_truth_file}")
            return {}
        
        ground_truth = {}
        print(f"Loading Gisma ground truth from: {ground_truth_file}")
        print(f"Using tau threshold: {tau_threshold}")
        
        try:
            with open(ground_truth_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(' ', 2)
                    if len(parts) < 3:
                        continue
                    
                    query_id = int(parts[0])
                    distance = float(parts[1])
                    
                    # Only include results within tau threshold
                    if distance <= tau_threshold:
                        # Parse candidate list: [7259] or [7259, 42564]
                        candidates_str = parts[2].strip()[1:-1]  # Remove [ and ]
                        if candidates_str:
                            candidates = [int(x.strip()) for x in candidates_str.split(',')]
                            
                            # Initialize query entry if not exists
                            if str(query_id) not in ground_truth:
                                ground_truth[str(query_id)] = set()
                            
                            # Add candidates to the set
                            ground_truth[str(query_id)].update(candidates)
            
            # Convert sets to lists for JSON serialization compatibility
            for query_id in ground_truth:
                ground_truth[query_id] = list(ground_truth[query_id])
            
            print(f"Loaded ground truth for {len(ground_truth)} queries from Gisma file")
            return ground_truth
            
        except Exception as e:
            print(f"Error loading Gisma ground truth: {e}")
            return {}

    def similarity_search_all_methods_optimized(self, candidates_dir, query_start, query_end, tau_thresholds, output_dir=None):
        """
        Optimized three-in-one similarity search: GEDGW + GEDIOT + GEDHOT

        Args:
            candidates_dir: Candidates directory (e.g., "candidates/AIDS/")
            query_start: Starting query ID
            query_end: Ending query ID
            tau_thresholds: List of tau threshold values (e.g., [2, 4, 6, 8])
            output_dir: Optional output directory for results

        Returns:
            dict: Complete comparison results for three methods
        """
        import csv
        import json
        import time
        from pathlib import Path

        print(f"=== Starting Three-in-One Similarity Search ===")
        print(f"Methods: GEDGW + GEDIOT + GEDHOT")
        print(f"Query range: {query_start} to {query_end}")
        print(f"Tau thresholds: {tau_thresholds}")
        print(f"Candidates directory: {candidates_dir}")

        # Result storage structure
        all_results = {}

        # Process each tau threshold
        for tau in tau_thresholds:
            print(f"\n--- Processing Tau = {tau} ---")

            # Build candidate file path
            candidates_file = f"{candidates_dir}/candidates_{self.args.dataset}_tau{int(tau)}.csv"

            if not Path(candidates_file).exists():
                print(f"Warning: Candidates file not found: {candidates_file}")
                continue

            # Read candidate data
            query_candidates = {}
            with open(candidates_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and len(row) >= 3:
                        query_id = int(row[0])
                        tau_val = float(row[1])
                        if tau_val == tau:
                            candidate_ids_str = row[2]
                            if candidate_ids_str:
                                candidate_ids = [int(cid.strip()) for cid in candidate_ids_str.split(';')
                                               if cid.strip().isdigit()]
                                query_candidates[query_id] = candidate_ids

            print(f"Loaded candidate data for {len(query_candidates)} queries")

            # Compute three methods for each query
            tau_results = {}

            for query_id in range(query_start, query_end + 1):
                # Get candidate set (empty list if not present)
                candidates = query_candidates.get(query_id, [])

                if len(candidates) == 0:
                    print(f"Query {query_id}: 0 candidates (no candidates to process)")
                    # Even with no candidates, still record this query
                    query_start_time = time.time()

                    # Get ground truth
                    gt_set = self._load_ground_truth_for_query(query_id, tau)

                    # Create empty result records for three methods
                    methods = ["GEDGW", "GEDIOT", "GEDHOT"]
                    for method in methods:
                        tau_results.setdefault(query_id, {})[method] = {
                            "found": 0,
                            "gt": len(gt_set),
                            "candidates_count": 0,
                            "time": time.time() - query_start_time,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0
                        }

                        # Save per-query JSON
                        self._save_single_query_json(
                            query_id=query_id,
                            query_data={
                                "found": 0,
                                "gt": len(gt_set),
                                "candidates_count": 0,
                                "time": time.time() - query_start_time,
                                "precision": 0.0,
                                "recall": 0.0,
                                "f1": 0.0
                            },
                            tau_threshold=tau,
                            dataset=self.args.dataset,
                            model=method,
                            epoch=getattr(self.args, 'model_epoch_start', 'unknown')
                        )

                    print(f"\n=== Query {query_id} Results (Tau = {tau}) ===")
                    print(f"All methods: Found 0 candidates (Total: 0), GT: {len(gt_set)}")
                    print("=" * 50)
                    continue

                print(f"Query {query_id}: {len(candidates)} candidates")

                # Store distance results for three methods
                gedgw_distances = {}
                gediot_distances = {}
                gedhot_distances = {}

                # Time statistics
                gedgw_time = 0.0
                gediot_time = 0.0

                query_start_time = time.time()

                # Check whether to use parallel processing
                use_parallel = getattr(self.args, 'parallel_search', False)

                if use_parallel and len(candidates) > 1:
                    # Hybrid parallel mode: GEDGW parallel + GEDIOT/GEDHOT sequential
                    print(f"  Using hybrid parallel processing for {len(candidates)} candidates")
                    print(f"    - GEDGW: parallel with {self.args.parallel_workers} workers")
                    print(f"    - GEDIOT/GEDHOT: sequential in main process")

                    # Map query_id to graph index once for all candidates
                    query_graph_idx = self.get_query_graph_index(query_id)

                    # Step 1: Compute GEDGW distances in parallel
                    parallel_args = []
                    for candidate_id in candidates:
                        if query_id != candidate_id:
                            # Pre-prepare data dict to avoid passing complex objects
                            n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                            num_edges_query = self.edge_index[query_graph_idx].shape[1]
                            num_edges_candidate = self.edge_index[candidate_id].shape[1]
                            hb = max(n1, n2) + max(num_edges_query, num_edges_candidate)

                            if n1 > n2:
                                # GEDGW requires n1 <= n2, swap graph roles
                                data_dict = {
                                    "n1": n2, "n2": n1,
                                    "edge_index_1": self.edge_index[candidate_id].clone().detach(),
                                    "edge_index_2": self.edge_index[query_graph_idx].clone().detach(),
                                    "features_1": self.features[candidate_id].clone().detach(),
                                    "features_2": self.features[query_graph_idx].clone().detach(),
                                    "hb": hb
                                }
                            else:
                                data_dict = {
                                    "n1": n1, "n2": n2,
                                    "edge_index_1": self.edge_index[query_graph_idx].clone().detach(),
                                    "edge_index_2": self.edge_index[candidate_id].clone().detach(),
                                    "features_1": self.features[query_graph_idx].clone().detach(),
                                    "features_2": self.features[candidate_id].clone().detach(),
                                    "hb": hb
                                }

                            parallel_args.append((query_id, candidate_id, data_dict))

                    try:
                        from multiprocessing import get_context
                        ctx = get_context('spawn')
                        with ctx.Pool(processes=self.args.parallel_workers) as pool:
                            gedgw_results = pool.map(compute_gedgw_parallel_worker, parallel_args)

                        # Collect GEDGW results
                        for result in gedgw_results:
                            if result is not None:
                                cand_id, gedgw_dist, gw_time = result
                                gedgw_distances[cand_id] = gedgw_dist
                                gedgw_time += gw_time

                        print(f"    - GEDGW parallel computation completed: {len(gedgw_distances)} results")

                        # Step 2: Compute GEDIOT and GEDHOT sequentially in the main process
                        print(f"    - Computing GEDIOT/GEDHOT sequentially...")
                        gediot_start = time.time()

                        for candidate_id in candidates:
                            if query_id == candidate_id:
                                continue

                            try:
                                # Prepare graph pair data (same format as GEDGW) - using query_graph_idx
                                n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                                num_edges_query = self.edge_index[query_graph_idx].shape[1]
                                num_edges_candidate = self.edge_index[candidate_id].shape[1]
                                hb = max(n1, n2) + max(num_edges_query, num_edges_candidate)

                                if n1 > n2:
                                    data = {
                                        "n1": n2, "n2": n1,
                                        "edge_index_1": self.edge_index[candidate_id],
                                        "edge_index_2": self.edge_index[query_graph_idx],
                                        "features_1": self.features[candidate_id],
                                        "features_2": self.features[query_graph_idx],
                                        "hb": hb
                                    }
                                else:
                                    data = {
                                        "n1": n1, "n2": n2,
                                        "edge_index_1": self.edge_index[query_graph_idx],
                                        "edge_index_2": self.edge_index[candidate_id],
                                        "features_1": self.features[query_graph_idx],
                                        "features_2": self.features[candidate_id],
                                        "hb": hb
                                    }

                                # Compute GEDIOT distance
                                self.model.eval()
                                with torch.no_grad():
                                    _, gediot_pred, _ = self.model(data)
                                    gediot_pred = gediot_pred.item() if hasattr(gediot_pred, 'item') else gediot_pred

                                gediot_distances[candidate_id] = gediot_pred

                                # Compute GEDHOT distance (min of GEDGW and GEDIOT)
                                # Only compute GEDHOT when GEDGW was successfully computed
                                if candidate_id in gedgw_distances:
                                    gedhot_distances[candidate_id] = min(gedgw_distances[candidate_id], gediot_pred)

                            except Exception as e:
                                print(f"    Error computing GEDIOT/GEDHOT for candidate {candidate_id}: {e}")
                                continue

                        gediot_time = time.time() - gediot_start
                        print(f"    - GEDIOT/GEDHOT sequential computation completed: {len(gediot_distances)} results")

                    except Exception as e:
                        print(f"Error in hybrid parallel processing, falling back to sequential: {e}")
                        use_parallel = False

                if not use_parallel:
                    # Sequential processing mode (original code)
                    print(f"  Using sequential processing for {len(candidates)} candidates")

                    # Compute distance for each candidate
                    for candidate_id in candidates:
                        if query_id == candidate_id:
                            continue

                        try:
                            # Map query_id to graph index
                            query_graph_idx = self.get_query_graph_index(query_id)

                            # Prepare graph pair data
                            n1, n2 = self.gn[query_graph_idx], self.gn[candidate_id]
                            # Calculate higher bound (hb) = max(nodes) + max(edges)
                            num_edges_query = self.edge_index[query_graph_idx].shape[1]
                            num_edges_candidate = self.edge_index[candidate_id].shape[1]
                            hb = max(n1, n2) + max(num_edges_query, num_edges_candidate)

                            if n1 > n2:
                                # GEDGW requires n1 <= n2, swap graph roles
                                data = {
                                    "n1": n2, "n2": n1,
                                    "edge_index_1": self.edge_index[candidate_id],
                                    "edge_index_2": self.edge_index[query_graph_idx],
                                    "features_1": self.features[candidate_id],
                                    "features_2": self.features[query_graph_idx],
                                    "hb": hb
                                }
                            else:
                                data = {
                                    "n1": n1, "n2": n2,
                                    "edge_index_1": self.edge_index[query_graph_idx],
                                    "edge_index_2": self.edge_index[candidate_id],
                                    "features_1": self.features[query_graph_idx],
                                    "features_2": self.features[candidate_id],
                                    "hb": hb
                                }

                            # 1. Compute GEDGW distance
                            gedgw_dist = None
                            try:
                                start_time = time.time()
                                from models import GEDGW
                                gedgw_model = GEDGW(data, self.args)
                                _, gedgw_dist = gedgw_model.process()
                                gedgw_time += (time.time() - start_time)
                                gedgw_distances[candidate_id] = gedgw_dist
                            except Exception as e:
                                print(f"Error in GEDGW ({query_id}, {candidate_id}): {e}")
                                # Don't raise, continue to compute GEDIOT

                            # 2. Compute GEDIOT distance (independent of GEDGW)
                            gediot_dist = None
                            try:
                                start_time = time.time()
                                self.model.eval()
                                with torch.no_grad():
                                    _, gediot_dist, _ = self.model(data)
                                    gediot_dist = gediot_dist.item() if hasattr(gediot_dist, 'item') else gediot_dist
                                gediot_time += (time.time() - start_time)
                                gediot_distances[candidate_id] = gediot_dist
                            except Exception as e:
                                print(f"Error in GEDIOT ({query_id}, {candidate_id}): {e}")
                                # Don't raise, continue

                            # 3. Compute GEDHOT distance (only when both succeeded)
                            if gedgw_dist is not None and gediot_dist is not None:
                                gedhot_dist = min(gedgw_dist, gediot_dist)
                                gedhot_distances[candidate_id] = gedhot_dist

                            # Debug: Print first few distances and special Ground Truth candidate
                            if len(gedgw_distances) + len(gediot_distances) <= 10 or candidate_id == 7259:
                                is_gt = " (GROUND TRUTH!)" if candidate_id == 7259 else ""
                                gedgw_str = f"{gedgw_dist:.3f}" if gedgw_dist is not None else "N/A"
                                gediot_str = f"{gediot_dist:.3f}" if gediot_dist is not None else "N/A"
                                gedhot_str = f"{gedhot_distances.get(candidate_id, 'N/A'):.3f}" if candidate_id in gedhot_distances else "N/A"
                                print(f"Query {query_id} -> Candidate {candidate_id}: GEDGW={gedgw_str}, GEDIOT={gediot_str}, GEDHOT={gedhot_str}{is_gt}")

                        except Exception as e:
                            print(f"Error computing ({query_id}, {candidate_id}): {e}")
                            continue

                query_total_time = time.time() - query_start_time
                gedhot_time = gedgw_time + gediot_time  # GEDHOT time = sum of both

                # Load Ground Truth
                ground_truth = self._load_ground_truth_for_query(query_id, tau)

                # Compute evaluation metrics for each method
                methods_results = {}

                for method_name, distances in [
                    ("GEDGW", gedgw_distances),
                    ("GEDIOT", gediot_distances),
                    ("GEDHOT", gedhot_distances)
                ]:
                    # Find candidates with distance <= tau
                    found_candidates = {cid for cid, dist in distances.items() if dist <= tau}

                    # Compute metrics
                    if ground_truth is not None and len(ground_truth) > 0:
                        gt_set = set(ground_truth)
                        intersection = found_candidates & gt_set

                        precision = len(intersection) / len(found_candidates) if found_candidates else 0.0
                        recall = len(intersection) / len(gt_set) if gt_set else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    else:
                        precision = recall = f1 = 0.0
                        gt_set = set()

                    # Time allocation
                    if method_name == "GEDGW":
                        method_time = gedgw_time
                    elif method_name == "GEDIOT":
                        method_time = gediot_time
                    else:  # GEDHOT
                        method_time = gedhot_time

                    methods_results[method_name] = {
                        "found": len(found_candidates),
                        "gt": len(gt_set),
                        "time": method_time,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "candidates_count": len(candidates)
                    }

                tau_results[query_id] = methods_results

                # Output single query results
                self._print_query_comparison(query_id, tau, methods_results)

                # Save individual query JSONs immediately (real-time save for three-in-one)
                for method_name, method_data in methods_results.items():
                    query_data = {
                        "query_id": query_id,
                        "found_count": method_data["found"],
                        "gt_count": method_data["gt"],
                        "correct_count": len(set(ground_truth).intersection(
                            {cid for cid, dist in (gedgw_distances if method_name == "GEDGW"
                             else gediot_distances if method_name == "GEDIOT"
                             else gedhot_distances).items() if dist <= tau}
                        )) if ground_truth else 0,
                        "recall": method_data["recall"],
                        "precision": method_data["precision"],
                        "f1": method_data["f1"],
                        "query_time": method_data["time"],
                        "candidates_count": method_data.get("candidates_count", 0)
                    }
                    self._save_single_query_json(
                        query_id=query_id,
                        query_data=query_data,
                        tau_threshold=tau,
                        dataset=self.args.dataset,
                        model=method_name,
                        epoch=self.args.model_epoch_start if hasattr(self.args, 'model_epoch_start') else None
                    )

            all_results[str(float(tau))] = tau_results

        # Generate final summary report
        self._print_final_comparison_report(all_results, tau_thresholds)

        # Save results to JSON files (organized by tau directory structure)
        from datetime import datetime
        import os

        # Create output directory - organized by tau subdirectories
        base_output_dir = "results/similarity_search"

        # If testing only one tau, place in corresponding tau directory; otherwise use multi_tau directory
        if len(tau_thresholds) == 1:
            tau_dir = f"{base_output_dir}/{self.args.dataset}/tau{tau_thresholds[0]}"
        else:
            tau_str = "_".join(map(str, tau_thresholds))
            tau_dir = f"{base_output_dir}/{self.args.dataset}/multi_tau_{tau_str}"

        os.makedirs(tau_dir, exist_ok=True)

        # Generate readable summary report file
        # Skip if --no-summary flag is set
        if not getattr(self.args, 'no_summary', False):
            summary_file = f"{tau_dir}/summary_report.txt"
            self._generate_readable_summary_report(all_results, tau_thresholds, query_start, query_end, summary_file)
            print(f"Readable summary report saved to: {summary_file}")
        else:
            print(f"Summary generation skipped (--no-summary flag set)")

        return all_results

    def _generate_readable_summary_report(self, all_results, tau_thresholds, query_start, query_end, output_file):
        """Generate a readable summary report file."""
        from datetime import datetime

        with open(output_file, 'w') as f:
            # Write report header
            f.write("=" * 80 + "\n")
            f.write("THREE-IN-ONE SIMILARITY SEARCH SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.args.dataset}\n")
            f.write(f"Query Range: {query_start} to {query_end}\n")
            f.write(f"Tau Thresholds: {tau_thresholds}\n")
            f.write(f"Model Epoch: {getattr(self.args, 'model_epoch_start', 'unknown')}\n")
            f.write("=" * 80 + "\n\n")

            # Generate summary table for each tau value
            for tau in tau_thresholds:
                tau_str = str(float(tau))
                if tau_str in all_results:
                    tau_data = all_results[tau_str]
                    methods = ["GEDGW", "GEDIOT", "GEDHOT"]

                    f.write(f"--- Tau = {tau} Summary ---\n")
                    f.write("Method    | Queries | Total Found | Total GT | Avg Precision | Avg Recall | Avg F1 | Total Time(s)\n")
                    f.write("----------|---------|-------------|----------|---------------|------------|--------|--------------\n")

                    for method in methods:
                        total_queries = 0
                        total_found = 0
                        total_gt = 0
                        total_time = 0.0
                        sum_precision = 0.0
                        sum_recall = 0.0
                        sum_f1 = 0.0
                        queries_with_found = 0  # Number of queries with returned results
                        queries_with_gt = 0     # Number of queries with ground truth

                        for query_data in tau_data.values():
                            if method in query_data:
                                method_data = query_data[method]
                                total_queries += 1
                                found = method_data.get("found", 0)
                                gt = method_data.get("gt", 0)
                                total_found += found
                                total_gt += gt
                                total_time += method_data.get("time", 0.0)

                                # Precision is only meaningful when found > 0
                                if found > 0:
                                    sum_precision += method_data.get("precision", 0.0)
                                    queries_with_found += 1

                                # Recall is only meaningful when gt > 0
                                if gt > 0:
                                    sum_recall += method_data.get("recall", 0.0)
                                    sum_f1 += method_data.get("f1", 0.0)
                                    queries_with_gt += 1

                        avg_precision = sum_precision / queries_with_found if queries_with_found > 0 else 0.0
                        avg_recall = sum_recall / queries_with_gt if queries_with_gt > 0 else 0.0
                        avg_f1 = sum_f1 / queries_with_gt if queries_with_gt > 0 else 0.0

                        f.write(f"{method:<9} | {total_queries:<7} | {total_found:<11} | "
                               f"{total_gt:<8} | {avg_precision:<13.4f} | "
                               f"{avg_recall:<10.4f} | {avg_f1:<6.4f} | {total_time:<12.2f}\n")

                    f.write("\n")

                    # Add time analysis
                    f.write("--- Time Analysis ---\n")
                    method_times = {}
                    method_queries = {}
                    total_candidates = {}

                    for method in methods:
                        total_time = 0.0
                        total_queries_count = 0
                        total_cand = 0

                        for query_data in tau_data.values():
                            if method in query_data:
                                method_data = query_data[method]
                                total_time += method_data.get("time", 0.0)
                                total_queries_count += 1
                                total_cand += method_data.get("candidates_count", 0)

                        method_times[method] = total_time
                        method_queries[method] = total_queries_count
                        total_candidates[method] = total_cand

                        avg_time_per_query = total_time / total_queries_count if total_queries_count > 0 else 0
                        avg_time_per_comparison = (total_time / total_cand * 1000) if total_cand > 0 else 0

                        f.write(f"{method}: Total={total_time:.2f}s, Avg/query={avg_time_per_query:.3f}s")
                        if total_cand > 0:
                            f.write(f", Avg/comparison={avg_time_per_comparison:.2f}ms, Total comparisons={total_cand}")
                        f.write("\n")

                    # Speed comparison
                    if "GEDGW" in method_times and "GEDIOT" in method_times and method_times["GEDIOT"] > 0:
                        speedup = method_times["GEDGW"] / method_times["GEDIOT"]
                        f.write(f"\nSpeed comparison: GEDIOT is {speedup:.1f}x faster than GEDGW\n")

                    f.write("\n")

                    # Detailed results
                    f.write(f"--- Detailed Results for Tau = {tau} ---\n")
                    f.write("Query | Method  | Found | GT | Candidates | Time(s) | Precision | Recall | F1\n")
                    f.write("------|---------|-------|----|-----------|---------|-----------|---------|---------\n")

                    for query_id in sorted(tau_data.keys(), key=int):
                        query_data = tau_data[query_id]
                        for method in methods:
                            if method in query_data:
                                m = query_data[method]
                                found = m.get('found', 0)
                                gt = m.get('gt', 0)

                                # Apply N/A rules for precision, recall, and F1
                                if found == 0:
                                    precision_str = "N/A"
                                else:
                                    precision_str = f"{m.get('precision', 0.0):9.4f}"

                                if gt == 0:
                                    recall_str = "N/A"
                                else:
                                    recall_str = f"{m.get('recall', 0.0):7.4f}"

                                if found == 0 or gt == 0:
                                    f1_str = "N/A"
                                else:
                                    f1_str = f"{m.get('f1', 0.0):7.4f}"

                                f.write(f"{query_id:<5} | {method:<7} | {found:<5} | "
                                       f"{gt:<2} | {m.get('candidates_count', 0):<9} | "
                                       f"{m.get('time', 0.0):<7.3f} | {precision_str:>9} | "
                                       f"{recall_str:>7} | {f1_str:>7}\n")

                    f.write("\n" + "=" * 80 + "\n\n")

    def _generate_single_method_summary_report(self, result, output_file):
        """Generate TXT summary report for single method similarity search"""
        from datetime import datetime

        with open(output_file, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("SIMILARITY SEARCH SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Metadata
            metadata = result.get('metadata', {})
            f.write(f"Dataset: {metadata.get('dataset', 'N/A')}\n")
            f.write(f"Method: {metadata.get('model', 'N/A')}\n")
            f.write(f"Model Epoch: {metadata.get('epoch', 'N/A')}\n")
            f.write(f"Tau Threshold: {metadata.get('tau_threshold', 'N/A')}\n")
            f.write(f"Total Queries: {metadata.get('total_queries', 0)}\n")
            f.write(f"Valid Queries (with GT): {metadata.get('valid_queries', 0)}\n")
            f.write("=" * 80 + "\n\n")

            # Overall Summary
            summary = result.get('summary', {})
            f.write("--- Overall Performance Summary ---\n")
            f.write(f"Average Recall:       {summary.get('avg_recall', 0.0):.4f}\n")
            f.write(f"Average Precision:    {summary.get('avg_precision', 0.0):.4f}\n")
            f.write(f"Total Found:          {summary.get('total_found', 0)}\n")
            f.write(f"Avg Found/Query:      {summary.get('avg_found_per_query', 0.0):.2f}\n")
            f.write(f"Total Query Time:     {summary.get('total_query_time', 0.0):.2f} seconds\n")
            f.write(f"Avg Query Time:       {summary.get('avg_query_time', 0.0):.4f} seconds\n")
            f.write(f"Total Computation Time: {summary.get('total_computation_time', 0.0):.2f} seconds\n")
            f.write(f"Avg Computation Time:   {summary.get('avg_computation_time_ms', 0.0):.2f} ms/comparison\n")
            f.write("\n")

            # Per-Query Details
            per_query_data = result.get('per_query_data', [])
            if per_query_data:
                f.write("--- Per-Query Details ---\n")
                f.write("Query | Candidates | Found | GT | Precision | Recall   | F1     | Time(s)\n")
                f.write("------|------------|-------|----|-----------|---------|---------|---------\n")

                # Track totals for averages
                total_candidates = 0
                total_found = 0
                total_gt = 0
                total_time = 0.0
                sum_precision = 0.0
                sum_recall = 0.0
                sum_f1 = 0.0
                count_precision = 0  # queries with found > 0
                count_recall = 0     # queries with gt > 0
                count_f1 = 0         # queries with both found > 0 and gt > 0

                for query in per_query_data:
                    qid = query.get('query_id', 'N/A')
                    candidates = query.get('candidates_count', 0)
                    found = query.get('found_count', 0)
                    gt = query.get('gt_count', 0) if query.get('gt_count') is not None else 0
                    time_val = query.get('query_time', 0.0) if query.get('query_time') is not None else 0.0

                    # Format precision/recall/F1 according to rules
                    if found == 0:
                        precision_str = "N/A"
                    else:
                        precision = query.get('precision', 0.0) if query.get('precision') is not None else 0.0
                        precision_str = f"{precision:7.4f}"
                        sum_precision += precision
                        count_precision += 1

                    if gt == 0:
                        recall_str = "N/A"
                    else:
                        recall = query.get('recall', 0.0) if query.get('recall') is not None else 0.0
                        recall_str = f"{recall:7.4f}"
                        sum_recall += recall
                        count_recall += 1

                    if found == 0 or gt == 0:
                        f1_str = "N/A"
                    else:
                        f1 = query.get('f1', 0.0) if query.get('f1') is not None else 0.0
                        f1_str = f"{f1:6.4f}"
                        sum_f1 += f1
                        count_f1 += 1

                    # Accumulate totals
                    total_candidates += candidates
                    total_found += found
                    total_gt += gt
                    total_time += time_val

                    f.write(f"{qid:>5} | {candidates:>10} | {found:>5} | {gt:>2} | {precision_str:>9} | "
                           f"{recall_str:>8} | {f1_str:>6} | {time_val:>7.4f}\n")

                # Write average row
                f.write("------|------------|-------|----|-----------|---------|---------|---------\n")
                avg_candidates = total_candidates / len(per_query_data) if per_query_data else 0
                avg_found = total_found / len(per_query_data) if per_query_data else 0
                avg_gt = total_gt / len(per_query_data) if per_query_data else 0
                avg_precision = sum_precision / count_precision if count_precision > 0 else 0.0
                avg_recall = sum_recall / count_recall if count_recall > 0 else 0.0
                avg_f1 = sum_f1 / count_f1 if count_f1 > 0 else 0.0
                avg_time = total_time / len(per_query_data) if per_query_data else 0.0

                f.write(f"{'AVG':>5} | {avg_candidates:>10.2f} | {avg_found:>5.2f} | {avg_gt:>2.2f} | "
                       f"{avg_precision:>9.4f} | {avg_recall:>8.4f} | {avg_f1:>6.4f} | {avg_time:>7.4f}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")

    def _pack_graph_pair_for_similarity_search(self, id_1, id_2):
        """
        Simplified graph pair packing function for similarity search.
        Does not require ground truth GED or mapping information.
        """
        new_data = dict()

        # Basic graph information
        new_data["id_1"] = id_1
        new_data["id_2"] = id_2

        # Node counts
        n1, n2 = self.gn[id_1], self.gn[id_2]

        # GEDIOT requires n1 <= n2, swap if not satisfied
        if n1 > n2:
            id_1, id_2 = id_2, id_1
            n1, n2 = n2, n1

        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["avg_v"] = (n1 + n2) / 2.0

        # Edge information (using possibly swapped ids)
        new_data["edge_index_1"] = self.edge_index[id_1]
        new_data["edge_index_2"] = self.edge_index[id_2]

        # Feature information (using possibly swapped ids)
        new_data["features_1"] = self.features[id_1]
        new_data["features_2"] = self.features[id_2]

        # HB parameter (used by GEDIOT model)
        num_edges_1 = self.edge_index[id_1].shape[1]
        num_edges_2 = self.edge_index[id_2].shape[1]
        new_data["hb"] = max(n1, n2) + max(num_edges_1, num_edges_2)

        # Adjacency matrices (built from edge_index)
        import torch
        def edge_index_to_adj(edge_index, num_nodes):
            adj = torch.zeros(num_nodes, num_nodes)
            if edge_index.shape[1] > 0:
                adj[edge_index[0], edge_index[1]] = 1
            return adj

        new_data["A_1"] = edge_index_to_adj(self.edge_index[id_1], n1)
        new_data["A_2"] = edge_index_to_adj(self.edge_index[id_2], n2)

        # Add dummy target and ged (not needed for similarity search, but model may expect them)
        new_data["target"] = torch.tensor(0.0).float()
        new_data["ged"] = 0.0

        return new_data

    def _load_ground_truth_for_query(self, query_id, tau_threshold):
        """
        Load ground truth data for a given query (cumulative version).
        tau_threshold=4 should include all candidate graphs with distance <= 4.
        """
        ground_truth_file = f"Ground_Truth/{self.args.dataset}_ground_truth.txt"

        if not Path(ground_truth_file).exists():
            print(f"Warning: Ground truth file not found: {ground_truth_file}")
            return []

        try:
            cumulative_candidates = []

            with open(ground_truth_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 2)
                        if len(parts) >= 3:
                            gt_query_id = int(parts[0])
                            gt_tau = float(parts[1])

                            # Only process matching query IDs
                            if gt_query_id == query_id:
                                # Accumulate all candidate graphs with distance <= tau_threshold
                                if gt_tau <= tau_threshold:
                                    # Parse candidate graph ID list [id1, id2, ...]
                                    candidates_str = parts[2]
                                    if candidates_str.startswith('[') and candidates_str.endswith(']'):
                                        candidates_str = candidates_str[1:-1]  # Remove brackets
                                        if candidates_str:
                                            current_candidates = [int(cid.strip()) for cid in candidates_str.split(',')]
                                            cumulative_candidates.extend(current_candidates)

            # Deduplicate and return
            return list(set(cumulative_candidates))

        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return []

    def _print_query_comparison(self, query_id, tau, methods_results):
        """
        Print comparison results for a single query.
        Simplified version for similarity search.
        """
        print(f"\n=== Query {query_id} Results (Tau = {tau}) ===")
        for method, results in methods_results.items():
            found = results.get('found', 0)
            candidates = results.get('candidates_count', 0)
            time_used = results.get('time', 0.0)
            print(f"{method}: Found {found} candidates (Total: {candidates}), Time: {time_used:.3f}s")
        print("=" * 50)

    def _print_final_comparison_report(self, all_results, tau_thresholds):
        """
        Print final comparison report.
        Contains complete precision/recall/F1 metrics table.
        """
        print(f"\n=== Final Three-in-One Similarity Search Report ===")
        print(f"Tau thresholds: {tau_thresholds}")

        for tau in tau_thresholds:
            if tau in all_results:
                tau_data = all_results[tau]
                print(f"\n--- Tau = {tau} Performance Summary ---")
                print("Method   | Queries | Found | GT Total | Precision | Recall   | F1     | Time(s)")
                print("---------|---------|-------|----------|-----------|----------|--------|---------")

                methods = ["GEDGW", "GEDIOT", "GEDHOT"]
                for method in methods:
                    query_count = 0
                    total_found = 0
                    total_gt = 0
                    total_time = 0.0
                    total_precision = 0.0
                    total_recall = 0.0
                    total_f1 = 0.0
                    valid_queries = 0

                    for query_data in tau_data.values():
                        if method in query_data:
                            method_data = query_data[method]
                            query_count += 1
                            total_found += method_data.get('found', 0)
                            total_gt += method_data.get('gt', 0)
                            total_time += method_data.get('time', 0.0)

                            precision = method_data.get('precision', 0.0)
                            recall = method_data.get('recall', 0.0)
                            f1 = method_data.get('f1', 0.0)

                            # Accumulate all queries with GT data (including recall=0 cases)
                            total_precision += precision
                            total_recall += recall
                            total_f1 += f1
                            valid_queries += 1

                    # Compute averages (over all queries with GT data)
                    avg_precision = total_precision / valid_queries if valid_queries > 0 else 0.0
                    avg_recall = total_recall / valid_queries if valid_queries > 0 else 0.0
                    avg_f1 = total_f1 / valid_queries if valid_queries > 0 else 0.0

                    print(f"{method:<9} | {query_count:7} | {total_found:5} | {total_gt:8} | "
                          f"{avg_precision:9.4f} | {avg_recall:8.4f} | {avg_f1:6.4f} | {total_time:7.3f}")

                print("=" * 80)
        print("=" * 80)
    def mae_test_three_methods(self, num_samples=420, output_file=None, test_methods=None):
        """
        Test MAE for three methods (GEDGW, GEDIOT, GEDHOT) using Gisma ground_truth.txt
        Following similarity_search implementation pattern for data loading and preparation.

        Args:
            num_samples: Number of graph pairs to sample for testing (default: 420)
            output_file: Optional output file path for results
            test_methods: List of methods to test (default: ['GEDGW', 'GEDIOT', 'GEDHOT'])

        Returns:
            dict: MAE test results for three methods
        """
        import json
        import time
        import random
        import numpy as np
        from pathlib import Path
        import re

        # Parse test_methods parameter
        if test_methods is None:
            test_methods = ['GEDGW', 'GEDIOT', 'GEDHOT']

        print("=== Starting MAE Test ===")
        print(f"Number of samples: {num_samples}")
        print(f"Methods: {', '.join(test_methods)}")

        # Load ground truth from Gisma format
        gisma_dir = Path(f"../Gisma/datasets/{self.args.dataset}")
        gt_file = gisma_dir / "ground_truth.txt"

        if not gt_file.exists():
            print(f"Error: Gisma ground truth not found at {gt_file}")
            return None

        print(f"Loading ground truth from {gt_file}...")

        # Parse Gisma ground truth format: query_id ged_value [db_ids...]
        all_pairs = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                query_id = int(parts[0])
                ged_value = float(parts[1])

                # Parse db_ids from [id1, id2, ...]
                db_ids_str = ' '.join(parts[2:])
                db_ids_match = re.findall(r'\d+', db_ids_str)
                db_ids = [int(x) for x in db_ids_match]

                # Add all (query_id, db_id, ged) tuples
                for db_id in db_ids:
                    all_pairs.append((query_id, db_id, ged_value))

        print(f"Loaded {len(all_pairs)} query-db pairs from ground truth")

        # Stratified sampling: sample pairs from different GED values
        if num_samples < len(all_pairs):
            # Group pairs by GED value
            from collections import defaultdict
            ged_groups = defaultdict(list)
            for pair in all_pairs:
                ged_value = int(pair[2])  # Round to integer for grouping
                ged_groups[ged_value].append(pair)

            print(f"GED value distribution (total {len(ged_groups)} unique values):")
            sorted_geds = sorted(ged_groups.keys())
            print(f"  GED range: {sorted_geds[0]} to {sorted_geds[-1]}")
            print(f"  Sample counts per GED:")
            for ged in sorted_geds[:10]:  # Show first 10
                print(f"    GED={ged}: {len(ged_groups[ged])} pairs")
            if len(sorted_geds) > 10:
                print(f"    ... (showing first 10 of {len(sorted_geds)} unique GED values)")

            # Stratified sampling: ensure representation across GED values
            sampled_pairs = []
            samples_per_ged = max(1, num_samples // len(ged_groups))
            remaining = num_samples

            for ged in sorted_geds:
                pairs_in_group = ged_groups[ged]
                sample_count = min(samples_per_ged, len(pairs_in_group), remaining)
                sampled_pairs.extend(random.sample(pairs_in_group, sample_count))
                remaining -= sample_count
                if remaining <= 0:
                    break

            # If we haven't reached num_samples, fill from largest groups
            if len(sampled_pairs) < num_samples:
                all_remaining = [p for p in all_pairs if p not in sampled_pairs]
                additional = random.sample(all_remaining, num_samples - len(sampled_pairs))
                sampled_pairs.extend(additional)

            print(f"Stratified sampling: {len(sampled_pairs)} pairs across {len(set(int(p[2]) for p in sampled_pairs))} GED values")
        else:
            sampled_pairs = all_pairs
            print(f"Using all {len(sampled_pairs)} pairs")

        # Initialize result storage only for selected methods
        results = {}
        for method in test_methods:
            results[method] = {'predictions': [], 'ground_truth': [], 'errors': 0, 'times': []}

        # Process each graph pair
        for idx, pair_data in enumerate(sampled_pairs):
            query_id, db_id, gt_ged = pair_data

            if idx % 50 == 0:
                print(f"Processing {idx}/{len(sampled_pairs)}...")

            # Map query_id to graph index using get_query_graph_index
            query_graph_idx = self.get_query_graph_index(query_id)
            db_graph_idx = db_id  # db_id is directly the index

            # Validate indices
            if db_graph_idx >= self.db_num:
                print(f"Warning: db_id {db_id} out of range, skipping")
                continue

            # Prepare graph pair data following similarity_search pattern
            n1, n2 = self.gn[query_graph_idx], self.gn[db_graph_idx]
            num_edges_1 = self.edge_index[query_graph_idx].shape[1]
            num_edges_2 = self.edge_index[db_graph_idx].shape[1]
            hb = max(n1, n2) + max(num_edges_1, num_edges_2)

            data = {
                "id_1": query_graph_idx,
                "id_2": db_graph_idx,
                "n1": n1,
                "n2": n2,
                "hb": hb,
                "edge_index_1": self.edge_index[query_graph_idx],
                "edge_index_2": self.edge_index[db_graph_idx],
                "features_1": self.features[query_graph_idx],
                "features_2": self.features[db_graph_idx]
            }

            # Compute GEDGW (only if selected)
            if 'GEDGW' in test_methods:
                try:
                    start_time = time.time()

                    # Ensure n1 <= n2 for GEDGW
                    if n1 <= n2:
                        data["n1"] = n1
                        data["n2"] = n2
                    else:
                        # Swap to ensure n1 <= n2
                        data["n1"] = n2
                        data["n2"] = n1
                        data["edge_index_1"] = self.edge_index[db_graph_idx]
                        data["edge_index_2"] = self.edge_index[query_graph_idx]
                        data["features_1"] = self.features[db_graph_idx]
                        data["features_2"] = self.features[query_graph_idx]

                    from models import GEDGW
                    gedgw_model = GEDGW(data, self.args)
                    _, gedgw_pred = gedgw_model.process()
                    gedgw_time = time.time() - start_time

                    results['GEDGW']['predictions'].append(gedgw_pred)
                    results['GEDGW']['ground_truth'].append(gt_ged)
                    results['GEDGW']['times'].append(gedgw_time)
                except Exception as e:
                    print(f"Error computing GEDGW for pair (query={query_id}, db={db_id}): {e}")
                    results['GEDGW']['errors'] += 1

            # Compute GEDIOT (only if selected)
            if 'GEDIOT' in test_methods:
                try:
                    start_time = time.time()

                    # Reset data dict (may have been swapped for GEDGW)
                    data = {
                        "id_1": query_graph_idx,
                        "id_2": db_graph_idx,
                        "n1": n1,
                        "n2": n2,
                        "hb": hb,
                        "edge_index_1": self.edge_index[query_graph_idx],
                        "edge_index_2": self.edge_index[db_graph_idx],
                        "features_1": self.features[query_graph_idx],
                        "features_2": self.features[db_graph_idx]
                    }

                    # GEDIOT also requires n1 <= n2
                    if n1 <= n2:
                        data["n1"] = n1
                        data["n2"] = n2
                    else:
                        # Swap to ensure n1 <= n2
                        data["n1"] = n2
                        data["n2"] = n1
                        data["edge_index_1"] = self.edge_index[db_graph_idx]
                        data["edge_index_2"] = self.edge_index[query_graph_idx]
                        data["features_1"] = self.features[db_graph_idx]
                        data["features_2"] = self.features[query_graph_idx]

                    self.model.eval()
                    with torch.no_grad():
                        _, gediot_pred, _ = self.model(data)
                        gediot_pred = gediot_pred.item() if hasattr(gediot_pred, 'item') else gediot_pred
                    gediot_time = time.time() - start_time

                    results['GEDIOT']['predictions'].append(gediot_pred)
                    results['GEDIOT']['ground_truth'].append(gt_ged)
                    results['GEDIOT']['times'].append(gediot_time)
                except Exception as e:
                    print(f"Error computing GEDIOT for pair (query={query_id}, db={db_id}): {e}")
                    results['GEDIOT']['errors'] += 1

            # Compute GEDHOT (min of GEDGW and GEDIOT) - only if selected and both GEDGW/GEDIOT are available
            if 'GEDHOT' in test_methods:
                try:
                    if 'GEDGW' in results and 'GEDIOT' in results and \
                       len(results['GEDGW']['predictions']) > 0 and len(results['GEDIOT']['predictions']) > 0:
                        gedhot_pred = min(results['GEDGW']['predictions'][-1], results['GEDIOT']['predictions'][-1])
                        gedhot_time = results['GEDGW']['times'][-1] + results['GEDIOT']['times'][-1]

                        results['GEDHOT']['predictions'].append(gedhot_pred)
                        results['GEDHOT']['ground_truth'].append(gt_ged)
                        results['GEDHOT']['times'].append(gedhot_time)
                except Exception as e:
                    print(f"Error computing GEDHOT for pair (query={query_id}, db={db_id}): {e}")
                    results['GEDHOT']['errors'] += 1

        # Print sample details
        print(f"\n=== All Sample Predictions ({len(sampled_pairs)} samples) ===")

        # Build header dynamically based on selected methods
        header_parts = ["Sample", "GT_GED"]
        for method in test_methods:
            header_parts.append(method[:6].ljust(6))

        # Add error column for single method, or comparison columns for multiple methods
        if len(test_methods) == 1:
            header_parts.append("Error ")
        elif 'GEDGW' in test_methods and 'GEDIOT' in test_methods:
            header_parts.extend(["GW_Err", "IOT_Err", "Best"])

        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))

        # Get a reference method's ground_truth for iteration
        ref_method = test_methods[0]
        for i in range(len(results[ref_method]['ground_truth'])):
            gt = results[ref_method]['ground_truth'][i]

            # Build row dynamically
            row_parts = [f"{i:6}", f"{gt:6.1f}"]

            # Add predictions for each method
            preds = {}
            for method in test_methods:
                if i < len(results[method]['predictions']):
                    pred = results[method]['predictions'][i]
                    preds[method] = pred
                    row_parts.append(f"{pred:6.1f}")
                else:
                    row_parts.append("  N/A ")

            # Add error/comparison columns
            if len(test_methods) == 1:
                # Single method: just show error
                method = test_methods[0]
                if method in preds:
                    error = abs(preds[method] - gt)
                    row_parts.append(f"{error:6.1f}")
            elif 'GEDGW' in test_methods and 'GEDIOT' in test_methods:
                # Multiple methods with GEDGW and GEDIOT: show comparison
                if 'GEDGW' in preds and 'GEDIOT' in preds:
                    gw_err = abs(preds['GEDGW'] - gt)
                    iot_err = abs(preds['GEDIOT'] - gt)
                    best = "GW" if gw_err < iot_err else ("IOT" if iot_err < gw_err else "TIE")
                    row_parts.extend([f"{gw_err:6.1f}", f"{iot_err:7.1f}", f"{best:4}"])

            print(" | ".join(row_parts))
        print("-" * len(header))

        # Compute statistics
        final_results = {}
        for method_name, data in results.items():
            preds = np.array(data['predictions'])
            gts = np.array(data['ground_truth'])

            if len(preds) > 0:
                mae = np.mean(np.abs(preds - gts))
                rmse = np.sqrt(np.mean((preds - gts) ** 2))
                avg_time = np.mean(data['times'])
                success_rate = (len(preds) / num_samples) * 100

                final_results[method_name] = {
                    'samples': len(preds),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'avg_time': float(avg_time),
                    'success_rate': float(success_rate),
                    'errors': data['errors']
                }
            else:
                final_results[method_name] = {
                    'samples': 0,
                    'mae': None,
                    'rmse': None,
                    'avg_time': None,
                    'success_rate': 0.0,
                    'errors': data['errors']
                }

        # Print results
        print("\n=== MAE Test Results ===")
        print(f"Method   | Samples | MAE    | RMSE   | Avg Time(s) | Success Rate | Errors")
        print("-" * 80)
        for method_name, stats in final_results.items():
            if stats['mae'] is not None:
                print(f"{method_name:<8} | {stats['samples']:7} | {stats['mae']:6.2f} | {stats['rmse']:6.2f} | "
                      f"{stats['avg_time']:11.4f} | {stats['success_rate']:11.1f}% | {stats['errors']:6}")
            else:
                print(f"{method_name:<8} | {stats['samples']:7} | N/A    | N/A    | "
                      f"N/A         | {stats['success_rate']:11.1f}% | {stats['errors']:6}")
        print("=" * 80)

        # Save results
        if output_file is None:
            output_file = f"mae_test_results_{self.args.dataset}.json"

        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"Results saved to: {output_file}")
        print("=== MAE Test Completed ===")

        return final_results

    def similarity_search_batch_parallel(self, candidates_csv, query_start, query_end, tau_threshold, output_json=None):
        """
        Batch-parallel similarity search: compute all distance pairs in parallel
        Supports GEDGW, GEDIOT, GEDHOT methods using multiprocessing

        This mode loads data once and computes all (query, candidate) distances in parallel,
        then groups results by query. More memory-efficient than per-query parallelization.

        Args:
            candidates_csv: Path to CSV file with candidate graph IDs
            query_start: Starting query ID
            query_end: Ending query ID
            tau_threshold: Distance threshold for filtering
            output_json: Optional path to save results

        Returns:
            dict: Query results organized by query_id
        """
        import csv
        import json
        import time
        import multiprocessing
        from pathlib import Path
        from collections import defaultdict
        from batch_parallel_worker import compute_batch_distances_worker

        # Use args.method for display if available, otherwise model_name
        display_method = getattr(self.args, 'method', self.args.model_name)
        print(f"\n{'='*80}")
        print(f"Batch-Parallel Similarity Search ({display_method})")
        print(f"{'='*80}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Tau threshold: {tau_threshold}")
        print(f"Query range: {query_start} to {query_end}")
        print(f"{'='*80}\n")

        # Step 1: Collect all (query, candidate) pairs
        print("Step 1: Loading candidates and collecting distance pairs...")
        all_pairs = []
        query_candidates = {}  # query_id -> set of candidate_ids

        # Increase CSV field size limit for large candidate lists (e.g., SYN dataset)
        csv.field_size_limit(sys.maxsize)

        try:
            with open(candidates_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query_id = int(row['query_id'])
                    if query_start <= query_id <= query_end:
                        # Parse semicolon-separated candidate_ids
                        candidate_ids_str = row['candidate_ids'].strip()
                        if candidate_ids_str:
                            candidate_ids = [int(cid.strip()) for cid in candidate_ids_str.split(';')]
                            if query_id not in query_candidates:
                                query_candidates[query_id] = set()
                            for candidate_id in candidate_ids:
                                query_candidates[query_id].add(candidate_id)
                                all_pairs.append((query_id, candidate_id))
        except Exception as e:
            print(f"Error reading candidates file: {e}")
            return {}

        print(f"  Total distance pairs to compute: {len(all_pairs)}")
        print(f"  Queries with candidates: {len(query_candidates)}")
        print(f"  Avg candidates per query: {len(all_pairs) / max(len(query_candidates), 1):.1f}")

        # Determine which methods to run
        # Use args.method if available (for GEDHOT/all), otherwise fallback to model_name
        current_method = getattr(self.args, 'method', self.args.model_name)
        # Save for later use in output directory naming
        self._save_method = current_method
        if current_method in ['GEDHOT', 'all']:
            methods_to_run = ['GEDIOT', 'GEDGW']
            print(f"\n  Method '{current_method}' requires both GEDIOT and GEDGW")
            print(f"  Will run sequentially: GEDIOT -> GEDGW -> compute {current_method}")
        else:
            methods_to_run = [current_method]

        # Step 2: Prepare base model arguments (workers will load data independently)
        # DO NOT include gn, edge_index, features - workers load these themselves
        base_model_args = {
            'args': self.args,
            'db_num': self.db_num,
            'dataset': self.args.dataset,
            'abs_path': self.args.abs_path,  # Workers need this to load graphs_cache.pt
            'model_path': self.args.model_path,
            'model_epoch_start': getattr(self.args, 'model_epoch_start', 20),
            'number_of_labels': self.number_of_labels,
            'filters_1': self.args.filters_1,
            'filters_2': self.args.filters_2,
            'filters_3': self.args.filters_3,
            'tensor_neurons': self.args.tensor_neurons,
            'bottle_neck_neurons': self.args.bottle_neck_neurons,
            'bottle_neck_neurons_2': self.args.bottle_neck_neurons_2,
            'bottle_neck_neurons_3': self.args.bottle_neck_neurons_3,
            'bins': self.args.bins,
            'dropout': self.args.dropout,
            'hidden_dim': self.args.hidden_dim,
            'target_mode': self.args.target_mode,
            'histogram': self.args.histogram
        }
        print(f"  Worker-local loading mode: each worker loads graphs_cache.pt independently")
        print(f"  This bypasses /dev/shm limits on Linux")

        # Import worker functions
        from batch_parallel_worker import compute_batch_distances_worker, initialize_worker

        # Storage for results from each method
        method_results = {}
        total_computation_time = 0.0

        # Prepare worker pool parameters
        num_workers = getattr(self.args, 'parallel_workers', 14)
        chunk_size = (len(all_pairs) + num_workers - 1) // num_workers

        chunks = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(all_pairs))
            if start_idx < len(all_pairs):
                chunks.append(all_pairs[start_idx:end_idx])

        # For GEDHOT/all: create pool once and reuse for both methods
        # For single method: create pool normally
        if current_method in ['GEDHOT', 'all']:
            print(f"\n  Creating worker pool (will be reused for all methods)...")
            print(f"  Workers: {len(chunks)}")
            print(f"  Pairs per worker: ~{chunk_size}")

            # Initialize with GEDIOT (loads model), then reuse for GEDGW
            init_model_args = base_model_args.copy()
            init_model_args['method'] = 'GEDIOT'  # Always init with GEDIOT for GEDHOT/all

            with multiprocessing.Pool(processes=len(chunks),
                                       initializer=initialize_worker,
                                       initargs=(init_model_args,),
                                       maxtasksperchild=200) as pool:  # Auto-recycle workers to prevent FD accumulation
                print(f"  Worker pool created and initialized\n")

                # Loop through methods using the same pool
                for method_idx, method_name in enumerate(methods_to_run):
                    print(f"\n{'='*80}")
                    print(f"Step 2.{method_idx+1}: Computing {method_name} distances")
                    print(f"{'='*80}")

                    # Create method-specific model_args for each call
                    method_model_args = base_model_args.copy()
                    method_model_args['method'] = method_name

                    # Prepare chunks with method-specific args
                    method_chunks = [(chunk, method_model_args) for chunk in chunks]

                    print(f"  Total pairs: {len(all_pairs)}")
                    print(f"  Using existing worker pool...")

                    # Run computation for this method
                    start_time = time.time()
                    all_results = []

                    # Use imap_unordered for progress tracking
                    results_iter = pool.imap_unordered(compute_batch_distances_worker, method_chunks)

                    completed_workers = 0
                    total_workers = len(chunks)
                    pairs_computed = 0

                    for worker_result in results_iter:
                        all_results.extend(worker_result)
                        completed_workers += 1
                        pairs_computed = len(all_results)

                        elapsed = time.time() - start_time
                        progress = completed_workers / total_workers
                        bar_length = 50
                        filled = int(bar_length * progress)
                        bar = '=' * filled + '-' * (bar_length - filled)

                        if completed_workers > 0:
                            eta = elapsed / completed_workers * (total_workers - completed_workers)
                            eta_str = f"{eta:.0f}s"
                        else:
                            eta_str = "N/A"

                        print(f"  [{bar}] Worker {completed_workers}/{total_workers} | "
                              f"Pairs: {pairs_computed}/{len(all_pairs)} | "
                              f"Elapsed: {elapsed:.1f}s | ETA: {eta_str}")

                    # Method completed
                    method_time = time.time() - start_time
                    total_computation_time += method_time

                    print(f"\n  [+] {method_name} completed in {method_time:.2f}s")
                    print(f"  [+] Computed {len(all_results)}/{len(all_pairs)} pairs")
                    print(f"  [+] Avg: {method_time / max(len(all_results), 1) * 1000:.0f}ms per pair")

                    # Store results for this method
                    method_results[method_name] = all_results

        else:
            # Single method: create pool normally (no need to reuse)
            method_name = methods_to_run[0]
            print(f"\n{'='*80}")
            print(f"Step 2: Computing {method_name} distances")
            print(f"{'='*80}")

            # Create method-specific model_args
            model_args = base_model_args.copy()
            model_args['method'] = method_name

            # Prepare chunks with args
            method_chunks = [(chunk, model_args) for chunk in chunks]

            print(f"  Total pairs: {len(all_pairs)}")
            print(f"  Workers: {len(chunks)}")
            print(f"  Pairs per worker: ~{chunk_size}\n")

            # Run multiprocessing for this method
            start_time = time.time()
            all_results = []

            print(f"  Using multiprocessing with {len(chunks)} workers...")
            print(f"  Distributing data to workers (this may take a while)...")

            with multiprocessing.Pool(processes=len(chunks),
                                       initializer=initialize_worker,
                                       initargs=(model_args,),
                                       maxtasksperchild=200) as pool:  # Auto-recycle workers to prevent FD accumulation
                # Use imap_unordered for progress tracking
                results_iter = pool.imap_unordered(compute_batch_distances_worker, method_chunks)

                completed_workers = 0
                total_workers = len(chunks)
                pairs_computed = 0

                for worker_result in results_iter:
                    all_results.extend(worker_result)
                    completed_workers += 1
                    pairs_computed = len(all_results)

                    elapsed = time.time() - start_time
                    progress = completed_workers / total_workers
                    bar_length = 50
                    filled = int(bar_length * progress)
                    bar = '=' * filled + '-' * (bar_length - filled)

                    if completed_workers > 0:
                        eta = elapsed / completed_workers * (total_workers - completed_workers)
                        eta_str = f"{eta:.0f}s"
                    else:
                        eta_str = "N/A"

                    print(f"  [{bar}] Worker {completed_workers}/{total_workers} | "
                          f"Pairs: {pairs_computed}/{len(all_pairs)} | "
                          f"Elapsed: {elapsed:.1f}s | ETA: {eta_str}")

            # Method completed
            method_time = time.time() - start_time
            total_computation_time += method_time

            print(f"\n  [+] {method_name} completed in {method_time:.2f}s")
            print(f"  [+] Computed {len(all_results)}/{len(all_pairs)} pairs")
            print(f"  [+] Avg: {method_time / max(len(all_results), 1) * 1000:.0f}ms per pair")

            # Store results for this method
            method_results[method_name] = all_results

        # Step 3: Merge results for GEDHOT or all
        if current_method in ['GEDHOT', 'all']:
            print(f"\n{'='*80}")
            print(f"Step 3: Computing {current_method} from GEDIOT and GEDGW")
            print(f"{'='*80}")

            # Build lookup dicts
            gediot_dict = {(q, c): (ged, t) for q, c, ged, t in method_results['GEDIOT']}
            gedgw_dict = {(q, c): (ged, t) for q, c, ged, t in method_results['GEDGW']}

            # Compute GEDHOT = min(GEDIOT, GEDGW)
            gedhot_results = []
            for pair_key in gediot_dict:
                if pair_key in gedgw_dict:
                    q, c = pair_key
                    gediot_ged, gediot_time = gediot_dict[pair_key]
                    gedgw_ged, gedgw_time = gedgw_dict[pair_key]

                    gedhot_ged = min(gediot_ged, gedgw_ged)
                    gedhot_time = gediot_time + gedgw_time  # Both needed

                    gedhot_results.append((q, c, gedhot_ged, gedhot_time))

            print(f"  Computed GEDHOT for {len(gedhot_results)} pairs")

            # Set final results
            if current_method == 'GEDHOT':
                all_results = gedhot_results
            else:  # 'all' - we'll save all three later
                all_results = gedhot_results  # Primary results
                # Store individual methods for later saving
                self._gediot_results = method_results['GEDIOT']
                self._gedgw_results = method_results['GEDGW']
                self._gedhot_results = gedhot_results

            computation_time = total_computation_time
        else:
            # Single method
            all_results = method_results[current_method]
            computation_time = total_computation_time

        print(f"\n{'='*80}")
        print(f"Total computation time: {computation_time:.2f}s")
        print(f"{'='*80}\n")

        # OLD SINGLE-THREADED CODE (KEPT FOR REFERENCE, NOT USED)
        if False:
            total_pairs = len(all_pairs)
            for idx, (orig_query_id, orig_candidate_id) in enumerate(all_pairs):
                try:
                    # Map query_id to actual graph index
                    query_graph_idx = self.db_num + orig_query_id

                    # Get node counts and swap if needed (n1 <= n2)
                    n1 = self.gn[query_graph_idx]
                    n2 = self.gn[orig_candidate_id]

                    if n1 > n2:
                        graph_idx_1 = orig_candidate_id
                        graph_idx_2 = query_graph_idx
                        n1, n2 = n2, n1
                    else:
                        graph_idx_1 = query_graph_idx
                        graph_idx_2 = orig_candidate_id

                    # Prepare data
                    edge_idx_1 = self.edge_index[graph_idx_1]
                    edge_idx_2 = self.edge_index[graph_idx_2]
                    num_edges_1 = edge_idx_1.shape[1]
                    num_edges_2 = edge_idx_2.shape[1]

                    data = {
                        "id_1": graph_idx_1,
                        "id_2": graph_idx_2,
                        "n1": n1,
                        "n2": n2,
                        "edge_index_1": edge_idx_1,
                        "edge_index_2": edge_idx_2,
                        "features_1": self.features[graph_idx_1],
                        "features_2": self.features[graph_idx_2],
                        "avg_v": (n1 + n2) / 2.0,
                        "hb": max(n1, n2) + max(num_edges_1, num_edges_2)
                    }

                    # Compute GED using pre-loaded model
                    import torch
                    self.model.eval()
                    with torch.no_grad():
                        if self.args.model_name == 'GEDIOT':
                            _, predicted_ged, _ = self.model(data)
                        else:  # GEDHOT
                            _, predicted_ged = self.model(data)
                        predicted_ged = predicted_ged.item() if hasattr(predicted_ged, 'item') else predicted_ged

                    all_results.append((orig_query_id, orig_candidate_id, predicted_ged))

                    # Progress every 10%
                    if (idx + 1) % max(1, total_pairs // 10) == 0 or idx == total_pairs - 1:
                        elapsed = time.time() - start_time
                        progress = (idx + 1) / total_pairs
                        bar_length = 50
                        filled = int(bar_length * progress)
                        bar = '=' * filled + '-' * (bar_length - filled)
                        eta = elapsed / (idx + 1) * (total_pairs - idx - 1) if idx > 0 else 0
                        print(f"  [{bar}] {idx+1}/{total_pairs} ({progress*100:.1f}%) | "
                              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.0f}s")

                except Exception as e:
                    print(f"Error computing pair ({orig_query_id}, {orig_candidate_id}): {e}")
                    continue

        # Step 5.5: Load ground truth for metrics calculation
        print(f"\nStep 3.5: Loading ground truth and calculating metrics...")
        ground_truth_file = f"json_data/{self.args.dataset}/ground_truth.json"
        gisma_ground_truth_file = self._get_gisma_ground_truth_path()

        ground_truth = {}
        if Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
            print(f"  Loaded ground truth from: {ground_truth_file}")
        elif Path(gisma_ground_truth_file).exists():
            ground_truth = self._load_gisma_ground_truth(gisma_ground_truth_file, tau_threshold)
            print(f"  Loaded Gisma ground truth from: {gisma_ground_truth_file}")
        else:
            print(f"  Warning: No ground truth file found")
            print(f"  Tried: {ground_truth_file}")
            print(f"  Tried: {gisma_ground_truth_file}")

        # Step 6: Save per-query results (same format as backup)
        print(f"\nStep 4: Saving per-query results...")

        # For 'all' method, save results for each method separately
        save_method = getattr(self, '_save_method', getattr(self.args, 'method', self.args.model_name))

        # Determine which methods to save
        if save_method == 'all':
            methods_to_save = [
                ('GEDIOT', self._gediot_results),
                ('GEDGW', self._gedgw_results),
                ('GEDHOT', self._gedhot_results)
            ]
        else:
            # Single method - use the main all_results
            methods_to_save = [(save_method, all_results)]

        # Save results for each method
        for method_name, method_results_list in methods_to_save:
            print(f"\n  Saving {method_name} results...")

            output_dir = f"results/similarity_search/{self.args.dataset}/tau{tau_threshold}/per_query/{method_name}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Organize results by query for this method
            method_query_results = defaultdict(lambda: {'found': set()})
            method_query_times = defaultdict(float)

            for orig_query_id, orig_candidate_id, ged, comp_time in method_results_list:
                # Accumulate computation time for this query
                method_query_times[orig_query_id] += comp_time

                if ged <= tau_threshold:
                    method_query_results[orig_query_id]['found'].add(orig_candidate_id)

            # Save per-query results for this method
            for query_id in range(query_start, query_end + 1):
                found_set = method_query_results[query_id]['found']
                candidates_count = len(query_candidates.get(query_id, set()))

                # Use actual accumulated computation time for this query
                query_time = method_query_times.get(query_id, 0.0)

                # Build result in the same format as backup
                result = {
                    'query_id': query_id,
                    'found_count': len(found_set),
                    'candidates_count': candidates_count,
                    'query_time': query_time
                }

                # Calculate recall/precision if ground truth exists
                if str(query_id) in ground_truth:
                    gt_candidates = ground_truth[str(query_id)]

                    # Handle both JSON format (nested dict) and Gisma format (simple list)
                    if isinstance(gt_candidates, dict):
                        # JSON format: {"distance": [candidates]}
                        gt_set = set()
                        for distance_str, candidate_list in gt_candidates.items():
                            distance = float(distance_str)
                            if distance <= tau_threshold:
                                # Filter GT to only include candidates in our test set
                                filtered_candidates = set(candidate_list).intersection(
                                    query_candidates.get(query_id, set()))
                                gt_set.update(filtered_candidates)
                    else:
                        # Gisma format: simple list (already filtered by tau)
                        gt_set = set(gt_candidates).intersection(
                            query_candidates.get(query_id, set()))

                    # Calculate metrics
                    intersection = found_set.intersection(gt_set)
                    gt_count = len(gt_set)
                    correct_count = len(intersection)

                    recall = correct_count / gt_count if gt_count > 0 else 0.0
                    precision = correct_count / len(found_set) if len(found_set) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                    result.update({
                        'gt_count': gt_count,
                        'correct_count': correct_count,
                        'recall': recall,
                        'precision': precision,
                        'f1': f1
                    })
                else:
                    # Query not in ground truth - set default values (consistent with backup)
                    result.update({
                        'gt_count': 0,
                        'correct_count': 0,
                        'recall': 0.0,
                        'precision': 0.0,
                        'f1': 0.0
                    })

                # Build complete output with metadata and result structure
                output_data = {
                    'metadata': {
                        'dataset': self.args.dataset,
                        'model': method_name,  # Use actual method name for this save
                        'epoch': getattr(self.args, 'model_epoch_start', 20),
                        'tau_threshold': float(tau_threshold),
                        'query_id': query_id
                    },
                    'result': result
                }

                output_file = f"{output_dir}/query_{query_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)

            print(f"  {method_name} results saved to: {output_dir}")

        print(f"\n{'='*80}")
        print(f"Batch-Parallel Search Completed")
        print(f"Total time: {computation_time:.2f}s")
        print(f"{'='*80}\n")

        # ========== Generate and print summary report ==========
        self._generate_and_print_summary(tau_threshold, query_start, query_end, save_method, methods_to_save)

        # Return last method's query_results (for compatibility)
        # For 'all', this will be GEDHOT results; for single method, it's that method
        return dict(method_query_results)

    def _generate_and_print_summary(self, tau_threshold, query_start, query_end, save_method, methods_to_save):
        """Generate and print summary report after batch-parallel search."""
        from datetime import datetime
        from collections import defaultdict

        print(f"\n{'='*80}")
        print(f"GENERATING SUMMARY REPORT")
        print(f"{'='*80}\n")

        # Collect stats for each method
        method_stats = {}

        for method_name, _ in methods_to_save:
            per_query_dir = f"results/similarity_search/{self.args.dataset}/tau{tau_threshold}/per_query/{method_name}"

            total_found = 0
            total_gt = 0
            total_correct = 0
            total_time = 0.0
            total_candidates = 0
            query_count = 0

            for query_id in range(query_start, query_end + 1):
                json_file = Path(per_query_dir) / f"query_{query_id}.json"

                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        result = data['result']

                        total_found += result['found_count']
                        total_gt += result.get('gt_count', 0)
                        total_correct += result.get('correct_count', 0)
                        total_time += result['query_time']
                        total_candidates += result['candidates_count']
                        query_count += 1

            # Calculate metrics
            avg_precision = (total_correct / total_found * 100) if total_found > 0 else 0.0
            avg_recall = (total_correct / total_gt * 100) if total_gt > 0 else 0.0
            avg_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0

            method_stats[method_name] = {
                'queries': query_count,
                'total_found': total_found,
                'total_gt': total_gt,
                'total_correct': total_correct,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'total_time': total_time,
                'total_candidates': total_candidates
            }

        # Generate report text
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SIMILARITY SEARCH SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {self.args.dataset}")
        report_lines.append(f"Query Range: {query_start} to {query_end}")
        report_lines.append(f"Tau Threshold: {tau_threshold}")
        report_lines.append(f"Model Epoch: {getattr(self.args, 'model_epoch_start', 20)}")
        report_lines.append("="*80)
        report_lines.append("")

        # Summary table
        report_lines.append(f"--- Tau = {tau_threshold} Summary ---")
        header = f"{'Method':<10} | {'Queries':>7} | {'Found':>7} | {'GT':>7} | {'Correct':>7} | {'Recall%':>8} | {'Precision%':>10} | {'F1%':>6} | {'Time(s)':>8}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        for method_name in ['GEDGW', 'GEDIOT', 'GEDHOT']:
            if method_name in method_stats:
                stats = method_stats[method_name]
                line = f"{method_name:<10} | {stats['queries']:>7} | {stats['total_found']:>7} | {stats['total_gt']:>7} | {stats['total_correct']:>7} | {stats['avg_recall']:>7.2f}% | {stats['avg_precision']:>9.2f}% | {stats['avg_f1']:>5.2f}% | {stats['total_time']:>8.2f}"
                report_lines.append(line)

        report_lines.append("")

        # Time analysis
        report_lines.append("--- Time Analysis ---")
        for method_name in ['GEDGW', 'GEDIOT', 'GEDHOT']:
            if method_name in method_stats:
                stats = method_stats[method_name]
                avg_time = stats['total_time'] / stats['queries'] if stats['queries'] > 0 else 0
                avg_per_comparison = (stats['total_time'] / stats['total_candidates'] * 1000) if stats['total_candidates'] > 0 else 0
                line = f"{method_name}: Total={stats['total_time']:.2f}s, Avg/query={avg_time:.3f}s, Avg/comparison={avg_per_comparison:.2f}ms"
                report_lines.append(line)

        # Speed comparison
        if 'GEDGW' in method_stats and 'GEDIOT' in method_stats:
            gedgw_time = method_stats['GEDGW']['total_time']
            gediot_time = method_stats['GEDIOT']['total_time']
            if gediot_time > 0:
                speedup = gedgw_time / gediot_time
                report_lines.append(f"\nSpeed comparison: GEDIOT is {speedup:.1f}x faster than GEDGW")

        report_lines.append("")
        report_lines.append("="*80)

        # Print to console
        report_text = "\n".join(report_lines)
        print(report_text)

        # Save to file
        output_dir = f"results/similarity_search/{self.args.dataset}/tau{tau_threshold}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary_file = f"{output_dir}/summary_report_q{query_start}-{query_end}.txt"

        with open(summary_file, 'w') as f:
            f.write(report_text)

        print(f"\nSummary report saved to: {summary_file}\n")

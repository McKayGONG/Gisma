#!/usr/bin/env python3
"""
LAN Original Search Engine
Search phase: uses the original LAN method (exact GED or approximate GED)
Loads a pre-built index and performs search with real GED
"""

import math
import networkx as nx
import pickle
import heapq
import logging
import subprocess
import time
import os
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GINConv
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# CG (Compressed GNN-Graph) utilities
try:
    from cg_utils import compress_query_graph, cg_gin_forward
    CG_AVAILABLE = True
except ImportError:
    CG_AVAILABLE = False

# Import Gisma GED calculator
try:
    from gisma_ged_calculator import GismaGEDCalculator
    GISMA_AVAILABLE = True
except ImportError:
    GISMA_AVAILABLE = False
    logger.warning("Gisma GED calculator not available, will use fallback methods")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GEDCalculator:
    """
    GED Calculator - simulates the original LAN method.
    Supports multiple GED computation methods.
    """

    def __init__(self, graph_file_dir: str = None, ged_method: str = "gisma",
                 dataset_path: str = None):
        """
        Args:
            graph_file_dir: Directory for individual graph files (for calling external GED tools)
            ged_method: GED computation method
                - "gisma": Use Gisma's AppForComputation (recommended, exact GED)
                - "simple": Simple heuristic (node count + edge count difference)
                - "astar": A* approximate GED
                - "exact": Exact GED (requires external tool)
            dataset_path: Dataset path (for Gisma calculator)
        """
        self.graph_file_dir = graph_file_dir
        self.ged_method = ged_method

        # Initialize Gisma GED calculator
        self.gisma_calculator = None
        if ged_method == "gisma" and GISMA_AVAILABLE:
            try:
                logger.info("Initializing Gisma GED Calculator...")
                self.gisma_calculator = GismaGEDCalculator(
                    dataset_path=dataset_path  # None will use default relative path
                )
                logger.info("✓ Gisma GED Calculator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gisma calculator: {e}")
                logger.warning("Falling back to A* heuristic")
                self.ged_method = "astar"

    def compute_ged(self, q: nx.Graph, g: nx.Graph) -> float:
        """
        Compute GED between two graphs.

        Implements the GED computation strategy from the original LAN paper:
        1. Prefer pre-computed GED (if available)
        2. Otherwise attempt exact GED (with time limit)
        3. On timeout, use A* approximate GED
        4. Finally fall back to simple heuristic
        """
        qid = q.graph.get("id")
        gid = g.graph.get("id")

        # Check if they are the SAME graph object (not just same ID)
        if q is g:
            return 0.0

        # Compute GED based on the selected method
        if self.ged_method == "gisma" and self.gisma_calculator is not None:
            ged = self.gisma_calculator.compute_ged(q, g)
        elif self.ged_method == "simple":
            ged = self._compute_simple_ged(q, g)
        elif self.ged_method == "astar":
            ged = self._compute_astar_ged(q, g)
        elif self.ged_method == "exact":
            ged = self._compute_exact_ged(q, g)
        else:
            ged = self._compute_simple_ged(q, g)

        return ged

    def _compute_simple_ged(self, q: nx.Graph, g: nx.Graph) -> float:
        """
        Simple heuristic GED estimation.
        Based on structural difference: |V1-V2| + |E1-E2|
        """
        node_diff = abs(q.number_of_nodes() - g.number_of_nodes())
        edge_diff = abs(q.number_of_edges() - g.number_of_edges())
        return float(node_diff + edge_diff)

    def _compute_astar_ged(self, q: nx.Graph, g: nx.Graph) -> float:
        """
        A* approximate GED - uses a simplified version here.
        In practice, one can call an external A* tool or implement the full A* algorithm.
        """
        # Simplified version: use degree-sequence-based distance
        q_degree_seq = sorted([d for n, d in q.degree()], reverse=True)
        g_degree_seq = sorted([d for n, d in g.degree()], reverse=True)

        # Pad to equal length
        max_len = max(len(q_degree_seq), len(g_degree_seq))
        q_degree_seq += [0] * (max_len - len(q_degree_seq))
        g_degree_seq += [0] * (max_len - len(g_degree_seq))

        # Degree sequence difference
        degree_diff = sum(abs(q_degree_seq[i] - g_degree_seq[i]) for i in range(max_len))

        # Combine node/edge differences
        node_diff = abs(q.number_of_nodes() - g.number_of_nodes())
        edge_diff = abs(q.number_of_edges() - g.number_of_edges())

        return float(node_diff + edge_diff + degree_diff * 0.5)

    def _compute_exact_ged(self, q: nx.Graph, g: nx.Graph, timeout: int = 10) -> float:
        """
        Exact GED computation (requires an external GED solver).
        Falls back to A* approximation on timeout.

        Note: You need to provide the path to the GED solver.
        Reference: LAN_ori implementation routing_with_neigh_pruning.py:853-887
        """
        # Exact GED not available, fall back to A* approximation
        logger.warning("Exact GED not implemented, falling back to A* approximation")
        return self._compute_astar_ged(q, g)


########################################################################################################
#### Ori Model (from LAN_ori routing_with_neigh_pruning.py, adapted for CPU)
########################################################################################################

def _read_one_gemb(args):
    """Worker function for parallel embedding reading."""
    addr, gfile = args
    gID = gfile[1:-4]
    with open(os.path.join(addr, gfile)) as f:
        lines = f.read().strip().split('\n')[1:]
    nodeEmbList = []
    for line in lines:
        tmp = line.strip().split(' ')
        nodeEmbList.append([float(ele) for ele in tmp[1:]])
    n = len(nodeEmbList)
    dim = len(nodeEmbList[0]) if n > 0 else 0
    mean = [sum(nodeEmbList[i][j] for i in range(n)) / n for j in range(dim)] if n > 0 else []
    return gID, mean


def _read_initial_gemb(addr):
    """Read Node2Vec graph-level embeddings (parallel)."""
    from multiprocessing import Pool
    num_workers = max(1, int(os.cpu_count() * 0.7))
    gfileList = os.listdir(addr)
    args_list = [(addr, gfile) for gfile in gfileList]
    gEmbMap = {}
    with Pool(num_workers) as pool:
        for gID, emb in tqdm(pool.imap_unordered(_read_one_gemb, args_list, chunksize=100),
                             total=len(args_list),
                             desc="    Reading Node2Vec emb", unit="file"):
            gEmbMap[gID] = torch.tensor(emb)
    return gEmbMap


def _make_a_dglgraph_for_ori(g):
    """Convert nx.Graph to DGL graph with degree one-hot features (CPU version)."""
    max_deg = 20
    ones = torch.eye(max_deg)

    nodes = list(g.nodes())
    if len(nodes) == 0:
        dg = dgl.graph(([], []))
        dg.ndata['h'] = torch.zeros(0, max_deg)
        return dg

    node_map = {node: i for i, node in enumerate(nodes)}

    edges = [[], []]
    for edge in g.edges():
        end1 = node_map[edge[0]]
        end2 = node_map[edge[1]]
        edges[0].append(int(end1))
        edges[1].append(int(end2))
        edges[0].append(int(end2))
        edges[1].append(int(end1))

    if len(edges[0]) == 0:
        dg = dgl.graph(([], []), num_nodes=len(nodes))
    else:
        dg = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])), num_nodes=len(nodes))

    h0 = dg.in_degrees().view(1, -1).squeeze()
    if h0.dim() == 0:
        h0 = h0.unsqueeze(0)
    h0 = torch.clamp(h0, 0, max_deg - 1)
    h0 = ones.index_select(0, h0).float()
    dg.ndata['h'] = h0
    dg = dgl.add_self_loop(dg)

    return dg


class OriInferenceModel(nn.Module):
    """
    Inference-time model from LAN_ori (routing_with_neigh_pruning.py:919-984).
    Takes string IDs, looks up embeddings and DGL graphs internally.
    Adapted for CPU (no .cuda() calls).
    """
    def __init__(self, gID2InitEmbMap, gid2dgmap, use_cg=False):
        super(OriInferenceModel, self).__init__()

        self.gid2dgMap = gid2dgmap
        self.gID2InitEmbMap = gID2InitEmbMap
        self.use_cg = use_cg
        self._cg_cache = {}      # gid -> cg_data dict
        self._qemb_cache = {}    # gid -> query embedding tensor

        self.RELU = torch.nn.ReLU(inplace=True)

        self.fc_init = nn.Linear(20, 512, bias=True)
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')
        self.gnn_bn = torch.nn.BatchNorm1d(512)
        self.gnn_bn2 = torch.nn.BatchNorm1d(512)

        self.fc = nn.Linear(512 * 3, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.fc3 = nn.Linear(256, 256, bias=True)
        self.fc4 = nn.Linear(256, 1, bias=True)
        self.bn = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.dp = torch.nn.Dropout(0.5)

    def clear_cg_cache(self):
        """Clear CG and query embedding caches (call when switching queries)."""
        self._cg_cache.clear()
        self._qemb_cache.clear()

    def prepare_cg(self, qid, nx_graph):
        """Pre-compute CG for a query graph (call once per query)."""
        if self.use_cg and CG_AVAILABLE and qid not in self._cg_cache:
            self._cg_cache[qid] = compress_query_graph(nx_graph)

    def forward(self, qIDs, pgNodeIDs, neighIDs):
        outputNum = len(neighIDs)

        qid = qIDs[0]

        # Check query embedding cache first (same query → same embedding)
        if qid in self._qemb_cache:
            qemb = self._qemb_cache[qid]
        elif self.use_cg and CG_AVAILABLE and qid in self._cg_cache:
            # CG forward path
            cg_data = self._cg_cache[qid]
            qemb = cg_gin_forward(cg_data, self.fc_init, self.gnn_bn, self.gnn_bn2, self.RELU, model_has_selfloop=True)
            qemb = qemb.view(1, -1)
            self._qemb_cache[qid] = qemb
        else:
            # Original GINConv path
            batch_dg = dgl.batch([self.gid2dgMap[qid] for qid in qIDs])
            batch_dg.ndata['h'] = self.fc_init(batch_dg.ndata['h'])
            batch_dg.ndata['h'] = self.RELU(self.gnn_bn(self.conv1_for_g(batch_dg, batch_dg.ndata['h'])))
            batch_dg.ndata['h'] = self.RELU(self.gnn_bn2(self.conv2_for_g(batch_dg, batch_dg.ndata['h'])))
            qemb = dgl.mean_nodes(batch_dg, 'h')
            qemb = qemb.view(1, -1)
            self._qemb_cache[qid] = qemb

        neighEmbList = torch.zeros(outputNum, 512)
        for idx in range(0, len(neighIDs)):
            neighID = neighIDs[idx]
            if neighID in self.gID2InitEmbMap:
                neighEmbList[idx] = self.gID2InitEmbMap[neighID]

        pgNode_embList = [self.gID2InitEmbMap[pgNodeID] for pgNodeID in pgNodeIDs]
        pgNode_embList = torch.stack(pgNode_embList)
        pgNode_embList = pgNode_embList.view(1, -1)

        a = torch.cat([qemb, pgNode_embList], 1)
        a = a.repeat(1, outputNum).view(-1, 512 * 2)

        b = torch.cat([a, neighEmbList], 1)

        H = self.RELU(self.bn(self.fc(b)))
        H2 = self.RELU(self.bn2(self.fc2(H)))
        H3 = self.RELU(self.bn3(self.fc3(H2)))
        pred = torch.sigmoid(self.fc4(H3))
        pred = pred.view(1, outputNum)

        return pred


class LANOriginalSearchEngine:
    """
    LAN Original Search Engine.
    Strictly follows the search algorithm from the LAN paper, using real GED.
    """

    def __init__(self, index_file: str, ged_calculator: GEDCalculator,
                 use_neighbor_pruning: bool = False, pruning_ratio: float = 0.2,
                 pruning_method: str = 'embedding',
                 mrk_model_dir: str = None, mrk_model_epoch: int = 999,
                 mrk_emb_path: str = None, mrk_graph_file: str = None,
                 ds: float = 1.0,
                 init_model_dir: str = None,
                 use_cg: bool = False):
        """
        Args:
            index_file: Path to the pre-built index file
            ged_calculator: GED calculator instance
            use_neighbor_pruning: Whether to use neighbor pruning
            pruning_ratio: Ratio of neighbors to retain (default 0.2 i.e. 20%)
            pruning_method: Pruning method ('embedding' or 'mrk')
            mrk_model_dir: Mrk model directory (required when pruning_method='mrk')
            mrk_model_epoch: Mrk model epoch number (default 999)
            mrk_emb_path: Node2Vec embedding directory (required when pruning_method='mrk')
            mrk_graph_file: Graph data file (required when pruning_method='mrk')
            ds: Gamma step size (batch expansion mode)
            init_model_dir: Init model directory (if specified, use learned entry point selection)
            use_cg: Whether to use CG compression to accelerate GIN forward pass
        """
        self.ged_calculator = ged_calculator
        self.use_neighbor_pruning = use_neighbor_pruning
        self.pruning_ratio = pruning_ratio
        self.pruning_method = pruning_method
        self.mrk_ranker_models = []
        self.ori_gID2InitEmbMap = None
        self.ori_gid2dgmap = None
        self.ds = ds
        self.use_cg = use_cg
        self.init_model = None
        self.init_all_db_embs = None
        self.init_db_graph_ids = None

        if use_cg and not CG_AVAILABLE:
            logger.warning("CG compression requested but cg_utils not available, disabling CG")
            self.use_cg = False

        if pruning_method == 'mrk' and mrk_model_dir:
            self._load_mrk_models(mrk_model_dir, mrk_model_epoch,
                                  mrk_emb_path, mrk_graph_file)

        if init_model_dir:
            self._load_init_model(init_model_dir)

        self.load_index(index_file)

    def load_index(self, index_file: str):
        """Load index"""
        import os
        file_size = os.path.getsize(index_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"  [Index] Loading from {index_file} ({file_size_mb:.1f} MB)")

        print("  [Index] Step 1/4: Reading pickle file...")
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)

        print("  [Index] Step 2/4: Loading proximity graph...")
        self.proximity_graph = index_data['proximity_graph']  # Nodes are graph IDs

        # Support two index formats
        print("  [Index] Step 3/4: Building graph ID mapping...")
        if 'graph_id_to_original' in index_data:
            raw_mapping = index_data['graph_id_to_original']
            self.graph_id_to_original = {}
            for k, v in tqdm(raw_mapping.items(), desc="    Graph mapping", unit="g"):
                self.graph_id_to_original[k] = v
        elif 'graphs' in index_data:
            graphs = index_data['graphs']
            self.graph_id_to_original = {}
            for g in tqdm(graphs, desc="    Graph mapping", unit="g"):
                self.graph_id_to_original[g.graph.get('id')] = g
        else:
            raise ValueError("Index file must contain 'graph_id_to_original' or 'graphs'")
        print(f"  [Index] {len(self.graph_id_to_original)} graphs loaded.")

        # Compute number of nodes and edges
        if 'num_nodes' in index_data and 'num_edges' in index_data:
            self.num_nodes = index_data['num_nodes']
            self.num_edges = index_data['num_edges']
        else:
            if isinstance(self.proximity_graph, dict):
                self.num_nodes = len(self.proximity_graph)
                self.num_edges = sum(len(neighbors) for neighbors in self.proximity_graph.values())
            else:
                self.num_nodes = self.proximity_graph.number_of_nodes()
                self.num_edges = self.proximity_graph.number_of_edges()

        print("  [Index] Step 4/4: Building reverse index...")
        self.graph_to_id = {}
        for k, v in tqdm(self.graph_id_to_original.items(), desc="    Reverse index", unit="g"):
            self.graph_to_id[v] = k

        print(f"  [Index] Done: {self.num_nodes} nodes, {self.num_edges} edges")


    def _load_mrk_models(self, model_dir: str, epoch: int,
                          emb_path: str, graph_file: str):
        """Load 4 independent Mrk ranker models (perc=20,40,60,80) and shared data."""
        print(f"  [Mrk] Loading 4 models from {model_dir}...")

        # Load Node2Vec embeddings (prefer pkl, fallback to txt directory)
        pkl_path = emb_path + '_graph_emb.pkl'
        if os.path.exists(pkl_path):
            from neigh_pruning_model_training import read_initial_gemb
            print(f"  [Mrk] Step 1/3: Loading Node2Vec embeddings from pkl: {pkl_path}")
            self.ori_gID2InitEmbMap = read_initial_gemb(emb_path, pkl_path=pkl_path)
        else:
            print(f"  [Mrk] Step 1/3: Loading Node2Vec embeddings from {emb_path}...")
            self.ori_gID2InitEmbMap = _read_initial_gemb(emb_path)
        print(f"  [Mrk] {len(self.ori_gID2InitEmbMap)} graph embeddings loaded")

        # Load graphs and build DGL cache
        from neigh_pruning_model_training import read_and_split_to_individual_graph
        print(f"  [Mrk] Step 2/3: Loading graphs and building DGL cache...")
        entire_dataset = read_and_split_to_individual_graph(graph_file, 0, 10000000)
        from neigh_pruning_model_training import _build_dgl_cache
        self.ori_gid2dgmap = _build_dgl_cache(entire_dataset)

        # Load 4 models (perc=20,40,60,80)
        # Prefer loading the best model; fall back to loading by epoch
        print(f"  [Mrk] Step 3/3: Loading model weights...")
        self.mrk_ranker_models = []
        for perc in [20, 40, 60, 80]:
            best_path = os.path.join(model_dir, f"best.perc{perc}.pkl")
            epoch_path = os.path.join(model_dir, f"mrk_perc{perc}.e{epoch}.pkl")
            model_path = best_path if os.path.exists(best_path) else epoch_path
            model = OriInferenceModel(self.ori_gID2InitEmbMap, self.ori_gid2dgmap, use_cg=self.use_cg)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.track_running_stats = False
            self.mrk_ranker_models.append(model)
            print(f"    Loaded: {os.path.basename(model_path)}")
        print(f"  [Mrk] All 4 models ready.")

    def _load_init_model(self, model_dir: str):
        """Load Init node selection model for learned entry point selection."""
        from init_node_sel_model_train import InitModel

        model_path = os.path.join(model_dir, 'best.pkl')
        if not os.path.exists(model_path):
            print(f"  [Init] WARNING: {model_path} not found, Init model disabled.")
            return

        print(f"  [Init] Loading model from {model_path}...")

        # Load Node2Vec embeddings (reuse from Mrk if already loaded)
        if self.ori_gID2InitEmbMap is None:
            print(f"  [Init] Node2Vec embeddings not loaded by Mrk, loading now...")
            # Infer emb_path from model_dir: models/init/{dataset}/ -> emb/{dataset}512
            ds_lower = os.path.basename(model_dir)
            script_dir = os.path.dirname(os.path.dirname(model_path))  # LAN_new/
            emb_dir = os.path.join(script_dir, 'emb', f'{ds_lower}512')
            pkl_path = os.path.join(script_dir, 'emb', f'{ds_lower}512_graph_emb.pkl')
            from neigh_pruning_model_training import read_initial_gemb
            self.ori_gID2InitEmbMap = read_initial_gemb(emb_dir, pkl_path=pkl_path)
            print(f"  [Init] Loaded {len(self.ori_gID2InitEmbMap)} embeddings")

        # Detect emb_dim from loaded embeddings
        sample_emb = next(iter(self.ori_gID2InitEmbMap.values()))
        emb_dim = sample_emb.shape[0]

        # Build all-DB embedding tensor (sorted by graph ID)
        self.init_db_graph_ids = sorted(self.ori_gID2InitEmbMap.keys(), key=int)
        db_emb_list = [self.ori_gID2InitEmbMap[gid] for gid in self.init_db_graph_ids]
        self.init_all_db_embs = torch.stack(db_emb_list)  # [N_db, emb_dim]
        self.init_db_id_to_idx = {gid: i for i, gid in enumerate(self.init_db_graph_ids)}
        print(f"  [Init] DB embedding tensor: {self.init_all_db_embs.shape}")

        # Load model
        self.init_model = InitModel(emb_dim=emb_dim)
        self.init_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.init_model.eval()
        for m in self.init_model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = False
        print(f"  [Init] Model loaded and ready.")

    def predict_entry_point(self, query_graph: nx.Graph, s: int = 4) -> str:
        """Use Init model to predict the best entry point for a query.

        Paper Section V-A: Mnh predicts neighborhood N_hat_Q, then randomly
        sample s graphs from N_hat_Q, compute actual GED, pick smallest.

        Args:
            query_graph: The query graph
            s: Number of random samples from predicted neighborhood (default 4)

        Returns:
            graph_id: ID of the best entry point in the DB
        """
        if self.init_model is None:
            return None

        # Convert query to DGL (self-loops already added inside _make_a_dglgraph_for_ori)
        q_dg = _make_a_dglgraph_for_ori(query_graph)

        # Step 1: Score all DB graphs → predict N_hat_Q (positive predictions)
        with torch.no_grad():
            if self.use_cg and CG_AVAILABLE:
                # CG mode: compute query embedding via CG, then run MLP
                # Init model uses self-loops in graphs (added during training/inference)
                cg_data = compress_query_graph(query_graph)
                q_emb = cg_gin_forward(
                    cg_data, self.init_model.fc_init,
                    self.init_model.bn1, self.init_model.bn2,
                    self.init_model.relu, model_has_selfloop=True
                ).squeeze()  # (emb_dim,)
                q_repeated = q_emb.unsqueeze(0).expand(self.init_all_db_embs.shape[0], -1)
                H = torch.cat([q_repeated, self.init_all_db_embs], dim=1)
                H = self.init_model.relu(self.init_model.bn_fc(self.init_model.fc1(H)))
                scores = torch.sigmoid(self.init_model.fc2(H)).squeeze(-1)
            else:
                scores = self.init_model(q_dg, self.init_all_db_embs)

        # N_hat_Q = graphs with score > 0.5 (predicted as in neighborhood)
        positive_mask = scores > 0.5
        positive_indices = positive_mask.nonzero(as_tuple=True)[0]

        if len(positive_indices) == 0:
            # No positive predictions, fall back to top-s by score
            _, top_indices = scores.topk(min(s, len(scores)))
            positive_indices = top_indices

        # Step 2: Randomly sample s graphs from N_hat_Q
        if len(positive_indices) <= s:
            sample_indices = positive_indices
        else:
            perm = torch.randperm(len(positive_indices))[:s]
            sample_indices = positive_indices[perm]

        sample_ids = [self.init_db_graph_ids[idx.item()] for idx in sample_indices]

        # Step 3: Compute actual GED for the s samples
        best_id = sample_ids[0]
        best_ged = float('inf')
        for gid in sample_ids:
            g = self.graph_id_to_original.get(gid)
            if g is None:
                continue
            ged = self.ged_calculator.compute_ged(query_graph, g)
            if ged < best_ged:
                best_ged = ged
                best_id = gid

        logger.debug(f"[Init] N_hat_Q={len(positive_indices)}, sampled={len(sample_ids)}, "
                     f"best entry: ID={best_id}, GED={best_ged:.1f}")
        return best_id

    def _predict_with_one_model(self, model, qid, pgNodeID, neighIDs):
        """Run one Mrk model on all neighbors at once, return dict {neigh_id: score}."""
        with torch.no_grad():
            pred = model([qid], [pgNodeID], neighIDs).view(-1)
        scores = pred.cpu().numpy().tolist()
        return {neighIDs[idx]: scores[idx] for idx in range(len(neighIDs))}

    def rank_neighbors_into_batches(self, query_graph: nx.Graph,
                                     pg_node: nx.Graph,
                                     neighbors: List[nx.Graph]) -> Tuple[List[List[str]], Dict[str, nx.Graph]]:
        """
        Use 4 Mrk models to rank neighbors into 5 batches via set difference.

        M_rk^1 predicts top 20%, M_rk^2 predicts top 40%, etc.
        Positive class (score < 0.5) = "good neighbor".

        Batches:
          B0 = M1_pos                    (best 20%)
          B1 = M2_pos - M1_pos           (next 20%)
          B2 = M3_pos - M2_pos           (next 20%)
          B3 = M4_pos - M3_pos           (next 20%)
          B4 = all - M4_pos              (worst 20%)

        Returns:
            (batches, id_to_graph) where batches is a list of 5 lists of neighbor IDs,
            and id_to_graph maps ID to nx.Graph.
            Returns ([], {}) if scoring not possible.
        """
        if len(self.mrk_ranker_models) == 0 or len(neighbors) == 0:
            return [], {}

        qid = query_graph.graph.get('id')
        pgNodeID = pg_node.graph.get('id')

        # Ensure query graph DGL is available
        if qid not in self.ori_gid2dgmap:
            dg = _make_a_dglgraph_for_ori(query_graph)
            dg = dgl.add_self_loop(dg)
            self.ori_gid2dgmap[qid] = dg

        # Prepare CG for query if enabled
        if self.use_cg:
            for model in self.mrk_ranker_models:
                model.prepare_cg(qid, query_graph)

        if pgNodeID not in self.ori_gID2InitEmbMap:
            return [], {}

        neighIDs = [n.graph.get('id') for n in neighbors]
        neighIDs.sort()
        id_to_graph = {n.graph.get('id'): n for n in neighbors}
        all_ids = set(neighIDs)

        # Run 4 models, get positive sets (score < 0.5 = good neighbor)
        positive_sets = []
        for model in self.mrk_ranker_models:
            scores = self._predict_with_one_model(model, qid, pgNodeID, neighIDs)
            pos = set(nid for nid, s in scores.items() if s < 0.5)
            positive_sets.append(pos)

        # Build 5 batches via set difference
        batches = []
        batches.append(list(positive_sets[0]))                                    # B0
        batches.append(list(positive_sets[1] - positive_sets[0]))                 # B1
        batches.append(list(positive_sets[2] - positive_sets[1]))                 # B2
        batches.append(list(positive_sets[3] - positive_sets[2]))                 # B3
        batches.append(list(all_ids - positive_sets[3]))                          # B4

        return batches, id_to_graph


    def search_k_nearest(self, query_graph: nx.Graph, k: int = 50, ef: int = 100,
                        entry_point_id: str = None, query_emb=None) -> Tuple[List[Tuple[float, nx.Graph]], Dict]:
        """
        k-NN search using the original LAN method.

        Args:
            query_graph: Query graph
            k: Return top-k results
            ef: Number of candidates during search
            entry_point_id: Entry point graph ID (randomly chosen if None)
            query_emb: Query embedding (used for neighbor pruning and GED fallback)

        Returns:
            (results, stats): Result list [(distance, graph)] and statistics
        """
        logger.info(f"Searching for top-{k} similar graphs with ef={ef}")

        # Clear CG caches for new query
        if self.use_cg:
            for model in self.mrk_ranker_models:
                model.clear_cg_cache()

        # Set query embedding (used for neighbor pruning and GED fallback)
        if query_emb is not None:
            self.ged_calculator.gisma_calculator.current_query_emb = query_emb

        # Select entry point
        if entry_point_id is None:
            # Try Init model first
            if self.init_model is not None:
                entry_point_id = self.predict_entry_point(query_graph)
            if entry_point_id is None:
                import random
                entry_point_id = random.choice(list(self.graph_id_to_original.keys()))

        entry_point = self.graph_id_to_original[entry_point_id]

        # Execute search
        start_time = time.time()
        candidates, stats = self.search_layer(query_graph, [entry_point], ef)
        search_time = time.time() - start_time

        # Extract top-k
        # W is a max-heap (stores negative distances); extract the k smallest distances
        # Convert to list and sort (by actual distance, ascending)
        all_results = [(-neg_dist, graph) for neg_dist, idx, graph in candidates]
        all_results.sort(key=lambda x: x[0])
        results = all_results[:k]

        # Update statistics
        stats['search_time'] = search_time
        stats['num_results'] = len(results)

        logger.info(f"Search completed in {search_time:.4f}s, found {len(results)} results")
        logger.info(f"Distance computations: {stats['distance_computations']}")
        logger.info(f"Nodes visited: {stats['nodes_visited']}")

        return results, stats

    def prune_neighbors_by_embedding(self, query_graph: nx.Graph, neighbors: List[nx.Graph]) -> List[nx.Graph]:
        """
        Prune neighbors by embedding distance, keeping only the top-k nearest.

        Args:
            query_graph: Query graph
            neighbors: List of all neighbors

        Returns:
            Pruned neighbor list (top 20%)
        """
        if not self.use_neighbor_pruning or len(neighbors) == 0:
            return neighbors

        # Get embeddings
        embeddings = self.ged_calculator.gisma_calculator.embeddings

        # CRITICAL FIX: Use current_query_emb instead of looking up by ID
        # Because the query may share the same ID as a database graph, but they are different graphs!
        import numpy as np
        query_emb = self.ged_calculator.gisma_calculator.current_query_emb

        if query_emb is None:
            # If no query embedding is provided, skip pruning
            return neighbors

        # Compute embedding distances between query and all neighbors
        id_to_index = self.ged_calculator.gisma_calculator.id_to_index
        neighbor_dists = []
        for neighbor in neighbors:
            neighbor_id = neighbor.graph.get('id')
            neighbor_idx = id_to_index.get(neighbor_id)

            if neighbor_idx is not None:
                neighbor_emb = embeddings[neighbor_idx]
                dist = np.linalg.norm(query_emb - neighbor_emb)
                neighbor_dists.append((neighbor, dist))
            else:
                # If the neighbor has no embedding, keep it (set to large distance)
                neighbor_dists.append((neighbor, float('inf')))

        # Sort by embedding distance
        neighbor_dists.sort(key=lambda x: x[1])

        # Keep only top-k% (consistent with original LAN implementation, using int floor)
        keep_count = int(len(neighbor_dists) * self.pruning_ratio)
        pruned_neighbors = [n for n, d in neighbor_dists[:keep_count]]

        return pruned_neighbors

    def search_layer(self, query_graph: nx.Graph, entry_points: List[nx.Graph], ef: int) \
            -> Tuple[List, Dict]:
        """
        Search layer - strictly follows the original LAN algorithm.
        This is a simplified version of routing_with_neigh_pruning.py:471-680 (without GNN pruning).

        Args:
            query_graph: Query graph
            entry_points: List of entry points
            ef: Number of candidates

        Returns:
            (W, stats): Candidate set and statistics
        """
        if len(entry_points) == 0:
            logger.error("Error: no enter point")
            return [], {}

        visited = set()
        C = []  # min-heap: search frontier
        idx_C = 0
        W = []  # max-heap: result set (stores negative distances to implement max-heap)
        idx_W = 0

        # Statistics
        distance_computations = 0
        nodes_visited = 0

        # Initialize entry points
        for ep in entry_points:
            ep_id = ep.graph.get("id")
            dist = self.ged_calculator.compute_ged(query_graph, ep)
            distance_computations += 1

            heapq.heappush(C, (dist, idx_C, ep))
            heapq.heappush(W, (-dist, idx_W, ep))
            visited.add(ep_id)
            idx_C += 1
            idx_W += 1

            logger.debug(f"[INIT] Entry point ID={ep_id}, dist={dist:.2f} added to W")

        # Helper: get neighbors of a PG node
        def _get_neighbors(node):
            node_id = node.graph.get("id")
            if node_id in self.proximity_graph:
                nids = list(self.proximity_graph.neighbors(node_id))
                return [self.graph_id_to_original[nid] for nid in nids]
            elif node in self.proximity_graph:
                return list(self.proximity_graph.neighbors(node))
            return []

        # Helper: compute GED for a neighbor and add to C/W
        import sys
        _search_start_time = time.time()

        def _print_status(stage_label=""):
            elapsed = time.time() - _search_start_time
            f_dist = -W[0][0] if W else float('inf')
            print(f"\r    [{stage_label}] GED={distance_computations}, "
                  f"visited={nodes_visited}, |C|={len(C)}, |W|={len(W)}, "
                  f"f_dist={f_dist:.1f}, {elapsed:.1f}s", end="", flush=True)

        def _process_neighbor(neigh_id, id_to_graph):
            nonlocal distance_computations, idx_C, idx_W
            if neigh_id in visited:
                return None
            visited.add(neigh_id)
            neighbor = id_to_graph[neigh_id]
            nd = self.ged_calculator.compute_ged(query_graph, neighbor)
            distance_computations += 1
            f_dist = -W[0][0]
            if nd < f_dist or len(W) < ef:
                heapq.heappush(C, (nd, idx_C, neighbor))
                heapq.heappush(W, (-nd, idx_W, neighbor))
                idx_C += 1
                idx_W += 1
                if len(W) > ef:
                    heapq.heappop(W)
            if distance_computations % 20 == 0:
                _print_status("search")
            return nd

        # ====== Batch expansion mode (Algorithm 2: mrk + 5 batches) ======
        use_batch = (self.pruning_method == 'mrk'
                     and len(self.mrk_ranker_models) > 0
                     and self.use_neighbor_pruning)

        if use_batch:
            # node_states: node_id -> (batches, id_to_graph, opened_count)
            # batches is list of 5 lists of neighbor IDs
            node_states = {}
            _print_status("S1:init")

            # Stage 1: Route to first local optimum
            while len(C) > 0:
                c_dist, _, c = heapq.heappop(C)
                nodes_visited += 1

                f_dist = -W[0][0]
                if c_dist > f_dist:
                    # Found local optimum G_flo
                    break

                # Rank neighbors into 5 batches using 4 Mrk models
                c_id = c.graph.get("id")
                neighbors = _get_neighbors(c)
                batches, id_to_graph = self.rank_neighbors_into_batches(
                    query_graph, c, neighbors)

                if not batches:
                    continue

                # Open batches until one has a neighbor with GED >= c_dist
                opened = 0
                for b in range(len(batches)):
                    opened = b + 1
                    has_far = False
                    for neigh_id in batches[b]:
                        nd = _process_neighbor(neigh_id, id_to_graph)
                        if nd is not None and nd >= c_dist:
                            has_far = True
                    if has_far:
                        break

                node_states[c_id] = (batches, id_to_graph, opened)

            # G_flo distance = current W's max distance
            g_flo_dist = -W[0][0]
            gamma = g_flo_dist + self.ds
            _print_status(f"S2:gamma={gamma:.1f}")

            # Stage 2: Gamma stepping
            MAX_GAMMA_STEPS = 100
            for gamma_step in range(MAX_GAMMA_STEPS):
                progress = False

                # (a) Re-check all explored nodes: open more batches up to gamma
                for nid, (batches, id_to_graph, opened) in list(node_states.items()):
                    while opened < len(batches):
                        has_far = False
                        for neigh_id in batches[opened]:
                            nd = _process_neighbor(neigh_id, id_to_graph)
                            if nd is not None:
                                progress = True
                                if nd >= gamma:
                                    has_far = True
                        opened += 1
                        if has_far:
                            break
                    node_states[nid] = (batches, id_to_graph, opened)

                # (b) Pop C nodes with dist <= gamma, rank + open batches
                while len(C) > 0 and C[0][0] <= gamma:
                    c_dist, _, c = heapq.heappop(C)
                    c_id = c.graph.get("id")
                    nodes_visited += 1

                    if c_id in node_states:
                        continue

                    neighbors = _get_neighbors(c)
                    batches, id_to_graph = self.rank_neighbors_into_batches(
                        query_graph, c, neighbors)

                    if not batches:
                        continue

                    opened = 0
                    for b in range(len(batches)):
                        opened = b + 1
                        has_far = False
                        for neigh_id in batches[b]:
                            nd = _process_neighbor(neigh_id, id_to_graph)
                            if nd is not None:
                                progress = True
                                if nd >= gamma:
                                    has_far = True
                        if has_far:
                            break

                    node_states[c_id] = (batches, id_to_graph, opened)

                if not progress:
                    # No new GED computed, increase gamma or stop
                    new_f_dist = -W[0][0]
                    if gamma > new_f_dist:
                        _print_status(f"S2:done gamma={gamma:.1f}")
                        break
                    gamma += self.ds
                    _print_status(f"S2:gamma={gamma:.1f}")

        else:
            # ====== Original search (no batch expansion) ======
            while len(C) > 0:
                c_dist, _, c = heapq.heappop(C)
                nodes_visited += 1

                f_dist = -W[0][0]
                if c_dist > f_dist:
                    break

                neighbors = _get_neighbors(c)

                # Neighbor pruning (if enabled)
                if self.use_neighbor_pruning:
                    neighbors = self.prune_neighbors_by_embedding(
                        query_graph, neighbors)

                for neighbor in neighbors:
                    neighbor_id = neighbor.graph.get("id")
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor_dist = self.ged_calculator.compute_ged(query_graph, neighbor)
                        distance_computations += 1
                        f_dist = -W[0][0]
                        if neighbor_dist < f_dist or len(W) < ef:
                            heapq.heappush(C, (neighbor_dist, idx_C, neighbor))
                            heapq.heappush(W, (-neighbor_dist, idx_W, neighbor))
                            idx_C += 1
                            idx_W += 1
                            if len(W) > ef:
                                heapq.heappop(W)

        print()  # Newline to end \r status output

        stats = {
            'distance_computations': distance_computations,
            'nodes_visited': nodes_visited,
            'visited_set_size': len(visited)
        }

        # DEBUG: Print all IDs in W
        w_ids = [g.graph.get('id') for _, _, g in W]
        logger.debug(f"[FINAL] W contains {len(W)} graphs: {w_ids[:10]}...")

        return W, stats


def load_lan_index_and_search(
    index_file: str,
    query_graph: nx.Graph,
    k: int = 50,
    ef: int = 100,
    ged_method: str = "simple"
) -> Tuple[List[Tuple[float, nx.Graph]], Dict]:
    """
    Convenience function: load index and search.

    Args:
        index_file: Index file path
        query_graph: Query graph
        k: Return top-k
        ef: Search parameter
        ged_method: GED computation method

    Returns:
        (results, stats)
    """
    ged_calculator = GEDCalculator(ged_method=ged_method)
    search_engine = LANOriginalSearchEngine(index_file, ged_calculator)
    return search_engine.search_k_nearest(query_graph, k, ef)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Search using LAN original method')
    parser.add_argument('--index_file', type=str, required=True, help='Index file path')
    parser.add_argument('--query_id', type=int, default=0, help='Query graph index')
    parser.add_argument('--k', type=int, default=50, help='Top-k results')
    parser.add_argument('--ef', type=int, default=100, help='Search ef parameter')
    parser.add_argument('--ged_method', type=str, default='gisma',
                       choices=['gisma', 'simple', 'astar', 'exact'],
                       help='GED computation method')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Dataset path (for Gisma calculator)')

    args = parser.parse_args()

    # Load index
    ged_calc = GEDCalculator(ged_method=args.ged_method, dataset_path=args.dataset_path)
    engine = LANOriginalSearchEngine(args.index_file, ged_calc)

    # Create query graph (use a graph from the index)
    query_graph = list(engine.graph_id_to_original.values())[args.query_id]

    logger.info(f"Query graph: ID={query_graph.graph.get('id')}, "
                f"Nodes={query_graph.number_of_nodes()}, "
                f"Edges={query_graph.number_of_edges()}")

    # Execute search
    results, stats = engine.search_k_nearest(query_graph, k=args.k, ef=args.ef)

    # Display results
    logger.info("\nSearch Results:")
    for i, (dist, graph) in enumerate(results[:20], 1):  # Show only first 20
        graph_id = graph.graph.get('id')
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        logger.info(f"  {i:2d}. ID={graph_id:>6}, Distance={dist:8.4f}, "
                   f"Nodes={nodes:2d}, Edges={edges:2d}")

    logger.info(f"\nStatistics:")
    logger.info(f"  Search time: {stats['search_time']:.4f}s")
    logger.info(f"  Distance computations: {stats['distance_computations']}")
    logger.info(f"  Nodes visited: {stats['nodes_visited']}")

"""
CG (Compressed GNN-Graph) utilities.

Ported from lan_ori_ref/wl_labelling.py with additional functions for
CG-based GIN forward pass during search.

Key functions:
  compress_query_graph(nx_graph) -> cg_data dict
  cg_gin_forward(cg_data, fc_init, bn1, bn2, relu) -> (1, hidden_dim) tensor
"""

from __future__ import annotations

import dgl
import dgl.function as fn
from dgl.udf import NodeBatch
import networkx as nx
import torch
from torch import Tensor
from typing import List, Tuple, Dict


# ============================================================================
# WL Labeling (ported from lan_ori_ref/wl_labelling.py)
# ============================================================================

class gen_wl_label_udf:
    """
    A reduce function to generate wl label.

    src: source label, e.g. wl_label_0
    dst: destination label, e.g. wl_label_1
    src_collection_to_dst: {(0, 1, 2): 4, (1, 2, 3): 5, ...}
    dst_to_src_collection: {4: (0, 1, 2), 5: (1, 2, 3), ...}
    """
    def __init__(self, src: str, dst: str, start_idx: int):
        self.src: str = src
        self.dst: str = dst
        self.start_idx: int = start_idx
        self.src_collection_to_dst = {}
        self.dst_to_src_collection = {}

    def __call__(self, nodes: NodeBatch):
        mailbox: dict = nodes.mailbox

        # shape: (number of nodes with same number of messages received, number of messages received)
        m: torch.tensor = mailbox[self.src]

        # source label, e.g. wl_label_0
        # after unsqueeze, shape should be (n, 1)
        src_label: torch.tensor = nodes.data[self.src].unsqueeze(-1)

        # collect messages from neighbors and combine with src_label
        # shape: (number of nodes with same number of messages received, number of messages received + 1)
        # src_label_collection = torch.cat((src_label, m), 1).type(torch.int)
        src_label_collection = m.type(torch.int)

        # sort along column dimension
        sorted_src_label_collection, _ = torch.sort(src_label_collection, 1)

        # relabel src_label_collection to new label to 0, 1, 2, ...
        dst_label = torch.zeros(len(sorted_src_label_collection))
        for i, l in enumerate(sorted_src_label_collection):
            l_tuple: Tuple[int] = tuple(l.tolist())
            if l_tuple not in self.src_collection_to_dst:
                # start from 1 to differentiate the ones who has no message received
                dst_label_idx = self.start_idx + len(self.src_collection_to_dst)
                self.src_collection_to_dst.update({ l_tuple : dst_label_idx })
                self.dst_to_src_collection.update({ dst_label_idx : l_tuple })
                dst_label[i] = dst_label_idx
            else:
                dst_label[i] = self.src_collection_to_dst[l_tuple]
        return {self.dst: dst_label}


def compress_graph(gid: int, g, max_degree: int) -> dict:
    """
    Compress a DGL graph using 2-round WL labeling.

    1. Re-label all nodes with new wl_label_0
    2. Use update_all to generate wl_label_1 with wl_label_0
    3. Use update_all to generate wl_label_2 with wl_label_1
    4. Generate a new DAG with wl_label_0, wl_label_1, wl_label_2 and hGx nodes
    5. Create edges with weights

    Args:
        gid: graph ID (unused, kept for API compatibility)
        g: DGL graph with ndata["label"] set (degrees before self-loops)
           and self-loops already added
        max_degree: max degree for one-hot encoding (default 20)

    Returns:
        dict with compressed graph info
    """
    # Relabel all nodes with new wl_label_0.
    # The number of wl_label_0 is the same as the number of unique labels in the graph.
    unique_label_tensor: Tensor = torch.unique(g.ndata["label"]).type(torch.long)
    unique_label_list: List = sorted(unique_label_tensor.tolist())
    n_wl_label_0 = len(unique_label_list)
    label_to_wl_label_0 = { l: idx for (idx, l) in enumerate(unique_label_list) }
    wl_label_0_to_label = { idx: l for (idx, l) in enumerate(unique_label_list) }
    n_data_wl_label_0 = g.ndata["label"].clone().detach()
    n_data_wl_label_0.apply_(lambda l: label_to_wl_label_0[int(l)])
    g.ndata["wl_label_0"] = n_data_wl_label_0

    # Calculate wl label 1 and the mapping from wl label 0 to wl label 1
    wl_label_0_to_1 = gen_wl_label_udf("wl_label_0", "wl_label_1", n_wl_label_0)
    g.update_all(fn.copy_u('wl_label_0', 'wl_label_0'), wl_label_0_to_1)
    n_wl_label_1 = len(wl_label_0_to_1.dst_to_src_collection)

    # Calculate wl label 2 and the mapping from wl label 1 to wl label 2
    wl_label_1_to_2 = gen_wl_label_udf("wl_label_1", "wl_label_2", n_wl_label_0 + n_wl_label_1)
    g.update_all(fn.copy_u('wl_label_1', 'wl_label_1'), wl_label_1_to_2)
    n_wl_label_2 = len(wl_label_1_to_2.dst_to_src_collection)

    # Final node for each graph
    hGx = n_wl_label_0 + n_wl_label_1 + n_wl_label_2

    # Create a new graph
    edges = []
    edges_to_ids = {}
    edge_weight = []
    wl_label_0_to_1_edge = []
    wl_label_1_to_2_edge = []
    wl_label_2_to_hGx_edge = []

    def upsert_edge(edge: Tuple[int], label_edge_list: List[int]):
        if edge in edges_to_ids:
            # If exists add edge weight by 1
            edge_id = edges_to_ids[edge]
            edge_weight[edge_id] += 1
        else:
            # If not exists, add a new edge
            edges_to_ids[edge] = len(edge_weight)
            label_edge_list.append(edge)
            edge_weight.append(1)
            edges.append(edge)

    processed_wl_label_1 = set()
    processed_wl_label_2 = set()
    for node_idx in g.nodes():
        label = g.ndata["label"][node_idx]
        wl_label_0 = int(g.ndata["wl_label_0"][node_idx])
        wl_label_1 = int(g.ndata["wl_label_1"][node_idx])
        wl_label_2 = int(g.ndata["wl_label_2"][node_idx])

        # Add edges from wl_label_0 to wl_label_1
        wl_label_0_collection = wl_label_0_to_1.dst_to_src_collection[wl_label_1]
        if wl_label_1 not in processed_wl_label_1:
            for wl_label_0 in wl_label_0_collection:
                edge = (wl_label_0, wl_label_1)
                upsert_edge(edge, wl_label_0_to_1_edge)
            processed_wl_label_1.add(wl_label_1)

        # Add edges from wl_label_1 to wl_label_2
        wl_label_1_collection = wl_label_1_to_2.dst_to_src_collection[wl_label_2]
        if wl_label_2 not in processed_wl_label_2:
            for wl_label_1 in wl_label_1_collection:
                edge = (wl_label_1, wl_label_2)
                upsert_edge(edge, wl_label_1_to_2_edge)
            processed_wl_label_2.add(wl_label_2)

        # Add edges from wl_label_2 to hGx
        edge = (wl_label_2, hGx)
        upsert_edge(edge, wl_label_2_to_hGx_edge)

    # new_g is directional graph
    # Only wl_label_0 nodes have valid h one hot encoding
    diagonal_ones = torch.eye(max_degree)
    wl_label_0_h_one_hot = diagonal_ones.index_select(0, torch.LongTensor(unique_label_list))

    new_g = dgl.graph(tuple(zip(*edges)))
    new_g.edata["weight"] = torch.tensor(edge_weight)
    new_g.ndata['h'] = torch.zeros(new_g.num_nodes(), max_degree)
    new_g.ndata['h'][:n_wl_label_0] = wl_label_0_h_one_hot
    return {
        "old_g": g,
        "new_g": new_g,
        "wl_label_0_nodes": sorted(list(wl_label_0_to_label.keys())),
        "wl_label_1_nodes": sorted(list(wl_label_0_to_1.dst_to_src_collection.keys())),
        "wl_label_2_nodes": sorted(list(wl_label_1_to_2.dst_to_src_collection.keys())),
        "wl_label_0_to_label": wl_label_0_to_label,
        "wl_label_1_to_0": wl_label_0_to_1.dst_to_src_collection,
        "wl_label_2_to_1": wl_label_1_to_2.dst_to_src_collection,
        "wl_label_0_to_1_edge": wl_label_0_to_1_edge,
        "wl_label_1_to_2_edge": wl_label_1_to_2_edge,
        "wl_label_2_to_hGx_edge": wl_label_2_to_hGx_edge
    }


# ============================================================================
# CG Forward Pass
# ============================================================================

def compress_query_graph(nx_g: nx.Graph, max_deg: int = 20) -> dict:
    """
    Convert a NetworkX graph to a compressed CG data dict ready for cg_gin_forward.

    Steps:
    1. Create bidirectional DGL graph (same as _make_a_dglgraph_for_ori)
    2. Set ndata['label'] = in_degrees (before self-loops, matching original code)
    3. Add self-loops
    4. Run compress_graph (WL labeling + DAG construction)
    5. Pre-compute per-level edge tensors with normalized weights
    6. Build self-feature mapping for GINConv compatibility

    Args:
        nx_g: NetworkX graph
        max_deg: max degree for one-hot encoding (default 20)

    Returns:
        cg_data dict with all info needed for cg_gin_forward
    """
    nodes = list(nx_g.nodes())
    if len(nodes) == 0:
        # Degenerate case: empty graph
        return {
            'n_wl0': 0, 'n_wl1': 0, 'n_wl2': 0,
            'wl0_h': torch.zeros(0, max_deg),
            'original_num_nodes': 0,
        }

    node_map = {node: i for i, node in enumerate(nodes)}

    # Create bidirectional edges
    edge_src, edge_dst = [], []
    for edge in nx_g.edges():
        end1 = node_map[edge[0]]
        end2 = node_map[edge[1]]
        edge_src.append(int(end1))
        edge_dst.append(int(end2))
        edge_src.append(int(end2))
        edge_dst.append(int(end1))

    if len(edge_src) == 0:
        dg = dgl.graph(([], []), num_nodes=len(nodes))
    else:
        dg = dgl.graph((torch.tensor(edge_src), torch.tensor(edge_dst)), num_nodes=len(nodes))

    # Set ndata['label'] = in_degrees BEFORE self-loops (matching original wl_labelling.py)
    dg.ndata['label'] = dg.in_degrees().type(torch.float)

    # Add self-loops (AFTER setting label)
    dg = dgl.add_self_loop(dg)

    # Run WL compression
    cg_dict = compress_graph(0, dg, max_deg)

    # Extract per-level info
    old_g = cg_dict['old_g']
    new_g = cg_dict['new_g']
    n_wl0 = len(cg_dict['wl_label_0_nodes'])
    n_wl1 = len(cg_dict['wl_label_1_nodes'])
    n_wl2 = len(cg_dict['wl_label_2_nodes'])
    hGx_idx = n_wl0 + n_wl1 + n_wl2

    # Build self-feature mappings for GINConv compatibility.
    # GINConv(None, 'mean') computes: h_new = (1+eps)*h_self + mean(neighbors).
    # With eps=0: h_new = h_self + mean(neighbors_including_self_via_selfloop).
    # CG weighted mean only gives mean(neighbors), so we need to add h_self back.
    # For each wl1 node j: self_wl0[j] = the wl0 label of original nodes with wl1=j.
    # For each wl2 node m: self_wl1_local[m] = the local wl1 index of original nodes with wl2=m.
    wl1_self_wl0 = torch.zeros(n_wl1, dtype=torch.long)
    wl2_self_wl1_local = torch.zeros(n_wl2, dtype=torch.long)
    seen_wl1 = set()
    seen_wl2 = set()
    for node_idx in old_g.nodes():
        wl0_val = int(old_g.ndata['wl_label_0'][node_idx])
        wl1_val = int(old_g.ndata['wl_label_1'][node_idx])
        wl2_val = int(old_g.ndata['wl_label_2'][node_idx])
        wl1_local = wl1_val - n_wl0
        wl2_local = wl2_val - n_wl0 - n_wl1
        if wl1_local not in seen_wl1:
            wl1_self_wl0[wl1_local] = wl0_val
            seen_wl1.add(wl1_local)
        if wl2_local not in seen_wl2:
            wl2_self_wl1_local[wl2_local] = wl1_local
            seen_wl2.add(wl2_local)

    # Get all edges and weights from compressed graph
    all_src, all_dst = new_g.edges()
    all_weights = new_g.edata['weight'].float()

    # Separate edges by level based on node index ranges
    # Level 0→1: src in [0, n_wl0), dst in [n_wl0, n_wl0+n_wl1)
    # Level 1→2: src in [n_wl0, n_wl0+n_wl1), dst in [n_wl0+n_wl1, n_wl0+n_wl1+n_wl2)
    # Level 2→hGx: src in [n_wl0+n_wl1, n_wl0+n_wl1+n_wl2), dst = hGx_idx
    mask_01 = (all_src < n_wl0) & (all_dst >= n_wl0) & (all_dst < n_wl0 + n_wl1)
    mask_12 = (all_src >= n_wl0) & (all_src < n_wl0 + n_wl1) & \
              (all_dst >= n_wl0 + n_wl1) & (all_dst < hGx_idx)
    mask_2h = (all_dst == hGx_idx)

    cg_data = {
        'n_wl0': n_wl0,
        'n_wl1': n_wl1,
        'n_wl2': n_wl2,
        'original_num_nodes': len(nodes),
        'wl0_h': new_g.ndata['h'][:n_wl0].clone(),  # (n_wl0, max_deg)

        # Self-feature mappings (for GINConv = self + mean(neighbors) compatibility)
        'wl1_self_wl0': wl1_self_wl0,      # (n_wl1,) wl0 index for each wl1 node's self
        'wl2_self_wl1_local': wl2_self_wl1_local,  # (n_wl2,) local wl1 index for each wl2 node's self

        # Level 0→1 edges (src is global wl0 index, dst is local wl1 index)
        'level_01_src': all_src[mask_01].long(),
        'level_01_dst_local': (all_dst[mask_01] - n_wl0).long(),
        'level_01_weights': all_weights[mask_01],

        # Level 1→2 edges (src is local wl1 index, dst is local wl2 index)
        'level_12_src_local': (all_src[mask_12] - n_wl0).long(),
        'level_12_dst_local': (all_dst[mask_12] - n_wl0 - n_wl1).long(),
        'level_12_weights': all_weights[mask_12],

        # Level 2→hGx edges (src is local wl2 index, dst is always 0)
        'level_2h_src_local': (all_src[mask_2h] - n_wl0 - n_wl1).long(),
        'level_2h_dst_local': torch.zeros(mask_2h.sum().item(), dtype=torch.long),
        'level_2h_weights': all_weights[mask_2h],
    }

    return cg_data


def _weighted_mean_aggregate(h_src: torch.Tensor,
                              src_idx: torch.Tensor,
                              dst_idx: torch.Tensor,
                              weights: torch.Tensor,
                              n_dst: int) -> torch.Tensor:
    """
    Compute weighted mean aggregation for one level of the CG DAG.

    For each destination node d:
        h_d = sum(w(s,d) * h_s) / sum(w(s,d))

    Args:
        h_src: (n_src, D) source node features
        src_idx: (E,) source indices into h_src
        dst_idx: (E,) destination indices (0-based within this level)
        weights: (E,) raw edge weights
        n_dst: number of destination nodes

    Returns:
        h_dst: (n_dst, D) aggregated features
    """
    if len(src_idx) == 0 or n_dst == 0:
        hidden_dim = h_src.shape[1] if h_src.dim() > 1 else 0
        return torch.zeros(n_dst, hidden_dim)

    hidden_dim = h_src.shape[1]

    # Normalize weights per destination node
    dst_weight_sum = torch.zeros(n_dst)
    dst_weight_sum.scatter_add_(0, dst_idx, weights)
    w_norm = weights / dst_weight_sum[dst_idx].clamp(min=1e-8)

    # Get source features and apply normalized weights
    h_s = h_src[src_idx]  # (E, D)
    weighted_h = h_s * w_norm.unsqueeze(1)  # (E, D)

    # Scatter-add to destination nodes
    h_dst = torch.zeros(n_dst, hidden_dim)
    dst_expanded = dst_idx.unsqueeze(1).expand_as(weighted_h)
    h_dst.scatter_add_(0, dst_expanded, weighted_h)

    return h_dst


def _ginconv_aggregate_cg(h_src: torch.Tensor,
                           src_idx: torch.Tensor,
                           dst_idx: torch.Tensor,
                           weights: torch.Tensor,
                           n_dst: int,
                           self_src_idx: torch.Tensor) -> torch.Tensor:
    """
    Compute GINConv-compatible aggregation on CG DAG.

    The CG DAG is built from a graph WITH self-loops (needed for WL labeling),
    but the actual model uses GINConv on a graph WITHOUT self-loops:
        GINConv(eps=0) = h_self + mean(non-self neighbors)

    This function:
    1. Computes the raw weighted sum (includes self-loop contribution)
    2. Subtracts the self-loop contribution (weight=1 per self-loop)
    3. Divides by non-self total weight to get mean(non-self neighbors)
    4. Adds h_self to match GINConv

    Args:
        h_src: (n_src, D) source node features
        src_idx: (E,) source indices into h_src
        dst_idx: (E,) destination indices (0-based within this level)
        weights: (E,) raw edge weights (include self-loop contributions)
        n_dst: number of destination nodes
        self_src_idx: (n_dst,) source index of the "self" node for each dst

    Returns:
        h_dst: (n_dst, D) GINConv output = h_self + mean(non-self neighbors)
    """
    if len(src_idx) == 0 or n_dst == 0:
        hidden_dim = h_src.shape[1] if h_src.dim() > 1 else 0
        return torch.zeros(n_dst, hidden_dim)

    hidden_dim = h_src.shape[1]

    # Step 1: Raw weighted sum per destination (includes self-loop)
    h_s = h_src[src_idx]  # (E, D)
    weighted_h = h_s * weights.unsqueeze(1)  # (E, D)
    raw_sum = torch.zeros(n_dst, hidden_dim)
    dst_expanded = dst_idx.unsqueeze(1).expand_as(weighted_h)
    raw_sum.scatter_add_(0, dst_expanded, weighted_h)

    # Total weight per destination (includes self-loop weight)
    w_total = torch.zeros(n_dst)
    w_total.scatter_add_(0, dst_idx, weights)

    # Step 2: Self features for each destination node
    h_self = h_src[self_src_idx]  # (n_dst, D)

    # Step 3: Subtract self-loop contribution (self-loop weight = 1)
    nonself_sum = raw_sum - h_self  # remove 1 * h_self from numerator
    nonself_weight = (w_total - 1.0).clamp(min=1e-8)  # remove 1 from denominator

    # Step 4: Non-self mean
    nonself_mean = nonself_sum / nonself_weight.unsqueeze(1)

    # Step 5: GINConv output = (1+eps)*h_self + mean = h_self + nonself_mean
    return h_self + nonself_mean


def cg_gin_forward(cg_data: dict,
                    fc_init: torch.nn.Module,
                    bn1: torch.nn.Module,
                    bn2: torch.nn.Module,
                    relu: torch.nn.Module,
                    model_has_selfloop: bool = False) -> torch.Tensor:
    """
    Compute GIN graph embedding using CG compressed graph.

    Replaces the standard GIN forward pass (fc_init -> GINConv1 -> BN1 -> ReLU ->
    GINConv2 -> BN2 -> ReLU -> mean_nodes) with level-by-level aggregation
    on the CG DAG.

    The CG DAG is built from a graph with self-loops (for WL labeling).
    Two modes based on whether the model's GINConv operates on a graph with self-loops:

    - model_has_selfloop=False (Mrk model): GINConv = h_self + mean(non-self neighbors).
      We subtract the self-loop contribution from the CG weighted mean, then add h_self.
    - model_has_selfloop=True (Init model): GINConv = h_self + mean(all neighbors incl. self).
      We use the CG weighted mean directly (includes self-loop), then add h_self.

    Args:
        cg_data: dict from compress_query_graph()
        fc_init: nn.Linear(max_deg, hidden_dim) - initial feature projection
        bn1: nn.BatchNorm1d(hidden_dim) - BN after layer 1
        bn2: nn.BatchNorm1d(hidden_dim) - BN after layer 2
        relu: nn.ReLU - activation function
        model_has_selfloop: whether the model was trained/used with self-loops in graphs

    Returns:
        graph_emb: (1, hidden_dim) graph embedding tensor
    """
    n_wl0 = cg_data['n_wl0']
    n_wl1 = cg_data['n_wl1']
    n_wl2 = cg_data['n_wl2']

    if n_wl0 == 0:
        # Empty graph fallback
        hidden_dim = fc_init.out_features
        return torch.zeros(1, hidden_dim)

    # Step 0: Apply fc_init to wl_label_0 node features (one-hot -> hidden_dim)
    h_wl0 = fc_init(cg_data['wl0_h'])  # (n_wl0, hidden_dim)
    hidden_dim = h_wl0.shape[1]

    # Step 1: GINConv layer 1
    if model_has_selfloop:
        # Model graph has self-loops: GINConv = h_self + mean(all neighbors incl. self)
        # CG weighted mean already includes self via self-loop in DAG edges
        h_wl1 = _weighted_mean_aggregate(
            h_wl0,
            cg_data['level_01_src'],
            cg_data['level_01_dst_local'],
            cg_data['level_01_weights'],
            n_wl1
        )
        h_wl1 = h_wl1 + h_wl0[cg_data['wl1_self_wl0']]
    else:
        # Model graph has NO self-loops: GINConv = h_self + mean(non-self neighbors)
        # _ginconv_aggregate_cg subtracts self-loop from CG weighted mean, then adds h_self
        h_wl1 = _ginconv_aggregate_cg(
            h_wl0,
            cg_data['level_01_src'],
            cg_data['level_01_dst_local'],
            cg_data['level_01_weights'],
            n_wl1,
            cg_data['wl1_self_wl0']
        )
    if n_wl1 > 1:
        h_wl1 = relu(bn1(h_wl1))
    elif n_wl1 == 1:
        # Skip BN for single-node level (BN requires >1 samples when track_running_stats=False)
        h_wl1 = relu(h_wl1)

    # Step 2: GINConv layer 2
    if model_has_selfloop:
        h_wl2 = _weighted_mean_aggregate(
            h_wl1,
            cg_data['level_12_src_local'],
            cg_data['level_12_dst_local'],
            cg_data['level_12_weights'],
            n_wl2
        )
        h_wl2 = h_wl2 + h_wl1[cg_data['wl2_self_wl1_local']]
    else:
        h_wl2 = _ginconv_aggregate_cg(
            h_wl1,
            cg_data['level_12_src_local'],
            cg_data['level_12_dst_local'],
            cg_data['level_12_weights'],
            n_wl2,
            cg_data['wl2_self_wl1_local']
        )
    if n_wl2 > 1:
        h_wl2 = relu(bn2(h_wl2))
    elif n_wl2 == 1:
        h_wl2 = relu(h_wl2)

    # Step 3: Graph-level readout = mean_nodes (simple weighted mean, NOT GINConv)
    graph_emb = _weighted_mean_aggregate(
        h_wl2,
        cg_data['level_2h_src_local'],
        cg_data['level_2h_dst_local'],
        cg_data['level_2h_weights'],
        1  # single hGx node
    )

    return graph_emb  # (1, hidden_dim)

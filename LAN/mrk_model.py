#!/usr/bin/env python3
"""
Mrk Model - Neighbor Ranking Model for LAN
Following the LAN_ori code architecture.

Architecture (no cross-graph attention):
- Q: degree one-hot (dim=20) -> GIN 2 layers -> mean pooling -> q_emb
- G (PG node): pre-computed Node2Vec embedding (dim=emb_dim)
- G' (neighbors): pre-computed Node2Vec embeddings (dim=emb_dim)
- concat(q_emb, g_emb, g'_emb) -> 3-layer MLP(256) -> sigmoid

4 binary rankers: M_rk^i predicts top i*20% neighbors (i=1..4)
M_rk^5 (top 100%) is trivial, handled as residual batch.
Shared backbone (GIN + MLP layers 1-3), separate output heads (fc4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import networkx as nx
from dgl.nn.pytorch.conv import GINConv


MAX_DEGREE = 20  # max degree for one-hot node features


class MrkModel(nn.Module):
    """
    Mrk - Neighbor Ranking Model (following LAN_ori code).

    4 binary rankers sharing a GIN backbone + MLP.
    """

    def __init__(self, num_rankers=4, emb_dim=512, num_layers=2):
        super().__init__()
        self.num_rankers = num_rankers
        self.emb_dim = emb_dim

        # GIN for query Q (degree one-hot -> hidden dim)
        self.fc_init = nn.Linear(MAX_DEGREE, emb_dim)
        self.gin_layers = nn.ModuleList()
        self.gin_bns = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINConv(None, 'mean'))
            self.gin_bns.append(nn.BatchNorm1d(emb_dim))

        self.relu = nn.ReLU(inplace=True)

        # Shared MLP layers: concat(q_emb, g_emb, g'_emb) -> 256
        self.fc1 = nn.Linear(emb_dim * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)

        # Separate output heads for each ranker
        self.heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_rankers)
        ])

    def encode_query(self, q_dgl):
        """
        Encode query graph Q through GIN.

        Args:
            q_dgl: DGL graph (possibly batched) with degree one-hot in ndata['h']

        Returns:
            q_emb: [batch_size, emb_dim]
        """
        h = self.fc_init(q_dgl.ndata['h'])
        for gin, bn in zip(self.gin_layers, self.gin_bns):
            h = self.relu(bn(gin(q_dgl, h)))
        q_dgl.ndata['_out'] = h
        q_emb = dgl.mean_nodes(q_dgl, '_out')
        return q_emb

    def forward(self, q_dgl, g_emb, neigh_embs):
        """
        Forward pass for a single training instance.

        Args:
            q_dgl: DGL graph for query Q (degree one-hot features)
            g_emb: [emb_dim] Node2Vec embedding of PG node G
            neigh_embs: [k, emb_dim] Node2Vec embeddings of G's neighbors

        Returns:
            logits: [k, num_rankers] raw logits (before sigmoid)
        """
        k = neigh_embs.size(0)

        # Encode Q
        q_emb = self.encode_query(q_dgl)  # [1, emb_dim]

        # Expand q_emb and g_emb for each neighbor
        qg = torch.cat([q_emb, g_emb.unsqueeze(0)], dim=1)  # [1, emb_dim*2]
        qg = qg.expand(k, -1)  # [k, emb_dim*2]

        # Concat with neighbor embeddings
        x = torch.cat([qg, neigh_embs], dim=1)  # [k, emb_dim*3]

        # Shared MLP
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))

        # Per-ranker output heads
        logits = torch.cat([head(x) for head in self.heads], dim=1)  # [k, num_rankers]
        return logits

    def forward_batch(self, q_dgls, g_embs, neigh_embs_list, neigh_counts):
        """
        Batched forward pass for multiple training instances.

        Args:
            q_dgls: batched DGL graph (multiple Q graphs batched together)
            g_embs: [batch_size, emb_dim] PG node embeddings
            neigh_embs_list: [total_neighbors, emb_dim] all neighbor embeddings concatenated
            neigh_counts: [batch_size] number of neighbors per instance

        Returns:
            all_logits: [total_neighbors, num_rankers]
        """
        # Encode all Q graphs at once
        q_embs = self.encode_query(q_dgls)  # [batch_size, emb_dim]

        # Expand q_emb and g_emb for each neighbor
        qg = torch.cat([q_embs, g_embs], dim=1)  # [batch_size, emb_dim*2]

        # Repeat qg for each neighbor
        qg_expanded = torch.repeat_interleave(qg, neigh_counts, dim=0)  # [total_neighbors, emb_dim*2]

        # Concat with neighbor embeddings
        x = torch.cat([qg_expanded, neigh_embs_list], dim=1)  # [total_neighbors, emb_dim*3]

        # Shared MLP
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))

        # Per-ranker output heads
        logits = torch.cat([head(x) for head in self.heads], dim=1)  # [total_neighbors, num_rankers]
        return logits

    def rank_neighbors(self, q_dgl, g_emb, neigh_embs):
        """
        Rank neighbors into 5 batches using the cascade of rankers.

        Args:
            q_dgl: DGL graph for query Q
            g_emb: [emb_dim] Node2Vec embedding of PG node G
            neigh_embs: [k, emb_dim] Node2Vec embeddings of G's neighbors

        Returns:
            batches: list of 5 lists, each containing indices into neighbors
                     B_0 (most promising) to B_4 (least promising)
        """
        k = neigh_embs.size(0)
        if k == 0:
            return [[] for _ in range(5)]

        with torch.no_grad():
            self.eval()
            # Disable running stats for BN during inference
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.track_running_stats = False

            logits = self.forward(q_dgl, g_emb, neigh_embs)  # [k, num_rankers]
            probs = torch.sigmoid(logits)  # [k, num_rankers]

            # Re-enable running stats
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.track_running_stats = True

        # Build batches via set difference
        pos = [set() for _ in range(self.num_rankers)]
        for i in range(self.num_rankers):
            for j in range(k):
                if probs[j, i].item() > 0.5:
                    pos[i].add(j)

        batches = [[] for _ in range(5)]
        batches[0] = sorted(pos[0])
        for i in range(1, self.num_rankers):
            batches[i] = sorted(pos[i] - pos[i - 1])

        assigned = set()
        for b in batches[:4]:
            assigned.update(b)
        batches[4] = sorted(j for j in range(k) if j not in assigned)

        return batches


# === Utility functions ===

def nx_to_dgl(g: nx.Graph) -> dgl.DGLGraph:
    """
    Convert a networkx graph to a DGL graph with degree one-hot node features.
    Adds self-loops for GIN aggregation.

    Args:
        g: networkx graph

    Returns:
        DGL graph with ndata['h'] as degree one-hot features (dim=MAX_DEGREE)
    """
    if g.number_of_nodes() == 0:
        dg = dgl.graph(([], []))
        dg.ndata['h'] = torch.zeros(0, MAX_DEGREE, dtype=torch.float32)
        return dg

    nodes = sorted(g.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build directed edge list (both directions for undirected graph)
    src, dst = [], []
    for u, v in g.edges():
        src.extend([node_to_idx[u], node_to_idx[v]])
        dst.extend([node_to_idx[v], node_to_idx[u]])

    dg = dgl.graph((src, dst), num_nodes=len(nodes))

    # Degree one-hot features (like LAN_ori make_a_dglgraph)
    ones = torch.eye(MAX_DEGREE)
    degrees = dg.in_degrees().clamp(0, MAX_DEGREE - 1)
    h = ones.index_select(0, degrees).float()
    dg.ndata['h'] = h

    # Self-loops for GIN
    dg = dgl.add_self_loop(dg)

    return dg

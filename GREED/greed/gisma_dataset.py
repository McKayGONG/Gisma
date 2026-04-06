"""
Gisma data adapter - converts Gisma's ged_results.txt to a format usable by the original GREED
Maintains a fully compatible interface with the original GREED; only the data source differs
"""
import torch
import torch_geometric as tg
import torch_geometric.data
from tqdm.auto import tqdm
import pandas as pd
import os

def load_graph_database(db_file):
    """
    Load graph database file, returning graph list and ID mappings

    Args:
        db_file: Path to graph database file (db.txt format)

    Returns:
        graphs: List of PyTorch Geometric graph objects
        graph_id_to_index: Mapping from graph ID to list index
        index_to_graph_id: Mapping from list index to graph ID
    """
    graphs = []
    graph_id_to_index = {}
    index_to_graph_id = {}
    all_node_features = []  # Collect all node features to determine unified dimension
    
    current_graph_id = None
    node_list = []
    edge_list = []
    node_features = []
    graph_data = []  # Temporary storage for all graph data
    
    print(f"Loading graph database from {db_file}")
    
    # First pass: read all data and collect node labels
    with open(db_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading db.txt")):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            if parts[0] == 't':
                # Save previous graph data
                if current_graph_id is not None:
                    if len(node_list) > 0:
                        graph_data.append((current_graph_id, node_features[:], edge_list[:]))
                        all_node_features.extend(node_features)

                # Start new graph
                current_graph_id = int(parts[2])
                node_list = []
                edge_list = []
                node_features = []

            elif parts[0] == 'ID':
                # Save previous graph data (Gisma format)
                if current_graph_id is not None:
                    if len(node_list) > 0:
                        graph_data.append((current_graph_id, node_features[:], edge_list[:]))
                        all_node_features.extend(node_features)

                # Start new graph
                current_graph_id = int(parts[1])
                node_list = []
                edge_list = []
                node_features = []
                
            elif parts[0] == 'v':
                node_id = int(parts[1])
                node_label = int(parts[2])
                node_list.append(node_id)
                node_features.append(node_label)
                
            elif parts[0] == 'e':
                src = int(parts[1])
                tgt = int(parts[2])
                edge_list.append([src, tgt])
    
    # Save the last graph data
    if current_graph_id is not None and len(node_list) > 0:
        graph_data.append((current_graph_id, node_features[:], edge_list[:]))
        all_node_features.extend(node_features)
    
    # Determine unified node feature dimension
    max_label = max(all_node_features) if all_node_features else 0
    print(f"Max node label: {max_label}, feature dimension: {max_label + 1}")
    
    # Second pass: create graph objects using unified dimension
    print("Creating PyTorch Geometric graphs...")
    for graph_id, node_features, edge_list in tqdm(graph_data, desc="Creating graphs"):
        graph = create_pyg_graph(node_features, edge_list, max_label)
        graphs.append(graph)
        graph_id_to_index[graph_id] = len(graphs) - 1
        index_to_graph_id[len(graphs) - 1] = graph_id
    
    print(f"Loaded {len(graphs)} graphs from database")
    return graphs, graph_id_to_index, index_to_graph_id

def create_pyg_graph(node_features, edge_list, max_label=None):
    """Create a PyTorch Geometric graph object"""
    # Remap node IDs to 0-based contiguous indices
    unique_nodes = sorted(list(set([n for n in range(len(node_features))])))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    # Node features (one-hot encoding) - use global max label dimension
    if max_label is None:
        max_label = max(node_features) + 1
    else:
        max_label = max_label + 1
        
    x = torch.zeros(len(unique_nodes), max_label)
    for i, label in enumerate(node_features):
        if label < max_label:
            x[i, label] = 1.0
    
    # Edge index (remap node IDs)
    if len(edge_list) > 0:
        edge_index = []
        for src, tgt in edge_list:
            if src < len(node_features) and tgt < len(node_features):
                edge_index.append([src, tgt])
                edge_index.append([tgt, src])  # Undirected graph
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return tg.data.Data(x=x, edge_index=edge_index)

def load_ged_results(ged_file):
    """
    Load the ged_results.txt file generated by Gisma

    Args:
        ged_file: Path to ged_results.txt file

    Returns:
        pandas DataFrame with columns: graph_id1, graph_id2, ged
    """
    print(f"Loading GED results from {ged_file}")
    
    data = []
    with open(ged_file, 'r') as f:
        for line in tqdm(f, desc="Reading GED results"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 3:
                graph_id1 = int(parts[0])
                graph_id2 = int(parts[1])
                ged_value = float(parts[2])
                data.append({
                    'graph_id1': graph_id1,
                    'graph_id2': graph_id2, 
                    'ged': ged_value
                })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} GED pairs")
    return df

def create_greed_dataset(db_file, ged_file, max_pairs=None, random_sample=False, seed=42):
    """
    Create a training dataset in GREED format

    Args:
        db_file: Graph database file
        ged_file: GED results file
        max_pairs: Maximum number of training pairs (None for all)
        random_sample: Whether to randomly sample (False selects the first N in order)
        seed: Random seed

    Returns:
        queries: List of query graphs
        targets: List of target graphs
        lb: Lower bound (equal to exact GED)
        ub: Upper bound (equal to exact GED)
    """
    # Load graph data
    graphs, graph_id_to_index, index_to_graph_id = load_graph_database(db_file)
    
    # Load GED data
    ged_df = load_ged_results(ged_file)
    
    if max_pairs is not None and max_pairs < len(ged_df):
        if random_sample:
            # Random sampling
            print(f"Randomly sampling {max_pairs} pairs from {len(ged_df)} total pairs (seed={seed})")
            ged_df = ged_df.sample(n=max_pairs, random_state=seed)
            print(f"Sampled {len(ged_df)} pairs for training")
        else:
            # Select the first N in order
            ged_df = ged_df.head(max_pairs)
            print(f"Selected first {len(ged_df)} pairs for training")
    
    # Convert to GREED format
    queries = []
    targets = []
    geds = []
    
    print("Converting to GREED format...")
    print(f"Graph database has {len(graphs)} graphs (indices 0-{len(graphs)-1})")
    
    valid_pairs = 0
    for _, row in tqdm(ged_df.iterrows(), total=len(ged_df), desc="Processing pairs"):
        graph_id1 = int(row['graph_id1'])
        graph_id2 = int(row['graph_id2'])
        ged_value = float(row['ged'])
        
        # Assume ged_results.txt uses array indices (0-based) rather than original graph IDs
        # Check if indices are within valid range
        if 0 <= graph_id1 < len(graphs) and 0 <= graph_id2 < len(graphs):
            queries.append(graphs[graph_id1])
            targets.append(graphs[graph_id2])
            geds.append(ged_value)
            valid_pairs += 1
        else:
            # Fall back to original ID matching (if index is out of range)
            if graph_id1 in graph_id_to_index and graph_id2 in graph_id_to_index:
                idx1 = graph_id_to_index[graph_id1]
                idx2 = graph_id_to_index[graph_id2]
                
                queries.append(graphs[idx1])
                targets.append(graphs[idx2])
                geds.append(ged_value)
                valid_pairs += 1
    
    print(f"Created dataset with {len(queries)} valid pairs")
    
    # Convert to tensors
    lb = torch.tensor(geds, dtype=torch.float32)
    ub = torch.tensor(geds, dtype=torch.float32)  # lb = ub = exact GED
    
    return queries, targets, lb, ub

def create_dataloader(queries, targets, lb, ub, batch_size=32, shuffle=True):
    """
    Create a data loader compatible with the original GREED

    Returns:
        DataLoader that yields (batch_queries, batch_targets, batch_lb, batch_ub)
    """
    from torch.utils.data import Dataset, DataLoader
    
    class GEDDataset(Dataset):
        def __init__(self, queries, targets, lb, ub):
            self.queries = queries
            self.targets = targets
            self.lb = lb
            self.ub = ub
            
        def __len__(self):
            return len(self.queries)
            
        def __getitem__(self, idx):
            return self.queries[idx], self.targets[idx], self.lb[idx], self.ub[idx]
    
    dataset = GEDDataset(queries, targets, lb, ub)
    
    def collate_fn(batch):
        queries, targets, lbs, ubs = zip(*batch)
        
        # Batch graph data
        batch_queries = tg.data.Batch.from_data_list(list(queries))
        batch_targets = tg.data.Batch.from_data_list(list(targets))
        
        # Batch scalar data
        batch_lb = torch.stack(list(lbs))
        batch_ub = torch.stack(list(ubs))
        
        return batch_queries, batch_targets, batch_lb, batch_ub
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    """Collate function for batching graph pairs"""
    queries, targets, lbs, ubs = zip(*batch)
    
    # Batch graph data
    batch_queries = tg.data.Batch.from_data_list(list(queries))
    batch_targets = tg.data.Batch.from_data_list(list(targets))

    # Batch scalar data
    batch_lb = torch.stack(list(lbs))
    batch_ub = torch.stack(list(ubs))
    
    return batch_queries, batch_targets, batch_lb, batch_ub

if __name__ == "__main__":
    # Test code
    db_file = "../datasets/AIDS/db.txt"
    ged_file = "../datasets/AIDS/ged_results.txt"
    
    if os.path.exists(db_file) and os.path.exists(ged_file):
        queries, targets, lb, ub = create_greed_dataset(db_file, ged_file, max_pairs=1000)
        
        loader = create_dataloader(queries, targets, lb, ub, batch_size=4)
        
        print("Testing dataloader...")
        for i, (batch_q, batch_t, batch_lb, batch_ub) in enumerate(loader):
            print(f"Batch {i}: Q={batch_q.num_graphs}, T={batch_t.num_graphs}, LB={batch_lb.shape}, UB={batch_ub.shape}")
            if i >= 2:
                break
        print("Test completed successfully!")
    else:
        print("Test files not found, skipping test")
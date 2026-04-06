"""Data processing utilities."""

from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    t.add_rows(rows)
    print(t.draw())

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format, parallel=True, max_workers=4):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :param parallel: Whether to use parallel loading.
    :param max_workers: Number of parallel workers.
    :return graphs: Networkx (dict) graphs.
    """
    import re

    assert file_format in ['gexf', 'json', 'onehot', 'anchor']

    files = get_file_paths(dir, file_format)

    def extract_gid(filename):
        """Extract gid from filename. Handles both '123.json' and 'query_123.json' formats."""
        name = basename(filename).split('.')[0]
        # Try to extract number from filename (e.g., "123" or "query_123")
        match = re.search(r'(\d+)$', name)
        if match:
            return int(match.group(1))
        else:
            # Fallback: try to convert entire name to int
            return int(name)

    if not parallel or len(files) < 10:
        # Use serial loading for small datasets
        graphs = []
        for file in files:
            gid = extract_gid(file)
            g = _load_single_graph(file, file_format, gid)
            graphs.append(g)
        return graphs

    # Use parallel loading for large datasets
    from concurrent.futures import ThreadPoolExecutor
    import os

    def load_with_gid(file):
        gid = extract_gid(file)
        return gid, _load_single_graph(file, file_format, gid)

    print(f"Loading {len(files)} {file_format} files in parallel with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_with_gid, files))

    # Sort by gid to maintain consistent order
    results.sort(key=lambda x: x[0])
    graphs = [result[1] for result in results]

    return graphs

def _load_single_graph(file, file_format, gid):
    """Helper function to load a single graph file."""
    if file_format == 'gexf':
        import networkx as nx
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
        return g
    elif file_format == 'json':
        # g is a dict
        g = json.load(open(file, 'r'))
        g['gid'] = gid
        return g
    elif file_format in ['onehot', 'anchor']:
        # g is a list of onehot labels
        g = json.load(open(file, 'r'))
        return g

def load_all_graphs(data_location, dataset_name, parallel=True, max_workers=4):
    import torch
    import os

    # Cache file path
    cache_file = f"{data_location}json_data/{dataset_name}/graphs_cache.pt"

    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading graphs from cache: {cache_file}")
        cached_data = torch.load(cache_file)
        return cached_data['db_num'], cached_data['query_num'], cached_data['test_num'], cached_data['graphs']

    # Original loading code if no cache
    print(f"No cache found, loading {dataset_name} graphs from JSON files...")

    # Load db graphs from train/ folder
    print("Loading database graphs from train/ folder...")
    graphs = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "json", parallel=parallel, max_workers=max_workers)
    db_num = len(graphs)
    print(f"Loaded {db_num} database graphs")

    # Load query graphs from queries/ folder
    queries_path = data_location + "json_data/" + dataset_name + "/queries"
    if os.path.exists(queries_path):
        print("Loading query graphs from queries/ folder...")
        query_graphs = iterate_get_graphs(queries_path, "json", parallel=parallel, max_workers=max_workers)
        query_num = len(query_graphs)
        print(f"Loaded {query_num} query graphs")
        graphs += query_graphs
    else:
        print(f"Warning: queries folder not found at {queries_path}")
        query_num = 0

    test_num = 0  # No longer using test placeholder graphs

    print(f"Total graphs loaded: {len(graphs)} (db: {db_num}, queries: {query_num})")

    # Save to cache for next time
    print(f"Saving graphs to cache: {cache_file}")
    torch.save({
        'db_num': db_num,
        'query_num': query_num,
        'test_num': test_num,
        'graphs': graphs
    }, cache_file)

    return db_num, query_num, test_num, graphs

def load_labels(data_location, dataset_name, parallel=True, max_workers=4, graphs=None):
    import torch
    import os
    
    print(f"Loading labels for {dataset_name} (real-time conversion from graphs_cache)...")
    
    # Load global labels mapping
    path = data_location + "json_data/" + dataset_name + "/labels.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")
    
    global_labels = json.load(open(path, 'r'))
    num_labels = len(global_labels)
    
    # Use provided graphs or load from cache if not provided
    if graphs is None:
        print("No graphs provided, loading from cache...")
        train_num, val_num, test_num, graphs = load_all_graphs(data_location, dataset_name, parallel, max_workers)
    else:
        print(f"Using provided graphs ({len(graphs)} graphs)")
    
    # Convert labels to one-hot features in real-time
    print(f"Converting {len(graphs)} graphs' labels to one-hot features (dim={num_labels})...")
    features = []

    # Create label-to-index mapping
    # labels.json format: {"original_label": one_hot_index}
    # The value in labels.json is already the 0-based one-hot index to use
    label_to_idx = {}
    for label_str, one_hot_idx in global_labels.items():
        # Map original label value to its one-hot index
        # In graphs_cache, labels are stored as original values (e.g., 0, 1, 2 for AIDS or 1, 2, 3 for PubChem)
        # We need to map these original values to the one-hot indices defined in labels.json
        original_label_val = int(label_str)
        label_to_idx[original_label_val] = one_hot_idx

    for i, graph in enumerate(graphs):
        if i % 5000 == 0 and i > 0:
            print(f"Converted {i}/{len(graphs)} graphs...")

        # Get labels for this graph
        graph_labels = graph['labels']

        # Convert to one-hot matrix [num_nodes, num_labels]
        onehot_matrix = []
        for label_val in graph_labels:
            onehot_vector = [0.0] * num_labels
            # Look up the one-hot index from labels.json mapping
            idx = label_to_idx.get(label_val)
            if idx is not None and 0 <= idx < num_labels:
                onehot_vector[idx] = 1.0
            else:
                print(f"Warning: label {label_val} not found in labels.json or idx={idx} out of range [0, {num_labels}), using zero vector")
            onehot_matrix.append(onehot_vector)

        features.append(onehot_matrix)
    
    print('Load one-hot label features (dim = {}) of {} (real-time converted).'.format(num_labels, dataset_name))
    return global_labels, features

def load_ged(ged_dict, data_location='', dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)

def load_features(data_location, dataset_name, feature_name):
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", feature_name) \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", feature_name)
    feature_dim = len(features[0][0])
    print('Load {} features (dim = {}) of {}.'.format(feature_name, feature_dim, dataset_name))
    return feature_dim, features
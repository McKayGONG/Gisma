"""Quick script to build pkl from existing txt embedding files.

Usage:
  python build_pkl_from_txt.py --dataset Chemical1M
  python build_pkl_from_txt.py --dataset Chemical1M --workers 64
"""
import os
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_txt_embedding_fast(filepath, dimensions=512):
    """Fast txt loading: read entire file at once, parse with numpy."""
    with open(filepath, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    header = lines[0].split()
    num_nodes = int(header[0])
    if num_nodes == 0:
        return np.zeros(dimensions, dtype=np.float32)
    embs = []
    for i in range(1, num_nodes + 1):
        parts = lines[i].split()
        embs.append(np.fromiter((float(x) for x in parts[1:]), dtype=np.float32, count=dimensions))
    return np.mean(np.stack(embs), axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Chemical1M')
    parser.add_argument('--dim', type=int, default=512)
    import multiprocessing
    default_workers = max(1, int(multiprocessing.cpu_count() * 0.7))
    parser.add_argument('--workers', type=int, default=default_workers)
    args = parser.parse_args()

    ds_lower = args.dataset.lower()
    emb_dir = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{args.dim}')
    pkl_path = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{args.dim}_graph_emb.pkl')

    # Find all txt files
    files = [f for f in os.listdir(emb_dir) if f.startswith('g') and f.endswith('.txt')]
    print(f"Found {len(files)} txt files in {emb_dir}")

    def _load_one(fname):
        gid = fname[1:-4]  # strip 'g' and '.txt'
        filepath = os.path.join(emb_dir, fname)
        try:
            emb = load_txt_embedding_fast(filepath, args.dim)
            return gid, emb
        except Exception:
            return gid, None

    graph_emb_map = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_load_one, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Loading"):
            gid, emb = future.result()
            if emb is not None:
                graph_emb_map[gid] = emb

    print(f"Saving {len(graph_emb_map)} embeddings to {pkl_path}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(graph_emb_map, f)
    print("Done!")


if __name__ == '__main__':
    main()

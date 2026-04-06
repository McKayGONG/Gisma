#!/usr/bin/env python3
"""
Build LAN indexes for all 4 datasets
"""
import os
import time
from lan_hybrid_index_builder import build_lan_index_with_greed

script_dir = os.path.dirname(os.path.abspath(__file__))
gisma_base = os.path.join(script_dir, '..', 'Gisma')

# Dataset configurations
DATASETS = [
    {
        'name': 'AIDS',
        'embedding_file': os.path.join(gisma_base, 'embeddings', 'AIDS', 'AIDS_embeddings.bin'),
        'graph_file': os.path.join(gisma_base, 'datasets', 'AIDS', 'db.txt'),
        'output_file': os.path.join('indexes', 'AIDS_LAN_index.pkl'),
        'distance_mode': 'ged',
        'timeout': 10.0
    },
    {
        'name': 'PubChem',
        'embedding_file': os.path.join(gisma_base, 'embeddings', 'PubChem', 'PubChem_embeddings.bin'),
        'graph_file': os.path.join(gisma_base, 'datasets', 'PubChem', 'db.txt'),
        'output_file': os.path.join('indexes', 'PubChem_LAN_index.pkl'),
        'distance_mode': 'ged',
        'timeout': 10.0
    },
    {
        'name': 'Chemical1M',
        'embedding_file': os.path.join(gisma_base, 'embeddings', 'Chemical1M', 'Chemical1M_embeddings.bin'),
        'graph_file': os.path.join(gisma_base, 'datasets', 'Chemical1M', 'db.txt'),
        'output_file': os.path.join('indexes', 'Chemical1M_LAN_index.pkl'),
        'distance_mode': 'embedding',
        'timeout': 10.0
    },
    {
        'name': 'SYN',
        'embedding_file': os.path.join(gisma_base, 'embeddings', 'SYN', 'SYN_embeddings.bin'),
        'graph_file': os.path.join(gisma_base, 'datasets', 'SYN', 'db.txt'),
        'output_file': os.path.join('indexes', 'SYN_LAN_index.pkl'),
        'distance_mode': 'embedding',
        'timeout': 10.0
    }
]

def main():
    print("=" * 80)
    print("Building LAN Indexes for All Datasets")
    print("=" * 80)
    print("All datasets: distance_mode=embedding")
    print("=" * 80)

    total_start = time.time()

    for i, ds in enumerate(DATASETS, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/4] {ds['name']} (distance_mode={ds['distance_mode']})")
        print("=" * 80)

        start = time.time()
        try:
            build_lan_index_with_greed(
                embedding_file=ds['embedding_file'],
                graph_file=ds['graph_file'],
                output_file=ds['output_file'],
                M=40,
                maxDeg0=80,
                efConst=200,
                distance_mode=ds['distance_mode'],
                timeout=ds['timeout']
            )
            elapsed = time.time() - start
            print(f"\n[{ds['name']}] Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n[{ds['name']}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"All indexes built in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Run experiments for a single dataset - split by tau and query range.

For each dataset, the smallest tau uses ef=10, all others use --ef (default 50).

Usage:
    python run_all_full.py --dataset AIDS --timeout 0.1 --beam_width 1
    python run_all_full.py --dataset AIDS --tau 2 4 --timeout 0.1 --beam_width 1
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
import time


DATASET_TAUS = {
    'AIDS':       [2, 4, 6, 8, 10, 12],
    'PubChem':    [2, 4, 6, 8, 10, 12],
    'SYN':        [1, 2, 3, 4],           # tau=5,6 CNF (>24h per Gisma paper)
    'Chemical1M': [2, 4, 6],              # tau=8,10,12 CNF (>24h per Gisma paper)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_TAUS.keys()))
    parser.add_argument('--tau', type=int, nargs='+', default=None,
                        help='Tau values to run. Default: all for the dataset')
    parser.add_argument('--ef', type=int, default=50, help='Search width (default: 50)')
    parser.add_argument('--ef_min_tau', type=int, default=10,
                        help='ef for the smallest tau (default: 10)')
    parser.add_argument('--timeout', type=float, default=0.1)
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=25,
                        help='Queries per worker (default: 25)')
    parser.add_argument('--query_start', type=int, default=0)
    parser.add_argument('--query_end', type=int, default=99)
    args = parser.parse_args()

    dataset = args.dataset
    taus = args.tau if args.tau else DATASET_TAUS[dataset]
    min_tau = min(taus)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_script = os.path.join(script_dir, 'similarity_search.py')

    # Build query chunks
    chunks = []
    q = args.query_start
    while q <= args.query_end:
        q_end = min(q + args.chunk_size - 1, args.query_end)
        chunks.append((q, q_end))
        q = q_end + 1

    total_workers = len(taus) * len(chunks)

    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Tau values: {taus}")
    print(f"ef: {args.ef} (min tau={min_tau} uses ef={args.ef_min_tau})")
    print(f"timeout: {args.timeout}s, beam_width: {args.beam_width}")
    print(f"Query range: {args.query_start}-{args.query_end}, chunk_size: {args.chunk_size}")
    print(f"Total workers: {total_workers} ({len(taus)} tau x {len(chunks)} chunks)")
    print("=" * 60)

    wall_clock_start = time.time()

    # Launch all workers
    processes = []
    for tau in taus:
        ef = args.ef_min_tau if tau == min_tau else args.ef
        for q_start, q_end in chunks:
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '4'
            env['MKL_NUM_THREADS'] = '4'

            cmd = [
                sys.executable, '-u', search_script,
                '--dataset', dataset,
                '--tau', str(tau),
                '--timeout', str(args.timeout),
                '--beam_width', str(args.beam_width),
                '--query_start', str(q_start),
                '--query_end', str(q_end),
                '--ef', str(ef),
                '--save'
            ]

            log_dir = os.path.join(script_dir, 'results', dataset, f'tau{int(tau)}')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'q{q_start}-{q_end}.log')

            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                processes.append((proc, dataset, tau, ef, q_start, q_end, log_file))

            print(f"  Started: {dataset} tau={tau} ef={ef} q={q_start}-{q_end}")

    print(f"\nAll {total_workers} workers launched!")
    print("Waiting for completion...\n")

    # Wait for all
    completed = 0
    failed = 0
    for proc, ds, tau, ef, q_start, q_end, log_file in processes:
        proc.wait()
        completed += 1
        if proc.returncode != 0:
            failed += 1
            print(f"  FAILED: {ds} tau={tau} q={q_start}-{q_end} (code {proc.returncode})")
        else:
            print(f"  Done: {ds} tau={tau} q={q_start}-{q_end} [{completed}/{total_workers}]")

    wall_clock_time = time.time() - wall_clock_start

    print("\n" + "=" * 60)
    print(f"All workers completed! ({dataset})")
    print(f"  Success: {total_workers - failed}")
    print(f"  Failed: {failed}")
    print(f"  Wall clock: {wall_clock_time:.1f}s ({wall_clock_time/60:.1f} min, {wall_clock_time/3600:.2f} h)")
    print("=" * 60)


if __name__ == '__main__':
    main()

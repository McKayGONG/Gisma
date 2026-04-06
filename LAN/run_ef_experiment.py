#!/usr/bin/env python3
"""
Run ef sensitivity experiments: one-to-one tau→ef mapping.
All workers launch simultaneously per dataset.

Each tau has a fixed ef. Each worker handles chunk_size queries.
  AIDS/PubChem: tau=2→ef=6, 4→12, 6→25, 8→50, 10→100, 12→200
  Chemical1M:   tau=2→ef=6, 4→12, 6→25
  SYN:          tau=1→ef=6, 2→12, 3→25, 4→50

Results: results/{dataset}/tau{X}/ef{Y}/per_query/query_*.json

Usage:
    python run_ef_experiment.py --dataset AIDS
    python run_ef_experiment.py --dataset SYN --chunk_size 10
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
import time


TAU_EF_MAP = {
    'AIDS':       [(2, 6), (4, 12), (6, 25), (8, 50), (10, 100), (12, 200)],
    'PubChem':    [(2, 6), (4, 12), (6, 25), (8, 50), (10, 100), (12, 200)],
    'Chemical1M': [(2, 6), (4, 12), (6, 25)],
    'SYN':        [(1, 6), (2, 12), (3, 25), (4, 50)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(TAU_EF_MAP.keys()))
    parser.add_argument('--timeout', type=float, default=0.1)
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--query_start', type=int, default=0)
    parser.add_argument('--query_end', type=int, default=99)
    parser.add_argument('--no_use_cg', action='store_true', help='Disable CG compression')
    args = parser.parse_args()

    dataset = args.dataset
    tau_ef_pairs = TAU_EF_MAP[dataset]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_script = os.path.join(script_dir, 'similarity_search.py')

    # Build query chunks
    chunks = []
    q = args.query_start
    while q <= args.query_end:
        q_end = min(q + args.chunk_size - 1, args.query_end)
        chunks.append((q, q_end))
        q = q_end + 1

    n_workers = len(tau_ef_pairs) * len(chunks)

    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Tau→ef pairs: {tau_ef_pairs}")
    print(f"timeout={args.timeout}s, beam_width={args.beam_width}, use_cg={not args.no_use_cg}")
    print(f"Queries: {args.query_start}-{args.query_end}, chunk_size={args.chunk_size}")
    print(f"Total workers: {n_workers} ({len(tau_ef_pairs)} tau x {len(chunks)} chunks)")
    print("=" * 60)

    wall_start = time.time()
    processes = []

    for tau, ef in tau_ef_pairs:
        for q_start, q_end in chunks:
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '4'
            env['MKL_NUM_THREADS'] = '4'

            cmd = [
                sys.executable, '-u', search_script,
                '--dataset', dataset,
                '--tau', str(tau),
                '--ef', str(ef),
                '--timeout', str(args.timeout),
                '--beam_width', str(args.beam_width),
                '--query_start', str(q_start),
                '--query_end', str(q_end),
                '--save',
            ]
            if not args.no_use_cg:
                cmd.append('--use_cg')

            log_dir = os.path.join(script_dir, 'results', dataset,
                                   f'tau{tau}', f'ef{ef}')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'q{q_start}-{q_end}.log')

            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                processes.append((proc, tau, ef, q_start, q_end))

            print(f"  Started: tau={tau} ef={ef} q={q_start}-{q_end}")

    print(f"\nAll {n_workers} workers launched. Waiting...")

    # Wait for all
    completed = 0
    failed = 0
    for proc, tau, ef, q_start, q_end in processes:
        proc.wait()
        completed += 1
        if proc.returncode != 0:
            failed += 1
            print(f"  FAILED: tau={tau} ef={ef} q={q_start}-{q_end} (code {proc.returncode})")
        else:
            print(f"  Done: tau={tau} ef={ef} q={q_start}-{q_end} [{completed}/{n_workers}]")

    wall_time = time.time() - wall_start

    print("\n" + "=" * 60)
    print(f"All done! ({dataset})")
    print(f"  Success: {n_workers - failed}")
    print(f"  Failed: {failed}")
    print(f"  Wall clock: {wall_time:.1f}s ({wall_time/60:.1f} min, {wall_time/3600:.2f} h)")
    print("=" * 60)


if __name__ == '__main__':
    main()

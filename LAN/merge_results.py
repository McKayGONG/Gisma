#!/usr/bin/env python3
"""
Merge all per_query JSON results and generate a summary report
"""
import os
import json
import argparse
from datetime import datetime


def merge_results(dataset, tau, query_start=0, query_end=99):
    """Merge results from per_query JSON files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    per_query_dir = os.path.join(script_dir, 'results', dataset, f'tau{int(tau)}', 'per_query', 'LAN')

    if not os.path.exists(per_query_dir):
        print(f"Error: {per_query_dir} does not exist")
        return None

    results = []
    for qid in range(query_start, query_end + 1):
        json_file = os.path.join(per_query_dir, f'query_{qid}.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)

    return results


def generate_report(results, dataset, tau, query_start, query_end, output_file=None):
    """Generate summary report"""
    if not results:
        print("No results to merge")
        return

    # Get metadata from the first result
    meta = results[0]['metadata']

    lines = []
    lines.append("LAN Similarity Search Results (Merged)")
    lines.append("=" * 120)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Query Range: {query_start} - {query_end}")
    lines.append(f"Tau: {tau}")
    lines.append(f"ef: {meta.get('ef', 'N/A')}, timeout: {meta.get('timeout', 'N/A')}s")
    lines.append(f"Neighbor Pruning: {'ON' if meta.get('pruning_ratio', 0) > 0 else 'OFF'} (ratio={meta.get('pruning_ratio', 0)})")
    lines.append("=" * 120)
    lines.append("")

    # Table header
    lines.append(f" {'Query':>5} | {'GT':>4} | {'Found':>5} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Time':>7} | {'GED#':>6} | {'ef':>5} | {'Iter':>4}")
    lines.append("-" * 120)

    # Statistics variables
    total_gt = 0
    total_found = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0
    total_ged = 0
    valid_prec = []
    valid_recall = []
    valid_f1 = []

    for data in results:
        r = data['result']
        qid = r['query_id']
        gt = r['gt_count']
        found = r['found_count']
        tp = r['correct_count']
        fp = found - tp
        fn = gt - tp
        prec = r['precision']
        recall = r['recall']
        f1 = r['f1']
        time_s = r['query_time']
        ged_count = r['ged_computations']
        ef = r['final_ef']
        iters = r['iterations']

        # Format output
        prec_str = f"{prec:.3f}" if prec is not None else "N/A"
        recall_str = f"{recall:.3f}" if recall is not None else "N/A"
        f1_str = f"{f1:.3f}" if f1 is not None else "N/A"

        lines.append(f" {qid:>5} | {gt:>4} | {found:>5} | {tp:>4} | {fp:>4} | {fn:>4} | {prec_str:>6} | {recall_str:>6} | {f1_str:>6} | {time_s:>6.2f}s | {ged_count:>6} | {ef:>5} | {iters:>4}")

        # Accumulate statistics
        total_gt += gt
        total_found += found
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_time += time_s
        total_ged += ged_count
        if prec is not None:
            valid_prec.append(prec)
        if recall is not None:
            valid_recall.append(recall)
        if f1 is not None:
            valid_f1.append(f1)

    # Averages
    n = len(results)
    avg_gt = total_gt / n
    avg_found = total_found / n
    avg_tp = total_tp / n
    avg_fp = total_fp / n
    avg_fn = total_fn / n
    avg_time = total_time / n
    avg_ged = total_ged / n
    avg_prec = sum(valid_prec) / len(valid_prec) if valid_prec else 0
    avg_recall = sum(valid_recall) / len(valid_recall) if valid_recall else 0
    avg_f1 = sum(valid_f1) / len(valid_f1) if valid_f1 else 0

    lines.append("-" * 120)
    lines.append(f" {'Avg':>5} | {avg_gt:>4.1f} | {avg_found:>5.1f} | {avg_tp:>4.1f} | {avg_fp:>4.1f} | {avg_fn:>4.1f} | {avg_prec:>6.3f} | {avg_recall:>6.3f} | {avg_f1:>6.3f} | {avg_time:>6.2f}s | {avg_ged:>6.1f} |       |     ")
    lines.append("=" * 120)
    lines.append("")

    # Summary statistics
    lines.append("Summary Statistics")
    lines.append("-" * 40)
    lines.append(f"Total queries: {n}")
    lines.append(f"Total search time: {total_time:.2f}s")
    lines.append(f"Total GED computations: {total_ged}")
    lines.append(f"Average search time: {avg_time:.2f}s")
    lines.append(f"Average GED computations: {avg_ged:.1f}")
    lines.append(f"Average Precision: {avg_prec:.3f}")
    lines.append(f"Average Recall: {avg_recall:.3f}")
    lines.append(f"Average F1: {avg_f1:.3f}")
    lines.append("")

    report = "\n".join(lines)

    # Output
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")

    return {
        'total_queries': n,
        'avg_precision': avg_prec,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_time': avg_time,
        'total_time': total_time
    }


def main():
    parser = argparse.ArgumentParser(description='Merge per-query results into summary report')
    parser.add_argument('--dataset', type=str, default='AIDS', help='Dataset name')
    parser.add_argument('--tau', type=float, required=True, help='Tau threshold')
    parser.add_argument('--query_start', type=int, default=0, help='Start query index')
    parser.add_argument('--query_end', type=int, default=99, help='End query index')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: auto)')
    args = parser.parse_args()

    # Merge results
    results = merge_results(args.dataset, args.tau, args.query_start, args.query_end)

    if not results:
        print("No results found")
        return

    # Generate output filename
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(
            script_dir, 'results', args.dataset, f'tau{int(args.tau)}',
            f'summary_report_q{args.query_start}-{args.query_end}.txt'
        )
    else:
        output_file = args.output

    # Generate report
    generate_report(results, args.dataset, args.tau, args.query_start, args.query_end, output_file)


if __name__ == '__main__':
    main()

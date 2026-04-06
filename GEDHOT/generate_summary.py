#!/usr/bin/env python3
"""
Standalone script to generate summary reports from per-query JSON files.
Allows flexible control over which queries to include in the summary.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_query_result(json_file):
    """Load a single query result from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def generate_single_method_summary(dataset, method, tau, query_start, query_end, output_file, model_epoch=20):
    """Generate summary report for a single method."""

    # Construct path to per-query JSON files
    base_dir = Path(__file__).parent / "results" / "similarity_search" / dataset / f"tau{tau}" / "per_query" / method

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return

    # Collect all query results
    results = []
    valid_queries = 0
    total_found = 0
    total_gt = 0
    total_correct = 0
    total_time = 0.0
    total_candidates = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    queries_with_found = 0
    queries_with_gt = 0

    for query_id in range(query_start, query_end + 1):
        json_file = base_dir / f"query_{query_id}.json"

        if not json_file.exists():
            print(f"Warning: Query file not found: {json_file}")
            continue

        data = load_query_result(json_file)
        result = data['result']

        # Handle null values
        gt_count = result.get('gt_count') or 0
        found_count = result.get('found_count') or 0
        correct_count = result.get('correct_count') or 0
        query_time = result.get('query_time') or 0.0
        candidates_count = result.get('candidates_count') or 0

        # Only count queries with ground truth for overall metrics
        if gt_count > 0:
            valid_queries += 1
            queries_with_gt += 1
            sum_recall += result.get('recall') or 0.0
            sum_f1 += result.get('f1') or 0.0

        if found_count > 0:
            queries_with_found += 1
            sum_precision += result.get('precision') or 0.0

        total_found += found_count
        total_gt += gt_count
        total_correct += correct_count
        total_time += query_time
        total_candidates += candidates_count

        results.append(result)

    if not results:
        print("Error: No query results found")
        return

    # Calculate overall metrics (average of per-query metrics, matching trainer.py)
    avg_recall = sum_recall / queries_with_gt if queries_with_gt > 0 else 0.0
    avg_precision = sum_precision / queries_with_found if queries_with_found > 0 else 0.0
    avg_found = total_found / len(results)
    avg_time = total_time / len(results)
    avg_candidates = total_candidates / len(results)

    # Calculate computation time per comparison
    total_comparisons = total_candidates
    avg_computation_time_ms = (total_time * 1000 / total_comparisons) if total_comparisons > 0 else 0.0

    # Generate summary report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIMILARITY SEARCH SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Model Epoch: {model_epoch}\n")
        f.write(f"Tau Threshold: {tau}\n")
        f.write(f"Total Queries: {len(results)}\n")
        f.write(f"Valid Queries (with GT): {valid_queries}\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- Overall Performance Summary ---\n")
        f.write(f"Average Recall:       {avg_recall:.4f}\n")
        f.write(f"Average Precision:    {avg_precision:.4f}\n")
        f.write(f"Total Found:          {total_found}\n")
        f.write(f"Avg Found/Query:      {avg_found:.2f}\n")
        f.write(f"Total Query Time:     {total_time:.2f} seconds\n")
        f.write(f"Avg Query Time:       {avg_time:.4f} seconds\n")
        f.write(f"Total Computation Time: {total_time:.2f} seconds\n")
        f.write(f"Avg Computation Time:   {avg_computation_time_ms:.2f} ms/comparison\n\n")

        f.write("--- Per-Query Details ---\n")
        f.write("Query | Candidates | Found | GT | Precision | Recall   | F1     | Time(s)\n")
        f.write("------|------------|-------|----|-----------|---------|---------|---------\n")

        for result in results:
            query_id = result['query_id']
            candidates = result.get('candidates_count') or 0
            found = result.get('found_count') or 0
            gt = result.get('gt_count') or 0
            precision = result.get('precision') or 0.0
            recall = result.get('recall') or 0.0
            f1 = result.get('f1') or 0.0
            time = result.get('query_time') or 0.0

            # Apply N/A rules
            if found == 0:
                precision_str = "N/A"
            else:
                precision_str = f"{precision:9.4f}"

            if gt == 0:
                recall_str = "N/A"
            else:
                recall_str = f"{recall:7.4f}"

            if found == 0 or gt == 0:
                f1_str = "N/A"
            else:
                f1_str = f"{f1:6.4f}"

            f.write(f"{query_id:5} | {candidates:10} | {found:5} | {gt:2} | "
                   f"{precision_str:>9} | {recall_str:>7} | {f1_str:>6} | {time:7.4f}\n")

        # Average row
        f.write("------|------------|-------|----|-----------|---------|---------|---------\n")
        avg_gt = total_gt / len(results)
        f.write(f"  AVG | {avg_candidates:10.2f} | {avg_found:5.2f} | {avg_gt:4.2f} | "
               f"{avg_precision:9.4f} | {avg_recall:7.4f} | {0.0:6.4f} | {avg_time:7.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report generated: {output_path}")


def generate_all_methods_summary(dataset, tau, query_start, query_end, output_file, model_epoch=20):
    """Generate comparison summary report for all three methods."""
    import time
    import os

    methods = ['GEDGW', 'GEDIOT', 'GEDHOT']

    # Use absolute path to avoid issues in different working directories
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "results" / "similarity_search" / dataset / f"tau{tau}" / "per_query"

    print(f"Looking for per_query directory at: {base_dir}")
    print(f"Current working directory: {os.getcwd()}")

    # Wait up to 10 seconds for directory to be created
    for i in range(10):
        if base_dir.exists():
            print(f"Found per_query directory!")
            break
        if i == 0:
            print(f"Waiting for per_query directory to be created...")
        time.sleep(1)

    if not base_dir.exists():
        print(f"Error: Directory not found after waiting: {base_dir}")
        print(f"Please check if the experiments completed successfully.")
        return

    # Collect results for all methods
    all_results = defaultdict(list)

    for method in methods:
        method_dir = base_dir / method
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_dir}")
            continue

        for query_id in range(query_start, query_end + 1):
            json_file = method_dir / f"query_{query_id}.json"

            if not json_file.exists():
                print(f"Warning: Query file not found: {json_file}")
                continue

            data = load_query_result(json_file)
            all_results[query_id].append({
                'method': method,
                'result': data['result']
            })

    if not all_results:
        print("Error: No query results found for any method")
        return

    # Calculate overall statistics per method
    method_stats = {}
    for method in methods:
        total_found = 0
        total_gt = 0
        total_correct = 0
        total_time = 0.0
        total_candidates = 0
        query_count = 0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_f1 = 0.0
        queries_with_found = 0  # Queries with found > 0 (for precision)
        queries_with_gt = 0     # Queries with gt > 0 (for recall)

        for query_id in sorted(all_results.keys()):
            for m in all_results[query_id]:
                if m['method'] == method:
                    result = m['result']
                    found = result.get('found_count') or 0
                    gt = result.get('gt_count') or 0

                    total_found += found
                    total_gt += gt
                    total_correct += result.get('correct_count') or 0
                    total_time += result.get('query_time') or 0.0
                    total_candidates += result.get('candidates_count') or 0
                    query_count += 1

                    # Only add precision when found > 0
                    if found > 0:
                        sum_precision += result.get('precision') or 0.0
                        queries_with_found += 1

                    # Only add recall when gt > 0
                    if gt > 0:
                        sum_recall += result.get('recall') or 0.0
                        sum_f1 += result.get('f1') or 0.0
                        queries_with_gt += 1

        avg_precision = sum_precision / queries_with_found if queries_with_found > 0 else 0.0
        avg_recall = sum_recall / queries_with_gt if queries_with_gt > 0 else 0.0
        avg_f1 = sum_f1 / queries_with_gt if queries_with_gt > 0 else 0.0

        method_stats[method] = {
            'queries': query_count,
            'total_found': total_found,
            'total_gt': total_gt,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'total_time': total_time,
            'total_candidates': total_candidates
        }

    # Generate summary report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THREE-IN-ONE SIMILARITY SEARCH SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Query Range: {query_start} to {query_end}\n")
        f.write(f"Tau Thresholds: [{tau}]\n")
        f.write(f"Model Epoch: {model_epoch}\n")
        f.write("=" * 80 + "\n\n")

        # Overall summary table
        f.write(f"--- Tau = {tau} Summary ---\n")
        f.write("Method    | Queries | Total Found | Total GT | Avg Precision | Avg Recall | Avg F1 | Total Time(s)\n")
        f.write("----------|---------|-------------|----------|---------------|------------|--------|--------------\n")

        for method in methods:
            stats = method_stats.get(method, {})
            f.write(f"{method:<9} | {stats.get('queries', 0):<7} | "
                   f"{stats.get('total_found', 0):<11} | {stats.get('total_gt', 0):<8} | "
                   f"{stats.get('avg_precision', 0.0):<13.4f} | {stats.get('avg_recall', 0.0):<10.4f} | "
                   f"{stats.get('avg_f1', 0.0):<6.4f} | {stats.get('total_time', 0.0):<12.2f}\n")

        # Time analysis
        f.write("\n--- Time Analysis ---\n")
        for method in methods:
            stats = method_stats.get(method, {})
            total_time = stats.get('total_time', 0.0)
            queries = stats.get('queries', 1)
            candidates = stats.get('total_candidates', 1)
            avg_time = total_time / queries if queries > 0 else 0.0
            avg_comp_time = (total_time * 1000 / candidates) if candidates > 0 else 0.0
            f.write(f"{method}: Total={total_time:.2f}s, Avg/query={avg_time:.3f}s, "
                   f"Avg/comparison={avg_comp_time:.2f}ms, Total comparisons={candidates}\n")

        # Speed comparison
        if 'GEDGW' in method_stats and 'GEDIOT' in method_stats:
            gedgw_time = method_stats['GEDGW'].get('total_time', 1.0)
            gediot_time = method_stats['GEDIOT'].get('total_time', 1.0)
            speedup = gedgw_time / gediot_time if gediot_time > 0 else 0.0
            f.write(f"\nSpeed comparison: GEDIOT is {speedup:.1f}x faster than GEDGW\n")

        # Detailed per-query table
        f.write(f"\n--- Detailed Results for Tau = {tau} ---\n")
        f.write("Query | Method  | Found | GT | Candidates | Time(s) | Precision | Recall | F1\n")
        f.write("------|---------|-------|----|-----------|---------|-----------|---------|---------\n")

        for query_id in sorted(all_results.keys()):
            for m in all_results[query_id]:
                method = m['method']
                result = m['result']
                found = result.get('found_count') or 0
                gt = result.get('gt_count') or 0
                candidates = result.get('candidates_count') or 0
                time = result.get('query_time') or 0.0
                precision = result.get('precision') or 0.0
                recall = result.get('recall') or 0.0
                f1 = result.get('f1') or 0.0

                # Apply N/A rules
                if found == 0:
                    precision_str = "N/A"
                else:
                    precision_str = f"{precision:9.4f}"

                if gt == 0:
                    recall_str = "N/A"
                else:
                    recall_str = f"{recall:7.4f}"

                if found == 0 or gt == 0:
                    f1_str = "N/A"
                else:
                    f1_str = f"{f1:7.4f}"

                f.write(f"{query_id:<5} | {method:<7} | {found:<5} | "
                       f"{gt:<2} | {candidates:<9} | "
                       f"{time:<7.3f} | {precision_str:>9} | "
                       f"{recall_str:>7} | {f1_str:>7}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary reports from per-query JSON files"
    )

    parser.add_argument("--dataset",
                       type=str,
                       default="AIDS",
                       help="Dataset name (default: AIDS)")

    parser.add_argument("--method",
                       type=str,
                       choices=["GEDGW", "GEDIOT", "GEDHOT", "all"],
                       default="all",
                       help="Method to summarize (GEDGW, GEDIOT, GEDHOT, or all)")

    parser.add_argument("--tau",
                       type=float,
                       required=True,
                       help="Tau threshold value")

    parser.add_argument("--query-start",
                       type=int,
                       default=0,
                       help="Starting query ID (default: 0)")

    parser.add_argument("--query-end",
                       type=int,
                       default=99,
                       help="Ending query ID (default: 99)")

    parser.add_argument("--model-epoch",
                       type=int,
                       default=20,
                       help="Model epoch number (default: 20)")

    parser.add_argument("--output",
                       type=str,
                       default=None,
                       help="Output file path (default: auto-generated based on method and query range)")

    args = parser.parse_args()

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        base_dir = Path(__file__).parent / "results" / "similarity_search" / args.dataset / f"tau{args.tau}"
        if args.method == "all":
            output_file = base_dir / f"summary_report_q{args.query_start}-{args.query_end}.txt"
        else:
            output_file = base_dir / f"summary_{args.method}_q{args.query_start}-{args.query_end}.txt"

    # Generate summary
    if args.method == "all":
        generate_all_methods_summary(
            args.dataset, args.tau, args.query_start, args.query_end,
            output_file, args.model_epoch
        )
    else:
        generate_single_method_summary(
            args.dataset, args.method, args.tau, args.query_start, args.query_end,
            output_file, args.model_epoch
        )


if __name__ == "__main__":
    main()

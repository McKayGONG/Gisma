#!/usr/bin/env python3
"""
Calculate similarity search metrics from GEDIOT-only results
"""
import json
import sys
import re

def load_ground_truth(gt_file):
    """Load ground truth from file"""
    gt_dict = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = int(parts[0])
                ged = float(parts[1])
                # Extract all db_ids from the rest of the line using regex
                db_list_str = ' '.join(parts[2:])
                db_ids = re.findall(r'\d+', db_list_str)
                db_ids = [int(x) for x in db_ids]

                if query_id not in gt_dict:
                    gt_dict[query_id] = {}
                for db_id in db_ids:
                    gt_dict[query_id][db_id] = ged
    return gt_dict

def calculate_metrics(results_file, gt_file, tau, query_start=0, query_end=99):
    """Calculate precision, recall, F1 for similarity search"""

    # Load results
    with open(results_file) as f:
        data = json.load(f)

    # Load ground truth
    gt_dict = load_ground_truth(gt_file)

    # Get tau data
    tau_key = str(float(tau))
    if tau_key not in data:
        print(f"Error: tau {tau_key} not found in results")
        return

    tau_data = data[tau_key]

    # Calculate metrics
    total_gt_answers = 0
    total_predicted = 0
    correct_predictions = 0
    total_queries = 0
    all_predictions = []

    for query_idx in range(query_start, query_end + 1):
        query_key = str(query_idx)

        if query_key not in tau_data:
            continue

        query_data = tau_data[query_key]
        total_queries += 1

        # Get predicted distances
        gediot_distances = query_data.get('gediot_distances', {})

        # Filter predictions by tau threshold
        predicted_answers = set()
        for db_id_str, dist in gediot_distances.items():
            if dist <= tau:
                predicted_answers.add(int(db_id_str))
            all_predictions.append(dist)

        # Get ground truth answers
        gt_answers = set()
        if query_idx in gt_dict:
            for db_id, ged in gt_dict[query_idx].items():
                if ged <= tau:
                    gt_answers.add(db_id)

        # Count
        total_gt_answers += len(gt_answers)
        total_predicted += len(predicted_answers)
        correct_predictions += len(gt_answers & predicted_answers)

    # Calculate final metrics
    precision = correct_predictions / total_predicted if total_predicted > 0 else 0
    recall = correct_predictions / total_gt_answers if total_gt_answers > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_pred = sum(all_predictions) / len(all_predictions) if all_predictions else 0

    print(f'=== Similarity Search Metrics (Tau={tau}) ===')
    print(f'Total Queries: {total_queries}')
    print(f'Total GT Answers: {total_gt_answers}')
    print(f'Total Predicted: {total_predicted}')
    print(f'Correct Predictions: {correct_predictions}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1 Score: {f1:.6f}')
    print(f'Mean Prediction: {mean_pred:.2f}')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_prediction': mean_pred,
        'total_queries': total_queries,
        'total_gt_answers': total_gt_answers,
        'total_predicted': total_predicted,
        'correct_predictions': correct_predictions
    }

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python calculate_similarity_metrics.py <results_json> <gt_file> <tau> [query_start] [query_end]")
        sys.exit(1)

    results_file = sys.argv[1]
    gt_file = sys.argv[2]
    tau = float(sys.argv[3])
    query_start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    query_end = int(sys.argv[5]) if len(sys.argv) > 5 else 99

    calculate_metrics(results_file, gt_file, tau, query_start, query_end)

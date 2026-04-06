#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_AIDS_ged_distribution_under_15.py
Read AIDS dataset GED dictionary data and plot GED value frequency distribution.
For analyzing distribution characteristics of GED results with node count <= 15.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# Set font
plt.rcParams['font.family'] = 'Cambria'

def load_ged_dict(pkl_file):
    """Load GED dictionary in pkl format"""
    print("Loading GED dictionary from pkl file...")

    with open(pkl_file, 'rb') as f:
        ged_dict = pickle.load(f)

    print(f"Loaded {len(ged_dict)} graph pairs")

    # Extract all GED values
    ged_values = list(ged_dict.values())

    return ged_values, ged_dict

def main():
    # AIDS dataset file path
    pkl_file = "./datasets/ged_dict_15node.pkl"

    # Check if file exists
    if not os.path.exists(pkl_file):
        print(f"Error: File not found: {pkl_file}")
        return

    # Load data
    ged_values, ged_dict = load_ged_dict(pkl_file)

    if not ged_values:
        print("No valid GED values found!")
        return

    print(f"Processing {len(ged_values)} GED values...")

    # Statistics
    min_ged = min(ged_values)
    max_ged = max(ged_values)
    mean_ged = np.mean(ged_values)
    std_ged = np.std(ged_values)

    print(f"\n=== GED Statistics ===")
    print(f"Total pairs: {len(ged_values)}")
    print(f"Min GED: {min_ged}")
    print(f"Max GED: {max_ged}")
    print(f"Mean GED: {mean_ged:.2f}")
    print(f"Std GED: {std_ged:.2f}")

    # Plot distribution
    plt.figure(figsize=(10, 6))

    # Calculate frequency percentage for each GED value
    ged_counter = Counter(ged_values)
    unique_values = sorted(ged_counter.keys())
    frequencies = []
    for val in unique_values:
        count = ged_counter[val]
        percentage = count/len(ged_values)*100
        frequencies.append(percentage)

    # Plot line chart
    plt.plot(unique_values, frequencies, 'purple', marker='+', linewidth=2, markersize=8, label='GED distribution')

    # Fill the 0-8 range area with dense gray dots
    x_fill = [val for val in unique_values if val <= 8]
    y_fill = [frequencies[i] for i, val in enumerate(unique_values) if val <= 8]

    if x_fill:  # ensure there are data points
        plt.fill_between(x_fill, y_fill, color='gray', alpha=0.3, hatch='...', edgecolor='none')

    # Add vertical dashed lines at specific GED values (from corresponding frequency to X-axis)
    vertical_lines = [6, 8, 12, 15, 16]
    for ged_val in vertical_lines:
        if ged_val in ged_counter:
            # Find the corresponding frequency value
            frequency_val = ged_counter[ged_val] / len(ged_values) * 100
            # Draw dashed line from frequency value down to X-axis
            plt.vlines(x=ged_val, ymin=0, ymax=frequency_val, colors='black', linestyles='dashed', alpha=0.7)

    # Set plot style
    plt.xlabel('Graph Edit Distance', fontsize=32)
    plt.ylabel('Frequency (%)', fontsize=32)
    plt.grid(True, alpha=0.3)

    # Set axes to start from origin
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Add X-axis labels at specific positions, remove 10
    special_ticks = [6, 12, 15]
    current_ticks = list(plt.xticks()[0])  # get current ticks
    # Ensure ticks at specific positions exist
    for tick in special_ticks:
        if tick not in current_ticks:
            current_ticks.append(tick)
    # Remove the tick at 10
    if 10 in current_ticks:
        current_ticks.remove(10)
    current_ticks.sort()
    plt.xticks(current_ticks, fontsize=32)

    # Get current Y-axis ticks and remove 0 to avoid double-marking at origin
    current_y_ticks = plt.yticks()[0]
    y_ticks = [tick for tick in current_y_ticks if tick > 0]
    plt.yticks(y_ticks, fontsize=32)

    # Save image
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)

    # Save in PNG and PDF formats
    png_path = os.path.join(output_dir, "AIDS_ged_distribution_under_15.png")
    pdf_path = os.path.join(output_dir, "AIDS_ged_distribution_under_15.pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

    print(f"\n=== Output Files ===")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")

    # Display plot
    plt.show()

    # Save GED value distribution statistics to file
    stats_dir = "./stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "AIDS_ged_distribution_under_15_stats.txt")

    # Calculate frequency for each GED value
    ged_counter = Counter(ged_values)
    unique_values = sorted(ged_counter.keys())

    with open(stats_file, 'w') as f:
        f.write("AIDS GED Distribution Statistics (Nodes <= 15)\n")
        f.write("="*50 + "\n")
        f.write(f"Total pairs: {len(ged_values)}\n")
        f.write(f"Min GED: {min_ged}\n")
        f.write(f"Max GED: {max_ged}\n")
        f.write(f"Mean GED: {mean_ged:.2f}\n")
        f.write(f"Std GED: {std_ged:.2f}\n")
        f.write("="*50 + "\n\n")
        f.write("GED Value\tCount\t\tPercentage\n")
        f.write("-"*40 + "\n")

        for val in unique_values:
            count = ged_counter[val]
            percentage = count/len(ged_values)*100
            f.write(f"{val:.1f}\t\t{count}\t\t{percentage:.2f}%\n")

    print(f"\n=== Statistics saved to: {stats_file} ===")

    # Print GED value distribution (top 10)
    print(f"\n=== Top 10 GED Value Distribution ===")
    for i, val in enumerate(unique_values[:10]):
        count = ged_counter[val]
        percentage = count/len(ged_values)*100
        print(f"GED {val:.1f}: {count} pairs ({percentage:.2f}%)")

    if len(unique_values) > 10:
        print(f"... and {len(unique_values)-10} more values (see stats file for complete list)")

if __name__ == "__main__":
    main()
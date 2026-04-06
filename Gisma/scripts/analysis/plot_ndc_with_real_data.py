#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ndc_with_real_data.py
Plot NDC vs alpha with theoretical curve and actual measured data points.
Similar to paper Fig 7b, showing only tau=8.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Set font
plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['font.size'] = 60
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['ytick.major.size'] = 12


def load_data(csv_file):
    """Load statistics CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    return df


def load_real_ndc_data(real_data_file):
    """Load actual measured NDC data"""
    if not os.path.exists(real_data_file):
        print(f"Warning: Real NDC data file not found: {real_data_file}")
        return None

    # Read CSV, skip comment lines
    df = pd.read_csv(real_data_file, comment='#')

    # Filter out empty ndc values
    df = df.dropna(subset=['ndc'])
    df = df[df['ndc'] != '']

    if len(df) == 0:
        print(f"Warning: No real NDC data found in {real_data_file}")
        return None

    # Convert to numeric types
    df['alpha'] = pd.to_numeric(df['alpha'])
    df['ndc'] = pd.to_numeric(df['ndc'])

    print(f"Loaded {len(df)} real NDC data points")
    return df


def calculate_ndc_for_alpha(df, alpha, tau):
    """
    Calculate the estimated NDC value for a given alpha

    NDC(α) ≈ Σ[i=0 to φ-1] (|B(α·2^(i+1) + 2τ)| / |B(α·2^(i+1))|) · Cover(α·2^i, α·2^(i+1)) + |B(2α + τ)|
    """
    # Calculate phi (number of layers)
    max_r = df['r_k_actual'].max()
    if max_r >= 1e6:
        max_r = df[df['r_k_actual'] < 1e6]['r_k_actual'].max()

    phi = int(np.log2(max_r / alpha)) + 1

    # Group by r_integer, take the last record for each integer r
    df_by_r = df.groupby('r_integer').last().reset_index()

    def get_cluster_size(r):
        r_int = int(round(r))
        row = df_by_r[df_by_r['r_integer'] == r_int]
        if len(row) > 0:
            return row['max_cluster_size'].values[0]
        return 0

    def get_cover_count(r):
        r_int = int(round(r))
        row = df_by_r[df_by_r['r_integer'] == r_int]
        if len(row) > 0:
            return row['max_friends_count'].values[0]
        return 0

    # Calculate NDC
    ndc = 0

    for i in range(phi):
        r1 = alpha * (2 ** i)
        r2 = alpha * (2 ** (i + 1))

        cover_r1_2r1 = get_cover_count(r1)
        b_r2_plus_2tau = get_cluster_size(r2 + 2 * tau)
        b_r2 = get_cluster_size(r2)

        if b_r2 > 0:
            ratio = b_r2_plus_2tau / b_r2
            layer_contrib = ratio * cover_r1_2r1
        else:
            layer_contrib = 0

        ndc += layer_contrib

    # Add the final term |B(2*alpha + tau)|
    b_2alpha_plus_tau = get_cluster_size(2 * alpha + tau)
    ndc += b_2alpha_plus_tau

    return ndc, phi


def plot_ndc_with_real_data(df, tau, alpha_range, real_ndc_df, output_dir):
    """
    Plot NDC(alpha) with theoretical curve and actual data points.
    Show a single tau value, using dual y-axes.
    """
    fig, ax = plt.subplots(figsize=(18, 14))

    # Calculate theoretical NDC curve
    alphas = []
    ndcs = []

    print(f"\nCalculating theoretical NDC for tau = {tau}")
    for alpha in alpha_range:
        ndc, phi = calculate_ndc_for_alpha(df, alpha, tau)
        alphas.append(alpha)
        ndcs.append(ndc)

    # Plot theoretical curve (left y-axis)
    line1, = ax.plot(alphas, ndcs,
            linewidth=6,
            color='blue',
            label='Estimated NDC',
            zorder=5)

    # Find and mark theoretical minimum
    min_idx = np.argmin(ndcs)
    optimal_alpha = alphas[min_idx]
    min_ndc = ndcs[min_idx]

    ax.plot(optimal_alpha, min_ndc, marker='*', markersize=50,
            color='blue',
            markeredgecolor='black',
            markeredgewidth=3,
            linestyle='None',
            label='Estimated Optimal α',
            zorder=15)
    print(f"Optimal theoretical alpha: {optimal_alpha}, Min NDC = {min_ndc:.1f}")

    # Set left y-axis labels and format
    ax.set_xlabel(r'$\alpha$', fontsize=72)
    ax.set_ylabel(r'Estimated $\mathrm{NDC}$', fontsize=68, color='black')
    ax.tick_params(axis='y', labelcolor='blue', labelsize=64)
    ax.tick_params(axis='x', labelsize=64)
    ax.grid(True, alpha=0.3, linewidth=3.5)

    # Set left y-axis format to scientific notation (e.g., 4x10^4)
    def scientific_formatter(x, pos):
        if x == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(x))))
        coef = x / (10 ** exp)
        # Only show ticks with integer coefficients
        if abs(coef - round(coef)) < 0.01:
            return r'$%d\!\!\times\!\!10^{%d}$' % (int(round(coef)), exp)
        else:
            return ''
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # Create right y-axis for actual data points
    ax2 = ax.twinx()

    if real_ndc_df is not None and len(real_ndc_df) > 0:
        scatter = ax2.scatter(real_ndc_df['alpha'], real_ndc_df['ndc'],
                   s=400,  # point size
                   color='red',
                   marker='o',
                   edgecolors='black',
                   linewidths=3,
                   label='Actual NDC',
                   zorder=10)

        # Connect actual data points with lines
        sorted_df = real_ndc_df.sort_values('alpha')
        ax2.plot(sorted_df['alpha'], sorted_df['ndc'],
                linewidth=4,
                color='red',
                linestyle='--',
                alpha=0.7,
                zorder=9)

        print(f"Plotted {len(real_ndc_df)} real data points")

    # Set right y-axis range and labels
    ax2.set_ylim(1.8e4, 4e4)
    ax2.set_ylabel(r'Actual $\mathrm{NDC}$', fontsize=68, color='black')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=64)
    ax2.set_yticks([2e4, 3e4, 4e4])
    ax2.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # Merge legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=48, loc='upper right', frameon=False)

    plt.tight_layout()

    # Save image
    png_path = os.path.join(output_dir, f'ndc_with_real_data_tau{int(tau)}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\nSaved plot to {png_path}")
    plt.close()

    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot NDC(α) with theoretical curve and real data points"
    )
    parser.add_argument('--csv', type=str,
                        default='./alpha_selection/alpha_statistics_summary_alpha1.csv.backup',
                        help='Path to summary statistics CSV file (for theoretical curve)')
    parser.add_argument('--real_data', type=str,
                        default='./alpha_selection/real_ndc_data.csv',
                        help='Path to real NDC data CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='./alpha_selection/plots',
                        help='Output directory for plots')
    parser.add_argument('--tau', type=float, default=8.0,
                        help='Tau value (default: 8.0)')
    parser.add_argument('--alpha_min', type=float, default=1.0,
                        help='Minimum alpha to test')
    parser.add_argument('--alpha_max', type=float, default=20.0,
                        help='Maximum alpha to test')
    parser.add_argument('--alpha_step', type=float, default=1.0,
                        help='Alpha step size')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load theoretical data
    df = load_data(args.csv)
    if df is None:
        return 1

    # Load actual data (optional)
    real_ndc_df = load_real_ndc_data(args.real_data)

    # Generate alpha range
    alpha_range = np.arange(args.alpha_min, args.alpha_max + args.alpha_step, args.alpha_step)

    print("\n" + "="*60)
    print(f"Plotting NDC with tau = {args.tau}")
    print("="*60)

    plot_ndc_with_real_data(df, args.tau, alpha_range, real_ndc_df, args.output_dir)

    print("\n" + "="*60)
    print("Plot completed!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

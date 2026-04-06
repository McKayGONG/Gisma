#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ndc_formula.py
Calculate and visualize estimated NDC(alpha) based on formula
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
    print(f"Loaded {len(df)} rows")
    return df


def calculate_ndc_for_alpha(df, alpha, tau):
    """
    Calculate the estimated NDC value for a given alpha

    NDC(α) ≈ Σ[i=0 to φ-1] (|B(α·2^(i+1) + 2τ)| / |B(α·2^(i+1))|) · Cover(α·2^i, α·2^(i+1)) + |B(2α + τ)|

    Cover uses max, cluster size uses max
    """
    # Calculate phi (number of layers)
    max_r = df['r_k_actual'].max()
    if max_r >= 1e6:  # filter out initial INF values
        max_r = df[df['r_k_actual'] < 1e6]['r_k_actual'].max()

    phi = int(np.log2(max_r / alpha)) + 1

    print(f"\nCalculating NDC for alpha={alpha}, tau={tau}, phi={phi}")

    # Group by r_integer, take the last record for each integer r
    df_by_r = df.groupby('r_integer').last().reset_index()

    # Create interpolation functions to get values at any r
    def get_cluster_size(r):
        r_int = int(round(r))
        row = df_by_r[df_by_r['r_integer'] == r_int]
        if len(row) > 0:
            return row['max_cluster_size'].values[0]  # use max cluster size
        return 0

    def get_cover_count(r):
        r_int = int(round(r))
        row = df_by_r[df_by_r['r_integer'] == r_int]
        if len(row) > 0:
            return row['max_friends_count'].values[0]  # use max cover count
        return 0

    # Calculate NDC
    ndc = 0
    components = []

    for i in range(phi):
        r1 = alpha * (2 ** i)
        r2 = alpha * (2 ** (i + 1))

        # Cover(α·2^i, α·2^(i+1)) ≈ Cover(r1, 2·r1) = max_cover_count at r1
        cover_r1_2r1 = get_cover_count(r1)

        # |B(α·2^(i+1) + 2τ)|
        b_r2_plus_2tau = get_cluster_size(r2 + 2 * tau)

        # |B(α·2^(i+1))|
        b_r2 = get_cluster_size(r2)

        # Calculate this layer's contribution
        if b_r2 > 0:
            ratio = b_r2_plus_2tau / b_r2
            layer_contrib = ratio * cover_r1_2r1
        else:
            layer_contrib = 0

        ndc += layer_contrib
        components.append({
            'layer': i,
            'r1': r1,
            'r2': r2,
            'cover': cover_r1_2r1,
            'ratio': ratio if b_r2 > 0 else 0,
            'contribution': layer_contrib
        })

        print(f"  Layer {i}: r={r1:.1f} to {r2:.1f}, Cover={cover_r1_2r1:.1f}, Ratio={ratio if b_r2 > 0 else 0:.3f}, Contrib={layer_contrib:.1f}")

    # Add the final term |B(2*alpha + tau)|
    b_2alpha_plus_tau = get_cluster_size(2 * alpha + tau)
    ndc += b_2alpha_plus_tau

    print(f"  Final term: |B(2*{alpha} + {tau})| = {b_2alpha_plus_tau:.1f}")
    print(f"  Total NDC({alpha}) ~= {ndc:.1f}")

    return ndc, phi, components


def plot_ndc_vs_alpha_multiple_tau(df, tau_values, alpha_range, output_dir):
    """
    Plot NDC(alpha) vs alpha for multiple tau values on one figure
    """
    fig, ax = plt.subplots(figsize=(18, 14))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    # Store all optimal points for overlap detection
    optimal_points = []

    for idx, tau in enumerate(tau_values):
        alphas = []
        ndcs = []

        print(f"\n{'='*60}")
        print(f"Processing tau = {tau}")
        print(f"{'='*60}")

        for alpha in alpha_range:
            ndc, phi, _ = calculate_ndc_for_alpha(df, alpha, tau)
            alphas.append(alpha)
            ndcs.append(ndc)

        # Plot curve
        # If tau is integer, display as integer; otherwise display with decimal
        tau_label = f'τ = {int(tau)}' if tau == int(tau) else f'τ = {tau}'
        ax.plot(alphas, ndcs,
                marker=markers[idx % len(markers)],
                linewidth=6,
                markersize=16,
                color=colors[idx % len(colors)],
                label=tau_label,
                zorder=5)

        # Find and mark minimum
        min_idx = np.argmin(ndcs)
        optimal_alpha = alphas[min_idx]
        min_ndc = ndcs[min_idx]

        optimal_points.append((optimal_alpha, min_ndc, idx))

        print(f"Optimal alpha for tau={tau}: {optimal_alpha}, Min NDC = {min_ndc:.1f}")

    # Plot all star markers, detect overlap and add offset
    # Smaller tau processed first (draw at original position), larger tau later (with offset), smaller tau has higher zorder (on top)
    alpha_positions = {}  # track how many stars drawn at each alpha position
    for optimal_alpha, min_ndc, idx in optimal_points:
        # Check if there is already a star at this alpha position
        if optimal_alpha in alpha_positions:
            # Overlap detected, add horizontal offset
            offset = 0.3 * alpha_positions[optimal_alpha]
            alpha_positions[optimal_alpha] += 1
            plot_alpha = optimal_alpha + offset
        else:
            alpha_positions[optimal_alpha] = 1
            plot_alpha = optimal_alpha

        ax.plot(plot_alpha, min_ndc, marker='*', markersize=40,
                color=colors[idx],
                markeredgecolor='black',
                markeredgewidth=3,
                zorder=100 + (len(tau_values) - idx))  # smaller tau (smaller idx) has higher zorder

    ax.set_xlabel(r'$\alpha$', fontsize=72)
    ax.set_ylabel(r'$\mathrm{NDC}$', fontsize=68)
    ax.legend(fontsize=64, loc='upper right', frameon=False)
    ax.grid(True, alpha=0.3, linewidth=3.5)
    ax.tick_params(labelsize=64)

    # Set y-axis format to scientific notation (e.g., 4x10^4)
    def scientific_formatter(x, pos):
        if x == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(x))))
        coef = x / (10 ** exp)
        if abs(coef - round(coef)) < 0.01:
            return r'$%d\!\!\times\!\!10^{%d}$' % (int(round(coef)), exp)
        else:
            return ''
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # y-axis auto-adjusts range

    plt.tight_layout()

    png_path = os.path.join(output_dir, 'ndc_estimation_multi_tau.png')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n{'='*60}")
    print(f"Saved NDC estimation plot to {png_path}")
    print(f"{'='*60}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot NDC(α) estimation based on formula"
    )
    parser.add_argument('--csv', type=str,
                       default='./alpha_selection/alpha_statistics_summary_alpha1.csv.backup',
                       help='Path to summary statistics CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='./alpha_selection/plots',
                       help='Output directory for plots')
    parser.add_argument('--tau', type=float, nargs='+', default=[2.0, 4.0, 6.0, 8.0, 10.0],
                       help='Tau values (can specify multiple)')
    parser.add_argument('--alpha_min', type=float, default=1.0,
                       help='Minimum alpha to test')
    parser.add_argument('--alpha_max', type=float, default=50.0,
                       help='Maximum alpha to test')
    parser.add_argument('--alpha_step', type=float, default=1.0,
                       help='Alpha step size')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.csv)
    if df is None:
        return 1

    # Generate alpha range
    alpha_range = np.arange(args.alpha_min, args.alpha_max + args.alpha_step, args.alpha_step)

    print("\n" + "="*60)
    print("Calculating NDC estimation for different alpha values...")
    print(f"Tau values: {args.tau}")
    print("="*60)

    plot_ndc_vs_alpha_multiple_tau(df, args.tau, alpha_range, args.output_dir)

    print("\n" + "="*60)
    print("NDC estimation completed!")
    print(f"Plot saved to: {args.output_dir}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

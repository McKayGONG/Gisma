#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_doubling_and_cluster.py
Plot cover count and cluster size as functions of r
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Set font - enlarge all fonts
plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['font.size'] = 60  # significantly increase base font size
plt.rcParams['axes.linewidth'] = 4  # thicken border
plt.rcParams['xtick.major.width'] = 4  # thicken tick lines
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['xtick.major.size'] = 12  # lengthen tick lines
plt.rcParams['ytick.major.size'] = 12


def load_data(csv_file):
    """Load statistics CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    return df


def plot_cover_count(df, output_dir, r_min=1, r_max=50):
    """
    Plot cover count vs r
    """
    # Filter r in [r_min, r_max] range
    df_filtered = df[(df['r_integer'] >= r_min) & (df['r_integer'] <= r_max)]

    # Group by r_integer, take the last record for each integer r (state when fully reached that integer)
    df_plot = df_filtered.groupby('r_integer').last().reset_index()

    fig, ax = plt.subplots(figsize=(18, 10.5))

    # Plot max cover count - significantly enlarge line width and points
    ax.plot(df_plot['r_integer'], df_plot['max_friends_count'],
            marker='^', linewidth=7, markersize=22,
            label=r'Max $\lambda(p,r)$ over $p$', color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel(r'$r$', fontsize=72)
    ax.set_ylabel(r'$\mathrm{max}_p\:\lambda(p,r)$', fontsize=68)
    # Do not show title
    ax.grid(True, alpha=0.3, linewidth=3.5)

    # Enlarge tick labels
    ax.tick_params(labelsize=64)

    # Set y-axis format to k (thousands)
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Set x-axis range
    ax.set_xlim(r_min, r_max)

    # Draw vertical dashed line at r=14
    ax.axvline(x=14, color='black', linestyle='--', linewidth=4, alpha=0.8, zorder=10)

    # Shade the r < 14 region in gray
    ax.axvspan(r_min, 14, color='gray', alpha=0.2, zorder=0)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'cover_count_r{r_min}_to_{r_max}.png')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Saved cover count plot to {png_path}")
    plt.close()


def plot_ball_size(df, output_dir, r_min=1, r_max=50):
    """
    Plot ball size vs r
    """
    # Filter r in [r_min, r_max] range
    df_filtered = df[(df['r_integer'] >= r_min) & (df['r_integer'] <= r_max)]

    # Group by r_integer, take the last record for each integer r
    df_plot = df_filtered.groupby('r_integer').last().reset_index()

    fig, ax = plt.subplots(figsize=(18, 14))

    # Plot max ball size - significantly enlarge line width and points
    ax.plot(df_plot['r_integer'], df_plot['max_cluster_size'],
            marker='s', linewidth=7, markersize=22,
            label='Max Ball Size', color='orange', linestyle='--', alpha=0.7)

    ax.set_xlabel(r'$r$', fontsize=72)
    ax.set_ylabel(r'$|B(r)|$', fontsize=68)
    # Do not show title
    ax.grid(True, alpha=0.3, linewidth=3.5)

    # Enlarge tick labels
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

    # Set x-axis range
    ax.set_xlim(r_min, r_max)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'ball_size_r{r_min}_to_{r_max}.png')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Saved ball size plot to {png_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot doubling constant and ball size vs r"
    )
    parser.add_argument('--csv', type=str,
                       default='./alpha_selection/alpha_statistics_summary.csv',
                       help='Path to summary statistics CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='./alpha_selection/plots',
                       help='Output directory for plots')
    parser.add_argument('--r_min', type=int, default=1,
                       help='Minimum r value to plot')
    parser.add_argument('--r_max', type=int, default=50,
                       help='Maximum r value to plot')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.csv)
    if df is None:
        return 1

    # Generate plots
    print("\n" + "="*60)
    print(f"Generating plots for r = {args.r_min} to {args.r_max}...")
    print("="*60)

    plot_cover_count(df, args.output_dir, args.r_min, args.r_max)
    plot_ball_size(df, args.output_dir, args.r_min, args.r_max)

    print("\n" + "="*60)
    print("Plotting completed!")
    print(f"All plots saved to: {args.output_dir}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

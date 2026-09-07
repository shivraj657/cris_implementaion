"""
Plotting utilities for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

def create_all_plots(standard_results, community_results, lambda_05_results, 
                    partition, community_sizes, standard_k20_seeds, 
                    community_k20_seeds, results_dir):
    """
    Create all required plots.
    """
    print("   Creating plots...")
    
    # 1. Spread vs k comparison
    create_spread_vs_k_plot(standard_results, lambda_05_results, results_dir)
    
    # 2. Coverage vs lambda
    create_coverage_vs_lambda_plot(community_results, results_dir)
    
    # 3. Seed distribution comparison
    create_seed_distribution_plot(
        standard_k20_seeds, community_k20_seeds, 
        partition, community_sizes, results_dir
    )

def create_spread_vs_k_plot(standard_results, lambda_05_results, results_dir):
    """
    Create spread vs k comparison plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    k_values = [r['k'] for r in standard_results]
    standard_spreads = [r['spread'] for r in standard_results]
    community_spreads = [r['spread'] for r in lambda_05_results]
    
    # Plot lines
    plt.plot(k_values, standard_spreads, 'o-', label='Standard RIS', linewidth=2, markersize=8)
    plt.plot(k_values, community_spreads, 's-', label='Community-Aware RIS (λ=0.5)', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Seeds (k)')
    plt.ylabel('Estimated Influence Spread')
    plt.title('Influence Spread Comparison: Standard vs Community-Aware RIS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "spread_vs_k.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_coverage_vs_lambda_plot(community_results, results_dir):
    """
    Create coverage and entropy vs lambda plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    lambda_values = [r['lambda'] for r in community_results]
    coverages = [r['community_coverage'] for r in community_results]
    entropies = [r['entropy'] for r in community_results]
    spreads = [r['spread'] for r in community_results]
    
    # Plot 1: Coverage and Entropy vs Lambda
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(lambda_values, coverages, 'o-', color='blue', label='Community Coverage', linewidth=2, markersize=8)
    line2 = ax1_twin.plot(lambda_values, entropies, 's-', color='red', label='Seed Distribution Entropy', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Community Coverage', color='blue')
    ax1_twin.set_ylabel('Entropy', color='red')
    ax1.set_title('Community Coverage and Entropy vs Lambda')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # Plot 2: Spread vs Lambda
    ax2.plot(lambda_values, spreads, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Estimated Influence Spread')
    ax2.set_title('Influence Spread vs Lambda')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "coverage_vs_lambda.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_seed_distribution_plot(standard_k20_seeds, community_k20_seeds, 
                                partition, community_sizes, results_dir):
    """
    Create seed distribution comparison bar chart.
    """
    # Get seed counts per community
    standard_counts = Counter(partition[seed] for seed in standard_k20_seeds)
    community_counts = Counter(partition[seed] for seed in community_k20_seeds)
    
    # Get top 20 communities by size
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:20]
    top_comm_ids = [comm_id for comm_id, size in top_communities]
    
    # Prepare data for plotting
    standard_seeds = [standard_counts.get(comm_id, 0) for comm_id in top_comm_ids]
    community_seeds = [community_counts.get(comm_id, 0) for comm_id in top_comm_ids]
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(top_comm_ids))
    width = 0.35
    
    plt.bar(x - width/2, standard_seeds, width, label='Standard RIS', alpha=0.8)
    plt.bar(x + width/2, community_seeds, width, label='Community-Aware RIS (λ=0.5)', alpha=0.8)
    
    plt.xlabel('Community ID (Top 20 by Size)')
    plt.ylabel('Number of Seeds')
    plt.title('Seed Distribution Comparison: Top 20 Communities by Size')
    plt.xticks(x, [f'C{comm_id}' for comm_id in top_comm_ids], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / "seed_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
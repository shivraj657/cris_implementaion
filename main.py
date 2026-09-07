#!/usr/bin/env python3
"""
Community-Aware RIS Implementation on Slashdot Dataset
B.Tech Project (BTP) - Influence Maximization

This script implements and compares Standard RIS vs Community-Aware RIS
on the Slashdot social network dataset.
"""

import random
import time
import os
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Import our modules
from graph_loader import load_graph
from community_detect import detect_communities, save_community_stats
from ris import standard_ris
from community_ris import community_aware_ris
from evaluate import evaluate_seed_set, monte_carlo_verification
from plot import create_all_plots

def main():
    print("=" * 60)
    print("Community-Aware RIS on Slashdot Dataset")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Step 1: Load graph
    print("\n1. Loading graph...")
    dataset_file = "processed_dataset.txt"
    G_forward, G_reverse, nodes = load_graph(dataset_file)
    n = len(nodes)
    print(f"   Loaded graph: {n} nodes, {sum(len(adj) for adj in G_forward.values())} edges")
    
    # Step 2: Community detection
    print("\n2. Detecting communities...")
    partition, modularity, community_sizes = detect_communities(G_forward, nodes)
    num_communities = len(community_sizes)
    print(f"   Detected {num_communities} communities")
    print(f"   Modularity Q = {modularity:.4f}")
    print(f"   Top 10 communities by size: {sorted(community_sizes.values(), reverse=True)[:10]}")
    
    # Save community stats
    save_community_stats(partition, community_sizes, results_dir)
    
    # Step 3: Standard RIS experiments
    print("\n3. Running Standard RIS experiments...")
    k_values = [5, 10, 15, 20]
    theta = 10000  # Fixed for reproducibility
    
    standard_results = []
    
    for k in k_values:
        print(f"   Running Standard RIS for k={k}...")
        start_time = time.time()
        
        seed_set, rrr_sets = standard_ris(G_reverse, nodes, k, theta)
        
        runtime = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_seed_set(seed_set, rrr_sets, partition, community_sizes, n, theta)
        metrics['k'] = k
        metrics['runtime'] = runtime
        
        standard_results.append(metrics)
        
        print(f"     Spread: {metrics['spread']:.2f}, Coverage: {metrics['community_coverage']:.3f}, "
              f"Entropy: {metrics['entropy']:.3f}, Runtime: {runtime:.2f}s")
    
    # Step 4: Community-Aware RIS experiments
    print("\n4. Running Community-Aware RIS experiments...")
    
    # Lambda sweep at k=20
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    community_results = []
    
    for lambda_val in lambda_values:
        print(f"   Running Community-Aware RIS for k=20, λ={lambda_val}...")
        start_time = time.time()
        
        seed_set, rrr_sets = community_aware_ris(G_reverse, nodes, 20, theta, partition, community_sizes, lambda_val)
        
        runtime = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_seed_set(seed_set, rrr_sets, partition, community_sizes, n, theta)
        metrics['k'] = 20
        metrics['lambda'] = lambda_val
        metrics['runtime'] = runtime
        
        community_results.append(metrics)
        
        print(f"     Spread: {metrics['spread']:.2f}, Coverage: {metrics['community_coverage']:.3f}, "
              f"Entropy: {metrics['entropy']:.3f}, Communities with 0 seeds: {metrics['communities_with_0_seeds']}, "
              f"Runtime: {runtime:.2f}s")
    
    # Full k sweep at lambda=0.5
    print("\n   Running full k sweep at λ=0.5...")
    lambda_05_results = []
    
    for k in k_values:
        print(f"   Running Community-Aware RIS for k={k}, λ=0.5...")
        start_time = time.time()
        
        seed_set, rrr_sets = community_aware_ris(G_reverse, nodes, k, theta, partition, community_sizes, 0.5)
        
        runtime = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_seed_set(seed_set, rrr_sets, partition, community_sizes, n, theta)
        metrics['k'] = k
        metrics['lambda'] = 0.5
        metrics['runtime'] = runtime
        
        lambda_05_results.append(metrics)
        
        print(f"     Spread: {metrics['spread']:.2f}, Coverage: {metrics['community_coverage']:.3f}, "
              f"Entropy: {metrics['entropy']:.3f}, Runtime: {runtime:.2f}s")
    
    # Step 5: Monte Carlo verification (optional)
    print("\n5. Monte Carlo verification...")
    
    # Get seed sets for verification
    standard_k20_seeds, _ = standard_ris(G_reverse, nodes, 20, theta)
    community_k20_seeds, _ = community_aware_ris(G_reverse, nodes, 20, theta, partition, community_sizes, 0.5)
    
    print("   Verifying Standard RIS k=20...")
    mc_standard = monte_carlo_verification(G_forward, standard_k20_seeds, runs=500)
    print(f"     Monte Carlo spread: {mc_standard:.2f}")
    
    print("   Verifying Community-Aware RIS k=20, λ=0.5...")
    mc_community = monte_carlo_verification(G_forward, community_k20_seeds, runs=500)
    print(f"     Monte Carlo spread: {mc_community:.2f}")
    
    # Step 6: Save results and create plots
    print("\n6. Saving results and creating plots...")
    
    # Save CSV results
    import pandas as pd
    
    # Standard RIS results
    df_standard = pd.DataFrame(standard_results)
    df_standard.to_csv(results_dir / "results_standard_ris.csv", index=False)
    
    # Community RIS results (combine lambda sweep and k sweep)
    all_community_results = community_results + lambda_05_results
    df_community = pd.DataFrame(all_community_results)
    df_community.to_csv(results_dir / "results_community_ris.csv", index=False)
    
    # Create all plots
    create_all_plots(
        standard_results, 
        community_results, 
        lambda_05_results,
        partition, 
        community_sizes,
        standard_k20_seeds,
        community_k20_seeds,
        results_dir
    )
    
    print("\n" + "=" * 60)
    print("Experiments completed successfully!")
    print(f"Results saved to: {results_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
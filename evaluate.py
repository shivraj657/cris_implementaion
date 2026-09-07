"""
Evaluation metrics for seed sets.
"""

import random
import math
from collections import Counter, defaultdict

def evaluate_seed_set(seed_set, rrr_sets, partition, community_sizes, n, theta):
    """
    Evaluate a seed set using various metrics.
    
    Args:
        seed_set: List of selected seed nodes
        rrr_sets: List of RRR sets
        partition: dict {node_id: community_id}
        community_sizes: dict {community_id: size}
        n: Total number of nodes
        theta: Number of RRR sets
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate influence spread
    covered_count = 0
    for rrr_set in rrr_sets:
        if any(seed in rrr_set for seed in seed_set):
            covered_count += 1
    
    spread = (covered_count / theta) * n
    
    # Calculate community coverage
    seed_communities = set(partition[seed] for seed in seed_set)
    community_coverage = len(seed_communities) / len(community_sizes)
    
    # Calculate seed distribution entropy
    seed_comm_counts = Counter(partition[seed] for seed in seed_set)
    k = len(seed_set)
    
    if k > 0:
        entropy = 0
        for count in seed_comm_counts.values():
            p_c = count / k
            if p_c > 0:
                entropy -= p_c * math.log2(p_c)
    else:
        entropy = 0
    
    # Count communities with 0 seeds
    communities_with_0_seeds = len(community_sizes) - len(seed_communities)
    
    return {
        'spread': spread,
        'community_coverage': community_coverage,
        'entropy': entropy,
        'communities_with_0_seeds': communities_with_0_seeds
    }

def monte_carlo_verification(G_forward, seed_set, p=0.01, runs=500):
    """
    Verify spread estimate using Monte Carlo simulation of Independent Cascade.
    
    Args:
        G_forward: Forward adjacency list
        seed_set: List of seed nodes
        p: Propagation probability
        runs: Number of simulation runs
        
    Returns:
        float: Average spread across runs
    """
    total_spread = 0
    
    for run in range(runs):
        # Initialize activation
        activated = set(seed_set)
        newly_activated = set(seed_set)
        
        # Propagate until no new activations
        while newly_activated:
            next_activated = set()
            
            for u in newly_activated:
                for v in G_forward[u]:
                    if v not in activated:
                        # Try to activate v with probability p
                        if random.random() < p:
                            activated.add(v)
                            next_activated.add(v)
            
            newly_activated = next_activated
        
        total_spread += len(activated)
        
        if (run + 1) % 100 == 0:
            print(f"     Completed {run + 1}/{runs} MC simulations")
    
    return total_spread / runs
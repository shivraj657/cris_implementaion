"""
Community-Aware RIS implementation.
"""

import random
import math
from collections import defaultdict
from ris import generate_rrr_set

def calculate_quotas(community_sizes, k, n):
    """
    Calculate per-community quotas for seed selection.
    
    Args:
        community_sizes: dict {community_id: size}
        k: Total number of seeds
        n: Total number of nodes
        
    Returns:
        quotas: dict {community_id: quota}
    """
    quotas = {}
    total_quota = 0
    
    # Calculate initial quotas proportional to community size
    for comm_id, size in community_sizes.items():
        quota = max(1, round(k * size / n))
        quotas[comm_id] = quota
        total_quota += quota
    
    # Redistribute if total quota exceeds k
    if total_quota > k:
        # Sort communities by size (largest first)
        sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Reduce quotas starting from largest communities
        excess = total_quota - k
        for comm_id, size in sorted_communities:
            if excess <= 0:
                break
            reduction = min(quotas[comm_id] - 1, excess)
            if reduction > 0:
                quotas[comm_id] -= reduction
                excess -= reduction
    
    # Ensure we don't exceed k
    actual_total = sum(quotas.values())
    if actual_total > k:
        # Final adjustment: reduce largest communities
        sorted_communities = sorted(quotas.items(), key=lambda x: x[1], reverse=True)
        excess = actual_total - k
        for comm_id, quota in sorted_communities:
            if excess <= 0:
                break
            reduction = min(quota - 1, excess)
            if reduction > 0:
                quotas[comm_id] -= reduction
                excess -= reduction
    
    return quotas

def community_aware_ris(G_reverse, nodes, k, theta, partition, community_sizes, lambda_val):
    """
    Community-Aware RIS algorithm.
    
    Args:
        G_reverse: Reverse adjacency list
        nodes: Set of all node IDs
        k: Number of seeds to select
        theta: Number of RRR sets to generate
        partition: dict {node_id: community_id}
        community_sizes: dict {community_id: size}
        lambda_val: Trade-off parameter between spread and fairness
        
    Returns:
        seed_set: list of selected seed nodes
        rrr_sets: list of RRR sets for evaluation
    """
    print(f"     Generating {theta} RRR sets...")
    
    # Generate RRR sets (same as standard RIS)
    rrr_sets = []
    for i in range(theta):
        rrr_set = generate_rrr_set(G_reverse, nodes)
        rrr_sets.append(rrr_set)
        
        if (i + 1) % 1000 == 0:
            print(f"       Generated {i + 1}/{theta} RRR sets")
    
    print(f"     Running community-aware greedy selection (λ={lambda_val})...")
    
    # Calculate quotas
    n = len(nodes)
    quotas = calculate_quotas(community_sizes, k, n)
    
    print(f"       Community quotas: {dict(list(quotas.items())[:5])}...")  # Show first 5
    
    # Initialize tracking
    seed_set = []
    covered_sets = set()
    community_counts = {comm_id: 0 for comm_id in community_sizes.keys()}
    
    # Build node-to-RRR mapping for efficient lookup
    node_to_rrr = defaultdict(set)
    for i, rrr_set in enumerate(rrr_sets):
        for node in rrr_set:
            node_to_rrr[node].add(i)
    
    # Greedy selection with community awareness
    for round_num in range(k):
        best_node = None
        best_effective_gain = -1
        
        # Find node with maximum effective gain
        for node in nodes:
            if node in seed_set:
                continue
                
            # Get node's community
            comm_id = partition[node]
            
            # Calculate coverage gain
            coverage_gain = len(node_to_rrr[node] - covered_sets)
            
            if coverage_gain == 0:
                continue
            
            # Calculate effective gain with community penalty
            if community_counts[comm_id] >= quotas[comm_id]:
                # Community over quota - apply penalty
                penalty_factor = community_counts[comm_id] / max(quotas[comm_id], 1)
                effective_gain = coverage_gain * (1 - lambda_val * penalty_factor)
            else:
                # Community under quota - no penalty
                effective_gain = coverage_gain
            
            if effective_gain > best_effective_gain:
                best_effective_gain = effective_gain
                best_node = node
        
        if best_node is None:
            print(f"       Warning: No more nodes with positive gain at round {round_num + 1}")
            break
        
        # Add best node to seed set
        seed_set.append(best_node)
        best_comm = partition[best_node]
        community_counts[best_comm] += 1
        
        # Update covered sets
        newly_covered = node_to_rrr[best_node] - covered_sets
        covered_sets.update(newly_covered)
        
        print(f"       Round {round_num + 1}: Selected node {best_node} (comm {best_comm}), "
              f"effective gain = {best_effective_gain:.2f}")
    
    # Print final community distribution
    communities_with_seeds = sum(1 for count in community_counts.values() if count > 0)
    print(f"       Final: {communities_with_seeds}/{len(community_sizes)} communities have seeds")
    
    return seed_set, rrr_sets
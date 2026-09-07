"""
Standard RIS (Reverse Influence Sampling) implementation.
"""

import random
import heapq
from collections import defaultdict

def generate_rrr_set(G_reverse, nodes, p=0.01):
    """
    Generate one Random Reverse Reachable (RRR) set using backward BFS.
    
    Args:
        G_reverse: Reverse adjacency list
        nodes: Set of all node IDs
        p: Propagation probability (default 0.01)
        
    Returns:
        set: RRR set (nodes that can reach the target)
    """
    # Sample random target node
    target = random.choice(list(nodes))
    
    # Backward BFS with live-edge sampling
    rrr_set = set()
    queue = [target]
    visited = {target}
    
    while queue:
        v = queue.pop(0)
        rrr_set.add(v)
        
        # Check all incoming edges to v
        for u in G_reverse[v]:
            if u not in visited:
                # Sample edge with probability p
                if random.random() < p:
                    visited.add(u)
                    queue.append(u)
    
    return rrr_set

def standard_ris(G_reverse, nodes, k, theta):
    """
    Standard RIS algorithm.
    
    Args:
        G_reverse: Reverse adjacency list
        nodes: Set of all node IDs
        k: Number of seeds to select
        theta: Number of RRR sets to generate
        
    Returns:
        seed_set: list of selected seed nodes
        rrr_sets: list of RRR sets for evaluation
    """
    print(f"     Generating {theta} RRR sets...")
    
    # Generate RRR sets
    rrr_sets = []
    for i in range(theta):
        rrr_set = generate_rrr_set(G_reverse, nodes)
        rrr_sets.append(rrr_set)
        
        if (i + 1) % 1000 == 0:
            print(f"       Generated {i + 1}/{theta} RRR sets")
    
    print(f"     Running greedy seed selection...")
    
    # Greedy seed selection with efficient coverage tracking
    seed_set = []
    covered_sets = set()
    
    # Build node-to-RRR mapping for efficient lookup
    node_to_rrr = defaultdict(set)
    for i, rrr_set in enumerate(rrr_sets):
        for node in rrr_set:
            node_to_rrr[node].add(i)
    
    for round_num in range(k):
        best_node = None
        best_gain = -1
        
        # Find node with maximum marginal gain
        for node in nodes:
            if node in seed_set:
                continue
                
            # Count uncovered RRR sets containing this node
            gain = len(node_to_rrr[node] - covered_sets)
            
            if gain > best_gain:
                best_gain = gain
                best_node = node
        
        if best_node is None:
            print(f"       Warning: No more nodes with positive gain at round {round_num + 1}")
            break
            
        # Add best node to seed set
        seed_set.append(best_node)
        
        # Update covered sets
        newly_covered = node_to_rrr[best_node] - covered_sets
        covered_sets.update(newly_covered)
        
        print(f"       Round {round_num + 1}: Selected node {best_node}, gain = {best_gain}")
    
    return seed_set, rrr_sets
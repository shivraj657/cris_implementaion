"""
Community detection using Louvain algorithm.
"""

import random
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

def detect_communities(G_forward, nodes):
    """
    Detect communities using Louvain algorithm.
    
    Args:
        G_forward: Forward adjacency list
        nodes: Set of all node IDs
        
    Returns:
        partition: dict {node_id: community_id}
        modularity: float - modularity Q score
        community_sizes: dict {community_id: size}
    """
    print("   Converting to undirected NetworkX graph...")
    
    # Create undirected NetworkX graph
    G_undirected = nx.Graph()
    
    # Add all nodes
    G_undirected.add_nodes_from(nodes)
    
    # Add all edges (ignore direction)
    edge_count = 0
    for u in G_forward:
        for v in G_forward[u]:
            if not G_undirected.has_edge(u, v):
                G_undirected.add_edge(u, v)
                edge_count += 1
    
    print(f"   Undirected graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
    
    # Run Louvain community detection
    print("   Running Louvain community detection...")
    partition = community_louvain.best_partition(G_undirected, random_state=42)
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G_undirected)
    
    # Calculate community sizes
    community_counter = Counter(partition.values())
    community_sizes = dict(community_counter)
    
    print(f"   Detected {len(community_sizes)} communities")
    print(f"   Modularity Q = {modularity:.4f}")
    
    return partition, modularity, community_sizes

def save_community_stats(partition, community_sizes, results_dir):
    """
    Save community statistics and create size distribution plot.
    """
    print("   Saving community statistics...")
    
    # Create community size distribution histogram
    sizes = list(community_sizes.values())
    
    plt.figure(figsize=(10, 6))
    
    # Log-binned histogram
    log_bins = np.logspace(0, np.log10(max(sizes)), 30)
    plt.hist(sizes, bins=log_bins, alpha=0.7, edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Community Size (nodes)')
    plt.ylabel('Frequency')
    plt.title('Community Size Distribution (Louvain)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "community_size_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed community stats
    community_data = []
    for comm_id, size in community_sizes.items():
        community_data.append({
            'community_id': comm_id,
            'size': size
        })
    
    df = pd.DataFrame(community_data)
    df = df.sort_values('size', ascending=False)
    df.to_csv(results_dir / "community_stats.csv", index=False)
    
    print(f"   Saved community size distribution plot and stats")
    print(f"   Largest community: {max(sizes)} nodes")
    print(f"   Smallest community: {min(sizes)} nodes")
    print(f"   Average community size: {np.mean(sizes):.1f} nodes")
"""
Graph loading utilities for the Slashdot dataset.
"""

from collections import defaultdict

def load_graph(filename):
    """
    Load the edge list from file and build forward and reverse adjacency lists.
    
    Args:
        filename: Path to the edge list file
        
    Returns:
        G_forward: defaultdict(list) - forward adjacency list
        G_reverse: defaultdict(list) - reverse adjacency list  
        nodes: set - all node IDs
    """
    print(f"   Loading edge list from {filename}...")
    
    G_forward = defaultdict(list)
    G_reverse = defaultdict(list)
    nodes = set()
    
    edge_count = 0
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comment lines
            if line.startswith('#') or not line:
                continue
                
            # Parse edge
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
                
            # Add to adjacency lists
            G_forward[u].append(v)
            G_reverse[v].append(u)
            
            # Track nodes
            nodes.add(u)
            nodes.add(v)
            
            edge_count += 1
            
            if edge_count % 100000 == 0:
                print(f"     Loaded {edge_count} edges...")
    
    print(f"   Loaded {edge_count} edges, {len(nodes)} nodes")
    
    # Verify against known stats
    expected_nodes = 82140
    expected_edges = 549202
    
    if len(nodes) != expected_nodes:
        print(f"   WARNING: Expected {expected_nodes} nodes, got {len(nodes)}")
    if edge_count != expected_edges:
        print(f"   WARNING: Expected {expected_edges} edges, got {edge_count}")
    
    return G_forward, G_reverse, nodes
#!/usr/bin/env python3
"""
Complete RIS Implementation: Standard RIS vs Community-Aware RIS
================================================================

This file contains a complete implementation of both Standard RIS and Community-Aware RIS
algorithms for influence maximization on social networks.

Author: Community-Aware RIS Research Team
Date: 2024
"""

import random
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

# External dependencies
import networkx as nx
import community as cl

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class RISImplementation:
    """
    Complete implementation of Standard RIS and Community-Aware RIS algorithms.
    """
    
    def __init__(self, dataset_path="processed_dataset.txt", propagation_prob=0.01):
        """
        Initialize the RIS implementation.
        
        Args:
            dataset_path: Path to the edge list file
            propagation_prob: Propagation probability for Independent Cascade model
        """
        self.dataset_path = dataset_path
        self.p = propagation_prob
        self.G_forward = defaultdict(list)
        self.G_reverse = defaultdict(list)
        self.nodes = set()
        self.partition = {}
        self.community_sizes = {}
        self.modularity = 0.0
        
        print("=" * 60)
        print("RIS Implementation: Standard vs Community-Aware")
        print("=" * 60)
    
    def load_graph(self):
        """
        Load the edge list from file and build adjacency lists.
        """
        print(f"\n1. Loading graph from {self.dataset_path}...")
        
        edge_count = 0
        with open(self.dataset_path, 'r') as f:
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
                self.G_forward[u].append(v)
                self.G_reverse[v].append(u)
                
                # Track nodes
                self.nodes.add(u)
                self.nodes.add(v)
                
                edge_count += 1
                
                if edge_count % 100000 == 0:
                    print(f"   Loaded {edge_count} edges...")
        
        print(f"   Graph loaded: {len(self.nodes)} nodes, {edge_count} edges")
        return len(self.nodes), edge_count
    
    def detect_communities(self):
        """
        Detect communities using Louvain algorithm.
        """
        print("\n2. Detecting communities using Louvain algorithm...")
        
        # Create undirected NetworkX graph
        G_undirected = nx.Graph()
        G_undirected.add_nodes_from(self.nodes)
        
        # Add edges (ignore direction)
        for u in self.G_forward:
            for v in self.G_forward[u]:
                if not G_undirected.has_edge(u, v):
                    G_undirected.add_edge(u, v)
        
        print(f"   Undirected graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
        
        # Run Louvain community detection
        self.partition = cl.best_partition(G_undirected, random_state=42)
        self.modularity = cl.modularity(self.partition, G_undirected)
        
        # Calculate community sizes
        community_counter = Counter(self.partition.values())
        self.community_sizes = dict(community_counter)
        
        print(f"   Detected {len(self.community_sizes)} communities")
        print(f"   Modularity Q = {self.modularity:.4f}")
        
        # Show top communities
        top_communities = sorted(self.community_sizes.values(), reverse=True)[:10]
        print(f"   Top 10 communities by size: {top_communities}")
        
        return len(self.community_sizes), self.modularity
    
    def generate_rrr_set(self):
        """
        Generate one Random Reverse Reachable (RRR) set using backward BFS.
        
        Returns:
            set: RRR set (nodes that can reach the target)
        """
        # Sample random target node
        target = random.choice(list(self.nodes))
        
        # Backward BFS with live-edge sampling
        rrr_set = set()
        queue = [target]
        visited = {target}
        
        while queue:
            v = queue.pop(0)
            rrr_set.add(v)
            
            # Check all incoming edges to v
            for u in self.G_reverse[v]:
                if u not in visited:
                    # Sample edge with probability p
                    if random.random() < self.p:
                        visited.add(u)
                        queue.append(u)
        
        return rrr_set
    
    def standard_ris(self, k, theta=10000):
        """
        Standard RIS algorithm implementation.
        
        Args:
            k: Number of seeds to select
            theta: Number of RRR sets to generate
            
        Returns:
            tuple: (seed_set, rrr_sets, metrics)
        """
        print(f"\n   Running Standard RIS for k={k}, θ={theta}...")
        start_time = time.time()
        
        # Generate RRR sets
        print(f"     Generating {theta} RRR sets...")
        rrr_sets = []
        for i in range(theta):
            rrr_set = self.generate_rrr_set()
            rrr_sets.append(rrr_set)
            
            if (i + 1) % 2000 == 0:
                print(f"       Generated {i + 1}/{theta} RRR sets")
        
        # Greedy seed selection
        print(f"     Running greedy seed selection...")
        seed_set = []
        covered_sets = set()
        
        # Build node-to-RRR mapping for efficient lookup
        node_to_rrr = defaultdict(set)
        for i, rrr_set in enumerate(rrr_sets):
            for node in rrr_set:
                node_to_rrr[node].add(i)
        
        # Greedy selection
        for round_num in range(k):
            best_node = None
            best_gain = -1
            
            # Find node with maximum marginal gain
            for node in self.nodes:
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
            
            if round_num < 5 or (round_num + 1) % 5 == 0:
                print(f"       Round {round_num + 1}: Selected node {best_node}, gain = {best_gain}")
        
        runtime = time.time() - start_time
        
        # Calculate metrics
        metrics = self.evaluate_seed_set(seed_set, rrr_sets, len(self.nodes), theta)
        metrics['runtime'] = runtime
        
        print(f"     Results: Spread={metrics['spread']:.2f}, Coverage={metrics['community_coverage']:.3f}, "
              f"Entropy={metrics['entropy']:.3f}, Runtime={runtime:.2f}s")
        
        return seed_set, rrr_sets, metrics
    
    def calculate_quotas(self, k):
        """
        Calculate per-community quotas for seed selection.
        
        Args:
            k: Total number of seeds
            
        Returns:
            dict: {community_id: quota}
        """
        quotas = {}
        n = len(self.nodes)
        total_quota = 0
        
        # Calculate initial quotas proportional to community size
        for comm_id, size in self.community_sizes.items():
            quota = max(1, round(k * size / n))
            quotas[comm_id] = quota
            total_quota += quota
        
        # Redistribute if total quota exceeds k
        if total_quota > k:
            # Sort communities by size (largest first)
            sorted_communities = sorted(self.community_sizes.items(), key=lambda x: x[1], reverse=True)
            
            # Reduce quotas starting from largest communities
            excess = total_quota - k
            for comm_id, size in sorted_communities:
                if excess <= 0:
                    break
                reduction = min(quotas[comm_id] - 1, excess)
                if reduction > 0:
                    quotas[comm_id] -= reduction
                    excess -= reduction
        
        return quotas
    
    def community_aware_ris(self, k, lambda_val, theta=10000):
        """
        Community-Aware RIS algorithm implementation.
        
        Args:
            k: Number of seeds to select
            lambda_val: Trade-off parameter between spread and fairness
            theta: Number of RRR sets to generate
            
        Returns:
            tuple: (seed_set, rrr_sets, metrics)
        """
        print(f"\n   Running Community-Aware RIS for k={k}, λ={lambda_val}, θ={theta}...")
        start_time = time.time()
        
        # Generate RRR sets (same as standard RIS)
        print(f"     Generating {theta} RRR sets...")
        rrr_sets = []
        for i in range(theta):
            rrr_set = self.generate_rrr_set()
            rrr_sets.append(rrr_set)
            
            if (i + 1) % 2000 == 0:
                print(f"       Generated {i + 1}/{theta} RRR sets")
        
        # Calculate quotas
        quotas = self.calculate_quotas(k)
        print(f"     Community quotas calculated (showing first 5): {dict(list(quotas.items())[:5])}...")
        
        # Initialize tracking
        seed_set = []
        covered_sets = set()
        community_counts = {comm_id: 0 for comm_id in self.community_sizes.keys()}
        
        # Build node-to-RRR mapping for efficient lookup
        node_to_rrr = defaultdict(set)
        for i, rrr_set in enumerate(rrr_sets):
            for node in rrr_set:
                node_to_rrr[node].add(i)
        
        # Community-aware greedy selection
        print(f"     Running community-aware greedy selection...")
        for round_num in range(k):
            best_node = None
            best_effective_gain = -1
            
            # Find node with maximum effective gain
            for node in self.nodes:
                if node in seed_set:
                    continue
                
                # Get node's community
                comm_id = self.partition[node]
                
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
            best_comm = self.partition[best_node]
            community_counts[best_comm] += 1
            
            # Update covered sets
            newly_covered = node_to_rrr[best_node] - covered_sets
            covered_sets.update(newly_covered)
            
            if round_num < 5 or (round_num + 1) % 5 == 0:
                print(f"       Round {round_num + 1}: Selected node {best_node} (comm {best_comm}), "
                      f"effective gain = {best_effective_gain:.2f}")
        
        runtime = time.time() - start_time
        
        # Calculate metrics
        metrics = self.evaluate_seed_set(seed_set, rrr_sets, len(self.nodes), theta)
        metrics['runtime'] = runtime
        metrics['lambda'] = lambda_val
        
        # Print final community distribution
        communities_with_seeds = sum(1 for count in community_counts.values() if count > 0)
        print(f"     Final: {communities_with_seeds}/{len(self.community_sizes)} communities have seeds")
        print(f"     Results: Spread={metrics['spread']:.2f}, Coverage={metrics['community_coverage']:.3f}, "
              f"Entropy={metrics['entropy']:.3f}, Runtime={runtime:.2f}s")
        
        return seed_set, rrr_sets, metrics
    
    def evaluate_seed_set(self, seed_set, rrr_sets, n, theta):
        """
        Evaluate a seed set using various metrics.
        
        Args:
            seed_set: List of selected seed nodes
            rrr_sets: List of RRR sets
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
        seed_communities = set(self.partition[seed] for seed in seed_set)
        community_coverage = len(seed_communities) / len(self.community_sizes)
        
        # Calculate seed distribution entropy
        seed_comm_counts = Counter(self.partition[seed] for seed in seed_set)
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
        communities_with_0_seeds = len(self.community_sizes) - len(seed_communities)
        
        return {
            'spread': spread,
            'community_coverage': community_coverage,
            'entropy': entropy,
            'communities_with_0_seeds': communities_with_0_seeds
        }
    
    def monte_carlo_verification(self, seed_set, runs=500):
        """
        Verify spread estimate using Monte Carlo simulation of Independent Cascade.
        
        Args:
            seed_set: List of seed nodes
            runs: Number of simulation runs
            
        Returns:
            float: Average spread across runs
        """
        print(f"     Running Monte Carlo verification ({runs} simulations)...")
        total_spread = 0
        
        for run in range(runs):
            # Initialize activation
            activated = set(seed_set)
            newly_activated = set(seed_set)
            
            # Propagate until no new activations
            while newly_activated:
                next_activated = set()
                
                for u in newly_activated:
                    for v in self.G_forward[u]:
                        if v not in activated:
                            # Try to activate v with probability p
                            if random.random() < self.p:
                                activated.add(v)
                                next_activated.add(v)
                
                newly_activated = next_activated
            
            total_spread += len(activated)
            
            if (run + 1) % 100 == 0:
                print(f"       Completed {run + 1}/{runs} simulations")
        
        return total_spread / runs
    
    def create_comparison_plot(self, standard_results, community_results):
        """
        Create comparison plots for the results.
        """
        print("\n   Creating comparison plots...")
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Plot 1: Spread vs k comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        k_values = [r['k'] for r in standard_results]
        standard_spreads = [r['spread'] for r in standard_results]
        community_spreads = [r['spread'] for r in community_results if r.get('lambda') == 0.5]
        
        plt.plot(k_values, standard_spreads, 'o-', label='Standard RIS', linewidth=2, markersize=8)
        plt.plot(k_values, community_spreads, 's-', label='Community-Aware RIS (λ=0.5)', linewidth=2, markersize=8)
        plt.xlabel('Number of Seeds (k)')
        plt.ylabel('Estimated Influence Spread')
        plt.title('Influence Spread Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs k comparison
        plt.subplot(1, 2, 2)
        standard_coverage = [r['community_coverage'] for r in standard_results]
        community_coverage = [r['community_coverage'] for r in community_results if r.get('lambda') == 0.5]
        
        plt.plot(k_values, standard_coverage, 'o-', label='Standard RIS', linewidth=2, markersize=8)
        plt.plot(k_values, community_coverage, 's-', label='Community-Aware RIS (λ=0.5)', linewidth=2, markersize=8)
        plt.xlabel('Number of Seeds (k)')
        plt.ylabel('Community Coverage')
        plt.title('Community Coverage Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "ris_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Plots saved to {results_dir}/ris_comparison.png")
    
    def save_results(self, standard_results, community_results):
        """
        Save results to CSV files.
        """
        print("\n   Saving results to CSV files...")
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save standard RIS results
        df_standard = pd.DataFrame(standard_results)
        df_standard.to_csv(results_dir / "standard_ris_results.csv", index=False)
        
        # Save community RIS results
        df_community = pd.DataFrame(community_results)
        df_community.to_csv(results_dir / "community_ris_results.csv", index=False)
        
        print(f"     Results saved to {results_dir}/")
    
    def run_complete_experiment(self):
        """
        Run the complete experimental comparison between Standard RIS and Community-Aware RIS.
        """
        # Load graph and detect communities
        self.load_graph()
        self.detect_communities()
        
        # Experimental parameters
        k_values = [5, 10, 15, 20]
        lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        theta = 10000
        
        # Run Standard RIS experiments
        print("\n3. Running Standard RIS experiments...")
        standard_results = []
        
        for k in k_values:
            seed_set, rrr_sets, metrics = self.standard_ris(k, theta)
            metrics['k'] = k
            metrics['method'] = 'Standard RIS'
            standard_results.append(metrics)
        
        # Run Community-Aware RIS experiments
        print("\n4. Running Community-Aware RIS experiments...")
        community_results = []
        
        # Lambda sweep at k=20
        print("\n   Lambda sensitivity analysis (k=20):")
        for lambda_val in lambda_values:
            seed_set, rrr_sets, metrics = self.community_aware_ris(20, lambda_val, theta)
            metrics['k'] = 20
            metrics['method'] = 'Community-Aware RIS'
            community_results.append(metrics)
        
        # K sweep at lambda=0.5
        print("\n   K scalability analysis (λ=0.5):")
        for k in k_values:
            if k != 20:  # Already done above
                seed_set, rrr_sets, metrics = self.community_aware_ris(k, 0.5, theta)
                metrics['k'] = k
                metrics['method'] = 'Community-Aware RIS'
                community_results.append(metrics)
        
        # Monte Carlo verification
        print("\n5. Monte Carlo verification...")
        
        # Get final seed sets for verification
        standard_k20_seeds, _, _ = self.standard_ris(20, theta)
        community_k20_seeds, _, _ = self.community_aware_ris(20, 0.5, theta)
        
        print("   Verifying Standard RIS (k=20)...")
        mc_standard = self.monte_carlo_verification(standard_k20_seeds, 500)
        print(f"     Monte Carlo spread: {mc_standard:.2f}")
        
        print("   Verifying Community-Aware RIS (k=20, λ=0.5)...")
        mc_community = self.monte_carlo_verification(community_k20_seeds, 500)
        print(f"     Monte Carlo spread: {mc_community:.2f}")
        
        # Create visualizations and save results
        print("\n6. Creating visualizations and saving results...")
        self.create_comparison_plot(standard_results, community_results)
        self.save_results(standard_results, community_results)
        
        # Print summary
        self.print_summary(standard_results, community_results, mc_standard, mc_community)
        
        return standard_results, community_results
    
    def print_summary(self, standard_results, community_results, mc_standard, mc_community):
        """
        Print experimental summary.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENTAL SUMMARY")
        print("=" * 60)
        
        # Find k=20 results for comparison
        std_k20 = next(r for r in standard_results if r['k'] == 20)
        comm_k20 = next(r for r in community_results if r['k'] == 20 and r.get('lambda') == 0.5)
        
        print(f"\nDataset: {len(self.nodes)} nodes, {sum(len(adj) for adj in self.G_forward.values())} edges")
        print(f"Communities: {len(self.community_sizes)} (Q = {self.modularity:.4f})")
        
        print(f"\nStandard RIS (k=20):")
        print(f"  Spread: {std_k20['spread']:.2f}")
        print(f"  Community Coverage: {std_k20['community_coverage']:.1%}")
        print(f"  Entropy: {std_k20['entropy']:.3f}")
        print(f"  Monte Carlo: {mc_standard:.2f}")
        
        print(f"\nCommunity-Aware RIS (k=20, λ=0.5):")
        print(f"  Spread: {comm_k20['spread']:.2f} ({((comm_k20['spread']/std_k20['spread']-1)*100):+.1f}%)")
        print(f"  Community Coverage: {comm_k20['community_coverage']:.1%} ({comm_k20['community_coverage']/std_k20['community_coverage']:.1f}×)")
        print(f"  Entropy: {comm_k20['entropy']:.3f} ({((comm_k20['entropy']/std_k20['entropy']-1)*100):+.1f}%)")
        print(f"  Monte Carlo: {mc_community:.2f}")
        
        print(f"\nKey Finding: {comm_k20['community_coverage']/std_k20['community_coverage']:.1f}× better community coverage")
        print(f"with only {abs((comm_k20['spread']/std_k20['spread']-1)*100):.1f}% spread reduction")
        
        print("\n" + "=" * 60)


def main():
    """
    Main function to run the complete RIS comparison experiment.
    """
    # Initialize RIS implementation
    ris = RISImplementation("processed_dataset.txt", propagation_prob=0.01)
    
    # Run complete experiment
    try:
        standard_results, community_results = ris.run_complete_experiment()
        print("\nExperiment completed successfully!")
        print("Check the 'results/' directory for output files and plots.")
        
    except FileNotFoundError:
        print(f"\nError: Dataset file '{ris.dataset_path}' not found.")
        print("Please ensure the Slashdot dataset is available in the current directory.")
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        raise


if __name__ == "__main__":
    main()
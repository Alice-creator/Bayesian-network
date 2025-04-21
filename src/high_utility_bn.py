import numpy as np
from collections import defaultdict
import heapq
from src.bayesian_network import BayesianNetwork, Node
import itertools

class ItemsetNode(Node):
    """
    Extended Node class for representing itemsets in uncertain databases.
    """
    def __init__(self, name, items=None, utility=0.0):
        super().__init__(name)
        self.items = items or set()  # The items in this itemset
        self.utility = utility       # Utility value for this itemset
        self.support = 0.0           # Support value in the database
    
    def __repr__(self):
        return f"ItemsetNode({self.name}, items={self.items}, utility={self.utility:.4f})"


class HighUtilityBN(BayesianNetwork):
    """
    Extended Bayesian Network for high-utility itemset mining.
    """
    def __init__(self):
        super().__init__()
        self.total_utility = 0.0  # Total utility of the network
    
    def add_itemset_node(self, name, items=None, utility=0.0):
        """Add an itemset node to the network."""
        if name not in self.nodes:
            self.nodes[name] = ItemsetNode(name, items, utility)
        return self.nodes[name]
    
    def calculate_total_utility(self):
        """Calculate the total utility of the network."""
        self.total_utility = sum(node.utility for node in self.nodes.values() 
                                if isinstance(node, ItemsetNode))
        return self.total_utility
    
    def calculate_expected_utility(self, evidence=None):
        """
        Calculate the expected utility of the network given evidence.
        
        Args:
            evidence: Dictionary of observed variables {var_name: value}
            
        Returns:
            Expected utility value
        """
        if evidence is None:
            evidence = {}
            
        # Generate samples to estimate expected utility
        n_samples = 10000
        samples = self.sample(n_samples)
        
        # Filter samples consistent with evidence
        if evidence:
            samples = [s for s in samples if all(s[var] == val for var, val in evidence.items())]
            
        if not samples:
            return 0.0
            
        # Calculate expected utility from samples
        total_utility = 0.0
        for sample in samples:
            sample_utility = sum(self.nodes[node_name].utility 
                                for node_name, val in sample.items() 
                                if val is True and isinstance(self.nodes[node_name], ItemsetNode))
        
        return total_utility / len(samples)


def mine_top_k_high_utility_networks(database, k=10, min_utility=0.0):
    """
    Mine top-K high utility Bayesian networks from a database.
    
    Args:
        database: Dictionary mapping transaction IDs to list of items with utilities
        k: Number of top networks to find
        min_utility: Minimum utility threshold
        
    Returns:
        List of top-K high utility Bayesian networks
    """
    # Step 1: Extract frequent itemsets and their utilities
    itemsets = extract_high_utility_itemsets(database, min_utility)
    
    # Step 2: Build candidate Bayesian networks
    candidates = generate_candidate_networks(itemsets)
    
    # Step 3: Evaluate candidate networks and find top-K
    top_k_networks = find_top_k_networks(candidates, database, k)
    
    return top_k_networks


def extract_high_utility_itemsets(database, min_utility=0.0):
    """
    Extract high utility itemsets from a database.
    
    Args:
        database: Dictionary mapping transaction IDs to list of (item, utility) tuples
        min_utility: Minimum utility threshold
        
    Returns:
        Dictionary mapping itemsets to their utilities
    """
    # Itemset utilities: {frozenset(items): utility}
    itemset_utilities = defaultdict(float)
    
    # Extract single items first
    single_items = set()
    for transaction in database.values():
        for item, utility in transaction:
            single_items.add(item)
            itemset_utilities[frozenset([item])] += utility
    
    # Filter single items by min_utility
    high_utility_items = {item for item in single_items 
                         if itemset_utilities[frozenset([item])] >= min_utility}
    
    # Generate larger itemsets (level-wise approach)
    current_itemsets = [{item} for item in high_utility_items]
    high_utility_itemsets = {frozenset(itemset): itemset_utilities[frozenset(itemset)] 
                            for itemset in current_itemsets}
    
    k = 2  # Size of itemsets to generate
    while current_itemsets:
        # Generate candidate k-itemsets
        candidates = []
        for i in range(len(current_itemsets)):
            for j in range(i+1, len(current_itemsets)):
                itemset1 = current_itemsets[i]
                itemset2 = current_itemsets[j]
                # Check if first k-2 items are common to join
                if len(itemset1.intersection(itemset2)) == k-2:
                    candidate = itemset1.union(itemset2)
                    if len(candidate) == k:
                        candidates.append(candidate)
        
        # Calculate utilities for candidate itemsets
        for candidate in candidates:
            # Reset utility for this candidate
            itemset_utilities[frozenset(candidate)] = 0.0
            
            # Calculate utility across all transactions
            for transaction in database.values():
                # Check if all items in candidate are in the transaction
                transaction_items = {item for item, _ in transaction}
                if candidate.issubset(transaction_items):
                    # Sum utilities of items in the candidate
                    for item, utility in transaction:
                        if item in candidate:
                            itemset_utilities[frozenset(candidate)] += utility
        
        # Filter by min_utility
        current_itemsets = [candidate for candidate in candidates 
                          if itemset_utilities[frozenset(candidate)] >= min_utility]
        
        # Add to high utility itemsets
        for itemset in current_itemsets:
            high_utility_itemsets[frozenset(itemset)] = itemset_utilities[frozenset(itemset)]
        
        k += 1
    
    return high_utility_itemsets


def generate_candidate_networks(itemsets):
    """
    Generate candidate Bayesian networks from high utility itemsets.
    
    Args:
        itemsets: Dictionary mapping itemsets to their utilities
        
    Returns:
        List of candidate HighUtilityBN instances
    """
    candidates = []
    
    # Strategy 1: Create networks with parent-child relationships based on subset relationships
    bn1 = HighUtilityBN()
    
    # Add all itemsets as nodes
    for i, (itemset, utility) in enumerate(itemsets.items()):
        node_name = f"Itemset_{i}"
        bn1.add_itemset_node(node_name, set(itemset), utility)
    
    # Add edges based on subset relationships (superset -> subset)
    nodes = list(bn1.nodes.values())
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                node_i = nodes[i]
                node_j = nodes[j]
                # If node_i's itemset is a subset of node_j's itemset, add edge j->i
                if node_i.items.issubset(node_j.items) and node_i.items != node_j.items:
                    bn1.add_edge(node_j.name, node_i.name)
    
    candidates.append(bn1)
    
    # Strategy 2: Create networks based on co-occurrence patterns
    bn2 = HighUtilityBN()
    
    # Add all itemsets as nodes
    for i, (itemset, utility) in enumerate(itemsets.items()):
        node_name = f"Itemset_{i}"
        bn2.add_itemset_node(node_name, set(itemset), utility)
    
    # For simplicity, in this strategy we connect itemsets that share items
    nodes = list(bn2.nodes.values())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            # If node_i and node_j share items, add edge i->j (arbitrary direction)
            if node_i.items.intersection(node_j.items):
                bn2.add_edge(node_i.name, node_j.name)
    
    candidates.append(bn2)
    
    # We could generate more complex candidate networks here
    
    return candidates


def find_top_k_networks(candidates, database, k=10):
    """
    Evaluate candidate networks and find the top-K high utility networks.
    
    Args:
        candidates: List of candidate HighUtilityBN instances
        database: Dictionary mapping transaction IDs to list of (item, utility) tuples
        k: Number of top networks to find
        
    Returns:
        List of top-K high utility networks
    """
    # Calculate utilities for each network
    network_utilities = []
    
    for i, bn in enumerate(candidates):
        # If the network isn't acyclic, try to make it acyclic
        if not bn.is_acyclic():
            # Simple approach: remove edges until acyclic
            make_network_acyclic(bn)
        
        # Skip if we still have cycles
        if not bn.is_acyclic():
            continue
        
        # Calculate probabilities for each node based on database
        set_network_probabilities(bn, database)
        
        # Calculate network utility
        utility = bn.calculate_total_utility()
        
        # Add to heap (negative utility for max-heap)
        heapq.heappush(network_utilities, (-utility, i, bn))
    
    # Extract top-K networks
    top_k = []
    for _ in range(min(k, len(network_utilities))):
        if network_utilities:
            _, _, bn = heapq.heappop(network_utilities)
            top_k.append(bn)
    
    return top_k


def make_network_acyclic(bn):
    """
    Make a network acyclic by removing edges.
    
    Args:
        bn: BayesianNetwork instance to modify
    """
    while not bn.is_acyclic():
        # Find a cycle
        cycle = find_cycle(bn)
        if not cycle:
            break
            
        # Remove an edge from the cycle
        if cycle:
            # Choose edge with minimum utility difference
            min_diff = float('inf')
            edge_to_remove = None
            
            for i in range(len(cycle)):
                node1 = cycle[i]
                node2 = cycle[(i+1) % len(cycle)]
                
                diff = abs(node1.utility - node2.utility)
                if diff < min_diff:
                    min_diff = diff
                    edge_to_remove = (node1, node2)
            
            if edge_to_remove:
                # Remove the edge
                parent, child = edge_to_remove
                if child in parent.children:
                    parent.children.remove(child)
                if parent in child.parents:
                    child.parents.remove(parent)


def find_cycle(bn):
    """
    Find a cycle in the Bayesian network.
    
    Args:
        bn: BayesianNetwork instance
        
    Returns:
        List of nodes in a cycle, or empty list if no cycle found
    """
    visited = set()
    path = []
    path_set = set()
    
    def dfs(node):
        visited.add(node)
        path.append(node)
        path_set.add(node)
        
        for child in node.children:
            if child in path_set:
                # Found a cycle
                cycle_start = path.index(child)
                return path[cycle_start:]
            
            if child not in visited:
                cycle = dfs(child)
                if cycle:
                    return cycle
        
        path.pop()
        path_set.remove(node)
        return []
    
    for node in bn.nodes.values():
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                return cycle
    
    return []


def set_network_probabilities(bn, database):
    """
    Set the conditional probability tables for each node based on database.
    
    Args:
        bn: HighUtilityBN instance
        database: Dictionary mapping transaction IDs to list of (item, utility) tuples
    """
    # Count occurrences of each itemset
    itemset_counts = defaultdict(int)
    total_transactions = len(database)
    
    for transaction in database.values():
        transaction_items = {item for item, _ in transaction}
        
        for node in bn.nodes.values():
            if isinstance(node, ItemsetNode) and node.items.issubset(transaction_items):
                itemset_counts[node.name] += 1
    
    # For each node, calculate conditional probabilities
    for node_name, node in bn.nodes.items():
        if not isinstance(node, ItemsetNode):
            continue
            
        # If node has no parents, set prior probability
        if not node.parents:
            support = itemset_counts[node_name] / total_transactions
            node.support = support
            node.set_cpt({
                True: support,
                False: 1.0 - support
            })
        else:
            # For each combination of parent values
            cpt = {}
            parent_names = [parent.name for parent in node.parents]
            
            # Generate all combinations of parent values (True/False)
            for parent_values in itertools.product([True, False], repeat=len(parent_names)):
                parent_assignment = dict(zip(parent_names, parent_values))
                
                # Count matching transactions
                matching_transactions = 0
                matching_with_node = 0
                
                for transaction in database.values():
                    transaction_items = {item for item, _ in transaction}
                    
                    # Check if transaction matches parent assignment
                    matches_parents = True
                    for parent_name, parent_value in parent_assignment.items():
                        parent_node = bn.nodes[parent_name]
                        parent_present = parent_node.items.issubset(transaction_items)
                        
                        if (parent_value and not parent_present) or (not parent_value and parent_present):
                            matches_parents = False
                            break
                    
                    if matches_parents:
                        matching_transactions += 1
                        if node.items.issubset(transaction_items):
                            matching_with_node += 1
                
                # Calculate conditional probability
                p_true = 0.0
                if matching_transactions > 0:
                    p_true = matching_with_node / matching_transactions
                
                # Add to CPT
                parent_key = tuple(parent_values)
                cpt[parent_key] = {True: p_true, False: 1.0 - p_true}
            
            node.set_cpt(cpt)


# Example usage function
def example_usage():
    # Create a sample database
    database = {
        1: [('A', 5), ('C', 1), ('D', 1)],
        2: [('A', 2), ('C', 6), ('E', 2), ('G', 5)],
        3: [('A', 1), ('B', 2), ('C', 1), ('D', 6), ('E', 1)],
        4: [('B', 4), ('C', 3), ('D', 3), ('E', 1)],
        5: [('B', 2), ('C', 2), ('E', 1), ('F', 5), ('G', 2)],
        6: [('A', 1), ('C', 1), ('D', 1), ('G', 3)],
        7: [('B', 1), ('C', 3), ('D', 1), ('E', 3), ('F', 2)],
        8: [('A', 4), ('C', 1), ('D', 2), ('E', 1)]
    }
    
    # Mine top-5 high utility networks
    min_utility = 5.0
    k = 5
    
    print(f"Mining top-{k} high utility Bayesian networks (min_utility={min_utility})...")
    networks = mine_top_k_high_utility_networks(database, k, min_utility)
    
    # Display results
    print(f"\nFound {len(networks)} high utility Bayesian networks:")
    for i, bn in enumerate(networks, 1):
        print(f"\nNetwork {i}:")
        print(f"Total Utility: {bn.calculate_total_utility():.2f}")
        print("Structure:")
        bn.print_structure()
        print("Nodes:")
        for node_name, node in bn.nodes.items():
            if isinstance(node, ItemsetNode):
                print(f"  {node_name}: items={node.items}, utility={node.utility:.2f}")


if __name__ == "__main__":
    example_usage() 
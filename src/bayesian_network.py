import numpy as np
from collections import defaultdict, deque
import copy

class Node:
    """
    A node in a Bayesian network representing a random variable.
    """
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.cpt = None  # Conditional Probability Table
    
    def add_parent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)
            if self not in parent.children:
                parent.children.append(self)
    
    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
            if self not in child.parents:
                child.add_parent(self)
    
    def set_cpt(self, cpt):
        """
        Set the conditional probability table for this node.
        
        Args:
            cpt: Dictionary mapping parent value combinations to probability distributions.
                 For a node with no parents, this is simply a probability distribution.
        """
        self.cpt = cpt
    
    def __repr__(self):
        return f"Node({self.name})"


class BayesianNetwork:
    """
    A Bayesian network, represented as a directed acyclic graph (DAG).
    """
    def __init__(self):
        self.nodes = {}  # name -> Node
    
    def add_node(self, name):
        """Add a node to the network if it doesn't already exist."""
        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]
    
    def add_edge(self, parent_name, child_name):
        """Add an edge from parent to child."""
        parent = self.add_node(parent_name)
        child = self.add_node(child_name)
        parent.add_child(child)
    
    def is_acyclic(self):
        """Check if the network is acyclic."""
        visited = set()
        path = set()
        
        def dfs(node):
            visited.add(node)
            path.add(node)
            
            for child in node.children:
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in path:
                    return True
            
            path.remove(node)
            return False
        
        for node in self.nodes.values():
            if node not in visited:
                if dfs(node):
                    return False
        
        return True
    
    def topological_sort(self):
        """Return nodes in topological order."""
        if not self.is_acyclic():
            raise ValueError("Network contains cycles")
        
        result = []
        visited = set()
        
        def dfs(node):
            visited.add(node)
            for child in node.children:
                if child not in visited:
                    dfs(child)
            result.append(node)
        
        # Find all root nodes (nodes with no parents)
        roots = [node for node in self.nodes.values() if not node.parents]
        
        # If no roots, start from any node
        if not roots and self.nodes:
            roots = [next(iter(self.nodes.values()))]
        
        for root in roots:
            if root not in visited:
                dfs(root)
        
        # Add any remaining unvisited nodes
        for node in self.nodes.values():
            if node not in visited:
                dfs(node)
        
        return list(reversed(result))
    
    def query(self, query_vars, evidence=None):
        """
        Basic inference for querying the network.
        This is a simple implementation using exact inference.
        
        Args:
            query_vars: Dictionary of variables to query {var_name: value}
            evidence: Dictionary of observed variables {var_name: value}
            
        Returns:
            Probability of the query variables given the evidence
        """
        if evidence is None:
            evidence = {}
        
        # For now, we'll use a simple approach - enumerate all possible values
        # A more efficient implementation would use Variable Elimination or other algorithms
        
        # Create a complete assignment by combining query and evidence
        assignment = {**query_vars, **evidence}
        
        # Check if assignment is complete
        if len(assignment) != len(self.nodes):
            missing_vars = set(self.nodes.keys()) - set(assignment.keys())
            raise ValueError(f"Incomplete assignment: missing variables {missing_vars}")
        
        # Calculate probability of this assignment
        probability = 1.0
        ordered_nodes = self.topological_sort()
        
        for node in ordered_nodes:
            var_name = node.name
            var_value = assignment[var_name]
            
            # Get parent values
            parent_values = tuple(assignment[parent.name] for parent in node.parents)
            
            # Look up probability in CPT
            if not node.parents:
                # Node has no parents
                prob = node.cpt[var_value]
            else:
                # Node has parents
                prob = node.cpt[parent_values][var_value]
            
            probability *= prob
        
        return probability
    
    def sample(self, n_samples=1):
        """
        Generate samples from the Bayesian network.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of dictionaries, where each dictionary maps variable names to values
        """
        ordered_nodes = self.topological_sort()
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for node in ordered_nodes:
                # Get parent values
                parent_values = tuple(sample[parent.name] for parent in node.parents)
                
                # Sample from conditional distribution
                if not node.parents:
                    # Node has no parents
                    dist = node.cpt
                    values = list(dist.keys())
                    probs = list(dist.values())
                else:
                    # Node has parents
                    dist = node.cpt[parent_values]
                    values = list(dist.keys())
                    probs = list(dist.values())
                
                # Sample according to distribution
                value = np.random.choice(values, p=probs)
                sample[node.name] = value
            
            samples.append(sample)
        
        return samples

    def print_structure(self):
        """Print the structure of the Bayesian network."""
        for node_name, node in self.nodes.items():
            parents = [parent.name for parent in node.parents]
            print(f"{node_name} | Parents: {parents}")
            
    def __repr__(self):
        return f"BayesianNetwork(nodes={list(self.nodes.keys())})"


class VariableElimination:
    """
    Implementation of the Variable Elimination algorithm for Bayesian networks.
    """
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
        
    def _factor_product(self, factor1, factor2):
        """Compute the product of two factors."""
        # To be implemented
        pass
    
    def _sum_out(self, factor, variable):
        """Sum out a variable from a factor."""
        # To be implemented
        pass
    
    def query(self, query_vars, evidence=None):
        """
        Perform inference using variable elimination.
        
        Args:
            query_vars: Dictionary of variables to query {var_name: value}
            evidence: Dictionary of observed variables {var_name: value}
            
        Returns:
            Probability of the query variables given the evidence
        """
        # To be implemented
        pass 
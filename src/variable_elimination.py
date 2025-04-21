import numpy as np
from collections import defaultdict
import itertools
from bayesian_network import BayesianNetwork

class Factor:
    """
    A factor in a Bayesian network, representing a function from variable assignments to real values.
    """
    def __init__(self, variables, values=None):
        """
        Initialize a factor.
        
        Args:
            variables: List of variable names that this factor depends on
            values: Dictionary mapping variable assignments to values (optional)
        """
        self.variables = list(variables)
        self.values = values or {}
        
    def get_value(self, assignment):
        """Get the value for a variable assignment."""
        # Convert assignment to a tuple of values in the correct order
        key = tuple(assignment[var] for var in self.variables)
        return self.values.get(key, 0.0)
    
    def set_value(self, assignment, value):
        """Set the value for a variable assignment."""
        key = tuple(assignment[var] for var in self.variables)
        self.values[key] = value
    
    def multiply(self, other):
        """Multiply this factor by another factor."""
        # Create a new factor that includes all variables from both factors
        all_variables = list(set(self.variables) | set(other.variables))
        result = Factor(all_variables)
        
        # For each possible assignment to all variables
        variable_domains = {var: [True, False] for var in all_variables}  # Assuming binary variables
        
        for assignment in self._get_all_assignments(variable_domains):
            # Get the value from each factor and multiply them
            value1 = self.get_value(assignment)
            value2 = other.get_value(assignment)
            result.set_value(assignment, value1 * value2)
        
        return result
    
    def marginalize(self, variable):
        """Marginalize (sum) over a variable."""
        if variable not in self.variables:
            return self
        
        # Create a new factor without the marginalized variable
        new_variables = [v for v in self.variables if v != variable]
        result = Factor(new_variables)
        
        # For each possible assignment to the remaining variables
        variable_domains = {var: [True, False] for var in self.variables}  # Assuming binary variables
        
        for assignment in self._get_all_assignments(variable_domains):
            # Skip this assignment if we've already processed an equivalent one
            assignment_key = tuple(assignment[var] for var in new_variables)
            if assignment_key in result.values:
                continue
            
            # Sum over the values for the marginalized variable
            assignment_copy = assignment.copy()
            sum_value = 0
            
            for val in [True, False]:  # Assuming binary variables
                assignment_copy[variable] = val
                sum_value += self.get_value(assignment_copy)
            
            # Set the summed value in the result factor
            result.set_value({k: assignment[k] for k in new_variables}, sum_value)
        
        return result
    
    def normalize(self):
        """Normalize the factor to sum to 1."""
        total = sum(self.values.values())
        if total == 0:
            return self  # Cannot normalize
        
        result = Factor(self.variables)
        for assignment, value in self.values.items():
            result.values[assignment] = value / total
        
        return result
    
    def _get_all_assignments(self, variable_domains):
        """Get all possible assignments for the variables."""
        variables = list(variable_domains.keys())
        domains = [variable_domains[var] for var in variables]
        
        for values in itertools.product(*domains):
            yield {variables[i]: values[i] for i in range(len(variables))}


class VariableElimination:
    """
    Implementation of the Variable Elimination algorithm for Bayesian network inference.
    """
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
    
    def query(self, query_variables, evidence=None):
        """
        Perform inference using variable elimination.
        
        Args:
            query_variables: Dictionary of variables to query {var_name: value}
            evidence: Dictionary of observed variables {var_name: value}
            
        Returns:
            Probability of the query variables given the evidence
        """
        if evidence is None:
            evidence = {}
        
        # Create a factor for each variable in the network
        factors = []
        for node_name, node in self.bn.nodes.items():
            scope = [parent.name for parent in node.parents] + [node_name]
            factor = self._make_factor_from_node(node, scope)
            factors.append(factor)
        
        # Incorporate evidence by reducing factors
        for var, val in evidence.items():
            factors = self._reduce_factors(factors, var, val)
        
        # Determine elimination order (this is a simple heuristic)
        elimination_order = self._get_elimination_order(
            set(self.bn.nodes.keys()) - set(query_variables.keys()) - set(evidence.keys())
        )
        
        # Eliminate variables one by one
        for var in elimination_order:
            # Find all factors that mention this variable
            relevant_factors = [f for f in factors if var in f.variables]
            
            if not relevant_factors:
                continue
            
            # Remove these factors from the list
            factors = [f for f in factors if var not in f.variables]
            
            # Multiply all relevant factors
            product = relevant_factors[0]
            for factor in relevant_factors[1:]:
                product = product.multiply(factor)
            
            # Sum out the variable
            summed_out = product.marginalize(var)
            
            # Add the resulting factor back to the list
            factors.append(summed_out)
        
        # Multiply remaining factors
        if not factors:
            return 0.0
        
        result = factors[0]
        for factor in factors[1:]:
            result = result.multiply(factor)
        
        # Normalize to get conditional probabilities
        return result.normalize()
    
    def _make_factor_from_node(self, node, scope):
        """Create a factor from a node's CPT."""
        factor = Factor(scope)
        
        # If the node has no parents, the factor is just the prior
        if not node.parents:
            for value, prob in node.cpt.items():
                factor.set_value({node.name: value}, prob)
            return factor
        
        # Otherwise, create a factor from the CPT
        parent_names = [parent.name for parent in node.parents]
        
        # For each combination of parent values
        parent_domains = {parent: [True, False] for parent in parent_names}  # Assuming binary variables
        
        for parent_assignment in self._get_all_assignments(parent_domains):
            parent_values = tuple(parent_assignment[parent] for parent in parent_names)
            
            # For each value of the node
            for value, prob in node.cpt[parent_values].items():
                # Create a complete assignment
                assignment = parent_assignment.copy()
                assignment[node.name] = value
                
                # Set the probability in the factor
                factor.set_value(assignment, prob)
        
        return factor
    
    def _reduce_factors(self, factors, var, val):
        """Reduce factors by incorporating evidence."""
        reduced_factors = []
        
        for factor in factors:
            if var not in factor.variables:
                reduced_factors.append(factor)
                continue
            
            # Create a new factor with the variable fixed to its observed value
            new_variables = [v for v in factor.variables if v != var]
            new_factor = Factor(new_variables)
            
            # For each assignment to the remaining variables
            variable_domains = {v: [True, False] for v in new_variables}  # Assuming binary variables
            
            for assignment in self._get_all_assignments(variable_domains):
                # Add the evidence variable with its fixed value
                full_assignment = assignment.copy()
                full_assignment[var] = val
                
                # Set the value in the new factor
                new_factor.set_value(assignment, factor.get_value(full_assignment))
            
            reduced_factors.append(new_factor)
        
        return reduced_factors
    
    def _get_elimination_order(self, variables):
        """
        Determine the elimination order for variables.
        This is a simple implementation that uses the min-fill heuristic.
        """
        variables = list(variables)
        return variables  # For simplicity, we'll use the natural order
    
    def _get_all_assignments(self, variable_domains):
        """Get all possible assignments for the variables."""
        variables = list(variable_domains.keys())
        domains = [variable_domains[var] for var in variables]
        
        for values in itertools.product(*domains):
            yield {variables[i]: values[i] for i in range(len(variables))} 
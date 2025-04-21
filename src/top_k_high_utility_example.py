import numpy as np
from src.high_utility_bn import mine_top_k_high_utility_networks
import sys

# Try importing visualization libraries, but make them optional
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    visualization_available = True
except ImportError:
    visualization_available = False
    print("Warning: matplotlib or networkx not available. Visualizations will be skipped.")

def create_synthetic_database(num_transactions=100, num_items=10, max_items_per_transaction=5, 
                             max_utility=10, random_seed=42):
    """
    Create a synthetic database with random transactions and utilities.
    
    Args:
        num_transactions: Number of transactions
        num_items: Number of unique items
        max_items_per_transaction: Maximum number of items per transaction
        max_utility: Maximum utility value for an item
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping transaction IDs to list of (item, utility) tuples
    """
    np.random.seed(random_seed)
    
    # Create items (using letters for simplicity)
    items = [chr(65 + i) for i in range(min(num_items, 26))]
    
    # Create database
    database = {}
    
    for tid in range(1, num_transactions + 1):
        # Decide number of items in this transaction
        num_items_in_transaction = np.random.randint(1, max_items_per_transaction + 1)
        
        # Select items for this transaction (without replacement)
        transaction_items = np.random.choice(items, size=num_items_in_transaction, replace=False)
        
        # Assign utilities to items
        transaction = []
        for item in transaction_items:
            utility = np.random.randint(1, max_utility + 1)
            transaction.append((item, utility))
        
        database[tid] = transaction
    
    return database

def calculate_statistics(database):
    """
    Calculate basic statistics about the database.
    
    Args:
        database: Dictionary mapping transaction IDs to list of (item, utility) tuples
        
    Returns:
        Dictionary of statistics
    """
    # Count items and their utilities
    item_occurrences = {}
    item_utilities = {}
    total_utility = 0
    
    for tid, transaction in database.items():
        for item, utility in transaction:
            item_occurrences[item] = item_occurrences.get(item, 0) + 1
            item_utilities[item] = item_utilities.get(item, 0) + utility
            total_utility += utility
    
    # Calculate statistics
    num_transactions = len(database)
    num_unique_items = len(item_occurrences)
    avg_transaction_length = sum(len(transaction) for transaction in database.values()) / num_transactions
    avg_item_utility = sum(item_utilities.values()) / sum(item_occurrences.values())
    
    return {
        "num_transactions": num_transactions,
        "num_unique_items": num_unique_items,
        "avg_transaction_length": avg_transaction_length,
        "avg_item_utility": avg_item_utility,
        "total_utility": total_utility,
        "item_occurrences": item_occurrences,
        "item_utilities": item_utilities
    }

def print_database_sample(database, max_transactions=5):
    """Print a sample of the database."""
    print(f"Database sample (showing up to {max_transactions} transactions):")
    for tid in list(database.keys())[:max_transactions]:
        transaction = database[tid]
        items_str = ', '.join([f"{item}({utility})" for item, utility in transaction])
        print(f"  Transaction {tid}: {items_str}")
    if len(database) > max_transactions:
        print(f"  ... and {len(database) - max_transactions} more transactions")

def visualize_network(bn, title=None):
    """
    Visualize a Bayesian network as a directed graph.
    
    Args:
        bn: HighUtilityBN instance
        title: Title for the plot
    """
    if not visualization_available:
        print(f"Skipping visualization for {title} (libraries not available)")
        return
    
    try:
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_name, node in bn.nodes.items():
            # Node label format: name (items) utility
            items_str = ','.join(sorted(node.items))
            label = f"{node_name}\n({items_str})\n{node.utility:.1f}"
            G.add_node(node_name, label=label, utility=node.utility)
        
        # Add edges
        for node_name, node in bn.nodes.items():
            for child in node.children:
                G.add_edge(node_name, child.name)
        
        # Create a custom layout
        pos = nx.spring_layout(G, seed=42)
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Draw the nodes
        node_utilities = [bn.nodes[node].utility * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_utilities, alpha=0.8, 
                              node_color=node_utilities, cmap=plt.cm.YlOrRd)
        
        # Draw the edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0, alpha=0.5)
        
        # Draw the labels
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif')
        
        # Add a title
        if title:
            plt.title(title)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        filename = f"{title.replace(' ', '_')}.png" if title else "network.png"
        plt.savefig(filename)
        print(f"Network visualization saved to {filename}")
        
        # Try to show the plot (might not work in all environments)
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            
    except Exception as e:
        print(f"Error during visualization: {e}")

def example_mining():
    """Example of mining top-K high utility Bayesian networks."""
    # Create a synthetic database
    print("Creating synthetic database...")
    database = create_synthetic_database(
        num_transactions=50,
        num_items=8,
        max_items_per_transaction=4,
        max_utility=10,
        random_seed=42
    )
    
    # Print database statistics
    stats = calculate_statistics(database)
    print(f"\nDatabase Statistics:")
    print(f"  Transactions: {stats['num_transactions']}")
    print(f"  Unique items: {stats['num_unique_items']}")
    print(f"  Avg transaction length: {stats['avg_transaction_length']:.2f}")
    print(f"  Avg item utility: {stats['avg_item_utility']:.2f}")
    print(f"  Total utility: {stats['total_utility']}")
    
    # Print sample of the database
    print_database_sample(database)
    
    # Mine top-K high utility networks
    k = 3
    min_utility = 50.0  # Adjust based on your database
    
    print(f"\nMining top-{k} high utility Bayesian networks (min_utility={min_utility})...")
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
            items_str = ','.join(sorted(node.items)) if hasattr(node, 'items') else ''
            utility = f", utility={node.utility:.2f}" if hasattr(node, 'utility') else ''
            print(f"  {node_name}: items=({items_str}){utility}")
        
        # Visualize the network
        if visualization_available and '--no-plots' not in sys.argv:
            try:
                visualize_network(bn, title=f"Network {i} (Utility: {bn.calculate_total_utility():.2f})")
            except Exception as e:
                print(f"Visualization error: {e}")
        else:
            print("Skipping visualization (no display available or --no-plots specified)")

def main():
    example_mining()

if __name__ == "__main__":
    main() 
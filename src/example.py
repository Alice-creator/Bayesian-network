import numpy as np
from bayesian_network import BayesianNetwork, Node

def create_sprinkler_network():
    """
    Create the classic 'sprinkler' Bayesian network example:
    
    Cloudy -> Sprinkler -> Wet Grass
        \                /
         \--> Rain -----/
         
    Variables:
    - Cloudy (True/False)
    - Sprinkler (True/False)
    - Rain (True/False)
    - Wet Grass (True/False)
    """
    bn = BayesianNetwork()
    
    # Add nodes
    cloudy = bn.add_node("Cloudy")
    sprinkler = bn.add_node("Sprinkler")
    rain = bn.add_node("Rain")
    wet_grass = bn.add_node("WetGrass")
    
    # Add edges
    bn.add_edge("Cloudy", "Sprinkler")
    bn.add_edge("Cloudy", "Rain")
    bn.add_edge("Sprinkler", "WetGrass")
    bn.add_edge("Rain", "WetGrass")
    
    # Set CPTs (Conditional Probability Tables)
    
    # P(Cloudy)
    cloudy.set_cpt({
        True: 0.5,
        False: 0.5
    })
    
    # P(Sprinkler | Cloudy)
    sprinkler.set_cpt({
        (True,): {True: 0.1, False: 0.9},   # P(Sprinkler | Cloudy=True)
        (False,): {True: 0.5, False: 0.5}   # P(Sprinkler | Cloudy=False)
    })
    
    # P(Rain | Cloudy)
    rain.set_cpt({
        (True,): {True: 0.8, False: 0.2},   # P(Rain | Cloudy=True)
        (False,): {True: 0.2, False: 0.8}   # P(Rain | Cloudy=False)
    })
    
    # P(WetGrass | Sprinkler, Rain)
    wet_grass.set_cpt({
        (True, True): {True: 0.99, False: 0.01},    # P(WetGrass | Sprinkler=True, Rain=True)
        (True, False): {True: 0.9, False: 0.1},     # P(WetGrass | Sprinkler=True, Rain=False)
        (False, True): {True: 0.9, False: 0.1},     # P(WetGrass | Sprinkler=False, Rain=True)
        (False, False): {True: 0.0, False: 1.0}     # P(WetGrass | Sprinkler=False, Rain=False)
    })
    
    return bn

def simple_inference_example(bn):
    """Demonstrate simple inference with the network."""
    # Print the network structure
    print("Bayesian Network Structure:")
    bn.print_structure()
    print()
    
    # Check if the network is acyclic
    print(f"Network is acyclic: {bn.is_acyclic()}")
    print()
    
    # Let's answer: P(WetGrass=True | Cloudy=True)
    # To calculate this directly, we need to marginalize over Sprinkler and Rain
    
    # We'll count the probability of all cases where WetGrass=True and Cloudy=True
    prob_wet_given_cloudy = 0
    
    for sprinkler in [True, False]:
        for rain in [True, False]:
            # P(Cloudy=True, Sprinkler=sprinkler, Rain=rain, WetGrass=True)
            prob = bn.query({
                "Cloudy": True,
                "Sprinkler": sprinkler,
                "Rain": rain,
                "WetGrass": True
            })
            prob_wet_given_cloudy += prob
    
    # Normalize by P(Cloudy=True)
    prob_cloudy = 0.5  # Given in the CPT
    
    result = prob_wet_given_cloudy / prob_cloudy
    print(f"P(WetGrass=True | Cloudy=True) = {result:.4f}")
    
    # Let's calculate: P(Rain=True | WetGrass=True)
    # We'll use sampling for this (Monte Carlo approximation)
    n_samples = 10000
    samples = bn.sample(n_samples)
    
    # Filter samples where WetGrass=True
    wet_samples = [s for s in samples if s["WetGrass"] is True]
    rain_given_wet = sum(1 for s in wet_samples if s["Rain"] is True) / len(wet_samples)
    
    print(f"P(Rain=True | WetGrass=True) ≈ {rain_given_wet:.4f} (estimated from {n_samples} samples)")

def main():
    np.random.seed(42)  # For reproducibility
    
    # Create the sprinkler network
    bn = create_sprinkler_network()
    
    # Run inference examples
    simple_inference_example(bn)

if __name__ == "__main__":
    main() 
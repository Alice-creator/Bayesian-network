import numpy as np
from bayesian_network import BayesianNetwork
from variable_elimination import VariableElimination

def create_asia_network():
    """
    Create the 'Asia' Bayesian network example:
    
    A simple Bayesian network representing a medical diagnosis scenario.
    
    Variables:
    - VisitToAsia (A): True/False
    - Tuberculosis (T): True/False
    - Smoking (S): True/False
    - LungCancer (L): True/False
    - Bronchitis (B): True/False
    - TuberculosisOrCancer (TC): True/False
    - XRayPositive (X): True/False
    - Dyspnea (D): True/False
    """
    bn = BayesianNetwork()
    
    # Add nodes
    visit_asia = bn.add_node("VisitToAsia")
    tuberculosis = bn.add_node("Tuberculosis")
    smoking = bn.add_node("Smoking")
    lung_cancer = bn.add_node("LungCancer")
    bronchitis = bn.add_node("Bronchitis")
    tb_or_cancer = bn.add_node("TuberculosisOrCancer")
    xray_positive = bn.add_node("XRayPositive")
    dyspnea = bn.add_node("Dyspnea")
    
    # Add edges
    bn.add_edge("VisitToAsia", "Tuberculosis")
    bn.add_edge("Smoking", "LungCancer")
    bn.add_edge("Smoking", "Bronchitis")
    bn.add_edge("Tuberculosis", "TuberculosisOrCancer")
    bn.add_edge("LungCancer", "TuberculosisOrCancer")
    bn.add_edge("TuberculosisOrCancer", "XRayPositive")
    bn.add_edge("TuberculosisOrCancer", "Dyspnea")
    bn.add_edge("Bronchitis", "Dyspnea")
    
    # Set CPTs
    visit_asia.set_cpt({
        True: 0.01,
        False: 0.99
    })
    
    tuberculosis.set_cpt({
        (True,): {True: 0.05, False: 0.95},   # P(T | A=True)
        (False,): {True: 0.01, False: 0.99}   # P(T | A=False)
    })
    
    smoking.set_cpt({
        True: 0.2,
        False: 0.8
    })
    
    lung_cancer.set_cpt({
        (True,): {True: 0.1, False: 0.9},     # P(L | S=True)
        (False,): {True: 0.01, False: 0.99}   # P(L | S=False)
    })
    
    bronchitis.set_cpt({
        (True,): {True: 0.6, False: 0.4},     # P(B | S=True)
        (False,): {True: 0.3, False: 0.7}     # P(B | S=False)
    })
    
    tb_or_cancer.set_cpt({
        (True, True): {True: 1.0, False: 0.0},   # P(TC | T=True, L=True)
        (True, False): {True: 1.0, False: 0.0},  # P(TC | T=True, L=False)
        (False, True): {True: 1.0, False: 0.0},  # P(TC | T=False, L=True)
        (False, False): {True: 0.0, False: 1.0}  # P(TC | T=False, L=False)
    })
    
    xray_positive.set_cpt({
        (True,): {True: 0.98, False: 0.02},   # P(X | TC=True)
        (False,): {True: 0.05, False: 0.95}   # P(X | TC=False)
    })
    
    dyspnea.set_cpt({
        (True, True): {True: 0.9, False: 0.1},    # P(D | TC=True, B=True)
        (True, False): {True: 0.7, False: 0.3},   # P(D | TC=True, B=False)
        (False, True): {True: 0.8, False: 0.2},   # P(D | TC=False, B=True)
        (False, False): {True: 0.1, False: 0.9}   # P(D | TC=False, B=False)
    })
    
    return bn

def variable_elimination_example(bn):
    """Demonstrate the Variable Elimination algorithm."""
    print("Bayesian Network Structure:")
    bn.print_structure()
    print()
    
    # Create a Variable Elimination instance
    ve = VariableElimination(bn)
    
    print("Performing inference using Variable Elimination...")
    print()
    
    # Query 1: P(LungCancer | Smoking=True)
    # This is a simple query that we can verify by looking at the CPT
    query1 = {"LungCancer": True}
    evidence1 = {"Smoking": True}
    result1 = ve.query(query1, evidence1)
    print(f"Query 1: P(LungCancer=True | Smoking=True)")
    print(f"Result: {result1.values.get((True,), 0):.4f}")
    print(f"Ground truth (from CPT): 0.1000")
    print()
    
    # Query 2: P(Tuberculosis | XRayPositive=True)
    # This requires actual inference across the network
    query2 = {"Tuberculosis": True}
    evidence2 = {"XRayPositive": True}
    result2 = ve.query(query2, evidence2)
    print(f"Query 2: P(Tuberculosis=True | XRayPositive=True)")
    print(f"Result: {result2.values.get((True,), 0):.4f}")
    print()
    
    # Query 3: P(Dyspnea | Smoking=True, VisitToAsia=True)
    query3 = {"Dyspnea": True}
    evidence3 = {"Smoking": True, "VisitToAsia": True}
    result3 = ve.query(query3, evidence3)
    print(f"Query 3: P(Dyspnea=True | Smoking=True, VisitToAsia=True)")
    print(f"Result: {result3.values.get((True,), 0):.4f}")
    print()
    
    # Query 4: Compare inference with and without evidence
    # P(Dyspnea | XRayPositive=True, Smoking=True) vs P(Dyspnea)
    query4a = {"Dyspnea": True}
    evidence4a = {"XRayPositive": True, "Smoking": True}
    result4a = ve.query(query4a, evidence4a)
    
    query4b = {"Dyspnea": True}
    evidence4b = {}
    result4b = ve.query(query4b, evidence4b)
    
    print(f"Query 4a: P(Dyspnea=True | XRayPositive=True, Smoking=True)")
    print(f"Result: {result4a.values.get((True,), 0):.4f}")
    print()
    
    print(f"Query 4b: P(Dyspnea=True) [No evidence]")
    print(f"Result: {result4b.values.get((True,), 0):.4f}")
    print()
    
    print("Diagnostic Reasoning:")
    # Query 5: P(Smoking | Dyspnea=True)
    query5 = {"Smoking": True}
    evidence5 = {"Dyspnea": True}
    result5 = ve.query(query5, evidence5)
    print(f"Query 5: P(Smoking=True | Dyspnea=True)")
    print(f"Result: {result5.values.get((True,), 0):.4f}")
    print(f"Prior P(Smoking=True): 0.2000")
    print()

def main():
    np.random.seed(42)  # For reproducibility
    
    # Create the Asia network
    bn = create_asia_network()
    
    # Run variable elimination examples
    variable_elimination_example(bn)

if __name__ == "__main__":
    main() 
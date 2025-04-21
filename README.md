# Bayesian-network

## 📌 Project Title (Tiếng Việt)

**Thiết kế và phân tích các thuật toán tìm kiếm top-K mạng Bayesian tiện ích cao của các tập mục trong cơ sở dữ liệu không chắc chắn**

## 📌 Project Title (English)

**Design and Analysis of Algorithms Searching for Top-K High-Utility Bayesian Networks of Itemsets in Uncertain Databases**

---
### Deadline: 01/05/2025

## 📋 Bayesian Network Implementation

This repository contains a from-scratch implementation of Bayesian networks in Python, including:

1. **Core Bayesian Network Classes**:
   - `Node`: Represents a random variable in the network
   - `BayesianNetwork`: Represents the complete network structure

2. **Inference Algorithms**:
   - Basic exact inference using enumeration
   - Variable Elimination algorithm
   - Monte Carlo sampling

3. **Example Networks**:
   - Classic Sprinkler network example
   - Asia medical diagnosis network

4. **Top-K High-Utility Bayesian Networks**:
   - Extended `ItemsetNode` for representing itemsets with utilities
   - `HighUtilityBN` class for managing high-utility Bayesian networks
   - Mining algorithms for finding top-K high-utility networks
   - Visualization of high-utility networks

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- NetworkX (for network visualization)

### Installation
```bash
pip install -r requirements.txt
```

### Running Examples
```bash
# Run all examples
python main.py

# Run specific examples
python main.py --example sprinkler    # Basic Bayesian network example
python main.py --example asia         # Variable elimination algorithm example
python main.py --example high_utility # Top-K high-utility networks example

# Alternative: Run individual examples directly
python src/example.py
python src/variable_elimination_example.py
python src/top_k_high_utility_example.py
```

## 🔍 Implementation Details

The implementation focuses on discrete (binary) random variables with the following features:
- Creating and manipulating Bayesian network structures
- Setting conditional probability tables (CPTs)
- Performing inference tasks
- Sampling from the joint distribution

### Top-K High-Utility Bayesian Networks

The high-utility implementation extends the base Bayesian network to find networks that maximize utility in uncertain databases:

1. **Itemset Mining**: First extracts high-utility itemsets from the database
2. **Network Generation**: Creates candidate Bayesian networks using different strategies:
   - Subset-based relationships (superset → subset)
   - Co-occurrence based relationships
3. **Network Evaluation**: Evaluates networks based on total utility and ensures they are acyclic
4. **Visualization**: Provides visualization tools for the discovered networks

## 🛠️ Future Work

- Support for continuous variables
- Additional inference algorithms (MCMC, importance sampling)
- Structure learning from data
- Improved utility-based reasoning for top-K networks
- Performance optimizations for large databases
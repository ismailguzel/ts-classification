# Topological Features Extraction

##  Overview

This directory contains experimental work for extracting topological features from time series using Topological Data Analysis (TDA) methods, specifically persistent homology.

**Status**: Experimental / Research-in-progress

##  Contents

- `topological_features.py` - Main module for computing topological features
- `test_topological.py` - Test script for validating topological feature extraction
- `topological_demo.ipynb` - Jupyter notebook demonstrating TDA on time series
- `ts-top-jupyter/` - Python virtual environment for Jupyter notebook

##  Methods

Topological features are extracted using:
- **Persistent Homology**: Captures topological properties across multiple scales
- **Takens Embedding**: Converts time series into point clouds
- **Persistence Diagrams**: Summarizes birth-death times of topological features
- **Statistical Summaries**: Mean, variance, entropy of persistence diagrams

##  Usage

### Setup Environment

```bash
# Activate the topological Jupyter environment
source ts-top-jupyter/bin/activate
```

### Run Tests

```bash
# Test topological feature extraction
python test_topological.py
```

### Jupyter Demo

```bash
# Launch Jupyter with the demo notebook
source ts-top-jupyter/bin/activate
jupyter notebook topological_demo.ipynb
```

##  Features Extracted

- **H0 (Connected Components)**: Connectivity information
- **H1 (Loops/Cycles)**: Periodic and cyclic patterns
- Statistical summaries of persistence diagrams:
  - Mean persistence
  - Variance of persistence
  - Entropy
  - Number of features

##  Integration

These features are **not** currently integrated into the main feature extraction pipeline (`extract_dask.py`). They represent experimental work for future enhancement.

To integrate:
1. Add topological features to `extract_dask.py`
2. Update feature selection to handle topological features
3. Benchmark performance vs TSFresh features

##  Dependencies

- `giotto-tda` or `ripser` - Persistent homology computation
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `jupyter` - Interactive exploration (in demo)

##  Future Work

- [ ] Benchmark topological vs TSFresh features
- [ ] Optimize computation for large datasets
- [ ] Integrate with main preprocessing pipeline
- [ ] Compare classification performance
- [ ] Parallelize computation with Dask

##  References

- Persistent Homology for Time Series: [Link to paper/resource]
- Giotto-TDA: https://giotto-ai.github.io/gtda-docs/
- Takens' Embedding Theorem

---

**Note**: This is experimental work and not part of the main training pipeline yet.

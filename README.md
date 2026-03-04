# Adaptive Early-Exit GNN for Link Prediction

Code for the L65 Geometric Deep Learning mini-project:

> *Adaptive Early-Exit Graph Neural Networks for Subgraph-Based Link Prediction*  
> Iva Jurkovic, University of Cambridge, 2025-2026

## Overview

We integrate an ACT-style halting mechanism into a SEAL-based subgraph GNN for
link prediction. The model adaptively selects propagation depth per edge query,
reducing expected depth by **41.2%** (3.0 → 1.76 layers) while achieving
**12.0% MRR** under the strict HeaRT evaluation protocol on Cora.

## Repository Structure

```
├── Early_Exit_Link_Prediction.ipynb   # Main notebook (run this)
├── src/
│   ├── model.py          # AdaptiveSAGE model class
│   ├── data_utils.py     # Data loading, negative generation, subgraph caching
│   ├── train_eval.py     # Training loop, evaluation, diagnostics
│   ├── visualisation.py  # All plotting functions and LaTeX table
│   └── utils.py          # Seed setting, JSON save/load, result printing
├── outputs/              # Generated plots and result JSONs (not tracked)
└── README.md
```

## Quickstart (Google Colab)

1. Clone this repo into Colab:
   ```bash
   !git clone https://github.com/ijurk/Early_Exit_Link_Prediction.git
   %cd Early_Exit_Link_Prediction
   ```

2. Open and run `Early_Exit_Link_Prediction.ipynb` top-to-bottom.

The notebook handles environment setup, HeaRT dataset download, training,
test evaluation, and figure generation automatically.

**Expected runtimes (Colab T4):**
- Environment setup: ~5 min
- Subgraph caching: ~20 min
- Training (50 epochs): ~4.5 h
- Test evaluation (500 negatives): ~35 min

## Results

| Method             | MRR (%) | Hits@20 (%) | Avg Depth | Exp. Depth Saved |
|--------------------|---------|-------------|-----------|-----------------|
| Common Neighbors   | 10.5    | 30.0        | —         | —               |
| Resource Allocation| 13.2    | 36.0        | —         | —               |
| GCN (HeaRT)        | 16.6    | 42.0        | 3.0       | 0%              |
| GraphSAGE (HeaRT)  | 16.1    | 40.0        | 3.0       | 0%              |
| **Ours (Adaptive)**| **12.0**| **34.5**    | **1.76**  | **41.2%**       |

Baselines from Zhu et al. (2023) HeaRT benchmark.

## Key References

- Zhang & Chen (2018). SEAL: Link Prediction Based on Graph Neural Networks.
- Zhu et al. (2023). HeaRT: Evaluating GNNs for Link Prediction.
- Graves (2016). Adaptive Computation Time for Recurrent Neural Networks.
- Di Francesco et al. (2025). Early-Exit Graph Neural Networks.

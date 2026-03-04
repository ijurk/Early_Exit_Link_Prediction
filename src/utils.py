"""
utils.py
--------
General-purpose utilities for the AdaptiveSAGE experiment.

Contains:
    - set_seed: Reproducibility helper.
    - save_results: JSON serialisation of results log.
    - load_results: JSON deserialisation.
    - print_test_results: Formatted console summary of final test metrics.
"""

import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set all random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, PyTorch (CPU and all GPUs),
    and configures cuDNN to deterministic mode.

    Args:
        seed (int): Random seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Random seed set to {seed}")


def save_results(results: dict, path: str):
    """
    Serialise a results dictionary to JSON.

    Args:
        results (dict): Results to save (must be JSON-serialisable).
        path (str): Output file path.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {path}")


def load_results(path: str) -> dict:
    """
    Load a results dictionary from JSON.

    Args:
        path (str): Path to JSON file.

    Returns:
        dict: Loaded results.
    """
    with open(path, "r") as f:
        return json.load(f)


def print_test_results(test_results: dict):
    """
    Print a formatted summary of final test evaluation results.

    Args:
        test_results (dict): Output of evaluate_test_500().
    """
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (500 NEGATIVES - FULL HEART PROTOCOL)")
    print("=" * 70)
    print(f"MRR:        {test_results['mrr']:.4f} ({test_results['mrr']*100:.2f}%)")
    print(f"Hits@1:     {test_results['hits@1']:.4f} ({test_results['hits@1']*100:.2f}%)")
    print(f"Hits@10:    {test_results['hits@10']:.4f} ({test_results['hits@10']*100:.2f}%)")
    print(f"Hits@20:    {test_results['hits@20']:.4f} ({test_results['hits@20']*100:.2f}%)")
    print(f"Hits@50:    {test_results['hits@50']:.4f} ({test_results['hits@50']*100:.2f}%)")
    print(f"Hits@100:   {test_results['hits@100']:.4f} ({test_results['hits@100']*100:.2f}%)")
    print("-" * 70)
    print(f"Avg Depth:  {test_results['avg_depth']:.2f} / 3")
    print(f"Depth Saved:{test_results['computation_saved']*100:.1f}%")
    print(f"Time:       {test_results.get('elapsed_minutes', 0):.1f} min")
    print("=" * 70)

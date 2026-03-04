"""

Data loading and preprocessing utilities for the Cora / HeaRT link prediction pipeline.

Responsibilities:
    - Loading the Cora graph via PyTorch Geometric
    - Reading HeaRT train/val/test positive edge splits
    - Loading pre-generated HeaRT hard negative samples (.npy)
    - Generating training negatives via Common Neighbour scoring
    - Precomputing and caching SEAL-style k-hop enclosing subgraphs

"""

import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Graph + split loading
# ---------------------------------------------------------------------------

def load_cora_with_heart(data_path: str, device: torch.device):
    """
    Load the Cora graph and HeaRT train/val/test splits.

    Expects HeaRT-format files under `data_path`:
        train_pos.txt, valid_pos.txt, test_pos.txt  (whitespace-separated u v per line)
        heart_valid_samples.npy, heart_test_samples.npy  (hard negatives, shape [N, K, 2])

    Args:
        data_path (str): Path to the HeaRT Cora dataset directory (e.g. 'dataset/cora').
        device (torch.device): Device to move the PyG Data object to.

    Returns:
        data (Data): PyG Data object (graph on `device`).
        train_pos (Tensor): [N_train, 2] positive training edges.
        val_pos (Tensor): [N_val, 2] positive validation edges.
        test_pos (Tensor): [N_test, 2] positive test edges.
        val_neg (Tensor or None): [N_val, K, 2] HeaRT hard validation negatives.
        test_neg (Tensor or None): [N_test, K, 2] HeaRT hard test negatives.
    """
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0].to(device)

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Features: {data.x.shape}")

    def _read_edges(filepath):
        edges = []
        with open(filepath, "r") as f:
            for line in f:
                u, v = map(int, line.strip().split())
                edges.append([u, v])
        return torch.tensor(edges, dtype=torch.long)

    train_pos = _read_edges(f"{data_path}/train_pos.txt")
    val_pos = _read_edges(f"{data_path}/valid_pos.txt")
    test_pos = _read_edges(f"{data_path}/test_pos.txt")

    # HeaRT hard negatives (optional — fall back to None if missing)
    val_neg = _load_negatives(f"{data_path}/heart_valid_samples.npy", split="val")
    test_neg = _load_negatives(f"{data_path}/heart_test_samples.npy", split="test")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pos)} edges")
    print(f"  Val:   {len(val_pos)} edges")
    print(f"  Test:  {len(test_pos)} edges")

    return data, train_pos, val_pos, test_pos, val_neg, test_neg


def _load_negatives(filepath: str, split: str):
    """
    Load a HeaRT hard-negative .npy file.

    Args:
        filepath (str): Path to the .npy file.
        split (str): Human-readable split name for logging only.

    Returns:
        Tensor [N, K, 2] or None if the file does not exist.
    """
    if os.path.exists(filepath):
        neg = np.load(filepath)
        neg_tensor = torch.from_numpy(neg).long()
        print(f"{split} negatives: {neg_tensor.shape} (HeaRT hard negatives)")
        return neg_tensor
    else:
        print(f"Warning: HeaRT negatives not found at {filepath}, will fall back to random.")
        return None


# ---------------------------------------------------------------------------
# Training negative generation
# ---------------------------------------------------------------------------

def precompute_heart_negatives_fast(train_pos: torch.Tensor, data, K: int = 10) -> dict:
    """
    Generate hard training negatives via Common Neighbour scoring.

    For each positive edge (u, v), generates K negatives by:
        - Corrupting v: ranking candidates by CN with u (descending), taking top K//2.
        - Corrupting u: ranking candidates by CN with v (descending), taking top K - K//2.

    Candidates that are already neighbours of the anchor node are excluded.

    Args:
        train_pos (Tensor): [N, 2] positive training edges.
        data (Data): PyG Data object (used for adjacency).
        K (int): Number of negatives per positive. Default: 10.

    Returns:
        train_negatives (dict): Maps positive index -> list of (u_neg, v_neg) tuples.
    """
    adj_list = defaultdict(set)
    for i in range(data.edge_index.size(1)):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        adj_list[u].add(v)
        adj_list[v].add(u)

    train_negatives = {}

    for idx, pos in enumerate(tqdm(train_pos, desc="Generating training negatives")):
        u, v = pos.tolist()
        negatives = []

        # Corrupt v
        candidates = set(range(data.num_nodes)) - {u, v} - adj_list[u]
        scored = [(c, len(adj_list[u] & adj_list[c])) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        for c, _ in scored[: K // 2]:
            negatives.append((u, c))

        # Corrupt u
        candidates = set(range(data.num_nodes)) - {u, v} - adj_list[v]
        scored = [(c, len(adj_list[v] & adj_list[c])) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        for c, _ in scored[: K - K // 2]:
            negatives.append((c, v))

        train_negatives[idx] = negatives

    return train_negatives


# ---------------------------------------------------------------------------
# SEAL-style subgraph caching
# ---------------------------------------------------------------------------

def precompute_all_subgraphs(
    edge_list: list,
    data,
    L_max: int = 3,
    device: str = "cpu",
) -> dict:
    """
    Precompute and cache k-hop enclosing subgraphs for all edges at all depths.

    For each (u, v) in `edge_list` and each depth k in 1..L_max, extracts the
    k-hop enclosing subgraph using PyTorch Geometric's `k_hop_subgraph` and
    stores the result as a plain dict (CPU tensors) for memory efficiency.

    Subgraphs are kept on CPU and moved to the target device during the forward
    pass to avoid GPU OOM on large graphs.

    Args:
        edge_list (list): List of (u, v) tuples to cache.
        data (Data): PyG Data object (full graph, used for edge_index).
        L_max (int): Maximum depth to cache. Default: 3.
        device (str): Unused — subgraphs always stored on CPU. Kept for API clarity.

    Returns:
        cached_subgraphs (dict): Nested dict:
            {(u, v): {depth: {'subset', 'edge_index', 'mapping', 'num_nodes'}}}
    """
    print(f"\n{'='*70}")
    print(f"Caching subgraphs for {len(edge_list):,} edges at depths 1-{L_max}")
    print(f"{'='*70}")

    cached_subgraphs = {}
    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes
    start_time = time.time()

    for u, v in tqdm(edge_list, desc="Caching", ncols=80):
        cached_subgraphs[(u, v)] = {}
        for depth in range(1, L_max + 1):
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=[u, v],
                num_hops=depth,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            cached_subgraphs[(u, v)][depth] = {
                "subset": subset.cpu(),
                "edge_index": sub_edge_index.cpu(),
                "mapping": mapping,
                "num_nodes": len(subset),
            }

    elapsed = time.time() - start_time

    # Rough statistics over first 100 edges
    sample_keys = list(cached_subgraphs.keys())[:100]
    avg_nodes = sum(
        cached_subgraphs[e][d]["num_nodes"]
        for e in sample_keys
        for d in range(1, L_max + 1)
    ) / (len(sample_keys) * L_max)

    total_subgraphs = len(edge_list) * L_max
    est_memory_mb = (total_subgraphs * avg_nodes * 8 * 2) / (1024**2)

    print(f"\n{'='*70}")
    print(f"✓ Caching complete!")
    print(f"  Total subgraphs: {total_subgraphs:,} ({len(edge_list):,} edges × {L_max} depths)")
    print(f"  Avg nodes/subgraph: ~{avg_nodes:.0f}")
    print(f"  Est. memory: ~{est_memory_mb:.0f} MB")
    print(f"  Time: {elapsed / 60:.1f} min  ({len(edge_list) / elapsed:.0f} edges/sec)")
    print(f"{'='*70}\n")

    return cached_subgraphs


def build_train_edge_list(train_pos: torch.Tensor, train_negatives: dict) -> list:
    """
    Build the deduplicated list of all edges (positives + negatives) to cache.

    Args:
        train_pos (Tensor): [N, 2] positive training edges.
        train_negatives (dict): Maps positive index -> list of (u_neg, v_neg).

    Returns:
        all_train_edges (list): Deduplicated list of (u, v) tuples.
    """
    edge_to_idx = {}
    all_train_edges = []

    for pos_idx, pos_edge in enumerate(tqdm(train_pos, desc="Building edge list", ncols=80)):
        u, v = pos_edge.tolist()
        if (u, v) not in edge_to_idx:
            edge_to_idx[(u, v)] = len(all_train_edges)
            all_train_edges.append((u, v))
        for u_neg, v_neg in train_negatives[pos_idx]:
            if (u_neg, v_neg) not in edge_to_idx:
                edge_to_idx[(u_neg, v_neg)] = len(all_train_edges)
                all_train_edges.append((u_neg, v_neg))

    print(f"Total unique training edges to cache: {len(all_train_edges):,}")
    return all_train_edges


def build_val_edge_list(val_pos: torch.Tensor, val_neg: torch.Tensor) -> list:
    """
    Build the flat list of validation edges (positives + negatives) to cache.

    Args:
        val_pos (Tensor): [N, 2] positive validation edges.
        val_neg (Tensor): [N, K, 2] HeaRT hard negatives.

    Returns:
        val_edges_list (list): List of (u, v) tuples (may contain duplicates across positives).
    """
    val_edges_list = []
    for pos_idx, pos_edge in enumerate(val_pos):
        u, v = pos_edge.tolist()
        val_edges_list.append((u, v))
        for neg_edge in val_neg[pos_idx]:
            u_neg, v_neg = neg_edge.tolist()
            val_edges_list.append((u_neg, v_neg))
    return val_edges_list

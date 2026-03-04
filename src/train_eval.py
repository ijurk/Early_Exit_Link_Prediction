"""

Training and evaluation routines for AdaptiveSAGE link prediction.

Contains:
    - train_one_epoch_cached: Single training epoch using cached subgraphs.
    - evaluate_heart: Ranking evaluation under the HeaRT protocol.
    - evaluate_test_500: Full test evaluation with all 500 hard negatives.
    - analyze_exit_distribution: Diagnostic utility for halting weight analysis.

Training uses binary cross-entropy loss over positive/negative edge scores,
with an optional depth-penalty term (lambda_depth * E[K]).

Evaluation follows HeaRT (Zhu et al., 2023): each positive is ranked against
K hard negatives; MRR and Hits@K are reported.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import types
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch_cached(
    model,
    data,
    train_pos: torch.Tensor,
    train_negatives: dict,
    cached_subgraphs: dict,
    args: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Run one epoch of training using precomputed subgraph cache.

    For each positive edge, scores the positive and its K hard negatives,
    computes BCE loss + optional depth penalty, and updates model weights.

    Args:
        model (AdaptiveSAGE): The model to train.
        data (Data): PyG Data object.
        train_pos (Tensor): [N, 2] positive training edges.
        train_negatives (dict): Maps positive index -> list of (u_neg, v_neg).
        cached_subgraphs (dict): Precomputed subgraph cache.
        args (dict): Hyperparameter dict. Required keys:
            'batch_size' (int), 'lambda_depth' (float).
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Target device.

    Returns:
        dict with keys: 'loss', 'rank_loss', 'depth_pen', 'avg_depth'.
    """
    model.train()

    total_loss = total_rank_loss = total_depth_pen = 0.0
    total_EK = total_pairs = num_batches = 0

    x_full = data.x.to(device)
    perm = torch.randperm(len(train_pos))

    pbar = tqdm(
        range(0, len(train_pos), args["batch_size"]),
        desc="Training",
        ncols=100,
    )

    for batch_start in pbar:
        batch_end = min(batch_start + args["batch_size"], len(train_pos))
        batch_indices = perm[batch_start:batch_end]
        batch_pos = train_pos[batch_indices]

        optimizer.zero_grad()

        batch_scores, batch_labels, batch_EK = [], [], []

        for batch_idx, pos_edge in enumerate(batch_pos):
            u, v = pos_edge.tolist()
            pos_idx = batch_indices[batch_idx].item()
            negatives = train_negatives[pos_idx]

            score_pos, EK_pos, _ = model.forward_pair_train_cached(
                cached_subgraphs, u, v, x_full
            )
            batch_scores.append(score_pos)
            batch_labels.append(1.0)
            batch_EK.append(EK_pos)

            for u_neg, v_neg in negatives:
                score_neg, EK_neg, _ = model.forward_pair_train_cached(
                    cached_subgraphs, u_neg, v_neg, x_full
                )
                batch_scores.append(score_neg)
                batch_labels.append(0.0)
                batch_EK.append(EK_neg)

        scores = torch.stack(batch_scores)
        labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)
        EKs = torch.stack(batch_EK)

        rank_loss = F.binary_cross_entropy_with_logits(scores, labels)
        depth_pen = args["lambda_depth"] * EKs.mean()
        loss = rank_loss + depth_pen

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rank_loss += rank_loss.item()
        total_depth_pen += depth_pen.item()
        total_EK += EKs.sum().item()
        total_pairs += len(batch_scores)
        num_batches += 1

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "depth": f"{EKs.mean().item():.2f}"}
        )

    return {
        "loss": total_loss / num_batches,
        "rank_loss": total_rank_loss / num_batches,
        "depth_pen": total_depth_pen / num_batches,
        "avg_depth": total_EK / total_pairs,
    }


# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------

def evaluate_heart(
    model,
    data,
    pos_edges: torch.Tensor,
    neg_edges_array,
    cached_subgraphs: dict,
    args: dict,
    mode: str = "val",
) -> dict:
    """
    Evaluate model under the HeaRT ranking protocol.

    Uses hard early exit (forward_pair_test_cached) for speed.
    Optionally evaluates on a random subset for fast mid-training checks.

    Args:
        model (AdaptiveSAGE): Trained or partially trained model.
        data (Data): PyG Data object.
        pos_edges (Tensor): [N, 2] positive edges to evaluate.
        neg_edges_array (Tensor or None): [N, K, 2] hard negatives.
            If None, evaluation falls back to random negatives (not recommended).
        cached_subgraphs (dict): Precomputed subgraph cache.
        args (dict): Config dict. Optional key: 'eval_subset' (int or None).
        mode (str): Label for progress bar. Default: 'val'.

    Returns:
        dict with keys: 'mrr', 'hits@10', 'hits@20', 'hits@50',
                        'avg_depth', 'computation_saved'.
    """
    model.eval()

    all_ranks, all_depths = [], []

    # Optionally sub-sample for speed during training
    if args.get("eval_subset") and args["eval_subset"] < len(pos_edges):
        indices = torch.randperm(len(pos_edges))[: args["eval_subset"]]
        pos_edges = pos_edges[indices]
        if neg_edges_array is not None:
            neg_edges_array = neg_edges_array[indices]

    with torch.no_grad():
        pbar = tqdm(
            enumerate(pos_edges),
            total=len(pos_edges),
            desc=f"{mode} eval",
        )

        for pos_idx, pos_edge in pbar:
            u, v = pos_edge.tolist()

            pos_score, pos_depth = model.forward_pair_test_cached(
                cached_subgraphs, u, v, data.x
            )
            all_depths.append(pos_depth)

            negatives = neg_edges_array[pos_idx] if neg_edges_array is not None else []
            neg_scores = []
            for neg_edge in negatives:
                u_neg, v_neg = neg_edge.tolist()
                neg_score, neg_depth = model.forward_pair_test_cached(
                    cached_subgraphs, u_neg, v_neg, data.x
                )
                neg_scores.append(neg_score)
                all_depths.append(neg_depth)

            rank = int((np.array(neg_scores) >= pos_score).sum() + 1)
            all_ranks.append(rank)

            if (pos_idx + 1) % 10 == 0:
                pbar.set_postfix(
                    {
                        "MRR": f"{np.mean([1.0/r for r in all_ranks]):.4f}",
                        "avg_depth": f"{np.mean(all_depths):.2f}",
                    }
                )

    mrr = np.mean([1.0 / r for r in all_ranks])
    avg_depth = np.mean(all_depths)

    return {
        "mrr": mrr,
        "hits@10": np.mean([r <= 10 for r in all_ranks]),
        "hits@20": np.mean([r <= 20 for r in all_ranks]),
        "hits@50": np.mean([r <= 50 for r in all_ranks]),
        "avg_depth": avg_depth,
        "computation_saved": 1 - (avg_depth / model.L_max),
    }


# ---------------------------------------------------------------------------
# Final test evaluation (all 500 negatives)
# ---------------------------------------------------------------------------

def evaluate_test_500(
    model,
    data,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Full HeaRT test evaluation ranking each positive against all 500 hard negatives.

    Uses soft ACT aggregation (forward_pair_train_cached) to match training-time
    scoring and ensure consistency with validation MRR.

    Note: This is slow (~30-45 min on Cora). Expects test subgraphs to be computed
    on-the-fly via model.forward_pair_train (no cache required — see notebook).

    Args:
        model (AdaptiveSAGE): Trained model (best checkpoint).
        data (Data): PyG Data object.
        test_pos (Tensor): [N_test, 2] positive test edges.
        test_neg (Tensor): [N_test, 500, 2] HeaRT hard test negatives.
        device (torch.device): Target device.

    Returns:
        dict with keys: 'mrr', 'hits@1', 'hits@10', 'hits@20', 'hits@50',
                        'hits@100', 'avg_depth', 'computation_saved',
                        'elapsed_minutes', 'rank_distribution'.
    """

    model.eval()
    all_ranks, all_depths = [], []
    start = time.time()

    with torch.no_grad():
        for i in tqdm(range(len(test_pos)), desc="Test"):
            u, v = test_pos[i].tolist()
            pos_score, pos_depth, _ = model.forward_pair_train(data, u, v, data.x)
            all_depths.append(float(pos_depth.item()))

            neg_scores = []
            for neg_edge in test_neg[i]:
                u_neg, v_neg = neg_edge.tolist()
                neg_score, neg_depth, _ = model.forward_pair_train(data, u_neg, v_neg, data.x)
                neg_scores.append(float(neg_score.item()))
                all_depths.append(float(neg_depth.item()))

            rank = int((np.array(neg_scores) >= float(pos_score.item())).sum() + 1)
            all_ranks.append(rank)

    elapsed = time.time() - start
    avg_depth = float(np.mean(all_depths))

    return {
        "mrr": float(np.mean([1.0 / r for r in all_ranks])),
        "hits@1": float(np.mean([r <= 1 for r in all_ranks])),
        "hits@10": float(np.mean([r <= 10 for r in all_ranks])),
        "hits@20": float(np.mean([r <= 20 for r in all_ranks])),
        "hits@50": float(np.mean([r <= 50 for r in all_ranks])),
        "hits@100": float(np.mean([r <= 100 for r in all_ranks])),
        "avg_depth": avg_depth,
        "computation_saved": 1 - avg_depth / 3.0,
        "elapsed_minutes": elapsed / 60,
        "num_negatives": 500,
        "protocol": "HeaRT_full",
        "rank_distribution": {
            "rank_1": int(sum(r == 1 for r in all_ranks)),
            "rank_2-3": int(sum(2 <= r <= 3 for r in all_ranks)),
            "rank_4-10": int(sum(4 <= r <= 10 for r in all_ranks)),
            "rank_11-20": int(sum(11 <= r <= 20 for r in all_ranks)),
            "rank_21-50": int(sum(21 <= r <= 50 for r in all_ranks)),
            "rank_51-100": int(sum(51 <= r <= 100 for r in all_ranks)),
            "rank_>100": int(sum(r > 100 for r in all_ranks)),
        },
    }


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def analyze_exit_distribution(
    model,
    data,
    edge_list: torch.Tensor,
    cached_subgraphs: dict,
    num_samples: int = 200,
) -> dict:
    """
    Compute the distribution of dominant exit layers over a sample of edges.

    The dominant exit layer is defined as argmax(alpha) + 1.

    Args:
        model (AdaptiveSAGE): Trained model in eval mode.
        data (Data): PyG Data object.
        edge_list (Tensor): [N, 2] edges to sample from.
        cached_subgraphs (dict): Precomputed subgraph cache.
        num_samples (int): Number of edges to sample. Default: 200.

    Returns:
        dict mapping exit layer (int) -> percentage (float) of edges.
    """
    model.eval()
    exit_counts = {k: 0 for k in range(1, model.L_max + 1)}

    with torch.no_grad():
        for i in range(min(num_samples, len(edge_list))):
            u, v = edge_list[i].tolist()
            _, _, alpha = model.forward_pair_train_cached(
                cached_subgraphs, u, v, data.x
            )
            exit_layer = int(torch.argmax(alpha).item()) + 1
            exit_counts[exit_layer] += 1

    total = sum(exit_counts.values())
    return {k: (v / total) * 100 for k, v in exit_counts.items()}

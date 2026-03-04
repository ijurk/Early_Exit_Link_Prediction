"""

AdaptiveSAGE: Adaptive early-exit GraphSAGE model for link prediction.

Architecture overview:
    - Weight-shared GraphSAGE backbone applied at depths 1..L_max
    - ACT-style halting network outputs per-depth halt probabilities
    - Soft aggregation of depth-specific logits during training and inference
    - Edge scoring via Hadamard + concatenation head on endpoint embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


class AdaptiveSAGE(nn.Module):
    """
    Adaptive depth GraphSAGE for subgraph-based link prediction.

    At each depth k (1..L_max), the model:
        1. Applies k shared GraphSAGE message-passing steps over a cached subgraph.
        2. Scores the candidate edge via a Hadamard + concat MLP head.
        3. Computes a halt probability via a lightweight MLP.

    Final prediction is a soft ACT-weighted combination of per-depth logits.
    Expected exit depth is reported as a measure of computational usage.

    Args:
        in_dim (int): Input node feature dimensionality.
        hidden_dim (int): Hidden embedding dimensionality. Default: 256.
        L_max (int): Maximum propagation depth. Default: 5.
        dropout (float): Dropout probability applied after each SAGE layer. Default: 0.1.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, L_max: int = 5, dropout: float = 0.1):
        super().__init__()

        self.L_max = L_max
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Projects raw node features to hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Single GraphSAGE layer, weight-shared across all depths
        self.sage_layer = SAGEConv(hidden_dim, hidden_dim)

        # Edge scoring head: [h_u || h_v || h_u ⊙ h_v] -> scalar logit
        self.edge_predictor = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Halting network: [h_u || h_v || score_k] -> halt probability in (0,1)
        self.halting_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward_pair_train_cached(
        self,
        cached_subgraphs: dict,
        u: int,
        v: int,
        x_full: torch.Tensor,
    ):
        """
        Training forward pass using precomputed subgraph cache.

        Runs all L_max depths and computes ACT soft-weighted combination.
        Gradients flow through all depths.

        Args:
            cached_subgraphs (dict): Precomputed subgraphs keyed by (u, v) then depth.
                Each entry contains 'subset', 'edge_index', 'mapping', 'num_nodes'.
            u (int): Source node ID.
            v (int): Target node ID.
            x_full (Tensor): Full graph node feature matrix [num_nodes, feat_dim].

        Returns:
            final_score (Tensor): Scalar edge logit (soft ACT aggregation).
            expected_depth (Tensor): Expected exit depth E[K].
            alpha (Tensor): Normalised halting weights [L_max].
        """
        device = x_full.device

        if isinstance(u, torch.Tensor):
            u = u.item()
        if isinstance(v, torch.Tensor):
            v = v.item()

        alpha_list = []
        scores_list = []
        p_not_halted = torch.tensor(1.0, device=device)

        for depth_k in range(1, self.L_max + 1):
            subgraph_data = cached_subgraphs[(u, v)][depth_k]

            subset = subgraph_data["subset"].to(device)
            edge_index = subgraph_data["edge_index"].to(device)
            mapping = subgraph_data["mapping"]

            # Project and run depth_k message-passing steps
            H = self.input_proj(x_full[subset])
            for _ in range(depth_k):
                H = F.relu(self.sage_layer(H, edge_index))
                H = F.dropout(H, p=self.dropout, training=self.training)

            u_idx = mapping[0].item()
            v_idx = mapping[1].item()
            h_u, h_v = H[u_idx], H[v_idx]

            score_k = self.edge_predictor(
                torch.cat([h_u, h_v, h_u * h_v], dim=-1)
            ).view(1)

            p_halt_k = self.halting_net(torch.cat([h_u, h_v, score_k], dim=-1))

            alpha_k = p_halt_k * p_not_halted
            alpha_list.append(alpha_k)
            scores_list.append(score_k)
            p_not_halted = p_not_halted * (1 - p_halt_k)

        # Normalised ACT weights
        alpha = torch.stack(alpha_list).squeeze()
        alpha = alpha / (alpha.sum() + 1e-8)
        scores = torch.stack(scores_list).squeeze()
        final_score = (alpha * scores).sum()

        depths = torch.arange(1, self.L_max + 1, dtype=torch.float32, device=device)
        expected_depth = (alpha * depths).sum()

        return final_score, expected_depth, alpha

    def forward_pair_test_cached(
        self,
        cached_subgraphs: dict,
        u: int,
        v: int,
        x_full: torch.Tensor,
        threshold: float = 0.5,
    ):
        """
        Test/inference forward pass with hard early exit.

        Iterates depths in order and stops as soon as halt probability
        exceeds `threshold`, or at L_max if never exceeded.

        Note: This provides realized compute savings at inference, unlike
        the soft aggregation used in training. Results may differ slightly
        from training-mode scoring due to discrete stopping.

        Args:
            cached_subgraphs (dict): Precomputed subgraph cache (same format as training).
            u (int): Source node ID.
            v (int): Target node ID.
            x_full (Tensor): Full graph node feature matrix.
            threshold (float): Halt probability threshold for early exit. Default: 0.5.

        Returns:
            score (float): Edge score at exit depth.
            exit_depth (int): Depth at which the model halted.
        """
        device = x_full.device

        if isinstance(u, torch.Tensor):
            u = u.item()
        if isinstance(v, torch.Tensor):
            v = v.item()

        score_k = None  # will be set on first iteration

        for depth_k in range(1, self.L_max + 1):
            subgraph_data = cached_subgraphs[(u, v)][depth_k]
            subset = subgraph_data["subset"].to(device)
            edge_index = subgraph_data["edge_index"].to(device)
            mapping = subgraph_data["mapping"]

            H = self.input_proj(x_full[subset])
            for _ in range(depth_k):
                H = F.relu(self.sage_layer(H, edge_index))

            u_idx, v_idx = mapping[0].item(), mapping[1].item()
            h_u, h_v = H[u_idx], H[v_idx]

            score_k = self.edge_predictor(
                torch.cat([h_u, h_v, h_u * h_v], dim=-1)
            ).view(1)
            p_halt_k = self.halting_net(torch.cat([h_u, h_v, score_k], dim=-1))

            if p_halt_k.item() >= threshold or depth_k == self.L_max:
                return score_k.item(), depth_k

        # Fallback (unreachable in practice, but satisfies type checker)
        return score_k.item(), self.L_max

    def forward_pair_train(
        self,
        data: Data,
        u: int,
        v: int,
        x_full: torch.Tensor,
    ):
        """
        Training forward pass computing subgraphs on-the-fly (no cache).

        Runs all L_max depths using k_hop_subgraph extraction at each depth.
        Computes ACT soft-weighted combination. Gradients flow through all depths.

        Args:
            data (torch_geometric.data.Data): Full graph object. Only
                data.edge_index (LongTensor [2, E]) and data.num_nodes (int)
                are accessed; node features are passed separately via x_full.
            u (int): Source node ID.
            v (int): Target node ID.
            x_full (Tensor): Full graph node feature matrix [num_nodes, feat_dim].

        Returns:
            final_score (Tensor): Scalar edge logit (soft ACT aggregation).
            expected_depth (Tensor): Expected exit depth E[K].
            alpha (Tensor): Normalised halting weights [L_max].
        """
        device = x_full.device

        if isinstance(u, torch.Tensor):
            u = u.item()
        if isinstance(v, torch.Tensor):
            v = v.item()

        alpha_list = []
        scores_list = []
        p_not_halted = torch.tensor(1.0, device=device)

        for depth_k in range(1, self.L_max + 1):

            nodes = torch.tensor([u, v], dtype=torch.long, device=device)
            subset, edge_index, mapping, _ = k_hop_subgraph(
                nodes, depth_k, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
            )

            H = self.input_proj(x_full[subset])

            for layer_idx in range(depth_k):
                H = F.relu(self.sage_layer(H, edge_index))
                H = F.dropout(H, p=self.dropout, training=self.training)

            u_idx, v_idx = mapping[0].item(), mapping[1].item()
            h_u, h_v = H[u_idx], H[v_idx]

            score_k = self.edge_predictor(torch.cat([h_u, h_v, h_u * h_v], dim=-1)).view(1)
            p_halt_k = self.halting_net(torch.cat([h_u, h_v, score_k], dim=-1))

            alpha_k = p_halt_k * p_not_halted
            alpha_list.append(alpha_k)
            scores_list.append(score_k)
            p_not_halted = p_not_halted * (1 - p_halt_k)

        alpha = torch.stack(alpha_list).squeeze()
        alpha = alpha / (alpha.sum() + 1e-8)
        scores = torch.stack(scores_list).squeeze()
        final_score = (alpha * scores).sum()

        depths = torch.arange(1, self.L_max + 1, dtype=torch.float32, device=device)
        expected_depth = (alpha * depths).sum()

        return final_score, expected_depth, alpha


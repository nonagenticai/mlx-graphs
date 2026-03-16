import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class TopKPooling(nn.Module):
    """Top-K pooling layer for hierarchical graph coarsening.

    Based on `"Graph U-Nets" <https://arxiv.org/abs/1905.05178>`_ paper.

    Scores each node with a learnable projection vector, retains the top-k
    scoring nodes, and filters edges accordingly.

    Args:
        in_channels: Size of input node features
        ratio: Ratio of nodes to keep (between 0 and 1). Default 0.5.
        min_score: Minimum score threshold. If set, uses score threshold
            instead of ratio. Default None.
        nonlinearity: Activation for scores. Default 'tanh'.

    Example:

    .. code-block:: python

        pool = TopKPooling(16, ratio=0.5)
        edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        x = mx.random.uniform(low=0, high=1, shape=(3, 16))

        x_out, edge_index_out, edge_attr_out, batch_out, perm, score = pool(
            edge_index, x
        )

        >>> x_out.shape
        [2, 16]
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        nonlinearity: str = "tanh",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score

        # Learnable projection vector
        glorot_init = nn.init.glorot_uniform()
        self.weight = glorot_init(mx.zeros((1, in_channels)), 1.0)

        if nonlinearity == "tanh":
            self.nonlinearity = mx.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = mx.sigmoid
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
        batch: Optional[mx.array] = None,
    ) -> Tuple[
        mx.array,
        mx.array,
        Optional[mx.array],
        Optional[mx.array],
        mx.array,
        mx.array,
    ]:
        """Forward pass.

        Args:
            edge_index: Edge index of shape [2, num_edges]
            node_features: Node features of shape [num_nodes, in_channels]
            edge_features: Optional edge features of shape [num_edges, edge_dim]
            batch: Optional batch assignment of shape [num_nodes]

        Returns:
            Tuple of:
            - node_features: Gated features of retained nodes [k, in_channels]
            - edge_index: Filtered and re-indexed edges [2, num_kept_edges]
            - edge_features: Filtered edge features (or None)
            - batch: Updated batch indices (or None)
            - perm: Indices of retained nodes in original graph [k]
            - score: Scores of retained nodes [k]
        """
        num_nodes = node_features.shape[0]

        # Compute node scores: score = activation(X * p / ||p||)
        weight_norm = self.weight / mx.sqrt((self.weight * self.weight).sum() + 1e-8)
        score = (node_features * weight_norm).sum(axis=-1)
        score = self.nonlinearity(score)

        if self.min_score is not None:
            # Use score threshold
            k = int((score >= self.min_score).sum().item())
            k = max(k, 1)  # Keep at least 1 node
        else:
            # Use ratio
            k = max(1, math.ceil(self.ratio * num_nodes))

        # Get top-k indices
        # argsort in descending order: negate scores, argsort ascending
        sorted_indices = mx.argsort(-score)
        perm = sorted_indices[:k]

        # Sort perm to maintain relative order
        perm = mx.sort(perm)

        # Select top-k nodes and gate by score
        node_features_out = node_features[perm] * mx.expand_dims(score[perm], -1)
        score_out = score[perm]

        # Filter edges: keep only edges where BOTH src and dst are in perm
        # Build a mapping from old node indices to new indices
        # Stop gradient on index operations (not differentiable)
        perm_sg = mx.stop_gradient(perm)
        node_map = mx.full((num_nodes,), -1, dtype=mx.int32)
        new_indices = mx.arange(k, dtype=mx.int32)
        # +1 to distinguish from -1 (unmapped)
        node_map = node_map.at[perm_sg].add(new_indices + 1)

        src_mapped = node_map[edge_index[0]]
        dst_mapped = node_map[edge_index[1]]

        # Keep edges where both endpoints are retained (mapped value > 0)
        edge_mask_src = src_mapped > 0
        edge_mask_dst = dst_mapped > 0
        edge_keep = edge_mask_src * edge_mask_dst  # logical AND via multiplication

        # Since we can't boolean-index directly, use the indices where mask is True
        keep_indices = mx.argsort(-edge_keep.astype(mx.float32))
        num_kept_edges = int(edge_keep.sum().item())

        if num_kept_edges > 0:
            keep_indices = keep_indices[:num_kept_edges]
            # Re-index edges (subtract 1 since we added 1 above)
            new_edge_index = mx.stack(
                [
                    node_map[edge_index[0][keep_indices]] - 1,
                    node_map[edge_index[1][keep_indices]] - 1,
                ]
            )
        else:
            new_edge_index = mx.zeros((2, 0), dtype=edge_index.dtype)

        # Filter edge features
        edge_features_out = None
        if edge_features is not None and num_kept_edges > 0:
            edge_features_out = edge_features[keep_indices]

        # Update batch indices
        batch_out = None
        if batch is not None:
            batch_out = batch[perm]

        return (
            node_features_out,
            new_edge_index,
            edge_features_out,
            batch_out,
            perm,
            score_out,
        )

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import add_self_loops, degree, invert_sqrt_degree


class GCNConv(MessagePassing):
    r"""Graph Convolutional Network layer from `"Semi-Supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (Kipf & Welling, 2017).

    .. math::
        \mathbf{h}_i = \sum_{j \in \mathcal{N}(i) \cup \{i\}}
        \frac{1}{\sqrt{\hat{d}_i \hat{d}_j}} \mathbf{W} \mathbf{x}_j

    where :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} 1` is the
    degree of node :math:`i` with added self-loop.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        bias: Whether to use bias in the node projection. Default ``True``
        add_self_loops: Whether to add a self-loop for each node. Default ``False``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GCNConv

        conv = GCNConv(16, 32)
        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        node_features = mx.random.uniform(low=0, high=1, shape=(4, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        [4, 32]
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        bias: bool = True,
        add_self_loops: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GCNConv, self).__init__(**kwargs)

        self.linear = nn.Linear(node_features_dim, out_features_dim, bias)
        self._add_self_loops = add_self_loops

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> mx.array:
        assert edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)"
        assert edge_index[1].size > 0, (
            "'col' component of edge_index should not be empty"
        )

        node_features = self.linear(node_features)

        if self._add_self_loops:
            edge_index = add_self_loops(edge_index)

        row, col = edge_index

        # Compute node degree normalization for the mean aggregation.
        norm: Optional[mx.array] = None
        if normalize:
            deg = degree(col, node_features.shape[0], edge_weights=edge_weights)
            # NOTE : need boolean indexing in order to zero out inf values
            deg_inv_sqrt = invert_sqrt_degree(deg)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Compute messages and aggregate them with sum and norm.
        node_features = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": norm},
        )

        return node_features

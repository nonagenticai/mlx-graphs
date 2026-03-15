from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import degree, scatter


class ChebConv(MessagePassing):
    """Chebyshev spectral graph convolution layer.

    Based on `"Convolutional Neural Networks on Graphs with Fast Localized
    Spectral Filtering" <https://arxiv.org/abs/1606.09375>`_ paper.

    Uses Chebyshev polynomials of the graph Laplacian to approximate
    spectral convolutions, enabling localized graph filters of order K.

    Args:
        in_channels: Size of input node features
        out_channels: Size of output node features
        K: Order of the Chebyshev polynomial (filter size)
        bias: Whether to use bias. Default True.
        normalization: Laplacian normalization ('sym' or None). Default 'sym'.

    Example:

    .. code-block:: python

        conv = ChebConv(16, 32, K=3)
        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        node_features = mx.random.uniform(low=0, high=1, shape=(4, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        [4, 32]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        bias: bool = True,
        normalization: Optional[str] = "sym",
    ):
        super(ChebConv, self).__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization

        # One linear transform per Chebyshev order
        self.lins = [Linear(in_channels, out_channels, bias=False) for _ in range(K)]

        if bias:
            self.bias = mx.zeros(out_channels)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            edge_index: Edge index of shape [2, num_edges]
            node_features: Node features of shape [num_nodes, in_channels]
            edge_weights: Optional edge weights of shape [num_edges]

        Returns:
            Updated node features of shape [num_nodes, out_channels]
        """
        num_nodes = node_features.shape[0]

        # Compute normalized Laplacian edge weights: L = I - D^{-1/2} A D^{-1/2}
        # For message passing, we need -D^{-1/2} A D^{-1/2} (negated adjacency part)
        norm_weights = self._compute_laplacian_weights(
            edge_index, edge_weights, num_nodes
        )

        # Chebyshev recurrence
        # T_0(L̃) · X = X
        Tx_0 = node_features
        out = self.lins[0](Tx_0)

        if self.K > 1:
            # T_1(L̃) · X = L̃ · X (one message-passing step with Laplacian)
            Tx_1 = self._laplacian_mul(edge_index, norm_weights, Tx_0, num_nodes)
            out = out + self.lins[1](Tx_1)

            # T_k(L̃) · X = 2 · L̃ · T_{k-1} - T_{k-2}
            for k in range(2, self.K):
                Tx_2 = (
                    2.0 * self._laplacian_mul(edge_index, norm_weights, Tx_1, num_nodes)
                    - Tx_0
                )
                out = out + self.lins[k](Tx_2)
                Tx_0, Tx_1 = Tx_1, Tx_2

        if "bias" in self:
            out = out + self.bias

        return out

    def _compute_laplacian_weights(
        self,
        edge_index: mx.array,
        edge_weights: Optional[mx.array],
        num_nodes: int,
    ) -> mx.array:
        """Compute normalized Laplacian weights for message passing.

        Returns weights corresponding to -D^{-1/2} A D^{-1/2} (the off-diagonal
        part of the normalized Laplacian, negated for the recurrence).
        """
        src, dst = edge_index

        if edge_weights is None:
            edge_weights = mx.ones(edge_index.shape[1])

        if self.normalization == "sym":
            # Symmetric normalization: D^{-1/2} A D^{-1/2}
            deg = degree(dst, num_nodes)
            deg_inv_sqrt = mx.where(deg > 0, mx.rsqrt(deg), 0.0)
            weights = deg_inv_sqrt[src] * edge_weights * deg_inv_sqrt[dst]
        else:
            weights = edge_weights

        # Negate for Laplacian (L = I - D^{-1/2}AD^{-1/2})
        # The self-loop (identity) part is handled by Chebyshev T_0 term
        return -weights

    def _laplacian_mul(
        self,
        edge_index: mx.array,
        norm_weights: mx.array,
        x: mx.array,
        num_nodes: int,
    ) -> mx.array:
        """Multiply normalized Laplacian with node features via message passing.

        Computes L̃ · x = x + scatter(norm_weights * x[src], dst)
        where norm_weights already contains the negated adjacency weights.
        """
        src_features = x[edge_index[0]]
        messages = norm_weights.reshape(-1, 1) * src_features
        # Aggregate messages (this gives -D^{-1/2}AD^{-1/2} · x)
        aggregated = scatter(messages, edge_index[1], num_nodes, "add")
        # L̃ · x = I · x + (-D^{-1/2}AD^{-1/2}) · x = x + aggregated
        return x + aggregated

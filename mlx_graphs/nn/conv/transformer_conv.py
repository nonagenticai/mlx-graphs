from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import scatter


class TransformerConv(MessagePassing):
    """Graph Transformer convolution layer.

    Based on `"Masked Label Prediction: Unified Message Passing Model for
    Semi-Supervised Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    Implements multi-head attention with Q/K/V projections and optional
    edge feature integration.

    Args:
        in_channels: Size of input node features
        out_channels: Size of output node features per head
        heads: Number of attention heads. Default 1.
        concat: Whether to concatenate or average head outputs. Default True.
        beta: If True, use learnable skip connection weight. Default False.
        dropout: Dropout probability on attention weights. Default 0.0.
        edge_dim: Size of edge features. Default None (no edge features).
        bias: Whether to use bias. Default True.
        root_weight: Whether to add skip connection. Default True.

    Example:

    .. code-block:: python

        conv = TransformerConv(16, 32, heads=2, concat=True)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        [5, 64]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(TransformerConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self._beta = beta
        self.root_weight = root_weight

        # Q, K, V projections
        self.W_query = Linear(in_channels, heads * out_channels, bias=False)
        self.W_key = Linear(in_channels, heads * out_channels, bias=False)
        self.W_value = Linear(in_channels, heads * out_channels, bias=False)

        if edge_dim is not None:
            self.W_edge = Linear(edge_dim, heads * out_channels, bias=False)

        if root_weight:
            self.W_skip = Linear(
                in_channels,
                heads * out_channels if concat else out_channels,
                bias=False,
            )

        if beta:
            self.beta_param = mx.zeros(1)

        if bias:
            bias_shape = heads * out_channels if concat else out_channels
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            edge_index: Edge index of shape [2, num_edges]
            node_features: Node features of shape [num_nodes, in_channels]
            edge_features: Optional edge features of shape [num_edges, edge_dim]

        Returns:
            Updated node features of shape [num_nodes, heads * out_channels] if
            concat, else [num_nodes, out_channels]
        """
        H, C = self.heads, self.out_channels

        query = self.W_query(node_features).reshape(-1, H, C)
        key = self.W_key(node_features).reshape(-1, H, C)
        value = self.W_value(node_features).reshape(-1, H, C)

        # Gather source and destination features
        src_key = key[edge_index[0]]
        src_value = value[edge_index[0]]
        dst_query = query[edge_index[1]]
        dst_idx = edge_index[1]

        out = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={
                "src_key": src_key,
                "src_value": src_value,
                "dst_query": dst_query,
                "index": dst_idx,
                "edge_features": edge_features,
            },
        )

        if self.concat:
            out = out.reshape(-1, H * C)
        else:
            out = mx.mean(out, axis=1)

        # Skip connection
        if self.root_weight:
            skip = self.W_skip(node_features)
            if self._beta:
                beta = mx.sigmoid(self.beta_param)
                out = beta * out + (1.0 - beta) * skip
            else:
                out = out + skip

        if "bias" in self:
            out = out + self.bias

        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        src_key: mx.array,
        src_value: mx.array,
        dst_query: mx.array,
        index: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute attention-weighted messages."""
        # Add edge features to key and value if provided
        if edge_features is not None and "W_edge" in self:
            edge_proj = self.W_edge(edge_features).reshape(
                -1, self.heads, self.out_channels
            )
            src_key = src_key + edge_proj
            src_value = src_value + edge_proj

        # Scaled dot-product attention
        scale = float(self.out_channels) ** 0.5
        alpha = (dst_query * src_key).sum(-1) / scale

        # Softmax over neighbors
        alpha = scatter(alpha, index, self.num_nodes, aggr="softmax")

        if "dropout" in self:
            alpha = self.dropout(alpha)

        return mx.expand_dims(alpha, -1) * src_value

import pytest

# mlx_cluster 0.0.7 ABI is incompatible with mlx 0.31.2+ (missing
# mlx::core::metal::Device::get_command_encoder symbol). Skip until a
# compatible mlx-cluster wheel is released.
pytest.importorskip("mlx_cluster", exc_type=ImportError)

import mlx.core as mx  # noqa: E402

from mlx_graphs.algorithms import Node2Vec  # noqa: E402

mx.random.seed(42)


def test_node2vec():
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    embedding_size = 5
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_size,
        walk_length=2,
        context_size=1,
        num_nodes=4,
    )
    embeddings = model(mx.arange(4).astype(mx.int64))
    assert embeddings.shape == (4, 5), "Embedding dimensions are not equal"

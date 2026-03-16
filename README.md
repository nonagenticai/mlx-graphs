# MLX-Graphs

Graph Neural Network library built on Apple's MLX framework for fast training and inference on Apple Silicon.

| | |
|---|---|
| **Stack** | Python 3.10+ · MLX · NumPy |
| **Hardware** | Apple Silicon (M1/M2/M3/M4) · Unified Memory GPU |

## What MLX-Graphs Provides

When a researcher or developer imports mlx-graphs, they get a complete toolkit for building, training, and evaluating Graph Neural Networks that runs natively on Apple Silicon GPUs. The library provides 9 convolutional layers (GCN, GAT, GATv2, SAGE, GIN, Chebyshev, Transformer, Relational, Simple), 3 normalization layers, global and hierarchical pooling operators, and a MessagePassing base class that handles all the low-level scatter/gather mechanics so that new layers can be defined purely in terms of message and aggregation functions.

The core data layer represents graphs as edge index arrays paired with node and edge feature matrices, all stored as MLX arrays in unified memory. This means graphs can be loaded once and accessed by both CPU and GPU without any device-to-device transfer. The Dataloader handles batching by merging multiple graphs into a single disconnected graph with shifted edge indices, and supports compile-friendly batch padding for use with `mx.compile`. Thirteen built-in datasets span citation networks (Cora, CiteSeer, PubMed), molecular graphs (QM7b, TU datasets), social networks (IMDB, DBLP), superpixel images, and more, each with automatic download, caching, and preprocessing.

Beyond the neural network layers, the library includes graph algorithms like Node2Vec for learning node embeddings via biased random walks, feature transforms for normalizing node attributes, and utility functions for topology manipulation, scatter operations, and format conversion. Benchmarks included in the repository show up to 10x speed improvements over PyTorch Geometric and DGL on large datasets when running on Apple Silicon, because the MLX primitives are designed from the ground up for the Metal GPU and unified memory architecture rather than adapting CUDA-oriented code paths.

## Features

- **MessagePassing base class** — handles scatter/gather, aggregation, and GPU dispatch so custom GNN layers only need to define message and update functions
- **9 convolutional layers** — GCN, GAT, GATv2, SAGE, GIN, Chebyshev, Transformer, Relational, and Simple convolutions covering spectral, attention-based, and sampling-based approaches
- **Graph normalization** — BatchNorm, LayerNorm, and InstanceNorm adapted for graph-structured data so training stability techniques from standard deep learning transfer directly
- **Hierarchical pooling** — TopKPooling for graph coarsening plus global add/mean/max pooling for graph-level readout, enabling graph classification tasks
- **Unified memory graph storage** — graphs live in shared CPU/GPU memory with zero-copy access from either device, allowing Macs with large RAM to train on graphs spanning tens of gigabytes
- **Compile-friendly batching** — `pad=True` in `batch()` pads batches to uniform sizes for compatibility with `mx.compile`, eliminating recompilation overhead on variable-size graphs
- **13 built-in datasets** — Planetoid, TU, OGB, QM7b, DBLP, IMDB, MovieLens, Elliptic, Karate Club, and SuperPixel datasets with automatic download, caching, and preprocessing
- **Node2Vec** — biased random walk algorithm for unsupervised node embedding learning
- **Property caching** — derived properties like `num_nodes` are cached and auto-invalidated on attribute change, avoiding redundant computation
- **forward_dict() convenience** — unpack graph attributes directly into conv layer calls with `conv(**graph.forward_dict())`
- **Framework comparison benchmarks** — included benchmarks against PyG and DGL on training loops, random walks, and individual layer operations across multiple Apple Silicon chips

## Quick Start

```bash
pip install mlx-graphs
# or: uv add mlx-graphs
pip install -e '.[test]'
pytest tests/ -v
ruff check mlx_graphs/ tests/ && ruff format --check mlx_graphs/ tests/
```

## Architecture

```
mlx_graphs/
├── __init__.py
├── algorithms/
│   └── node2vec.py                  # Biased random walk node embeddings
├── data/
│   ├── data.py                      # GraphData and HeteroGraphData containers
│   ├── batch.py                     # GraphDataBatch, batch(), unbatch()
│   ├── collate.py                   # Merge multiple graphs into batched graph
│   └── utils.py                     # Edge index utilities (self-loops, degree)
├── datasets/
│   ├── base_dataset.py              # Abstract dataset base class
│   ├── dataset.py                   # Concrete Dataset with transforms
│   ├── hetero_dataset.py            # Heterogeneous graph dataset
│   ├── planetoid.py                 # Cora, CiteSeer, PubMed
│   ├── tu_dataset.py                # TU benchmark collection
│   ├── ogb_dataset.py               # Open Graph Benchmark wrapper
│   ├── qm7b.py                      # QM7b molecular dataset
│   ├── dblp.py                      # DBLP citation network
│   ├── imdb.py                      # IMDB movie dataset
│   ├── movie_lens_100k.py           # MovieLens 100K recommendations
│   ├── elliptic.py                  # Elliptic Bitcoin transaction dataset
│   ├── karate_club.py               # Zachary's Karate Club
│   ├── superpixel.py                # MNIST/CIFAR superpixel graphs
│   └── utils/                       # Download and I/O helpers
├── loaders/
│   └── dataloaders.py               # Dataloader with batching and padding
├── nn/
│   ├── message_passing.py           # MessagePassing base class
│   ├── linear.py                    # Linear layer for graphs
│   ├── graph_network_block.py       # GraphNetworkBlock composite layer
│   ├── conv/                        # 9 convolutional layers
│   │   ├── gcn_conv.py              # Graph Convolutional Network
│   │   ├── gat_conv.py              # Graph Attention Network
│   │   ├── gatv2_conv.py            # Dynamic Graph Attention (v2)
│   │   ├── sage_conv.py             # GraphSAGE
│   │   ├── gin_conv.py              # Graph Isomorphism Network
│   │   ├── cheb_conv.py             # Chebyshev Spectral Convolution
│   │   ├── transformer_conv.py      # Graph Transformer
│   │   ├── rel_conv.py              # Generalized Relational Convolution
│   │   └── simple_conv.py           # Simple aggregation convolution
│   ├── norm/                        # 3 normalization layers
│   │   ├── batch_norm.py            # Graph BatchNorm
│   │   ├── layer_norm.py            # Graph LayerNorm
│   │   └── instance_norm.py         # Graph InstanceNorm
│   └── pooling/                     # Pooling operators
│       ├── global_pooling.py        # Global add/mean/max pooling
│       └── topk_pool.py             # TopK hierarchical pooling
├── transforms/
│   ├── base_transform.py            # Abstract transform base class
│   └── normalize_features.py        # Feature normalization transform
└── utils/
    ├── array_ops.py                 # MLX array operations
    ├── scatter.py                   # Scatter add/mean/max primitives
    ├── topology.py                  # Graph topology utilities
    ├── sorting.py                   # Sorting helpers
    ├── transformations.py           # Adjacency/edge index conversions
    ├── convert.py                   # Format conversion (NetworkX, PyG)
    ├── validators.py                # Input validation
    └── fs.py                        # File system utilities
```

## Modules

### Convolutional Layers

| Layer | What It Does |
|-------|-------------|
| `GCNConv` | Spectral graph convolution with symmetric normalization (Kipf & Welling 2017) |
| `GATConv` | Attention-weighted neighbor aggregation with learnable attention coefficients (Velickovic et al. 2018) |
| `GATv2Conv` | Dynamic attention that conditions on both source and target nodes (Brody et al. 2022) |
| `SAGEConv` | Inductive representation learning via sampling and aggregating neighbor features (Hamilton et al. 2017) |
| `GINConv` | Maximally expressive aggregation matching the Weisfeiler-Leman graph isomorphism test (Xu et al. 2019) |
| `ChebConv` | Chebyshev polynomial spectral filters for localized graph convolution (Defferrard et al. 2016) |
| `TransformerConv` | Multi-head attention over graph neighborhoods (Shi et al. 2021) |
| `GeneralizedRelationalConv` | Relation-type-aware message passing for multi-relational graphs |
| `SimpleConv` | Parameter-free neighbor aggregation for baselines and ablation studies |

### Datasets

| Dataset | What It Does |
|---------|-------------|
| `PlanetoidDataset` | Load Cora, CiteSeer, or PubMed citation networks with train/val/test splits |
| `TUDataset` | Load any of 120+ TU benchmark graph classification datasets |
| `OGBDataset` | Load Open Graph Benchmark datasets with standard evaluation splits |
| `QM7bDataset` | Load QM7b molecular property prediction dataset (7,211 molecules, 14 targets) |
| `DBLP` | Load DBLP heterogeneous citation network |
| `IMDB` | Load IMDB heterogeneous movie dataset |
| `MovieLens100K` | Load MovieLens 100K recommendation bipartite graph |
| `EllipticBitcoinDataset` | Load Elliptic Bitcoin transaction network for fraud detection |
| `KarateClubDataset` | Load Zachary's Karate Club social network (34 nodes) |
| `SuperPixelDataset` | Load MNIST/CIFAR as superpixel graphs |

## Design Principles

1. **MLX-native execution** — all operations use MLX primitives that compile to Metal GPU shaders, avoiding the CUDA-to-Metal translation overhead that limits other frameworks on Apple Silicon
2. **Unified memory first** — graph data stays in shared CPU/GPU memory with zero-copy access, so large graphs that would exceed discrete GPU VRAM can be processed directly
3. **MessagePassing abstraction** — custom layers implement only `message()` and optionally `update()`, while the base class handles edge indexing, scatter operations, and aggregation on GPU
4. **PyG-compatible API** — class names, method signatures, and data structures mirror PyTorch Geometric conventions so users can port existing GNN code with minimal changes

## Testing

```bash
pip install -e '.[test]'
pytest tests/ -v                              # All tests (52 test files)
pytest tests/nn/ -v                           # Neural network layer tests
pytest tests/data/ -v                         # Data container and batching tests
pytest tests/datasets/ -v                     # Dataset loading tests
pytest tests/utils/ -v                        # Utility function tests
ruff check mlx_graphs/ tests/                 # Lint
ruff format --check mlx_graphs/ tests/        # Format check
```

## Installation

Install from PyPI:
```bash
pip install mlx-graphs
```
Or with [uv](https://docs.astral.sh/uv/):
```bash
uv add mlx-graphs
```

### Build from source

```bash
git clone git@github.com:mlx-graphs/mlx-graphs.git && cd mlx-graphs
pip install -e .
# or: uv pip install -e .
```

## Usage

Tutorial notebooks are available in the [documentation](https://mlx-graphs.github.io/mlx-graphs/):

- [Quickstart guide](https://mlx-graphs.github.io/mlx-graphs/tutorials/quickstart.html)
- [Graph classification guide](https://mlx-graphs.github.io/mlx-graphs/tutorials/graph_classification.html)

### Example: Custom GraphSAGE with edge weights

```python
import mlx.core as mx
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing

class SAGEConv(MessagePassing):
    def __init__(
        self, node_features_dim: int, out_features_dim: int, bias: bool = True, **kwargs
    ):
        super(SAGEConv, self).__init__(aggr="mean", **kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim

        self.neigh_proj = Linear(node_features_dim, out_features_dim, bias=False)
        self.self_proj = Linear(node_features_dim, out_features_dim, bias=bias)

    def __call__(self, edge_index: mx.array, node_features: mx.array, edge_weights: mx.array) -> mx.array:
         """Forward layer of the custom SAGE layer."""
         neigh_features = self.propagate( # Message passing directly on GPU
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": edge_weights},
         )
         neigh_features = self.neigh_proj(neigh_features)

        out_features = self.self_proj(node_features) + neigh_features
        return out_features

   def message(self, src_features: mx.array, dst_features: mx.array, **kwargs) -> mx.array:
         """Message function called by propagate(). Computes messages for all edges in the graph."""
        edge_weights = kwargs.get("edge_weights", None)

        return edge_weights.reshape(-1, 1) * src_features
```

## Dependencies

- **[MLX](https://github.com/ml-explore/mlx)** (>=0.18) — Apple's array framework providing Metal GPU primitives
- **[mlx_cluster](https://pypi.org/project/mlx-cluster/)** (>=0.0.5) — Cluster operations for MLX arrays
- **NumPy** (1.26.3) — Array utilities and interop
- **requests** (2.31.0) — HTTP client for dataset downloads
- **fsspec** (2024.2.0) — Filesystem abstraction for dataset storage
- **tqdm** (4.66.1) — Progress bars for dataset downloads and training loops

## Contributing

We are at an early stage of development, which means contributions can have a large impact. Everyone is welcome to contribute -- just open an issue with your idea and we will work together on the implementation. Contributions such as new layers and datasets are especially valuable.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pre-commit hook instructions.

**[Documentation](https://mlx-graphs.github.io/mlx-graphs/)** | **[Discord](https://discord.gg/K3mWFCxxM7)** | **[GitHub](https://github.com/mlx-graphs/mlx-graphs)**
